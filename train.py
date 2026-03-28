"""
训练 1D 回归网络（V2_last 主线默认：V2 + 8 labels + 截断输入）

支持 V1/V2/V3 三种模型，但默认只推荐 V2。

断点续训：
  --resume <ckpt_path>  从指定检查点恢复训练（恢复 epoch、优化器、scheduler 状态）
  每个 epoch 结束后除保存最优模型外，还额外保存 resume checkpoint（out_ckpt + ".resume.pt"）
  这样 Ctrl+C 中断后可以用 --resume 接续训练。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
import functions as fn


def log(msg: str):
    print(msg, flush=True)


def load_and_normalize(h5_path: Path, norm: fn.Normalization, max_samples: int | None = None):
    """加载 h5 并在内存中完成归一化，返回 TensorDataset"""
    log(f"[data] Loading {h5_path}...")
    with h5py.File(h5_path, "r") as h5f:
        x = np.array(h5f["x"][:max_samples], dtype=np.float32)
        y = np.array(h5f["y"][:max_samples], dtype=np.float32)
    log(f"[data]   raw shape: x={x.shape}, y={y.shape}")

    x_n = fn.normalize_x(x, norm).astype(np.float32)
    y_n = fn.normalize_y(y, norm).astype(np.float32)

    x_t = torch.from_numpy(x_n).unsqueeze(1)  # (N, 1, S)
    y_t = torch.from_numpy(y_n)               # (N, D)
    y_raw_t = torch.from_numpy(y)             # (N, D)

    log(f"[data]   normalized, tensor shapes: x={x_t.shape}, y={y_t.shape}")
    return TensorDataset(x_t, y_t), y_raw_t


def load_exp_dataset(h5_path: Path, norm: fn.Normalization):
    """加载实验弱监督数据；若 h5 不含 is_prepump，则默认全部视为 pre-pump。"""
    log(f"[exp] Loading {h5_path}...")
    with h5py.File(h5_path, "r") as h5f:
        x = np.array(h5f["x"], dtype=np.float32)
        time_idx = np.array(h5f["time_idx"], dtype=np.int64)
        if "is_prepump" in h5f:
            is_prepump = np.array(h5f["is_prepump"], dtype=np.int64)
        else:
            is_prepump = np.ones(len(x), dtype=np.int64)
    log(f"[exp]   raw shape: x={x.shape}, time_idx={time_idx.shape}, prepump={int(is_prepump.sum())}/{len(is_prepump)}")
    x_n = fn.normalize_x(x, norm).astype(np.float32)
    x_t = torch.from_numpy(x_n).unsqueeze(1)
    time_t = torch.from_numpy(time_idx)
    prepump_t = torch.from_numpy(is_prepump)
    log(f"[exp]   normalized tensor shape: x={x_t.shape}")
    return TensorDataset(x_t, time_t, prepump_t)


def evaluate_loader_mae(model, loader, y_raw: torch.Tensor, norm: fn.Normalization,
                        device: torch.device) -> tuple[float, np.ndarray]:
    preds = []
    with torch.no_grad():
        for xb, _yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())
    pred_n = torch.cat(preds, dim=0).numpy()
    pred_real = fn.denormalize_y(pred_n, norm)
    y_true = y_raw.numpy()
    err = np.abs(pred_real - y_true)
    return float(err.mean()), err.mean(axis=0)


def save_inference_ckpt(path: Path, model, args, norm: fn.Normalization):
    torch.save({
        "model_state": model.state_dict(),
        "model_version": args.model_version,
        "base_ch": args.base_ch,
        "n_blocks": args.n_blocks,
        "norm_json": str(Path(args.norm_json)),
        "config": {
            "input_len": len(norm.x_mean),
            "label_dim": cfg.LABEL_FLAT_DIM,
            "label_names": fn.label_names_for_dim(cfg.LABEL_FLAT_DIM),
            "label_atoms": cfg.LABEL_ATOMS,
            "mainline_tag": cfg.V2_LAST_TAG,
        },
    }, path)


def next_or_restart(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def exp_consistency_loss(pred: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
    """同一 pre-pump 时间点的 bootstrap 预测应彼此接近。"""
    uniq = torch.unique(time_idx)
    losses = []
    for t in uniq:
        mask = time_idx == t
        if int(mask.sum()) > 1:
            losses.append(pred[mask].var(dim=0, unbiased=False).mean())
    if losses:
        return torch.stack(losses).mean()
    return pred.var(dim=0, unbiased=False).mean()


def evaluate_exp_prepump(model, dl_exp_eval, norm: fn.Normalization, device: torch.device,
                         eq_real: np.ndarray):
    preds = []
    prepump_masks = []
    with torch.no_grad():
        for xb, _time_idx, is_prepump in dl_exp_eval:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().astype(np.float32)
            preds.append(pred)
            prepump_masks.append(is_prepump.numpy().astype(bool))
    pred_n = np.concatenate(preds, axis=0)
    pred_real = fn.denormalize_y(pred_n, norm)
    prepump_mask = np.concatenate(prepump_masks, axis=0)
    pred_prepump = pred_real[prepump_mask]
    dev = np.abs(pred_prepump - eq_real[None, :])
    per_dim = dev.mean(axis=0)
    mean_dev = float(per_dim.mean())
    key_dev = float(per_dim[:min(3, len(per_dim))].mean())
    return mean_dev, key_dev, per_dim


def save_resume_ckpt(path: Path, model, opt, scheduler, epoch: int,
                     best_val_mae: float, patience_counter: int,
                     global_step: int, args):
    """保存可续训检查点（包含完整训练状态）"""
    norm = fn.load_normalization(Path(args.norm_json))
    torch.save({
        "model_state":     model.state_dict(),
        "opt_state":       opt.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch":           epoch,
        "best_val_mae":    best_val_mae,
        "patience_counter": patience_counter,
        "global_step":     global_step,
        "model_version":   args.model_version,
        "base_ch":         args.base_ch,
        "n_blocks":        args.n_blocks,
        "norm_json":       str(Path(args.norm_json)),
        "config":          {
            "input_len": len(norm.x_mean),
            "label_dim": cfg.LABEL_FLAT_DIM,
            "label_names": fn.label_names_for_dim(cfg.LABEL_FLAT_DIM),
            "label_atoms": cfg.LABEL_ATOMS,
            "mainline_tag": cfg.V2_LAST_TAG,
        },
        "train_args":      vars(args),
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_h5",     type=str, default=str(cfg.PATHS.train_h5))
    ap.add_argument("--val_h5",       type=str, default=str(cfg.PATHS.val_h5))
    ap.add_argument("--norm_json",    type=str, default=str(cfg.PATHS.norm_json))
    ap.add_argument("--out_ckpt",     type=str, default=str(cfg.PATHS.checkpoint_pt))

    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=1.5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base_ch",      type=int,   default=128)
    ap.add_argument("--n_blocks",     type=int,   default=8)
    ap.add_argument("--model_version",type=str,   default=cfg.DEFAULT_MODEL_VERSION, choices=["v1", "v2", "v3"])
    ap.add_argument("--warmup_epochs",type=int,   default=3)
    ap.add_argument("--patience",     type=int,   default=15, help="Early stop patience")
    ap.add_argument("--seed",         type=int,   default=cfg.GEN_RANDOM_SEED)
    ap.add_argument("--max_samples",  type=int,   default=None,
                    help="截取前 N 个训练样本（smoke test 用，None = 全量）")
    ap.add_argument("--resume",       type=str,   default=None,
                    help="从此 .resume.pt 检查点恢复训练（可中断后续训）")
    ap.add_argument("--init_ckpt",    type=str,   default=None,
                    help="仅加载模型权重作为初始化，不恢复优化器/调度器（适合 domain adaptation fine-tune）")
    ap.add_argument("--loss_type",    type=str,   default="smooth_l1",
                    choices=["smooth_l1", "combined"],
                    help="Loss function: smooth_l1 or combined (0.7*SmoothL1 + 0.3*MSE)")
    ap.add_argument("--mixup_alpha",  type=float, default=0.0,
                    help="Mixup alpha (0.0 = disabled). When > 0, applies mixup augmentation.")
    ap.add_argument("--exp_h5",       type=str, default=None,
                    help="实验 pre-pump 弱监督 h5（由 inference.py --mode export_prepump_h5 生成）")
    ap.add_argument("--exp_batch_size", type=int, default=64)
    ap.add_argument("--lambda_exp_eq", type=float, default=0.0,
                    help="实验 pre-pump 平衡态弱监督损失权重")
    ap.add_argument("--lambda_exp_cons", type=float, default=0.0,
                    help="实验 pre-pump bootstrap consistency 损失权重")
    ap.add_argument("--balanced_exp_weight", type=float, default=0.8,
                    help="balanced checkpoint score = val_mae + w * exp_dev")
    ap.add_argument("--joint_exp_h5", type=str, default=None,
                    help="joint-train: 实验 pre-pump 训练 h5（含 y=equilibrium）")
    ap.add_argument("--joint_exp_holdout_h5", type=str, default=None,
                    help="joint-train: 实验 pre-pump holdout h5（仅评估，不参与训练）")
    ap.add_argument("--joint_exp_batch_size", type=int, default=4,
                    help="每个 synthetic batch 额外混入的 experimental 样本数")
    ap.add_argument("--save_each_epoch_dir", type=str, default=None,
                    help="若提供，则每个 epoch 额外保存一个可推理 checkpoint 到该目录")
    args = ap.parse_args()

    fn.set_global_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"[train] device={device}, model={args.model_version}")

    norm = fn.load_normalization(Path(args.norm_json))

    ds_tr, _ = load_and_normalize(Path(args.train_h5), norm, max_samples=args.max_samples)
    val_max = args.max_samples // 10 if args.max_samples else None
    ds_va, y_val_raw = load_and_normalize(Path(args.val_h5), norm, max_samples=val_max)
    log(f"[train] train={len(ds_tr)}, val={len(ds_va)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=False)

    dl_exp = None
    dl_exp_eval = None
    dl_joint_exp = None
    dl_joint_holdout = None
    y_joint_holdout_raw = None
    eq_real = fn.equilibrium_labels_for_dim(cfg.LABEL_FLAT_DIM).astype(np.float32)
    eq_norm = fn.normalize_y(eq_real[None, :], norm).astype(np.float32)[0]
    eq_target = torch.from_numpy(eq_norm).float().to(device)
    if args.exp_h5:
        ds_exp = load_exp_dataset(Path(args.exp_h5), norm)
        dl_exp = DataLoader(ds_exp, batch_size=args.exp_batch_size, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=False)
        dl_exp_eval = DataLoader(ds_exp, batch_size=args.exp_batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False)
        log(f"[exp] train/eval samples={len(ds_exp)}  lambda_eq={args.lambda_exp_eq} "
            f"lambda_cons={args.lambda_exp_cons}")

    if args.joint_exp_h5:
        ds_joint_exp, _y_joint_exp_raw = load_and_normalize(Path(args.joint_exp_h5), norm)
        dl_joint_exp = DataLoader(
            ds_joint_exp,
            batch_size=args.joint_exp_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        log(f"[joint_exp] train samples={len(ds_joint_exp)} batch={args.joint_exp_batch_size}")

    if args.joint_exp_holdout_h5:
        ds_joint_holdout, y_joint_holdout_raw = load_and_normalize(Path(args.joint_exp_holdout_h5), norm)
        dl_joint_holdout = DataLoader(
            ds_joint_holdout,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        log(f"[joint_exp] holdout samples={len(ds_joint_holdout)}")

    if args.model_version == "v3":
        model = fn.NMMRegressorV3(base_ch=args.base_ch, n_blocks=args.n_blocks,
                                  out_dim=cfg.LABEL_FLAT_DIM).to(device)
    elif args.model_version == "v2":
        model = fn.NMMRegressorV2(base_ch=args.base_ch, n_blocks=args.n_blocks,
                                  out_dim=cfg.LABEL_FLAT_DIM).to(device)
    else:
        model = fn.NMMRegressor1D(base_ch=args.base_ch, n_blocks=args.n_blocks,
                                  out_dim=cfg.LABEL_FLAT_DIM).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[train] model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + CosineAnnealing（基于总步数，resume 时保持一致）
    warmup_steps = args.warmup_epochs * len(dl_tr)
    total_steps  = args.epochs * len(dl_tr)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 续训状态
    start_epoch     = 1
    best_val_mae    = float("inf")
    patience_counter = 0
    global_step     = 0

    if args.init_ckpt and args.resume:
        raise ValueError("--init_ckpt and --resume are mutually exclusive")

    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.exists():
            raise FileNotFoundError(f"--init_ckpt: {init_path} not found")
        log(f"[train] Initializing model weights from {init_path}")
        ckpt_init = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_init["model_state"], strict=True)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume: {resume_path} not found")
        log(f"[train] Resuming from {resume_path}")
        ckpt_r = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_r["model_state"])
        opt.load_state_dict(ckpt_r["opt_state"])
        scheduler.load_state_dict(ckpt_r["scheduler_state"])
        start_epoch      = ckpt_r["epoch"] + 1
        best_val_mae     = ckpt_r["best_val_mae"]
        patience_counter = ckpt_r["patience_counter"]
        global_step      = ckpt_r["global_step"]
        log(f"[train] Resumed: start_epoch={start_epoch} best_val_mae={best_val_mae:.4f}A "
            f"patience_counter={patience_counter}")

    # 加权损失：根据输出维度自动设置
    if cfg.LABEL_FLAT_DIM == 3:
        # 3-dim: [O-N, O-C5, N-C5]，N-C5 变化范围最大加权
        loss_weights = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32, device=device)
    elif cfg.LABEL_FLAT_DIM == 8:
        # 8-dim: 7 个距离 + h_C5_signed
        loss_weights = torch.tensor([1.0, 1.0, 2.5, 1.5, 1.5, 3.0, 3.0, 2.0],
                                    dtype=torch.float32, device=device)
    else:
        # 7-dim: [O-N, O-C5, N-C5, N-C2, N-C4, C5-C2, C5-C4]
        # C5-C2/C5-C4 权重提高以改善 C5 重建精度
        loss_weights = torch.tensor([1.0, 1.0, 2.5, 1.5, 1.5, 3.0, 3.0], dtype=torch.float32, device=device)
    loss_weights = loss_weights / loss_weights.sum() * len(loss_weights)  # 归一化使均值=1
    base_loss_fn = nn.SmoothL1Loss(reduction="none")
    if args.loss_type == "combined":
        mse_loss_fn = nn.MSELoss(reduction="none")
        log(f"[train] Using combined loss: 0.7*SmoothL1 + 0.3*MSE")
    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    resume_ckpt = out_ckpt.with_suffix(".resume.pt")
    expbest_ckpt = out_ckpt.with_name(out_ckpt.stem + "_expbest.pt")
    balanced_ckpt = out_ckpt.with_name(out_ckpt.stem + "_balanced.pt")
    epoch_ckpt_dir = Path(args.save_each_epoch_dir) if args.save_each_epoch_dir else None
    if epoch_ckpt_dir is not None:
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_exp_dev = float("inf")
    best_balanced_score = float("inf")

    log("[train] Starting training loop...")
    exp_iter = iter(dl_exp) if dl_exp is not None else None
    joint_exp_iter = iter(dl_joint_exp) if dl_joint_exp is not None else None
    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        tr_exp_eq_losses = []
        tr_exp_cons_losses = []
        log(f"[train] ep{ep} starting...")

        for batch_idx, (xb, yb) in enumerate(dl_tr, 1):
            xb = xb.to(device)
            yb = yb.to(device)

            if dl_joint_exp is not None:
                (xe, ye), joint_exp_iter = next_or_restart(joint_exp_iter, dl_joint_exp)
                xb = torch.cat([xb, xe.to(device)], dim=0)
                yb = torch.cat([yb, ye.to(device)], dim=0)

            # Mixup augmentation
            if args.mixup_alpha > 0.0:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                perm = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb = lam * yb + (1 - lam) * yb[perm]

            pred = model(xb)
            if args.loss_type == "combined":
                loss_sup = (0.7 * (base_loss_fn(pred, yb) * loss_weights).mean()
                            + 0.3 * (mse_loss_fn(pred, yb) * loss_weights).mean())
            else:
                loss_sup = (base_loss_fn(pred, yb) * loss_weights).mean()
            loss = loss_sup

            if dl_exp is not None and (args.lambda_exp_eq > 0.0 or args.lambda_exp_cons > 0.0):
                (xe, time_idx_e, is_prepump_e), exp_iter = next_or_restart(exp_iter, dl_exp)
                xe = xe.to(device)
                time_idx_e = time_idx_e.to(device)
                is_prepump_e = is_prepump_e.to(device).bool()
                pred_e = model(xe)

                if args.lambda_exp_eq > 0.0:
                    if bool(is_prepump_e.any()):
                        pred_eq = pred_e[is_prepump_e]
                        eq_batch = eq_target.unsqueeze(0).expand_as(pred_eq)
                        loss_exp_eq = (base_loss_fn(pred_eq, eq_batch) * loss_weights).mean()
                        loss = loss + args.lambda_exp_eq * loss_exp_eq
                        tr_exp_eq_losses.append(float(loss_exp_eq.item()))

                if args.lambda_exp_cons > 0.0:
                    loss_exp_cons = exp_consistency_loss(pred_e, time_idx_e)
                    loss = loss + args.lambda_exp_cons * loss_exp_cons
                    tr_exp_cons_losses.append(float(loss_exp_cons.item()))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            global_step += 1
            tr_losses.append(float(loss.item()))

            if batch_idx % 50 == 0 or batch_idx == len(dl_tr):
                cur    = float(np.mean(tr_losses[-50:]))
                lr_now = opt.param_groups[0]['lr']
                log(f"  ep{ep} [{batch_idx}/{len(dl_tr)}] loss={cur:.6f} lr={lr_now:.2e}")

        # Validation
        model.eval()
        va_losses = []
        va_preds  = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                if args.loss_type == "combined":
                    v_loss = (0.7 * (base_loss_fn(pred, yb) * loss_weights).mean()
                              + 0.3 * (mse_loss_fn(pred, yb) * loss_weights).mean())
                else:
                    v_loss = (base_loss_fn(pred, yb) * loss_weights).mean()
                va_losses.append(float(v_loss.item()))
                va_preds.append(pred.cpu())

        va_preds_all  = torch.cat(va_preds, dim=0).numpy()
        va_preds_real = fn.denormalize_y(va_preds_all, norm)
        y_val_np      = y_val_raw.numpy()
        val_mae       = float(np.abs(va_preds_real - y_val_np).mean())
        per_dim_mae   = np.abs(va_preds_real - y_val_np).mean(axis=0)
        # key_mae: 前 3 维为关键距离（O-N, O-C5, N-C5）
        key_mae       = float(per_dim_mae[:min(3, len(per_dim_mae))].mean())

        tr_loss = float(np.mean(tr_losses))
        val_mse = float(np.mean(va_losses))
        dt      = time.time() - t0
        exp_dev = None
        exp_key_dev = None
        exp_per_dim = None
        joint_holdout_dev = None
        joint_holdout_per_dim = None
        balanced_score = None
        if dl_exp_eval is not None:
            exp_dev, exp_key_dev, exp_per_dim = evaluate_exp_prepump(
                model, dl_exp_eval, norm, device, eq_real
            )
            balanced_score = val_mae + args.balanced_exp_weight * exp_dev
        if dl_joint_holdout is not None and y_joint_holdout_raw is not None:
            joint_holdout_dev, joint_holdout_per_dim = evaluate_loader_mae(
                model, dl_joint_holdout, y_joint_holdout_raw, norm, device
            )

        improved = ""
        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            patience_counter = 0
            save_inference_ckpt(out_ckpt, model, args, norm)
            improved += " *BEST_VAL*"
        else:
            patience_counter += 1

        if exp_dev is not None and exp_dev < best_exp_dev:
            best_exp_dev = exp_dev
            save_inference_ckpt(expbest_ckpt, model, args, norm)
            improved += " *BEST_EXP*"

        if balanced_score is not None and balanced_score < best_balanced_score:
            best_balanced_score = balanced_score
            save_inference_ckpt(balanced_ckpt, model, args, norm)
            improved += " *BEST_BAL*"

        dim_str = " ".join(f"{cfg.LABEL_PAIR_NAMES[i]}={per_dim_mae[i]:.4f}" for i in range(len(per_dim_mae)))
        exp_loss_str = ""
        if tr_exp_eq_losses:
            exp_loss_str += f" exp_eq={float(np.mean(tr_exp_eq_losses)):.6f}"
        if tr_exp_cons_losses:
            exp_loss_str += f" exp_cons={float(np.mean(tr_exp_cons_losses)):.6f}"
        exp_metric_str = ""
        if exp_dev is not None:
            exp_metric_str = (f" | exp_dev={exp_dev:.4f}A exp_key={exp_key_dev:.4f}A"
                              f" bal={balanced_score:.4f}")
        if joint_holdout_dev is not None:
            exp_metric_str += f" | exp_holdout={joint_holdout_dev:.4f}A"
        log(f"[ep {ep:03d}/{args.epochs}] train_loss={tr_loss:.6f}{exp_loss_str} | "
            f"val_mse={val_mse:.6f} val_mae={val_mae:.4f}A key_mae={key_mae:.4f}A{exp_metric_str} | "
            f"{dim_str} | {dt:.1f}s{improved}")

        # 每个 epoch 保存续训检查点（可中断后接续）
        save_resume_ckpt(resume_ckpt, model, opt, scheduler, ep,
                         best_val_mae, patience_counter, global_step, args)
        log(f"[train] resume ckpt saved -> {resume_ckpt.name}")
        if epoch_ckpt_dir is not None:
            epoch_ckpt = epoch_ckpt_dir / f"epoch_{ep:03d}.pt"
            save_inference_ckpt(epoch_ckpt, model, args, norm)
            log(f"[train] epoch ckpt saved -> {epoch_ckpt}")

        if patience_counter >= args.patience:
            log(f"[train] Early stopping at epoch {ep} (patience={args.patience})")
            break

    if dl_exp_eval is not None:
        log(f"[train] Done. Best val MAE: {best_val_mae:.4f} A -> {out_ckpt}")
        log(f"[train] Best experimental pre-pump dev: {best_exp_dev:.4f} A -> {expbest_ckpt}")
        log(f"[train] Best balanced score: {best_balanced_score:.4f} -> {balanced_ckpt}")
    else:
        log(f"[train] Done. Best val MAE: {best_val_mae:.4f} A, saved to {out_ckpt}")


if __name__ == "__main__":
    main()
