"""
训练 1D 回归网络（输入 681 长度信号 -> 输出 9 维距离矩阵 flatten）

支持 V1 (NMMRegressor1D) 和 V2 (NMMRegressorV2) 两种模型。

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
    y_t = torch.from_numpy(y_n)               # (N, 9)
    y_raw_t = torch.from_numpy(y)              # (N, 9)

    log(f"[data]   normalized, tensor shapes: x={x_t.shape}, y={y_t.shape}")
    return TensorDataset(x_t, y_t), y_raw_t


def save_resume_ckpt(path: Path, model, opt, scheduler, epoch: int,
                     best_val_mae: float, patience_counter: int,
                     global_step: int, args):
    """保存可续训检查点（包含完整训练状态）"""
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
        "config":          {"S_LEN": cfg.S_LEN, "LABEL_ATOMS": cfg.LABEL_ATOMS},
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
    ap.add_argument("--model_version",type=str,   default="v2", choices=["v1", "v2"])
    ap.add_argument("--warmup_epochs",type=int,   default=3)
    ap.add_argument("--patience",     type=int,   default=15, help="Early stop patience")
    ap.add_argument("--seed",         type=int,   default=cfg.GEN_RANDOM_SEED)
    ap.add_argument("--max_samples",  type=int,   default=None,
                    help="截取前 N 个训练样本（smoke test 用，None = 全量）")
    ap.add_argument("--resume",       type=str,   default=None,
                    help="从此 .resume.pt 检查点恢复训练（可中断后续训）")
    ap.add_argument("--loss_type",    type=str,   default="smooth_l1",
                    choices=["smooth_l1", "combined"],
                    help="Loss function: smooth_l1 or combined (0.7*SmoothL1 + 0.3*MSE)")
    ap.add_argument("--mixup_alpha",  type=float, default=0.0,
                    help="Mixup alpha (0.0 = disabled). When > 0, applies mixup augmentation.")
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

    if args.model_version == "v2":
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

    log("[train] Starting training loop...")
    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        log(f"[train] ep{ep} starting...")

        for batch_idx, (xb, yb) in enumerate(dl_tr, 1):
            xb = xb.to(device)
            yb = yb.to(device)

            # Mixup augmentation
            if args.mixup_alpha > 0.0:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                perm = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb = lam * yb + (1 - lam) * yb[perm]

            pred = model(xb)
            if args.loss_type == "combined":
                loss = (0.7 * (base_loss_fn(pred, yb) * loss_weights).mean()
                        + 0.3 * (mse_loss_fn(pred, yb) * loss_weights).mean())
            else:
                loss = (base_loss_fn(pred, yb) * loss_weights).mean()
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

        improved = ""
        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            patience_counter = 0
            # 保存最优模型（仅模型权重，推理用）
            torch.save({
                "model_state":  model.state_dict(),
                "model_version": args.model_version,
                "base_ch":      args.base_ch,
                "n_blocks":     args.n_blocks,
                "norm_json":    str(Path(args.norm_json)),
                "config":       {"S_LEN": cfg.S_LEN, "LABEL_ATOMS": cfg.LABEL_ATOMS},
            }, out_ckpt)
            improved = " *BEST*"
        else:
            patience_counter += 1

        dim_str = " ".join(f"{cfg.LABEL_PAIR_NAMES[i]}={per_dim_mae[i]:.4f}" for i in range(len(per_dim_mae)))
        log(f"[ep {ep:03d}/{args.epochs}] train_loss={tr_loss:.6f} | "
            f"val_mse={val_mse:.6f} val_mae={val_mae:.4f}A key_mae={key_mae:.4f}A | "
            f"{dim_str} | {dt:.1f}s{improved}")

        # 每个 epoch 保存续训检查点（可中断后接续）
        save_resume_ckpt(resume_ckpt, model, opt, scheduler, ep,
                         best_val_mae, patience_counter, global_step, args)
        log(f"[train] resume ckpt saved -> {resume_ckpt.name}")

        if patience_counter >= args.patience:
            log(f"[train] Early stopping at epoch {ep} (patience={args.patience})")
            break

    log(f"[train] Done. Best val MAE: {best_val_mae:.4f} A, saved to {out_ckpt}")


if __name__ == "__main__":
    main()
