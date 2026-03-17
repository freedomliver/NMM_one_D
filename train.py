"""
训练 1D 回归网络（输入 681 长度信号 -> 输出 9 维距离矩阵 flatten）

你最常修改的参数：
- batch size / 学习率 / epoch：命令行参数
- 网络结构：functions.NMMRegressor1D(base_ch, n_blocks)
- 训练/验证数据路径：--train_h5 / --val_h5
- 归一化文件：--norm_json（由 train_gen_NMM.py 生成）
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
import functions as fn


def progress_bar(current, total, prefix="", decimals=1, length=40, fill="█"):
    """简单的进度条显示"""
    if total == 0:
        return ""
    percent = current / total
    filled = int(length * percent)
    bar = fill * filled + "-" * (length - filled)
    return f"{prefix} |{bar}| {100*percent:.1f}% ({current}/{total})"


def format_time(seconds):
    """将秒数格式化为易读的时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_h5", type=str, default=str(cfg.PATHS.train_h5))
    ap.add_argument("--val_h5", type=str, default=str(cfg.PATHS.val_h5))
    ap.add_argument("--norm_json", type=str, default=str(cfg.PATHS.norm_json))
    ap.add_argument("--out_ckpt", type=str, default=str(cfg.PATHS.checkpoint_pt))

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=cfg.GEN_RANDOM_SEED)
    args = ap.parse_args()

    fn.set_global_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train] device={device}")

    print(f"[train] loading datasets...")
    train_h5 = Path(args.train_h5)
    val_h5 = Path(args.val_h5)
    norm = fn.load_normalization(Path(args.norm_json))

    ds_tr = fn.H5Dataset(train_h5)
    ds_va = fn.H5Dataset(val_h5)
    print(f"[train] train set size: {len(ds_tr)}, val set size: {len(ds_va)}")
    
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[train] train batches: {len(dl_tr)}, val batches: {len(dl_va)}")

    model = fn.NMMRegressor1D(base_ch=args.base_ch, n_blocks=args.n_blocks, out_dim=cfg.LABEL_FLAT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    loss_fn = nn.MSELoss()

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    epoch_times = []  # 用于计算 ETA
    
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        
        # 训练循环，带进度条
        for batch_idx, (xb, yb) in enumerate(dl_tr, 1):
            xb = xb.numpy()  # (B,1,S)
            yb = yb.numpy()  # (B,9)
            xb = fn.normalize_x(xb[:, 0, :], norm)[:, None, :]  # normalize on (B,S) then restore (B,1,S)
            yb = fn.normalize_y(yb, norm)

            xb = torch.from_numpy(xb).float().to(device)
            yb = torch.from_numpy(yb).float().to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.detach().cpu().item()))
            
            # 每 100 batch 或最后一次输出进度
            if batch_idx % max(1, len(dl_tr) // 10) == 0 or batch_idx == len(dl_tr):
                bar = progress_bar(batch_idx, len(dl_tr), prefix=f"  [train]", length=30)
                current_loss = float(np.mean(tr_losses[-100:])) if tr_losses else float("nan")
                print(f"\r{bar} loss={current_loss:.6f}", end="", flush=True)
        
        print()  # 换行

        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(dl_va, 1):
                xb = xb.numpy()
                yb = yb.numpy()
                xb = fn.normalize_x(xb[:, 0, :], norm)[:, None, :]
                yb = fn.normalize_y(yb, norm)

                xb = torch.from_numpy(xb).float().to(device)
                yb = torch.from_numpy(yb).float().to(device)

                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(float(loss.detach().cpu().item()))
                
                # 验证进度条（简要显示）
                if batch_idx % max(1, len(dl_va) // 5) == 0 or batch_idx == len(dl_va):
                    bar = progress_bar(batch_idx, len(dl_va), prefix=f"  [val]  ", length=30)
                    print(f"\r{bar}", end="", flush=True)
        
        print()  # 换行

        sch.step()
        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        dt = time.time() - t0
        epoch_times.append(dt)
        
        # 计算 ETA
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = args.epochs - ep
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = format_time(eta_seconds)
        
        print(f"[train] ep {ep:03d}/{args.epochs} | train_mse={tr:.6f} val_mse={va:.6f} | lr={sch.get_last_lr()[0]:.2e} | {dt:.1f}s (ETA: {eta_str})")

        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "norm_json": str(Path(args.norm_json)),
                    "config": {
                        "S_LEN": cfg.S_LEN,
                        "LABEL_ATOMS": cfg.LABEL_ATOMS,
                    },
                },
                out_ckpt,
            )
            print(f"[train] ✓ saved best checkpoint: {out_ckpt} (val_mse={best_val:.6f})")


if __name__ == "__main__":
    main()

