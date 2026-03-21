"""
实时训练曲线绘图 — 解析训练日志并生成/更新图像

用法:
  python plot_training.py train_3dim_80k.log           # 一次性绘图
  python plot_training.py train_3dim_80k.log --watch    # 每 30 秒自动刷新
  python plot_training.py log1.log log2.log             # 对比多个训练
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path: str) -> dict:
    """解析训练日志，提取每 epoch 的指标"""
    # Format with per-dim: [ep 001/40] train_loss=... | val_mse=... | O-N=... | 418.1s
    pattern_dim = re.compile(
        r"\[ep\s+(\d+)/(\d+)\]\s+"
        r"train_loss=([\d.]+)\s*\|\s*"
        r"val_mse=([\d.]+)\s+val_mae=([\d.]+)A\s+key_mae=([\d.]+)A\s*\|"
        r"\s*(.*?)\s*\|\s*([\d.]+)s"
    )
    # Format without per-dim: [ep 001/60] train_loss=... | val_mse=... | 481.6s
    pattern_nodim = re.compile(
        r"\[ep\s+(\d+)/(\d+)\]\s+"
        r"train_loss=([\d.]+)\s*\|\s*"
        r"val_mse=([\d.]+)\s+val_mae=([\d.]+)A\s+key_mae=([\d.]+)A\s*\|"
        r"\s*([\d.]+)s"
    )
    dim_pattern = re.compile(r"([\w-]+)=([\d.]+)")

    data = {
        "epoch": [], "train_loss": [], "val_mse": [],
        "val_mae": [], "key_mae": [], "time_s": [],
    }
    dim_names = []
    dim_data = {}

    with open(log_path) as f:
        for line in f:
            m = pattern_dim.search(line)
            if m:
                data["epoch"].append(int(m.group(1)))
                data["train_loss"].append(float(m.group(3)))
                data["val_mse"].append(float(m.group(4)))
                data["val_mae"].append(float(m.group(5)))
                data["key_mae"].append(float(m.group(6)))
                data["time_s"].append(float(m.group(8)))
                # Parse per-dim MAE
                dim_str = m.group(7)
                for dm in dim_pattern.finditer(dim_str):
                    name = dm.group(1)
                    val = float(dm.group(2))
                    if name not in dim_data:
                        dim_data[name] = []
                        dim_names.append(name)
                    dim_data[name].append(val)
                continue

            m2 = pattern_nodim.search(line)
            if m2:
                data["epoch"].append(int(m2.group(1)))
                data["train_loss"].append(float(m2.group(3)))
                data["val_mse"].append(float(m2.group(4)))
                data["val_mae"].append(float(m2.group(5)))
                data["key_mae"].append(float(m2.group(6)))
                data["time_s"].append(float(m2.group(7)))

    data["dim_names"] = dim_names
    data["dim_data"] = dim_data
    return data


def plot(log_paths: list[str], out_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Progress", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10.colors
    multi = len(log_paths) > 1
    d = {}

    for idx, lp in enumerate(log_paths):
        d = parse_log(lp)
        if not d["epoch"]:
            print(f"  [WARN] No epoch data in {lp}")
            continue

        label_prefix = Path(lp).stem + " " if multi else ""
        c = colors[idx % len(colors)]
        ep = d["epoch"]

        # Left: Train loss vs Val loss on the same axes
        ax = axes[0]
        ax.plot(ep, d["train_loss"], "-o", ms=3, color=c, label=f"{label_prefix}train loss")
        ax.plot(ep, d["val_mse"], "-s", ms=3, color=colors[(idx * 2 + 1) % len(colors)],
                label=f"{label_prefix}val loss (MSE)")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.set_title("Train vs Validation Loss")
        ax.grid(True, alpha=0.3)

        # Right: Per-dimension MAE
        ax = axes[1]
        dim_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
        for di, name in enumerate(d["dim_names"]):
            vals = d["dim_data"][name]
            ax.plot(ep[:len(vals)], vals, "-o", ms=3, color=dim_colors[di % len(dim_colors)],
                    label=f"{label_prefix}{name}")
        ax.set_ylabel("MAE (Å)")
        ax.set_xlabel("Epoch")
        ax.set_title("Per-Distance MAE")
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(fontsize=8)

    # Add summary text
    if not multi and d.get("epoch"):
        best_ep = d["epoch"][np.argmin(d["key_mae"])]
        best_km = min(d["key_mae"])
        fig.text(0.5, -0.02,
                 f"Best key_mae: {best_km:.4f} Å at epoch {best_ep}  |  "
                 f"Latest: ep{d['epoch'][-1]} key_mae={d['key_mae'][-1]:.4f} Å",
                 ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Training log file(s)")
    ap.add_argument("--out", type=str, default="training_curves.png")
    ap.add_argument("--watch", action="store_true", help="Auto-refresh every 30s")
    args = ap.parse_args()

    if args.watch:
        print("[plot] Watch mode — Ctrl+C to stop")
        while True:
            try:
                plot(args.logs, args.out)
                time.sleep(30)
            except KeyboardInterrupt:
                break
    else:
        plot(args.logs, args.out)


if __name__ == "__main__":
    main()
