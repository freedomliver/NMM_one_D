"""
实验推理结果绘图脚本

用法:
  python plot_inference.py --npz results/inference/inference_7dim_80k_exp.npz \
      --out results/inference/inference_7dim_80k_errbar.png \
      --title "nmm_v2_7dim_80k, alpha=0.03, mean ± std of 200 bootstraps"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import functions as fn

COLORS = {
    "O-N":   "#e74c3c",
    "O-C5":  "#2ecc71",
    "N-C5":  "#3498db",
    "N-C2":  "#f39c12",
    "N-C4":  "#9b59b6",
    "C5-C2": "#1abc9c",
    "C5-C4": "#e67e22",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",   type=str, required=True, help="推理结果 npz 文件")
    ap.add_argument("--out",   type=str, required=True, help="输出 png 路径")
    ap.add_argument("--title", type=str, default="Experimental Inference — NMM Key Distances vs Time Delay")
    ap.add_argument("--timedelay", type=str, default="TimeDelay.mat")
    ap.add_argument("--prepump_end", type=int, default=7,
                    help="pre-pump 时间点上界（默认7，即 t00-t06）")
    args = ap.parse_args()

    # 加载时间轴
    td = sio.loadmat(args.timedelay)["TimeDelay"].flatten()  # (45,) fs

    # 加载推理结果
    d = np.load(args.npz, allow_pickle=True)
    y_mean = d["y_mean"]   # (45, D)
    y_std  = d["y_std"]    # (45, D)
    n_t, n_dim = y_mean.shape

    # 确定维度标签与平衡值
    if "label_names" in d:
        labels = [str(x) for x in d["label_names"].tolist()]
    else:
        labels = fn.label_names_for_dim(n_dim)
    if "equilibrium" in d:
        eq_vals = np.asarray(d["equilibrium"], dtype=np.float32)
    else:
        eq_vals = fn.equilibrium_labels_for_dim(n_dim)

    # pre-pump 均值（t00 ~ prepump_end-1）
    pre_pump_mean = y_mean[:args.prepump_end].mean(axis=0)

    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 3.2 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]

    fig.suptitle(args.title, fontsize=11, fontweight="bold")

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        eq  = float(eq_vals[i]) if i < len(eq_vals) else None
        c   = COLORS.get(lbl, "#333333")
        mu  = y_mean[:, i]
        sig = y_std[:, i]

        # pre-pump 阴影
        ax.axvspan(td[0], 0, alpha=0.08, color="steelblue")
        # t=0 分界线
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
        # 平衡值水平线
        if eq is not None:
            ax.axhline(eq, color="red", lw=1.0, ls="--", alpha=0.7)
        # pre-pump 均值水平线
        ax.axhline(pre_pump_mean[i], color="blue", lw=1.0, ls="--", alpha=0.6)

        # 误差带 + 均值曲线
        ax.fill_between(td, mu - sig, mu + sig, alpha=0.2, color=c)
        ax.plot(td, mu, "-o", ms=3, lw=1.5, color=c,
                label=f"mean ± std (bootstrap)")

        ax.set_ylabel(f"{lbl} (Å)", fontsize=9)
        ax.grid(True, alpha=0.25)

        # legend：标注平衡值和 pre-pump 均值
        legend_lines = [
            matplotlib.lines.Line2D([0], [0], color="red",  ls="--", lw=1.0),
            matplotlib.lines.Line2D([0], [0], color="blue", ls="--", lw=1.0),
            matplotlib.lines.Line2D([0], [0], color=c, lw=1.5),
        ]
        legend_labels = []
        if eq is not None:
            legend_labels.append(f"equilibrium = {eq:.3f} Å")
        else:
            legend_labels.append("equilibrium N/A")
        legend_labels.append(f"pre-pump mean = {pre_pump_mean[i]:.3f} Å")
        legend_labels.append("mean ± std (bootstrap)")
        ax.legend(legend_lines, legend_labels, fontsize=7.5, loc="upper right")

    axes[-1].set_xlabel("Time delay (fs)", fontsize=10)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved: {out}")


if __name__ == "__main__":
    import matplotlib.lines
    main()
