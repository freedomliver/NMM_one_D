"""
Generate publication-quality figures for paper_nmm.

Usage:
  python plot_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config as cfg
import functions as fn


# Correct equilibrium distances
EQ = fn.label_from_backbone(cfg.NMM_BASE_COORDS_ANG)
NAMES = cfg.LABEL_PAIR_NAMES
COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

OUT_DIR = Path("paper_nmm/Figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig3_training_curves():
    """Fig 3: Training curves comparison (3-dim vs 7-dim vs 7-dim_eq)."""
    from plot_training import parse_log

    logs = {
        "3-dim 80K": "results/training/train_3dim_80k.log",
        "7-dim 80K": "results/training/train_7dim_80k.log",
        "7-dim 80K (eq)": "results/training/train_7dim_80k_eq.log",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    model_colors = ["#3498db", "#e74c3c", "#2ecc71"]
    model_ls = ["-", "--", "-"]

    for idx, (label, path) in enumerate(logs.items()):
        if not Path(path).exists():
            continue
        d = parse_log(path)
        if not d["epoch"]:
            continue
        c = model_colors[idx]
        ep = d["epoch"]

        # Left: val MAE
        axes[0].plot(ep, d["val_mae"], ls=model_ls[idx], lw=1.8, color=c, label=label)
        axes[0].scatter([ep[np.argmin(d["val_mae"])]], [min(d["val_mae"])],
                        s=40, color=c, zorder=5, edgecolors="k", linewidths=0.5)

    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Validation MAE (Å)", fontsize=11)
    axes[0].set_title("(a) Overall Validation MAE", fontsize=12)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(True, alpha=0.2)
    axes[0].set_ylim(0, 0.6)

    # Right: per-distance MAE for best model (7dim_eq)
    d = parse_log("results/training/train_7dim_80k_eq.log")
    ax = axes[1]
    for di, name in enumerate(d["dim_names"]):
        vals = d["dim_data"][name]
        ax.plot(d["epoch"][:len(vals)], vals, "-", lw=1.5,
                color=COLORS[di % len(COLORS)], label=name)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Per-Distance MAE (Å)", fontsize=11)
    ax.set_title("(b) Per-Distance MAE (7-dim eq)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_training_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] saved")


def fig4_parity():
    """Fig 4: Compact parity plots (2x4 grid)."""
    d = np.load("results/validation/synth_inference_7dim_eq.npz")
    y_hat, y_true = d["y_hat"], d["y_true"]
    ndim = y_hat.shape[1]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes_flat = axes.flatten()

    for i in range(ndim):
        ax = axes_flat[i]
        pred, true = y_hat[:, i], y_true[:, i]
        mae = np.abs(pred - true).mean()
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        margin = (hi - lo) * 0.05

        ax.hexbin(true, pred, gridsize=40, cmap="YlOrRd", mincnt=1)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{NAMES[i]}", fontsize=11, fontweight="bold")
        ax.annotate(f"MAE={mae:.3f} Å\nR²={r2:.3f}",
                    xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8,
                    va="top", bbox=dict(fc="white", alpha=0.8, pad=2))
        if i >= 4:
            ax.set_xlabel("True (Å)", fontsize=9)
        if i % 4 == 0:
            ax.set_ylabel("Predicted (Å)", fontsize=9)
        ax.tick_params(labelsize=7)

    # Summary bar chart
    ax = axes_flat[7]
    maes = np.abs(y_hat - y_true).mean(axis=0)
    bars = ax.bar(range(ndim), maes, color=COLORS[:ndim], edgecolor="k", linewidth=0.3)
    ax.set_xticks(range(ndim))
    ax.set_xticklabels(NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (Å)", fontsize=9)
    ax.set_title("Summary", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f"{val:.3f}", ha="center", fontsize=7)

    fig.suptitle("Validation: Predicted vs True (8,000 samples)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_parity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_parity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] saved")


def fig5_trajectories():
    """Fig 5: Time-resolved distance trajectories (compact 3+4 layout)."""
    td = sio.loadmat("TimeDelay.mat")["TimeDelay"].flatten()
    d = np.load("results/inference/inference_7dim_eq_alpha020.npz", allow_pickle=True)
    y_mean = d["y_mean"]
    n_t, n_dim = y_mean.shape

    fig, axes = plt.subplots(n_dim, 1, figsize=(8, 2.5 * n_dim), sharex=True)

    for i in range(n_dim):
        ax = axes[i]
        mu = y_mean[:, i]
        eq = EQ[i]

        ax.axvspan(td[0], 0, alpha=0.06, color="steelblue")
        ax.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.axhline(eq, color="red", lw=1.0, ls="--", alpha=0.6)
        ax.plot(td, mu, "-o", ms=2.5, lw=1.2, color=COLORS[i])

        pre_mean = mu[:9].mean()
        ax.axhline(pre_mean, color="blue", lw=0.8, ls=":", alpha=0.5)

        ax.set_ylabel(f"{NAMES[i]} (Å)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)

        # Compact legend
        ax.text(0.98, 0.95, f"eq={eq:.3f}  pre={pre_mean:.3f}",
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                bbox=dict(fc="white", alpha=0.8, pad=1.5))

    axes[-1].set_xlabel("Time delay (fs)", fontsize=10)
    fig.suptitle("Time-Resolved Distances (α=0.02, bootstrap mean)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_trajectories.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] saved")


if __name__ == "__main__":
    print("Generating publication figures...")
    fig3_training_curves()
    fig4_parity()
    fig5_trajectories()
    print(f"\nAll figures saved to {OUT_DIR}/")
