"""
Validation diagnostic plots for NMM ML project.

Generates parity plots and error analysis figures from synth inference results.

Usage:
    python plot_validation.py --npz results/validation/synth_7dim_80k.npz \
        --out_dir results/validation \
        --prefix 7dim_eq
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import LABEL_PAIR_NAMES


def load_data(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load y_hat and y_true from an npz file."""
    data = np.load(npz_path)
    y_hat = data["y_hat"]    # (N, 7)
    y_true = data["y_true"]  # (N, 7)
    return y_hat, y_true


def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def compute_r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ──────────────────────────────────────────────
# Figure 1: Parity Plots
# ──────────────────────────────────────────────

def plot_parity(y_hat: np.ndarray, y_true: np.ndarray, out_path: Path) -> None:
    """2x4 grid: 7 hexbin parity plots + 1 MAE bar chart."""
    ndim = y_hat.shape[1]
    names = LABEL_PAIR_NAMES[:ndim]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    maes = []
    for i in range(ndim):
        ax = axes_flat[i]
        pred = y_hat[:, i]
        true = y_true[:, i]

        mae = compute_mae(pred, true)
        r2 = compute_r2(pred, true)
        maes.append(mae)

        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        margin = (hi - lo) * 0.05
        lo -= margin
        hi += margin

        hb = ax.hexbin(true, pred, gridsize=50, cmap="YlOrRd", mincnt=1)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True (Å)", fontsize=10)
        ax.set_ylabel("Pred (Å)", fontsize=10)
        ax.set_title(names[i], fontsize=12, fontweight="bold")
        ax.annotate(
            f"MAE = {mae:.4f} Å\nR² = {r2:.4f}",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        fig.colorbar(hb, ax=ax, shrink=0.7)

    # Summary panel: MAE bar chart
    ax_sum = axes_flat[7]
    colors = plt.cm.Set2(np.linspace(0, 1, ndim))
    bars = ax_sum.bar(range(ndim), maes, color=colors, edgecolor="k", linewidth=0.5)
    ax_sum.set_xticks(range(ndim))
    ax_sum.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_sum.set_ylabel("MAE (Å)", fontsize=10)
    ax_sum.set_title("Overall MAE per dim", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, maes):
        ax_sum.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Parity Plots — Predicted vs True", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parity plot: {out_path}")


# ──────────────────────────────────────────────
# Figure 2: Error Analysis
# ──────────────────────────────────────────────

def plot_error_analysis(y_hat: np.ndarray, y_true: np.ndarray, out_path: Path) -> None:
    """
    3 rows:
      Row 1 (1x7): Error histograms per dimension
      Row 2 (1x7): Error vs true scatter (heteroscedasticity check)
      Row 3 (1x1): 7x7 absolute-error correlation matrix heatmap
    """
    ndim = y_hat.shape[1]
    names = LABEL_PAIR_NAMES[:ndim]
    errors = y_hat - y_true           # signed errors (N, 7)
    abs_errors = np.abs(errors)       # absolute errors (N, 7)

    fig = plt.figure(figsize=(22, 18))

    # Use GridSpec: rows 1&2 have 7 cols, row 3 spans full width
    gs = fig.add_gridspec(3, ndim, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.35)

    # ── Row 1: Error histograms ──
    for i in range(ndim):
        ax = fig.add_subplot(gs[0, i])
        err = errors[:, i]
        ax.hist(err, bins=60, color="steelblue", edgecolor="k", linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Error (Å)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(names[i], fontsize=11, fontweight="bold")
        ax.annotate(
            f"μ={np.mean(err):.4f}\nσ={np.std(err):.4f}",
            xy=(0.95, 0.92),
            xycoords="axes fraction",
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    # ── Row 2: Error vs true scatter ──
    for i in range(ndim):
        ax = fig.add_subplot(gs[1, i])
        true = y_true[:, i]
        err = errors[:, i]
        ax.scatter(true, err, s=1, alpha=0.3, c="teal", rasterized=True)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("True (Å)", fontsize=9)
        ax.set_ylabel("Error (Å)", fontsize=9)
        ax.set_title(names[i], fontsize=11, fontweight="bold")

    # ── Row 3: Error correlation matrix ──
    ax_corr = fig.add_subplot(gs[2, :])
    # Pearson correlation of absolute errors across dimensions
    corr = np.corrcoef(abs_errors.T)  # (7, 7)
    im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax_corr.set_xticks(range(ndim))
    ax_corr.set_xticklabels(names, fontsize=10, rotation=45, ha="right")
    ax_corr.set_yticks(range(ndim))
    ax_corr.set_yticklabels(names, fontsize=10)
    ax_corr.set_title("Absolute-Error Correlation Matrix", fontsize=12, fontweight="bold")

    # Annotate each cell
    for row in range(ndim):
        for col in range(ndim):
            val = corr[row, col]
            color = "white" if abs(val) > 0.6 else "black"
            ax_corr.text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax_corr, fraction=0.02, pad=0.02)

    fig.suptitle("Error Analysis", fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved error analysis: {out_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation diagnostic plots for NMM ML.")
    parser.add_argument("--npz", required=True, help="Path to inference npz file (keys: y_hat, y_true)")
    parser.add_argument("--out_dir", default="results/validation", help="Output directory for figures")
    parser.add_argument("--prefix", default="7dim_eq", help="Filename prefix for output figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_hat, y_true = load_data(args.npz)
    print(f"Loaded {y_hat.shape[0]} samples, {y_hat.shape[1]} dimensions from {args.npz}")

    plot_parity(y_hat, y_true, out_dir / f"parity_plots_{args.prefix}.png")
    plot_error_analysis(y_hat, y_true, out_dir / f"error_analysis_{args.prefix}.png")

    print("Done.")


if __name__ == "__main__":
    main()
