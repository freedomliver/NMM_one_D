"""
Visualize the experimental signal preprocessing pipeline.

Loads bootstrap data and shows 6 progressive preprocessing steps for
3 selected time points (pre-pump, pump onset, late).

Usage:
    python plot_signal_diagnostics.py --exp_dir bootstrapping/results_240 \
        --alpha 0.03 --out results/inference/preprocessing_diagnostics.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import config as cfg
import functions as fn
from inference import (
    EXP_S_GRID,
    EXP_S_LEN,
    load_all_bootstrap,
    _fill_nan,
    _baseline_correct,
    preprocess_single,
    compute_delta_calibration,
    preprocess_delta,
)


def generate_training_sample() -> np.ndarray:
    """Generate one synthetic delta-sM signal for visual comparison."""
    rng = np.random.default_rng(42)
    # Equilibrium signal
    coords_eq, _, elems_eq = fn.build_full_coords_with_locked_H(cfg.NMM_BASE_COORDS_ANG)
    sM_ground = fn.compute_1d_scattering_signal(coords_eq, elems_eq)  # (681,)

    # Try random configurations until one succeeds
    for _ in range(200):
        dof = fn.sample_dof(rng)
        try:
            backbone = fn.generate_backbone_coords_from_dof(dof)
            coords_all, _names, elems = fn.build_full_coords_with_locked_H(backbone)
            sig = fn.compute_1d_scattering_signal(coords_all, elems)
            delta_sM = sig - sM_ground
            delta_sM = fn.add_noise(delta_sM, rng)
            return delta_sM  # (681,)
        except Exception:
            continue
    # Fallback: return zeros
    return np.zeros(cfg.S_LEN, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Preprocessing diagnostics plot")
    ap.add_argument("--exp_dir", type=str,
                    default=str(cfg.PATHS.root / "bootstrapping" / "results_240"))
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--prepump_end", type=int, default=9)
    ap.add_argument("--s_low_mask", type=float, default=1.5)
    ap.add_argument("--s_high_mask", type=float, default=9.0)
    ap.add_argument("--baseline_start", type=float, default=5.0)
    ap.add_argument("--out", type=str,
                    default="results/inference/preprocessing_diagnostics.png")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load bootstrap data
    # ------------------------------------------------------------------
    exp_dir = Path(args.exp_dir)
    raw = load_all_bootstrap(exp_dir)  # (B, T, 637)
    raw_mean = raw.mean(axis=0)        # (T, 637)

    # Selected time points
    time_indices = [4, 9, 35]
    time_labels = ["t=4 (pre-pump)", "t=9 (pump onset)", "t=35 (late)"]
    # Clamp to available range
    n_t = raw_mean.shape[0]
    time_indices = [min(t, n_t - 1) for t in time_indices]

    # ------------------------------------------------------------------
    # Calibration: compute pre-pump reference and alpha
    # ------------------------------------------------------------------
    pre_pump_637, alpha, sM_ground = compute_delta_calibration(
        raw, args.prepump_end, alpha_override=args.alpha
    )

    # Pre-pump reference on model grid (for delta subtraction)
    s0_ref_model = preprocess_single(
        pre_pump_637,
        s_low=args.s_low_mask,
        s_high=args.s_high_mask,
        bl_start=args.baseline_start,
    )  # (681,)

    # ------------------------------------------------------------------
    # Generate a training sample for row 6 overlay
    # ------------------------------------------------------------------
    training_sample = generate_training_sample()  # (681,)

    # ------------------------------------------------------------------
    # Build the 6-row x 3-col figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(6, 3, figsize=(18, 20), dpi=180)

    row_labels = [
        "1. Raw signal (637 pts)",
        "2. After NaN fill",
        "3. After ref subtraction",
        "4. After baseline correction",
        "5. After alpha scaling",
        "6. Final (681 pts) + training",
    ]

    for col, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
        sig_raw = raw_mean[tidx].copy()  # (637,)

        # ---- Row 0: Raw signal ----
        ax = axes[0, col]
        ax.plot(EXP_S_GRID, sig_raw, color="gray", linewidth=0.6)
        ax.set_title(tlabel, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(row_labels[0], fontsize=9)
        ax.set_xlim(0, 12)
        ax.tick_params(labelsize=8)

        # ---- Row 1: After NaN fill ----
        sig_filled = _fill_nan(sig_raw)
        ax = axes[1, col]
        ax.plot(EXP_S_GRID, sig_filled, color="gray", linewidth=0.6)
        if col == 0:
            ax.set_ylabel(row_labels[1], fontsize=9)
        ax.set_xlim(0, 12)
        ax.tick_params(labelsize=8)

        # ---- Row 2: After reference subtraction ----
        sig_refsub = sig_filled - pre_pump_637
        ax = axes[2, col]
        ax.plot(EXP_S_GRID, sig_refsub, color="gray", linewidth=0.6)
        if col == 0:
            ax.set_ylabel(row_labels[2], fontsize=9)
        ax.set_xlim(0, 12)
        ax.tick_params(labelsize=8)

        # ---- Row 3: After baseline correction (2nd order poly, s>5.0) ----
        sig_blcorr = _baseline_correct(sig_refsub, EXP_S_GRID, args.baseline_start)
        ax = axes[3, col]
        ax.plot(EXP_S_GRID, sig_blcorr, color="blue", linewidth=0.6)
        ax.axvline(args.baseline_start, color="orange", linestyle="--",
                   linewidth=0.5, alpha=0.7, label=f"bl_start={args.baseline_start}")
        if col == 0:
            ax.set_ylabel(row_labels[3], fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, 12)
        ax.tick_params(labelsize=8)

        # ---- Row 4: After alpha scaling ----
        sig_scaled = sig_blcorr * alpha
        ax = axes[4, col]
        ax.plot(EXP_S_GRID, sig_scaled, color="blue", linewidth=0.6)
        if col == 0:
            ax.set_ylabel(row_labels[4], fontsize=9)
        ax.set_xlim(0, 12)
        ax.tick_params(labelsize=8)
        ax.text(0.97, 0.95, f"alpha={alpha:.4f}", transform=ax.transAxes,
                fontsize=7, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        # ---- Row 5: Final signal (range mask + interp to 681 pts) ----
        # Reproduce the full pipeline to get the final 681-pt signal:
        # preprocess_single does: NaN fill -> baseline correct -> mask -> interp
        # Then preprocess_delta does: (sig - s0_ref) * alpha with range zeroing
        sig_preprocessed = preprocess_single(
            sig_raw,
            s_low=args.s_low_mask,
            s_high=args.s_high_mask,
            bl_start=args.baseline_start,
        )  # (681,)
        sig_final = preprocess_delta(sig_preprocessed, s0_ref_model, alpha)  # (681,)

        ax = axes[5, col]
        ax.plot(cfg.S_GRID, sig_final, color="blue", linewidth=0.7, label="Exp final")
        ax.plot(cfg.S_GRID, training_sample, color="red", linewidth=0.5,
                alpha=0.5, label="Training sample")
        ax.axvline(args.s_low_mask, color="green", linestyle=":", linewidth=0.5,
                   alpha=0.6)
        ax.axvline(args.s_high_mask, color="green", linestyle=":", linewidth=0.5,
                   alpha=0.6)
        if col == 0:
            ax.set_ylabel(row_labels[5], fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, 12)
        ax.tick_params(labelsize=8)
        ax.set_xlabel(r"$s\;(\AA^{-1})$", fontsize=9)

    # ------------------------------------------------------------------
    # Global formatting
    # ------------------------------------------------------------------
    fig.suptitle(
        "Experimental Signal Preprocessing Pipeline (bootstrap mean)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"[plot] Saved preprocessing diagnostics to {out_path}")


if __name__ == "__main__":
    main()
