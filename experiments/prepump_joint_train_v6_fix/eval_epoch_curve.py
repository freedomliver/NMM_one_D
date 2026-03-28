from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
import functions as fn
import inference as inf


def evaluate_ckpt(ckpt_path: Path, alpha: float, exp_dir: Path, prepump_end: int,
                  s_low_mask: float, s_high_mask: float, baseline_start: float):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, norm, meta = inf.load_model(ckpt_path, device)
    s_model = meta["input_grid"]
    out_dim = meta["out_dim"]
    label_names = list(meta["label_names"])
    eq = fn.equilibrium_labels_for_dim(out_dim)

    raw = inf.load_all_bootstrap(exp_dir, "s0")
    n_b, n_t, _ = raw.shape

    pre_pump_637, _, _ = inf.compute_delta_calibration(raw, prepump_end, alpha_override=alpha)
    s0_ref = inf.preprocess_single(
        pre_pump_637,
        s_model=s_model,
        s_low=s_low_mask,
        s_high=s_high_mask,
        bl_start=baseline_start,
    )

    final = np.zeros((n_b, n_t, len(s_model)), dtype=np.float32)
    for b in range(n_b):
        preprocessed = inf.batch_preprocess(
            raw[b],
            s_model=s_model,
            s_low=s_low_mask,
            s_high=s_high_mask,
            bl_start=baseline_start,
        )
        for t in range(n_t):
            final[b, t] = inf.preprocess_delta(
                preprocessed[t],
                s0_ref,
                alpha,
                s_model=s_model,
                s_low=s_low_mask,
                s_high=s_high_mask,
            )

    y_all = np.zeros((n_b, n_t, out_dim), dtype=np.float32)
    for b in range(n_b):
        y_all[b] = inf.predict_batch(model, norm, final[b], device)

    y_mean = y_all.mean(axis=0)
    y_std = y_all.std(axis=0)
    prepump_pred = y_mean[:prepump_end].mean(axis=0)

    dev_all = np.abs(prepump_pred - eq)
    dev7 = np.abs(prepump_pred[:7] - eq[:7])
    span = y_mean.max(axis=0) - y_mean.min(axis=0)

    return {
        "epoch": int(ckpt_path.stem.split("_")[-1]),
        "ckpt": str(ckpt_path),
        "prepump_dev_7": float(dev7.mean()),
        "prepump_key_7": float(dev7[:3].mean()),
        "prepump_dev_all": float(dev_all.mean()),
        "span_n_c5": float(span[2]),
        "span_oc5": float(span[1]),
        "span_hc5": float(span[7]) if len(span) > 7 else np.nan,
        "label_names": label_names,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--exp_dir", type=str, default=str(cfg.PATHS.root / "bootstrapping" / "results_240"))
    ap.add_argument("--alpha", type=float, default=0.015)
    ap.add_argument("--prepump_end", type=int, default=7)
    ap.add_argument("--s_low_mask", type=float, default=1.5)
    ap.add_argument("--s_high_mask", type=float, default=9.0)
    ap.add_argument("--baseline_start", type=float, default=5.0)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_times_csv", type=str, default=None)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no epoch_*.pt in {ckpt_dir}")

    rows = []
    times_rows = []
    for ckpt in ckpts:
        row = evaluate_ckpt(
            ckpt_path=ckpt,
            alpha=args.alpha,
            exp_dir=Path(args.exp_dir),
            prepump_end=args.prepump_end,
            s_low_mask=args.s_low_mask,
            s_high_mask=args.s_high_mask,
            baseline_start=args.baseline_start,
        )
        rows.append(row)
        label_names = row["label_names"]
        y_mean = row["y_mean"]
        y_std = row["y_std"]
        for t in range(y_mean.shape[0]):
            item = {
                "epoch": row["epoch"],
                "time_idx": t,
            }
            for i, name in enumerate(label_names):
                safe = name.replace("-", "_")
                item[f"{safe}_mean"] = float(y_mean[t, i])
                item[f"{safe}_std"] = float(y_std[t, i])
            times_rows.append(item)
        print(
            f"[epoch {row['epoch']:03d}] prepump7={row['prepump_dev_7']:.6f} "
            f"key7={row['prepump_key_7']:.6f} span(N-C5)={row['span_n_c5']:.4f}"
        )

    rows = sorted(rows, key=lambda r: r["epoch"])
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [k for k in rows[0].keys() if k != "label_names"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    if args.out_times_csv:
        out_times_csv = Path(args.out_times_csv)
        out_times_csv.parent.mkdir(parents=True, exist_ok=True)
        times_rows = sorted(times_rows, key=lambda r: (r["epoch"], r["time_idx"]))
        time_fields = list(times_rows[0].keys())
        with out_times_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=time_fields)
            writer.writeheader()
            for row in times_rows:
                writer.writerow(row)

    epochs = np.array([r["epoch"] for r in rows], dtype=int)
    prepump_dev_7 = np.array([r["prepump_dev_7"] for r in rows], dtype=float)
    prepump_key_7 = np.array([r["prepump_key_7"] for r in rows], dtype=float)
    span_oc5 = np.array([r["span_oc5"] for r in rows], dtype=float)
    span_n_c5 = np.array([r["span_n_c5"] for r in rows], dtype=float)
    span_hc5 = np.array([r["span_hc5"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, prepump_dev_7, "-o", label="pre-pump dev (7)")
    ax.plot(epochs, prepump_key_7, "-o", label="pre-pump key (7)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean absolute deviation (A)")
    ax.set_title("Epoch vs Experimental Quality")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, span_oc5, "-o", label="span O-C5")
    ax.plot(epochs, span_n_c5, "-o", label="span N-C5")
    if np.isfinite(span_hc5).any():
        ax.plot(epochs, span_hc5, "-o", label="span h_C5_signed")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time span (A)")
    ax.set_title("Epoch vs Dynamics Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"[saved] csv -> {out_csv}")
    if args.out_times_csv:
        print(f"[saved] times csv -> {out_times_csv}")
    print(f"[saved] png -> {out_png}")


if __name__ == "__main__":
    main()
