from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import functions as fn


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as h5f:
        x = np.array(h5f["x"], dtype=np.float32)
        y = np.array(h5f["y"], dtype=np.float32) if "y" in h5f else None
        eq = np.array(h5f["equilibrium"], dtype=np.float32) if "equilibrium" in h5f else None
    return x, y, eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_h5", type=str, required=True)
    ap.add_argument("--exp_h5", type=str, required=True)
    ap.add_argument("--out_h5", type=str, required=True)
    ap.add_argument("--norm_json", type=str, required=True)
    ap.add_argument("--exp_repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260327)
    args = ap.parse_args()

    synthetic_h5 = Path(args.synthetic_h5)
    exp_h5 = Path(args.exp_h5)
    out_h5 = Path(args.out_h5)
    norm_json = Path(args.norm_json)

    x_syn, y_syn, _eq_syn = load_xy(synthetic_h5)
    x_exp, y_exp, eq_exp = load_xy(exp_h5)
    if args.exp_repeat < 1:
        raise ValueError("--exp_repeat must be >= 1")

    if y_exp is None or y_exp.shape[1] != y_syn.shape[1]:
        eq_now = fn.equilibrium_labels_for_dim(y_syn.shape[1]).astype(np.float32)
        y_exp = np.repeat(eq_now[None, :], len(x_exp), axis=0).astype(np.float32)
        src = "current equilibrium labels"
        if eq_exp is not None and len(eq_exp) == y_syn.shape[1]:
            y_exp = np.repeat(eq_exp[None, :], len(x_exp), axis=0).astype(np.float32)
            src = "exp_h5 equilibrium"
        print(
            f"[merge] exp labels regenerated from {src} "
            f"(exp_dim={0 if y_exp is None else y_exp.shape[1]}, train_dim={y_syn.shape[1]})"
        )

    x_exp_rep = np.concatenate([x_exp] * args.exp_repeat, axis=0)
    y_exp_rep = np.concatenate([y_exp] * args.exp_repeat, axis=0)

    x_all = np.concatenate([x_syn, x_exp_rep], axis=0)
    y_all = np.concatenate([y_syn, y_exp_rep], axis=0)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x_all))
    x_all = x_all[perm]
    y_all = y_all[perm]

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h5f:
        h5f.create_dataset("x", data=x_all, compression="gzip", compression_opts=4, shuffle=True)
        h5f.create_dataset("y", data=y_all, compression="gzip", compression_opts=4, shuffle=True)
        h5f.attrs["synthetic_h5"] = str(synthetic_h5)
        h5f.attrs["exp_h5"] = str(exp_h5)
        h5f.attrs["exp_repeat"] = int(args.exp_repeat)
        h5f.attrs["n_synthetic"] = int(len(x_syn))
        h5f.attrs["n_experimental_raw"] = int(len(x_exp))
        h5f.attrs["n_experimental_merged"] = int(len(x_exp_rep))

    norm = fn.Normalization(
        x_mean=x_all.mean(axis=0).astype(np.float32),
        x_std=(x_all.std(axis=0) + 1e-8).astype(np.float32),
        y_mean=y_all.mean(axis=0).astype(np.float32),
        y_std=(y_all.std(axis=0) + 1e-8).astype(np.float32),
    )
    fn.save_normalization(norm, norm_json)

    print(
        f"[merge] synthetic={len(x_syn)} exp_raw={len(x_exp)} exp_repeat={args.exp_repeat} "
        f"exp_merged={len(x_exp_rep)} total={len(x_all)}"
    )
    print(f"[merge] saved train -> {out_h5}")
    print(f"[merge] saved norm  -> {norm_json}")


if __name__ == "__main__":
    main()
