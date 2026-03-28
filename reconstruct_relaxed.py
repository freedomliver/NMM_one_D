"""
解析重建失败时的 relaxed fallback 重建。

策略：
1. 先调用现有的物理约束解析重建（N3 同侧硬约束，C5 同侧优先）
2. 对解析失败的帧，再做数值优化：
   - 允许 O7, N3, C5 共同微调
   - C2/C4 仍固定
   - O7 受强正则约束，保持接近基态
   - N3/C5 受弱参考正则约束，并保持与基态同侧
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

import config as cfg
import reconstruct as rec


def _signed_height(point: np.ndarray, O: np.ndarray, C2: np.ndarray, C4: np.ndarray) -> float:
    normal = np.cross(C2 - O, C4 - O)
    n = float(np.linalg.norm(normal))
    if n < 1e-12:
        return 0.0
    normal = normal / n
    return float(np.dot(point - O, normal))


def _interp_refs(traj: np.ndarray, success: np.ndarray) -> np.ndarray:
    out = traj.copy()
    idx = np.arange(len(traj))
    ok = np.where(success)[0]
    if len(ok) == 0:
        return out
    for dim in range(traj.shape[1]):
        out[:, dim] = np.interp(idx, ok, traj[ok, dim])
    return out


def _distance_residuals(O: np.ndarray, N: np.ndarray, C5: np.ndarray, C2: np.ndarray, C4: np.ndarray,
                        target: np.ndarray) -> np.ndarray:
    d = np.array([
        np.linalg.norm(O - N),
        np.linalg.norm(O - C5),
        np.linalg.norm(N - C5),
        np.linalg.norm(N - C2),
        np.linalg.norm(N - C4),
        np.linalg.norm(C5 - C2),
        np.linalg.norm(C5 - C4),
    ], dtype=np.float64)
    return d - target[:7]


def _relaxed_optimize_frame(
    target: np.ndarray,
    O_eq: np.ndarray,
    N_ref: np.ndarray,
    C5_ref: np.ndarray,
    C2: np.ndarray,
    C4: np.ndarray,
    lambda_O: float = 6.0,
    lambda_N: float = 0.8,
    lambda_C5: float = 0.8,
    lambda_side: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    eq_sign_N = np.sign(_signed_height(cfg.NMM_BASE_COORDS_ANG[cfg.NMM_ATOM_INDEX["N3"]], O_eq, C2, C4))
    eq_sign_C5 = np.sign(_signed_height(cfg.NMM_BASE_COORDS_ANG[cfg.NMM_ATOM_INDEX["C5"]], O_eq, C2, C4))
    if abs(eq_sign_N) < 1e-12:
        eq_sign_N = 1.0
    if abs(eq_sign_C5) < 1e-12:
        eq_sign_C5 = 1.0

    x0 = np.concatenate([O_eq, N_ref, C5_ref], axis=0).astype(np.float64)

    def fun(x: np.ndarray) -> np.ndarray:
        O = x[0:3]
        N = x[3:6]
        C5 = x[6:9]
        res = []
        res.extend(_distance_residuals(O, N, C5, C2, C4, target))
        res.extend(np.sqrt(lambda_O) * (O - O_eq))
        res.extend(np.sqrt(lambda_N) * (N - N_ref))
        res.extend(np.sqrt(lambda_C5) * (C5 - C5_ref))

        n_side = eq_sign_N * _signed_height(N, O, C2, C4)
        c5_side = eq_sign_C5 * _signed_height(C5, O, C2, C4)
        if n_side < 0:
            res.append(np.sqrt(lambda_side) * (-n_side))
        if c5_side < 0:
            res.append(np.sqrt(lambda_side) * (-c5_side))
        return np.asarray(res, dtype=np.float64)

    sol = least_squares(fun, x0=x0, method="trf", max_nfev=400)
    O = sol.x[0:3]
    N = sol.x[3:6]
    C5 = sol.x[6:9]
    dist_res = _distance_residuals(O, N, C5, C2, C4, target)
    mean_abs = float(np.mean(np.abs(dist_res)))
    ok = bool(sol.success and mean_abs < 0.12)
    return O, N, C5, mean_abs, ok


def reconstruct_relaxed_trajectory(
    y_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = cfg.NMM_BASE_COORDS_ANG
    iO = cfg.NMM_ATOM_INDEX["O7"]
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]
    O_eq = np.asarray(base[iO], dtype=np.float64)
    C2 = np.asarray(base[iC2], dtype=np.float64)
    C4 = np.asarray(base[iC4], dtype=np.float64)

    N_base, C5_base, success_base = rec.reconstruct_trajectory(y_mean)
    N_ref = _interp_refs(N_base, success_base)
    C5_ref = _interp_refs(C5_base, success_base)

    n_t = y_mean.shape[0]
    N_traj = N_base.copy()
    C5_traj = C5_base.copy()
    success = success_base.copy()
    fallback_mask = np.zeros(n_t, dtype=bool)
    O_traj = np.repeat(O_eq[None, :], n_t, axis=0)

    for t in range(n_t):
        if success[t]:
            continue
        O_opt, N_opt, C5_opt, _mean_abs, ok = _relaxed_optimize_frame(
            y_mean[t], O_eq=O_eq, N_ref=N_ref[t], C5_ref=C5_ref[t], C2=C2, C4=C4
        )
        O_traj[t] = O_opt
        if ok:
            N_traj[t] = N_opt
            C5_traj[t] = C5_opt
            success[t] = True
            fallback_mask[t] = True
        else:
            O_traj[t] = O_eq
    return O_traj, N_traj, C5_traj, success, fallback_mask


def bootstrap_displacement_stats(
    y_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个 bootstrap 单独重建，统计 N3/C5 总位移 |Δr| 的 std。
    返回:
      N3_mag_std: (T,)
      C5_mag_std: (T,)
    """
    base = cfg.NMM_BASE_COORDS_ANG
    iN = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    N_eq = base[iN]
    C5_eq = base[iC5]

    n_b, n_t, _ = y_all.shape
    n3_mag_all = np.zeros((n_b, n_t), dtype=np.float32)
    c5_mag_all = np.zeros((n_b, n_t), dtype=np.float32)
    for b in range(n_b):
        _O, N_traj, C5_traj, _success, _fallback = reconstruct_relaxed_trajectory(y_all[b])
        n3_mag_all[b] = np.linalg.norm(N_traj - N_eq[None, :], axis=1).astype(np.float32)
        c5_mag_all[b] = np.linalg.norm(C5_traj - C5_eq[None, :], axis=1).astype(np.float32)
    return n3_mag_all.std(axis=0), c5_mag_all.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/reconstruction_relaxed")
    ap.add_argument("--timedelay", type=str, default="TimeDelay.mat")
    ap.add_argument("--n_frames", type=int, default=12)
    ap.add_argument("--view_elev", type=float, default=0.0)
    ap.add_argument("--view_azim", type=float, default=0.0)
    ap.add_argument("--view_roll", type=float, default=270.0)
    args = ap.parse_args()

    td = sio.loadmat(args.timedelay)["TimeDelay"].flatten()
    d = np.load(args.npz, allow_pickle=True)
    y_mean = d["y_mean"]
    print(f"[reconstruct_relaxed] loaded: {args.npz}, shape={y_mean.shape}")
    y_all = d["y_all"] if "y_all" in d.files else None

    O_traj, N3_traj, C5_traj, success, fallback_mask = reconstruct_relaxed_trajectory(y_mean)
    print(f"[reconstruct_relaxed] success: {int(success.sum())}/{len(success)} time points")
    print(f"[reconstruct_relaxed] fallback recovered: {int(fallback_mask.sum())} time points")

    N3_mag_std = None
    C5_mag_std = None
    if y_all is not None:
        print(f"[reconstruct_relaxed] bootstrap trajectories: {y_all.shape[0]} samples")
        N3_mag_std, C5_mag_std = bootstrap_displacement_stats(y_all)
        print(
            f"[reconstruct_relaxed] bootstrap disp std | "
            f"N3 mean/max = {float(N3_mag_std.mean()):.4f}/{float(N3_mag_std.max()):.4f} A, "
            f"C5 mean/max = {float(C5_mag_std.mean()):.4f}/{float(C5_mag_std.max()):.4f} A"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "reconstruction.npz",
             O_traj=O_traj, N3_traj=N3_traj, C5_traj=C5_traj,
             success=success, fallback_mask=fallback_mask, td=td, y_mean=y_mean,
             N3_mag_std=N3_mag_std, C5_mag_std=C5_mag_std)

    rec.plot_structure_snapshots(N3_traj, C5_traj, success, td,
                                 str(out_dir / "structure_snapshots.png"),
                                 n_frames=args.n_frames,
                                 view_elev=args.view_elev,
                                 view_azim=args.view_azim,
                                 view_roll=args.view_roll)
    rec.plot_trajectory_2d(N3_traj, C5_traj, success, td,
                           str(out_dir / "coordinate_trajectories.png"))
    rec.plot_reconstruction_residuals(y_mean, N3_traj, C5_traj, success, td,
                                      str(out_dir / "reconstruction_residuals.png"))
    rec.plot_displacement_from_equilibrium(N3_traj, C5_traj, success, td,
                                           str(out_dir / "displacement_from_equilibrium.png"),
                                           N3_mag_std=N3_mag_std, C5_mag_std=C5_mag_std)
    rec.plot_trajectory_projections(N3_traj, C5_traj, success, td,
                                    str(out_dir / "trajectory_projections.png"))


if __name__ == "__main__":
    main()
