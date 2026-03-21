"""
从预测距离重建分子3D结构并可视化

输入: 7 个距离 [O-N, O-C5, N-C5, N-C2, N-C4, C5-C2, C5-C4]
输出: N3 和 C5 的 3D 坐标（固定原子 O7, C1, C2, C4, C6 已知）

重建步骤:
1. N3 定位: trilaterate(O7, C2, C4, d_ON, d_NC2, d_NC4) → 2 个镜像解
2. C5 定位: trilaterate(O7, C2, C4, d_OC5, d_C5C2, d_C5C4) → 2 个镜像解
3. 用 d(N-C5) 从 4 种组合中选出最匹配的解
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import config as cfg
import functions as fn


def reconstruct_from_distances(
    distances: np.ndarray,  # (7,) = [O-N, O-C5, N-C5, N-C2, N-C4, C5-C2, C5-C4]
    base_coords: np.ndarray = cfg.NMM_BASE_COORDS_ANG,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    从 7 个距离重建 N3 和 C5 的 3D 坐标。

    返回: (N3_xyz, C5_xyz, success)
    """
    iO  = cfg.NMM_ATOM_INDEX["O7"]
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]

    O  = base_coords[iO]
    C2 = base_coords[iC2]
    C4 = base_coords[iC4]

    d_ON, d_OC5, d_NC5, d_NC2, d_NC4, d_C5C2, d_C5C4 = distances

    try:
        # N3 定位: 3 个锚点 (O, C2, C4)，3 个距离
        N_sol1, N_sol2 = fn.trilaterate_three_spheres(O, C2, C4, d_ON, d_NC2, d_NC4, min_h=1e-6)

        # C5 定位: 3 个锚点 (O, C2, C4)，3 个距离
        C5_sol1, C5_sol2 = fn.trilaterate_three_spheres(O, C2, C4, d_OC5, d_C5C2, d_C5C4, min_h=1e-6)

        # 用 d(N-C5) 从 4 种组合中选最匹配的
        candidates = [
            (N_sol1, C5_sol1),
            (N_sol1, C5_sol2),
            (N_sol2, C5_sol1),
            (N_sol2, C5_sol2),
        ]
        best_err = float("inf")
        best_N, best_C5 = N_sol1, C5_sol1
        for N_cand, C5_cand in candidates:
            nc5_pred = float(np.linalg.norm(N_cand - C5_cand))
            err = abs(nc5_pred - d_NC5)
            if err < best_err:
                best_err = err
                best_N, best_C5 = N_cand, C5_cand

        return best_N, best_C5, True

    except ValueError:
        # 重建失败（距离不自洽），返回平衡态坐标
        iN  = cfg.NMM_ATOM_INDEX["N3"]
        iC5 = cfg.NMM_ATOM_INDEX["C5"]
        return base_coords[iN].copy(), base_coords[iC5].copy(), False


def reconstruct_trajectory(
    y_mean: np.ndarray,  # (T, 7)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对每个时间点重建 N3 和 C5 坐标。

    返回: (N3_traj, C5_traj, success_flags)
        N3_traj: (T, 3)
        C5_traj: (T, 3)
        success_flags: (T,) bool
    """
    n_t = y_mean.shape[0]
    N3_traj = np.zeros((n_t, 3), dtype=np.float64)
    C5_traj = np.zeros((n_t, 3), dtype=np.float64)
    success = np.zeros(n_t, dtype=bool)

    for t in range(n_t):
        N3_traj[t], C5_traj[t], success[t] = reconstruct_from_distances(y_mean[t])

    return N3_traj, C5_traj, success


def plot_structure_snapshots(
    N3_traj: np.ndarray,
    C5_traj: np.ndarray,
    success: np.ndarray,
    td: np.ndarray,
    out_path: str,
    n_frames: int = 6,
):
    """
    绘制多帧分子结构快照（3D），展示 N3 和 C5 随时间的运动。
    """
    base = cfg.NMM_BASE_COORDS_ANG
    iO  = cfg.NMM_ATOM_INDEX["O7"]
    iC1 = cfg.NMM_ATOM_INDEX["C1"]
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]
    iC6 = cfg.NMM_ATOM_INDEX["C6"]
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]

    fixed_atoms = {
        "C1": base[iC1], "C2": base[iC2], "C4": base[iC4],
        "C6": base[iC6], "O7": base[iO],
    }
    # 环骨架连接顺序: C1-C2-N3-C4-C6-O7-C1 (6元环)
    ring_order = ["C1", "C2", "N3", "C4", "C6", "O7"]

    # 选取时间帧
    n_t = len(td)
    # 选关键时刻: pre-pump, t=0附近, 以及几个后泵浦时刻
    frame_indices = [0, 4, 9, 15, 25, n_t-1]  # pre-pump, pre, pump rise, early, mid, late
    frame_indices = [i for i in frame_indices if i < n_t and success[i]]
    if len(frame_indices) < n_frames:
        # 补充成功重建的帧
        valid_frames = np.where(success)[0]
        step = max(1, len(valid_frames) // n_frames)
        frame_indices = valid_frames[::step][:n_frames].tolist()

    n_frames = len(frame_indices)
    n_cols = min(3, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle("NMM Structure Reconstruction from Predicted Distances",
                 fontsize=13, fontweight="bold")

    for panel_idx, t_idx in enumerate(frame_indices):
        ax = fig.add_subplot(n_rows, n_cols, panel_idx + 1, projection='3d')

        N3 = N3_traj[t_idx]
        C5 = C5_traj[t_idx]

        # 构建当前帧的所有原子位置
        atoms = dict(fixed_atoms)
        atoms["N3"] = N3
        atoms["C5"] = C5

        # 画固定原子（灰色）
        for name, pos in fixed_atoms.items():
            color = "red" if name == "O7" else "gray"
            ax.scatter(*pos, s=80, c=color, edgecolors="black", linewidths=0.5, zorder=5)
            ax.text(pos[0], pos[1], pos[2] + 0.15, name, fontsize=7, ha="center")

        # 画 N3（蓝色）和 C5（绿色）
        ax.scatter(*N3, s=120, c="blue", edgecolors="black", linewidths=0.5, zorder=5)
        ax.text(N3[0], N3[1], N3[2] + 0.15, "N3", fontsize=7, ha="center", color="blue")
        ax.scatter(*C5, s=120, c="green", edgecolors="black", linewidths=0.5, zorder=5)
        ax.text(C5[0], C5[1], C5[2] + 0.15, "C5", fontsize=7, ha="center", color="green")

        # 画平衡态 N3/C5 位置（半透明）
        ax.scatter(*base[iN], s=60, c="blue", alpha=0.2, marker="x", zorder=3)
        ax.scatter(*base[iC5], s=60, c="green", alpha=0.2, marker="x", zorder=3)

        # 画环骨架连接
        for i in range(len(ring_order)):
            a1 = ring_order[i]
            a2 = ring_order[(i + 1) % len(ring_order)]
            p1, p2 = atoms[a1], atoms[a2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    "k-", lw=1.5, alpha=0.6)

        # 画 N3-C5 键
        ax.plot([N3[0], C5[0]], [N3[1], C5[1]], [N3[2], C5[2]],
                "g--", lw=1.5, alpha=0.7)

        # 键长标注
        d_NC5 = float(np.linalg.norm(N3 - C5))
        d_ON  = float(np.linalg.norm(fixed_atoms["O7"] - N3))
        mid_nc5 = (N3 + C5) / 2
        mid_on  = (fixed_atoms["O7"] + N3) / 2
        ax.text(mid_nc5[0], mid_nc5[1], mid_nc5[2] + 0.2,
                f"{d_NC5:.2f}", fontsize=6, color="green", ha="center")
        ax.text(mid_on[0] + 0.2, mid_on[1], mid_on[2],
                f"{d_ON:.2f}", fontsize=6, color="red", ha="center")

        # 设置视角和标签
        ax.set_xlabel("X (Å)", fontsize=7)
        ax.set_ylabel("Y (Å)", fontsize=7)
        ax.set_zlabel("Z (Å)", fontsize=7)
        ax.set_title(f"t = {td[t_idx]:.0f} fs", fontsize=10)
        ax.view_init(elev=25, azim=135)

        # 统一轴范围
        all_pos = np.array(list(atoms.values()))
        center = all_pos.mean(axis=0)
        max_range = max(all_pos.max(axis=0) - all_pos.min(axis=0)) / 2 + 1.0
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[reconstruct] saved: {out_path}")


def plot_trajectory_2d(
    N3_traj: np.ndarray,
    C5_traj: np.ndarray,
    success: np.ndarray,
    td: np.ndarray,
    out_path: str,
):
    """
    绘制 N3 和 C5 随时间的坐标变化（2D 投影）。
    """
    base = cfg.NMM_BASE_COORDS_ANG
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle("N3 and C5 Coordinate Trajectories (Reconstructed)",
                 fontsize=13, fontweight="bold")

    mask = success
    coord_labels = ["X", "Y", "Z"]

    for j in range(3):
        # N3 坐标
        ax = axes[0, j]
        ax.axhline(base[iN, j], color="blue", ls="--", alpha=0.5, label=f"N3 eq ({base[iN,j]:.2f})")
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axvspan(td[0], 0, alpha=0.08, color="steelblue")
        ax.plot(td[mask], N3_traj[mask, j], "-o", ms=2, color="blue", label="N3 recon")
        if not mask.all():
            ax.plot(td[~mask], N3_traj[~mask, j], "x", ms=4, color="red", label="failed")
        ax.set_ylabel(f"N3 {coord_labels[j]} (Å)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

        # C5 坐标
        ax = axes[1, j]
        ax.axhline(base[iC5, j], color="green", ls="--", alpha=0.5, label=f"C5 eq ({base[iC5,j]:.2f})")
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axvspan(td[0], 0, alpha=0.08, color="steelblue")
        ax.plot(td[mask], C5_traj[mask, j], "-o", ms=2, color="green", label="C5 recon")
        if not mask.all():
            ax.plot(td[~mask], C5_traj[~mask, j], "x", ms=4, color="red", label="failed")
        ax.set_ylabel(f"C5 {coord_labels[j]} (Å)", fontsize=9)
        ax.set_xlabel("Time delay (fs)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[reconstruct] saved: {out_path}")


def plot_reconstruction_residuals(
    y_mean: np.ndarray,
    N3_traj: np.ndarray,
    C5_traj: np.ndarray,
    success: np.ndarray,
    td: np.ndarray,
    out_path: str,
):
    """
    Plot reconstruction residuals: |d_recon - d_input| for each distance at each time point.
    Validates that trilateration is self-consistent.
    """
    base = cfg.NMM_BASE_COORDS_ANG
    iO  = cfg.NMM_ATOM_INDEX["O7"]
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]
    O, C2, C4 = base[iO], base[iC2], base[iC4]

    n_t = len(td)
    pair_names = cfg.LABEL_PAIR_NAMES[:7]
    residuals = np.full((n_t, 7), np.nan)

    for t in range(n_t):
        if not success[t]:
            continue
        N3, C5 = N3_traj[t], C5_traj[t]
        d_recon = np.array([
            np.linalg.norm(O - N3),    # O-N
            np.linalg.norm(O - C5),    # O-C5
            np.linalg.norm(N3 - C5),   # N-C5
            np.linalg.norm(N3 - C2),   # N-C2
            np.linalg.norm(N3 - C4),   # N-C4
            np.linalg.norm(C5 - C2),   # C5-C2
            np.linalg.norm(C5 - C4),   # C5-C4
        ])
        residuals[t] = np.abs(d_recon - y_mean[t, :7])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Reconstruction Residuals: |d_reconstructed - d_input|",
                 fontsize=13, fontweight="bold")

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

    # Per-distance residual vs time
    ax = axes[0]
    for i, name in enumerate(pair_names):
        mask = ~np.isnan(residuals[:, i])
        ax.plot(td[mask], residuals[mask, i], "-o", ms=3, color=colors[i], label=name)
    ax.set_ylabel("Residual (Å)", fontsize=10)
    ax.set_title("Per-Distance Reconstruction Residual vs Time")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.25)
    ax.axvline(0, color="gray", ls="--", lw=0.8)

    # Total residual (mean across distances)
    ax = axes[1]
    mean_res = np.nanmean(residuals, axis=1)
    mask = ~np.isnan(mean_res)
    ax.bar(td[mask], mean_res[mask], width=np.diff(td).mean() * 0.6,
           color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.5)
    ax.set_ylabel("Mean Residual (Å)", fontsize=10)
    ax.set_xlabel("Time delay (fs)", fontsize=10)
    ax.set_title("Mean Reconstruction Residual")
    ax.grid(True, alpha=0.25)
    ax.axvline(0, color="gray", ls="--", lw=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[reconstruct] saved: {out_path}")


def plot_displacement_from_equilibrium(
    N3_traj: np.ndarray,
    C5_traj: np.ndarray,
    success: np.ndarray,
    td: np.ndarray,
    out_path: str,
):
    """
    Plot N3 and C5 displacement from equilibrium position vs time.
    Shows total displacement magnitude and per-axis components.
    """
    base = cfg.NMM_BASE_COORDS_ANG
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    N3_eq = base[iN]
    C5_eq = base[iC5]

    N3_disp = N3_traj - N3_eq[None, :]  # (T, 3)
    C5_disp = C5_traj - C5_eq[None, :]
    N3_mag = np.linalg.norm(N3_disp, axis=1)
    C5_mag = np.linalg.norm(C5_disp, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Atomic Displacement from Equilibrium",
                 fontsize=13, fontweight="bold")

    mask = success
    coord_labels = ["X", "Y", "Z"]

    # Total displacement magnitude
    ax = axes[0]
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axvspan(td[0], 0, alpha=0.08, color="steelblue")
    ax.plot(td[mask], N3_mag[mask], "-o", ms=3, color="blue", label="N3 |Δr|")
    ax.plot(td[mask], C5_mag[mask], "-o", ms=3, color="green", label="C5 |Δr|")
    ax.set_ylabel("Displacement (Å)", fontsize=10)
    ax.set_title("Total Displacement Magnitude")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # N3 per-axis displacement
    ax = axes[1]
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axvspan(td[0], 0, alpha=0.08, color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    for j, (lbl, c) in enumerate(zip(coord_labels, ["#e74c3c", "#2ecc71", "#3498db"])):
        ax.plot(td[mask], N3_disp[mask, j], "-o", ms=2, color=c, label=f"N3 Δ{lbl}")
    ax.set_ylabel("N3 Displacement (Å)", fontsize=10)
    ax.set_title("N3 Per-Axis Displacement from Equilibrium")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.25)

    # C5 per-axis displacement
    ax = axes[2]
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axvspan(td[0], 0, alpha=0.08, color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    for j, (lbl, c) in enumerate(zip(coord_labels, ["#e74c3c", "#2ecc71", "#3498db"])):
        ax.plot(td[mask], C5_disp[mask, j], "-o", ms=2, color=c, label=f"C5 Δ{lbl}")
    ax.set_ylabel("C5 Displacement (Å)", fontsize=10)
    ax.set_xlabel("Time delay (fs)", fontsize=10)
    ax.set_title("C5 Per-Axis Displacement from Equilibrium")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[reconstruct] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="推理结果 npz 文件（需含 y_mean, 7-dim）")
    ap.add_argument("--out_dir", type=str, default="results/reconstruction")
    ap.add_argument("--timedelay", type=str, default="TimeDelay.mat")
    args = ap.parse_args()

    td = sio.loadmat(args.timedelay)["TimeDelay"].flatten()

    d = np.load(args.npz, allow_pickle=True)
    y_mean = d["y_mean"]  # (45, 7)
    n_t, n_dim = y_mean.shape
    print(f"[reconstruct] loaded: {args.npz}, shape={y_mean.shape}")

    if n_dim < 7:
        print(f"[reconstruct] ERROR: need 7-dim distances for reconstruction, got {n_dim}")
        return

    # 重建轨迹
    N3_traj, C5_traj, success = reconstruct_trajectory(y_mean)
    n_success = success.sum()
    print(f"[reconstruct] success: {n_success}/{n_t} time points")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存重建结果
    np.savez(out_dir / "reconstruction.npz",
             N3_traj=N3_traj, C5_traj=C5_traj, success=success,
             td=td, y_mean=y_mean)

    # 3D 结构快照
    plot_structure_snapshots(N3_traj, C5_traj, success, td,
                            str(out_dir / "structure_snapshots.png"))

    # 坐标轨迹
    plot_trajectory_2d(N3_traj, C5_traj, success, td,
                       str(out_dir / "coordinate_trajectories.png"))

    # 重建残差图
    plot_reconstruction_residuals(y_mean, N3_traj, C5_traj, success, td,
                                  str(out_dir / "reconstruction_residuals.png"))

    # 平衡态位移图
    plot_displacement_from_equilibrium(N3_traj, C5_traj, success, td,
                                       str(out_dir / "displacement_from_equilibrium.png"))


if __name__ == "__main__":
    main()
