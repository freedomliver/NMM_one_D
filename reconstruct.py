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


def aligned_reference_positions() -> dict[str, np.ndarray]:
    """
    将参考态 NMM_pl / NMM_ax 刚体对齐到基态固定骨架，返回对齐后的关键位置。
    这些位置只用于可视化参考，不参与重建本身。
    """
    # 仅用四个环平面碳原子对齐，避免 O7 的微小结构差异把平面参考拉偏。
    anchor_labels = ["C1", "C2", "C4", "C6"]
    eq = {
        label: np.asarray(cfg.NMM_BASE_COORDS_ANG[cfg.NMM_ATOM_INDEX[label]], dtype=np.float64)
        for label in ["C1", "C2", "N3", "C4", "C5", "C6", "O7"]
    }

    out: dict[str, np.ndarray] = {}
    target_anchor = np.stack([eq[l] for l in anchor_labels], axis=0)
    for state_name in ["NMM_pl.xyz", "NMM_ax.xyz"]:
        state = fn._load_reference_state_xyz(state_name)
        moving_anchor = np.stack([state[l] for l in anchor_labels], axis=0)
        _aligned, r, t = fn._kabsch_align(moving_anchor, target_anchor)
        for key in state:
            state[key] = state[key] @ r + t
        tag = "pl" if "pl" in state_name else "ax"
        out[f"{tag}_C5"] = state["C5"]
        out[f"{tag}_N3"] = state["N3"]
    return out


def reconstruct_from_distances(
    distances: np.ndarray,  # (7,) = [O-N, O-C5, N-C5, N-C2, N-C4, C5-C2, C5-C4]
    base_coords: np.ndarray = cfg.NMM_BASE_COORDS_ANG,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    从 7 个距离重建 N3 和 C5 的 3D 坐标。

    返回: (N3_xyz, C5_xyz, success)
    """
    candidates = reconstruct_candidate_solutions(distances, base_coords=base_coords)
    if candidates:
        best_N, best_C5, _best_err = choose_preferred_candidate(candidates, base_coords=base_coords)
        return best_N.copy(), best_C5.copy(), True

    # 重建失败（距离不自洽），返回平衡态坐标
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    return base_coords[iN].copy(), base_coords[iC5].copy(), False


def reconstruct_candidate_solutions(
    distances: np.ndarray,
    base_coords: np.ndarray = cfg.NMM_BASE_COORDS_ANG,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    返回当前 7 个距离对应的 4 组候选镜像解及其局部 N-C5 误差。
    若三球定位失败，则返回空列表。
    """
    iO  = cfg.NMM_ATOM_INDEX["O7"]
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]

    O  = base_coords[iO]
    C2 = base_coords[iC2]
    C4 = base_coords[iC4]

    d_ON, d_OC5, d_NC5, d_NC2, d_NC4, d_C5C2, d_C5C4 = distances

    try:
        N_sol1, N_sol2 = fn.trilaterate_three_spheres(O, C2, C4, d_ON, d_NC2, d_NC4, min_h=1e-6)
        C5_sol1, C5_sol2 = fn.trilaterate_three_spheres(O, C2, C4, d_OC5, d_C5C2, d_C5C4, min_h=1e-6)
    except ValueError:
        return []

    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for N_cand, C5_cand in [
        (N_sol1, C5_sol1),
        (N_sol1, C5_sol2),
        (N_sol2, C5_sol1),
        (N_sol2, C5_sol2),
    ]:
        nc5_pred = float(np.linalg.norm(N_cand - C5_cand))
        err = abs(nc5_pred - d_NC5)
        out.append((
            np.asarray(N_cand, dtype=np.float64),
            np.asarray(C5_cand, dtype=np.float64),
            float(err),
        ))
    return out


def _c5_plane_signed_distance(
    c5_xyz: np.ndarray,
    base_coords: np.ndarray = cfg.NMM_BASE_COORDS_ANG,
) -> float:
    """
    计算 C5 相对 O-C2-C4 平面的有符号距离。
    约定“上方/同侧”由基态 C5 的符号决定。
    """
    iO = cfg.NMM_ATOM_INDEX["O7"]
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]
    O = np.asarray(base_coords[iO], dtype=np.float64)
    C2 = np.asarray(base_coords[iC2], dtype=np.float64)
    C4 = np.asarray(base_coords[iC4], dtype=np.float64)
    normal = np.cross(C2 - O, C4 - O)
    n = float(np.linalg.norm(normal))
    if n < 1e-12:
        return 0.0
    normal = normal / n
    return float(np.dot(np.asarray(c5_xyz, dtype=np.float64) - O, normal))


def choose_preferred_candidate(
    candidates: list[tuple[np.ndarray, np.ndarray, float]],
    base_coords: np.ndarray = cfg.NMM_BASE_COORDS_ANG,
    tie_tol: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    优先按单帧 N-C5 误差选解；当上下翻转两支误差接近时，
    优先选择与基态 C5 位于锚点平面同一侧的候选。
    """
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    eq_C5 = np.asarray(base_coords[iC5], dtype=np.float64)
    eq_sign = np.sign(_c5_plane_signed_distance(eq_C5, base_coords=base_coords))
    if abs(eq_sign) < 1e-12:
        eq_sign = 1.0

    best_err = min(item[2] for item in candidates)
    near_best = [item for item in candidates if item[2] <= best_err + tie_tol]
    same_side = [
        item for item in near_best
        if np.sign(_c5_plane_signed_distance(item[1], base_coords=base_coords)) == eq_sign
    ]
    pool = same_side if same_side else near_best
    return min(pool, key=lambda item: item[2])


def _choose_n_same_side(
    n_candidates: list[np.ndarray],
    base_coords: np.ndarray = cfg.NMM_BASE_COORDS_ANG,
) -> np.ndarray | None:
    """
    N3 采用硬约束：优先选与基态 N3 位于锚点平面同侧的解。
    若没有同侧解，则返回 None，交由上层判为失败。
    """
    iN = cfg.NMM_ATOM_INDEX["N3"]
    eq_N = np.asarray(base_coords[iN], dtype=np.float64)
    eq_sign = np.sign(_c5_plane_signed_distance(eq_N, base_coords=base_coords))
    if abs(eq_sign) < 1e-12:
        eq_sign = 1.0

    same_side = [
        cand for cand in n_candidates
        if np.sign(_c5_plane_signed_distance(cand, base_coords=base_coords)) == eq_sign
    ]
    if not same_side:
        return None
    return min(same_side, key=lambda cand: abs(_c5_plane_signed_distance(cand, base_coords=base_coords)
                                               - _c5_plane_signed_distance(eq_N, base_coords=base_coords)))


def reconstruct_from_distances_physically_constrained(
    distances: np.ndarray,
    base_coords: np.ndarray = cfg.NMM_BASE_COORDS_ANG,
    c5_tie_tol: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    带物理先验的重建：
    1. N3 不允许翻面，必须与基态同侧
    2. 在 N3 固定后，C5 在误差接近时优先选与基态同侧的解
    """
    iO  = cfg.NMM_ATOM_INDEX["O7"]
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]

    O  = base_coords[iO]
    C2 = base_coords[iC2]
    C4 = base_coords[iC4]
    distances = np.asarray(distances, dtype=np.float64)[:7]
    d_ON, d_OC5, d_NC5, d_NC2, d_NC4, d_C5C2, d_C5C4 = distances

    try:
        N_sol1, N_sol2 = fn.trilaterate_three_spheres(O, C2, C4, d_ON, d_NC2, d_NC4, min_h=1e-6)
        C5_sol1, C5_sol2 = fn.trilaterate_three_spheres(O, C2, C4, d_OC5, d_C5C2, d_C5C4, min_h=1e-6)
    except ValueError:
        return base_coords[iN].copy(), base_coords[iC5].copy(), False

    N_fixed = _choose_n_same_side(
        [np.asarray(N_sol1, dtype=np.float64), np.asarray(N_sol2, dtype=np.float64)],
        base_coords=base_coords,
    )
    if N_fixed is None:
        return base_coords[iN].copy(), base_coords[iC5].copy(), False

    c5_candidates: list[tuple[np.ndarray, float]] = []
    for C5_cand in [np.asarray(C5_sol1, dtype=np.float64), np.asarray(C5_sol2, dtype=np.float64)]:
        err = abs(float(np.linalg.norm(N_fixed - C5_cand)) - d_NC5)
        c5_candidates.append((C5_cand, err))

    eq_C5 = np.asarray(base_coords[iC5], dtype=np.float64)
    eq_sign = np.sign(_c5_plane_signed_distance(eq_C5, base_coords=base_coords))
    if abs(eq_sign) < 1e-12:
        eq_sign = 1.0

    best_err = min(err for _c5, err in c5_candidates)
    near_best = [(c5, err) for c5, err in c5_candidates if err <= best_err + c5_tie_tol]
    same_side = [
        (c5, err) for c5, err in near_best
        if np.sign(_c5_plane_signed_distance(c5, base_coords=base_coords)) == eq_sign
    ]
    pool = same_side if same_side else near_best
    C5_fixed, _ = min(pool, key=lambda item: item[1])
    return N_fixed.copy(), C5_fixed.copy(), True


def reconstruct_trajectory(
    y_mean: np.ndarray,  # (T, >=7)
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
        N3_traj[t], C5_traj[t], success[t] = reconstruct_from_distances_physically_constrained(y_mean[t])

    return N3_traj, C5_traj, success


def plot_structure_snapshots(
    N3_traj: np.ndarray,
    C5_traj: np.ndarray,
    success: np.ndarray,
    td: np.ndarray,
    out_path: str,
    n_frames: int = 12,
    view_elev: float = 25.0,
    view_azim: float = 135.0,
    view_roll: float = 0.0,
):
    """
    绘制多帧分子结构快照（3D），展示完整分子随时间的重建结构。
    """
    base = cfg.NMM_BASE_COORDS_ANG
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    # 环骨架连接顺序: C1-C2-N3-C4-C6-O7-C1 (6元环)
    ring_order = ["C1", "C2", "N3", "C4", "C6", "O7"]
    ref_pos = aligned_reference_positions()

    n_t = len(td)
    valid_frames = np.where(success)[0]
    if len(valid_frames) == 0:
        raise ValueError("no successful reconstruction frames available")
    if len(valid_frames) <= n_frames:
        frame_indices = valid_frames.tolist()
    else:
        grid = np.linspace(0, len(valid_frames) - 1, n_frames)
        frame_indices = valid_frames[np.round(grid).astype(int)].tolist()

    n_frames = len(frame_indices)
    n_cols = min(4, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4.6 * n_cols, 4.6 * n_rows))
    fig.suptitle("NMM Structure Reconstruction from Predicted Distances",
                 fontsize=13, fontweight="bold")

    for panel_idx, t_idx in enumerate(frame_indices):
        ax = fig.add_subplot(n_rows, n_cols, panel_idx + 1, projection='3d')

        frame_backbone = base.copy()
        frame_backbone[iN] = N3_traj[t_idx]
        frame_backbone[iC5] = C5_traj[t_idx]
        coords_all, atom_names, atom_elems = fn.build_full_coords_with_locked_H(frame_backbone)
        atom_map = {name: coords_all[idx] for idx, name in enumerate(atom_names)}

        # H 原子作为结构参考，尽量弱化视觉权重
        for name, elem, pos in zip(atom_names, atom_elems, coords_all):
            if elem != "H":
                continue
            ax.scatter(*pos, s=12, c="#cccccc", alpha=0.45, edgecolors="none", zorder=1)

        # 固定骨架原子
        for name in ["C1", "C2", "C4", "C6", "O7"]:
            pos = atom_map[name]
            color = "red" if name == "O7" else "gray"
            ax.scatter(*pos, s=70, c=color, edgecolors="black", linewidths=0.4, zorder=4)
            ax.text(pos[0], pos[1], pos[2] + 0.13, name, fontsize=6.5, ha="center")

        # 可动原子
        N3 = atom_map["N3"]
        C5 = atom_map["C5"]
        ax.scatter(*N3, s=115, c="blue", edgecolors="black", linewidths=0.5, zorder=5)
        ax.text(N3[0], N3[1], N3[2] + 0.13, "N3", fontsize=7, ha="center", color="blue")
        ax.scatter(*C5, s=115, c="green", edgecolors="black", linewidths=0.5, zorder=5)
        ax.text(C5[0], C5[1], C5[2] + 0.13, "C5", fontsize=7, ha="center", color="green")

        # 画平衡态 N3/C5 位置（半透明）
        ax.scatter(*base[iN], s=60, c="blue", alpha=0.2, marker="x", zorder=3)
        ax.scatter(*base[iC5], s=60, c="green", alpha=0.2, marker="x", zorder=3)

        # 画参考态 N3 / C5 位置：仅作对比，不参与重建
        ax.scatter(*ref_pos["pl_C5"], s=70, c="#f39c12", marker="^",
                   edgecolors="black", linewidths=0.4, alpha=0.85, zorder=4)
        ax.scatter(*ref_pos["ax_C5"], s=70, c="#8e44ad", marker="s",
                   edgecolors="black", linewidths=0.4, alpha=0.85, zorder=4)
        ax.text(ref_pos["pl_C5"][0], ref_pos["pl_C5"][1], ref_pos["pl_C5"][2] + 0.10,
                "pl-C5", fontsize=6, ha="center", color="#b9770e")
        ax.text(ref_pos["ax_C5"][0], ref_pos["ax_C5"][1], ref_pos["ax_C5"][2] + 0.10,
                "ax-C5", fontsize=6, ha="center", color="#6c3483")
        # N3 参考位置最后画，避免被重建点或参考 C5 遮住。
        ax.scatter(*ref_pos["pl_N3"], s=62, c="#5dade2", marker="^",
                   edgecolors="black", linewidths=0.5, alpha=0.95, zorder=6)
        ax.scatter(*ref_pos["ax_N3"], s=62, c="#1f618d", marker="s",
                   edgecolors="black", linewidths=0.5, alpha=0.95, zorder=6)
        ax.text(ref_pos["pl_N3"][0], ref_pos["pl_N3"][1], ref_pos["pl_N3"][2] + 0.10,
                "pl-N3", fontsize=6, ha="center", color="#2e86c1")
        ax.text(ref_pos["ax_N3"][0], ref_pos["ax_N3"][1], ref_pos["ax_N3"][2] + 0.10,
                "ax-N3", fontsize=6, ha="center", color="#1f618d")

        # 画环骨架连接
        for i in range(len(ring_order)):
            a1 = ring_order[i]
            a2 = ring_order[(i + 1) % len(ring_order)]
            p1, p2 = atom_map[a1], atom_map[a2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    "k-", lw=1.5, alpha=0.6)

        # 画 N3-C5 键
        ax.plot([N3[0], C5[0]], [N3[1], C5[1]], [N3[2], C5[2]],
                "g--", lw=1.5, alpha=0.7)

        # 键长标注
        d_NC5 = float(np.linalg.norm(N3 - C5))
        d_ON  = float(np.linalg.norm(atom_map["O7"] - N3))
        mid_nc5 = (N3 + C5) / 2
        mid_on  = (atom_map["O7"] + N3) / 2
        ax.text(mid_nc5[0], mid_nc5[1], mid_nc5[2] + 0.2,
                f"{d_NC5:.2f}", fontsize=6, color="green", ha="center")
        ax.text(mid_on[0] + 0.2, mid_on[1], mid_on[2],
                f"{d_ON:.2f}", fontsize=6, color="red", ha="center")

        # 设置视角和标签
        ax.set_xlabel("X (Å)", fontsize=7)
        ax.set_ylabel("Y (Å)", fontsize=7)
        ax.set_zlabel("Z (Å)", fontsize=7)
        ax.set_title(f"t = {td[t_idx]:.0f} fs", fontsize=10)
        ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)

        # 统一轴范围
        all_pos = coords_all
        center = all_pos.mean(axis=0)
        max_range = max(all_pos.max(axis=0) - all_pos.min(axis=0)) / 2 + 1.0
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.tick_params(labelsize=6)

        if panel_idx == 0:
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
                           markeredgecolor="black", markersize=7, label="reconstructed N3"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
                           markeredgecolor="black", markersize=7, label="reconstructed C5"),
                plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#5dade2",
                           markeredgecolor="black", markersize=7, label="aligned pl-N3"),
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f618d",
                           markeredgecolor="black", markersize=7, label="aligned ax-N3"),
                plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#f39c12",
                           markeredgecolor="black", markersize=7, label="aligned pl-C5"),
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#8e44ad",
                           markeredgecolor="black", markersize=7, label="aligned ax-C5"),
            ]
            ax.legend(handles=handles, fontsize=6.5, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[reconstruct] saved: {out_path}")


def plot_trajectory_projections(
    N3_traj: np.ndarray,
    C5_traj: np.ndarray,
    success: np.ndarray,
    td: np.ndarray,
    out_path: str,
):
    """
    绘制 N3 / C5 在三个平面上的轨迹投影，方便看运动方向和幅度。
    """
    base = cfg.NMM_BASE_COORDS_ANG
    iN = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    mask = success

    proj_pairs = [(0, 1, "X", "Y"), (1, 2, "Y", "Z"), (0, 2, "X", "Z")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Reconstructed Trajectory Projections", fontsize=13, fontweight="bold")

    for ax, (i, j, lx, ly) in zip(axes, proj_pairs):
        ax.plot(N3_traj[mask, i], N3_traj[mask, j], "-o", ms=2.5, lw=1.0,
                color="blue", alpha=0.9, label="N3")
        ax.plot(C5_traj[mask, i], C5_traj[mask, j], "-o", ms=2.5, lw=1.0,
                color="green", alpha=0.9, label="C5")
        ax.scatter(base[iN, i], base[iN, j], c="blue", marker="x", s=70, alpha=0.6)
        ax.scatter(base[iC5, i], base[iC5, j], c="green", marker="x", s=70, alpha=0.6)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.2)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.2)
        ax.set_xlabel(f"{lx} (Å)")
        ax.set_ylabel(f"{ly} (Å)")
        ax.set_title(f"{lx}-{ly} projection")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

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
    N3_mag_std: np.ndarray | None = None,
    C5_mag_std: np.ndarray | None = None,
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
    if N3_mag_std is not None:
        n_lo = np.clip(N3_mag - N3_mag_std, 0.0, None)
        n_hi = N3_mag + N3_mag_std
        ax.fill_between(td[mask], n_lo[mask], n_hi[mask], color="blue", alpha=0.14, label="N3 ±1σ")
    if C5_mag_std is not None:
        c_lo = np.clip(C5_mag - C5_mag_std, 0.0, None)
        c_hi = C5_mag + C5_mag_std
        ax.fill_between(td[mask], c_lo[mask], c_hi[mask], color="green", alpha=0.14, label="C5 ±1σ")
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
    ap.add_argument("--n_frames", type=int, default=12,
                    help="structure snapshot 图中展示的时间帧数量")
    ap.add_argument("--view_elev", type=float, default=25.0,
                    help="3D snapshot 视角仰角")
    ap.add_argument("--view_azim", type=float, default=135.0,
                    help="3D snapshot 视角方位角")
    ap.add_argument("--view_roll", type=float, default=0.0,
                    help="3D snapshot 屏幕内旋转角")
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
                            str(out_dir / "structure_snapshots.png"),
                            n_frames=args.n_frames,
                            view_elev=args.view_elev,
                            view_azim=args.view_azim,
                            view_roll=args.view_roll)

    # 坐标轨迹
    plot_trajectory_2d(N3_traj, C5_traj, success, td,
                       str(out_dir / "coordinate_trajectories.png"))

    # 重建残差图
    plot_reconstruction_residuals(y_mean, N3_traj, C5_traj, success, td,
                                  str(out_dir / "reconstruction_residuals.png"))

    # 平衡态位移图
    plot_displacement_from_equilibrium(N3_traj, C5_traj, success, td,
                                       str(out_dir / "displacement_from_equilibrium.png"))

    # 轨迹投影
    plot_trajectory_projections(N3_traj, C5_traj, success, td,
                                str(out_dir / "trajectory_projections.png"))


if __name__ == "__main__":
    main()
