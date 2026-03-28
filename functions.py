"""
核心函数集合（几何/信号/噪声/归一化/数据IO/网络）

本文件的设计目标：
- 你需要改的“物理参数/数据约束”尽量都在 `config.py`，这里主要提供实现与工具函数
- 每个关键函数的 docstring 都明确“哪些参数常改、怎么改”

重要提醒（对应 manual.md 的要求）：
- 输入信号是一维长度 681 的 `int(s)` / `delta sM(s)`（V2_last 主线默认截断到 431 点）
- H 原子参与干涉项计算，但 H 的坐标不作为自由度：H 坐标 = parent_C 坐标 + 固定偏移（见 `config.H_LOCKED_OFFSETS_ANG`）
- 当前主线标签是 7 个距离（O-N, O-C5, N-C5, N-C2, N-C4, C5-C2, C5-C4）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import json
import math
import numpy as np
import h5py

import torch
import torch.nn as nn

import config as cfg


# ============================================================
# 基础工具：随机数、向量、几何
# ============================================================

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def pairwise_dist_matrix(coords: np.ndarray) -> np.ndarray:
    """
    输入 (N,3)，输出 (N,N) 距离矩阵
    """
    dif = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(dif, axis=-1)


def trilaterate_three_spheres(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    r1: float,
    r2: float,
    r3: float,
    min_h: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    三球交点（经典 trilateration），返回两组解（镜像）。

    你会改的参数：
    - `min_h`: 为了避免几何病态（接近共线导致 h≈0），设置最小高度阈值；对应 config.FILTERS

    失败时抛 ValueError。
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    ex = unit(p2 - p1)
    i = float(np.dot(ex, p3 - p1))
    tmp = p3 - p1 - i * ex
    ey = unit(tmp)
    ez = np.cross(ex, ey)

    d = float(np.linalg.norm(p2 - p1))
    j = float(np.dot(ey, p3 - p1))

    if d < 1e-12 or abs(j) < 1e-12:
        raise ValueError("trilateration: degenerate anchor geometry")

    x = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    y = (r1 * r1 - r3 * r3 + i * i + j * j) / (2.0 * j) - (i / j) * x
    h2 = r1 * r1 - x * x - y * y
    if h2 < 0:
        raise ValueError("trilateration: no real intersection")
    h = math.sqrt(h2)
    if h < min_h:
        raise ValueError("trilateration: unstable (h too small)")

    sol1 = p1 + x * ex + y * ey + h * ez
    sol2 = p1 + x * ex + y * ey - h * ez
    return sol1, sol2


def circle_intersections_in_plane(
    c1: np.ndarray,
    c2: np.ndarray,
    r1: float,
    r2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在同一平面（这里用 xy 平面）求两圆交点。
    c1,c2: (3,) 但只取 x,y；z 由调用者控制。
    返回两组交点（z=0）。
    """
    x0, y0 = float(c1[0]), float(c1[1])
    x1, y1 = float(c2[0]), float(c2[1])
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)
    if d < 1e-12:
        raise ValueError("circle intersection: coincident centers")

    # 无解条件
    if d > (r1 + r2) or d < abs(r1 - r2):
        raise ValueError("circle intersection: no intersection")

    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < 0:
        raise ValueError("circle intersection: no real intersection")
    h = math.sqrt(max(0.0, h2))

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    pA = np.array([xm + rx, ym + ry, 0.0], dtype=np.float64)
    pB = np.array([xm - rx, ym - ry, 0.0], dtype=np.float64)
    return pA, pB


# ============================================================
# 分子构型生成（按 manual.md 的自由度思想）
# ============================================================

@dataclass(frozen=True)
class SampledDOF:
    """
    当前主线采样自由度：
    - n_offset: N3 的小范围各向同性位移
    - c5_offset: C5 的偏置椭球 / 局部各向同性位移
    """
    n_offset: np.ndarray
    c5_offset: np.ndarray


_REFERENCE_STATE_CACHE: Dict[str, np.ndarray] | None = None


def _load_reference_state_xyz(name: str) -> Dict[str, np.ndarray]:
    path = cfg.PATHS.root / "params" / "coords" / name
    lines = path.read_text(encoding="utf-8").strip().splitlines()[2:]
    atoms: List[str] = []
    coords: List[List[float]] = []
    for line in lines:
        sp = line.split()
        if sp[0] == "H":
            continue
        atoms.append(sp[0])
        coords.append([float(sp[1]), float(sp[2]), float(sp[3])])
    heavy = np.array(coords, dtype=np.float64)
    idx_n = atoms.index("N")
    idx_o = atoms.index("O")
    carbons = [heavy[i] for i, a in enumerate(atoms) if a == "C"]
    c5 = max(carbons, key=lambda c: float(c[2]))
    ring = [c for c in carbons if not np.allclose(c, c5)]
    upper = [c for c in ring if float(c[2]) >= 0.0]
    lower = [c for c in ring if float(c[2]) < 0.0]
    if len(upper) != 2 or len(lower) != 2:
        raise ValueError(f"{name}: failed to identify ring carbons in mainline order")
    c2 = max(upper, key=lambda c: float(c[0]))
    c4 = min(upper, key=lambda c: float(c[0]))
    c1 = max(lower, key=lambda c: float(c[0]))
    c6 = min(lower, key=lambda c: float(c[0]))
    return {
        "C1": np.asarray(c1, dtype=np.float64),
        "C2": np.asarray(c2, dtype=np.float64),
        "N3": heavy[idx_n],
        "C4": np.asarray(c4, dtype=np.float64),
        "C5": np.asarray(c5, dtype=np.float64),
        "C6": np.asarray(c6, dtype=np.float64),
        "O7": heavy[idx_o],
    }


def _kabsch_align(moving: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moving = np.asarray(moving, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    moving_centroid = moving.mean(axis=0)
    target_centroid = target.mean(axis=0)
    moving_c = moving - moving_centroid
    target_c = target - target_centroid
    h = moving_c.T @ target_c
    u, _s, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = u @ vt
    t = target_centroid - moving_centroid @ r
    aligned = moving @ r + t
    return aligned, r, t


def _reference_sampling_basis() -> Dict[str, np.ndarray]:
    global _REFERENCE_STATE_CACHE
    if _REFERENCE_STATE_CACHE is None:
        # NMM_pl / NMM_ax 仅用于提取 C5 的参考方向。
        # 它们先对齐到基态固定骨架，再只把方向信息用于采样偏置；
        # 后续分子重建仍然以基态参考结构为准，不直接依赖这些位移数值。
        anchor_labels = ["C1", "C2", "C4", "C6", "O7"]
        eq = {
            label: np.asarray(cfg.NMM_BASE_COORDS_ANG[cfg.NMM_ATOM_INDEX[label]], dtype=np.float64)
            for label in ["C1", "C2", "N3", "C4", "C5", "C6", "O7"]
        }
        pl = _load_reference_state_xyz("NMM_pl.xyz")
        ax = _load_reference_state_xyz("NMM_ax.xyz")
        target_anchor = np.stack([eq[l] for l in anchor_labels], axis=0)
        for state in (pl, ax):
            moving_anchor = np.stack([state[l] for l in anchor_labels], axis=0)
            _aligned, r, t = _kabsch_align(moving_anchor, target_anchor)
            for key in state:
                state[key] = state[key] @ r + t

        c5_pl = pl["C5"] - eq["C5"]
        c5_ax = ax["C5"] - eq["C5"]
        e1 = unit(c5_pl + c5_ax)
        diff = c5_ax - c5_pl
        e2 = diff - np.dot(diff, e1) * e1
        if float(np.linalg.norm(e2)) < 1e-8:
            fallback = eq["C4"] - eq["C2"]
            e2 = fallback - np.dot(fallback, e1) * e1
        e2 = unit(e2)
        e3 = unit(np.cross(e1, e2))
        _REFERENCE_STATE_CACHE = {
            "c5_pl": c5_pl,
            "c5_ax": c5_ax,
            "c5_e1": e1,
            "c5_e2": e2,
            "c5_e3": e3,
        }
    return _REFERENCE_STATE_CACHE


def _sample_truncated_isotropic(
    rng: np.random.Generator,
    sigma: float,
    max_radius: float,
    max_tries: int = 256,
) -> np.ndarray:
    for _ in range(max_tries):
        v = rng.normal(0.0, sigma, size=3).astype(np.float64)
        if float(np.linalg.norm(v)) <= max_radius:
            return v
    v = rng.normal(0.0, sigma, size=3).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n > max_radius and n > 1e-12:
        v *= max_radius / n
    return v


def sample_dof(rng: np.random.Generator) -> SampledDOF:
    """
    当前主线采样：
    1. N3: 小范围各向同性采样
    2. C5: 宽松偏置椭球 + 少量局部各向同性采样
    """
    n_offset = _sample_truncated_isotropic(
        rng, cfg.GEN_N_ISO_SIGMA_ANG, cfg.GEN_N_ISO_MAX_RADIUS_ANG
    )

    if float(rng.random()) < cfg.GEN_C5_LOCAL_ISO_PROB:
        c5_offset = _sample_truncated_isotropic(
            rng, cfg.GEN_C5_LOCAL_ISO_SIGMA_ANG, cfg.GEN_C5_LOCAL_MAX_RADIUS_ANG
        )
    else:
        basis = _reference_sampling_basis()
        a1 = float(np.clip(
            rng.normal(cfg.GEN_C5_MAIN_MEAN_ANG, cfg.GEN_C5_MAIN_SIGMA_ANG),
            cfg.GEN_C5_MAIN_MIN_ANG,
            cfg.GEN_C5_MAIN_MAX_ANG,
        ))
        a2 = float(rng.normal(0.0, cfg.GEN_C5_PERP_SIGMA_ANG))
        a3 = float(rng.normal(0.0, cfg.GEN_C5_NORMAL_SIGMA_ANG))
        c5_offset = (
            a1 * basis["c5_e1"]
            + a2 * basis["c5_e2"]
            + a3 * basis["c5_e3"]
        ).astype(np.float64)
        n = float(np.linalg.norm(c5_offset))
        if n > cfg.GEN_C5_TOTAL_MAX_RADIUS_ANG and n > 1e-12:
            c5_offset *= cfg.GEN_C5_TOTAL_MAX_RADIUS_ANG / n

    return SampledDOF(n_offset=n_offset, c5_offset=c5_offset)


def generate_backbone_near_equil(rng: np.random.Generator,
                                  max_disp: float = 0.3) -> np.ndarray:
    """
    直接生成近平衡态骨架（Franck-Condon 区域）。

    在基态坐标基础上，对 N3 和 C5 各加小幅 Gaussian 扰动
    （sigma=max_disp/3 Å），用于补充靠近平衡态的训练样本。
    这里不再使用旧的圆柱/球采样参数，也不强加方向先验。

    返回：7x3 骨架坐标（Å）
    """
    base = np.asarray(cfg.NMM_BASE_COORDS_ANG, dtype=np.float64).copy()
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]

    sigma = max_disp / 3.0  # 3-sigma ≈ max_disp Å

    # Small Gaussian displacement on N and C5
    base[iN]  += rng.normal(0.0, sigma, 3)
    base[iC5] += rng.normal(0.0, sigma, 3)

    return base


def build_full_coords_with_locked_H(
    backbone_coords_7x3: np.ndarray,
    h_locked_offsets: Dict[str, List[np.ndarray]] = cfg.H_LOCKED_OFFSETS_ANG,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    把 7 个骨架原子坐标扩展为 “骨架 + H” 的完整坐标。

    返回：
    - coords_all: (N,3)
    - atom_names_all: 长度 N 的名字（如 C1,H_C1_0,...）
    - atom_elements_all: 长度 N 的元素符号（C/N/O/H）

    你常改的点：
    - H 的数量和偏移：config.H_LOCKED_OFFSETS_ANG
    """
    backbone_coords_7x3 = np.asarray(backbone_coords_7x3, dtype=np.float64)
    if backbone_coords_7x3.shape != (7, 3):
        raise ValueError("backbone_coords must be (7,3) in NMM_ATOM_ORDER")

    names: List[str] = []
    elems: List[str] = []
    coords: List[np.ndarray] = []

    # 7 骨架
    for name, xyz in zip(cfg.NMM_ATOM_ORDER, backbone_coords_7x3):
        names.append(name)
        if name.startswith("C"):
            elems.append("C")
        elif name.startswith("N"):
            elems.append("N")
        elif name.startswith("O"):
            elems.append("O")
        else:
            raise ValueError(f"unknown backbone atom name: {name}")
        coords.append(xyz.astype(np.float64))

    # H 展开
    for parent_name, offsets in h_locked_offsets.items():
        if parent_name not in cfg.NMM_ATOM_INDEX:
            raise ValueError(f"H parent {parent_name} not in NMM_ATOM_ORDER")
        parent_idx = cfg.NMM_ATOM_INDEX[parent_name]
        parent_xyz = backbone_coords_7x3[parent_idx]
        for k, off in enumerate(offsets):
            names.append(f"H_{parent_name}_{k}")
            elems.append("H")
            coords.append(parent_xyz + np.asarray(off, dtype=np.float64))

    coords_all = np.stack(coords, axis=0)
    return coords_all, names, elems


def generate_backbone_coords_from_dof(
    dof: SampledDOF,
    base_coords: np.ndarray = cfg.NMM_BASE_COORDS_ANG,

    min_h: float = 0.0,
) -> np.ndarray:
    """
    根据当前主线采样自由度生成 7x3 骨架坐标：
    - 固定：O7、C1/C2/C4/C6
    - 生成：N3（小范围各向同性）、C5（偏置椭球/局部各向同性）
    """
    base_coords = np.asarray(base_coords, dtype=np.float64)
    coords = base_coords.copy()

    iO = cfg.NMM_ATOM_INDEX["O7"]
    iN = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]

    O = coords[iO]
    N_base = coords[iN]
    C5_base = coords[iC5]

    N = N_base + np.asarray(dof.n_offset, dtype=np.float64)
    C5 = C5_base + np.asarray(dof.c5_offset, dtype=np.float64)

    coords[iN] = N
    coords[iC5] = C5
    
    # ========== 采样后的键长约束检查 ==========
    # 约束：N-O 键长 < C5-O 键长（外面那个 C 原子）
    if cfg.GEN_CHECK_N_O_SHORTER_THAN_C5_O:
        r_no = float(np.linalg.norm(N - O))
        r_c5o = float(np.linalg.norm(C5 - O))
        if r_no >= r_c5o:
            raise ValueError(
                f"constraint violated: r(N-O)={r_no:.3f} >= r(C5-O)={r_c5o:.3f}; "
                f"requires r(N-O) < r(C5-O)"
            )
    
    return coords


# ============================================================
# 散射因子与一维信号计算
# ============================================================

# ============================================================
# DPWA 散射因子加载（一次性缓存）
# params/DPWA/fC.mat, fH.mat, fN.mat, fO.mat
# 680 点复数，对应 s = S_GRID[1:]（0.022~15.0 Å⁻¹），取实部用于 sM 计算
# ============================================================

_DPWA_CACHE: Dict[str, np.ndarray] | None = None

def _load_dpwa_factors() -> Dict[str, np.ndarray]:
    """加载 DPWA 散射因子并插值到 cfg.S_GRID（681 点）。首次调用时执行，结果缓存。"""
    global _DPWA_CACHE
    if _DPWA_CACHE is not None:
        return _DPWA_CACHE

    import scipy.io as sio
    from scipy.interpolate import interp1d

    dpwa_dir = Path(__file__).resolve().parent / "params" / "DPWA"
    name_map = {"C": "fC", "H": "fH", "N": "fN", "O": "fO"}
    # DPWA 680 点对应 s = S_GRID[1:] (跳过 s=0)
    s_dpwa = cfg.S_GRID[1:]  # (680,)
    s_full = cfg.S_GRID       # (681,)

    # 先加载所有元素的原始数据，找全局归一化参考（Carbon s=0 处的值）
    raw: Dict[str, np.ndarray] = {}
    for elem, fname in name_map.items():
        fpath = dpwa_dir / f"{fname}.mat"
        if not fpath.exists():
            raise FileNotFoundError(f"DPWA file not found: {fpath}")
        data = sio.loadmat(str(fpath))[fname].flatten()  # (680,) complex
        raw[elem] = np.abs(data).astype(np.float64)

    # 归一化：除以 Carbon 在 s→0 处的值，使 C(s=0) ≈ Z_C = 6
    # 这样所有元素保持相对比例，同时量纲与 Cromer-Mann 兼容
    f_C0 = raw["C"][0]      # Carbon 在最低 s 点的散射振幅
    Z_C = 6.0               # Carbon 的电子数（作为归一化目标）
    scale = Z_C / f_C0      # 将所有因子缩放到 electron 单位

    cache: Dict[str, np.ndarray] = {}
    for elem, f_raw in raw.items():
        f_scaled = f_raw * scale
        interp = interp1d(s_dpwa, f_scaled, kind="linear",
                          bounds_error=False, fill_value=(f_scaled[0], f_scaled[-1]))
        f_grid = np.empty(len(s_full), dtype=np.float64)
        f_grid[0] = f_scaled[0]
        f_grid[1:] = interp(s_full[1:])
        cache[elem] = f_grid

    _DPWA_CACHE = cache
    return cache


def atomic_scattering_factor(element: str, s: np.ndarray) -> np.ndarray:
    """
    返回元素的 DPWA 电子散射因子 f(s)。

    V2_last 主线固定要求：
    - 内部散射信号在 cfg.S_GRID (681 点) 上计算
    - 之后再统一截断到 cfg.S_GRID_TRUNC (431 点)
    """
    s = np.asarray(s, dtype=np.float64)
    if len(s) != len(cfg.S_GRID) or not np.allclose(s, cfg.S_GRID):
        raise ValueError(
            "V2_last requires scattering signals to be computed on cfg.S_GRID "
            f"(got len={len(s)})"
        )
    try:
        return _load_dpwa_factors()[element]
    except KeyError as exc:
        raise KeyError(f"unsupported element for DPWA scattering factor: {element}") from exc


def compute_1d_scattering_signal(
    coords_all: np.ndarray,
    elements_all: List[str],
    s: np.ndarray = cfg.S_GRID,
    s_cut_min: float = cfg.S_CUTOFF_MIN,
    s_cut_max: float = cfg.S_CUTOFF_MAX,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    从“骨架+H”的坐标计算一维信号（长度 len(s)）。

    模型（简化但可用）：
    - IA(s) = sum_i f_i(s)^2
    - IM(s) = sum_{i<j} 2 f_i(s) f_j(s) sin(s*r_ij)/(s*r_ij)
    - signal = s * IM / IA

    你会改的点：
    - 如果你有更严谨/更贴近实验的定义（例如 delta sM、背景扣除、归一化方式），
      请在这里集中修改。
    """
    s = np.asarray(s, dtype=np.float64)
    coords_all = np.asarray(coords_all, dtype=np.float64)

    n = coords_all.shape[0]
    if coords_all.shape != (n, 3):
        raise ValueError("coords_all must be (N,3)")
    if len(elements_all) != n:
        raise ValueError("elements_all length mismatch")

    # 预计算每个元素的 f(s)
    uniq = sorted(set(elements_all))
    f_cache: Dict[str, np.ndarray] = {e: atomic_scattering_factor(e, s) for e in uniq}
    f = np.stack([f_cache[e] for e in elements_all], axis=0)  # (N, S)

    IA = np.sum(f * f, axis=0)  # (S,)

    # pairwise distances
    dif = coords_all[:, None, :] - coords_all[None, :, :]
    rij = np.linalg.norm(dif, axis=-1) + np.eye(n)  # avoid zero on diagonal

    IM = np.zeros_like(s, dtype=np.float64)
    for i in range(n):
        fi = f[i]
        for j in range(i + 1, n):
            fj = f[j]
            r = float(rij[i, j])
            IM += 2.0 * fi * fj * (np.sin(s * r) / (s * r + eps))

    signal = s * IM / (IA + eps)

    # 有效区间裁剪
    signal = signal.copy()
    signal[s < s_cut_min] = 0.0
    signal[s > s_cut_max] = 0.0
    return signal.astype(np.float32)


# ============================================================
# 噪声模型
# ============================================================

def smooth_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window % 2 == 0:
        raise ValueError("window must be odd")
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    k = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(xp, k, mode="valid")


def add_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    s: np.ndarray = cfg.S_GRID,
    noise_profile: np.ndarray | None = None,
    noise_profile_scale_range: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    叠加噪声（可在 config.py 开关与调量级）
    """
    if not cfg.NOISE_ENABLE:
        return signal

    y = signal.astype(np.float64).copy()

    # 1) 高斯噪声（主线 joint-train 可使用实验经验包络）
    if noise_profile is not None:
        profile = np.asarray(noise_profile, dtype=np.float64)
        if profile.shape != y.shape:
            raise ValueError(
                f"noise_profile shape mismatch: got {profile.shape}, expected {y.shape}"
            )
        amp = float(rng.uniform(*noise_profile_scale_range))
        y += rng.normal(0.0, 1.0, size=y.shape) * profile * amp
    else:
        std = float(rng.uniform(*cfg.NOISE_GAUSS_STD_RANGE))
        if cfg.NOISE_GAUSS_SCALE_WITH_S:
            scale = 1.0 / (s + 0.5)  # 避免 s=0 发散
            y += rng.normal(0.0, std, size=y.shape) * scale
        else:
            y += rng.normal(0.0, std, size=y.shape)

    # 2) 低频漂移
    if cfg.NOISE_DRIFT_ENABLE:
        dstd = float(rng.uniform(*cfg.NOISE_DRIFT_STD_RANGE))
        drift = rng.normal(0.0, dstd, size=y.shape)
        drift = smooth_moving_average(drift, cfg.NOISE_DRIFT_SMOOTH_WINDOW)
        y += drift

    # 3) 少量尖峰
    if cfg.NOISE_SPIKE_ENABLE:
        mask = rng.random(size=y.shape) < float(cfg.NOISE_SPIKE_PROB)
        spikes = rng.normal(0.0, float(cfg.NOISE_SPIKE_SCALE), size=y.shape)
        y += mask * spikes

    return y.astype(np.float32)


# ============================================================
# 标签与评估基准
# ============================================================

def label_names_for_dim(dim: int) -> List[str]:
    """
    返回当前主线的标签名。
    """
    if dim != cfg.LABEL_FLAT_DIM:
        raise ValueError(
            f"V2_last only supports {cfg.LABEL_FLAT_DIM} labels, got {dim}"
        )
    return list(cfg.LABEL_PAIR_NAMES)


def legacy_matrix3_label_from_backbone(backbone_7x3: np.ndarray) -> np.ndarray:
    """
    Legacy 9 维标签：(O, N, C5) 的 3x3 距离矩阵 flatten。
    仅用于兼容旧 checkpoint / 旧报告，不作为 V2_last 主线。
    """
    coords = np.asarray(backbone_7x3, dtype=np.float64)
    idx = [cfg.NMM_ATOM_INDEX["O7"], cfg.NMM_ATOM_INDEX["N3"], cfg.NMM_ATOM_INDEX["C5"]]
    mat = pairwise_dist_matrix(coords[idx])
    return mat.astype(np.float32).reshape(-1)


def label_from_backbone(backbone_7x3: np.ndarray) -> np.ndarray:
    """
    返回当前主线的 8 维标签：
    [d(O-N), d(O-C5), d(N-C5), d(N-C2), d(N-C4), d(C5-C2), d(C5-C4), h_C5_signed]
    其中 h_C5_signed 为 C5 到刚性环平面（C1/C2/C4/C6 best-fit plane）的有符号距离。
    """
    iO  = cfg.NMM_ATOM_INDEX["O7"]
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    coords = backbone_7x3
    d_ON   = float(np.linalg.norm(coords[iO]  - coords[iN]))
    d_OC5  = float(np.linalg.norm(coords[iO]  - coords[iC5]))
    d_NC5  = float(np.linalg.norm(coords[iN]  - coords[iC5]))
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]
    d_NC2  = float(np.linalg.norm(coords[iN]  - coords[iC2]))
    d_NC4  = float(np.linalg.norm(coords[iN]  - coords[iC4]))
    d_C5C2 = float(np.linalg.norm(coords[iC5] - coords[iC2]))
    d_C5C4 = float(np.linalg.norm(coords[iC5] - coords[iC4]))
    iC1 = cfg.NMM_ATOM_INDEX["C1"]
    iC6 = cfg.NMM_ATOM_INDEX["C6"]
    ring = np.stack([coords[iC1], coords[iC2], coords[iC4], coords[iC6]], axis=0).astype(np.float64)
    centroid = ring.mean(axis=0)
    _u, _s, vh = np.linalg.svd(ring - centroid, full_matrices=False)
    normal = vh[-1]
    n = float(np.linalg.norm(normal))
    if n < 1e-12:
        h_c5_signed = 0.0
    else:
        normal = normal / n
        eq_coords = np.asarray(cfg.NMM_BASE_COORDS_ANG, dtype=np.float64)
        eq_ring = np.stack([eq_coords[iC1], eq_coords[iC2], eq_coords[iC4], eq_coords[iC6]], axis=0)
        eq_centroid = eq_ring.mean(axis=0)
        _u_eq, _s_eq, vh_eq = np.linalg.svd(eq_ring - eq_centroid, full_matrices=False)
        eq_normal = vh_eq[-1]
        eq_sign = float(np.dot(eq_coords[iC5] - eq_centroid, eq_normal))
        if eq_sign < 0:
            normal = -normal
        h_c5_signed = float(np.dot(coords[iC5] - centroid, normal))
    return np.array([d_ON, d_OC5, d_NC5, d_NC2, d_NC4, d_C5C2, d_C5C4, h_c5_signed], dtype=np.float32)


def equilibrium_labels_for_dim(dim: int) -> np.ndarray:
    """
    从当前 config 中的基态坐标显式计算“平衡距离”。
    所有评估/绘图都应调用这里，避免把坐标分量误当作距离。
    """
    if dim != cfg.LABEL_FLAT_DIM:
        raise ValueError(
            f"V2_last only supports {cfg.LABEL_FLAT_DIM} labels, got {dim}"
        )
    return label_from_backbone(cfg.NMM_BASE_COORDS_ANG)


# ============================================================
# 归一化参数保存/加载
# ============================================================

@dataclass
class Normalization:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def to_json(self) -> Dict:
        return dict(
            x_mean=self.x_mean.tolist(),
            x_std=self.x_std.tolist(),
            y_mean=self.y_mean.tolist(),
            y_std=self.y_std.tolist(),
        )

    @staticmethod
    def from_json(d: Dict) -> "Normalization":
        return Normalization(
            x_mean=np.array(d["x_mean"], dtype=np.float32),
            x_std=np.array(d["x_std"], dtype=np.float32),
            y_mean=np.array(d["y_mean"], dtype=np.float32),
            y_std=np.array(d["y_std"], dtype=np.float32),
        )


def compute_normalization(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> Normalization:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0) + eps
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + eps
    return Normalization(x_mean.astype(np.float32), x_std.astype(np.float32), y_mean.astype(np.float32), y_std.astype(np.float32))


def save_normalization(norm: Normalization, path: Path = cfg.PATHS.norm_json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(norm.to_json(), f, ensure_ascii=False, indent=2)


def load_normalization(path: Path = cfg.PATHS.norm_json) -> Normalization:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Normalization.from_json(d)


def normalize_x(x: np.ndarray, norm: Normalization) -> np.ndarray:
    return (x - norm.x_mean) / norm.x_std


def normalize_y(y: np.ndarray, norm: Normalization) -> np.ndarray:
    return (y - norm.y_mean) / norm.y_std


def denormalize_y(y_hat: np.ndarray, norm: Normalization) -> np.ndarray:
    return y_hat * norm.y_std + norm.y_mean


# ============================================================
# H5 数据集写入/读取（按 manual.md：保存 .h5 到 file_training_data）
# ============================================================

def write_h5_dataset(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("x", data=x, compression="gzip", compression_opts=4, shuffle=True)
        h5f.create_dataset("y", data=y, compression="gzip", compression_opts=4, shuffle=True)


def read_h5_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5f:
        return np.array(h5f["x"]), np.array(h5f["y"])


class H5Dataset(torch.utils.data.Dataset):
    """逐条从 h5 读取（适合超大数据集）"""
    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as h5f:
            self.n = int(h5f["x"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as h5f:
            x = h5f["x"][idx]
            y = h5f["y"][idx]
        # x: (S,) -> (1,S) for Conv1d
        return torch.from_numpy(x[None, :]).float(), torch.from_numpy(y).float()


class InMemoryDataset(torch.utils.data.Dataset):
    """一次性加载到内存（16GB 机器上 500k*681*4B ≈ 1.3GB 可以放下）"""
    def __init__(self, h5_path: Path):
        print(f"[data] Loading {h5_path} into memory...", flush=True)
        with h5py.File(h5_path, "r") as h5f:
            self.x = torch.from_numpy(np.array(h5f["x"])).float()  # (N, S)
            self.y = torch.from_numpy(np.array(h5f["y"])).float()  # (N, 7)
        print(f"[data] Loaded {self.x.shape[0]} samples, x={self.x.shape}, y={self.y.shape}", flush=True)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx].unsqueeze(0), self.y[idx]  # (1, S), (9,)


# ============================================================
# 1D 回归网络（具备“倒空间反演能力”的实践型结构：Conv1d + 残差 + 多尺度池化）
# ============================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, ch: int, k: int = 7):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad),
            nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=pad),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class NMMRegressor1D(nn.Module):
    """
    旧版模型（保留用于加载已有 checkpoint）。
    输入 (B,1,S), 输出 (B,7)
    """

    def __init__(self, base_ch: int = 64, n_blocks: int = 6, out_dim: int = cfg.LABEL_FLAT_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock1D(base_ch, k=7) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


# ============================================================
# 改进版模型 V2：多尺度 + SE注意力 + 更大容量
# ============================================================

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.ReLU(),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)  # (B, ch, 1)
        return x * w


class ResBlockV2(nn.Module):
    """带 BN + SE 的残差块"""
    def __init__(self, ch: int, k: int = 7):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=pad),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=pad),
        )
        self.se = SEBlock1D(ch)

    def forward(self, x):
        return x + self.se(self.net(x))


class MultiScaleStem(nn.Module):
    """多尺度卷积 stem: 不同 kernel size 捕获不同频率特征"""
    def __init__(self, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 4
        self.b3 = nn.Conv1d(1, branch_ch, kernel_size=3, padding=1)
        self.b7 = nn.Conv1d(1, branch_ch, kernel_size=7, padding=3)
        self.b15 = nn.Conv1d(1, branch_ch, kernel_size=15, padding=7)
        self.b31 = nn.Conv1d(1, branch_ch, kernel_size=31, padding=15)
        self.merge = nn.Sequential(
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
        )

    def forward(self, x):
        out = torch.cat([self.b3(x), self.b7(x), self.b15(x), self.b31(x)], dim=1)
        return self.merge(out)


class NMMRegressorV2(nn.Module):
    """
    改进版模型：多尺度stem + SE残差块 + 渐进下采样 + 更大MLP head
    输入 (B,1,S), 输出 (B,7)
    """

    def __init__(self, base_ch: int = 128, n_blocks: int = 8, out_dim: int = cfg.LABEL_FLAT_DIM):
        super().__init__()
        self.stem = MultiScaleStem(base_ch)

        # 残差块组 + 渐进下采样
        layers = []
        for i in range(n_blocks):
            layers.append(ResBlockV2(base_ch, k=7))
            if i in (2, 5):  # 在第3和第6个块后下采样
                layers.append(nn.MaxPool1d(2))
        self.body = nn.Sequential(*layers)

        # 全局池化 + MLP head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)


# ============================================================
# V3 模型：更深卷积 + 增强注意力 + 适配截断输入（~431 点）
# ============================================================

class CBAM1D(nn.Module):
    """Convolutional Block Attention Module (1D): 通道注意力 + 空间注意力"""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        # 通道注意力 (SE-like, but with both avg+max pooling)
        self.ch_fc = nn.Sequential(
            nn.Linear(ch * 2, ch // reduction),
            nn.ReLU(),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )
        # 空间注意力
        self.sp_conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention
        avg_pool = x.mean(dim=-1)           # (B, C)
        max_pool = x.max(dim=-1).values     # (B, C)
        ch_attn = self.ch_fc(torch.cat([avg_pool, max_pool], dim=1))  # (B, C)
        x = x * ch_attn.unsqueeze(-1)
        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)  # (B, 1, L)
        max_s = x.max(dim=1, keepdim=True).values  # (B, 1, L)
        sp_attn = self.sp_conv(torch.cat([avg_s, max_s], dim=1))  # (B, 1, L)
        return x * sp_attn


class ResBlockV3(nn.Module):
    """V3 残差块: BN-GELU-Conv-BN-GELU-Conv + CBAM + 可选 bottleneck"""
    def __init__(self, ch: int, k: int = 5, bottleneck_ratio: float = 1.0):
        super().__init__()
        mid_ch = max(ch // 4, int(ch * bottleneck_ratio)) if bottleneck_ratio < 1 else ch
        pad = k // 2
        self.net = nn.Sequential(
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Conv1d(ch, mid_ch, 1) if mid_ch != ch else nn.Identity(),
            nn.BatchNorm1d(mid_ch) if mid_ch != ch else nn.Identity(),
            nn.GELU() if mid_ch != ch else nn.Identity(),
            nn.Conv1d(mid_ch, mid_ch, k, padding=pad, groups=1),
            nn.BatchNorm1d(mid_ch),
            nn.GELU(),
            nn.Conv1d(mid_ch, ch, 1) if mid_ch != ch else nn.Conv1d(ch, ch, k, padding=pad),
        )
        self.attn = CBAM1D(ch)

    def forward(self, x):
        return x + self.attn(self.net(x))


class MultiScaleStemV3(nn.Module):
    """V3 多尺度 stem: 5 个分支 + 更多特征"""
    def __init__(self, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 5
        extra = out_ch - branch_ch * 5
        self.b3 = nn.Conv1d(1, branch_ch, kernel_size=3, padding=1)
        self.b5 = nn.Conv1d(1, branch_ch, kernel_size=5, padding=2)
        self.b11 = nn.Conv1d(1, branch_ch, kernel_size=11, padding=5)
        self.b21 = nn.Conv1d(1, branch_ch, kernel_size=21, padding=10)
        self.b41 = nn.Conv1d(1, branch_ch + extra, kernel_size=41, padding=20)
        self.merge = nn.Sequential(
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
        )

    def forward(self, x):
        out = torch.cat([self.b3(x), self.b5(x), self.b11(x),
                         self.b21(x), self.b41(x)], dim=1)
        return self.merge(out)


class NMMRegressorV3(nn.Module):
    """
    V3 模型：适配截断输入 (~431 点)
    - 5 分支多尺度 stem (更多频率覆盖)
    - 10 个 CBAM 残差块 (更深)
    - 3 次下采样 (8x)
    - 更大 MLP head
    输入 (B, 1, S), 输出 (B, out_dim)
    """

    def __init__(self, base_ch: int = 160, n_blocks: int = 10,
                 out_dim: int = cfg.LABEL_FLAT_DIM):
        super().__init__()
        self.stem = MultiScaleStemV3(base_ch)

        # 残差块组 + 渐进下采样
        layers = []
        for i in range(n_blocks):
            layers.append(ResBlockV3(base_ch, k=5))
            if i in (2, 5, 8):  # 3 次下采样
                layers.append(nn.MaxPool1d(2))
        self.body = nn.Sequential(*layers)

        # 全局池化 + 更大 MLP head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)
