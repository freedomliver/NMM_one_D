"""
核心函数集合（几何/信号/噪声/归一化/数据IO/网络）

本文件的设计目标：
- 你需要改的“物理参数/数据约束”尽量都在 `config.py`，这里主要提供实现与工具函数
- 每个关键函数的 docstring 都明确“哪些参数常改、怎么改”

重要提醒（对应 manual.md 的要求）：
- 输入信号是一维长度 681 的 `int(s)` / `delta sM(s)`（本工程使用 `config.S_GRID`）
- H 原子参与干涉项计算，但 H 的坐标不作为自由度：H 坐标 = parent_C 坐标 + 固定偏移（见 `config.H_LOCKED_OFFSETS_ANG`）
- 标签是 (O7, N3, C5) 的 3x3 全距离矩阵（flatten 为 9 维）
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
    新采样策略的参数（2026/3/12）：
    - N_center, N_orth_dir：N3 在圆柱中的采样位置参数
    - C5_offset：C5 相对其基态坐标的偏移向量
    """
    n_center: np.ndarray  # (3,) - N 采样圆柱轴上的点（实际 N 将在此附近圆柱内）
    n_radius_t: float     # [0, 1] - N 采样的圆柱半径归一化参数
    n_angle_t: float      # [0, 2π) - N 采样的方位角
    c5_offset: np.ndarray # (3,) - C5 相对基态的偏移（球体采样）


def sample_dof(rng: np.random.Generator) -> SampledDOF:
    """
    新采样策略：
    1. 对 N3 进行圆柱采样：在 NO 中轴面上，离中轴面 GEN_N_CYLINDER_OFF_PLANE_ANG，
       以原坐标为圆心、半径 GEN_N_CYLINDER_RADIUS_ANG 的圆柱内均匀随机采样。
    2. 对 C5 进行球体采样：以原坐标为圆心、半径 GEN_C5_SPHERE_RADIUS_ANG 的球体内均匀随机采样。
    
    实际几何约束（包含 NO 中轴面定义等）延迟到 generate_backbone_coords_from_dof 中实现。
    """
    # 取基态 N3 坐标作为圆柱采样的中心
    n_base = cfg.NMM_BASE_COORDS_ANG[cfg.NMM_ATOM_INDEX["N3"]]
    
    # N3 圆柱采样参数
    # - 圆柱半径：从 0 到 GEN_N_CYLINDER_RADIUS_ANG 的平方根均匀分布（保证圆盘面积均匀）
    r_sq = float(rng.uniform(0.0, cfg.GEN_N_CYLINDER_RADIUS_ANG ** 2))
    n_radius_t = float(np.sqrt(r_sq / (cfg.GEN_N_CYLINDER_RADIUS_ANG ** 2)))
    
    # - 方位角：[0, 2π)
    n_angle_t = float(rng.uniform(0.0, 2.0 * np.pi))
    
    # C5 球体采样参数
    # 在单位球内均匀采样，然后缩放到 GEN_C5_SPHERE_RADIUS_ANG
    # 使用标准的球面坐标均匀采样
    c5_r_sq = float(rng.uniform(0.0, cfg.GEN_C5_SPHERE_RADIUS_ANG ** 3))
    c5_r = float(np.cbrt(c5_r_sq))
    c5_theta = float(np.arccos(rng.uniform(-1.0, 1.0)))
    c5_phi = float(rng.uniform(0.0, 2.0 * np.pi))
    
    c5_offset = c5_r * np.array([
        np.sin(c5_theta) * np.cos(c5_phi),
        np.sin(c5_theta) * np.sin(c5_phi),
        np.cos(c5_theta)
    ], dtype=np.float64)
    
    return SampledDOF(
        n_center=n_base.copy(),
        n_radius_t=n_radius_t,
        n_angle_t=n_angle_t,
        c5_offset=c5_offset,
    )


def generate_backbone_near_equil(rng: np.random.Generator,
                                  max_disp: float = 0.3) -> np.ndarray:
    """
    直接生成近平衡态骨架（Franck-Condon 区域）。

    在基态坐标基础上，对 N3 和 C5 各加 Gaussian 扰动（sigma=max_disp/3 Å），
    N3 的面外偏移从 [0, GEN_N_CYLINDER_OFF_PLANE_ANG] 均匀采样，
    允许生成真正的近基态结构（off_plane ≈ 0 → N-C5 ≈ 1.45 Å）。

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
    根据新采样策略生成 7x3 骨架坐标：
    - 固定：O7、C1/C2/C4/C6
    - 生成：N3（圆柱约束）、C5（球体约束）
    
    采样策略（2026/3/12）：
    1. N3：在 NO 中轴面上离中轴面 GEN_N_CYLINDER_OFF_PLANE_ANG，
       以原坐标为圆心、半径 GEN_N_CYLINDER_RADIUS_ANG 的圆柱内均匀随机采样。
    2. C5：以原坐标为圆心、半径 GEN_C5_SPHERE_RADIUS_ANG 的球体内均匀随机采样。
    
    采样后限制：N-O 键长 < C2-O 键长，不满足则抛异常（由外层逻辑重采样）。
    """
    base_coords = np.asarray(base_coords, dtype=np.float64)
    coords = base_coords.copy()

    iO = cfg.NMM_ATOM_INDEX["O7"]
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iN = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    iC1 = cfg.NMM_ATOM_INDEX["C1"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]

    O = coords[iO]
    C2 = coords[iC2]
    C1 = coords[iC1]
    C4 = coords[iC4]
    N_base = coords[iN]
    C5_base = coords[iC5]

    # ========== N3 圆柱采样 ==========
    # 定义 NO 中轴面：
    # - 以 O 为原点
    # - 中轴线指向 N（基态）
    # - 平面由 N 和 C1/C4（平面原子）定义
    
    # 中轴线方向（从 O 指向基态 N）
    no_axis = unit(N_base - O)
    
    # 平面法向（用 C1-O 和 C4-O 叉积定义；这是固定平面，不依赖采样的 N）
    v1 = unit(C1 - O)
    v2 = unit(C4 - O)
    plane_normal = unit(np.cross(v1, v2))
    
    # 圆柱采样参数
    r_cyl = cfg.GEN_N_CYLINDER_RADIUS_ANG * np.sqrt(dof.n_radius_t)  # 径向半径
    theta = dof.n_angle_t  # 方位角
    
    # 圆柱内的方向向量（垂直于 NO 轴）
    # 构造垂直于 no_axis 的两个正交向量：v_perp1, v_perp2
    v_perp1 = unit(np.cross(no_axis, plane_normal))
    v_perp2 = unit(np.cross(no_axis, v_perp1))
    
    # 圆柱坐标系内的径向位置
    cyl_radial = r_cyl * (np.cos(theta) * v_perp1 + np.sin(theta) * v_perp2)
    
    # N 的位置：O + 沿中轴线偏移 + 径向偏移 + 离平面偏移
    # 沿中轴线：取基态 N 到 O 的距离作为默认偏移
    n_axial_offset = float(np.linalg.norm(N_base - O))
    n_candidate_center = O + n_axial_offset * no_axis + cyl_radial
    
    # 离平面偏移：GEN_N_CYLINDER_OFF_PLANE_ANG（可正可负）
    off_plane = cfg.GEN_N_CYLINDER_OFF_PLANE_ANG * plane_normal
    N = n_candidate_center + off_plane
    
    coords[iN] = N

    # ========== C5 球体采样 ==========
    C5 = C5_base + dof.c5_offset
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
    返回元素的电子散射因子 f(s)，优先使用 DPWA（params/DPWA/）。
    s 必须与 cfg.S_GRID 长度相同（681 点）；若不同则回退到 Cromer-Mann。
    """
    if len(s) == len(cfg.S_GRID) and np.allclose(s, cfg.S_GRID):
        try:
            return _load_dpwa_factors()[element]
        except (FileNotFoundError, KeyError):
            pass
    # 回退：Cromer-Mann 参数化
    pars = cfg.SCATTERING_COEFFS[element]
    a = np.array(pars["a"], dtype=np.float64)
    b = np.array(pars["b"], dtype=np.float64)
    c = float(pars["c"])
    q = s / (4.0 * math.pi)
    out = np.zeros_like(s, dtype=np.float64)
    for ai, bi in zip(a, b):
        out += ai * np.exp(-bi * (q * q))
    out += c
    return out


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


def add_noise(signal: np.ndarray, rng: np.random.Generator, s: np.ndarray = cfg.S_GRID) -> np.ndarray:
    """
    叠加噪声（可在 config.py 开关与调量级）
    """
    if not cfg.NOISE_ENABLE:
        return signal

    y = signal.astype(np.float64).copy()

    # 1) 高斯噪声（可随 s 衰减）
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
# 标签：7 维关键距离（O-N, O-C5, N-C5, N-C2, N-C4, C5-C2, C5-C4）
# ============================================================

def label_from_backbone(backbone_7x3: np.ndarray) -> np.ndarray:
    """
    根据 cfg.LABEL_FLAT_DIM 返回对应维度的距离标签。
    3-dim: [d(O-N), d(O-C5), d(N-C5)]
    7-dim: [d(O-N), d(O-C5), d(N-C5), d(N-C2), d(N-C4), d(C5-C2), d(C5-C4)]
    """
    iO  = cfg.NMM_ATOM_INDEX["O7"]
    iN  = cfg.NMM_ATOM_INDEX["N3"]
    iC5 = cfg.NMM_ATOM_INDEX["C5"]
    coords = backbone_7x3
    d_ON   = float(np.linalg.norm(coords[iO]  - coords[iN]))
    d_OC5  = float(np.linalg.norm(coords[iO]  - coords[iC5]))
    d_NC5  = float(np.linalg.norm(coords[iN]  - coords[iC5]))
    if cfg.LABEL_FLAT_DIM == 3:
        return np.array([d_ON, d_OC5, d_NC5], dtype=np.float32)
    # 7-dim: 添加到固定锚点 C2/C4 的距离
    iC2 = cfg.NMM_ATOM_INDEX["C2"]
    iC4 = cfg.NMM_ATOM_INDEX["C4"]
    d_NC2  = float(np.linalg.norm(coords[iN]  - coords[iC2]))
    d_NC4  = float(np.linalg.norm(coords[iN]  - coords[iC4]))
    d_C5C2 = float(np.linalg.norm(coords[iC5] - coords[iC2]))
    d_C5C4 = float(np.linalg.norm(coords[iC5] - coords[iC4]))
    return np.array([d_ON, d_OC5, d_NC5, d_NC2, d_NC4, d_C5C2, d_C5C4], dtype=np.float32)


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
            self.y = torch.from_numpy(np.array(h5f["y"])).float()  # (N, 9)
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
    输入 (B,1,S=681), 输出 (B,9)
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
    输入 (B,1,S=681), 输出 (B,9)
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

