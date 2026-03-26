"""
本项目配置文件（重要：你需要经常改的参数都集中在这里）

你最常修改的内容（建议只改这里，不要到处改脚本）：
- S 轴基网格 / 主线截断区间：`S_GRID` / `S_GRID_TRUNC`
- 分子原子顺序与映射：`NMM_ATOM_ORDER`、`NMM_ATOM_Z`、`NMM_ATOM_INDEX`
- H 原子“锁死”策略（相对对应 C 的固定偏移量）：`H_LOCKED_OFFSETS_ANG`
- 训练数据生成自由度范围/限制条件：`GEN_*` 一组参数
- 训练集筛选条件（例如键长范围、夹角范围、异常样本剔除）：`FILTERS`
- 噪声策略与量级：`NOISE_*`
- 数据/模型输出路径：`PATHS`

环境建议：
- Mac (M3) 上用 conda 建一个 python>=3.11 的环境，按 requirements.txt 安装即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


V2_LAST_TAG = "v2_last"
DEFAULT_MODEL_VERSION = "v2"
DEFAULT_DELTA_SM = True
DEFAULT_EQUIL_FRACTION = 0.30
DEFAULT_NO_DRIFT = True
DEFAULT_TRUNCATE_S = True


# =========================
# 路径配置
# =========================

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent
    params_dir: Path = root / "params"
    training_data_dir: Path = root / "file_training_data"
    models_dir: Path = root / "params" / "models"
    results_dir: Path = root / "results"

    # V2_last 默认产物（训练集统计量 / 数据集 / 模型权重）
    norm_json: Path = params_dir / f"normalization_{V2_LAST_TAG}.json"
    train_h5: Path = training_data_dir / f"train_NMM_{V2_LAST_TAG}.h5"
    val_h5: Path = training_data_dir / f"val_NMM_{V2_LAST_TAG}.h5"
    checkpoint_pt: Path = models_dir / f"nmm_{V2_LAST_TAG}.pt"
    benchmark_md: Path = results_dir / "BENCHMARK_V2_LAST.md"

    exp_prepump_h5: Path = training_data_dir / f"exp_prepump_{V2_LAST_TAG}.h5"
    exp_full_h5: Path = training_data_dir / f"exp_full_{V2_LAST_TAG}.h5"


PATHS = Paths()


# =========================
# S 轴配置（一维信号输入长度）
# =========================

# 说明：
# - 内部散射信号仍在 681 点基网格上计算，再统一截断到 V2_last 主线输入
# - 如果你的实验/Matlab径向积分得到的 s0 已经有明确的 s 数组，请把这里替换为真实 s 数组
S_LEN = 681
S_MIN = 0.0
S_MAX = 15.0
S_GRID = np.linspace(S_MIN, S_MAX, S_LEN).astype(np.float64)

# 计算信号时常用的”有效区间”裁剪（来自旧脚本经验：s<1 与 s>12 置零）
S_CUTOFF_MIN = 1.0
S_CUTOFF_MAX = 12.0


# =========================
# V2_last 默认截断 S 轴（实验优化：只保留有信息的区间）
# =========================
# 实验数据有效范围约 [1.5, 9.0]，信号计算 cutoff [1.0, 12.0]
# 截断到 [0.5, 10.0] 保留 buffer，从 681 → 431 点（减少 37%）
S_TRUNC_MIN = 0.5
S_TRUNC_MAX = 10.0
_s_trunc_mask = (S_GRID >= S_TRUNC_MIN) & (S_GRID <= S_TRUNC_MAX)
S_GRID_TRUNC = S_GRID[_s_trunc_mask].copy()
S_LEN_TRUNC = len(S_GRID_TRUNC)  # ~431


# =========================
# NMM 原子顺序与元素信息
# =========================

# 统一原子顺序（参考 reference/inference_NMM.py 里的标签）
NMM_ATOM_ORDER = ["C1", "C2", "N3", "C4", "C5", "C6", "O7"]

# 原子序数（用于散射因子；你也可以改成别的分子）
NMM_ATOM_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}

# 名字到索引
NMM_ATOM_INDEX = {name: i for i, name in enumerate(NMM_ATOM_ORDER)}


# =========================
# 基态骨架坐标（Å）
# =========================

# 重要说明（你很可能需要改这里）：
# - 这里必须提供 7x3 的骨架坐标（不含 H），顺序必须与 NMM_ATOM_ORDER 一致
# - 目前直接从 `params/coords/NMM_eq.xyz` 读取得到的真实 NMM 基态结构（C/N/O 七个重原子）
# - 顺序按 `NMM_ATOM_ORDER = ["C1","C2","N3","C4","C5","C6","O7"]` 对应 xyz 文件中的
#   C, C, N, C, C, C, O
NMM_BASE_COORDS_ANG = np.array(
    [
        [ 1.1738643646, -0.2150038459, -1.1426381082],  # C1
        [ 1.2005823301,  0.1916224380,  0.3269223126],  # C2
        [-0.0000000094, -0.2957761321,  1.0045123511],  # N3（将被约束在平面）
        [-1.2005823170,  0.1916224591,  0.3269223282],  # C4
        [ 0.0000000090,  0.0222585972,  2.4240707563],  # C5（将被采样变化）
        [-1.1738643499, -0.2150038773, -1.1426380616],  # C6
        [-0.0000000389,  0.2643605276, -1.7911323609],  # O7
    ],
    dtype=np.float64,
)


# =========================
# H 原子“锁死”偏移（Å）
# =========================

# 你要求：H 原子参与干涉项，但不作为自由度；每个 H 与对应 C “完全锁死”。
# 实现方式：H 的坐标 = parent_C 坐标 + 固定偏移向量（该偏移向量在基态结构中确定，后续只做平移不做旋转/拉伸）。
#
# 你需要做的事情：
# 1) 先确定每个 C 对应的 H 数量（例如平面 C1/C2/C4/C6 各 1 个 H，C5 甲基 3 个 H 等）
# 2) 把每个 H 在基态下相对 parent_C 的偏移向量写进这里
#
# 下面的偏移是从 `params/coords/NMM_eq.xyz` 基态结构中自动推断得到的真实值：
# - 先用最近邻原则把每个 H 绑定到最近的重原子（C1/C2/N3/C4/C5/C6/O7）
# - 再用 H 坐标减去对应重原子坐标，得到“锁死偏移”
H_LOCKED_OFFSETS_ANG = {
    # parent atom name -> list of offsets (x,y,z)
    "C1": [
        np.array([ 0.0480368408, -1.0960326673, -0.0810621195]),
        np.array([ 0.8470925666,  0.4315569991, -0.5366668726]),
    ],
    "C2": [
        np.array([0.8820305916, -0.4336429506, 0.4828809427]),
        np.array([0.0841669519,  1.1018956088, 0.0701654072]),
    ],
    "C4": [
        np.array([-0.0841669594,  1.1018956149, 0.0701654166]),
        np.array([-0.8820305073, -0.4336430084, 0.4828809998]),
    ],
    "C5": [
        np.array([-0.0000000180,  1.0886413436, 0.1968581524]),
        np.array([ 0.8850168553, -0.4317612451, 0.4749975387]),
        np.array([-0.8850168324, -0.4317612689, 0.4749975536]),
    ],
    "C6": [
        np.array([-0.0480368693, -1.0960326682, -0.0810621430]),
        np.array([-0.8470923942,  0.4315570916, -0.5366669858]),
    ],
    # N3 / O7 在该基态结构中没有直接绑定的 H
}


# =========================
# 训练数据生成：自由度与范围（你会经常改）
# =========================

# 目标：只让 N 基相关原子动（N3 与 C5），O7 + 平面四个 C（C1/C2/C4/C6）锁死。
FIXED_ATOMS = ["C1", "C2", "C4", "C6", "O7"]
MOVING_ATOMS = ["N3", "C5"]

# 新采样策略（2026/3/12 修改）：
# - N3：在 NO 中轴面上离中轴面 2Å，以原坐标为圆心半径 3.5Å 的圆柱内均匀随机采样
# - C5：以原坐标为圆心的半径 5Å 的球体内均匀随机采样
# 采样后限制：N-O 键长 < C2-O 键长（即 r_NO < r_C2O），否则丢弃

GEN_N_SAMPLES_TRAIN = 20000  # 默认先小一点方便验证；你可改到 1_000_000
GEN_N_SAMPLES_VAL = 2000
GEN_RANDOM_SEED = 20260311

# ========== N 原子采样参数（圆柱约束）==========
# N3 在 NO 中轴面上离中轴面的距离（Å）【调试标记】
GEN_N_CYLINDER_OFF_PLANE_ANG = 2.0

# N3 采样圆柱的半径（Å）【调试标记】
GEN_N_CYLINDER_RADIUS_ANG = 2.0  # was 3.5; focused on relevant dynamics range

# ========== C5 原子采样参数（球体约束）==========
# C5 采样球体的半径（Å）【调试标记】
GEN_C5_SPHERE_RADIUS_ANG = 3.0  # was 5.0; caps N-C5 ~5 Å to match experimental range

# ========== 采样后的键长约束 ==========
# N-O 键长必须 < C5-O 键长（外面那个 C 原子），否则丢弃该样本
# （这个约束在 generate_backbone_coords_from_dof 中实现）
GEN_CHECK_N_O_SHORTER_THAN_C5_O = True


# =========================
# 训练集筛选条件（不合理样本剔除）
# =========================

# 说明：生成时会先尝试构型求解；若无解/数值不稳定会丢弃并重采样。
# 你当前希望“完全去掉采样阶段的额外限制条件”，因此这里不再施加任何二次筛选。
FILTERS = {
    # 为了兼容现有代码，这里保留键名但取“无约束”数值：
    # - min_trilateration_height_ang = 0.0: 不再因为高度过小而额外丢弃样本
    # - max_abs_oop_deg = 180.0: 相当于不过滤 out-of-plane 角度
    "min_trilateration_height_ang": 0.0,
    "max_abs_oop_deg": 180.0,
}


# =========================
# 噪声策略（你会经常改）
# =========================

# 你要求：训练信号需要噪声，可叠加多种类型。
# 我们提供 3 种噪声，可按开关叠加：
# - gaussian: 高斯白噪声（可随 s 衰减）
# - drift: 低频漂移（用平滑随机曲线模拟）
# - spikes: 少量尖峰（模拟坏点）

NOISE_ENABLE = True

NOISE_GAUSS_STD_RANGE = (0.002, 0.02)  # 对 signal 的相对量级（你需要按真实实验量级调）
NOISE_GAUSS_SCALE_WITH_S = True        # True: std * (1/(s+eps)) 形式

NOISE_DRIFT_ENABLE = True
NOISE_DRIFT_STD_RANGE = (0.0, 0.01)
NOISE_DRIFT_SMOOTH_WINDOW = 31         # 必须是奇数，越大越“低频”

NOISE_SPIKE_ENABLE = False
NOISE_SPIKE_PROB = 0.002
NOISE_SPIKE_SCALE = 0.2


# =========================
# 标签定义（V2_last 主线：7 个距离）
# =========================

LABEL_ATOMS = ["O7", "N3", "C5", "C2", "C4"]
LABEL_PAIR_NAMES = ["O-N", "O-C5", "N-C5", "N-C2", "N-C4", "C5-C2", "C5-C4"]
LABEL_FLAT_DIM = 7
LEGACY_LABEL_NAME_MAP = {
    3: LABEL_PAIR_NAMES[:3],
    7: LABEL_PAIR_NAMES,
    9: ["O-O", "O-N", "O-C5", "N-O", "N-N", "N-C5", "C5-O", "C5-N", "C5-C5"],
}


# =========================
# 散射因子
# =========================

# V2_last 主线固定使用 params/DPWA/f*.mat。
# 不再保留 Cromer-Mann 或其他回退散射因子实现。
