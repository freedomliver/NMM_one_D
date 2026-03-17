"""
本文件是给大模型阅读的“项目框架描述”，便于后续自动改进/扩展本工程。
请优先阅读本文件，再结合各源码文件的 docstring。
"""

## 顶层设计概览

### 问题与数据
- 输入：UED 径向积分得到的一维衍射信号 \(int(s)\) / \(\Delta sM(s)\)，在代码中统一为长度 `config.S_LEN`（默认 681）的一维数组。
- 输出：选定原子子集的全距离矩阵，本项目当前为 `["O7","N3","C5"]` 的 \(3\times 3\) 距离矩阵，训练/推理时展平成长度 9 的向量。
- 分子：NMM。骨架原子为 7 个（`NMM_ATOM_ORDER`），H 原子通过“锁死偏移”从对应 C 原子生成，只参与散射，不作为自由度。

### 主要流程（端到端）
1. 通过自由度采样生成一批 NMM 骨架构型（7×3）。
2. 展开为“骨架+H”的完整坐标，并计算所有原子散射信号得到一维 `signal(s)`。
3. 对信号加噪声，生成训练/验证集（`x`）与对应距离矩阵标签（`y`），写入 h5 + 归一化参数 json。
4. 使用 1D CNN（`NMMRegressor1D`）从 `x` 回归到 `y`，训练并保存最佳权重。
5. 在推理阶段，对合成或实验数据的一维信号进行同样的归一化与前向传播，得到预测的距离矩阵。

---

## config.py

### 角色
- 集中管理所有“可改参数”，包括：
  - 路径（数据、模型、参数）
  - S 轴（长度与范围）
  - 原子顺序、索引映射与标签原子集合
  - NMM 基态骨架坐标（7×3）
  - H 原子锁死偏移定义
  - 自由度采样范围（键长、面外角等）
  - 构型筛选条件（几何稳定性等）
  - 噪声开关与量级
  - 散射因子模型与参数

### 关键对象与常用入口
- `Paths` / `PATHS`：统一的路径配置。
- `S_LEN`, `S_MIN`, `S_MAX`, `S_GRID`, `S_CUTOFF_MIN`, `S_CUTOFF_MAX`：
  - 若更换实验/模拟的 s 轴定义，先改这里。
- `NMM_ATOM_ORDER`, `NMM_ATOM_Z`, `NMM_ATOM_INDEX`：
  - 控制原子顺序与散射元素类型。
- `NMM_BASE_COORDS_ANG`：
  - 7×3 NMM 基态骨架坐标；后续自由度是在此基础上施加的。
- `H_LOCKED_OFFSETS_ANG`：
  - dict[parent_atom_name -> List[偏移向量]]；定义每个 H 相对其 C 的固定坐标偏移。
- `FIXED_ATOMS`, `MOVING_ATOMS`：
  - 定义哪些骨架原子在自由度中保持锁死，哪些参与变化。
- `GEN_N_SAMPLES_TRAIN`, `GEN_N_SAMPLES_VAL`, `GEN_*R_*`, `GEN_C5_OOP_DEG_RANGE`：
  - 控制训练数据规模与几何自由度采样范围。
- `FILTERS`：
  - 如 `min_trilateration_height_ang`, `max_abs_oop_deg` 控制几何解的稳定性筛选。
- `NOISE_*`：
  - 开关及各类噪声的量级与形态。
- `LABEL_ATOMS`, `LABEL_SIZE`, `LABEL_FLAT_DIM`：
  - 控制标签原子的集合与展平维度（当前是 O7/N3/C5 的 3×3）。
- `SCATTERING_FACTOR_MODEL`, `SCATTERING_COEFFS`：
  - 散射因子模型类型及其参数（可换成基于 DPWA/mat 文件的 tabulated 模型）。

---

## functions.py

### 角色
- 集中实现本项目的所有“纯函数”和核心组件，包括：
  - 几何与自由度采样（从参数空间到 7×3 骨架坐标）
  - H 锁死展开与完整原子坐标构造
  - 散射因子计算与一维信号生成
  - 噪声叠加
  - 标签计算（距离矩阵）
  - 归一化参数计算/保存/加载及归一化操作
  - h5 IO 与 PyTorch Dataset 封装
  - 1D CNN 回归模型定义

### 函数/类一览与用途

#### 基础工具
- `set_global_seed(seed)`：
  - 统一设置 numpy / torch 的随机种子。
- `unit(v, eps)`：
  - 计算单位向量；长度过小则返回零向量。
- `pairwise_dist_matrix(coords)`：
  - 输入 (N,3)，输出 (N,N) 的距离矩阵。

#### 几何构型与自由度

##### 几何采样策略（2026/3/12 更新）

本项目采用**几何约束法**进行采样，而非传统的多距离参数化：

###### 固定与移动原子
- **固定原子**（6个）：O7、C1、C2、C4、C6（及其锁死的 H）
- **移动原子**（2个）：N3（圆柱面约束）、C5（球面约束）

###### 采样参数（config.py, lines 161-172）
| 参数名 | 含义 | 默认值 | 单位 | 调试标记 |
|-------|------|---------|------|---------|
| `GEN_N_CYLINDER_OFF_PLANE_ANG` | N3 离 NO 轴向平面的距离 | 2.0 | Å | 【调试】 |
| `GEN_N_CYLINDER_RADIUS_ANG` | N3 圆柱采样半径 | 3.5 | Å | 【调试】 |
| `GEN_C5_SPHERE_RADIUS_ANG` | C5 球采样半径 | 5.0 | Å | 【调试】 |
| `GEN_CHECK_N_O_SHORTER_THAN_C5_O` | 是否检查键长约束 | True | - | - |

###### 采样方法

**N3 采样（圆柱面）：**
1. 建立 NO 轴向坐标系（以 O7 为原点）
   - Z 轴：O7→N3 轴向
   - X、Y 轴：垂直于 Z 且相互正交的两个向量
2. 在 (X,Y) 平面内均匀采样圆：
   - 随机 $\theta \in [0, 2\pi)$，$u \in [0,1)$
   - 使用 $r = R \sqrt{u}$ 实现面积均匀分布（关键）
   - 圆柱点：$(X_i, Y_i, Z_i) = (r\cos\theta, r\sin\theta, z_{off})$，其中 $R=$ `GEN_N_CYLINDER_RADIUS_ANG`，$z_{off}=$ `GEN_N_CYLINDER_OFF_PLANE_ANG`
3. 变换回全局坐标系

**C5 采样（球面）：**
1. 以 N3 为中心构造球面采样：
   - 随机 $\phi \in [0, 2\pi)$，$\theta \in [0, \pi]$（标准球坐标）
   - 随机 $v \in [0,1)$
   - 使用 $r = R v^{1/3}$ 实现体积均匀分布（关键）
   - 球面点：O7 + $r(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$，其中 $R=$ `GEN_C5_SPHERE_RADIUS_ANG`

**键长约束（后处理）：**
- 检查是否满足：$r(N_3-O_7) < r(C_5-O_7)$（C5 必须比 N3 离 O7 更远）
- 若不满足，触发异常，采样逻辑重试
- 当前实验数据显示约 80% 的采样通过此约束

###### 核心数据结构

`SampledDOF` 数据类（functions.py, lines 145-151）：
```python
@dataclass
class SampledDOF:
    n_center: np.ndarray        # (3,) - N3 采样中心（对应 O7 位置）
    n_radius_t: np.ndarray      # (1,) - 圆柱半径参数 t ∈ [0,1)
    n_angle_t: np.ndarray       # (1,) - 圆周角参数 ∈ [0, 2π)
    c5_offset: np.ndarray       # (3,) - 相对 N3 的偏移向量
```

###### 坐标生成流程

`generate_backbone_coords_from_dof()` 函数（functions.py, lines 253-338）完成以下步骤：

1. **建立 NO 轴向坐标系**
   - 计算 $\vec{O_7}$ 和 $\vec{N_3}$ 方向
   - 通过 Gram-Schmidt 正交化得到垂直向量

2. **N3 采样点变换**
   - 圆柱面上的点 $(r\cos\theta, r\sin\theta, z_{off})$
   - 变换至全局坐标

3. **C5 采样点变换**
   - 相对于 N3 的球面偏移
   - 变换至全局坐标

4. **键长约束检验**
   - 若 $r(N-O) \geq r(C5-O)$，抛异常
   - 触发上层采样循环重试

###### 函数接口

- `trilaterate_three_spheres(p1,p2,p3,r1,r2,r3,min_h)`：
  - 三球交点（两解），用于从三个已知原子和三条距离恢复目标原子坐标。
  - `min_h` 用于剔除几何病态的解（高度太小）。
- `circle_intersections_in_plane(c1,c2,r1,r2)`：
  - 在平面内求两圆交点（用于把 N3 限制在给定平面，例如 z=0 平面）。
- `SampledDOF`（dataclass）：
  - 描述一次采样的几何自由度。
- `sample_dof(rng)`：
  - 根据 `config.GEN_*` 中定义的圆柱/球约束随机采样一组 `SampledDOF`。
- `build_full_coords_with_locked_H(backbone_coords_7x3, h_locked_offsets)`：
  - 输入 7×3 骨架坐标，依据 `H_LOCKED_OFFSETS_ANG` 展开出所有 H。
  - 输出 `(coords_all, atom_names_all, atom_elements_all)`。
- `generate_backbone_coords_from_dof(dof, base_coords, min_h)`：
  - 给定一份自由度（`SampledDOF`），在 `NMM_BASE_COORDS_ANG` 附近构造新的 7×3 骨架。
  - 建立 NO 轴向平面，应用圆柱/球约束，执行键长约束检验。
  - 若几何无解/不稳定/违反约束，抛异常，由上层采样逻辑重试。

#### 散射与噪声
- `atomic_scattering_factor_cromer_like(element, s)`：
  - 根据 `SCATTERING_COEFFS` 与类 Cromer-Mann 形式计算 f(s)。
  - 若未来换为 tabulated f(s) 数据，可在此函数中重写逻辑。
- `compute_1d_scattering_signal(coords_all,elements_all,s,s_cut_min,s_cut_max,eps)`：
  - 基于全原子（含 H）的坐标与元素类型，计算一维散射信号：
    - 先构造每个原子的 f_i(s)，再累积 IA / IM，最后计算 `signal = s*IM/IA` 并在给定 s 区间外置零。
  - 若需要更复杂的 delta sM/背景扣除，可以在此处统一修改。
- `smooth_moving_average(x,window)`：
  - 简单的滑动平均滤波，用于生成低频漂移信号。
- `add_noise(signal,rng,s)`：
  - 根据 `NOISE_*` 叠加多种噪声（高斯白噪 + 低频漂移 + 尖峰）。

#### 标签与归一化
- `label_from_backbone(backbone_7x3)`：
  - 从骨架坐标中提取 `LABEL_ATOMS`（默认 O7/N3/C5），构造 3×3 距离矩阵并展平为 9 维。
- `Normalization`（dataclass）：
  - 保存 x/y 的均值与标准差，支持 json 序列化。
- `compute_normalization(x,y,eps)`：
  - 从整个训练集的 x/y 计算归一化参数。
- `save_normalization(norm,path)` / `load_normalization(path)`：
  - 读写归一化 json。
- `normalize_x(x,norm)` / `normalize_y(y,norm)` / `denormalize_y(y_hat,norm)`：
  - 归一化与反归一化操作。

#### 数据集与网络
- `write_h5_dataset(path,x,y)` / `read_h5_dataset(path)`：
  - 将 (N,S) 输入与 (N,D) 标签写入/读出 h5 文件。
- `H5Dataset(h5_path)`：
  - 基于 h5 的 PyTorch Dataset，把每个样本加载为 `(1,S)` 输入与标签向量。
- `ResidualBlock1D(nn.Module)`：
  - 简单的 1D 残差块：Conv1d-GELU-Conv1d + skip。
- `NMMRegressor1D(nn.Module)`：
  - 整体 1D CNN 结构：
    - Stem：Conv1d 若干层提取局部特征。
    - Blocks：多个 `ResidualBlock1D`。
    - Head：全局平均池化 + MLP 输出 `LABEL_FLAT_DIM`。
  - 可通过参数 `base_ch` / `n_blocks` 控制宽度与深度。

---

## train_gen_NMM.py

### 角色
- 从自由度空间采样构型，调用 `functions.py` 中工具生成训练/验证数据，并保存：
  - 训练/验证集 h5：`file_training_data/train_NMM.h5`、`val_NMM.h5`
  - 归一化参数 json：`params/normalization.json`

### 关键函数
- `generate_dataset(n_samples,seed,max_tries_factor)`：
  - 循环：
    - 采样自由度：`sample_dof`。
    - 几何构建：`generate_backbone_coords_from_dof`。
    - H 展开：`build_full_coords_with_locked_H`。
    - 信号计算 + 噪声：`compute_1d_scattering_signal` + `add_noise`。
    - 标签计算：`label_from_backbone`。
  - 若几何/筛选失败则丢弃并重试，直到收集到指定样本数或达到最大尝试次数。
- `main()`：
  - 解析命令行参数（样本数、输出路径、seed 等）。
  - 分别生成 train/val 集。
  - 用 train 集计算归一化参数并保存。
  - 写入 h5。

---

## train.py

### 角色
- 加载 h5 数据与归一化参数，训练 1D 回归网络并保存最优模型。

### 关键逻辑
- `main()`：
  - 解析命令行超参数（数据路径、epochs、batch_size、lr、网络宽度/深度等）。
  - 读取归一化参数：`load_normalization`。
  - 构建 `H5Dataset` 与 `DataLoader`。
  - 实例化 `NMMRegressor1D`，优化器 AdamW，学习率退火 `CosineAnnealingLR`。
  - 训练循环：
    - 对每个 batch：
      - 用归一化参数对输入 `(B,1,S)` 与标签 `(B,D)` 做标准化。
      - 前向、计算 MSE 损失、反向传播、更新权重。
    - 计算验证集 MSE，若变好则保存 checkpoint。
  - Checkpoint 内容包括：
    - `model_state`（权重）
    - 归一化 json 的路径
    - 部分关键 config（如 S_LEN 与 LABEL_ATOMS）。

---

## inference.py

### 角色
- 封装两类推理场景：
  - 合成数据：评估模型在可控几何变化下的精度。
  - 实验数据：对实际 UED 径向积分信号进行距离矩阵回归。

### 核心函数
- `load_model(ckpt_path,device)`：
  - 从 checkpoint 加载 `NMMRegressor1D` 权重与归一化参数。
- `predict_batch(model,norm,x,device)`：
  - 对一批 (N,S) 输入做归一化与前向传播，输出去归一化后的预测距离矩阵。
- `synth_feature_set(n,seed)`：
  - 生成带有特定几何趋势的合成数据集，用于 sanity-check：
    - 当前策略：让 C2-N 键长在其范围内线性扫描，其它自由度随机。
- `load_experiment_signals(exp_dir,mat_key)`：
  - 从指定目录读取 `s0*.mat` 文件，取变量 `mat_key`（默认 "s0"）：
    - 对形状 (S,T)/(T,S) 自动调整为 (T,S) 后按文件拼接成 (N,S)。
    - 要求 S 维度与 `config.S_LEN` 一致，否则抛异常。
- `main()`：
  - 根据参数 `--mode` 切换：
    - `"synth"`：生成合成数据，推理并计算 MAE，结果写入 npz。
    - `"exp"`：读取实验数据，推理并写入 npz（包含 `y_hat` 和输入 `x`）。

---

## manual.md

### 角色
- 面向人类用户的“项目说明 + 使用指南”，包括：
  - 原始物理背景与 UED 数据处理流程。
  - 当前实现的文件结构与各文件职责。
  - 如何在本机配置环境与运行生成/训练/推理脚本。
  - 需要用户亲自确认/修改的物理参数清单。

---

## 未来改进入口建议（给大模型）

- 若要修改“几何自由度定义/采样策略”，首选修改：
  - `config.py` 中的 GEN_* / FILTERS；
  - `functions.generate_backbone_coords_from_dof` 中的构型构造逻辑；
  - 必要时扩展 `SampledDOF` 字段。
- 若要换别的标签（例如输出更多原子距离/角度）：
  - 在 `config.LABEL_ATOMS` / `LABEL_SIZE` / `LABEL_FLAT_DIM` 中定义新的标签集合。
  - 对应修改 `functions.label_from_backbone` 的构造方式。
- 若要引入更精确的散射模型：
  - 用 DPWA 或实验数据替换 `atomic_scattering_factor_cromer_like` 的实现；
  - 或增加新的 `SCATTERING_FACTOR_MODEL` 分支。
- 若要升级/更换网络结构：
  - 在 `functions.NMMRegressor1D` 内替换 CNN 结构，或添加新的模型类，在 `train.py` / `inference.py` 中改用新的类。

---

## 运行说明（真实终端 vs 沙箱，给大模型）

- **真实终端（用户自己跑）**：
  - 先在 shell 中执行：`conda activate ./nmm_ml`，激活项目环境。
  - 然后按 `manual.md` 的说明依次运行：
    - 环境安装：`pip install -r requirements.txt`（或用 `conda install` 安装主要依赖和 PyTorch）。
    - 生成数据：`python train_gen_NMM.py ...`
    - 训练模型：`python train.py ...`
    - 推理分析：`python inference.py ...`
  - 若在这一步遇到 SSL / 网络 / 镜像等安装问题，只能在真实系统层面解决，这不在本项目代码范围内。

- **沙箱（大模型内部使用的 `python`/`pip`）**：
  - 理论上也应先尝试执行一次 `conda activate ./nmm_ml`，以便在“若沙箱真的能看到同名环境”的情况下保持行为一致；
  - 但在当前实现中，沙箱通常**看不到**用户的 conda 环境，上述命令大概率不会生效，因此实际仍是使用沙箱自带的 `python` / `pip`。
  - 为了避免与你本机的依赖安装策略冲突，**沙箱内不再主动执行 `pip install -r requirements.txt`**；依赖安装全部交由你在真实终端完成。
  - 每次进入沙箱时，推荐按以下顺序操作，**且在每次运行任何 `python` 命令之前都重复执行第 1 步用于确认环境**：
    1. 用 `pwd` / `ls` / `python --version` / `pip --version` **以及** `python -c "import sys; print(sys.executable)"` 确认：
       - 当前工作目录是 `/Users/jinc_air/NMM_ML`；
       - Python 来自你希望使用的环境（若沙箱能看到 `./nmm_ml`，可从 `sys.executable` 路径中看到 `nmm_ml` 字样），否则视为环境未正确激活。
    2. 直接尝试导入关键依赖而**不做任何安装**，例如：
       - `python -c "import torch, h5py, numpy; print('imports_ok')"`（根据需要增减模块）。
    3. 只有在 `import torch` 等核心依赖**已经成功**的前提下，才做小规模自测，例如：
       - `python train_gen_NMM.py --train_n 200 --val_n 20 --seed 123`
       - `python train.py --epochs 2 --batch_size 64 --lr 3e-4 --seed 123`
       - `python inference.py --mode synth --synth_n 2000 --seed 123`
    4. 一旦出现 `ModuleNotFoundError: No module named 'torch'` 或类似缺包错误：
       - **不要在沙箱里尝试 `pip install`**；
       - 只在对话中说明“需要用户在真实终端的 conda 环境中安装对应包（例如先在本机运行 `pip install -r requirements.txt` 或 `conda install pytorch -c pytorch`）”，然后继续专注于代码/逻辑层面的修改。

