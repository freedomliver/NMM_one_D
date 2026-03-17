"""
PROJECT_FRAMEWORK.md 快速导航索引
用于快速定位不同功能模块的位置，避免重复读整个文档
"""

## 📍 核心章节快速导航

### 1. 顶层设计 (L1-L40)
- **问题与数据** (L3-L6)：输入输出定义、分子结构
- **主要流程（端到端）** (L8-L14)：5步核心流程概览

### 2. config.py (L17-L85)
- **角色** (L18-L20)：文件功能描述
- **关键对象与常用入口** (L22-L56)：所有可改参数的快速查询

### 3. functions.py - 基础工具 (L88-L142)
- **set_global_seed()** (L91-L92)
- **unit()** (L93-L94)
- **pairwise_dist_matrix()** (L95-L96)

### 4. functions.py - 几何与采样 (L98-L175) ⭐ **重点**
| 子章节 | 行号范围 | 用途 |
|--------|---------|------|
| **几何采样策略** | L86-L176 | 新的圆柱/球约束采样方法 |
| 采样参数表 | L92-L100 | GEN_N_* 和 GEN_C5_* 参数 |
| N3采样算法 | L107-L113 | 圆柱面均匀采样 |
| C5采样算法 | L115-L119 | 球面均匀采样 |
| 键长约束 | L121-L124 | 后处理检验规则 |
| SampledDOF类 | L126-L131 | 数据结构定义 |
| 坐标生成流程 | L133-L157 | generate_backbone_coords_from_dof() |
| 函数接口 | L159-L175 | 6个关键函数说明 |

### 5. functions.py - 散射与噪声 (L177-L198)
- **atomic_scattering_factor_cromer_like()** (L179-L181)
- **compute_1d_scattering_signal()** (L182-L186)
- **smooth_moving_average()** (L187-L188)
- **add_noise()** (L189-L190)

### 6. functions.py - 标签与归一化 (L192-L211)
- **label_from_backbone()** (L193-L194)
- **Normalization dataclass** (L195-L196)
- **compute_normalization()** (L197-L198)
- **save/load_normalization()** (L199-L200)
- **normalize/denormalize** (L201-L202)

### 7. functions.py - 数据集与网络 (L213-L236)
- **write/read_h5_dataset()** (L214-L215)
- **H5Dataset** (L216-L217)
- **ResidualBlock1D** (L218-L221)
- **NMMRegressor1D** (L222-L236)

### 8. train_gen_NMM.py (L238-L305)
- **角色** (L239-L241)
- **命令行参数** (L243-L256)
- **核心流程** (L257-L263)
- **返回值** (L264-L268)
- **采样失败排查** (L269-L305)

### 9. train.py (L307-L328)
- **角色** (L308-L310)
- **使用建议** (L311-L328)

### 10. inference.py (L330-L348)
- **角色** (L331-L337)
- **使用建议** (L338-L348)

---

## 🔧 常见任务快速查询

### 修改采样参数
➜ 先改 **config.py L161-172**（新采样参数），然后查看：
- PROJECT_FRAMEWORK_INDEX.md L92-L100（参数表说明）
- PROJECT_FRAMEWORK.md L92-L100（参数表）

### 修改采样算法
➜ 查看 PROJECT_FRAMEWORK.md：
- L107-L113：N3采样逻辑
- L115-L119：C5采样逻辑
- L133-L157：坐标生成流程

### 修改噪声模型
➜ 查看 PROJECT_FRAMEWORK.md L189-L190（add_noise函数说明）
➜ 修改 functions.py 中的 add_noise() 实现

### 修改神经网络
➜ 查看 PROJECT_FRAMEWORK.md L222-L236（NMMRegressor1D说明）
➜ 修改 functions.py 中的网络类定义

### 调试采样失败
➜ 查看 PROJECT_FRAMEWORK.md L269-L305（train_gen_NMM.py采样失败排查）

---

## 📊 文件统计

- **总行数**：348 行
- **顶层设计**：40 行（L1-40）
- **配置说明**：68 行（L17-85）
- **采样设计**：94 行（L86-180）⭐ **最常改**
- **散射/噪声**：22 行（L177-198）
- **标签/归一化**：20 行（L192-211）
- **数据/网络**：24 行（L213-236）
- **三个脚本说明**：112 行（L238-348）

---

## 💡 使用建议

1. **任务开始前**：先读这个索引，确定要修改的章节行号范围
2. **定位快速**：用 `read_file` 时直接读对应行号范围，不需要读整个文件
3. **批量修改**：同一个文件的多个位置修改时，用 `multi_replace_string_in_file` 一次性做完
4. **验证修改**：修改后只需查看该章节几行，不用重新读整个文档

