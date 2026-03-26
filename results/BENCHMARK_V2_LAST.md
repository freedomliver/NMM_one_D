# V2_last Benchmark

## Canonical Mainline

`V2_last` 现在只保留这一条主线：

- labels: `7`
- model: `V2`
- scattering: `DPWA`
- signal: `delta_sm`
- sampling: `equil_fraction = 0.30`
- noise: `nodrift`
- input: `trunc` (`s = [0.5, 10.0]`, `431` pts)
- experimental adaptation: `pre-pump weak supervision`

## Explicit Baseline Check

平衡距离只能从基态坐标显式计算，不能手填。

- 距离定义位置：[functions.py](/Users/jinc_air/NMM_ML/functions.py#L575)
- 平衡距离计算位置：[functions.py](/Users/jinc_air/NMM_ML/functions.py#L596)
- 推理结果写入 `equilibrium` 的位置：[inference.py](/Users/jinc_air/NMM_ML/inference.py#L585)
- 绘图读取 `equilibrium` 的位置：[plot_inference.py](/Users/jinc_air/NMM_ML/plot_inference.py)

当前 7-label 平衡值：

| label | equilibrium (A) |
|---|---:|
| O-N | 2.8512 |
| O-C5 | 4.2222 |
| N-C5 | 1.4547 |
| N-C2 | 1.4622 |
| N-C4 | 1.4622 |
| C5-C2 | 2.4224 |
| C5-C4 | 2.4224 |

## Recommended Commands

数据生成：

```bash
./nmm_ml/bin/python train_gen_NMM.py \
  --train_n 100000 \
  --val_n 10000
```

导出实验 pre-pump 弱监督集：

```bash
./nmm_ml/bin/python inference.py \
  --mode export_prepump_h5 \
  --out_h5 file_training_data/exp_prepump_v2_last.h5 \
  --alpha 0.015
```

阶段 1，合成基线训练：

```bash
./nmm_ml/bin/python train.py \
  --train_h5 file_training_data/train_NMM_v2_last.h5 \
  --val_h5 file_training_data/val_NMM_v2_last.h5 \
  --norm_json params/normalization_v2_last.json \
  --out_ckpt params/models/nmm_v2_last_base.pt \
  --model_version v2
```

阶段 2，实验弱监督微调：

```bash
./nmm_ml/bin/python train.py \
  --train_h5 file_training_data/train_NMM_v2_last.h5 \
  --val_h5 file_training_data/val_NMM_v2_last.h5 \
  --norm_json params/normalization_v2_last.json \
  --out_ckpt params/models/nmm_v2_last.pt \
  --model_version v2 \
  --init_ckpt params/models/nmm_v2_last_base.pt \
  --exp_h5 file_training_data/exp_prepump_v2_last.h5 \
  --exp_batch_size 64 \
  --lambda_exp_eq 0.30 \
  --lambda_exp_cons 0.05 \
  --lr 2e-5 \
  --epochs 8 \
  --patience 6 \
  --balanced_exp_weight 0.8
```

验证集推理：

```bash
./nmm_ml/bin/python inference.py \
  --ckpt params/models/nmm_v2_last.pt \
  --mode synth \
  --val_h5 file_training_data/val_NMM_v2_last.h5 \
  --out_npz results/v2_last/inference_v2_last_weak_val.npz
```

实验 `alpha sweep`：

```bash
./nmm_ml/bin/python inference.py \
  --alpha_sweep \
  --ckpt params/models/nmm_v2_last_balanced.pt \
  --exp_dir bootstrapping/results_240 \
  --alpha_values 0.006,0.008,0.010,0.012,0.015,0.018,0.020 \
  --out_npz results/v2_last/alpha_balanced/alpha_sweep.npz
```

实验正式推理（保留 200 组 bootstrap 方差）：

```bash
./nmm_ml/bin/python inference.py \
  --ckpt params/models/nmm_v2_last_balanced.pt \
  --mode exp \
  --alpha 0.008 \
  --out_npz results/v2_last/inference_v2_last_balanced_alpha008.npz
```

## Current Status

旧的 `full_s` / 非 DPWA / 3dim / 9dim benchmark 已退出主线。

当前 benchmark 将按这三个 checkpoint 重新生成：

- `params/models/nmm_v2_last_base.pt`：synthetic-only baseline
- `params/models/nmm_v2_last.pt`：best validation
- `params/models/nmm_v2_last_expbest.pt`：best experimental pre-pump deviation
- `params/models/nmm_v2_last_balanced.pt`：best balanced score

## Stage-2 Weak Supervision Results

阶段 2 训练日志：

- [train_v2_last_finetune.log](/Users/jinc_air/NMM_ML/results/v2_last/train_v2_last_finetune.log)

最终训练选择：

- `params/models/nmm_v2_last.pt`：best validation, `val_mae = 0.1834 A`
- `params/models/nmm_v2_last_expbest.pt`：best experimental pre-pump dev, `exp_dev = 0.0535 A`
- `params/models/nmm_v2_last_balanced.pt`：best balanced score, `bal = 0.2293`

验证集与实验域对照：

| checkpoint | val_mae (A) | key_mae (A) | best alpha | pre-pump mean abs dev (A) | notes |
|---|---:|---:|---:|---:|---|
| `nmm_v2_last.pt` | 0.1834 | 0.1346 | 0.012 | 0.0401 | best validation |
| `nmm_v2_last_expbest.pt` | 0.1875 | 0.1395 | 0.006 | 0.0243 | best experimental during training |
| `nmm_v2_last_balanced.pt` | 0.1844 | 0.1368 | 0.008 | 0.0238 | current overall best tradeoff |

当前推荐：

- 当前 stage-2 默认实验推理 checkpoint：[nmm_v2_last_balanced.pt](/Users/jinc_air/NMM_ML/params/models/nmm_v2_last_balanced.pt)
- 当前 stage-2 默认实验推理 alpha：`0.008`
- 当前 stage-2 默认实验结果：[inference_v2_last_balanced_alpha008.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_balanced_alpha008.npz)
- 当前 stage-2 默认实验 errorbar 图：[inference_v2_last_balanced_alpha008_errbar.png](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_balanced_alpha008_errbar.png)

相关产物：

- [inference_v2_last_weak_val.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_weak_val.npz)
- [inference_v2_last_expbest_val.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_expbest_val.npz)
- [inference_v2_last_balanced_val.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_balanced_val.npz)
- [alpha_bestval/alpha_sweep.log](/Users/jinc_air/NMM_ML/results/v2_last/alpha_bestval/alpha_sweep.log)
- [alpha_expbest/alpha_sweep.log](/Users/jinc_air/NMM_ML/results/v2_last/alpha_expbest/alpha_sweep.log)
- [alpha_balanced/alpha_sweep.log](/Users/jinc_air/NMM_ML/results/v2_last/alpha_balanced/alpha_sweep.log)
- [inference_v2_last_exp_alpha012_errbar.png](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_exp_alpha012_errbar.png)
- [inference_v2_last_expbest_alpha006_errbar.png](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_expbest_alpha006_errbar.png)
- [inference_v2_last_balanced_alpha008_errbar.png](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_balanced_alpha008_errbar.png)

## Stage-3 Full-Experiment Consistency

新增弱监督集：

- [exp_full_v2_last.h5](/Users/jinc_air/NMM_ML/file_training_data/exp_full_v2_last.h5)
  `9000 = 200 x 45` 条实验样本，含 `is_prepump`

训练方式：

- 从 [nmm_v2_last_balanced.pt](/Users/jinc_air/NMM_ML/params/models/nmm_v2_last_balanced.pt) 初始化
- pre-pump 样本使用平衡态约束
- 全部时间点使用 bootstrap consistency

训练日志：

- [train_v2_last_stage3.log](/Users/jinc_air/NMM_ML/results/v2_last/train_v2_last_stage3.log)

stage-3 结果：

| checkpoint | val_mae (A) | key_mae (A) | best alpha | pre-pump mean abs dev (A) | notes |
|---|---:|---:|---:|---:|---|
| `nmm_v2_last_stage3.pt` | 0.1727 | 0.1244 | 0.008 | 0.0200 | best validation / best balanced |
| `nmm_v2_last_stage3_expbest.pt` | 0.1759 | 0.1287 | 0.008 | 0.0168 | best experimental |
| `nmm_v2_last_stage3_balanced.pt` | 0.1727 | 0.1244 | 0.008 | 0.0200 | same metrics as best-val |

相对 stage-2 `balanced` 的提升：

- `val_mae`: `0.1844 -> 0.1727 A`
- `key_mae`: `0.1368 -> 0.1244 A`
- `pre-pump mean abs dev`: `0.0238 -> 0.0168 A`（以 `stage3_expbest` 计）

当前总推荐：

- 验证集与实验域折中最好：[nmm_v2_last_stage3.pt](/Users/jinc_air/NMM_ML/params/models/nmm_v2_last_stage3.pt)
- 如果只追求实验 pre-pump 最优：[nmm_v2_last_stage3_expbest.pt](/Users/jinc_air/NMM_ML/params/models/nmm_v2_last_stage3_expbest.pt)
- 默认实验推理 alpha：`0.008`
- 推荐实验图：[inference_v2_last_stage3_expbest_alpha008_errbar.png](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_expbest_alpha008_errbar.png)

stage-3 相关产物：

- [inference_v2_last_stage3_val.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_val.npz)
- [inference_v2_last_stage3_expbest_val.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_expbest_val.npz)
- [inference_v2_last_stage3_balanced_val.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_balanced_val.npz)
- [alpha_stage3/alpha_sweep.log](/Users/jinc_air/NMM_ML/results/v2_last/alpha_stage3/alpha_sweep.log)
- [alpha_stage3_expbest/alpha_sweep.log](/Users/jinc_air/NMM_ML/results/v2_last/alpha_stage3_expbest/alpha_sweep.log)
- [alpha_stage3_balanced/alpha_sweep.log](/Users/jinc_air/NMM_ML/results/v2_last/alpha_stage3_balanced/alpha_sweep.log)
- [inference_v2_last_stage3_alpha008.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_alpha008.npz)
- [inference_v2_last_stage3_expbest_alpha008.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_expbest_alpha008.npz)
- [inference_v2_last_stage3_balanced_alpha008.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_balanced_alpha008.npz)
- [inference_v2_last_stage3_alpha008_errbar.png](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_alpha008_errbar.png)
- [inference_v2_last_stage3_expbest_alpha008_errbar.png](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_expbest_alpha008_errbar.png)

## Stage-3 Reconstruction

基于当前实验域最优结果 [inference_v2_last_stage3_expbest_alpha008.npz](/Users/jinc_air/NMM_ML/results/v2_last/inference_v2_last_stage3_expbest_alpha008.npz)，
使用 [reconstruct.py](/Users/jinc_air/NMM_ML/reconstruct.py) 对 `N3` 和 `C5` 做 trilateration 重建。

重建结论：

- `45 / 45` 个时间点可解
- mean reconstruction residual: `0.02635 A`
- 残差主要集中在 `N-C5`
- 最大位移幅度：
  `N3 max = 0.3021 A`
  `C5 max = 1.0899 A`

推荐展示文件：

- [structure_snapshots.png](/Users/jinc_air/NMM_ML/results/v2_last/reconstruction_stage3_expbest/structure_snapshots.png)
- [coordinate_trajectories.png](/Users/jinc_air/NMM_ML/results/v2_last/reconstruction_stage3_expbest/coordinate_trajectories.png)
- [displacement_from_equilibrium.png](/Users/jinc_air/NMM_ML/results/v2_last/reconstruction_stage3_expbest/displacement_from_equilibrium.png)
- [reconstruction_residuals.png](/Users/jinc_air/NMM_ML/results/v2_last/reconstruction_stage3_expbest/reconstruction_residuals.png)

对照版：

- [results/v2_last/reconstruction_stage3_bestval/structure_snapshots.png](/Users/jinc_air/NMM_ML/results/v2_last/reconstruction_stage3_bestval/structure_snapshots.png)
