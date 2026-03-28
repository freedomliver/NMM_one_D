# prepump_joint_train_v6_fix

修正版目标：
- 对标 `V3` 的训练规模与低噪声口径
- 使用 `8 labels = 7 distances + h_C5_signed`
- experimental pre-pump 样本固定使用 `alpha=0.015` 重新导出的 `1400` 条
- merged dataset
- 保存每个 epoch checkpoint，并逐个跑实验推理

固定配置：
- model: `V2`
- signal: `delta_sm`
- scattering: `DPWA`
- input: `trunc 431`
- synthetic noise: `experiments/prepump_joint_train_v2/noise_profile_joint_train_v2.npz`
- exp h5: `file_training_data/exp_prepump_joint_alpha015_1400.h5`
- train synthetic: `60K`
- val synthetic: `10K`
- exp repeat: `2x`
- alpha for experiment inference sweep curve: fixed `0.015`

执行顺序：

1. 生成 synthetic 数据：

```bash
./nmm_ml/bin/python train_gen_NMM.py \
  --train_n 60000 \
  --val_n 10000 \
  --equil_fraction 0 \
  --no_drift \
  --noise_profile_npz experiments/prepump_joint_train_v2/noise_profile_joint_train_v2.npz \
  --train_h5 file_training_data/train_NMM_prepump_joint_v6_fix_syn.h5 \
  --val_h5 file_training_data/val_NMM_prepump_joint_v6_fix.h5 \
  --norm_json params/normalization_prepump_joint_v6_fix_syn.json
```

2. 合并训练集：

```bash
./nmm_ml/bin/python experiments/prepump_joint_train_v3/build_merged_train_h5.py \
  --synthetic_h5 file_training_data/train_NMM_prepump_joint_v6_fix_syn.h5 \
  --exp_h5 file_training_data/exp_prepump_joint_alpha015_1400.h5 \
  --out_h5 file_training_data/train_NMM_prepump_joint_v6_fix.h5 \
  --norm_json params/normalization_prepump_joint_v6_fix.json \
  --exp_repeat 2
```

3. 训练并保存每个 epoch checkpoint：

```bash
./nmm_ml/bin/python train.py \
  --train_h5 file_training_data/train_NMM_prepump_joint_v6_fix.h5 \
  --val_h5 file_training_data/val_NMM_prepump_joint_v6_fix.h5 \
  --norm_json params/normalization_prepump_joint_v6_fix.json \
  --out_ckpt params/models/nmm_prepump_joint_train_v6_fix.pt \
  --model_version v2 \
  --epochs 22 \
  --batch_size 112 \
  --lr 1.5e-4 \
  --save_each_epoch_dir experiments/prepump_joint_train_v6_fix/epoch_ckpts
```

4. 逐 epoch 实验评估：

```bash
./nmm_ml/bin/python experiments/prepump_joint_train_v6_fix/eval_epoch_curve.py \
  --ckpt_dir experiments/prepump_joint_train_v6_fix/epoch_ckpts \
  --alpha 0.015 \
  --out_csv experiments/prepump_joint_train_v6_fix/epoch_vs_exp_quality.csv \
  --out_png experiments/prepump_joint_train_v6_fix/figures/epoch_vs_exp_quality.png
```
