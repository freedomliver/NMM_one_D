# V2_last Runbook

## Goal

当前唯一主线：

- labels: `7`
- model: `V2`
- scattering: `DPWA`
- signal: `delta_sm`
- sampling: `equil_fraction=0.30`
- noise: `nodrift`
- input: `trunc` (`431` pts)
- adaptation: `pre-pump weak supervision`

## Outputs

- train h5: `file_training_data/train_NMM_v2_last.h5`
- val h5: `file_training_data/val_NMM_v2_last.h5`
- exp pre-pump h5: `file_training_data/exp_prepump_v2_last.h5`
- norm: `params/normalization_v2_last.json`
- synthetic baseline ckpt: `params/models/nmm_v2_last_base.pt`
- best-val ckpt: `params/models/nmm_v2_last.pt`
- best-exp ckpt: `params/models/nmm_v2_last_expbest.pt`
- best-balanced ckpt: `params/models/nmm_v2_last_balanced.pt`
- resume ckpt: `params/models/nmm_v2_last.resume.pt`

## Commands

1. Generate synthetic train/val:

```bash
./nmm_ml/bin/python train_gen_NMM.py \
  --train_n 100000 \
  --val_n 10000
```

2. Export pre-pump weak-supervision set:

```bash
./nmm_ml/bin/python inference.py \
  --mode export_prepump_h5 \
  --out_h5 file_training_data/exp_prepump_v2_last.h5 \
  --alpha 0.015
```

3. Stage 1, train synthetic baseline:

```bash
./nmm_ml/bin/python train.py \
  --train_h5 file_training_data/train_NMM_v2_last.h5 \
  --val_h5 file_training_data/val_NMM_v2_last.h5 \
  --norm_json params/normalization_v2_last.json \
  --out_ckpt params/models/nmm_v2_last_base.pt \
  --model_version v2
```

4. Stage 2, fine-tune with pre-pump weak supervision:

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

5. Validation inference:

```bash
./nmm_ml/bin/python inference.py \
  --ckpt params/models/nmm_v2_last.pt \
  --mode synth \
  --val_h5 file_training_data/val_NMM_v2_last.h5 \
  --out_npz results/v2_last/inference_v2_last_val.npz
```

6. Experimental inference:

```bash
./nmm_ml/bin/python inference.py \
  --ckpt params/models/nmm_v2_last_expbest.pt \
  --mode exp \
  --avg_bootstrap \
  --alpha 0.015 \
  --out_npz results/v2_last/inference_v2_last_exp.npz
```

## Baseline Check

- 7-label definition: [functions.py](/Users/jinc_air/NMM_ML/functions.py#L575)
- equilibrium computation: [functions.py](/Users/jinc_air/NMM_ML/functions.py#L596)

## Status

- [ ] synthetic train/val regenerated
- [ ] experimental pre-pump h5 exported
- [ ] synthetic baseline training finished
- [ ] weak-supervision fine-tune finished
- [ ] validation inference finished
- [ ] experimental inference finished
