# V6 Corrected Summary

## Purpose

`V6_fix` is the corrected rerun of `V6`, intended to match the successful `V3` training scale and low-noise regime while adding one extra geometric label to reduce mirror ambiguity in reconstruction.

This run fixes the experimental pre-pump training sample definition:
- use pre-pump export with fixed `alpha = 0.015`
- use all `200 x 7 = 1400` pre-pump samples
- use merged-dataset training with `repeat = 2`

## Fixed Configuration

### Model / Input
- model: `V2`
- scattering: `DPWA`
- signal target: `delta_sm`
- input grid: truncated `431` points

### Labels
- total labels: `8`
- first 7 labels:
  - `O-N`
  - `O-C5`
  - `N-C5`
  - `N-C2`
  - `N-C4`
  - `C5-C2`
  - `C5-C4`
- extra label:
  - `h_C5_signed`
  - signed distance from `C5` to the rigid ring plane defined from `C1/C2/C4/C6`

### Synthetic Data
- train synthetic samples: `60000`
- val synthetic samples: `10000`
- `equil_fraction = 0`
- `no_drift = true`
- low-noise profile reused from `V2/V3`:
  - `experiments/prepump_joint_train_v2/noise_profile_joint_train_v2.npz`

### Experimental Data
- source h5:
  - `file_training_data/exp_prepump_joint_alpha015_1400.h5`
- export mode:
  - `inference.py --mode export_prepump_h5 --alpha 0.015`
- total pre-pump samples:
  - `1400`
- merged training repeat:
  - `2x`
- merged experimental count:
  - `2800`

### Merged Training Set
- synthetic train:
  - `60000`
- experimental merged:
  - `2800`
- merged total:
  - `62800`

### Training
- epochs: `22`
- batch size: `112`
- learning rate: `1.5e-4`
- optimizer: `AdamW`
- scheduler: warmup + cosine decay
- save every epoch checkpoint: `true`

## Main Files

### Runbook / Scripts
- `experiments/prepump_joint_train_v6_fix/RUNBOOK.md`
- `experiments/prepump_joint_train_v6_fix/PLAN.md`
- `experiments/prepump_joint_train_v6_fix/eval_epoch_curve.py`

### Checkpoint
- final best-val checkpoint:
  - `params/models/nmm_prepump_joint_train_v6_fix.pt`

### Logs
- training:
  - `experiments/prepump_joint_train_v6_fix/logs/train.log`
- epoch sweep:
  - `experiments/prepump_joint_train_v6_fix/logs/eval_epoch_curve.log`
- epoch 13 inference:
  - `experiments/prepump_joint_train_v6_fix/logs/inference_epoch013_alpha015.log`
- epoch 22 inference:
  - `experiments/prepump_joint_train_v6_fix/logs/inference_epoch022_alpha015.log`
- epoch 13 relaxed reconstruction:
  - `experiments/prepump_joint_train_v6_fix/logs/reconstruct_epoch013_alpha015_relaxed.log`
- epoch 22 relaxed reconstruction:
  - `experiments/prepump_joint_train_v6_fix/logs/reconstruct_epoch022_alpha015_relaxed.log`

### Result Tables / Figures
- epoch summary curve:
  - `experiments/prepump_joint_train_v6_fix/epoch_vs_exp_quality.csv`
- epoch full time-series table:
  - `experiments/prepump_joint_train_v6_fix/epoch_vs_exp_quality_all_times.csv`
- epoch summary figure:
  - `experiments/prepump_joint_train_v6_fix/figures/epoch_vs_exp_quality.png`

### Reconstructions
- epoch 13:
  - `experiments/prepump_joint_train_v6_fix/reconstruction_epoch013_alpha015_relaxed/`
- epoch 22:
  - `experiments/prepump_joint_train_v6_fix/reconstruction_epoch022_alpha015_relaxed/`

## Validation Accuracy

From the final training log (`epoch 22`):
- `val_mae = 0.0794 A`
- `key_mae = 0.0250 A`

Per-dimension MAE:
- `O-N = 0.0193 A`
- `O-C5 = 0.0200 A`
- `N-C5 = 0.0357 A`
- `N-C2 = 0.0902 A`
- `N-C4 = 0.0905 A`
- `C5-C2 = 0.1604 A`
- `C5-C4 = 0.1598 A`
- `h_C5_signed = 0.0594 A`

## Epoch Sweep Conclusion

The corrected `V6` does not behave monotonically with epoch.

Two representative epochs are worth keeping:

### Epoch 13
- `prepump_dev_7 = 0.00230 A`
- `prepump_key_7 = 0.00318 A`
- `span(O-C5) = 0.5508 A`
- `span(N-C5) = 0.8743 A`
- relaxed reconstruction:
  - `45/45` success
  - `fallback = 6`

Interpretation:
- better dynamics amplitude
- still physically stable enough
- likely the best balance if amplitude is the priority

### Epoch 22
- `prepump_dev_7 = 0.00081 A`
- `prepump_key_7 = 0.00077 A`
- `span(O-C5) = 0.4374 A`
- `span(N-C5) = 0.6682 A`
- relaxed reconstruction:
  - `45/45` success
  - `fallback = 1`

Interpretation:
- best pre-pump agreement
- more conservative dynamics
- best-val / most stable end-state checkpoint

## Reconstruction Method

### Main Reconstruction
- `N3` is constrained to stay on the same side of the anchor plane as equilibrium
- `C5` prefers the same side as equilibrium
- reference `pl/ax` markers are aligned using only the four ring carbons:
  - `C1/C2/C4/C6`

### Relaxed Fallback
Used only when analytic reconstruction fails:
- allow small optimization of `O7`, `N3`, `C5`
- keep `C2/C4` fixed
- keep same-side priors for `N3/C5`
- save `fallback_mask` into `reconstruction.npz`

### Bootstrap Error on Reconstruction
No re-inference is needed; bootstrap uncertainty is computed directly from saved `y_all` in the detailed inference npz.

For `displacement_from_equilibrium.png`, the first panel now shows `±1σ` bootstrap bands.

#### Epoch 13 bootstrap displacement std
- `N3`: mean `0.0438 A`, max `0.0895 A`
- `C5`: mean `0.3896 A`, max `0.9148 A`

#### Epoch 22 bootstrap displacement std
- `N3`: mean `0.0469 A`, max `0.0842 A`
- `C5`: mean `0.3126 A`, max `0.9212 A`

## Practical Takeaway

- `V6_fix` confirms that adding `h_C5_signed` does not destroy dynamics if the training scale stays close to `V3`.
- The corrected experimental sample definition matters.
- `epoch 13` and `epoch 22` should both be retained:
  - `epoch 13` for stronger motion amplitude
  - `epoch 22` for tighter pre-pump consistency and cleaner reconstruction
