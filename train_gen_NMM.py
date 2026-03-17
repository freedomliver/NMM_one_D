"""
生成 NMM 训练数据（.h5）并保存归一化参数（.json）

对照 manual.md 的核心要求：
- 输入：一维信号 int(s) / delta sM(s)，长度 681（见 config.S_GRID）
- 输出标签：O7, N3, C5 的 3x3 全距离矩阵（flatten 9 维）
- H 原子参与散射干涉项，但 H 坐标锁死在对应 C 的相对位置（见 config.H_LOCKED_OFFSETS_ANG）
- 自由度只让 N3/C5 动，O7 与平面 C 锁死（见 config.GEN_* / config.FIXED_ATOMS）
- 噪声可叠加多类型（见 config.NOISE_*）

你最常修改的参数：
- 样本量：config.GEN_N_SAMPLES_TRAIN / config.GEN_N_SAMPLES_VAL
- S 长度：config.S_LEN / config.S_GRID
- 自由度范围：config.GEN_R_* / config.GEN_C5_OOP_DEG_RANGE
- 训练集筛选：config.FILTERS
- H 的相对固定偏移：config.H_LOCKED_OFFSETS_ANG

运行示例：
  python train_gen_NMM.py --train_n 20000 --val_n 2000
  python train_gen_NMM.py --train_n 1000000 --val_n 20000
"""

from __future__ import annotations

import argparse
import time
import numpy as np

import config as cfg
import functions as fn


def generate_dataset(n_samples: int, seed: int, max_tries_factor: int = 50):
    rng = np.random.default_rng(seed)
    x_list = []
    y_list = []

    tries = 0
    max_tries = int(n_samples * max_tries_factor)

    t0 = time.time()
    while len(x_list) < n_samples and tries < max_tries:
        tries += 1
        dof = fn.sample_dof(rng)
        try:
            backbone = fn.generate_backbone_coords_from_dof(dof)
            coords_all, _names, elems = fn.build_full_coords_with_locked_H(backbone)
            signal = fn.compute_1d_scattering_signal(coords_all, elems)
            signal = fn.add_noise(signal, rng)
            label = fn.label_from_backbone(backbone)
        except Exception:
            # 构型无解 / 病态 / 不满足筛选，直接丢弃重采样
            continue

        x_list.append(signal)
        y_list.append(label)

        if len(x_list) % 2000 == 0:
            dt = time.time() - t0
            print(f"[gen] {len(x_list)}/{n_samples} collected | tries={tries} | elapsed={dt:.1f}s")

    if len(x_list) < n_samples:
        raise RuntimeError(f"only collected {len(x_list)}/{n_samples}, increase ranges/filters or max_tries_factor")

    x = np.stack(x_list, axis=0).astype(np.float32)  # (N,S)
    y = np.stack(y_list, axis=0).astype(np.float32)  # (N,9)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=cfg.GEN_N_SAMPLES_TRAIN)
    ap.add_argument("--val_n", type=int, default=cfg.GEN_N_SAMPLES_VAL)
    ap.add_argument("--seed", type=int, default=cfg.GEN_RANDOM_SEED)
    ap.add_argument("--train_h5", type=str, default=str(cfg.PATHS.train_h5))
    ap.add_argument("--val_h5", type=str, default=str(cfg.PATHS.val_h5))
    ap.add_argument("--norm_json", type=str, default=str(cfg.PATHS.norm_json))
    args = ap.parse_args()

    from pathlib import Path
    train_h5 = Path(args.train_h5)
    val_h5 = Path(args.val_h5)
    norm_json = Path(args.norm_json)

    print("[gen] generating train set...")
    x_tr, y_tr = generate_dataset(args.train_n, seed=args.seed)
    print("[gen] generating val set...")
    x_va, y_va = generate_dataset(args.val_n, seed=args.seed + 1)

    print("[gen] computing normalization from train set...")
    norm = fn.compute_normalization(x_tr, y_tr)
    fn.save_normalization(norm, norm_json)

    # 保存原始（未归一化）到 h5，训练时再读 norm 做归一化（便于复用同一 norm）
    fn.write_h5_dataset(train_h5, x_tr, y_tr)
    fn.write_h5_dataset(val_h5, x_va, y_va)

    print(f"[gen] saved train: {train_h5}")
    print(f"[gen] saved val:   {val_h5}")
    print(f"[gen] saved norm:  {norm_json}")


if __name__ == "__main__":
    main()

