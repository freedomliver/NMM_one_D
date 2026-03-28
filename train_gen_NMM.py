"""
生成 NMM 训练数据（.h5）并保存归一化参数（.json）

V2_last 默认配置：
- 固定 `delta_sm`
- `--no_drift`
- 固定 `trunc` (431 pts)

改进点：
- 实时 flush 输出，随时可见进度
- 分块写入 h5，避免大量数据堆在内存
- 打印内存使用，方便监控
- 固定生成差分信号 ΔsM = sM(structure) - sM(equilibrium)
  该模式与实验泵浦-探测数据约定一致（实验 s0 信号为差分信号）
- 若提供经验噪声包络，则按实验 pre-pump 统计的 sigma(s) 加噪

差分模式说明：
  - 地态平衡结构 → ΔsM = 0（无信号）
  - 激发态结构 → ΔsM ≠ 0（结构变化编码在差分信号中）
  - 推理时实验数据直接以 ΔsM 形式喂入，无需加回地态信号
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

import config as cfg
import functions as fn


def get_rss_mb() -> float:
    """获取当前进程 RSS（MB）"""
    import resource
    # macOS: ru_maxrss in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def log(msg: str):
    """实时输出，立即 flush"""
    print(msg, flush=True)


def generate_dataset_to_h5(
    h5_path: Path,
    n_samples: int,
    seed: int,
    chunk_size: int = 10000,
    max_tries_factor: int = 50,
    equil_fraction: float = 0.0,
    noise_profile_full: np.ndarray | None = None,
):
    """
    分块生成数据并直接写入 h5 文件，不在内存中保留全部数据。
    返回 (x_mean, x_sq_mean, y_mean, y_sq_mean, count) 用于计算归一化参数。
    """
    rng = np.random.default_rng(seed)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    s_len = cfg.S_LEN_TRUNC
    s_mask = (cfg.S_GRID >= cfg.S_TRUNC_MIN) & (cfg.S_GRID <= cfg.S_TRUNC_MAX)

    # V2_last 主线固定差分模式，预计算平衡态信号一次
    coords_eq, _, elems_eq = fn.build_full_coords_with_locked_H(cfg.NMM_BASE_COORDS_ANG)
    sM_ground = fn.compute_1d_scattering_signal(coords_eq, elems_eq)
    log(f"[gen] ΔsM mode: sM_ground computed (std in [1.5,9] = "
        f"{sM_ground[(cfg.S_GRID>=1.5)&(cfg.S_GRID<=9.)].std():.4f})")

    noise_std_lo, noise_std_hi = cfg.NOISE_GAUSS_STD_RANGE
    log(f"[gen] noise_gauss_std range: [{noise_std_lo}, {noise_std_hi}]")

    max_tries = int(n_samples * max_tries_factor)
    tries = 0
    total_collected = 0
    rejected = 0

    # 用于在线计算 mean/std（Welford 算法简化版：累加 sum 和 sum_sq）
    x_sum = np.zeros(s_len, dtype=np.float64)
    x_sq_sum = np.zeros(s_len, dtype=np.float64)
    y_sum = np.zeros(cfg.LABEL_FLAT_DIM, dtype=np.float64)
    y_sq_sum = np.zeros(cfg.LABEL_FLAT_DIM, dtype=np.float64)

    t0 = time.time()
    last_log_time = t0

    with h5py.File(h5_path, "w") as h5f:
        # 创建可扩展的 dataset（maxshape=None 表示无限制）
        chunk_s = min(chunk_size, max(1, n_samples))
        ds_x = h5f.create_dataset(
            "x", shape=(0, s_len), maxshape=(None, s_len),
            dtype=np.float32, chunks=(chunk_s, s_len),
            compression="gzip", compression_opts=4,
        )
        ds_y = h5f.create_dataset(
            "y", shape=(0, cfg.LABEL_FLAT_DIM), maxshape=(None, cfg.LABEL_FLAT_DIM),
            dtype=np.float32, chunks=(chunk_s, cfg.LABEL_FLAT_DIM),
            compression="gzip", compression_opts=4,
        )

        x_buf = []
        y_buf = []

        def flush_buf():
            """将缓冲区写入 h5"""
            nonlocal total_collected
            if not x_buf:
                return
            x_chunk = np.stack(x_buf, axis=0).astype(np.float32)
            y_chunk = np.stack(y_buf, axis=0).astype(np.float32)
            old_len = ds_x.shape[0]
            new_len = old_len + len(x_buf)
            ds_x.resize(new_len, axis=0)
            ds_y.resize(new_len, axis=0)
            ds_x[old_len:new_len] = x_chunk
            ds_y[old_len:new_len] = y_chunk
            total_collected = new_len
            x_buf.clear()
            y_buf.clear()

        while total_collected + len(x_buf) < n_samples and tries < max_tries:
            tries += 1
            try:
                # Near-equilibrium sampling: directly perturb ground-state backbone
                if equil_fraction > 0 and rng.random() < equil_fraction:
                    backbone = fn.generate_backbone_near_equil(rng)
                else:
                    dof = fn.sample_dof(rng)
                    backbone = fn.generate_backbone_coords_from_dof(dof)
                coords_all, _names, elems = fn.build_full_coords_with_locked_H(backbone)
                signal = fn.compute_1d_scattering_signal(coords_all, elems)
                signal = signal - sM_ground
                signal = fn.add_noise(signal, rng, noise_profile=noise_profile_full)
                label = fn.label_from_backbone(backbone)
            except Exception:
                rejected += 1
                continue

            signal = signal[s_mask]

            x_buf.append(signal)
            y_buf.append(label)

            # 累计统计
            x_sum += signal.astype(np.float64)
            x_sq_sum += (signal.astype(np.float64)) ** 2
            y_sum += label.astype(np.float64)
            y_sq_sum += (label.astype(np.float64)) ** 2

            # 缓冲区满了就写入 h5
            if len(x_buf) >= chunk_size:
                flush_buf()

                # 实时进度
                dt = time.time() - t0
                rate = total_collected / dt if dt > 0 else 0
                eta = (n_samples - total_collected) / rate if rate > 0 else 0
                accept_rate = total_collected / tries * 100
                log(
                    f"[gen] {total_collected:>8d}/{n_samples} "
                    f"({total_collected/n_samples*100:5.1f}%) | "
                    f"rejected={rejected} accept={accept_rate:.1f}% | "
                    f"{rate:.0f} samples/s | "
                    f"elapsed={dt:.0f}s ETA={eta:.0f}s | "
                    f"RSS={get_rss_mb():.0f}MB"
                )

            # 即使缓冲区没满，也定期输出进度（每 10 秒）
            now = time.time()
            if now - last_log_time > 10:
                dt = now - t0
                done = total_collected + len(x_buf)
                rate = done / dt if dt > 0 else 0
                eta = (n_samples - done) / rate if rate > 0 else 0
                log(
                    f"[gen] {done:>8d}/{n_samples} "
                    f"({done/n_samples*100:5.1f}%) | "
                    f"rate={rate:.0f}/s ETA={eta:.0f}s | "
                    f"RSS={get_rss_mb():.0f}MB (buf={len(x_buf)})"
                )
                last_log_time = now

        # 写入剩余缓冲
        flush_buf()

    dt = time.time() - t0
    log(f"[gen] DONE: {total_collected}/{n_samples} in {dt:.1f}s | RSS={get_rss_mb():.0f}MB")

    if total_collected < n_samples:
        raise RuntimeError(
            f"only collected {total_collected}/{n_samples}, "
            f"rejected={rejected}, increase ranges or max_tries_factor"
        )

    # 返回在线统计量用于归一化
    eps = 1e-8
    n = float(total_collected)
    x_mean = (x_sum / n).astype(np.float32)
    x_std = (np.sqrt(x_sq_sum / n - (x_sum / n) ** 2) + eps).astype(np.float32)
    y_mean = (y_sum / n).astype(np.float32)
    y_std = (np.sqrt(y_sq_sum / n - (y_sum / n) ** 2) + eps).astype(np.float32)

    return fn.Normalization(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=cfg.GEN_N_SAMPLES_TRAIN)
    ap.add_argument("--val_n", type=int, default=cfg.GEN_N_SAMPLES_VAL)
    ap.add_argument("--seed", type=int, default=cfg.GEN_RANDOM_SEED)
    ap.add_argument("--train_h5", type=str, default=str(cfg.PATHS.train_h5))
    ap.add_argument("--val_h5", type=str, default=str(cfg.PATHS.val_h5))
    ap.add_argument("--norm_json", type=str, default=str(cfg.PATHS.norm_json))
    ap.add_argument("--chunk_size", type=int, default=10000)
    ap.add_argument("--noise_profile_npz", type=str, default=None,
                    help="实验 pre-pump 经验噪声包络 npz；若提供则使用 sigma_full")
    ap.add_argument("--equil_fraction", type=float, default=cfg.DEFAULT_EQUIL_FRACTION,
                    help="近平衡态样本比例 [0,1]，推荐 0.25-0.35 平衡训练分布")
    ap.add_argument("--no_drift", dest="no_drift", action="store_true",
                    help="关闭低频漂移噪声（V2_last 默认开启）")
    ap.add_argument("--with_drift", dest="no_drift", action="store_false",
                    help="Legacy: 保留低频漂移噪声")
    ap.set_defaults(no_drift=cfg.DEFAULT_NO_DRIFT)
    args = ap.parse_args()

    train_h5 = Path(args.train_h5)
    val_h5 = Path(args.val_h5)
    norm_json = Path(args.norm_json)
    noise_profile_full = None
    if args.noise_profile_npz:
        prof = np.load(args.noise_profile_npz)
        noise_profile_full = np.asarray(prof["sigma_full"], dtype=np.float32)
        if noise_profile_full.shape != cfg.S_GRID.shape:
            raise ValueError(
                f"sigma_full shape mismatch: got {noise_profile_full.shape}, expected {cfg.S_GRID.shape}"
            )
        log(
            f"[gen] using empirical noise profile: {args.noise_profile_npz} "
            f"(mean={noise_profile_full.mean():.5f}, max={noise_profile_full.max():.5f})"
        )

    log(f"[gen] *** DELTA_SM MODE: training on ΔsM = sM(struct) - sM(equil) ***")
    if args.equil_fraction > 0:
        log(f"[gen] equil_fraction={args.equil_fraction:.2f} (near-equilibrium sampling)")

    if args.no_drift:
        log(f"[gen] *** NO DRIFT: disabling low-frequency drift noise ***")
        cfg.NOISE_DRIFT_ENABLE = False
    log(f"[gen] *** TRUNCATE S: s=[{cfg.S_TRUNC_MIN}, {cfg.S_TRUNC_MAX}], "
        f"{cfg.S_LEN_TRUNC} pts (was {cfg.S_LEN}) ***")

    log(f"[gen] === Generating train set: {args.train_n} samples ===")
    norm = generate_dataset_to_h5(
        train_h5, args.train_n, seed=args.seed, chunk_size=args.chunk_size,
        equil_fraction=args.equil_fraction, noise_profile_full=noise_profile_full,
    )

    # 保存归一化参数（基于训练集统计）
    fn.save_normalization(norm, norm_json)
    log(f"[gen] saved normalization: {norm_json}")

    log(f"[gen] === Generating val set: {args.val_n} samples ===")
    generate_dataset_to_h5(
        val_h5, args.val_n, seed=args.seed + 1, chunk_size=args.chunk_size,
        equil_fraction=args.equil_fraction, noise_profile_full=noise_profile_full,
    )

    log(f"[gen] ALL DONE.")
    log(f"[gen]   train: {train_h5}")
    log(f"[gen]   val:   {val_h5}")
    log(f"[gen]   norm:  {norm_json}")


if __name__ == "__main__":
    main()
