"""
推理脚本（两种模式）

模式 A：合成数据 sanity-check
- 生成 N 条“具有特征变化”的构型（例如 C2-N 键长单调增加等）
- 计算信号 -> 用模型回归 -> 与真值标签比较误差

模式 B：实验数据推理
- 读取 bootstrapping/results_240/ 下的 s0_*.mat 或 s0.mat
- 合并所有时间点/重复实验为一批一维信号
- 输出预测的 (O,N,C5) 距离矩阵（均值/方差）并保存

你最常修改的参数：
- 实验数据路径与 key：--exp_dir / --mat_key（默认 's0'）
- 选择使用哪些列（训练集限制条件类似的筛选）：--col_slice
- S 轴一致性：确保实验 s 轴与 config.S_GRID 对齐（若不一致，需要重采样/插值）
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import scipy.io as sio

import config as cfg
import functions as fn


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = fn.NMMRegressor1D(out_dim=cfg.LABEL_FLAT_DIM)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    norm = fn.load_normalization(Path(ckpt["norm_json"]))
    return model, norm


def predict_batch(model, norm, x: np.ndarray, device: torch.device) -> np.ndarray:
    # x: (N,S) float32
    x_n = fn.normalize_x(x, norm).astype(np.float32)
    xb = torch.from_numpy(x_n[:, None, :]).float().to(device)
    with torch.no_grad():
        y_hat_n = model(xb).cpu().numpy().astype(np.float32)
    y_hat = fn.denormalize_y(y_hat_n, norm)
    return y_hat


def synth_feature_set(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    生成一批“有特征”的合成数据，用于评估推理误差。
    这里提供一个简单策略：
    - 让 C5 球采样半径在范围内线性扫描，其它自由度随机
    你可以在这里扩展更多模式（旋转、夹角扫描等）。
    """
    rng = np.random.default_rng(seed)
    x_list, y_list = [], []

    # 在新的几何约束采样中，C5 半径从 GEN_C5_SPHERE_RADIUS_ANG 采样
    # 这里我们让 C5 球半径在 4.0-6.0 Å 范围内线性变化来生成特征样本
    r_c5_lo, r_c5_hi = 4.0, 6.0
    r_c5_scan = np.linspace(r_c5_lo, r_c5_hi, n).astype(np.float64)
    
    for k in range(n):
        dof = fn.sample_dof(rng)
        # 创建新的 DOF 对象，修改 C5 偏移向量的范数
        if np.linalg.norm(dof.c5_offset) > 1e-6:
            c5_offset_scaled = dof.c5_offset / np.linalg.norm(dof.c5_offset) * r_c5_scan[k]
        else:
            c5_offset_scaled = dof.c5_offset
        
        dof = fn.SampledDOF(
            n_center=dof.n_center,
            n_radius_t=dof.n_radius_t,
            n_angle_t=dof.n_angle_t,
            c5_offset=c5_offset_scaled,
        )
        
        try:
            backbone = fn.generate_backbone_coords_from_dof(dof)
            coords_all, _names, elems = fn.build_full_coords_with_locked_H(backbone)
            sig = fn.compute_1d_scattering_signal(coords_all, elems)
            sig = fn.add_noise(sig, rng)
            lab = fn.label_from_backbone(backbone)
            x_list.append(sig)
            y_list.append(lab)
        except Exception:
            continue

    return np.stack(x_list).astype(np.float32), np.stack(y_list).astype(np.float32)


def load_experiment_signals(exp_dir: Path, mat_key: str = "s0") -> np.ndarray:
    """
    读取实验 s0 矩阵并合并为 (N,S)。
    约定：mat 内变量 mat_key 的形状为 (S, T) 或 (T, S)。
    若 (T,S) 会自动转置。
    """
    mats = sorted(list(exp_dir.glob("s0*.mat")))
    if not mats:
        raise FileNotFoundError(f"no s0*.mat found under {exp_dir}")

    all_cols = []
    for p in mats:
        m = sio.loadmat(p)
        if mat_key not in m:
            raise KeyError(f"{p} missing key '{mat_key}'")
        a = np.array(m[mat_key], dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"{p} {mat_key} must be 2D, got {a.shape}")
        # force (S,T)
        if a.shape[0] != cfg.S_LEN and a.shape[1] == cfg.S_LEN:
            a = a.T
        if a.shape[0] != cfg.S_LEN:
            raise ValueError(f"{p} {mat_key} S dimension mismatch: expected {cfg.S_LEN}, got {a.shape}")
        all_cols.append(a.T)  # (T,S)

    x = np.concatenate(all_cols, axis=0)  # (N,S)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(cfg.PATHS.checkpoint_pt))
    ap.add_argument("--mode", type=str, choices=["synth", "exp"], default="synth")
    ap.add_argument("--out_npz", type=str, default=str(cfg.PATHS.params_dir / "inference_out.npz"))

    ap.add_argument("--synth_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=cfg.GEN_RANDOM_SEED + 7)

    ap.add_argument("--exp_dir", type=str, default=str(cfg.PATHS.root / "bootstrapping" / "results_240"))
    ap.add_argument("--mat_key", type=str, default="s0")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    model, norm = load_model(ckpt, device)

    if args.mode == "synth":
        x, y_true = synth_feature_set(args.synth_n, seed=args.seed)
        y_hat = predict_batch(model, norm, x, device)
        err = np.abs(y_hat - y_true)
        print(f"[inf:synth] mae per-dim: {err.mean(axis=0)}")
        print(f"[inf:synth] overall mae: {err.mean():.6f}")
        out = dict(y_hat=y_hat, y_true=y_true, x=x, mae_dim=err.mean(axis=0), mae=float(err.mean()))
    else:
        x = load_experiment_signals(Path(args.exp_dir), mat_key=args.mat_key)
        y_hat = predict_batch(model, norm, x, device)
        out = dict(y_hat=y_hat, x=x)
        print(f"[inf:exp] predicted {y_hat.shape[0]} samples")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out)
    print(f"[inf] saved: {out_npz}")


if __name__ == "__main__":
    main()

