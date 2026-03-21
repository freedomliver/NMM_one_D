"""
推理脚本（三种模式）

模式 A：合成数据 sanity-check（synth）
- 生成 N 条"具有特征变化"的构型
- 计算信号 -> 用模型回归 -> 与真值标签比较误差

模式 B：实验数据推理（exp）
- 读取 bootstrapping/results_240/ 下的 s0_*.mat (200 个文件, 每个 45×637)
- 汇聚 200 次 bootstrap → 每个时间点的均值 / 标准差
- 自动选择信号约定（--signal_mode）：
    abs   : 旧绝对 sM 约定（当前 V2 模型，有域偏移，实验效果差）
    delta : ΔsM 约定（s0 本身即 ΔsM，推荐）

模式 C：中间诊断（avg_only）
- 仅计算 200 个 bootstrap 文件的时间平均信号，保存到 npz，不做推理

实验数据 s 轴（来自参考代码 train50_gen_NMM.py）：
  sPixel = 0.0245 Å⁻¹
  s[k] = (k + 0.5) * sPixel, k = 0…636
  → 0.01225 ~ 15.594 Å⁻¹，共 637 点

实验数据说明：
  - s0 本身为差分信号 ΔsM = (pump_on - pump_off)，pre-pump 均值约为 0
  - 正确标定：alpha ≈ 理论 ΔsM 幅度 / 实验 ΔsM 幅度（physics-based）
    推荐 --alpha 0.03（基于后泵浦时刻幅度与理论训练信号匹配）
  - 错误的做法：把 s0_ref_std 当 sM_ground_std 来计算 alpha（旧版本）

实验数据预处理步骤（delta 模式）：
  1. NaN 填充（线性插值）
  2. 减去 pre-pump 参考信号 s0_ref（t00-t<prepump_end> 均值）—— 去除残余基线
  3. 乘以 alpha（从实验单位转到 Å⁻¹）
  4. 二阶多项式 baseline 扣除（s > baseline_start = 5.0 Å⁻¹）
  5. 范围掩蔽：s < s_low_mask（默认 1.5）和 s > s_high_mask（默认 9.0）置零
  6. 插值到模型 681 点网格

常用参数：
  --ckpt            模型权重（自动优先 nmm_v2_dpwa100k_eq.pt > nmm_v2_dpwa50k.pt）
  --mode            synth / exp / avg_only
  --signal_mode     abs（旧绝对sM模型）/ delta（ΔsM 新模型，推荐）
  --alpha           实验信号缩放因子（默认 0.03，基于物理标定）
  --s_low_mask      遮蔽低 s 阈值（Å⁻¹，默认 1.5）
  --s_high_mask     遮蔽高 s 阈值（Å⁻¹，默认 9.0）
  --baseline_start  二阶多项式基线校正起始（Å⁻¹，默认 5.0；设 0 关闭）
  --prepump_end     pre-pump 时间点上界，用于计算参考信号（默认 9，即 t00-t08）
  --out_npz         输出文件路径
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import scipy.io as sio
from scipy.interpolate import interp1d

import config as cfg
import functions as fn


# ============================================================
# 实验数据 s 轴（637 点）
# ============================================================
EXP_S_PIXEL = 0.0245
EXP_S_LEN   = 637
EXP_S_GRID  = (np.arange(EXP_S_LEN) + 0.5) * EXP_S_PIXEL   # (637,)


# ============================================================
# 模型加载（自动识别 V1 / V2）
# ============================================================

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = list(ckpt["model_state"].keys())
    # 从 checkpoint 推断 out_dim（兼容旧 9 维和新 3/7 维模型）
    head_weight_key = [k for k in keys if k.endswith("weight") and "head" in k]
    out_dim = ckpt["model_state"][head_weight_key[-1]].shape[0] if head_weight_key else cfg.LABEL_FLAT_DIM
    if any("stem.b3" in k or "stem.b7" in k or "body." in k for k in keys):
        model = fn.NMMRegressorV2(out_dim=out_dim)
        ver = "V2"
    else:
        model = fn.NMMRegressor1D(out_dim=out_dim)
        ver = "V1"
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    norm = fn.load_normalization(Path(ckpt["norm_json"]))
    print(f"[load_model] {ver} loaded from {ckpt_path} (out_dim={out_dim})")
    return model, norm, out_dim


# ============================================================
# 推理
# ============================================================

def predict_batch(model, norm, x: np.ndarray, device: torch.device) -> np.ndarray:
    x_n = fn.normalize_x(x, norm).astype(np.float32)
    xb  = torch.from_numpy(x_n[:, None, :]).float().to(device)
    with torch.no_grad():
        y_hat_n = model(xb).cpu().numpy().astype(np.float32)
    return fn.denormalize_y(y_hat_n, norm)


# ============================================================
# 实验数据预处理（通用部分）
# ============================================================

def _fill_nan(sig: np.ndarray) -> np.ndarray:
    """线性插值填充 NaN，边缘置零"""
    sig = sig.copy().astype(float)
    nan_m = np.isnan(sig)
    if nan_m.any():
        vidx = np.where(~nan_m)[0]
        if len(vidx) > 1:
            f = interp1d(vidx, sig[vidx], bounds_error=False, fill_value=0.)
            sig[nan_m] = f(np.where(nan_m)[0])
        else:
            sig[nan_m] = 0.
    return sig


def _baseline_correct(sig: np.ndarray, s_exp: np.ndarray, bl_start: float) -> np.ndarray:
    """s > bl_start 区间用二阶多项式扣除 baseline"""
    if bl_start <= 0:
        return sig
    mask_bl = s_exp > bl_start
    if mask_bl.sum() > 5:
        sig = sig.copy()
        c = np.polyfit(s_exp[mask_bl], sig[mask_bl], 2)
        sig[mask_bl] -= np.polyval(c, s_exp[mask_bl])
    return sig


def preprocess_single(
    sig_raw: np.ndarray,           # (637,)
    s_exp: np.ndarray = EXP_S_GRID,
    s_model: np.ndarray = cfg.S_GRID,
    s_low: float = 1.5,
    s_high: float = 9.0,
    bl_start: float = 5.0,
) -> np.ndarray:
    """
    通用预处理：NaN 填充 → baseline 校正 → 掩蔽 → 插值到 681 点。
    返回 (681,) float32，掩蔽区间保持原始信号值（仅清零 ΔsM 贡献）。
    注意：不做幅度缩放，调用方负责缩放。
    """
    sig = _fill_nan(sig_raw)
    sig = _baseline_correct(sig, s_exp, bl_start)
    sig[s_exp < s_low]  = 0.
    sig[s_exp > s_high] = 0.
    f = interp1d(s_exp, sig, bounds_error=False, fill_value=0.)
    return f(s_model).astype(np.float32)


def batch_preprocess(
    x_raw: np.ndarray,     # (N, 637)
    s_low: float = 1.5,
    s_high: float = 9.0,
    bl_start: float = 5.0,
) -> np.ndarray:           # (N, 681)
    n = x_raw.shape[0]
    out = np.zeros((n, cfg.S_LEN), dtype=np.float32)
    for i in range(n):
        out[i] = preprocess_single(x_raw[i], s_low=s_low, s_high=s_high, bl_start=bl_start)
    return out


# ============================================================
# 实验数据加载与 bootstrap 汇聚
# ============================================================

def load_all_bootstrap(exp_dir: Path, mat_key: str = "s0") -> np.ndarray:
    """
    加载全部 s0_*.mat 并 NaN 填充。
    返回 (B, T, 637) float32。
    """
    mats = sorted(exp_dir.glob("s0*.mat"),
                  key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0)
    if not mats:
        raise FileNotFoundError(f"no s0*.mat in {exp_dir}")

    raw_list = []
    for p in mats:
        m = sio.loadmat(p)
        if mat_key not in m:
            raise KeyError(f"{p} missing '{mat_key}'")
        a = np.array(m[mat_key], dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"{p} must be 2D, got {a.shape}")
        if a.shape[0] == EXP_S_LEN and a.shape[1] != EXP_S_LEN:
            a = a.T
        # NaN fill
        for t in range(a.shape[0]):
            a[t] = _fill_nan(a[t])
        raw_list.append(a)   # (T, S)

    raw = np.stack(raw_list, axis=0).astype(np.float32)  # (B, T, S)
    print(f"[load] {raw.shape[0]} bootstrap × {raw.shape[1]} time pts × {raw.shape[2]} s-pts")
    return raw


# ============================================================
# Delta 模式校准：计算 pre-pump 参考和 alpha 缩放因子
# ============================================================

def compute_delta_calibration(raw: np.ndarray, prepump_end: int = 9,
                               alpha_override: float | None = None):
    """
    raw: (B, T, 637)

    s0 本身是 ΔsM 差分信号（pump_on - pump_off），pre-pump 均值约为 0。
    正确 alpha 由物理估算：理论 ΔsM std ≈ 0.47 Å⁻¹，除以实验峰值时刻 delta std。
    推荐通过 --alpha 0.03 直接指定；也可由 auto 模式从峰值时刻自动估算。

    返回:
      s0_ref: (637,) pre-pump 参考基线（减掉后去除残余偏置）
      alpha:  标量缩放因子
      sM_ground: (681,) 理论平衡态 sM（供参考）
    """
    coords_eq, _, elems_eq = fn.build_full_coords_with_locked_H(cfg.NMM_BASE_COORDS_ANG)
    sM_ground = fn.compute_1d_scattering_signal(coords_eq, elems_eq)  # (681,)

    pre_pump = raw[:, :prepump_end, :].mean(axis=(0, 1))  # (637,) residual baseline
    valid = (EXP_S_GRID >= 1.5) & (EXP_S_GRID <= 9.0)

    if alpha_override is not None:
        alpha = alpha_override
        print(f"[calib] alpha override={alpha:.5f}  pre_pump_residual_std={pre_pump[valid].std():.4f}")
    else:
        # Auto: match peak post-pump signal amplitude to training ΔsM std ≈ 0.47 Å⁻¹
        raw_mean = raw.mean(axis=0)   # (T, 637)
        post_pump_stds = np.array([
            (raw_mean[t] - pre_pump)[valid].std() for t in range(raw.shape[1])
        ])
        peak_t = int(np.argmax(post_pump_stds))
        peak_std_exp = post_pump_stds[peak_t]
        alpha = 0.47 / (peak_std_exp + 1e-8)
        print(f"[calib] auto alpha: peak_t={peak_t}  peak_std_exp={peak_std_exp:.4f}  alpha={alpha:.5f}")

    return pre_pump, alpha, sM_ground


def preprocess_delta(
    sig_preprocessed: np.ndarray,   # (681,) already baseline-corrected + masked
    s0_ref_model: np.ndarray,        # (681,) pre-pump reference on model grid
    alpha: float,
) -> np.ndarray:
    """
    ΔsM 最终信号 = (sig - s0_ref_model) * alpha。
    Masked regions (= 0 in sig_preprocessed) also give 0 - s0_ref * alpha = noise;
    we zero them explicitly since the training model is trained on ΔsM with zero baseline.
    """
    delta = (sig_preprocessed - s0_ref_model) * alpha
    # Keep only valid s range (zero outside to match training ΔsM convention)
    delta[(cfg.S_GRID < 1.5) | (cfg.S_GRID > 9.0)] = 0.
    return delta.astype(np.float32)


# ============================================================
# 合成数据 sanity-check
# ============================================================

def synth_feature_set(n: int, seed: int, signal_mode: str = "abs") -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_list, y_list = [], []
    r_c5_scan = np.linspace(4.0, 6.0, n).astype(np.float64)
    # 平衡态信号（delta 模式用）
    sM_ground = None
    if signal_mode == "delta":
        coords_eq, _, elems_eq = fn.build_full_coords_with_locked_H(cfg.NMM_BASE_COORDS_ANG)
        sM_ground = fn.compute_1d_scattering_signal(coords_eq, elems_eq)

    for k in range(n):
        dof = fn.sample_dof(rng)
        if np.linalg.norm(dof.c5_offset) > 1e-6:
            c5_scaled = dof.c5_offset / np.linalg.norm(dof.c5_offset) * r_c5_scan[k]
        else:
            c5_scaled = dof.c5_offset
        dof = fn.SampledDOF(n_center=dof.n_center, n_radius_t=dof.n_radius_t,
                            n_angle_t=dof.n_angle_t, c5_offset=c5_scaled)
        try:
            backbone = fn.generate_backbone_coords_from_dof(dof)
            coords_all, _names, elems = fn.build_full_coords_with_locked_H(backbone)
            sig = fn.compute_1d_scattering_signal(coords_all, elems)
            if signal_mode == "delta" and sM_ground is not None:
                sig = sig - sM_ground
            sig = fn.add_noise(sig, rng)
            lab = fn.label_from_backbone(backbone)
            x_list.append(sig)
            y_list.append(lab)
        except Exception:
            continue
    return np.stack(x_list).astype(np.float32), np.stack(y_list).astype(np.float32)


# ============================================================
# 主函数
# ============================================================

def main():
    # 默认 ckpt 优先级
    default_ckpt = str(cfg.PATHS.checkpoint_pt)
    for candidate in ["nmm_v2_7dim_80k.pt", "nmm_v2_3dim_80k.pt", "nmm_v2_dpwa100k_eq.pt", "nmm_v2_dpwa50k.pt", "nmm_v2.pt", "nmm_regressor.pt"]:
        cpath = cfg.PATHS.models_dir / candidate
        if cpath.exists():
            default_ckpt = str(cpath)
            break

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           type=str, default=default_ckpt)
    ap.add_argument("--mode",           type=str, choices=["synth", "exp", "avg_only"],
                    default="synth")
    ap.add_argument("--signal_mode",    type=str, choices=["abs", "delta"], default="delta",
                    help="abs=绝对sM(旧), delta=ΔsM(新, 推荐)")
    ap.add_argument("--out_npz",        type=str,
                    default=str(cfg.PATHS.params_dir / "inference_out.npz"))
    ap.add_argument("--synth_n",        type=int, default=10000)
    ap.add_argument("--seed",           type=int, default=cfg.GEN_RANDOM_SEED + 7)
    ap.add_argument("--val_h5",         type=str, default=None,
                    help="直接从 h5 验证集推理（跳过 synth_feature_set，与训练完全一致）")
    ap.add_argument("--exp_dir",        type=str,
                    default=str(cfg.PATHS.root / "bootstrapping" / "results_240"))
    ap.add_argument("--mat_key",        type=str, default="s0")
    ap.add_argument("--s_low_mask",     type=float, default=1.5)
    ap.add_argument("--s_high_mask",    type=float, default=9.0)
    ap.add_argument("--baseline_start", type=float, default=5.0)
    ap.add_argument("--prepump_end",    type=int, default=9,
                    help="t=0..prepump_end-1 作为 pre-pump 参考（默认 9，即 t00-t08）")
    ap.add_argument("--alpha",          type=float, default=None,
                    help="实验信号缩放因子（默认 None = 自动估算；推荐 0.03 物理标定）")
    ap.add_argument("--avg_bootstrap",  action="store_true",
                    help="先对200个bootstrap取均值再推理（降噪，推荐用于实验数据）")
    args = ap.parse_args()

    print(f"[inf] signal_mode={args.signal_mode}  mode={args.mode}")
    print(f"[inf] preprocessing: s_low={args.s_low_mask} s_high={args.s_high_mask} "
          f"bl_start={args.baseline_start}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if args.mode == "avg_only":
        # 仅计算均值，不做推理
        raw = load_all_bootstrap(Path(args.exp_dir), args.mat_key)
        s0_mean = np.nanmean(raw, axis=0)
        s0_std  = np.nanstd(raw, axis=0)
        np.savez(Path(args.out_npz), s0_mean=s0_mean, s0_std=s0_std,
                 s_exp=EXP_S_GRID)
        print(f"[inf] saved avg: {args.out_npz}")
        return

    ckpt = Path(args.ckpt)
    model, norm, out_dim = load_model(ckpt, device)
    # 根据模型实际输出维度选择对应的标签名称
    label_names = cfg.LABEL_PAIR_NAMES[:out_dim]

    if args.mode == "synth":
        if args.val_h5:
            # 直接从 h5 验证集推理，与训练完全一致
            x_raw, y_true = fn.read_h5_dataset(Path(args.val_h5))
            n = min(args.synth_n, len(x_raw))
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(x_raw), n, replace=False)
            x_raw, y_true = x_raw[idx], y_true[idx]
            # 反归一化的 x 不需要，直接用原始 x + norm 推理
            x_n = fn.normalize_x(x_raw, norm).astype(np.float32)
            xb  = torch.from_numpy(x_n[:, None, :]).float().to(device)
            with torch.no_grad():
                y_hat_n = model(xb).cpu().numpy().astype(np.float32)
            y_hat = fn.denormalize_y(y_hat_n, norm)
            x = x_raw
            print(f"[inf:synth] using val_h5={args.val_h5}, n={n}")
        else:
            x, y_true = synth_feature_set(args.synth_n, seed=args.seed,
                                           signal_mode=args.signal_mode)
            y_hat = predict_batch(model, norm, x, device)
        err   = np.abs(y_hat - y_true)
        per_dim = err.mean(axis=0)
        for i, name in enumerate(label_names):
            print(f"[inf:synth] {name} mae: {per_dim[i]:.6f} Å")
        print(f"[inf:synth] overall mae: {err.mean():.6f} Å")
        out = dict(y_hat=y_hat, y_true=y_true, x=x,
                   mae_dim=per_dim, mae=float(err.mean()))

    else:  # exp
        exp_dir = Path(args.exp_dir)
        raw = load_all_bootstrap(exp_dir, args.mat_key)  # (B, T, 637)
        n_b, n_t, n_s = raw.shape

        if args.avg_bootstrap:
            # ── Average-first mode ──────────────────────────────────────────
            # Average 200 bootstrap samples → noise reduced by sqrt(B)
            # Predict from mean; bootstrap std gives input uncertainty estimate.
            print(f"[inf:exp] avg_bootstrap: averaging {n_b} bootstraps first...")
            raw_mean = raw.mean(axis=0, keepdims=True)   # (1, T, 637)
            raw_std  = raw.std(axis=0)                    # (T, 637) for saving

            if args.signal_mode == "delta":
                pre_pump_637, alpha, sM_ground = compute_delta_calibration(
                    raw_mean, args.prepump_end,
                    alpha_override=args.alpha)           # use mean for calib
                s0_ref_preprocessed = preprocess_single(
                    pre_pump_637,
                    s_low=args.s_low_mask,
                    s_high=args.s_high_mask,
                    bl_start=args.baseline_start,
                )
                preprocessed_mean = batch_preprocess(
                    raw_mean[0],
                    s_low=args.s_low_mask,
                    s_high=args.s_high_mask,
                    bl_start=args.baseline_start,
                )   # (T, 681)
                final_mean = np.zeros_like(preprocessed_mean)
                for t in range(n_t):
                    final_mean[t] = preprocess_delta(
                        preprocessed_mean[t], s0_ref_preprocessed, alpha)
                print(f"[inf:exp] avg_bootstrap calib done. "
                      f"Mean delta std (t00): "
                      f"{final_mean[0][(cfg.S_GRID>=1.5)&(cfg.S_GRID<=9.)].std():.4f}")
            else:
                preprocessed_mean = batch_preprocess(
                    raw_mean[0],
                    s_low=args.s_low_mask,
                    s_high=args.s_high_mask,
                    bl_start=args.baseline_start,
                )
                final_mean = preprocessed_mean
                sM_ground = None
                alpha = None

            # Single prediction from mean signal
            y_mean = predict_batch(model, norm, final_mean, device)  # (T, D)
            y_std  = np.zeros_like(y_mean)   # no per-prediction bootstrap std

            print(f"\n[inf:exp] Key distances (avg_bootstrap) over {n_t} time points:")
            print(f"{'t':>5}  " + "  ".join(f"{n+' (Å)':>12}" for n in label_names))
            for t in range(n_t):
                vals = "  ".join(f"{y_mean[t,i]:8.3f}" for i in range(out_dim))
                print(f"  t{t:02d}:  {vals}")

            out = dict(
                y_mean=y_mean,
                y_std=y_std,
                final=final_mean,
                preprocessed=preprocessed_mean,
                raw_mean=raw_mean[0],
                raw_std=raw_std,
                s_exp=EXP_S_GRID,
                s_model=cfg.S_GRID,
                signal_mode=np.array(args.signal_mode, dtype='U10'),
                preprocess_params=np.array([args.s_low_mask, args.s_high_mask,
                                            args.baseline_start, args.prepump_end]),
            )
            if alpha is not None:
                out["alpha"] = np.array(alpha)
            print(f"\n[inf:exp] mean key distances across all time pts: "
                  f"{y_mean.mean(axis=0)}")

        else:
            # ── Per-bootstrap mode (original) ────────────────────────────────
            # Step 1: 通用预处理 (B, T, 637) → (B, T, 681)
            print(f"[inf:exp] preprocessing {n_b}×{n_t} signals...")
            preprocessed = np.zeros((n_b, n_t, cfg.S_LEN), dtype=np.float32)
            for b in range(n_b):
                preprocessed[b] = batch_preprocess(
                    raw[b],
                    s_low=args.s_low_mask,
                    s_high=args.s_high_mask,
                    bl_start=args.baseline_start,
                )
                if (b + 1) % 50 == 0:
                    print(f"  {b+1}/{n_b}")

            if args.signal_mode == "delta":
                # Step 2: Delta 校准
                pre_pump_637, alpha, sM_ground = compute_delta_calibration(
                    raw, args.prepump_end, alpha_override=args.alpha)
                # 将 s0_ref 也预处理到 681 点（baseline + mask + interp）
                s0_ref_preprocessed = preprocess_single(
                    pre_pump_637,
                    s_low=args.s_low_mask,
                    s_high=args.s_high_mask,
                    bl_start=args.baseline_start,
                )   # (681,)

                # Step 3: 每个信号做 delta = (sig - s0_ref) * alpha
                final = np.zeros_like(preprocessed)
                for b in range(n_b):
                    for t in range(n_t):
                        final[b, t] = preprocess_delta(
                            preprocessed[b, t], s0_ref_preprocessed, alpha)
                print(f"[inf:exp] delta calibration done. "
                      f"Sample norm stats (t00): "
                      f"std={final[0,0][(cfg.S_GRID>=1.5)&(cfg.S_GRID<=9.)].std():.4f}")
            else:
                final = preprocessed
                sM_ground = None
                alpha = None

            # Step 4: 推理
            print(f"[inf:exp] running inference (signal_mode={args.signal_mode})...")
            y_all = np.zeros((n_b, n_t, out_dim), dtype=np.float32)
            for b in range(n_b):
                y_all[b] = predict_batch(model, norm, final[b], device)

            y_mean = y_all.mean(axis=0)
            y_std  = y_all.std(axis=0)

            print(f"\n[inf:exp] Key distances over {n_t} time points (mean ± bootstrap_std):")
            print(f"{'t':>5}  " + "  ".join(f"{n+' (Å)':>16}" for n in label_names))
            for t in range(n_t):
                vals = "  ".join(f"{y_mean[t,i]:6.3f}±{y_std[t,i]:.3f}" for i in range(out_dim))
                print(f"  t{t:02d}:  {vals}")

            out = dict(
                y_all=y_all,
                y_mean=y_mean,
                y_std=y_std,
                final=final,
                preprocessed=preprocessed,
                raw=raw,
                s_exp=EXP_S_GRID,
                s_model=cfg.S_GRID,
                signal_mode=np.array(args.signal_mode, dtype='U10'),
                preprocess_params=np.array([args.s_low_mask, args.s_high_mask,
                                            args.baseline_start, args.prepump_end]),
            )
            if alpha is not None:
                out["alpha"] = np.array(alpha)
            print(f"\n[inf:exp] mean key distances across all time pts: "
                  f"{y_mean.mean(axis=0)}")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out)
    print(f"[inf] saved: {out_npz}")


# ============================================================
# Alpha sensitivity sweep
# ============================================================

def alpha_sweep(args):
    """
    Sweep alpha values, select the one that minimizes pre-pump distance
    deviation from equilibrium. Outputs diagnostic plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    model, norm, out_dim = load_model(ckpt, device)
    label_names = cfg.LABEL_PAIR_NAMES[:out_dim]

    # Equilibrium distances
    eq_label = fn.label_from_backbone(cfg.NMM_BASE_COORDS_ANG)

    # Load and preprocess bootstrap data
    exp_dir = Path(args.exp_dir)
    raw = load_all_bootstrap(exp_dir, args.mat_key)
    n_b, n_t, n_s = raw.shape
    raw_mean = raw.mean(axis=0, keepdims=True)

    alphas = [float(a) for a in args.alpha_values.split(",")]
    print(f"[alpha_sweep] testing alphas: {alphas}")

    results = {}
    for alpha in alphas:
        pre_pump_637, _, sM_ground = compute_delta_calibration(
            raw_mean, args.prepump_end, alpha_override=alpha)
        s0_ref_preprocessed = preprocess_single(
            pre_pump_637, s_low=args.s_low_mask, s_high=args.s_high_mask,
            bl_start=args.baseline_start)
        preprocessed_mean = batch_preprocess(
            raw_mean[0], s_low=args.s_low_mask, s_high=args.s_high_mask,
            bl_start=args.baseline_start)
        final_mean = np.zeros_like(preprocessed_mean)
        for t in range(n_t):
            final_mean[t] = preprocess_delta(preprocessed_mean[t], s0_ref_preprocessed, alpha)

        y_pred = predict_batch(model, norm, final_mean, device)
        pre_pump_pred = y_pred[:args.prepump_end].mean(axis=0)
        deviation_pct = np.abs(pre_pump_pred - eq_label[:out_dim]) / eq_label[:out_dim] * 100
        mean_dev = deviation_pct.mean()
        results[alpha] = {
            "pre_pump_pred": pre_pump_pred,
            "deviation_pct": deviation_pct,
            "mean_dev": mean_dev,
            "y_pred": y_pred,
        }
        print(f"  alpha={alpha:.4f}: mean_dev={mean_dev:.2f}%  "
              + "  ".join(f"{label_names[i]}={deviation_pct[i]:.1f}%" for i in range(out_dim)))

    # Find best alpha
    best_alpha = min(results, key=lambda a: results[a]["mean_dev"])
    print(f"\n[alpha_sweep] BEST alpha = {best_alpha:.4f} "
          f"(mean_dev = {results[best_alpha]['mean_dev']:.2f}%)")

    # Diagnostic plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Alpha Sensitivity Sweep — Pre-pump Deviation from Equilibrium",
                 fontsize=12, fontweight="bold")

    # Left: deviation % per alpha
    ax = axes[0]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    for i, name in enumerate(label_names):
        devs = [results[a]["deviation_pct"][i] for a in alphas]
        ax.plot(alphas, devs, "-o", ms=5, color=colors[i % len(colors)], label=name)
    ax.axvline(best_alpha, color="black", ls="--", lw=1.5, alpha=0.7, label=f"best={best_alpha}")
    ax.set_xlabel("Alpha", fontsize=10)
    ax.set_ylabel("Pre-pump Deviation (%)", fontsize=10)
    ax.set_title("Per-Distance Deviation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Right: mean deviation
    ax = axes[1]
    mean_devs = [results[a]["mean_dev"] for a in alphas]
    bars = ax.bar(range(len(alphas)), mean_devs, color="steelblue", alpha=0.7,
                  edgecolor="navy", linewidth=0.5)
    best_idx = alphas.index(best_alpha)
    bars[best_idx].set_color("#e74c3c")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.3f}" for a in alphas])
    ax.set_xlabel("Alpha", fontsize=10)
    ax.set_ylabel("Mean Deviation (%)", fontsize=10)
    ax.set_title("Mean Pre-pump Deviation")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    out_path = Path(args.out_npz).parent / "alpha_sweep_diagnostics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[alpha_sweep] saved: {out_path}")

    return best_alpha


if __name__ == "__main__":
    import sys
    # Check for alpha_sweep mode before regular argparse
    if "--alpha_sweep" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--alpha_sweep", action="store_true")
        ap.add_argument("--ckpt", type=str, default=str(cfg.PATHS.checkpoint_pt))
        ap.add_argument("--exp_dir", type=str,
                        default=str(cfg.PATHS.root / "bootstrapping" / "results_240"))
        ap.add_argument("--mat_key", type=str, default="s0")
        ap.add_argument("--s_low_mask", type=float, default=1.5)
        ap.add_argument("--s_high_mask", type=float, default=9.0)
        ap.add_argument("--baseline_start", type=float, default=5.0)
        ap.add_argument("--prepump_end", type=int, default=9)
        ap.add_argument("--alpha_values", type=str, default="0.02,0.025,0.03,0.035,0.04")
        ap.add_argument("--out_npz", type=str,
                        default=str(cfg.PATHS.params_dir / "inference_out.npz"))
        args = ap.parse_args()
        alpha_sweep(args)
    else:
        main()
