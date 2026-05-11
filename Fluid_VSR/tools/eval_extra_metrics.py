from __future__ import annotations

import numpy as np


DATASET_SPECTRUM_BANDS: dict[str, tuple[int, int]] = {
    "RB": (8, 32),
    "RBC": (8, 32),
    "ShallowWater": (8, 32),
    "SW": (8, 32),
    "KF256": (16, 64),
    "ERA5V": (16, 64),
    "ERA5": (16, 64),
    "ERA5_72": (16, 64),
    "ERA5_72frames": (16, 64),
}


def _ensure_seq_field(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Expected (T, H, W) or (T, H, W, 1), got {arr.shape}")


def compute_1d_xspec(seq: np.ndarray, remove_zmean: bool = True) -> np.ndarray:
    """Mirror /data/yc/transfer_to_a100/spectrum_metrics/metrics.py exactly."""
    field = _ensure_seq_field(seq)
    single = field.ndim == 2
    if single:
        field = field[None]

    field = field.astype(np.float64, copy=False)
    _, nx, _ = field.shape

    if remove_zmean:
        field = field - field.mean(axis=0, keepdims=True)

    fhat = np.fft.rfft(field, axis=1) / nx
    power = np.abs(fhat) ** 2
    spec = power.mean(axis=2)

    return spec[0] if single else spec


def spectrum_error(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    k_lo: int,
    k_hi: int,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Mirror /data/yc/transfer_to_a100/spectrum_metrics/metrics.py exactly."""
    pred_spec = compute_1d_xspec(pred)
    gt_spec = compute_1d_xspec(gt)
    n_modes = pred_spec.shape[1]

    errs = np.linalg.norm(pred_spec - gt_spec, axis=1) / (np.linalg.norm(gt_spec, axis=1) + eps)
    log_errs = np.linalg.norm(np.log(pred_spec + eps) - np.log(gt_spec + eps), axis=1) / (
        np.linalg.norm(np.log(gt_spec + eps), axis=1) + eps
    )

    def band_err(k_start: int, k_end: int) -> float:
        pred_band = pred_spec[:, k_start:k_end]
        gt_band = gt_spec[:, k_start:k_end]
        band = np.linalg.norm(pred_band - gt_band, axis=1) / (np.linalg.norm(gt_band, axis=1) + eps)
        return float(band.mean())

    return {
        "mean_spec_err": float(errs.mean()),
        "max_spec_err": float(errs.max()),
        "low_k_err": band_err(1, k_lo),
        "mid_k_err": band_err(k_lo, k_hi),
        "high_k_err": band_err(k_hi, n_modes),
        "mean_log_spec_err": float(log_errs.mean()),
        "spec_err_over_time": errs.tolist(),
    }


def ldiff_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mirror compute_ldiff in eval_all_datasets.py exactly."""
    pred_seq = _ensure_seq_field(pred).astype(np.float64, copy=False)
    gt_seq = _ensure_seq_field(gt).astype(np.float64, copy=False)
    dp = np.diff(pred_seq, axis=0)
    dg = np.diff(gt_seq, axis=0)
    return float(np.abs(dp - dg).mean())


def compute_rollout_extra_metrics(
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    *,
    frames_per_seq: int,
    dataset_name: str,
) -> dict[str, float]:
    if frames_per_seq <= 1:
        raise ValueError(f"frames_per_seq must be > 1, got {frames_per_seq}")
    if dataset_name not in DATASET_SPECTRUM_BANDS:
        raise ValueError(f"Unsupported dataset_name for spectrum bands: {dataset_name}")

    n_frames = min(len(pred_arr), len(gt_arr))
    usable = (n_frames // frames_per_seq) * frames_per_seq
    if usable == 0:
        raise ValueError(
            f"Not enough frames for one sequence: n_frames={n_frames}, frames_per_seq={frames_per_seq}"
        )

    pred_seq = pred_arr[:usable].reshape(-1, frames_per_seq, *pred_arr.shape[1:])
    gt_seq = gt_arr[:usable].reshape(-1, frames_per_seq, *gt_arr.shape[1:])
    k_lo, k_hi = DATASET_SPECTRUM_BANDS[dataset_name]

    eps_lo_list = []
    eps_hi_list = []
    eps_spec_list = []
    ldiff_list = []
    for pred_one, gt_one in zip(pred_seq, gt_seq):
        ref_spec = spectrum_error(pred_one, gt_one, k_lo=k_lo, k_hi=k_hi)
        eps_spec_list.append(ref_spec["mean_spec_err"])
        eps_lo_list.append(ref_spec["low_k_err"])
        eps_hi_list.append(ref_spec["high_k_err"])
        ldiff_list.append(ldiff_l1(pred_one, gt_one))

    return {
        "eps_spec": float(np.mean(eps_spec_list)),
        "eps_lo": float(np.mean(eps_lo_list)),
        "eps_hi": float(np.mean(eps_hi_list)),
        "ldiff_l1": float(np.mean(ldiff_list)),
    }
