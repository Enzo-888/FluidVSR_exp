"""
Standalone video super-resolution evaluation script.

Computes metrics from saved pred.npz / gt.npz files (output of save_predictions.py).
Can also be used with predictions saved by other frameworks (BasicVSR, RVRT, etc.)
as long as they are saved in the same npz format.

Dependencies:
    pip install torchmetrics clean-fid

Usage:
    # Basic metrics only (MSE, RMSE, PSNR, SSIM, Relative-L2)
    python tools/eval_metrics.py --pred results/model_a/pred.npz --gt results/model_a/gt.npz

    # With FID (per-frame)
    python tools/eval_metrics.py --pred pred.npz --gt gt.npz --metrics basic fid

    # With FVD (per-sequence, requires frames_per_seq)
    python tools/eval_metrics.py --pred pred.npz --gt gt.npz --frames_per_seq 100 --metrics basic fid fvd

    # Read frames_per_seq automatically from meta.json
    python tools/eval_metrics.py --pred pred.npz --gt gt.npz --meta meta.json --metrics basic fid fvd

Input format:
    pred.npz / gt.npz: npz file with key 'data', array shape (N, H, W, C) or (N, H, W)
                       values should be in physical (denormalized) space.

FVD note:
    FVD uses torchvision's R3D-18 (pretrained on Kinetics-400) as the feature extractor.
    This is NOT the original I3D-based FVD from "Towards Accurate Generative Models of Video",
    but is a consistent and dependency-free alternative. Label results as "FVD(R3D-18)".
    Single-channel physical fields are replicated to 3 channels before feature extraction.
"""

import os
import sys
import json
import argparse
import tempfile
import warnings

import numpy as np
import torch
import torch.nn.functional as F


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, help='Path to pred.npz')
    p.add_argument('--gt',   required=True, help='Path to gt.npz')
    p.add_argument('--meta', default=None,
                   help='Path to meta.json (to auto-read frames_per_seq)')
    p.add_argument('--frames_per_seq', type=int, default=None,
                   help='Frames per video sequence (required for FVD)')
    p.add_argument('--metrics', nargs='+', default=['basic'],
                   choices=['basic', 'fid', 'fvd'],
                   help='Which metric groups to compute (default: basic)')
    p.add_argument('--vorticity', action='store_true',
                   help='Compute vorticity-specific metrics (VE, EE) - ONLY for KF256 (vorticity field), NOT for RB (temperature field)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Batch size for FID/FVD feature extraction')
    p.add_argument('--output', default=None,
                   help='Optional path to save results as JSON')
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_npz(path):
    """Load npz and return array of shape (N, H, W, C)."""
    d = np.load(path)
    # Support key 'data' (our format) or 'arr_0' (numpy default) or first key
    if 'data' in d:
        arr = d['data']
    else:
        key = list(d.keys())[0]
        warnings.warn(f"Key 'data' not found in {path}, using '{key}'")
        arr = d[key]
    if arr.ndim == 3:          # (N, H, W) -> (N, H, W, 1)
        arr = arr[..., np.newaxis]
    return arr.astype(np.float32)


# ── PSDD: Power Spectrum Density Discrepancy ──────────────────────────────────

def compute_psdd(pred_arr, gt_arr, eps=1e-12):
    """
    Power Spectrum Density Discrepancy (PSDD).
    Implements equations (31)-(38):
      1. Mean-center each sample over spatial axes (H, W).
      2. 2D FFT + fftshift (zero-frequency centred).
      3. Overflow-safe scaling: divide Re/Im by max(|Re|, |Im|) per sample.
      4. PSD = Re^2 + Im^2, normalised to a probability distribution.
      5. PSDD = mean L1 distance between normalised PSDs, averaged over N.

    pred_arr, gt_arr: numpy (N, H, W, C), physical values.
    Returns: scalar float (lower = better spectral agreement).
    """
    x = pred_arr.astype(np.float64) - pred_arr.mean(axis=(1, 2), keepdims=True)
    y = gt_arr.astype(np.float64)   - gt_arr.mean(axis=(1, 2), keepdims=True)

    Fx = np.fft.fftshift(np.fft.fft2(x, axes=(1, 2)), axes=(1, 2))
    Fy = np.fft.fftshift(np.fft.fft2(y, axes=(1, 2)), axes=(1, 2))

    def norm_psd(F):
        s = np.maximum(
            np.abs(F.real).max(axis=(1, 2, 3), keepdims=True),
            np.abs(F.imag).max(axis=(1, 2, 3), keepdims=True),
        )
        s = np.maximum(s, eps)
        P = (F.real / s) ** 2 + (F.imag / s) ** 2
        return P / (P.sum(axis=(1, 2, 3), keepdims=True) + eps)

    return float(np.abs(norm_psd(Fx) - norm_psd(Fy)).mean(axis=(1, 2, 3)).mean())


# ── Vorticity-specific metrics ───────────────────────────────────────────────

@torch.no_grad()
def compute_vorticity_error(pred_arr, gt_arr, device, batch_size=256):
    """
    计算涡度误差 (Vorticity Error, VE)

    VE = (1/N) * Σ[(1/M) * Σ(ω_i^gen - ω_i^real)^2]

    其中:
    - N: 时间步数（batch size）
    - M: 空间网格点数
    - ω^gen: 生成涡度
    - ω^real: 真实（DNS）涡度

    Args:
        pred_arr: 预测涡度场，numpy array shape (N, H, W, C)
        gt_arr: 真实涡度场，numpy array shape (N, H, W, C)
        device: torch device
        batch_size: batch size for processing

    Returns:
        涡度误差的标量值
    """
    N = pred_arr.shape[0]
    ve_list = []

    for i in range(0, N, batch_size):
        pred = torch.tensor(pred_arr[i:i+batch_size]).to(device)
        target = torch.tensor(gt_arr[i:i+batch_size]).to(device)

        # Convert to (B, C, H, W) format
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        # 计算每个空间点的误差平方
        point_error_squared = (pred - target) ** 2

        # 对空间和通道维度求平均: (1/M) * Σ(ω_i^gen - ω_i^real)^2
        spatial_mse = torch.mean(point_error_squared, dim=[1, 2, 3])  # (B,)

        ve_list.append(spatial_mse.cpu().numpy())

    # 对所有时间步求平均后开方 (RMSE of vorticity)
    ve = np.sqrt(np.concatenate(ve_list).mean())

    return float(ve)


@torch.no_grad()
def compute_enstrophy_error(pred_arr, gt_arr, device, batch_size=256):
    """
    计算拟涡能误差 (Enstrophy Error, EE)

    定义:
    - Enstrophy: ε(t) = (1/2M) * Σ(ω_i^2)
    - Enstrophy Error: EE = (1/N) * Σ(ε^gen(t) - ε^real(t))^2

    其中:
    - N: 时间步数（batch size）
    - M: 空间网格点数
    - ω: 涡度
    - ε: 拟涡能（enstrophy）

    Args:
        pred_arr: 预测涡度场，numpy array shape (N, H, W, C)
        gt_arr: 真实涡度场，numpy array shape (N, H, W, C)
        device: torch device
        batch_size: batch size for processing

    Returns:
        拟涡能误差的标量值
    """
    N = pred_arr.shape[0]
    ee_list = []

    for i in range(0, N, batch_size):
        pred = torch.tensor(pred_arr[i:i+batch_size]).to(device)
        target = torch.tensor(gt_arr[i:i+batch_size]).to(device)

        # Convert to (B, C, H, W) format
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        # 计算每个时间步的enstrophy
        # ε(t) = (1/2M) * Σ(ω_i^2)
        enstrophy_pred = torch.mean(pred ** 2, dim=[1, 2, 3]) / 2.0  # (B,)
        enstrophy_target = torch.mean(target ** 2, dim=[1, 2, 3]) / 2.0  # (B,)

        # 计算enstrophy的误差平方
        enstrophy_diff_squared = (enstrophy_pred - enstrophy_target) ** 2  # (B,)

        ee_list.append(enstrophy_diff_squared.cpu().numpy())

    # 对所有时间步求平均
    ee = np.concatenate(ee_list).mean()

    return float(ee)


# ── Basic metrics (torchmetrics + PSDD) ───────────────────────────────────────

def compute_basic_metrics(pred_arr, gt_arr, device, batch_size=256, include_vorticity=False):
    """
    Compute MSE, RMSE, PSNR, SSIM, Relative-L2, PSDD.
    Optionally compute vorticity-specific metrics (VE, EE) for KF256 dataset.
    All metrics are computed per-frame and averaged.

    pred_arr, gt_arr: numpy (N, H, W, C)
    include_vorticity: if True, compute VE and EE (for vorticity fields like KF256)
    """
    try:
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )
    except ImportError:
        raise ImportError("pip install torchmetrics")

    N, H, W, C = pred_arr.shape

    # Compute data range from the actual data
    data_range = float(max(pred_arr.max(), gt_arr.max()) - min(pred_arr.min(), gt_arr.min()))

    psnr_fn = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    mse_list, psnr_list, ssim_list, rel_l2_list = [], [], [], []

    for i in range(0, N, batch_size):
        p = torch.tensor(pred_arr[i:i+batch_size]).to(device)
        g = torch.tensor(gt_arr[i:i+batch_size]).to(device)

        p = p.permute(0, 3, 1, 2)   # (B, C, H, W)
        g = g.permute(0, 3, 1, 2)

        mse_val  = F.mse_loss(p, g, reduction='mean').item()
        psnr_val = psnr_fn(p, g).item()
        ssim_val = ssim_fn(p, g).item()

        diff_norm  = (p - g).pow(2).sum(dim=(1,2,3)).sqrt()
        gt_norm    = g.pow(2).sum(dim=(1,2,3)).sqrt().clamp_min(1e-12)
        rel_l2_val = (diff_norm / gt_norm).mean().item()

        b = p.shape[0]
        mse_list.append((mse_val,  b))
        psnr_list.append((psnr_val, b))
        ssim_list.append((ssim_val, b))
        rel_l2_list.append((rel_l2_val, b))

    def weighted_mean(lst):
        return sum(v * n for v, n in lst) / sum(n for _, n in lst)

    mse = weighted_mean(mse_list)

    # PSDD (computed on full arrays, not batched — uses numpy FFT)
    psdd = compute_psdd(pred_arr, gt_arr)

    results = {
        'mse':         mse,
        'rmse':        mse ** 0.5,
        'psnr':        weighted_mean(psnr_list),
        'ssim':        weighted_mean(ssim_list),
        'relative_l2': weighted_mean(rel_l2_list),
        'psdd':        psdd,
    }

    # Add vorticity-specific metrics if requested
    if include_vorticity:
        ve = compute_vorticity_error(pred_arr, gt_arr, device, batch_size)
        ee = compute_enstrophy_error(pred_arr, gt_arr, device, batch_size)
        results['vorticity_error'] = ve
        results['enstrophy_error'] = ee

    return results


# ── FID (cleanfid, per-frame) ─────────────────────────────────────────────────

def _frames_to_png_dir(arr, out_dir, vmin, vmax):
    """Save (N, H, W, C) array as PNG files into out_dir."""
    from PIL import Image
    for i, frame in enumerate(arr):
        img = frame.squeeze()                                   # (H, W)
        img = ((img - vmin) / (vmax - vmin + 1e-12) * 255).clip(0, 255).astype(np.uint8)
        img_rgb = np.stack([img, img, img], axis=-1)            # (H, W, 3)
        Image.fromarray(img_rgb).save(os.path.join(out_dir, f'{i:07d}.png'))


def compute_fid(pred_arr, gt_arr, device, batch_size):
    """FID between predicted frames and ground truth frames (distribution-level)."""
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        raise ImportError("pip install clean-fid")

    # Shared normalization range so real/fake are on the same scale
    vmin = float(min(pred_arr.min(), gt_arr.min()))
    vmax = float(max(pred_arr.max(), gt_arr.max()))

    with tempfile.TemporaryDirectory() as pred_dir, \
         tempfile.TemporaryDirectory() as gt_dir:

        print('  Saving frames for FID...')
        _frames_to_png_dir(pred_arr, pred_dir, vmin, vmax)
        _frames_to_png_dir(gt_arr,   gt_dir,   vmin, vmax)

        print('  Computing FID...')
        score = cleanfid.compute_fid(
            gt_dir, pred_dir,
            device=device,
            batch_size=batch_size,
            verbose=False,
        )

    return {'fid': score}


# ── FVD (R3D-18 features, per-sequence) ──────────────────────────────────────

def _load_r3d18(device):
    """Load R3D-18 pretrained on Kinetics-400, strip classifier head."""
    import torchvision.models.video as vm
    model = vm.r3d_18(weights=vm.R3D_18_Weights.KINETICS400_V1)
    # Remove avgpool + fc, keep stem + layer1-4
    feature_extractor = torch.nn.Sequential(
        model.stem,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        torch.nn.AdaptiveAvgPool3d(1),
    )
    feature_extractor.eval().to(device)
    return feature_extractor


def _extract_video_features(model, seqs_arr, device, batch_size):
    """
    seqs_arr: numpy (N_seqs, T, H, W, C), values in physical space
    Returns: numpy (N_seqs, feat_dim)
    """
    # Normalize to [0, 1] then to ImageNet mean/std (R3D-18 expectation)
    vmin, vmax = seqs_arr.min(), seqs_arr.max()
    seqs_norm = (seqs_arr - vmin) / (vmax - vmin + 1e-12)          # [0,1]

    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)

    all_feats = []
    N = seqs_norm.shape[0]

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = seqs_norm[i:i+batch_size]                      # (B, T, H, W, C)
            B, T, H, W, C = batch.shape

            # Single-channel → replicate to 3 channels
            if C == 1:
                batch = np.repeat(batch, 3, axis=-1)               # (B, T, H, W, 3)

            # Resize spatial dims to 112 (minimum for R3D-18)
            t = torch.tensor(batch).permute(0, 4, 1, 2, 3).float()  # (B, 3, T, H, W)
            if H != 112 or W != 112:
                B_, C_, T_, H_, W_ = t.shape
                t = t.reshape(B_ * T_, C_, H_, W_)
                t = F.interpolate(t, size=(112, 112), mode='bilinear', align_corners=False)
                t = t.reshape(B_, C_, T_, 112, 112)

            # Apply ImageNet normalization
            mn = torch.tensor(mean).view(1, 3, 1, 1, 1)
            sd = torch.tensor(std).view(1, 3, 1, 1, 1)
            t = (t - mn) / sd

            t = t.to(device)
            feats = model(t).squeeze(-1).squeeze(-1).squeeze(-1)   # (B, feat_dim)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def _frechet_distance(feats_real, feats_fake):
    """Compute Fréchet distance between two sets of feature vectors."""
    from scipy.linalg import sqrtm

    mu1, sigma1 = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    mu2, sigma2 = feats_fake.mean(0), np.cov(feats_fake, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fvd)


def compute_fvd(pred_arr, gt_arr, frames_per_seq, device, batch_size):
    """
    FVD(R3D-18) between predicted and ground truth video sequences.
    pred_arr, gt_arr: (N_frames, H, W, C)
    frames_per_seq:   int, must evenly divide N_frames
    """
    N = pred_arr.shape[0]
    T = frames_per_seq
    assert N % T == 0, (
        f'N_frames={N} is not divisible by frames_per_seq={T}. '
        f'Trailing frames will be dropped — set frames_per_seq correctly.'
    )
    N_seqs = N // T
    H, W, C = pred_arr.shape[1:]

    pred_seqs = pred_arr.reshape(N_seqs, T, H, W, C)
    gt_seqs   = gt_arr.reshape(N_seqs, T, H, W, C)

    if N_seqs < 2:
        warnings.warn(f'Only {N_seqs} sequence(s) — FVD covariance estimate is unreliable.')

    print('  Loading R3D-18 feature extractor...')
    model = _load_r3d18(device)

    print(f'  Extracting features from {N_seqs} GT sequences...')
    gt_feats   = _extract_video_features(model, gt_seqs,   device, batch_size)
    print(f'  Extracting features from {N_seqs} pred sequences...')
    pred_feats = _extract_video_features(model, pred_seqs, device, batch_size)

    score = _frechet_distance(gt_feats, pred_feats)
    return {'fvd_r3d18': score}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Read frames_per_seq from meta.json if provided
    frames_per_seq = args.frames_per_seq
    if args.meta and os.path.isfile(args.meta):
        with open(args.meta) as f:
            meta = json.load(f)
        if frames_per_seq is None:
            frames_per_seq = meta.get('frames_per_seq')
            print(f'Read frames_per_seq={frames_per_seq} from {args.meta}')

    print(f'Loading pred: {args.pred}')
    pred_arr = load_npz(args.pred)
    print(f'Loading gt:   {args.gt}')
    gt_arr   = load_npz(args.gt)

    assert pred_arr.shape == gt_arr.shape, \
        f'Shape mismatch: pred={pred_arr.shape}, gt={gt_arr.shape}'
    print(f'Shape: {pred_arr.shape}  (N, H, W, C)')

    results = {}

    # ── Basic metrics ─────────────────────────────────────────────────────────
    if 'basic' in args.metrics:
        print('\n[basic] Computing MSE / RMSE / PSNR / SSIM / Relative-L2 / PSDD...')
        if args.vorticity:
            print('         + Vorticity Error (VE) / Enstrophy Error (EE)')
        r = compute_basic_metrics(pred_arr, gt_arr, args.device,
                                   batch_size=256, include_vorticity=args.vorticity)
        results.update(r)
        for k, v in r.items():
            print(f'  {k:<18s}: {v:.6f}')

    # ── FID ───────────────────────────────────────────────────────────────────
    if 'fid' in args.metrics:
        print('\n[fid] Computing FID (per-frame, cleanfid)...')
        r = compute_fid(pred_arr, gt_arr, args.device, args.batch_size)
        results.update(r)
        for k, v in r.items():
            print(f'  {k:<14s}: {v:.4f}')

    # ── FVD ───────────────────────────────────────────────────────────────────
    if 'fvd' in args.metrics:
        if frames_per_seq is None:
            print('\n[fvd] SKIPPED — frames_per_seq not provided. '
                  'Use --frames_per_seq or --meta.')
        else:
            print(f'\n[fvd] Computing FVD(R3D-18) with frames_per_seq={frames_per_seq}...')
            r = compute_fvd(pred_arr, gt_arr, frames_per_seq, args.device, args.batch_size)
            results.update(r)
            for k, v in r.items():
                print(f'  {k:<14s}: {v:.4f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n── Results ──────────────────────────────────────────────────')
    for k, v in results.items():
        print(f'  {k:<14s}: {v:.6f}')

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved to: {args.output}')


if __name__ == '__main__':
    main()
