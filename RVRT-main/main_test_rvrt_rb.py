"""Test RVRT on RayleighBenard test split and save predictions.

Usage:
    python main_test_rvrt_rb.py --checkpoint experiments/rvrt_rb/20000_G.pth

Output (saved to experiments/rvrt_rb/test_predictions/):
    pred.npz   shape (N, H, W, 1), physical values
    gt.npz     shape (N, H, W, 1), physical values
    lr.npz     shape (N, h, w, 1), physical values
    meta.json
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from models.network_rvrt import RVRT
from data.dataset_video_rb import VideoRecurrentRBDataset

# ── same model config as training ──────────────────────────────────────────
MODEL_CFG = dict(
    spynet_path       = '/data/yc/KAIR-master/model_zoo/vrt/spynet_sintel_final-3d2a1287.pth',
    upscale           = 4,
    clip_size         = 2,
    img_size          = [2, 32, 32],
    window_size       = [2, 8, 8],
    num_blocks        = [1, 1, 1],
    depths            = [2, 2, 2],
    embed_dims        = [64, 64, 64],
    num_heads         = [4, 4, 4],
    inputconv_groups  = [1, 1, 1, 1, 1, 1],
    deformable_groups = 4,
    attention_heads   = 4,
    attention_window  = [3, 3],
    cpu_cache_length  = 100,
    in_chans          = 1,
)

DATA_CFG = dict(
    dataroot_gt  = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/HR',
    dataroot_lq  = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/LR',
    num_samples  = 120,
    train_ratio  = 0.8,
    valid_ratio  = 0.1,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to *_G.pth checkpoint')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or osp.join(osp.dirname(args.checkpoint), 'test_predictions')
    os.makedirs(output_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_opt = dict(**DATA_CFG, num_frame=None, split='test', test_mode=True)
    dataset  = VideoRecurrentRBDataset(test_opt)
    loader   = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    n_seqs   = len(dataset)
    fps      = dataset.frames_per_seq
    print(f'Test sequences: {n_seqs}, frames/seq: {fps}')

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    model = RVRT(**MODEL_CFG).to(args.device)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state['params'], strict=True)
    model.eval()

    # ── Efficiency tracker ────────────────────────────────────────────────────
    from efficiency_tracker import EfficiencyTracker
    tracker = EfficiencyTracker(
        model_name='RVRT',
        dataset_name='RB',
        save_dir=output_dir
    )

    # ── Warmup ────────────────────────────────────────────────────────────────
    print('Warming up (3 iterations)...')
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            lq = batch['L'].to(args.device)
            _ = model(lq)

    # ── Inference ─────────────────────────────────────────────────────────────
    tracker.start_inference(device=args.device)

    all_pred, all_gt, all_lr = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            lq = batch['L'].to(args.device)   # (1, T, 1, h, w)
            gt = batch['H'].to(args.device)   # (1, T, 1, H, W)

            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            start_time = time.time()

            pred = model(lq).clamp(0, 1)      # (1, T, 1, H, W)

            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            tracker.record_sequence_time(elapsed)

            # (1, T, 1, H, W) → (T, H, W, 1), denorm to physical
            pred_np = pred.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)
            gt_np   = gt.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)
            lq_np   = lq.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)

            pred_np = dataset.denorm_hr(pred_np)
            gt_np   = dataset.denorm_hr(gt_np)
            lq_np   = dataset.denorm_lr(lq_np)

            all_pred.append(pred_np)
            all_gt.append(gt_np)
            all_lr.append(lq_np)
            print(f'  Sequence {i+1}/{n_seqs} - {elapsed:.4f}s')

    # ── Save efficiency stats ─────────────────────────────────────────────────
    tracker.save_stats(checkpoint_path=args.checkpoint)

    # ── Save ──────────────────────────────────────────────────────────────────
    pred_arr = np.concatenate(all_pred, axis=0)
    gt_arr   = np.concatenate(all_gt,   axis=0)
    lr_arr   = np.concatenate(all_lr,   axis=0)

    np.savez_compressed(osp.join(output_dir, 'pred.npz'), data=pred_arr)
    np.savez_compressed(osp.join(output_dir, 'gt.npz'),   data=gt_arr)
    np.savez_compressed(osp.join(output_dir, 'lr.npz'),   data=lr_arr)

    meta = {
        'checkpoint':     args.checkpoint,
        'n_frames':       int(pred_arr.shape[0]),
        'frames_per_seq': fps,
        'n_seqs':         n_seqs,
        'pred_shape':     list(pred_arr.shape),
        'norm_stats': {
            'hr_min': dataset.hr_min, 'hr_max': dataset.hr_max,
            'lr_min': dataset.lr_min, 'lr_max': dataset.lr_max,
        },
    }
    with open(osp.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved to: {output_dir}')
    print(f'  pred: {pred_arr.shape}')
    print(f'  gt:   {gt_arr.shape}')
    print(f'  lr:   {lr_arr.shape}')


if __name__ == '__main__':
    main()
