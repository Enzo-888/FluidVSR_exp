"""
Test BasicVSR++ on the ShallowWater dataset and save predictions.

Usage:
    python tools/test_rb.py CONFIG CHECKPOINT [--output-dir DIR] [--device cuda]

Output (saved to output_dir, default: <work_dir>/test_predictions/):
    pred.npz   — model predictions,     shape (N, H, W, 1), physical values
    gt.npz     — ground truth HR frames, shape (N, H, W, 1), physical values
    lr.npz     — LR input frames,        shape (N, h, w, 1), physical values
    meta.json  — frames_per_seq, n_seqs, shape, norm stats, checkpoint path

Results in pred/gt/lr can be fed directly into eval_rb.ipynb for metrics.
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

# Allow running from repo root
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmcv.parallel import DataContainer
from mmedit.models import build_model
from mmedit.datasets import build_dataset


def dc_collate(batch):
    """Unwrap DataContainer before collating."""
    import torch
    unwrapped = []
    for sample in batch:
        out = {}
        for k, v in sample.items():
            out[k] = v.data if isinstance(v, DataContainer) else v
        unwrapped.append(out)
    # Now use default collate on plain tensors/arrays
    return torch.utils.data.dataloader.default_collate(unwrapped)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config',     help='Config file path')
    p.add_argument('checkpoint', help='Checkpoint file path')
    p.add_argument('--output-dir', default=None,
                   help='Directory to save pred/gt/lr npz files '
                        '(default: <work_dir>/test_predictions)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    output_dir = args.output_dir or osp.join(cfg.work_dir, 'test_predictions')
    os.makedirs(output_dir, exist_ok=True)

    # ── Build test dataset ────────────────────────────────────────────────────
    print('Building test dataset...')
    dataset = build_dataset(cfg.data.test)

    # dataset exposes norm stats for denormalisation
    frames_per_seq = dataset.frames_per_seq
    n_seqs         = len(dataset)   # one entry per full sequence at test time

    print(f'  Test sequences : {n_seqs}')
    print(f'  Frames per seq : {frames_per_seq}')

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dc_collate,
    )

    # ── Build model & load checkpoint ─────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval().to(args.device)

    # ── Inference ─────────────────────────────────────────────────────────────
    print('Running inference...')
    all_pred, all_gt, all_lr = [], [], []

    # Initialize efficiency tracker
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from efficiency_tracker import EfficiencyTracker

    tracker = EfficiencyTracker(
        model_name='BasicVSR++',
        dataset_name='SW',
        save_dir=output_dir
    )

    # Warmup
    print('Warming up (3 iterations)...')
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            lq = batch['lq'].to(args.device)
            _ = model.generator(lq)

    # Start tracking
    tracker.start_inference(device=args.device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            lq = batch['lq'].to(args.device)   # (1, T, 1, h, w)
            gt = batch['gt'].to(args.device)   # (1, T, 1, H, W)

            # Time this sequence
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            start_time = time.time()

            # model.generator returns (1, T, 1, H, W)
            pred = model.generator(lq).clamp(0, 1)

            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed = time.time() - start_time
            tracker.record_sequence_time(elapsed)

            # Move to CPU, squeeze batch dim → (T, 1, H, W) → (T, H, W, 1)
            pred_np = pred.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)
            gt_np   = gt.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)
            lq_np   = lq.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)

            # Denormalise to physical values
            pred_np = dataset.denorm_hr(pred_np)
            gt_np   = dataset.denorm_hr(gt_np)
            lq_np   = dataset.denorm_lr(lq_np)

            all_pred.append(pred_np)
            all_gt.append(gt_np)
            all_lr.append(lq_np)

            print(f'  Sequence {i + 1}/{n_seqs} - {elapsed:.4f}s')

    # Save efficiency stats
    tracker.save_stats(checkpoint_path=args.checkpoint)

    # ── Concatenate & save ────────────────────────────────────────────────────
    pred_arr = np.concatenate(all_pred, axis=0)   # (N_frames, H, W, 1)
    gt_arr   = np.concatenate(all_gt,   axis=0)
    lr_arr   = np.concatenate(all_lr,   axis=0)

    np.savez_compressed(osp.join(output_dir, 'pred.npz'), data=pred_arr)
    np.savez_compressed(osp.join(output_dir, 'gt.npz'),   data=gt_arr)
    np.savez_compressed(osp.join(output_dir, 'lr.npz'),   data=lr_arr)

    meta = {
        'checkpoint':    args.checkpoint,
        'config':        args.config,
        'n_frames':      int(pred_arr.shape[0]),
        'pred_shape':    list(pred_arr.shape),
        'lr_shape':      list(lr_arr.shape),
        'frames_per_seq': frames_per_seq,
        'n_seqs':         n_seqs,
        'norm_stats': {
            'hr_min': dataset.hr_min, 'hr_max': dataset.hr_max,
            'lr_min': dataset.lr_min, 'lr_max': dataset.lr_max,
        },
    }
    with open(osp.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved to: {output_dir}')
    print(f'  pred.npz : {pred_arr.shape}  (physical values)')
    print(f'  gt.npz   : {gt_arr.shape}')
    print(f'  lr.npz   : {lr_arr.shape}')


if __name__ == '__main__':
    main()
