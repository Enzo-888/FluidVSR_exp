"""
Save test-set predictions for a trained model.

Usage:
    python tools/save_predictions.py --model_dir logs/my_model
    python tools/save_predictions.py --model_dir logs/my_model --output_dir results/my_model

Output (saved to output_dir):
    pred.npz  — model predictions,      shape (N, H, W, C), denormalized (physical values)
    gt.npz    — ground truth HR frames,  shape (N, H, W, C), denormalized (physical values)
    lr.npz    — LR input frames,         shape (N, h, w, C), normalized (model input space)
    meta.json — dataset metadata (frames_per_seq, n_seqs, shape, model_dir, ...)

NOTE: pred and gt are in physical space (after hr_normalizer.decode).
      lr is kept in normalized space because lr_normalizer is not cached separately.
      For eval_metrics.py you only need pred.npz + gt.npz + meta.json.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecastors.base import BaseForecaster
from datasets import _dataset_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True,
                   help='Path to model directory (must contain config.yaml + best_model.pth)')
    p.add_argument('--output_dir', default=None,
                   help='Where to save output files (default: <model_dir>/test_predictions)')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Override eval batch size from config')
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(args.model_dir, 'test_predictions')
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load model (config.yaml + best_model.pth) ─────────────────────────
    print(f'Loading model from: {args.model_dir}')
    forecaster = BaseForecaster(args.model_dir)

    # ── 2. Rebuild dataset from the same config ───────────────────────────────
    data_args = forecaster.data_args
    data_name = data_args['name']
    print(f'Building dataset: {data_name}')
    dataset = _dataset_dict[data_name](data_args)
    normalizer = dataset.normalizer       # hr_normalizer, used for decode

    frames_per_seq = getattr(dataset, 'frames_per_seq', None)

    batch_size = args.batch_size or data_args.get('eval_batchsize', 10)
    test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ── 3. Run inference on test set ──────────────────────────────────────────
    print(f'Running inference on {len(dataset.test_dataset)} test frames...')
    all_pred, all_gt, all_lr = [], [], []

    forecaster.model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(forecaster.device, non_blocking=True)
            y = y.to(forecaster.device, non_blocking=True)

            y_pred = forecaster.inference(x, y)
            y_pred = normalizer.decode(y_pred)
            y      = normalizer.decode(y)

            all_pred.append(y_pred.cpu().numpy())
            all_gt.append(y.cpu().numpy())
            all_lr.append(x.cpu().numpy())   # normalized LR input

            if (i + 1) % 10 == 0:
                print(f'  batch {i+1}/{len(test_loader)}')

    pred_arr = np.concatenate(all_pred, axis=0)   # (N, H, W, C)
    gt_arr   = np.concatenate(all_gt,   axis=0)
    lr_arr   = np.concatenate(all_lr,   axis=0)   # (N, h, w, C)

    # ── 4. Save ───────────────────────────────────────────────────────────────
    np.savez_compressed(os.path.join(output_dir, 'pred.npz'), data=pred_arr)
    np.savez_compressed(os.path.join(output_dir, 'gt.npz'),   data=gt_arr)
    np.savez_compressed(os.path.join(output_dir, 'lr.npz'),   data=lr_arr)

    meta = {
        'model_dir':      args.model_dir,
        'data_name':      data_name,
        'n_frames':       int(pred_arr.shape[0]),
        'pred_shape':     list(pred_arr.shape),
        'lr_shape':       list(lr_arr.shape),
        'frames_per_seq': frames_per_seq,
        'n_seqs':         (pred_arr.shape[0] // frames_per_seq
                           if frames_per_seq else None),
        'lr_note':        'LR frames are in normalized (model input) space, not physical values',
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved to: {output_dir}')
    print(f'  pred.npz : {pred_arr.shape}  (physical values)')
    print(f'  gt.npz   : {gt_arr.shape}    (physical values)')
    print(f'  lr.npz   : {lr_arr.shape}    (normalized input)')
    print(f'  meta.json: frames_per_seq={frames_per_seq}, n_seqs={meta["n_seqs"]}')


if __name__ == '__main__':
    main()
