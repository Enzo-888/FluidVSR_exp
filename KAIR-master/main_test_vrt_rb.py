"""
Test VRT on the RayleighBenard dataset and save predictions.

Usage:
    python main_test_vrt_rb.py --opt options/vrt/010_train_vrt_videosr_rb.json \
                               --checkpoint experiments/010_train_vrt_videosr_rb/models/200000_G.pth \
                               [--output-dir DIR] [--device cuda]

Output (saved to output_dir, default: experiments/<task>/test_predictions/):
    pred.npz   — model predictions,      shape (N, H, W, 1), physical values
    gt.npz     — ground truth HR frames, shape (N, H, W, 1), physical values
    lr.npz     — LR input frames,        shape (N, h, w, 1), physical values
    meta.json  — frames_per_seq, n_seqs, norm stats, checkpoint path
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

from utils import utils_option as option
from models.select_network import define_G
from data.dataset_video_rb import VideoRecurrentRBDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--opt',        required=True, help='Path to training JSON config')
    p.add_argument('--checkpoint', required=True, help='Path to *_G.pth checkpoint')
    p.add_argument('--output-dir', default=None,
                   help='Directory to save outputs (default: experiments/<task>/test_predictions)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()

    # Save CUDA_VISIBLE_DEVICES before option.parse overwrites it
    _saved_cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    opt = option.parse(args.opt, is_train=False)

    # Restore CUDA_VISIBLE_DEVICES and force single-GPU
    if _saved_cuda_env is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = _saved_cuda_env
    opt['gpu_ids'] = [0]
    opt = option.dict_to_nonedict(opt)

    output_dir = args.output_dir or osp.join(
        opt['path']['root'], opt['task'], 'test_predictions')
    os.makedirs(output_dir, exist_ok=True)

    # ── Build test dataset ────────────────────────────────────────────────────
    print('Building test dataset...')
    test_opt = dict(opt['datasets']['test'])   # copy so we can override split
    test_opt['split'] = 'test'
    test_opt['test_mode'] = True
    test_opt['num_frame'] = None               # full sequence at test time

    dataset = VideoRecurrentRBDataset(test_opt)
    n_seqs         = len(dataset)
    frames_per_seq = dataset.frames_per_seq

    print(f'  Test sequences : {n_seqs}')
    print(f'  Frames per seq : {frames_per_seq}')

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=True)

    # ── Build model & load checkpoint ─────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    model = define_G(opt)
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    # KAIR saves checkpoints with a 'params' or bare state_dict
    if 'params' in state_dict:
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(args.device)

    # ── Inference ─────────────────────────────────────────────────────────────
    print('Running inference...')
    all_pred, all_gt, all_lr = [], [], []

    # Initialize efficiency tracker
    from efficiency_tracker import EfficiencyTracker

    tracker = EfficiencyTracker(
        model_name='VRT',
        dataset_name='RB',
        save_dir=output_dir
    )

    # Warmup
    print('Warming up (3 iterations)...')
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            lq = batch['L'].to(args.device)
            _ = model(lq)

    # Start tracking
    tracker.start_inference(device=args.device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            lq = batch['L'].to(args.device)   # (1, T, 1, h, w)
            gt = batch['H'].to(args.device)   # (1, T, 1, H, W)

            # Time this sequence
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            start_time = time.time()

            pred = model(lq).clamp(0, 1)      # (1, T, 1, H, W)

            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed = time.time() - start_time
            tracker.record_sequence_time(elapsed)

            # Squeeze batch dim → (T, 1, H, W) → (T, H, W, 1)
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
        'checkpoint':     args.checkpoint,
        'config':         args.opt,
        'n_frames':       int(pred_arr.shape[0]),
        'pred_shape':     list(pred_arr.shape),
        'lr_shape':       list(lr_arr.shape),
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
