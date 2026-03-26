# Copyright (c) OpenMMLab. All rights reserved.
import glob
import json
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import DATASETS


@DATASETS.register_module()
class SRSWDataset(Dataset):
    """Kolmogorov Flow (SW) VSR dataset for BasicVSR++.

    Loads HR/LR pairs from sample_*.npz files.
    Each .npz contains:
        'output': (T, H, W) — T frames at HR (128×128) or LR (32×32)

    Args:
        hr_path (str): Directory containing HR sample_*.npz files.
        lr_path (str): Directory containing LR sample_*.npz files.
        pipeline (list[dict]): Pipeline transforms.
        scale (int): SR upscale factor. Default: 4.
        split (str): One of 'train', 'valid', 'test'. Default: 'train'.
        num_samples (int): Number of trajectories to load. Default: 200.
        num_input_frames (int | None): Frames per training clip. If None,
            the full sequence is returned. Default: None.
        train_ratio (float): Fraction for training. Default: 0.8.
        valid_ratio (float): Fraction for validation. Default: 0.1.
        use_mirror_sequence (bool): Temporally mirror training clips.
            Default: True.
        test_mode (bool): Set True for val/test splits. Default: False.
    """

    def __init__(self,
                 hr_path,
                 lr_path,
                 pipeline,
                 scale=4,
                 split='train',
                 num_samples=200,
                 num_input_frames=None,
                 train_ratio=0.8,
                 valid_ratio=0.1,
                 use_mirror_sequence=True,
                 test_mode=False):

        self.hr_path = hr_path
        self.lr_path = lr_path
        self.scale = scale
        self.split = split
        self.num_input_frames = num_input_frames
        self.use_mirror_sequence = use_mirror_sequence and (not test_mode)
        self.test_mode = test_mode

        # ── Load sequences ────────────────────────────────────────────────────
        hr_files = sorted(glob.glob(osp.join(hr_path, 'sample_*.npz')))[:num_samples]
        lr_files = sorted(glob.glob(osp.join(lr_path, 'sample_*.npz')))[:num_samples]
        assert len(hr_files) == len(lr_files) == num_samples, (
            f'Expected {num_samples} files, got HR={len(hr_files)}, '
            f'LR={len(lr_files)}')

        # ── Train/val/test split by sequence index ────────────────────────────
        n = num_samples
        train_end = max(1, min(int(n * train_ratio), n - 2))
        valid_end = max(train_end + 1, min(int(n * (train_ratio + valid_ratio)), n - 1))

        split_ranges = {
            'train': (0, train_end),
            'valid': (train_end, valid_end),
            'test':  (valid_end, n),
        }
        assert split in split_ranges, f'split must be train/valid/test, got {split}'
        s, e = split_ranges[split]

        # ── Load data into RAM ────────────────────────────────────────────────
        self.hr_seqs = []   # list of (T, H, W) float32
        self.lr_seqs = []   # list of (T, h, w) float32
        for hf, lf in zip(hr_files[s:e], lr_files[s:e]):
            self.hr_seqs.append(np.load(hf)['output'].astype(np.float32))
            self.lr_seqs.append(np.load(lf)['output'].astype(np.float32))

        self.frames_per_seq = self.hr_seqs[0].shape[0]

        # ── Normalization stats (computed from train split, reused elsewhere) ─
        stats_path = osp.join(
            osp.dirname(hr_path.rstrip('/')),
            f'KF_{num_samples}_norm_stats.json')
        self.stats_path = stats_path

        if split == 'train':
            all_hr = np.concatenate(self.hr_seqs, axis=0)
            all_lr = np.concatenate(self.lr_seqs, axis=0)
            stats = {
                'hr_min': float(all_hr.min()),
                'hr_max': float(all_hr.max()),
                'lr_min': float(all_lr.min()),
                'lr_max': float(all_lr.max()),
            }
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f'[KF Dataset] Norm stats saved to {stats_path}')
        else:
            assert osp.exists(stats_path), (
                f'Norm stats not found at {stats_path}. '
                f'Build the train dataset first to generate them.')
            with open(stats_path) as f:
                stats = json.load(f)

        self.hr_min = stats['hr_min']
        self.hr_max = stats['hr_max']
        self.lr_min = stats['lr_min']
        self.lr_max = stats['lr_max']

        # ── Normalise sequences in-place ──────────────────────────────────────
        hr_range = self.hr_max - self.hr_min + 1e-12
        lr_range = self.lr_max - self.lr_min + 1e-12
        self.hr_seqs = [(s - self.hr_min) / hr_range for s in self.hr_seqs]
        self.lr_seqs = [(s - self.lr_min) / lr_range for s in self.lr_seqs]

        # ── Build data_infos ──────────────────────────────────────────────────
        self.data_infos = []
        T = num_input_frames if num_input_frames is not None else self.frames_per_seq

        for seq_idx in range(len(self.hr_seqs)):
            seq_len = self.hr_seqs[seq_idx].shape[0]
            if test_mode or num_input_frames is None:
                self.data_infos.append(
                    dict(seq_idx=seq_idx, start=0, T=seq_len))
            else:
                for start in range(seq_len - T + 1):
                    self.data_infos.append(
                        dict(seq_idx=seq_idx, start=start, T=T))

        from mmedit.datasets.pipelines import Compose
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        info = self.data_infos[idx]
        seq_idx = info['seq_idx']
        start   = info['start']
        T       = info['T']

        lq = self.lr_seqs[seq_idx][start:start + T].copy()  # (T, h, w)
        gt = self.hr_seqs[seq_idx][start:start + T].copy()  # (T, H, W)

        # ── Spatial augmentation (training only) ──────────────────────────────
        if not self.test_mode:
            if np.random.random() < 0.5:
                lq = lq[:, :, ::-1].copy()
                gt = gt[:, :, ::-1].copy()
            if np.random.random() < 0.5:
                lq = lq[:, ::-1, :].copy()
                gt = gt[:, ::-1, :].copy()
            if np.random.random() < 0.5:
                lq = lq.transpose(0, 2, 1).copy()
                gt = gt.transpose(0, 2, 1).copy()

        # ── Temporal mirror extension (training only) ─────────────────────────
        if self.use_mirror_sequence:
            lq = np.concatenate([lq, lq[::-1]], axis=0)
            gt = np.concatenate([gt, gt[::-1]], axis=0)

        # ── To tensor: (T, H, W) -> (T, 1, H, W) ─────────────────────────────
        lq_t = torch.from_numpy(lq).unsqueeze(1).float()
        gt_t = torch.from_numpy(gt).unsqueeze(1).float()

        key = f'traj{seq_idx:04d}'
        results = dict(
            lq=lq_t,
            gt=gt_t,
            lq_path=osp.join(self.lr_path, f'sample_{seq_idx:03d}.npz'),
            gt_path=osp.join(self.hr_path, f'sample_{seq_idx:03d}.npz'),
            key=key,
            scale=self.scale,
        )
        return self.pipeline(results)

    def evaluate(self, results, logger=None):
        return {}

    def denorm_hr(self, x):
        return x * (self.hr_max - self.hr_min) + self.hr_min

    def denorm_lr(self, x):
        return x * (self.lr_max - self.lr_min) + self.lr_min
