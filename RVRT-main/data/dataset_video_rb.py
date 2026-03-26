"""RayleighBenard VSR dataset for RVRT.

Returns {'L': (T,1,h,w), 'H': (T,1,H,W)} following the same convention
as the VRT/KAIR dataset so the training loop can be shared.
"""

import glob
import json
import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoRecurrentRBDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        hr_path    = opt['dataroot_gt']
        lr_path    = opt['dataroot_lq']
        num_samples = opt.get('num_samples', 120)
        self.num_frame  = opt.get('num_frame', None)
        train_ratio = opt.get('train_ratio', 0.8)
        valid_ratio = opt.get('valid_ratio', 0.1)
        split       = opt.get('split', 'train')
        self.test_mode = opt.get('test_mode', False)
        self.use_mirror_sequence = opt.get('use_mirror_sequence', False) and not self.test_mode

        # file lists
        hr_files = sorted(glob.glob(osp.join(hr_path, 'sample_*.npz')))[:num_samples]
        lr_files = sorted(glob.glob(osp.join(lr_path, 'sample_*.npz')))[:num_samples]
        assert len(hr_files) == len(lr_files) == num_samples, (
            f'Expected {num_samples} files, got HR={len(hr_files)}, LR={len(lr_files)}')

        # split
        n = num_samples
        train_end = max(1, min(int(n * train_ratio), n - 2))
        valid_end = max(train_end + 1, min(int(n * (train_ratio + valid_ratio)), n - 1))
        split_ranges = {'train': (0, train_end), 'valid': (train_end, valid_end), 'test': (valid_end, n)}
        assert split in split_ranges, f'split must be train/valid/test, got {split}'
        s, e = split_ranges[split]

        # load into RAM
        self.hr_seqs = [np.load(f)['output'].astype(np.float32) for f in hr_files[s:e]]
        self.lr_seqs = [np.load(f)['output'].astype(np.float32) for f in lr_files[s:e]]
        self.frames_per_seq = self.hr_seqs[0].shape[0]

        # normalisation stats
        stats_path = osp.join(osp.dirname(hr_path.rstrip('/')),
                               f'RB_{num_samples}_norm_stats.json')
        if split == 'train':
            all_hr = np.concatenate(self.hr_seqs, axis=0)
            all_lr = np.concatenate(self.lr_seqs, axis=0)
            stats = {'hr_min': float(all_hr.min()), 'hr_max': float(all_hr.max()),
                     'lr_min': float(all_lr.min()), 'lr_max': float(all_lr.max())}
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
        else:
            assert osp.exists(stats_path), f'Norm stats not found at {stats_path}. Build train dataset first.'
            with open(stats_path) as f:
                stats = json.load(f)

        self.hr_min = stats['hr_min']; self.hr_max = stats['hr_max']
        self.lr_min = stats['lr_min']; self.lr_max = stats['lr_max']
        hr_range = self.hr_max - self.hr_min + 1e-12
        lr_range = self.lr_max - self.lr_min + 1e-12
        self.hr_seqs = [(s - self.hr_min) / hr_range for s in self.hr_seqs]
        self.lr_seqs = [(s - self.lr_min) / lr_range for s in self.lr_seqs]

        # build index
        self.data_infos = []
        for seq_idx in range(len(self.hr_seqs)):
            seq_len = self.hr_seqs[seq_idx].shape[0]
            if self.test_mode or self.num_frame is None:
                self.data_infos.append(dict(seq_idx=seq_idx, start=0, T=seq_len))
            else:
                T = self.num_frame
                for start in range(seq_len - T + 1):
                    self.data_infos.append(dict(seq_idx=seq_idx, start=start, T=T))

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        info = self.data_infos[idx]
        seq_idx, start, T = info['seq_idx'], info['start'], info['T']

        lq = self.lr_seqs[seq_idx][start:start + T].copy()   # (T, h, w)
        gt = self.hr_seqs[seq_idx][start:start + T].copy()   # (T, H, W)

        if not self.test_mode:
            if np.random.random() < 0.5:
                lq = lq[:, :, ::-1].copy(); gt = gt[:, :, ::-1].copy()
            if np.random.random() < 0.5:
                lq = lq[:, ::-1, :].copy(); gt = gt[:, ::-1, :].copy()
            if np.random.random() < 0.5:
                lq = lq.transpose(0, 2, 1).copy(); gt = gt.transpose(0, 2, 1).copy()

        if self.use_mirror_sequence:
            lq = np.concatenate([lq, lq[::-1]], axis=0)
            gt = np.concatenate([gt, gt[::-1]], axis=0)

        # (T, H, W) → (T, 1, H, W)
        lq_t = torch.from_numpy(lq).unsqueeze(1).float()
        gt_t = torch.from_numpy(gt).unsqueeze(1).float()
        return {'L': lq_t, 'H': gt_t}

    def denorm_hr(self, x):
        return x * (self.hr_max - self.hr_min) + self.hr_min

    def denorm_lr(self, x):
        return x * (self.lr_max - self.lr_min) + self.lr_min
