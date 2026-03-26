# Rayleigh-Benard convection dataset with pre-computed HR/LR pairs
# Each .npz file contains one simulation sequence:
#   - 'input':  (H, W)         initial condition (1 frame)
#   - 'output': (100, H, W)    100 future frames
# HR resolution: 128x128, LR resolution: 32x32, scale factor: 4x
# We treat every frame as an independent image for image super-resolution.
# Data is split by sequence (not by frame) to avoid data leakage.

import torch
import os.path as osp
import glob
import numpy as np

from torch.utils.data import Dataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class RayleighBenardDataset:
    def __init__(self, data_args, **kwargs):
        hr_path = data_args['hr_path']
        lr_path = data_args['lr_path']
        num_samples = data_args.get('num_samples', 100)
        train_ratio = data_args.get('train_ratio', 0.8)
        valid_ratio = data_args.get('valid_ratio', 0.1)
        test_ratio = data_args.get('test_ratio', 0.1)
        normalize = data_args.get('normalize', True)
        normalizer_type = data_args.get('normalizer_type', 'PGN')
        use_input_frame = data_args.get('use_input_frame', False)

        cache_name = f'RB_{num_samples}samples_cache.pt'
        cache_path = osp.join(osp.dirname(hr_path.rstrip('/')), cache_name)

        if osp.exists(cache_path):
            print(f'Loading cached data from {cache_path}')
            cached = torch.load(cache_path)
            if len(cached) == 8:
                train_x, train_y, valid_x, valid_y, test_x, test_y, hr_normalizer, lr_normalizer = cached
            else:
                # legacy cache without lr_normalizer — delete and fall through to rebuild
                import os
                os.remove(cache_path)
                print(f'  旧格式缓存（{len(cached)}值），已删除，重新生成...')

        if not osp.exists(cache_path):
            print(f'Loading RayleighBenard data ({num_samples} sequences)...')
            hr_files = sorted(glob.glob(osp.join(hr_path, 'sample_*.npz')))[:num_samples]
            lr_files = sorted(glob.glob(osp.join(lr_path, 'sample_*.npz')))[:num_samples]
            assert len(hr_files) == len(lr_files) == num_samples, \
                f'Expected {num_samples} files, got HR={len(hr_files)}, LR={len(lr_files)}'

            # Load all frames: each npz has output (100, H, W)
            hr_all, lr_all = [], []
            for hf, lf in zip(hr_files, lr_files):
                hd, ld = np.load(hf), np.load(lf)
                hr_frames = hd['output']  # (100, 128, 128)
                lr_frames = ld['output']  # (100, 32, 32)
                if use_input_frame:
                    hr_frames = np.concatenate([hd['input'][None], hr_frames], axis=0)
                    lr_frames = np.concatenate([ld['input'][None], lr_frames], axis=0)
                hr_all.append(hr_frames)
                lr_all.append(lr_frames)

            n_seqs = len(hr_all)

            # Split by sequence index (ensure each split has at least 1 sequence)
            train_end = int(n_seqs * train_ratio)
            valid_end = int(n_seqs * (train_ratio + valid_ratio))
            train_end = max(1, min(train_end, n_seqs - 2))
            valid_end = max(train_end + 1, min(valid_end, n_seqs - 1))

            def concat_and_reshape(arr_list):
                # list of (T, H, W) -> (N, H, W, 1)
                arr = np.concatenate(arr_list, axis=0)  # (N, H, W)
                return torch.tensor(arr, dtype=torch.float32).unsqueeze(-1)  # (N, H, W, 1)

            train_hr = concat_and_reshape(hr_all[:train_end])
            train_lr = concat_and_reshape(lr_all[:train_end])
            valid_hr = concat_and_reshape(hr_all[train_end:valid_end])
            valid_lr = concat_and_reshape(lr_all[train_end:valid_end])
            test_hr = concat_and_reshape(hr_all[valid_end:])
            test_lr = concat_and_reshape(lr_all[valid_end:])

            print(f'Split: train={len(train_hr)}, valid={len(valid_hr)}, test={len(test_hr)} frames')
            print(f'HR shape per frame: {train_hr.shape[1:]}, LR shape per frame: {train_lr.shape[1:]}')

            # Normalize HR and LR independently (different resolutions → separate normalizers)
            train_x, train_y, hr_normalizer, lr_normalizer = self.pre_process(
                train_lr, train_hr, mode='train',
                normalize=normalize, normalizer_type=normalizer_type)
            valid_x, valid_y = self.pre_process(
                valid_lr, valid_hr, mode='valid',
                normalize=normalize, hr_normalizer=hr_normalizer, lr_normalizer=lr_normalizer)
            test_x, test_y = self.pre_process(
                test_lr, test_hr, mode='test',
                normalize=normalize, hr_normalizer=hr_normalizer, lr_normalizer=lr_normalizer)

            print(f'Saving cache to {cache_path}')
            torch.save((train_x, train_y, valid_x, valid_y, test_x, test_y, hr_normalizer, lr_normalizer), cache_path)

        self.normalizer    = hr_normalizer
        self.lr_normalizer = lr_normalizer
        self.train_dataset = RayleighBenardBase(train_x, train_y, mode='train')
        self.valid_dataset = RayleighBenardBase(valid_x, valid_y, mode='valid')
        self.test_dataset = RayleighBenardBase(test_x, test_y, mode='test')
        self.frames_per_seq = 101 if use_input_frame else 100

    def pre_process(self, lr_data, hr_data, mode='train', normalize=False,
                    normalizer_type='PGN', hr_normalizer=None, lr_normalizer=None):
        """
        lr_data: (N, h, w, 1) - pre-computed low-resolution frames
        hr_data: (N, H, W, 1) - high-resolution ground truth frames
        No blur/downsampling needed since LR is from physical simulation.

        HR and LR have different spatial resolutions (128x128 vs 32x32),
        so UnitGaussianNormalizer (per-pixel mean/std) requires separate instances.
        """
        B_hr, H, W, C = hr_data.shape
        B_lr, h, w, C_lr = lr_data.shape

        if normalize:
            hr_flat = hr_data.reshape(B_hr, -1, C)
            lr_flat = lr_data.reshape(B_lr, -1, C_lr)

            if mode == 'train':
                if normalizer_type == 'PGN':
                    hr_normalizer = UnitGaussianNormalizer(hr_flat)
                    lr_normalizer = UnitGaussianNormalizer(lr_flat)
                else:
                    hr_normalizer = GaussianNormalizer(hr_flat)
                    lr_normalizer = GaussianNormalizer(lr_flat)

            hr_flat = hr_normalizer.encode(hr_flat)
            lr_flat = lr_normalizer.encode(lr_flat)

            hr_data = hr_flat.reshape(B_hr, H, W, C)
            lr_data = lr_flat.reshape(B_lr, h, w, C_lr)
        else:
            hr_normalizer = None
            lr_normalizer = None

        if mode == 'train':
            return lr_data, hr_data, hr_normalizer, lr_normalizer
        else:
            return lr_data, hr_data


class RayleighBenardBase(Dataset):
    """
    PyTorch Dataset wrapper for RayleighBenard HR/LR pairs.

    Args:
        x: (N, h, w, 1) low-resolution frames (32x32)
        y: (N, H, W, 1) high-resolution frames (128x128)
    """
    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
