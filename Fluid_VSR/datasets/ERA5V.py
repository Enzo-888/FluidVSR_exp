# ERA5V dataset (ERA5 200hPa wind, pre-computed HR/LR daily npz pairs)
# Each .npz file contains one day of data:
#   - 'data': (24, H, W)  24 hourly frames
# HR resolution: 256x256, LR resolution: 64x64, scale factor: 4x
# We treat every frame as an independent image for image super-resolution.
# Data is split by day (not by frame) to avoid data leakage.

import torch
import os.path as osp
import glob
import numpy as np

from torch.utils.data import Dataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class ERA5VDataset:
    def __init__(self, data_args, **kwargs):
        hr_path = data_args['hr_path']
        lr_path = data_args['lr_path']
        num_samples = data_args.get('num_samples', 578)
        train_ratio = data_args.get('train_ratio', 0.98)
        valid_ratio = data_args.get('valid_ratio', 0.01)
        test_ratio = data_args.get('test_ratio', 0.01)
        normalize = data_args.get('normalize', True)
        normalizer_type = data_args.get('normalizer_type', 'PGN')

        cache_name = f'ERA5V_{num_samples}samples_cache.pt'
        cache_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'cache')
        import os; os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, cache_name)

        if osp.exists(cache_path):
            print(f'Loading cached data from {cache_path}')
            cached = torch.load(cache_path)
            if len(cached) == 8:
                train_x, train_y, valid_x, valid_y, test_x, test_y, hr_normalizer, lr_normalizer = cached
            else:
                import os
                os.remove(cache_path)
                print(f'  旧格式缓存（{len(cached)}值），已删除，重新生成...')

        if not osp.exists(cache_path):
            print(f'Loading ERA5V data ({num_samples} days)...')
            hr_files = sorted(glob.glob(osp.join(hr_path, '*.npz')))[:num_samples]
            lr_files = sorted(glob.glob(osp.join(lr_path, '*.npz')))[:num_samples]
            assert len(hr_files) == len(lr_files) == num_samples, \
                f'Expected {num_samples} files, got HR={len(hr_files)}, LR={len(lr_files)}'

            hr_all, lr_all = [], []
            for hf, lf in zip(hr_files, lr_files):
                hd, ld = np.load(hf), np.load(lf)
                hr_frames = hd['data']  # (24, 256, 256)
                lr_frames = ld['data']  # (24, 64, 64)
                hr_all.append(hr_frames)
                lr_all.append(lr_frames)

            n_seqs = len(hr_all)

            train_end = int(n_seqs * train_ratio)
            valid_end = int(n_seqs * (train_ratio + valid_ratio))
            train_end = max(1, min(train_end, n_seqs - 2))
            valid_end = max(train_end + 1, min(valid_end, n_seqs - 1))

            def concat_and_reshape(arr_list):
                arr = np.concatenate(arr_list, axis=0)
                return torch.tensor(arr, dtype=torch.float32).unsqueeze(-1)

            train_hr = concat_and_reshape(hr_all[:train_end])
            train_lr = concat_and_reshape(lr_all[:train_end])
            valid_hr = concat_and_reshape(hr_all[train_end:valid_end])
            valid_lr = concat_and_reshape(lr_all[train_end:valid_end])
            test_hr = concat_and_reshape(hr_all[valid_end:])
            test_lr = concat_and_reshape(lr_all[valid_end:])

            print(f'Split: train={len(train_hr)}, valid={len(valid_hr)}, test={len(test_hr)} frames')
            print(f'HR shape per frame: {train_hr.shape[1:]}, LR shape per frame: {train_lr.shape[1:]}')

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
        self.train_dataset = ERA5VBase(train_x, train_y, mode='train')
        self.valid_dataset = ERA5VBase(valid_x, valid_y, mode='valid')
        self.test_dataset  = ERA5VBase(test_x, test_y, mode='test')
        self.frames_per_seq = 24

    def pre_process(self, lr_data, hr_data, mode='train', normalize=False,
                    normalizer_type='PGN', hr_normalizer=None, lr_normalizer=None):
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


class ERA5VBase(Dataset):
    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
