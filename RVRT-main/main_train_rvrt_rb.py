"""Train RVRT on RayleighBenard VSR dataset.

Usage (4 GPUs):
    torchrun --nproc_per_node=4 main_train_rvrt_rb.py

Usage (single GPU):
    python main_train_rvrt_rb.py
"""

import os
import os.path as osp
import math
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from models.network_rvrt import RVRT
from data.dataset_video_rb import VideoRecurrentRBDataset

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    # data
    dataroot_gt  = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/HR',
    dataroot_lq  = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/LR',
    num_samples  = 120,
    num_frame    = 6,       # training clip length; must be divisible by clip_size=2
    train_ratio  = 0.8,
    valid_ratio  = 0.1,
    batch_size   = 4,       # per GPU
    num_workers  = 4,

    # model
    spynet_path  = '/data/yc/KAIR-master/model_zoo/vrt/spynet_sintel_final-3d2a1287.pth',
    upscale      = 4,
    clip_size    = 2,
    img_size     = [2, 32, 32],
    window_size  = [2, 8, 8],
    num_blocks   = [1, 1, 1],
    depths       = [2, 2, 2],
    embed_dims   = [64, 64, 64],
    num_heads    = [4, 4, 4],
    inputconv_groups = [1, 1, 1, 1, 1, 1],
    deformable_groups = 4,
    attention_heads   = 4,
    attention_window  = [3, 3],
    cpu_cache_length  = 100,
    in_chans     = 1,

    # training
    total_iter   = 20000,
    lr           = 1e-4,
    weight_decay = 0,
    manual_seed  = 42,

    # output
    save_dir     = 'experiments/rvrt_rb',
)
# ─────────────────────────────────────────────────────────────────────────────


def charbonnier_loss(pred, target, eps=1e-9):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # ── DDP init ──────────────────────────────────────────────────────────────
    use_ddp = 'RANK' in os.environ
    if use_ddp:
        dist.init_process_group('nccl')
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device     = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        rank = 0; world_size = 1; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main = (rank == 0)
    set_seed(CFG['manual_seed'] + rank)

    if is_main:
        os.makedirs(CFG['save_dir'], exist_ok=True)
        with open(osp.join(CFG['save_dir'], 'config.json'), 'w') as f:
            json.dump(CFG, f, indent=2)

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_opt = dict(
        dataroot_gt = CFG['dataroot_gt'], dataroot_lq = CFG['dataroot_lq'],
        num_samples = CFG['num_samples'], num_frame = CFG['num_frame'],
        train_ratio = CFG['train_ratio'], valid_ratio = CFG['valid_ratio'],
        split='train', test_mode=False,
    )
    train_dataset = VideoRecurrentRBDataset(train_opt)

    sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=CFG['batch_size'],
        sampler=sampler, shuffle=(sampler is None),
        num_workers=CFG['num_workers'], pin_memory=True, drop_last=True,
    )

    if is_main:
        print(f'Train samples: {len(train_dataset)}, iters/epoch: {len(train_loader)}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RVRT(
        upscale           = CFG['upscale'],
        clip_size         = CFG['clip_size'],
        img_size          = CFG['img_size'],
        window_size       = CFG['window_size'],
        num_blocks        = CFG['num_blocks'],
        depths            = CFG['depths'],
        embed_dims        = CFG['embed_dims'],
        num_heads         = CFG['num_heads'],
        inputconv_groups  = CFG['inputconv_groups'],
        spynet_path       = CFG['spynet_path'],
        deformable_groups = CFG['deformable_groups'],
        attention_heads   = CFG['attention_heads'],
        attention_window  = CFG['attention_window'],
        cpu_cache_length  = CFG['cpu_cache_length'],
        in_chans          = CFG['in_chans'],
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'],
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['total_iter'], eta_min=1e-7,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    current_iter = 0
    model.train()
    loader_iter  = iter(train_loader)

    while current_iter < CFG['total_iter']:
        # refill iterator each epoch
        try:
            batch = next(loader_iter)
        except StopIteration:
            if use_ddp:
                sampler.set_epoch(current_iter)
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        lq = batch['L'].to(device)   # (B, T, 1, h, w)
        gt = batch['H'].to(device)   # (B, T, 1, H, W)

        optimizer.zero_grad()
        pred = model(lq)             # (B, T, 1, H, W)
        loss = charbonnier_loss(pred, gt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_iter += 1

        if is_main and current_iter % 200 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'iter {current_iter:6d}/{CFG["total_iter"]} | '
                  f'loss {loss.item():.4e} | lr {lr_now:.3e}')

        # save at iter 1 (sanity check) and final iter
        if is_main and current_iter in (1, CFG['total_iter']):
            net = model.module if use_ddp else model
            ckpt_path = osp.join(CFG['save_dir'], f'{current_iter}_G.pth')
            torch.save({'params': net.state_dict()}, ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')

    if use_ddp:
        dist.destroy_process_group()



if __name__ == '__main__':
    main()
