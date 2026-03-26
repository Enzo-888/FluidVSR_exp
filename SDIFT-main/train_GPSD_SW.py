# train_GPSD_SW.py
# Adapted from train_GPSD_RB.py for ShallowWater VSR task.
#
# Supports single-GPU and multi-GPU (DDP via torchrun).
#
# Single GPU:
#   CUDA_VISIBLE_DEVICES=0 python train_GPSD_SW.py --core_path ...
#
# 4-GPU DDP:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_GPSD_SW.py --core_path ...
#
# Memory note (single 24GB GPU):
#   model_channels=16, num_blocks=4, num_temporal_latent=2: ~23.5GB (max that fits)
#   model_channels=32 OOMs regardless of ntl.

import argparse
import os
import copy
import random
import logging
from datetime import datetime
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import scipy.io as sio
import scipy.io as scio
import matplotlib.pyplot as plt

from networks_edm import Spatial_temporal_UNet


# ---- EDM sampler ----

@torch.no_grad()
def edm_sampler(edm, latents, t, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7):
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1/rho) + step_indices / (num_steps-1) * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])])
    x_next = latents.to(torch.float64) * i_steps[0]
    for i, (i_cur, i_next) in enumerate(zip(i_steps[:-1], i_steps[1:])):
        x_hat = x_next; i_hat = i_cur
        denoised = edm(x_hat, i_hat, t).to(torch.float64)
        d_cur  = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur
        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t).to(torch.float64)
            d_prime = (x_next - denoised) / i_next
            x_next  = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def get_gp_covariance(t):
    gp_gamma = 50
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5
    return torch.exp(-torch.square(s) * gp_gamma) + diag


# ---- EDM model ----

class EDM:
    def __init__(self, model, cfg):
        self.cfg        = cfg
        self.device     = cfg.device
        self.model      = model  # may be DDP-wrapped
        # EMA tracks the underlying module (not the DDP wrapper)
        raw = model.module if isinstance(model, DDP) else model
        self.ema        = copy.deepcopy(raw).eval().requires_grad_(False).to(self.device)
        self.sigma_min  = cfg.sigma_min
        self.sigma_max  = cfg.sigma_max
        self.rho        = cfg.rho
        self.sigma_data = 0.5
        self.P_mean     = -1.2
        self.P_std      = 1.2
        self.ema_rampup_ratio   = 0.05
        self.ema_halflife_kimg  = 500

    def _raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def model_forward_wrapper(self, x, sigma, t, use_ema=False):
        sigma[sigma == 0] = self.sigma_min
        c_skip  = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out   = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in    = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        x_in = torch.einsum('b,btijk->btijk', c_in, x)
        nl   = c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1)
        net  = self.ema if use_ema else self.model
        out  = net(x_in, nl, t)
        return torch.einsum('b,btijk->btijk', c_skip, x) + torch.einsum('b,btijk->btijk', c_out, out)

    def train_step(self, signals, t):
        rnd_normal = torch.randn([signals.shape[0]], device=signals.device)
        sigma  = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        cov  = get_gp_covariance(t)
        L    = torch.linalg.cholesky(cov)
        noise = torch.randn_like(signals)
        noise = (L @ noise.view(signals.shape[0], signals.shape[1], -1)).view(signals.shape)
        n    = torch.einsum('b,btijk->btijk', sigma, noise)
        D_yn = self.model_forward_wrapper(signals + n, sigma, t)
        if self.cfg.gt_guide_type == 'l2':
            loss = torch.einsum('b,btijk->btijk', weight, (D_yn - signals) ** 2)
        else:
            loss = torch.einsum('b,btijk->btijk', weight, torch.abs(D_yn - signals))
        return loss.mean()

    def update_ema(self, step, batch_size_total):
        """batch_size_total = per-GPU batch × world_size (effective batch per step)."""
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, step * batch_size_total * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size_total / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self._raw_model().parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    def __call__(self, x, sigma, t, use_ema=True):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), t.float(), use_ema=use_ema)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ---- Helpers ----

def create_model(config):
    unet = Spatial_temporal_UNet(
        in_channels=config.channels,
        out_channels=config.channels,
        num_blocks=config.layers_per_block,
        num_temporal_latent=config.num_temporal_latent,
        attn_resolutions=config.attn_resolutions,
        model_channels=config.model_channels,
        channel_mult=config.channel_mult,
        dropout=0,
        img_resolution=config.img_size,
        label_dim=0,
        embedding_type='positional',
        encoder_type='standard',
        decoder_type='standard',
        augment_dim=9,
        channel_mult_noise=1,
        resample_filter=[1, 1],
    )
    n = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"[rank {config.rank}] Trainable params: {n:,}")
    return unet


def normalize_data(tr_y):
    data_mean = tr_y.min()
    data_std  = tr_y.max() - tr_y.min()
    return (tr_y - data_mean) / data_std, data_mean, data_std


def load_train_data(config):
    d    = sio.loadmat(config.core_path)
    # Keep core on CPU; batches are moved to device in the training loop (standard DDP pattern)
    core = torch.tensor(d['core'], dtype=torch.float32)
    core, core_mean, core_std = normalize_data(core)
    if config.rank == 0:
        print(f"core shape: {core.shape}  mean: {core_mean:.4f}  std: {core_std:.4f}")
    t = torch.linspace(0, 1, core.shape[1]).view(1, -1, 1).repeat(core.shape[0], 1, 1)
    dataset = TensorDataset(core, t)
    sampler = DistributedSampler(dataset, num_replicas=config.world_size,
                                 rank=config.rank, shuffle=True) if config.world_size > 1 else None
    loader  = DataLoader(dataset, batch_size=config.train_batch_size,
                         sampler=sampler, shuffle=(sampler is None), num_workers=0)
    return loader, sampler, core_mean, core_std


def save_plots(samples, t_grid, sample_dir, step):
    for i in range(samples.shape[0]):
        plt.plot(t_grid[i].squeeze().detach().cpu().numpy(),
                 samples[i, :, 0, 0, 0].squeeze().detach().cpu().numpy(),
                 color='C0', alpha=1 / (i + 1))
    plt.title('5 new realizations')
    plt.xlabel('t')
    plt.savefig(f'{sample_dir}/samples_{step}.png')
    plt.close()


# ---- Main ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr",             type=str,   default="gp-edm")
    parser.add_argument("--dataset",          type=str,   default="sw")
    parser.add_argument("--core_path",        type=str,   default="./data/core_sw_1x128x128_latest.mat")
    parser.add_argument('--seed',             type=int,   default=231)
    parser.add_argument("--train_batch_size", type=int,   default=1)
    parser.add_argument("--num_steps",        type=int,   default=15000)
    parser.add_argument("--learning_rate",    type=float, default=2e-4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--save_model_iters", type=int,   default=2000)
    parser.add_argument("--log_step",         type=int,   default=200)
    parser.add_argument("--warmup",           type=int,   default=1500)
    parser.add_argument("--train_progress_bar", action='store_true', default=True)
    parser.add_argument('--gt_guide_type',    type=str,   default='l2')
    parser.add_argument('--sigma_min',        type=float, default=0.002)
    parser.add_argument('--sigma_max',        type=float, default=80.0)
    parser.add_argument('--rho',              type=float, default=7.)
    parser.add_argument('--sigma_data',       type=float, default=0.5)
    parser.add_argument('--total_steps',      type=int,   default=20)
    parser.add_argument("--save_signals_step",type=int,   default=1000)
    parser.add_argument('--begin_ckpt',       type=int,   default=0)
    # Model architecture
    parser.add_argument("--img_size",         type=int,   default=128)
    parser.add_argument('--channels',         type=int,   default=1)
    parser.add_argument('--model_channels',   type=int,   default=16)   # mc=32 OOMs on 24GB
    parser.add_argument('--channel_mult',     type=int,   nargs='+', default=[1, 2, 4, 4])
    parser.add_argument('--attn_resolutions', type=int,   nargs='+', default=[])
    parser.add_argument('--layers_per_block', type=int,   default=4)
    parser.add_argument('--num_temporal_latent', type=int, default=2)   # mc=16+ntl=2 is 24GB limit

    config = parser.parse_args()

    # ---- DDP init ----
    local_rank  = int(os.environ.get('LOCAL_RANK', 0))
    rank        = int(os.environ.get('RANK', 0))
    world_size  = int(os.environ.get('WORLD_SIZE', 1))
    config.local_rank  = local_rank
    config.rank        = rank
    config.world_size  = world_size
    is_main = (rank == 0)

    if world_size > 1:
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    config.device = device

    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed_all(config.seed + rank)
    random.seed(config.seed + rank)

    # ---- Workdir (rank 0 only) ----
    config.expr = f"{config.expr}_{config.dataset}"
    run_id   = datetime.now().strftime("%Y%m%d-%H%M")
    outdir   = f"exps/{config.expr}_{run_id}"
    if is_main:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(f'{outdir}/samples',     exist_ok=True)
        os.makedirs(f'{outdir}/checkpoints', exist_ok=True)
        logging.basicConfig(
            filename=f'{outdir}/std.log', filemode='w',
            format='%(asctime)s %(levelname)s --> %(message)s',
            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger()
        logger.info(f"world_size={world_size}")
        for arg in vars(config):
            logger.info(f"  {arg}: {getattr(config, arg)}")
        print(f"outdir: {outdir}  world_size: {world_size}")
    else:
        logger = logging.getLogger()  # no-op logger on non-main ranks

    if world_size > 1:
        dist.barrier()

    # ---- Data ----
    train_loader, sampler, core_mean, core_std = load_train_data(config)
    if is_main:
        scio.savemat(f'{outdir}/core_mean_std.mat',
                     {"core_mean": core_mean.cpu().numpy(), "core_std": core_std.cpu().numpy()})

    # ---- Model ----
    mynet = create_model(config).to(device)
    if world_size > 1:
        mynet = DDP(mynet, device_ids=[local_rank], find_unused_parameters=True)
    edm = EDM(model=mynet, cfg=config)
    edm.model.train()

    optimizer = torch.optim.Adam(
        mynet.parameters() if not isinstance(mynet, DDP) else mynet.module.parameters(),
        lr=config.learning_rate)

    # effective batch per optimizer step (for EMA decay calculation)
    effective_batch = config.train_batch_size * world_size * config.accumulation_steps

    # Initialize efficiency tracker (rank 0 only)
    tracker = None
    if is_main:
        from efficiency_tracker import EfficiencyTracker

        tracker = EfficiencyTracker(
            model_name='SDIFT_GPSD',
            dataset_name='SW',
            save_dir=outdir
        )
        tracker.start_training(batch_size=effective_batch, device=str(device))

    train_loss_values = 0.0
    if is_main and config.train_progress_bar:
        progress_bar = tqdm(total=config.num_steps)

    data_iterator = iter(train_loader)
    epoch_counter = 0

    for step in range(config.num_steps):
        edm.model.train()
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)

        for acc_i in range(config.accumulation_steps):
            try:
                signal_batch, t_batch = next(data_iterator)
            except StopIteration:
                epoch_counter += 1
                if sampler is not None:
                    sampler.set_epoch(epoch_counter)
                data_iterator = iter(train_loader)
                signal_batch, t_batch = next(data_iterator)

            # Move batch to device (DataLoader yields CPU tensors after CPU-core fix)
            signal_batch = signal_batch.to(device)
            t_batch      = t_batch.to(device)

            # DDP: no_sync for all but the last accumulation step
            import contextlib
            ctx = edm.model.no_sync() if (isinstance(edm.model, DDP) and acc_i < config.accumulation_steps - 1) else contextlib.nullcontext()
            with ctx:
                loss = edm.train_step(signal_batch, t_batch) / config.accumulation_steps
                loss.backward()
            batch_loss += loss.detach()

        # lr schedule
        for g in optimizer.param_groups:
            g['lr'] = config.learning_rate * min(step / max(config.warmup, 1), 1.0)
        # gradient clipping
        raw = edm._raw_model()
        for param in raw.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()
        edm.update_ema(step, effective_batch)

        # aggregate loss across ranks for logging
        if world_size > 1:
            dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        train_loss_values += batch_loss.item()

        if is_main:
            if config.train_progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{batch_loss.item():.5f}")
            if step % config.log_step == 0 or step == config.num_steps - 1:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'step: {step:06d}  lr: {current_lr:.6f}  '
                            f'avg_loss: {train_loss_values/(step+1):.6f}  '
                            f'batch_loss: {batch_loss.item():.6f}')

        # Sampling visualization (rank 0 only)
        if is_main and config.save_signals_step and (step % config.save_signals_step == 0 or step == config.num_steps - 1):
            edm.model.eval()
            n_vis, T_vis = 5, 24
            sample_shape = [n_vis, T_vis, config.channels, config.img_size, config.img_size]
            t_grid  = torch.linspace(0, 1, T_vis).view(1, -1, 1).to(device).repeat(n_vis, 1, 1)
            cov_s   = get_gp_covariance(t_grid)
            L_s     = torch.linalg.cholesky(cov_s)
            x_T     = (L_s @ torch.randn(sample_shape, device=device).view(n_vis, T_vis, -1)).view(sample_shape)
            with torch.no_grad():
                sample = edm_sampler(edm, x_T, t_grid, num_steps=config.total_steps).detach()
            sample = (sample * core_std + core_mean).cpu()
            if step >= config.save_model_iters:
                sio.savemat(f'{outdir}/samples/core_{step}.mat', {"core": sample.numpy()})
            save_plots(sample, t_grid, f'{outdir}/samples', step)

        # Save checkpoint (rank 0 only)
        if is_main and config.save_model_iters and step > 0 and \
                (step % config.save_model_iters == 0 or step == config.num_steps - 1):
            ckpt_path = f"{outdir}/checkpoints/ema_{step}.pth"
            torch.save(edm.ema.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save efficiency stats (rank 0 only)
    if is_main and tracker:
        tracker.end_training(total_iters=config.num_steps)
        tracker.save_stats(checkpoint_path=ckpt_path)

    if world_size > 1:
        dist.destroy_process_group()
