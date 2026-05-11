# inference_RB.py
# Adapted from message_passing_DPS.py for RayleighBenard VSR task.
#
# Key changes vs original:
#   - Observations come from paired LR 32x32 frames (not random sparse points)
#   - Posterior guidance matches the true forward model: decode HR -> bilinear x4 -> compare to LR
#   - sample_shape uses [1, 100, 1, 128, 128]
#   - Output saved as pred.npz / gt.npz / lr.npz in (N_frames, H, W, 1) physical values
#     (for eval by /data/yc/Fluid_VSR/tools/eval_metrics.py)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 python inference_RB.py \
#     --basis_path ./ckp/basis_rb_1x128x128_XXXX.pth \
#     --model_path ./exps/gp-edm_rb_XXXX/checkpoints/ema_XXXXX.pth \
#     --core_mean_std_path ./exps/gp-edm_rb_XXXX/core_mean_std.mat \
#     --output_dir ./results/rb

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import random
import scipy.io as sio
import time
import sys

from networks_edm import Spatial_temporal_UNet
from sr_posterior_utils import (
    build_lr_observations as build_paired_lr_observations,
    compute_bilinear_sr_posterior_grad,
    precompute_decoder_bases,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_ktT(y_tt, core_t, gp_gamma=1, gp_sigma=1):
    r = torch.sqrt(torch.square(y_tt - core_t))
    return gp_sigma * torch.exp(-torch.square(r) * gp_gamma)


def get_kTT_inv(t, gp_gamma=1, gp_sigma=1):
    r = torch.sqrt(torch.square(t - t.transpose(-1, -2)))
    diag = torch.eye(t.shape[-2]).to(t) * 1e-3
    K = gp_sigma * torch.exp(-torch.square(r) * gp_gamma) + diag
    return torch.inverse(K)


def get_gp_covariance(t, gp_gamma=50):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5
    return torch.exp(-torch.square(s) * gp_gamma) + diag


# ---- Observation construction ----

def build_lr_observations(lr_seq_norm, t_ind_uni):
    return build_paired_lr_observations(lr_seq_norm, t_ind_uni)


# ---- Posterior gradient (MPDPS) ----

def compute_continuous_poest(x_0, basis_function, core_mean, core_std, core_t,
                              y_group, ind_conti_group, y_time_group, y_time_ind_group,
                              decoder_bases, lr_hw, MPDPS=0.4):
    del basis_function, ind_conti_group
    return compute_bilinear_sr_posterior_grad(
        x_0=x_0,
        decoder_bases=decoder_bases,
        core_mean=core_mean,
        core_std=core_std,
        core_t=core_t,
        y_group=y_group,
        y_time_group=y_time_group,
        y_time_ind_group=y_time_ind_group,
        lr_hw=lr_hw,
        get_ktT_fn=get_ktT,
        get_kTT_inv_fn=get_kTT_inv,
        MPDPS=MPDPS,
    )


# ---- EDM model (EDM class from train_GPSD_RB, copied here for standalone use) ----

class EDM:
    def __init__(self, model=None, cfg=None):
        self.cfg = cfg
        self.device = cfg.device
        self.model  = model.to(self.device)
        self.ema    = None   # not needed for inference
        self.sigma_min  = cfg.sigma_min
        self.sigma_max  = cfg.sigma_max
        self.rho        = cfg.rho
        self.sigma_data = cfg.sigma_data
        self.P_mean     = -1.2
        self.P_std      = 1.2

    def model_forward_wrapper(self, x, sigma, t, use_ema=False):
        sigma[sigma == 0] = self.sigma_min
        c_skip  = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out   = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in    = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_output = self.model(
            torch.einsum('b,btijk->btijk', c_in, x),
            c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1),
            t)
        return torch.einsum('b,btijk->btijk', c_skip, x) + torch.einsum('b,btijk->btijk', c_out, model_output)

    def __call__(self, x, sigma, t, use_ema=False):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), t.float())

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ---- Posterior sampler ----

@torch.no_grad()
def edm_post_sampler(edm, basis_function, latents, t, y_group, ind_conti_group,
                     y_time_group, y_time_ind_group,
                     decoder_bases, lr_hw,
                     num_steps=20, sigma_min=0.002, sigma_max=80, rho=7,
                     zeta=0.009, MPDPS=0.4):
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])])

    x_next = latents.to(torch.float64) * i_steps[0]
    t_start = time.time()

    for i, (i_cur, i_next) in tqdm(enumerate(zip(i_steps[:-1], i_steps[1:]))):
        x_hat = x_next
        i_hat = i_cur

        # Euler step
        denoised = edm(x_hat, i_hat, t).to(torch.float64)
        denoised_core1 = denoised.detach().clone()
        d_cur  = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur

        # 2nd order correction + MPDPS gradient
        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t).to(torch.float64)
            denoised_core2 = denoised.detach().clone()
            d_prime = (x_next - denoised) / i_next
            x_next  = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)

            denoised_core = (denoised_core1 + denoised_core2) / 2
            llk_grad = compute_continuous_poest(
                denoised_core, basis_function, core_mean, core_std, t,
                y_group, ind_conti_group, y_time_group, y_time_ind_group,
                decoder_bases, lr_hw,
                MPDPS=MPDPS)
            x_next = x_next + (zeta / (i + 1)) * llk_grad

    print(f"  sampling done in {time.time() - t_start:.1f}s")
    return x_next


# ---- Tucker decoder ----

def decoder(u_ind_uni, v_ind_uni, w_ind_uni, core, basis_function):
    """
    core: [T, 1, 128, 128] tensor (on device)
    Returns numpy array [T, 1, 128, 128]
    """
    u_t = torch.FloatTensor(u_ind_uni).to(device)
    v_t = torch.FloatTensor(v_ind_uni).to(device)
    w_t = torch.FloatTensor(w_ind_uni).to(device)

    basis_function.eval()
    basis_function.mode = "training"
    core = core.to(torch.float32)

    basises = basis_function(input_ind_train=(u_t, v_t, w_t))
    output  = torch.einsum("mi, tijk->tmjk", basises[0], core)
    output  = torch.einsum("nj, tmjk->tmnk", basises[1], output)
    output  = torch.einsum("ok, tmnk->tmno", basises[2], output)
    return output.cpu().detach().numpy()  # [T, 1, 128, 128]


def create_model(config):
    return Spatial_temporal_UNet(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',               type=int,   default=123)
    parser.add_argument('--basis_path',         type=str,   required=True)
    parser.add_argument('--model_path',         type=str,   required=True)
    parser.add_argument('--core_mean_std_path', type=str,   required=True)
    parser.add_argument('--norm_stats_path',    type=str,   required=True,
                        help='norm_stats_rb_*.json saved by train_FTM_RB.py')
    parser.add_argument('--output_dir',         type=str,   default='./results/rb')
    # DPS hyperparameters
    parser.add_argument('--MPDPS',              type=float, default=0.4)
    parser.add_argument('--zeta',               type=float, default=0.009)
    parser.add_argument('--total_steps',        type=int,   default=20)
    # Model architecture (must match trained model)
    parser.add_argument('--sigma_min',          type=float, default=0.002)
    parser.add_argument('--sigma_max',          type=float, default=80.0)
    parser.add_argument('--rho',                type=float, default=7.)
    parser.add_argument('--sigma_data',         type=float, default=0.5)
    parser.add_argument("--img_size",           type=int,   default=128)
    parser.add_argument('--channels',           type=int,   default=1)
    parser.add_argument('--model_channels',     type=int,   default=16)
    parser.add_argument('--channel_mult',       type=int,   nargs='+', default=[1, 2, 4, 4])
    parser.add_argument('--attn_resolutions',   type=int,   nargs='+', default=[])
    parser.add_argument('--layers_per_block',   type=int,   default=4)
    parser.add_argument('--num_temporal_latent',type=int,   default=2)

    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    set_seed(config.seed)
    print("Device:", device)

    os.makedirs(config.output_dir, exist_ok=True)

    # ---- Load norm stats ----
    d = sio.loadmat(config.core_mean_std_path)
    core_mean = torch.tensor(d['core_mean'], dtype=torch.float32).to(device)
    core_std  = torch.tensor(d['core_std'],  dtype=torch.float32).to(device)

    # ---- Load norm stats (saved by train_FTM_RB.py) ----
    import json
    with open(config.norm_stats_path) as f:
        stats = json.load(f)
    data_min = float(stats['data_min'])
    data_max = float(stats['data_max'])
    print(f"data_min={data_min:.4f}  data_max={data_max:.4f}")

    # ---- Fixed coordinates ----
    u_ind_uni = np.array([1.0], dtype=np.float32)
    hr_v_ind  = np.linspace(0, 1, 128, dtype=np.float32)
    hr_w_ind  = np.linspace(0, 1, 128, dtype=np.float32)
    t_ind_uni = np.linspace(0, 1, 100, dtype=np.float32)

    # ---- Load test data directly from npz ----
    HR_DIR = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/HR'
    LR_DIR = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/LR'
    N_TRAIN, N_VAL = 96, 12
    test_indices = list(range(N_TRAIN + N_VAL, 120))  # 108-119

    print(f"Loading {len(test_indices)} test sequences from npz...")
    hr_raw_list, lr_raw_list = [], []
    for i in test_indices:
        hr_raw_list.append(np.load(f'{HR_DIR}/sample_{i:03d}.npz')['output'].astype(np.float32))
        lr_raw_list.append(np.load(f'{LR_DIR}/sample_{i:03d}.npz')['output'].astype(np.float32))

    hr_raw = np.stack(hr_raw_list)  # [12, 100, 128, 128]
    lr_raw = np.stack(lr_raw_list)  # [12, 100, 32, 32]

    # Normalize
    hr_norm = (hr_raw[:, :, np.newaxis, :, :] - data_min) / (data_max - data_min)  # [12, 100, 1, 128, 128]
    lr_norm = (lr_raw[:, :, np.newaxis, :, :] - data_min) / (data_max - data_min)  # [12, 100, 1, 32, 32]

    N_test = hr_norm.shape[0]
    T      = hr_norm.shape[1]
    lr_hw  = tuple(lr_norm.shape[-2:])
    print(f"Test sequences: {N_test}, frames: {T}")

    # ---- Load basis function ----
    basis_function = torch.load(config.basis_path, map_location=device)
    basis_function.eval()
    basis_function.mode = "sampling"
    decoder_bases = precompute_decoder_bases(
        basis_function, u_ind_uni, hr_v_ind, hr_w_ind, device=device
    )

    # ---- Load diffusion model ----
    my_net = create_model(config)
    edm    = EDM(model=my_net, cfg=config)
    checkpoint = torch.load(config.model_path, map_location=device)
    edm.model.load_state_dict(checkpoint)
    for param in edm.model.parameters():
        param.requires_grad = False
    edm.model.eval()
    print("Models loaded.")

    # ---- Initialize efficiency tracker ----
    from efficiency_tracker import EfficiencyTracker

    tracker = EfficiencyTracker(
        model_name='SDIFT',
        dataset_name='RB',
        save_dir=config.output_dir
    )

    # Warmup (3 iterations)
    print("Warming up (3 iterations)...")
    for warmup_i in range(min(3, N_test)):
        lr_seq_norm = lr_norm[warmup_i]
        y_group, ind_conti_group, y_time_group, y_time_ind_group = \
            build_lr_observations(lr_seq_norm, t_ind_uni)
        sample_shape = [1, T, config.channels, config.img_size, config.img_size]
        t_grid = (torch.linspace(0, 1, T).view(1, -1, 1).to(device)).repeat(1, 1, 1).double()
        cov_sample  = get_gp_covariance(t_grid)
        L_sample    = torch.linalg.cholesky(cov_sample).to(device)
        noise_sample = torch.randn(sample_shape).to(device).double()
        x_T = (L_sample @ noise_sample.view(1, T, -1)).view(sample_shape)
        _ = edm_post_sampler(
            edm, basis_function, x_T, t_grid,
            y_group, ind_conti_group, y_time_group, y_time_ind_group,
            decoder_bases, lr_hw,
            num_steps=config.total_steps,
            zeta=config.zeta,
            MPDPS=config.MPDPS,
        )

    # Start tracking
    tracker.start_inference(device=str(device))

    # ---- Run inference ----
    pred_list = []
    gt_list   = []
    lr_list   = []

    for i in range(N_test):
        print(f"\n[{i+1}/{N_test}] Sequence {i}")

        # Build LR observations
        lr_seq_norm = lr_norm[i]   # [100, 1, 32, 32]
        y_group, ind_conti_group, y_time_group, y_time_ind_group = \
            build_lr_observations(lr_seq_norm, t_ind_uni)

        # GP-driven initial noise
        sample_shape = [1, T, config.channels, config.img_size, config.img_size]
        t_grid = (torch.linspace(0, 1, T).view(1, -1, 1).to(device)).repeat(1, 1, 1).double()
        cov_sample  = get_gp_covariance(t_grid)
        L_sample    = torch.linalg.cholesky(cov_sample).to(device)
        noise_sample = torch.randn(sample_shape).to(device).double()
        x_T = (L_sample @ noise_sample.view(1, T, -1)).view(sample_shape)

        # Time this sequence
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start_time = time.time()

        # Posterior sampling
        sample = edm_post_sampler(
            edm, basis_function, x_T, t_grid,
            y_group, ind_conti_group, y_time_group, y_time_ind_group,
            decoder_bases, lr_hw,
            num_steps=config.total_steps,
            zeta=config.zeta,
            MPDPS=config.MPDPS,
        ).detach()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.time() - start_time
        tracker.record_sequence_time(elapsed)

        # Denormalize core (GPSD scale → [0,1])
        core_sample = (sample * core_std + core_mean)  # [1, T, 1, 128, 128]

        # Tucker decode: [T, 1, 128, 128] normalized [0,1]
        basis_function.mode = "sampling"
        out_norm = decoder(u_ind_uni, hr_v_ind, hr_w_ind, core_sample[0], basis_function)
        # out_norm: [T, 1, 128, 128]

        # Denormalize to physical values
        out_phys = out_norm * (data_max - data_min) + data_min  # [T, 1, 128, 128]

        # Reformat to (T, H, W, 1)
        pred_phys = out_phys.transpose(0, 2, 3, 1)   # [T, 128, 128, 1]
        gt_phys   = hr_raw[i][:, :, :, np.newaxis]   # [T, 128, 128, 1]
        lr_phys   = lr_raw[i][:, :, :, np.newaxis]   # [T, 32, 32, 1]

        rmse = np.sqrt(np.mean((pred_phys - gt_phys) ** 2))
        print(f"  RMSE (physical): {rmse:.6f}  Time: {elapsed:.4f}s")

        pred_list.append(pred_phys)
        gt_list.append(gt_phys)
        lr_list.append(lr_phys)

        basis_function.mode = "sampling"

    # ---- Save efficiency stats ----
    tracker.save_stats(checkpoint_path=config.model_path)

    # ---- Save results ----
    # Stack: [N_test*T, H, W, 1] to match eval_metrics.py convention
    pred_all = np.concatenate(pred_list, axis=0).astype(np.float32)  # [1200, 128, 128, 1]
    gt_all   = np.concatenate(gt_list,   axis=0).astype(np.float32)
    lr_all   = np.concatenate(lr_list,   axis=0).astype(np.float32)  # [1200, 32, 32, 1]

    np.savez(os.path.join(config.output_dir, 'pred.npz'), data=pred_all)
    np.savez(os.path.join(config.output_dir, 'gt.npz'),   data=gt_all)
    np.savez(os.path.join(config.output_dir, 'lr.npz'),   data=lr_all)

    # ---- Save meta.json ----
    import json as json_mod
    meta = {
        'frames_per_seq': T,
        'n_seqs': N_test,
        'norm_stats': {'data_min': data_min, 'data_max': data_max},
        'checkpoint': config.model_path,
    }
    with open(os.path.join(config.output_dir, 'meta.json'), 'w') as f:
        json_mod.dump(meta, f, indent=2)

    print(f"\nSaved pred/gt/lr.npz + meta.json to {config.output_dir}")
    print(f"  pred shape: {pred_all.shape}  gt shape: {gt_all.shape}  lr shape: {lr_all.shape}")

    overall_rmse = np.sqrt(np.mean((pred_all - gt_all) ** 2))
    print(f"Overall RMSE: {overall_rmse:.6f}")
