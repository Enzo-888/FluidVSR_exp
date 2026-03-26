# inference_SW.py
# Adapted from inference_RB.py for ShallowWater VSR task.
#
# Key differences vs RB:
#   - 200 sequences (train 160, valid 20, test 20), 72 frames per sequence
#   - HR 128x128, LR 32x32, same spatial resolution as RB
#   - sample_shape uses [1, 72, 1, 128, 128]
#   - Output saved as pred.npz / gt.npz / lr.npz in (N_frames, H, W, 1) physical values
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 python inference_SW.py \
#     --basis_path ./ckp/basis_sw_1x128x128_XXXX.pth \
#     --model_path ./exps/gp-edm_sw_XXXX/checkpoints/ema_XXXXX.pth \
#     --core_mean_std_path ./exps/gp-edm_sw_XXXX/core_mean_std.mat \
#     --norm_stats_path ./data/norm_stats_sw_XXXX.json \
#     --output_dir ./output_sw

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
    """
    Build grouped observations from LR normalized data.

    Args:
        lr_seq_norm: [T, 1, 32, 32] normalized LR for one test sequence
        t_ind_uni:   [T] time indices in [0,1]

    Returns:
        y_group:          list[T] of np.array [1024] — LR pixel values per frame
        ind_conti_group:  list[T] of np.array [1024, 3] — (u,v,w) coords (same for all T)
        y_time_group:     list[T] of float — continuous time index
        y_time_ind_group: list[T] of int   — integer time index
    """
    T = lr_seq_norm.shape[0]

    # LR coordinates: u=1.0 (D=1 single point), v and w on linspace(0,1,32)
    u_val  = np.array([1.0], dtype=np.float32)
    lr_v   = np.linspace(0, 1, 32, dtype=np.float32)
    lr_w   = np.linspace(0, 1, 32, dtype=np.float32)
    vv, ww = np.meshgrid(lr_v, lr_w, indexing='ij')
    uu     = np.ones_like(vv) * u_val[0]
    lr_coords = np.stack([uu.ravel(), vv.ravel(), ww.ravel()], axis=1)  # [1024, 3]

    y_group          = []
    ind_conti_group  = []
    y_time_group     = []
    y_time_ind_group = []

    for t in range(T):
        y_t = lr_seq_norm[t, 0, :, :].ravel().astype(np.float64)  # [1024]
        y_group.append(y_t)
        ind_conti_group.append(lr_coords)
        y_time_group.append(float(t_ind_uni[t]))
        y_time_ind_group.append(t)

    return y_group, ind_conti_group, y_time_group, y_time_ind_group


# ---- Posterior gradient (MPDPS) ----

def compute_continuous_poest(x_0, basis_function, core_mean, core_std, core_t,
                              y_group, ind_conti_group, y_time_group, y_time_ind_group,
                              MPDPS=0.4):
    core_tensor_shape = x_0.shape
    x_0_vec = x_0.view(core_tensor_shape[0], core_tensor_shape[1], -1)  # [1, T, R1R2R3]
    poest_matrix1 = torch.zeros_like(x_0_vec).to(device)
    poest_matrix2 = torch.zeros_like(x_0_vec).to(device)

    # Denormalize core (GPSD normalization → FTM [0,1] scale)
    x_0_denorm = x_0_vec * core_std + core_mean

    for y, y_tt, y_t_ind, ind in zip(y_group, y_time_group, y_time_ind_group, ind_conti_group):
        if len(y) == 0:
            continue

        # Stage 1: direct observation constraint at this time step
        x_0_t    = x_0_denorm[0, y_t_ind, :]          # [R1R2R3]
        y_tensor = torch.DoubleTensor(y).to(device)     # [1024]
        ind_tensor = torch.FloatTensor(ind).to(device)  # [1024, 3]
        A = basis_function(input_ind_sampl=ind_tensor).detach().double()  # [1024, R1R2R3]
        poest_matrix1[0, y_t_ind, :] = A.T @ (y_tensor - A @ x_0_t)

        # Stage 2: temporal message passing to all other time steps
        t_remove_group = y_time_ind_group.copy()
        t_remove_group.remove(y_t_ind)
        if len(t_remove_group) == 0:
            continue

        core_t_remove = core_t[:, t_remove_group, :]      # [1, T-1, 1]
        x_0_remove    = x_0_denorm[:, t_remove_group, :]  # [1, T-1, R1R2R3]

        ktT     = get_ktT(y_tt, core_t_remove).squeeze(2)  # [1, T-1]
        KTT_inv = get_kTT_inv(core_t_remove)                # [1, T-1, T-1]
        coeff   = (ktT @ KTT_inv).to(device).squeeze(1)     # [1, T-1] -> squeeze -> [T-1] when squeezed...

        # coeff shape after squeeze: handle both [1, T-1] and scalar cases
        if coeff.dim() == 1:
            coeff = coeff.unsqueeze(0)  # [1, T-1]

        x_0_aggregate = (coeff @ x_0_remove).squeeze()          # [R1R2R3]
        post          = A.T @ (y_tensor - A @ x_0_aggregate)     # [R1R2R3]

        temp = torch.zeros_like(x_0_vec).to(device)
        temp[:, t_remove_group, :] = coeff.unsqueeze(-1) * post.float().view(1, 1, -1)
        poest_matrix2 += temp

    return (poest_matrix1 + MPDPS * poest_matrix2).view(core_tensor_shape)


# ---- EDM model (EDM class from train_GPSD_SW, copied here for standalone use) ----

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
                        help='norm_stats_sw_*.json saved by train_FTM_SW.py')
    parser.add_argument('--output_dir',         type=str,   default='./output_sw')
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
    parser.add_argument('--model_channels',     type=int,   default=32)
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

    # ---- Load norm stats (saved by train_FTM_SW.py) ----
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
    t_ind_uni = np.linspace(0, 1, 72, dtype=np.float32)

    # ---- Load test data directly from npz ----
    HR_DIR = '/data/yc/dataset/RB/code/shallow-water-32/data1/HR'
    LR_DIR = '/data/yc/dataset/RB/code/shallow-water-32/data1/LR'
    N_TRAIN, N_VAL = 160, 20
    test_indices = list(range(N_TRAIN + N_VAL, 200))  # 180-199

    print(f"Loading {len(test_indices)} test sequences from npz...")
    hr_raw_list, lr_raw_list = [], []
    for i in test_indices:
        hr_raw_list.append(np.load(f'{HR_DIR}/sample_{i:03d}.npz')['output'].astype(np.float32))
        lr_raw_list.append(np.load(f'{LR_DIR}/sample_{i:03d}.npz')['output'].astype(np.float32))

    hr_raw = np.stack(hr_raw_list)  # [20, 72, 128, 128]
    lr_raw = np.stack(lr_raw_list)  # [20, 72, 32, 32]

    # Normalize
    hr_norm = (hr_raw[:, :, np.newaxis, :, :] - data_min) / (data_max - data_min)  # [20, 72, 1, 128, 128]
    lr_norm = (lr_raw[:, :, np.newaxis, :, :] - data_min) / (data_max - data_min)  # [20, 72, 1, 32, 32]

    N_test = hr_norm.shape[0]
    T      = hr_norm.shape[1]
    print(f"Test sequences: {N_test}, frames: {T}")

    # ---- Load basis function ----
    basis_function = torch.load(config.basis_path, map_location=device)
    basis_function.eval()
    basis_function.mode = "sampling"

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
        dataset_name='SW',
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
        lr_seq_norm = lr_norm[i]   # [72, 1, 32, 32]
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
    pred_all = np.concatenate(pred_list, axis=0).astype(np.float32)  # [1440, 128, 128, 1]
    gt_all   = np.concatenate(gt_list,   axis=0).astype(np.float32)
    lr_all   = np.concatenate(lr_list,   axis=0).astype(np.float32)  # [1440, 32, 32, 1]

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
