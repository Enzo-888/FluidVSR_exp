import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from efficiency_tracker import EfficiencyTracker
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
    diag = torch.eye(t.shape[-2], device=t.device, dtype=t.dtype) * 1e-3
    K = gp_sigma * torch.exp(-torch.square(r) * gp_gamma) + diag
    return torch.inverse(K)


def get_gp_covariance(t, gp_gamma=50):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2], device=t.device, dtype=t.dtype) * 1e-5
    return torch.exp(-torch.square(s) * gp_gamma) + diag


def build_lr_observations(lr_seq_norm, t_ind_uni):
    return build_paired_lr_observations(lr_seq_norm, t_ind_uni)


def compute_continuous_poest(
    x_0,
    basis_function,
    core_mean,
    core_std,
    core_t,
    y_group,
    ind_conti_group,
    y_time_group,
    y_time_ind_group,
    decoder_bases,
    lr_hw,
    MPDPS=0.4,
):
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


class EDM:
    def __init__(self, model=None, cfg=None):
        self.cfg = cfg
        self.device = cfg.device
        self.model = model.to(self.device)
        self.ema = None
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.rho = cfg.rho
        self.sigma_data = cfg.sigma_data
        self.P_mean = -1.2
        self.P_std = 1.2

    def model_forward_wrapper(self, x, sigma, t, use_ema=False):
        del use_ema
        sigma = sigma.clone()
        sigma[sigma == 0] = self.sigma_min
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        model_output = self.model(
            torch.einsum("b,btijk->btijk", c_in, x),
            c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1),
            t,
        )
        return torch.einsum("b,btijk->btijk", c_skip, x) + torch.einsum(
            "b,btijk->btijk", c_out, model_output
        )

    def __call__(self, x, sigma, t, use_ema=False):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]], device=x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), t.float(), use_ema=use_ema)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


@torch.no_grad()
def edm_post_sampler(
    edm,
    basis_function,
    latents,
    t,
    y_group,
    ind_conti_group,
    y_time_group,
    y_time_ind_group,
    decoder_bases,
    lr_hw,
    num_steps=20,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    zeta=0.009,
    MPDPS=0.4,
):
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])])

    x_next = latents.to(torch.float64) * i_steps[0]
    t_start = time.time()

    for i, (i_cur, i_next) in tqdm(enumerate(zip(i_steps[:-1], i_steps[1:]))):
        x_hat = x_next
        i_hat = i_cur

        denoised = edm(x_hat, i_hat, t).to(torch.float64)
        denoised_core1 = denoised.detach().clone()
        d_cur = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur

        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t).to(torch.float64)
            denoised_core2 = denoised.detach().clone()
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)

            denoised_core = (denoised_core1 + denoised_core2) / 2
            llk_grad = compute_continuous_poest(
                denoised_core,
                basis_function,
                core_mean,
                core_std,
                t,
                y_group,
                ind_conti_group,
                y_time_group,
                y_time_ind_group,
                decoder_bases,
                lr_hw,
                MPDPS=MPDPS,
            )
            x_next = x_next + (zeta / (i + 1)) * llk_grad

    print(f"  sampling done in {time.time() - t_start:.1f}s")
    return x_next


def decoder(u_coords, v_coords, w_coords, core, basis_function):
    u_t = torch.as_tensor(u_coords, dtype=torch.float32, device=device)
    v_t = torch.as_tensor(v_coords, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(w_coords, dtype=torch.float32, device=device)

    basis_function.eval()
    basis_function.mode = "training"
    core = core.to(torch.float32)

    bases = basis_function(input_ind_train=(u_t, v_t, w_t))
    output = torch.einsum("mi,tijk->tmjk", bases[0], core)
    output = torch.einsum("nj,tmjk->tmnk", bases[1], output)
    output = torch.einsum("ok,tmnk->tmno", bases[2], output)
    return output.cpu().detach().numpy()


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
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        augment_dim=9,
        channel_mult_noise=1,
        resample_filter=[1, 1],
    )


def load_split_files(config):
    hr_dir = Path(config.hr_dir)
    lr_dir = Path(config.lr_dir)
    hr_files = sorted(hr_dir.glob(config.file_glob))
    lr_files = sorted(lr_dir.glob(config.file_glob))
    n_available = min(len(hr_files), len(lr_files))
    if n_available == 0:
        raise RuntimeError(f"No paired files found in {hr_dir} and {lr_dir}")

    n_total = min(config.num_total, n_available) if config.num_total > 0 else n_available
    hr_files = hr_files[:n_total]
    lr_files = lr_files[:n_total]

    test_start = config.num_train + config.num_val
    if test_start >= n_total:
        raise RuntimeError(f"test_start={test_start} exceeds n_total={n_total}")

    if config.num_test > 0:
        test_end = min(test_start + config.num_test, n_total)
    else:
        test_end = n_total

    test_indices = list(range(test_start, test_end))
    return hr_files, lr_files, test_indices


def load_sequences(file_list, indices, data_key):
    seqs = []
    for idx in indices:
        seqs.append(np.load(file_list[idx])[data_key].astype(np.float32))
    return np.stack(seqs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--basis_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--core_mean_std_path", type=str, required=True)
    parser.add_argument("--norm_stats_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--file_glob", type=str, default="*.npz")
    parser.add_argument("--data_key", type=str, default="output")
    parser.add_argument("--num_total", type=int, default=-1)
    parser.add_argument("--num_train", type=int, required=True)
    parser.add_argument("--num_val", type=int, required=True)
    parser.add_argument("--num_test", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--target_hr_size", type=int, required=True)

    parser.add_argument("--MPDPS", type=float, default=0.4)
    parser.add_argument("--zeta", type=float, default=0.009)
    parser.add_argument("--total_steps", type=int, default=20)

    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--model_channels", type=int, default=16)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4, 4])
    parser.add_argument("--attn_resolutions", type=int, nargs="+", default=[])
    parser.add_argument("--layers_per_block", type=int, default=4)
    parser.add_argument("--num_temporal_latent", type=int, default=2)

    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    set_seed(config.seed)
    print("Device:", device)

    os.makedirs(config.output_dir, exist_ok=True)

    d = sio.loadmat(config.core_mean_std_path)
    core_mean = torch.tensor(d["core_mean"], dtype=torch.float32).to(device)
    core_std = torch.tensor(d["core_std"], dtype=torch.float32).to(device)

    with open(config.norm_stats_path) as f:
        stats = json.load(f)
    data_min = float(stats["data_min"])
    data_max = float(stats["data_max"])
    print(f"data_min={data_min:.4f}  data_max={data_max:.4f}")

    hr_files, lr_files, test_indices = load_split_files(config)
    print(f"Loading {len(test_indices)} test sequences from npz...")
    print(f"  HR dir: {config.hr_dir}")
    print(f"  LR dir: {config.lr_dir}")
    print(f"  test indices: {test_indices[0]}..{test_indices[-1]}")

    hr_raw = load_sequences(hr_files, test_indices, config.data_key)
    lr_raw = load_sequences(lr_files, test_indices, config.data_key)

    hr_norm = (hr_raw[:, :, np.newaxis, :, :] - data_min) / (data_max - data_min)
    lr_norm = (lr_raw[:, :, np.newaxis, :, :] - data_min) / (data_max - data_min)

    N_test = hr_norm.shape[0]
    T = hr_norm.shape[1]
    lr_hw = tuple(lr_norm.shape[-2:])
    t_ind_uni = np.linspace(0, 1, T, dtype=np.float32)
    print(f"Test sequences: {N_test}, frames: {T}, lr_hw={lr_hw}")

    u_ind_uni = np.array([1.0], dtype=np.float32)
    hr_v_ind = np.linspace(0, 1, config.target_hr_size, dtype=np.float32)
    hr_w_ind = np.linspace(0, 1, config.target_hr_size, dtype=np.float32)

    basis_function = torch.load(config.basis_path, map_location=device)
    basis_function.eval()
    basis_function.mode = "sampling"
    decoder_bases = precompute_decoder_bases(
        basis_function, u_ind_uni, hr_v_ind, hr_w_ind, device=device
    )

    my_net = create_model(config)
    edm = EDM(model=my_net, cfg=config)
    checkpoint = torch.load(config.model_path, map_location=device)
    edm.model.load_state_dict(checkpoint)
    for param in edm.model.parameters():
        param.requires_grad = False
    edm.model.eval()
    print("Models loaded.")

    tracker = EfficiencyTracker(
        model_name="SDIFT",
        dataset_name=config.dataset_name,
        save_dir=config.output_dir,
    )

    print("Warming up (3 iterations)...")
    for warmup_i in range(min(3, N_test)):
        lr_seq_norm = lr_norm[warmup_i]
        y_group, ind_conti_group, y_time_group, y_time_ind_group = build_lr_observations(
            lr_seq_norm, t_ind_uni
        )
        sample_shape = [1, T, config.channels, config.img_size, config.img_size]
        t_grid = torch.linspace(0, 1, T).view(1, -1, 1).to(device).double()
        cov_sample = get_gp_covariance(t_grid)
        L_sample = torch.linalg.cholesky(cov_sample).to(device)
        noise_sample = torch.randn(sample_shape, device=device).double()
        x_T = (L_sample @ noise_sample.view(1, T, -1)).view(sample_shape)
        _ = edm_post_sampler(
            edm,
            basis_function,
            x_T,
            t_grid,
            y_group,
            ind_conti_group,
            y_time_group,
            y_time_ind_group,
            decoder_bases,
            lr_hw,
            num_steps=config.total_steps,
            zeta=config.zeta,
            MPDPS=config.MPDPS,
        )

    tracker.start_inference(device=str(device))

    pred_list = []
    gt_list = []
    lr_list = []

    for i in range(N_test):
        print(f"\n[{i + 1}/{N_test}] Sequence {i}")

        lr_seq_norm = lr_norm[i]
        y_group, ind_conti_group, y_time_group, y_time_ind_group = build_lr_observations(
            lr_seq_norm, t_ind_uni
        )

        sample_shape = [1, T, config.channels, config.img_size, config.img_size]
        t_grid = torch.linspace(0, 1, T).view(1, -1, 1).to(device).double()
        cov_sample = get_gp_covariance(t_grid)
        L_sample = torch.linalg.cholesky(cov_sample).to(device)
        noise_sample = torch.randn(sample_shape, device=device).double()
        x_T = (L_sample @ noise_sample.view(1, T, -1)).view(sample_shape)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.time()

        sample = edm_post_sampler(
            edm,
            basis_function,
            x_T,
            t_grid,
            y_group,
            ind_conti_group,
            y_time_group,
            y_time_ind_group,
            decoder_bases,
            lr_hw,
            num_steps=config.total_steps,
            zeta=config.zeta,
            MPDPS=config.MPDPS,
        ).detach()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start_time
        tracker.record_sequence_time(elapsed)

        core_sample = sample * core_std + core_mean
        basis_function.mode = "sampling"
        out_norm = decoder(u_ind_uni, hr_v_ind, hr_w_ind, core_sample[0], basis_function)
        out_phys = out_norm * (data_max - data_min) + data_min

        pred_phys = out_phys.transpose(0, 2, 3, 1)
        gt_phys = hr_raw[i][:, :, :, np.newaxis]
        lr_phys = lr_raw[i][:, :, :, np.newaxis]

        rmse = np.sqrt(np.mean((pred_phys - gt_phys) ** 2))
        print(f"  RMSE (physical): {rmse:.6f}  Time: {elapsed:.4f}s")

        pred_list.append(pred_phys)
        gt_list.append(gt_phys)
        lr_list.append(lr_phys)

    tracker.save_stats(checkpoint_path=config.model_path)

    pred_all = np.concatenate(pred_list, axis=0).astype(np.float32)
    gt_all = np.concatenate(gt_list, axis=0).astype(np.float32)
    lr_all = np.concatenate(lr_list, axis=0).astype(np.float32)

    np.savez(os.path.join(config.output_dir, "pred.npz"), data=pred_all)
    np.savez(os.path.join(config.output_dir, "gt.npz"), data=gt_all)
    np.savez(os.path.join(config.output_dir, "lr.npz"), data=lr_all)

    meta = {
        "checkpoint": config.model_path,
        "basis_path": config.basis_path,
        "core_mean_std_path": config.core_mean_std_path,
        "norm_stats_path": config.norm_stats_path,
        "hr_dir": config.hr_dir,
        "lr_dir": config.lr_dir,
        "data_key": config.data_key,
        "n_frames": int(pred_all.shape[0]),
        "frames_per_seq": T,
        "n_seqs": N_test,
        "pred_shape": list(pred_all.shape),
        "norm_stats": {"data_min": data_min, "data_max": data_max},
        "target_hr_size": config.target_hr_size,
        "test_indices": test_indices,
        "inference_fix": "paired_bilinear_forward_model",
    }
    with open(os.path.join(config.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved pred/gt/lr.npz + meta.json to {config.output_dir}")
    print(f"  pred shape: {pred_all.shape}  gt shape: {gt_all.shape}  lr shape: {lr_all.shape}")

    overall_rmse = np.sqrt(np.mean((pred_all - gt_all) ** 2))
    print(f"Overall RMSE: {overall_rmse:.6f}")
