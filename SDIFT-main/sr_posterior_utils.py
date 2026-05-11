import numpy as np
import torch
import torch.nn.functional as F


def build_lr_observations(lr_seq_norm, t_ind_uni):
    """Build per-frame LR observations for paired SR datasets."""
    num_frames = lr_seq_norm.shape[0]
    y_group = []
    y_time_group = []
    y_time_ind_group = []

    for frame_idx in range(num_frames):
        y_group.append(lr_seq_norm[frame_idx].astype(np.float64, copy=False))
        y_time_group.append(float(t_ind_uni[frame_idx]))
        y_time_ind_group.append(frame_idx)

    # Keep the legacy 4-tuple return shape so the inference scripts need fewer edits.
    return y_group, None, y_time_group, y_time_ind_group


def precompute_decoder_bases(basis_function, u_coords, v_coords, w_coords, device, dtype=torch.float64):
    """Precompute full-grid basis matrices for differentiable HR decoding."""
    prev_mode = getattr(basis_function, "mode", None)
    basis_function.eval()
    basis_function.mode = "training"

    with torch.no_grad():
        u_t = torch.as_tensor(u_coords, dtype=torch.float32, device=device)
        v_t = torch.as_tensor(v_coords, dtype=torch.float32, device=device)
        w_t = torch.as_tensor(w_coords, dtype=torch.float32, device=device)
        bases = basis_function(input_ind_train=(u_t, v_t, w_t))

    if prev_mode is not None:
        basis_function.mode = prev_mode

    return tuple(base.detach().to(device=device, dtype=dtype) for base in bases)


def decode_core_to_hr(core_frame_denorm, decoder_bases):
    """Decode one denormalized Tucker core frame to an HR field."""
    u_basis, v_basis, w_basis = decoder_bases
    output = torch.einsum("mi,ijk->mjk", u_basis, core_frame_denorm)
    output = torch.einsum("nj,mjk->mnk", v_basis, output)
    output = torch.einsum("ok,mnk->mno", w_basis, output)
    return output


def apply_bilinear_degradation(hr_frame, lr_hw):
    """Apply the true paired-SR forward model used by RBC/SW/ERA5 bilinear datasets."""
    return F.interpolate(
        hr_frame.unsqueeze(0).float(),
        size=lr_hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0).to(dtype=hr_frame.dtype)


def compute_bilinear_sr_posterior_grad(
    x_0,
    decoder_bases,
    core_mean,
    core_std,
    core_t,
    y_group,
    y_time_group,
    y_time_ind_group,
    lr_hw,
    get_ktT_fn,
    get_kTT_inv_fn,
    MPDPS=0.4,
):
    """
    Compute the log-likelihood gradient for paired SR where LR is produced by
    bilinear downsampling of the decoded HR field.
    """
    device = x_0.device
    core_shape = x_0.shape
    x_0_vec = x_0.view(core_shape[0], core_shape[1], -1)
    poest_matrix1 = torch.zeros_like(x_0_vec, dtype=torch.float64, device=device)
    poest_matrix2 = torch.zeros_like(x_0_vec, dtype=torch.float64, device=device)

    core_mean = core_mean.to(device=device, dtype=torch.float64).reshape(1)
    core_std = core_std.to(device=device, dtype=torch.float64).reshape(1)

    with torch.enable_grad():
        for y_obs, y_tt, y_t_ind in zip(y_group, y_time_group, y_time_ind_group):
            y_tensor = torch.as_tensor(y_obs, dtype=torch.float64, device=device)

            core_t_norm = x_0_vec[0, y_t_ind].detach().clone().requires_grad_(True)
            core_t_denorm = (core_t_norm * core_std + core_mean).view(*core_shape[2:])
            pred_hr = decode_core_to_hr(core_t_denorm, decoder_bases)
            pred_lr = apply_bilinear_degradation(pred_hr, lr_hw)
            log_likelihood = -0.5 * (pred_lr - y_tensor).pow(2).sum()
            grad_t = torch.autograd.grad(log_likelihood, core_t_norm)[0]
            poest_matrix1[0, y_t_ind, :] = grad_t

            t_remove_group = y_time_ind_group.copy()
            t_remove_group.remove(y_t_ind)
            if not t_remove_group:
                continue

            core_t_remove = core_t[:, t_remove_group, :]
            ktT = get_ktT_fn(y_tt, core_t_remove).squeeze(2)
            KTT_inv = get_kTT_inv_fn(core_t_remove)
            coeff = (ktT @ KTT_inv).to(device=device, dtype=torch.float64).squeeze(1)
            if coeff.dim() == 1:
                coeff = coeff.unsqueeze(0)

            core_remove_norm = x_0_vec[:, t_remove_group, :].detach().clone().requires_grad_(True)
            core_agg_norm = torch.matmul(coeff, core_remove_norm[0])
            core_agg_denorm = (core_agg_norm.squeeze(0) * core_std + core_mean).view(*core_shape[2:])
            pred_hr_agg = decode_core_to_hr(core_agg_denorm, decoder_bases)
            pred_lr_agg = apply_bilinear_degradation(pred_hr_agg, lr_hw)
            log_likelihood_agg = -0.5 * (pred_lr_agg - y_tensor).pow(2).sum()
            grad_remove = torch.autograd.grad(log_likelihood_agg, core_remove_norm)[0]
            poest_matrix2[:, t_remove_group, :] += grad_remove

    return (poest_matrix1 + MPDPS * poest_matrix2).view(core_shape)
