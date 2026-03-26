"""
validate_pipeline.py
Quick end-to-end validation on a small subset:
  - 5 training sequences (FTM + GPSD)
  - 2 test sequences (inference)

Purpose: verify FTM reconstruction quality and pipeline correctness
before committing to full training (~1.5h total on one 4090).

All outputs saved to ./validation_output/
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as scio
import copy, os, json, time, random
import matplotlib.pyplot as plt
from tqdm import tqdm

from FTM_model import Tensor_inr_3D
from networks_edm import Spatial_temporal_UNet
from utils import total_variation_loss

# ============================================================
# Config
# ============================================================
HR_DIR   = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/HR'
LR_DIR   = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/LR'
OUT_DIR  = '/data/yc/SDIFT-main/validation_output'

TRAIN_IDS = list(range(5))      # sequences 0-4
TEST_IDS  = [108, 109]          # 2 test sequences

FTM_ITER     = 500              # ~20-30 min; watch RMSE curve
GPSD_STEPS   = 2000             # ~15 min; enough to check loss decrease
INFER_STEPS  = 20               # same as production

R = [1, 128, 128]               # Tucker rank
DEVICE = torch.device('cuda:0')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/ckp', exist_ok=True)

# ============================================================
# Helpers
# ============================================================
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# 1. Load data
# ============================================================
print("=" * 60)
print("Step 1: Loading data")
print("=" * 60)

hr_list, lr_list = [], []
for i in TRAIN_IDS:
    hr_list.append(np.load(f'{HR_DIR}/sample_{i:03d}.npz')['output'].astype(np.float32))
hr_train = np.stack(hr_list)    # [5, 100, 128, 128]

hr_te, lr_te = [], []
for i in TEST_IDS:
    hr_te.append(np.load(f'{HR_DIR}/sample_{i:03d}.npz')['output'].astype(np.float32))
    lr_te.append(np.load(f'{LR_DIR}/sample_{i:03d}.npz')['output'].astype(np.float32))
hr_te = np.stack(hr_te)    # [2, 100, 128, 128]
lr_te = np.stack(lr_te)    # [2, 100, 32, 32]

data_min = float(hr_train.min())
data_max = float(hr_train.max())
print(f"  train seqs: {TRAIN_IDS}  test seqs: {TEST_IDS}")
print(f"  data_min={data_min:.4f}  data_max={data_max:.4f}")

hr_norm   = (hr_train - data_min) / (data_max - data_min)
hr_norm5d = hr_norm[:, :, np.newaxis, :, :]          # [5, 100, 1, 128, 128]
hr_te_norm = (hr_te - data_min) / (data_max - data_min)
lr_te_norm = (lr_te - data_min) / (data_max - data_min)

data_t    = torch.FloatTensor(hr_norm5d).to(DEVICE)
batch_ind = torch.arange(len(TRAIN_IDS)).to(DEVICE)

# Coordinates
u_ind = torch.FloatTensor([1.0]).to(DEVICE)
v_ind = torch.FloatTensor(np.linspace(0, 1, 128, dtype=np.float32)).to(DEVICE)
w_ind = torch.FloatTensor(np.linspace(0, 1, 128, dtype=np.float32)).to(DEVICE)
ind_input = (u_ind, v_ind, w_ind)

# ============================================================
# 2. FTM Training
# ============================================================
print("\n" + "=" * 60)
print(f"Step 2: FTM Training  ({FTM_ITER} iterations, {len(TRAIN_IDS)} sequences)")
print("=" * 60)

tucker_core = (torch.ones(len(TRAIN_IDS), 100, *R) / 2).to(DEVICE)
tucker_core.requires_grad = True
basis_fn = Tensor_inr_3D(R, omega=20).to(DEVICE)
params   = list(basis_fn.parameters()) + [tucker_core]
opt_ftm  = optim.AdamW(params, lr=2e-4)

loader = DataLoader(TensorDataset(data_t, batch_ind), batch_size=len(TRAIN_IDS), shuffle=False)

ftm_rmse_log = []
best_ftm_rmse = 1e9
t0 = time.time()

for it in tqdm(range(FTM_ITER), desc="FTM"):
    basis_fn.train(); basis_fn.mode = "training"
    for data, bidx in loader:
        opt_ftm.zero_grad()
        bs = basis_fn(input_ind_train=ind_input)
        out = torch.einsum("mi,btijk->btmjk", bs[0], tucker_core[bidx])
        out = torch.einsum("nj,btmjk->btmnk", bs[1], out)
        out = torch.einsum("ok,btmnk->btmno", bs[2], out)
        loss = torch.sqrt(torch.mean((out - data) ** 2)) + total_variation_loss(tucker_core[bidx], 1e-7)
        loss.backward(); opt_ftm.step()

    if (it + 1) % 50 == 0 or it == FTM_ITER - 1:
        basis_fn.eval()
        with torch.no_grad():
            bs = basis_fn(input_ind_train=ind_input)
            out = torch.einsum("mi,btijk->btmjk", bs[0], tucker_core)
            out = torch.einsum("nj,btmjk->btmnk", bs[1], out)
            out = torch.einsum("ok,btmnk->btmno", bs[2], out)
            ev_rmse = float(torch.sqrt(torch.mean((out - data_t) ** 2)).item())
        ftm_rmse_log.append((it + 1, ev_rmse))
        print(f"  iter {it+1:4d}  RMSE={ev_rmse:.6f}")
        if ev_rmse < best_ftm_rmse:
            best_ftm_rmse = ev_rmse
            scio.savemat(f'{OUT_DIR}/core_val.mat', {"core": tucker_core.detach().cpu().numpy()})
            torch.save(basis_fn, f'{OUT_DIR}/ckp/basis_val.pth')

ftm_time = time.time() - t0
print(f"\nFTM done in {ftm_time/60:.1f} min  best RMSE={best_ftm_rmse:.6f}")

# Free GPU memory before GPSD (FTM optimizer states take several GB cached)
del tucker_core, basis_fn, data_t, opt_ftm, loader, params, bs, out, loss
torch.cuda.empty_cache()
print(f"GPU memory after FTM cleanup: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB allocated")

# Save RMSE curve
iters, rmses = zip(*ftm_rmse_log)
plt.figure(); plt.plot(iters, rmses, 'b-o')
plt.xlabel('iter'); plt.ylabel('RMSE'); plt.title('FTM Training RMSE')
plt.grid(True); plt.savefig(f'{OUT_DIR}/ftm_rmse_curve.png', dpi=100)
plt.close()

print(f"\n>>> FTM reconstruction RMSE: {best_ftm_rmse:.6f}")
if best_ftm_rmse < 0.05:
    print(">>> [OK] Tucker decomposition quality is good — proceed with full training.")
elif best_ftm_rmse < 0.10:
    print(">>> [WARN] Moderate RMSE. Consider more FTM iterations or larger rank.")
else:
    print(">>> [FAIL] High RMSE. Tucker rank R2=R3=128 may be insufficient. "
          "Try R=(1,192,192) or deeper basis network.")

with open(f'{OUT_DIR}/norm_stats.json', 'w') as f:
    json.dump({'data_min': data_min, 'data_max': data_max}, f)

# ============================================================
# 3. GPSD Training (mini)
# ============================================================
print("\n" + "=" * 60)
print(f"Step 3: GPSD Training  ({GPSD_STEPS} steps)")
print("=" * 60)

# Load best core
import scipy.io as sio
d = sio.loadmat(f'{OUT_DIR}/core_val.mat')
core = torch.tensor(d['core'], dtype=torch.float32).to(DEVICE)
core_mean_val = core.min()
core_std_val  = core.max() - core.min()
core_norm = (core - core_mean_val) / core_std_val
print(f"  core shape: {core_norm.shape}  norm min={core_norm.min():.3f} max={core_norm.max():.3f}")

t_seq = (torch.linspace(0, 1, 100).view(1, -1, 1).to(DEVICE)).repeat(len(TRAIN_IDS), 1, 1)
gpsd_dataset = TensorDataset(core_norm, t_seq)
gpsd_loader  = DataLoader(gpsd_dataset, batch_size=1, shuffle=True)  # 1 seq at a time; 100T×128×128 already large

def get_gp_covariance(t, gp_gamma=50):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5
    return torch.exp(-s ** 2 * gp_gamma) + diag

unet = Spatial_temporal_UNet(
    in_channels=1, out_channels=1,
    num_blocks=2, num_temporal_latent=1,   # reduced for validation (avoids OOM at 128x128)
    attn_resolutions=[], model_channels=16,  # 16 for validation; use 32 for full training
    channel_mult=[1, 2, 4, 4], dropout=0,
    img_resolution=128, label_dim=0,
    embedding_type='positional', encoder_type='standard',
    decoder_type='standard', augment_dim=9,
    channel_mult_noise=1, resample_filter=[1, 1],
)
unet.to(DEVICE).train()
ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
opt_gpsd  = torch.optim.Adam(unet.parameters(), lr=2e-4)

sigma_data = 0.5
P_mean, P_std = -1.2, 1.2
WARMUP = 500
gpsd_loss_log = []
t0 = time.time()

for step in tqdm(range(GPSD_STEPS), desc="GPSD"):
    try: sig_b, t_b = next(data_iter)
    except: data_iter = iter(gpsd_loader); sig_b, t_b = next(data_iter)

    opt_gpsd.zero_grad()
    rnd = torch.randn([sig_b.shape[0]], device=DEVICE)
    sigma  = (rnd * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    cov    = get_gp_covariance(t_b)
    L      = torch.linalg.cholesky(cov)
    noise  = torch.randn_like(sig_b)
    noise  = (L @ noise.view(sig_b.shape[0], sig_b.shape[1], -1)).view(sig_b.shape)
    n      = torch.einsum('b,btijk->btijk', sigma, noise)
    yn     = sig_b + n

    c_skip  = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out   = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in    = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
    c_noise = sigma.log() / 4

    x_in  = torch.einsum('b,btijk->btijk', c_in, yn)
    noise_labels = c_noise.view(-1, 1, 1).repeat(1, t_b.shape[1], 1)
    pred  = unet(x_in, noise_labels, t_b)
    D_yn  = torch.einsum('b,btijk->btijk', c_skip, yn) + torch.einsum('b,btijk->btijk', c_out, pred)
    loss  = torch.einsum('b,btijk->btijk', weight, (D_yn - sig_b) ** 2).mean()
    loss.backward()

    lr_now = 2e-4 * min(step / WARMUP, 1)
    for g in opt_gpsd.param_groups: g['lr'] = lr_now
    for p in unet.parameters():
        if p.grad is not None:
            torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
    opt_gpsd.step()

    # EMA
    ema_beta = 0.5 ** (len(TRAIN_IDS) / max(500_000 * 0.05, 1e-8))
    for pe, pn in zip(ema_model.parameters(), unet.parameters()):
        pe.copy_(pn.detach().lerp(pe, ema_beta))

    if (step + 1) % 200 == 0 or step == GPSD_STEPS - 1:
        gpsd_loss_log.append((step + 1, loss.item()))
        print(f"  step {step+1:4d}  loss={loss.item():.6f}  lr={lr_now:.6f}")

torch.save(unet.state_dict(), f'{OUT_DIR}/ckp/gpsd_val.pth')
scio.savemat(f'{OUT_DIR}/core_mean_std_val.mat',
             {"core_mean": core_mean_val.cpu().numpy(),
              "core_std":  core_std_val.cpu().numpy()})
gpsd_time = time.time() - t0
print(f"\nGPSD done in {gpsd_time/60:.1f} min")

steps_g, losses_g = zip(*gpsd_loss_log)
plt.figure(); plt.plot(steps_g, losses_g, 'r-o')
plt.xlabel('step'); plt.ylabel('loss'); plt.title('GPSD Training Loss')
plt.grid(True); plt.savefig(f'{OUT_DIR}/gpsd_loss_curve.png', dpi=100)
plt.close()

# ============================================================
# 4. Inference on 2 test sequences
# ============================================================
print("\n" + "=" * 60)
print(f"Step 4: Inference  ({len(TEST_IDS)} test sequences, {INFER_STEPS} ODE steps)")
print("=" * 60)

unet.eval()
for p in unet.parameters(): p.requires_grad = False

def get_ktT(y_tt, core_t, gp_gamma=1):
    r = torch.sqrt((y_tt - core_t) ** 2)
    return torch.exp(-r ** 2 * gp_gamma)

def get_kTT_inv(t, gp_gamma=1):
    r = torch.sqrt((t - t.transpose(-1, -2)) ** 2)
    K = torch.exp(-r ** 2 * gp_gamma) + torch.eye(t.shape[-2]).to(t) * 1e-3
    return torch.inverse(K)

def build_lr_obs(lr_seq_norm, t_ind):
    T = lr_seq_norm.shape[0]
    lr_v = np.linspace(0, 1, 32, dtype=np.float32)
    lr_w = np.linspace(0, 1, 32, dtype=np.float32)
    vv, ww = np.meshgrid(lr_v, lr_w, indexing='ij')
    uu = np.ones_like(vv)
    lr_coords = np.stack([uu.ravel(), vv.ravel(), ww.ravel()], axis=1)  # [1024, 3]
    y_group, ind_group, yt_group, yti_group = [], [], [], []
    for t in range(T):
        y_group.append(lr_seq_norm[t, 0, :, :].ravel().astype(np.float64))
        ind_group.append(lr_coords)
        yt_group.append(float(t_ind[t]))
        yti_group.append(t)
    return y_group, ind_group, yt_group, yti_group

def compute_posterior(x_0, basis_fn, core_mean, core_std, core_t,
                      y_group, ind_group, yt_group, yti_group, MPDPS=0.4):
    shape = x_0.shape
    x_vec = x_0.view(shape[0], shape[1], -1)
    pm1 = torch.zeros_like(x_vec)
    pm2 = torch.zeros_like(x_vec)
    x_d = x_vec * core_std + core_mean
    for y, y_tt, y_t_ind, ind in zip(y_group, yt_group, yti_group, ind_group):
        if len(y) == 0: continue
        x_t = x_d[0, y_t_ind, :]
        y_t = torch.DoubleTensor(y).to(DEVICE)
        A   = basis_fn(input_ind_sampl=torch.FloatTensor(ind).to(DEVICE)).detach().double()
        pm1[0, y_t_ind, :] = (A.T @ (y_t - A @ x_t)).float()

        t_rem = yti_group.copy(); t_rem.remove(y_t_ind)
        if not t_rem: continue
        ct_rem = core_t[:, t_rem, :]
        xr     = x_d[:, t_rem, :]
        ktT    = get_ktT(y_tt, ct_rem).squeeze(2)
        Ki     = get_kTT_inv(ct_rem)
        coeff  = (ktT @ Ki).to(DEVICE)
        if coeff.dim() == 1: coeff = coeff.unsqueeze(0)
        x_agg  = (coeff @ xr).squeeze()
        post   = (A.T @ (y_t - A @ x_agg.double())).float()
        temp   = torch.zeros_like(x_vec)
        # outer product: [1, len(t_rem), 1] * [1, 1, 16384] → [1, len(t_rem), 16384]
        temp[:, t_rem, :] = coeff.unsqueeze(-1) * post.float().view(1, 1, -1)
        pm2 += temp
    return (pm1 + MPDPS * pm2).view(shape)

def edm_call(x, sigma, t):
    if sigma.shape == torch.Size([]): sigma = sigma * torch.ones([x.shape[0]]).to(DEVICE)
    sigma_data_v = 0.5
    c_skip  = sigma_data_v ** 2 / (sigma ** 2 + sigma_data_v ** 2)
    c_out   = sigma * sigma_data_v / (sigma ** 2 + sigma_data_v ** 2).sqrt()
    c_in    = 1 / (sigma_data_v ** 2 + sigma ** 2).sqrt()
    c_noise = sigma.log() / 4
    x_in = torch.einsum('b,btijk->btijk', c_in.float(), x.float())
    nl   = c_noise.float().view(-1, 1, 1).repeat(1, t.shape[1], 1)
    pred = unet(x_in, nl, t.float())
    return torch.einsum('b,btijk->btijk', c_skip.float(), x.float()) + \
           torch.einsum('b,btijk->btijk', c_out.float(), pred)

basis_fn = torch.load(f'{OUT_DIR}/ckp/basis_val.pth', map_location=DEVICE)
basis_fn.eval(); basis_fn.mode = "sampling"

core_mean_v = torch.tensor(sio.loadmat(f'{OUT_DIR}/core_mean_std_val.mat')['core_mean'], dtype=torch.float32).to(DEVICE)
core_std_v  = torch.tensor(sio.loadmat(f'{OUT_DIR}/core_mean_std_val.mat')['core_std'],  dtype=torch.float32).to(DEVICE)

t_ind_uni = np.linspace(0, 1, 100, dtype=np.float32)
hr_v = np.linspace(0, 1, 128, dtype=np.float32)
hr_w = np.linspace(0, 1, 128, dtype=np.float32)

pred_list, gt_list, lr_list = [], [], []
sigma_min, sigma_max, rho_v = 0.002, 80.0, 7.0

for i in range(len(TEST_IDS)):
    print(f"\n  Sequence {TEST_IDS[i]}")
    lr_seq = lr_te_norm[i]   # [100, 32, 32]  → add dim
    lr_seq_5d = lr_seq[:, np.newaxis, :, :]   # [100, 1, 32, 32]
    y_group, ind_group, yt_group, yti_group = build_lr_obs(lr_seq_5d, t_ind_uni)

    T = 100
    sample_shape = [1, T, 1, 128, 128]
    t_grid = (torch.linspace(0, 1, T).view(1, -1, 1).to(DEVICE)).repeat(1, 1, 1).double()
    cov   = get_gp_covariance(t_grid)
    L_gp  = torch.linalg.cholesky(cov).to(DEVICE)
    noise = torch.randn(sample_shape).to(DEVICE).double()
    x_T   = (L_gp @ noise.view(1, T, -1)).view(sample_shape)

    # ODE steps
    step_idx = torch.arange(INFER_STEPS, dtype=torch.float64, device=DEVICE)
    i_steps  = (sigma_max ** (1/rho_v) + step_idx / (INFER_STEPS-1) * (sigma_min**(1/rho_v) - sigma_max**(1/rho_v))) ** rho_v
    i_steps  = torch.cat([i_steps, torch.zeros_like(i_steps[:1])])
    x_next   = x_T * i_steps[0]

    t0_inf = time.time()
    for si, (i_cur, i_next) in enumerate(zip(i_steps[:-1], i_steps[1:])):
        x_hat = x_next; i_hat = i_cur
        with torch.no_grad():
            den1 = edm_call(x_hat, i_hat, t_grid).double()
        d_cur  = (x_hat - den1) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur
        if si < INFER_STEPS - 1:
            with torch.no_grad():
                den2 = edm_call(x_next, i_next, t_grid).double()
            d_prime = (x_next - den2) / i_next
            x_next  = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)
            dc_avg  = (den1 + den2) / 2
            llk = compute_posterior(dc_avg, basis_fn, core_mean_v, core_std_v, t_grid,
                                    y_group, ind_group, yt_group, yti_group)
            x_next = x_next + (0.009 / (si + 1)) * llk

    print(f"    inference: {time.time()-t0_inf:.1f}s")

    # Decode
    core_samp = (x_next * core_std_v + core_mean_v)[0]  # [T, 1, 128, 128]
    basis_fn.mode = "training"
    u_t = torch.FloatTensor([1.0]).to(DEVICE)
    v_t = torch.FloatTensor(hr_v).to(DEVICE)
    w_t = torch.FloatTensor(hr_w).to(DEVICE)
    core_samp = core_samp.to(torch.float32)
    bs = basis_fn(input_ind_train=(u_t, v_t, w_t))
    out = torch.einsum("mi,tijk->tmjk", bs[0], core_samp)
    out = torch.einsum("nj,tmjk->tmnk", bs[1], out)
    out = torch.einsum("ok,tmnk->tmno", bs[2], out)
    out_norm = out.cpu().detach().numpy()  # [T, 1, 128, 128]

    # Denormalize → physical
    out_phys = out_norm * (data_max - data_min) + data_min   # [T, 1, 128, 128]
    pred_phys = out_phys.transpose(0, 2, 3, 1)               # [T, 128, 128, 1]
    gt_phys   = hr_te[i][:, :, :, np.newaxis]                # [T, 128, 128, 1]
    lr_phys   = lr_te[i][:, :, :, np.newaxis]                # [T, 32, 32, 1]

    r = rmse(pred_phys, gt_phys)
    print(f"    RMSE (physical): {r:.6f}")
    pred_list.append(pred_phys)
    gt_list.append(gt_phys)
    lr_list.append(lr_phys)
    basis_fn.mode = "sampling"

# Save
pred_all = np.concatenate(pred_list, axis=0).astype(np.float32)
gt_all   = np.concatenate(gt_list,   axis=0).astype(np.float32)
lr_all   = np.concatenate(lr_list,   axis=0).astype(np.float32)
np.savez(f'{OUT_DIR}/pred.npz', data=pred_all)
np.savez(f'{OUT_DIR}/gt.npz',   data=gt_all)
np.savez(f'{OUT_DIR}/lr.npz',   data=lr_all)

overall = rmse(pred_all, gt_all)
print(f"\nOverall RMSE (physical): {overall:.6f}")
print(f"pred/gt/lr.npz saved to {OUT_DIR}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"  FTM best RMSE (normalized) : {best_ftm_rmse:.6f}  ({ftm_time/60:.1f} min)")
print(f"  GPSD final loss            : {gpsd_loss_log[-1][1]:.6f}  ({gpsd_time/60:.1f} min)")
print(f"  Inference RMSE (physical)  : {overall:.6f}")
print(f"  RMSE curve plot            : {OUT_DIR}/ftm_rmse_curve.png")
print(f"  Loss curve plot            : {OUT_DIR}/gpsd_loss_curve.png")
print(f"  Outputs                    : {OUT_DIR}/pred|gt|lr.npz")

if best_ftm_rmse < 0.05:
    print("\n[VERDICT] FTM reconstruction OK → safe to run full training.")
elif best_ftm_rmse < 0.10:
    print("\n[VERDICT] FTM RMSE moderate → may need more iterations in full training.")
else:
    print("\n[VERDICT] FTM RMSE high → consider increasing Tucker rank before full training.")
