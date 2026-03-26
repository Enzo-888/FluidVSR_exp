# train_FTM_RB.py
# Adapted from train_FTM.py for RayleighBenard VSR task.
#
# Reads RB npz files directly — no preprocessing step needed.
# Saves norm stats (data_min, data_max) alongside the core tensor .mat file.
#
# Key changes vs original:
#   - R = (1, 128, 128) to match HR 128x128 resolution
#   - Full-field supervision (mask = all ones, training on complete HR field)
#   - Loads data directly from npz files
#   - max_iter = 1500

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from FTM_model import Tensor_inr_3D
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as scio
from datetime import datetime
from utils import total_variation_loss
import argparse
import os
import json
import sys

HR_DIR = '/data/yc/dataset/RB/code/rayleigh-benard-32/data1/HR'

N_TRAIN = 96   # sequences 0-95


def loss_fn(pred, gt, mask=None):
    diff = pred - gt
    if mask is not None:
        diff = diff[mask == 1]
    return torch.sqrt(torch.mean(diff ** 2))


def loss_fn2(pred, gt, mask=None):
    diff = torch.abs(pred - gt)
    if mask is not None:
        diff = diff[mask == 1]
    return torch.mean(diff)


def load_rb_train_data(device):
    """Load HR training sequences directly from npz files, return normalized tensor."""
    print("Loading HR training data from npz...")
    hr_list = []
    for i in range(N_TRAIN):
        hr = np.load(f'{HR_DIR}/sample_{i:03d}.npz')['output'].astype(np.float32)  # (100, 128, 128)
        hr_list.append(hr)
    hr_data = np.stack(hr_list)  # [96, 100, 128, 128]

    data_min = float(hr_data.min())
    data_max = float(hr_data.max())
    print(f"data_min={data_min:.4f}  data_max={data_max:.4f}")

    hr_norm = (hr_data - data_min) / (data_max - data_min)
    # Add D=1 dim: [96, 100, 1, 128, 128]
    hr_norm_5d = hr_norm[:, :, np.newaxis, :, :]

    data_tensor = torch.FloatTensor(hr_norm_5d).to(device)
    return data_tensor, data_min, data_max


def training(basis_function, tucker_core, train_loader, optimizer):
    basis_function.train()
    basis_function.mode = "training"
    loss_list = []
    for data, batch_ind in train_loader:
        optimizer.zero_grad()
        basises = basis_function(input_ind_train=ind_input)
        output = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core[batch_ind])
        output = torch.einsum("nj, btmjk->btmnk", basises[1], output)
        output = torch.einsum("ok, btmnk->btmno", basises[2], output)
        loss = loss_fn(output, data) + total_variation_loss(tucker_core[batch_ind], weight=1e-7)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    return np.mean(loss_list)


def evaluating(basis_function, tucker_core, loader):
    basis_function.eval()
    with torch.no_grad():
        for data, batch_ind in loader:
            basises = basis_function(input_ind_train=ind_input)
            output = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core[batch_ind])
            output = torch.einsum("nj, btmjk->btmnk", basises[1], output)
            output = torch.einsum("ok, btmnk->btmno", basises[2], output)
            rmse = loss_fn(output, data)
            mae  = loss_fn2(output, data)
    return rmse.item(), mae.item()


def train_parallel(basis_function, tucker_core, loader, optimizer, max_iter,
                   data_min, data_max, config, tracker):
    os.makedirs('./ckp', exist_ok=True)
    current_time = datetime.now().strftime("%Y_%m_%d_%H")
    R = config.R
    RMSE_min = 10.0

    print("Initial eval:", evaluating(basis_function, tucker_core, loader))

    for iter in tqdm(range(max_iter)):
        basis_function.mode = "training"
        loss = training(basis_function, tucker_core, loader, optimizer)
        if iter % 20 == 0:
            print(f"epoch {iter:4d}  loss: {loss:.6f}")
        if iter > 800 and (loss < getattr(train_parallel, '_loss_min', 10) or iter % 50 == 0):
            train_parallel._loss_min = min(getattr(train_parallel, '_loss_min', 10), loss)
            rmse, mae = evaluating(basis_function, tucker_core, loader)
            if rmse < RMSE_min:
                RMSE_min = rmse
                core_path  = f"./data/core_rb_{R[0]}x{R[1]}x{R[2]}_{current_time}.mat"
                basis_path = f"./ckp/basis_rb_{R[0]}x{R[1]}x{R[2]}_{current_time}.pth"
                stats_path = f"./data/norm_stats_rb_{current_time}.json"
                os.makedirs('./data', exist_ok=True)
                scio.savemat(core_path, {"core": tucker_core.detach().cpu().numpy()})
                torch.save(basis_function, basis_path)
                with open(stats_path, 'w') as f:
                    json.dump({'data_min': data_min, 'data_max': data_max}, f)
                print(f"  [saved] core={core_path}")
                print(f"          basis={basis_path}")
                print(f"          stats={stats_path}")
            print(f"  iter {iter:4d}  eval RMSE: {rmse:.6f}  MAE: {mae:.6f}")

    # Save efficiency stats at end
    if tracker:
        tracker.end_training(total_iters=max_iter)
        tracker.save_stats(checkpoint_path=basis_path)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--R",             type=int, nargs=3, default=[1, 128, 128])
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_iter",      type=int,   default=1500)
    parser.add_argument("--batch_size",    type=int,   default=96)  # all 96 seqs in one shot
    config = parser.parse_args()

    # Load data directly from npz
    data, data_min, data_max = load_rb_train_data(device)
    # data: [96, 100, 1, 128, 128] on GPU

    # Coordinates: u=D(single point), v=H(128), w=W(128)
    u_ind = torch.FloatTensor([1.0]).to(device)
    v_ind = torch.FloatTensor(np.linspace(0, 1, 128, dtype=np.float32)).to(device)
    w_ind = torch.FloatTensor(np.linspace(0, 1, 128, dtype=np.float32)).to(device)
    ind_input = (u_ind, v_ind, w_ind)

    R = config.R
    batch_ind = torch.arange(data.size(0)).to(device)
    tucker_core = (torch.ones(data.size(0), data.size(1), R[0], R[1], R[2]) / 2).to(device)
    tucker_core.requires_grad = True
    basis_function = Tensor_inr_3D(R, omega=20).to(device)

    params = list(basis_function.parameters()) + [tucker_core]
    optimizer = optim.AdamW(params, config.learning_rate)

    loader = DataLoader(TensorDataset(data, batch_ind), batch_size=config.batch_size, shuffle=False)

    # Initialize efficiency tracker
    from efficiency_tracker import EfficiencyTracker

    tracker = EfficiencyTracker(
        model_name='SDIFT_FTM',
        dataset_name='RB',
        save_dir='./exps'
    )
    tracker.start_training(batch_size=config.batch_size, device=str(device))

    train_parallel(basis_function, tucker_core, loader, optimizer, config.max_iter,
                   data_min, data_max, config, tracker)
