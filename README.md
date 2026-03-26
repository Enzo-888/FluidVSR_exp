# Fluid-VSR: A Unified Benchmark for Fluid Video Super-Resolution

A research framework for evaluating and comparing video super-resolution methods on scientific fluid dynamics datasets. It integrates 6 internal models and 4 external video SR baselines, with unified evaluation and efficiency tracking.

## Overview

This project consists of:

- **Fluid_VSR** (main framework): trains and evaluates 6 image-based SR models
- **4 external repos** (video SR baselines): VRT, RVRT, BasicVSR++, SDIFT

All models are benchmarked on 4 fluid dynamics datasets with a unified evaluation pipeline.

## Datasets

| Dataset | HR Resolution | LR Resolution | Scale | Sequences | Frames/Seq | Split (train/val/test) |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| RayleighBenard (RB) | 128x128 | 32x32 | 4x | 120 | 100 | 96/12/12 |
| ShallowWater (SW) | 128x128 | 32x32 | 4x | 200 | 72 | 160/20/20 |
| KolmogorovFlow (KF256) | 256x256 | 64x64 | 4x | 70 | 180 | 56/7/7 |
| ERA5V (ERA5 wind) | 256x256 | 64x64 | 4x | 578 | 24 | 566/6/6 |

Data paths:

```
dataset/
  RB/code/rayleigh-benard-32/data1/   HR/ LR/   (sample_000.npz ~ sample_119.npz)
  RB/code/shallow-water-32/data1/     HR/ LR/   (sample_000.npz ~ sample_199.npz)
  KF256_data/x4/                      HR/ LR/   (trajectory_000.npz ~ trajectory_069.npz)
  era5_wind_200hPa_daily_cut/         HR/ LR/   (20160201.npz ~ 20171031.npz)
```

---

## 1. Fluid_VSR (Internal Framework)

### Models

FNO, EDSR, SwinIR, SRNO, ResShift, ReMD

### Code Structure

```
Fluid_VSR/
  main.py / main_ddp.py          # training entry points
  config.py                       # CLI arg parser
  datasets/                       # dataset loaders (RB, SW, KF256, ERA5V, ...)
  models/                         # model definitions
  trainers/                       # training logic per model type
  forecastors/                    # inference wrappers
  template_configs/               # YAML configs per dataset per model
    RayleighBenard/  ShallowWater/  KolmogorovFlow/  ERA5V/
  template_notebook/              # eval & efficiency notebooks
  tools/                          # save_predictions.py, eval_metrics.py
  efficiency_stats/               # unified efficiency JSON directory
  vsr_baseline/                   # external model prediction outputs (npz)
  logs/                           # training logs & checkpoints
```

### Training

Single GPU:
```bash
python main.py --config template_configs/RayleighBenard/fno.yaml
```

Multi-GPU (DDP):
```bash
torchrun --nproc_per_node=4 main_ddp.py --config template_configs/RayleighBenard/fno.yaml
```

Config YAML controls everything: model architecture, dataset paths, batch size, learning rate, log directory, etc. Each dataset directory under `template_configs/` has 6 YAML files (edsr, fno, swinIR, sronet, resshift, remd).

### Inference

Inference is done via the eval notebook (see Section 3) or via CLI tools:

```bash
python tools/save_predictions.py --model_dir logs/RayleighBenard/FNO2d_10_07_25
python tools/eval_metrics.py --pred pred.npz --gt gt.npz --metrics basic fid fvd
```

---

## 2. External Models

All external models share the same `efficiency_tracker.py`, which automatically syncs efficiency JSON to `Fluid_VSR/efficiency_stats/`.

### 2.1 VRT (`KAIR-master/`)

**Training:**
```bash
# Single GPU
python main_train_vrt.py --opt options/vrt/010_train_vrt_videosr_rb.json

# Multi-GPU (DDP, handled internally by KAIR framework)
python -m torch.distributed.launch --nproc_per_node=4 main_train_vrt.py \
    --opt options/vrt/010_train_vrt_videosr_rb.json
```

Config files:
- `options/vrt/010_train_vrt_videosr_rb.json`
- `options/vrt/011_train_vrt_videosr_kf256.json`
- `options/vrt/012_train_vrt_videosr_sw.json`

Outputs: `experiments/<config_name>/models/` (checkpoints), `experiments/<config_name>/` (logs)

**Inference:**
```bash
python main_test_vrt_rb.py \
    --opt options/vrt/010_train_vrt_videosr_rb.json \
    --checkpoint experiments/010_train_vrt_videosr_rb/models/200000_G.pth

python main_test_vrt_sw.py \
    --opt options/vrt/012_train_vrt_videosr_sw.json \
    --checkpoint experiments/012_train_vrt_videosr_sw/models/200000_G.pth

python main_test_vrt_kf256.py \
    --opt options/vrt/011_train_vrt_videosr_kf256.json \
    --checkpoint experiments/011_train_vrt_videosr_kf256/models/200000_G.pth
```

Outputs: `experiments/<config_name>/test_predictions/` containing `pred.npz`, `gt.npz`, `lr.npz`, `meta.json`

### 2.2 RVRT (`RVRT-main/`)

**Training:**
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python main_train_rvrt_rb.py

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_train_rvrt_rb.py
```

Per-dataset scripts: `main_train_rvrt_rb.py`, `main_train_rvrt_kf256.py`, `main_train_rvrt_sw.py`

Outputs: `experiments/rvrt_<dataset>/` (checkpoints + logs)

**Inference:**
```bash
python main_test_rvrt_rb.py --checkpoint experiments/rvrt_rb/20000_G.pth
python main_test_rvrt_sw.py --checkpoint experiments/rvrt_sw/20000_G.pth
python main_test_rvrt_kf256.py --checkpoint experiments/rvrt_kf256/20000_G.pth
```

Outputs: `experiments/rvrt_<dataset>/test_predictions/` containing `pred.npz`, `gt.npz`, `lr.npz`, `meta.json`

### 2.3 BasicVSR++ (`BasicVSR_PlusPlus-master/`)

**Training:**
```bash
python tools/train.py configs/basicvsr_plusplus_rb.py --work-dir work_dirs/basicvsr_pp_rb
```

Per-dataset configs: `configs/basicvsr_plusplus_rb.py`, `configs/basicvsr_plusplus_sw.py`, `configs/basicvsr_plusplus_kf256.py`

**Inference:**
```bash
python tools/test_rb.py configs/basicvsr_plusplus_rb.py work_dirs/basicvsr_pp_rb/latest.pth
python tools/test_sw.py configs/basicvsr_plusplus_sw.py work_dirs/basicvsr_pp_sw/latest.pth
```

Outputs: `<work_dir>/test_predictions/` containing `pred.npz`, `gt.npz`, `lr.npz`, `meta.json`

### 2.4 SDIFT (`SDIFT-main/`)

SDIFT has a 3-stage pipeline: FTM (basis learning) -> GPSD (diffusion training) -> Inference (posterior sampling).

**Stage 1 - FTM (Tucker decomposition):**
```bash
python train_FTM_RB.py --R 1 128 128 --max_iter 1500 --batch_size 16
```

Outputs in `data/` and `ckp/`:
- `data/core_<dataset>_<R>_<timestamp>.mat` (Tucker core)
- `ckp/basis_<dataset>_<R>_<timestamp>.pth` (basis functions)
- `data/norm_stats_<dataset>_<timestamp>.json` (min/max normalization)

**Stage 2 - GPSD (diffusion model training):**
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train_GPSD_RB.py --core_path ./data/core_rb_1x128x128_*.mat

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_GPSD_RB.py \
    --core_path ./data/core_rb_1x128x128_*.mat
```

Outputs in `exps/gp-edm_<dataset>_<timestamp>/`:
- `checkpoints/ema_<step>.pth` (model weights)
- `core_mean_std.mat` (normalization stats for core)

**Stage 3 - Inference (MPDPS posterior sampling):**
```bash
python inference_RB.py \
    --basis_path ./ckp/basis_rb_*.pth \
    --model_path ./exps/gp-edm_rb_*/checkpoints/ema_15000.pth \
    --core_mean_std_path ./exps/gp-edm_rb_*/core_mean_std.mat \
    --norm_stats_path ./data/norm_stats_rb_*.json \
    --output_dir ./output_rb
```

Per-dataset scripts: `train_FTM_{RB,SW,KF256}.py`, `train_GPSD_{RB,SW,KF256}.py`, `inference_{RB,SW,KF256}.py`

Outputs: `output_<dataset>/` containing `pred.npz`, `gt.npz`, `lr.npz`, `meta.json`

---

## 3. Unified Evaluation

### eval.ipynb (`template_notebook/eval.ipynb`)

A single notebook that evaluates all models (internal + external) on any dataset.

**How to use:**
1. Set `GPU_ID` in the first cell
2. Set `DATASET_NAME` to `'RB'` / `'ShallowWater'` / `'KF256'` / `'ERA5V'`
3. Configure the `models` list:
   - Internal models: set `type: 'fluid_vsr'` with `model_dir` pointing to the training log directory
   - External models: set `type: 'external'` with paths to `pred.npz`, `gt.npz`, `lr.npz`, `meta.json`
4. Run all cells

**What it computes:**
- Basic metrics: MSE, RMSE, PSNR, SSIM, Relative L2
- PSDD (Power Spectral Density Discrepancy)
- FID (Frechet Inception Distance)
- FVD (Frechet Video Distance, R3D-18)
- Vorticity error and enstrophy error (KF256 only)

**What it generates:**
- Comparison table (printed in notebook)
- Per-model visualization: LR / GT / Pred / AbsError (saved to `/data/yc/output_images/<dataset>/`)
- Cross-model error comparison images
- Error-over-time plots (per frame index)
- Efficiency JSON for internal models (saved to `efficiency_stats/`)

### efficiency_summary.ipynb (`template_notebook/efficiency_summary.ipynb`)

Aggregates efficiency statistics across all models for a given dataset.

**How to use:**
1. Set `DATASET` to `'RB'` / `'KF256'` / `'ShallowWater'` / `'ERA5V'` / `'all'`
2. Run all cells

**What it shows:**
- Parameters (M), model size (MB)
- Training: batch size, epochs/iters, total time, time per iter
- Inference: time per sequence, peak GPU memory

It reads from `Fluid_VSR/efficiency_stats/`. External models auto-sync their JSON here via `efficiency_tracker.py`.

---

## 4. Efficiency Tracking

All external models use a shared `efficiency_tracker.py` that:
1. Records training time, iterations, peak memory during training
2. Records per-sequence inference time and peak memory during testing
3. Saves a local JSON in the experiment directory
4. Automatically syncs a merged copy to `Fluid_VSR/efficiency_stats/{ModelName}_{DatasetName}.json`

Training and inference stats are merged into a single JSON per model per dataset. The merge uses a "non-null overwrites" strategy, so running training first and inference later produces a complete record.

---

## 5. External Model Outputs for eval.ipynb

External model predictions should be placed in `Fluid_VSR/vsr_baseline/` for the eval notebook to load:

```
vsr_baseline/
  rb_dataset/
    vrt_pred/        pred.npz  gt.npz  lr.npz  meta.json
    rvrt_pred/       ...
    basicvsr++_pred/ ...
    sdift_pred/      ...
  sw_dataset/
    vrt/  rvrt/  sdift/
  kf256_dataset/
    vrt/  rvrt/  basicvsr++/
```

Each directory contains `pred.npz` (shape `[N_frames, H, W, 1]`), `gt.npz`, `lr.npz`, and `meta.json` with `frames_per_seq` and `n_seqs`.

---

## 6. Quick Start Example (RayleighBenard)

```bash
# 1. Train FNO on RB (single GPU)
cd Fluid_VSR
python main.py --config template_configs/RayleighBenard/fno.yaml

# 2. Train VRT on RB
cd KAIR-master
python main_train_vrt.py --opt options/vrt/010_train_vrt_videosr_rb.json

# 3. Test VRT on RB
python main_test_vrt_rb.py \
    --opt options/vrt/010_train_vrt_videosr_rb.json \
    --checkpoint experiments/010_train_vrt_videosr_rb/models/200000_G.pth

# 4. Open eval.ipynb, set DATASET_NAME='RB', configure model paths, run all cells
# 5. Open efficiency_summary.ipynb, set DATASET='RB', run all cells
```
