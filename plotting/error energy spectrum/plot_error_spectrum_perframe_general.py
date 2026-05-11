"""plot_error_spectrum_perframe_general.py — 多数据集逐帧误差能量谱

对 ERA5 / KF256 / SW 三个数据集，按 seq/frame 逐帧保存误差能量谱对比图。

输出结构（各数据集独立子目录）：
    figures/spec_per_frame_ERA5/seq_00/frame_000.png
    figures/spec_per_frame_KF256/seq_00/frame_000.png
    figures/spec_per_frame_SW/seq_00/frame_000.png

用法：
    # 跑单个数据集
    conda run -n 25cvpr python "/home/gjx/error energy spectrum/plot_error_spectrum_perframe_general.py" --dataset ERA5
    conda run -n 25cvpr python "/home/gjx/error energy spectrum/plot_error_spectrum_perframe_general.py" --dataset KF256
    conda run -n 25cvpr python "/home/gjx/error energy spectrum/plot_error_spectrum_perframe_general.py" --dataset SW

    # 跑全部（串行）
    conda run -n 25cvpr python "/home/gjx/error energy spectrum/plot_error_spectrum_perframe_general.py" --dataset all
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

# ── 修改这两行以适配本地环境 ──────────────────────────────────────────────────
PRED_BASE = Path('/data_new/FluidVSR_data/expriments')       # 推理结果根目录
OUT_BASE  = Path('/home/gjx/VSR_temporal_metrics/figures')   # 图片输出根目录（会在其下创建 spec_per_frame/{DS}/）

# ── 数据集配置 ────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    'ERA5': {
        'nx':           256,
        'frames_per_seq': 24,
        'n_seqs':       6,
        'k_max':        110,
        'lr_nyquist':   32,
        'mid_boundary': 64,
        'models': {
            'basicvsr++':       'BasicVSR++',
            'srno':             'SRNO',
            'edsr':             'EDSR',
            'rvrt':             'RVRT',
            'resshift50M':      'ResShift',
            'WRD':              'WRD',
            'WRD_sgf_KCS+WSDF': 'WRD_sgf_KCS+WSDF',
        },
    },
    'KF256': {
        'nx':           256,
        'frames_per_seq': 180,
        'n_seqs':       7,
        'k_max':        110,
        'lr_nyquist':   32,
        'mid_boundary': 64,
        'models': {
            'basicvsr++':       'BasicVSR++',
            'srno':             'SRNO',
            'edsr':             'EDSR',
            'rvrt':             'RVRT',
            'resshift50M':      'ResShift',
            'WRD':              'WRD',
            'WRD_sgf_KCS+WSDF': 'WRD_sgf_KCS+WSDF',
        },
    },
    'SW': {
        'nx':           128,
        'frames_per_seq': 72,
        'n_seqs':       20,
        'k_max':        56,
        'lr_nyquist':   16,
        'mid_boundary': 32,
        'models': {
            'basicvsr++':       'BasicVSR++',
            'srno':             'SRNO',
            'edsr':             'EDSR',
            'rvrt':             'RVRT',
            'resshift50M':      'ResShift',
            'WRD':              'WRD',
            'WRD_sgf_KCS+WSDF': 'WRD_sgf_KCS+WSDF',
        },
    },
}

# ── 固定样式 ──────────────────────────────────────────────────────────────────
_BASELINE_COLORS = {
    'basicvsr++':    '#1f77b4',
    'srno':          '#17becf',
    'edsr':          '#ff7f0e',
    'rvrt':          '#e377c2',
    'resshift112M':  '#7b2d8b',
    'resshift50M':   '#7b2d8b',
}
_WRD_COLORS = {
    'WRD':              '#e6550d',
    'WRD_KCS':          '#fd8d3c',
    'WRD_KCS+WSDF':     '#a63603',
    'WRD_sgf':          '#fdae6b',
    'WRD_sgf_KCS':      '#8B4513',
    'WRD_sgf_WSDF':     '#d94801',
    'WRD_sgf_KCS+WSDF': '#7f2704',
}

def model_style(dir_name):
    if dir_name.startswith('WRD'):
        return dict(color=_WRD_COLORS.get(dir_name, '#e6550d'), ls='-.', lw=2.0, alpha=0.95)
    return dict(color=_BASELINE_COLORS.get(dir_name, 'gray'), ls='-', lw=1.6, alpha=0.92)


# ── 核心计算 ──────────────────────────────────────────────────────────────────
def compute_error_spec_frame(pred_frame: np.ndarray, gt_frame: np.ndarray,
                             err_mean: np.ndarray, nx: int, smooth: int = 3) -> np.ndarray:
    """单帧 1D x 方向误差能量谱（减去序列时均误差场，k 空间平滑）。
    pred_frame, gt_frame, err_mean: (Nx, Nz)  →  返回 (Nx//2+1,)
    """
    from scipy.ndimage import uniform_filter1d
    err = (pred_frame - gt_frame) - err_mean
    F   = np.fft.rfft(err, axis=0) / nx
    E   = (np.abs(F) ** 2).mean(axis=1)
    return uniform_filter1d(E, size=smooth, mode='nearest')


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_model(path: Path, frames_per_seq: int):
    """(N*T, Nx, Nz, 1) → (n_seqs, T, Nx, Nz) float32"""
    d = np.load(path)['data'][..., 0].astype(np.float32)
    n = d.shape[0] // frames_per_seq
    return d.reshape(n, frames_per_seq, d.shape[1], d.shape[2])


# ── 单数据集处理 ──────────────────────────────────────────────────────────────
def run_dataset(ds_name: str):
    cfg        = DATASET_CONFIGS[ds_name]
    pred_root  = PRED_BASE / ds_name / '×4' / 'test_predictions'
    out_root   = OUT_BASE / 'spec_per_frame' / ds_name
    nx         = cfg['nx']
    fps        = cfg['frames_per_seq']
    k_max      = cfg['k_max']
    lr_nyq     = cfg['lr_nyquist']
    mid_bnd    = cfg['mid_boundary']
    models_cfg = cfg['models']

    print(f'\n{"="*60}')
    print(f'Dataset: {ds_name}  NX={nx}  frames_per_seq={fps}')
    print(f'{"="*60}')

    # 加载所有模型
    def _find_npz(model_dir, stem):
        p = model_dir / f'{stem}.npz'
        if p.exists():
            return p
        candidates = sorted(model_dir.glob(f'{stem}_*.npz'))
        return candidates[0] if candidates else None

    model_data = {}
    gt = None
    for dir_name, label in models_cfg.items():
        p = _find_npz(pred_root / dir_name, 'pred')
        g = _find_npz(pred_root / dir_name, 'gt')
        if p is None:
            print(f'  [SKIP] {dir_name}')
            continue
        model_data[dir_name] = load_model(p, fps)
        if gt is None and g is not None:
            gt = load_model(g, fps)
        print(f'  {dir_name:25s}  shape={model_data[dir_name].shape}')

    if gt is None:
        print(f'[ERROR] GT not found for {ds_name}')
        return

    n_seqs  = gt.shape[0]
    kx      = np.fft.rfftfreq(nx) * nx           # [0, 1, ..., nx//2]
    kx_plot = kx[1: k_max + 1]                   # k=1..k_max

    # 预计算每个 (model, seq) 的误差时均场
    err_means = {
        dir_name: [
            (model_data[dir_name][s] - gt[s]).mean(axis=0)
            for s in range(model_data[dir_name].shape[0])
        ]
        for dir_name in model_data
    }

    plt.rcParams.update({'font.family': 'serif', 'axes.linewidth': 1.2})

    for seq_i in range(n_seqs):
        out_dir = out_root / f'seq_{seq_i:02d}'
        out_dir.mkdir(parents=True, exist_ok=True)

        for t in range(fps):
            fig, ax = plt.subplots(figsize=(9, 4.5))

            for dir_name, label in models_cfg.items():
                if dir_name not in model_data:
                    continue
                arr = model_data[dir_name]
                if seq_i >= arr.shape[0]:
                    continue
                E  = compute_error_spec_frame(arr[seq_i, t], gt[seq_i, t],
                                              err_means[dir_name][seq_i], nx)
                st = model_style(dir_name)
                ax.semilogy(kx_plot, E[1: k_max + 1], label=label, **st)

            # 竖线
            for kv, lbl in [(lr_nyq, f'LR Nyquist\n(k={lr_nyq})'), (mid_bnd, f'k={mid_bnd}')]:
                ax.axvline(kv, color='#666666', linestyle='--', linewidth=1.0, alpha=0.7)
                ax.text(kv + 0.4 * (k_max / 56), 1.0, lbl,
                        transform=blended_transform_factory(ax.transData, ax.transAxes),
                        fontsize=6.5, color='#555555', va='top', ha='left')

            # 频段标签
            btrans = blended_transform_factory(ax.transData, ax.transAxes)
            lo_c  = lr_nyq // 2
            mid_c = (lr_nyq + mid_bnd) // 2
            hi_c  = (mid_bnd + k_max) // 2
            for x_c, seg in [(lo_c, 'LR band'), (mid_c, 'Mid band'), (hi_c, 'HR band')]:
                ax.text(x_c, 0.03, seg, transform=btrans,
                        fontsize=7.5, color='#888888', ha='center', va='bottom')

            ax.set_xlabel('FREQUENCY', fontsize=11, labelpad=6, fontfamily='serif')
            ax.set_ylabel('ERROR ENERGY', fontsize=11, labelpad=6, fontfamily='serif')
            ax.set_title(
                f'{ds_name} — Error-Energy Spectrum  seq {seq_i:02d} / frame {t:03d}',
                fontsize=11, fontweight='bold', pad=8
            )
            ax.set_xlim([1, k_max])
            step = 16 if nx == 256 else 8
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(step // 2))
            ax.grid(True, which='major', alpha=0.25, color='gray', linewidth=0.7)
            ax.grid(True, which='minor', alpha=0.10, color='gray', linewidth=0.4)

            ax.legend(fontsize=6.5, ncol=3, loc='upper right',
                      framealpha=0.9, edgecolor='#aaaaaa',
                      borderpad=0.5, labelspacing=0.25)

            fig.tight_layout()
            fig.savefig(out_dir / f'frame_{t:03d}.png', dpi=120, bbox_inches='tight')
            plt.close(fig)

        print(f'  seq {seq_i:02d} done → {out_dir}')

    print(f'{ds_name} 完成 → {out_root}')


# ── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ERA5', 'KF256', 'SW', 'all'], default='all')
    args = parser.parse_args()

    targets = ['ERA5', 'KF256', 'SW'] if args.dataset == 'all' else [args.dataset]
    for ds in targets:
        run_dataset(ds)
