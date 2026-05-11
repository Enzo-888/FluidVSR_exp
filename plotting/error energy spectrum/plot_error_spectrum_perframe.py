"""plot_error_spectrum_perframe.py — 逐帧误差能量谱图

对 /data_new/FluidVSR_data/expriments/RBC/×4/test_predictions/ 下每个模型，
按 seq / frame 逐帧保存误差能量谱对比图。

输出结构：
    OUT_ROOT/seq_00/frame_000.png
    OUT_ROOT/seq_00/frame_001.png
    ...
    OUT_ROOT/seq_11/frame_099.png

用法：
    conda run -n 25cvpr python "/home/gjx/error energy spectrum/plot_error_spectrum_perframe.py"
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

# ── 修改这两行以适配本地环境 ──────────────────────────────────────────────────
PRED_ROOT = Path('/data_new/FluidVSR_data/expriments/RBC/×4/test_predictions')  # RBC 推理结果目录
OUT_ROOT  = Path('/home/gjx/VSR_temporal_metrics/figures/spec_per_frame/RBC')   # 图片输出目录

# ── 参数（一般不需要改） ───────────────────────────────────────────────────────
T_PER_SEQ = 100
NX        = 128
K_MAX     = 56   # 截断 Nyquist 附近

# 参与绘图的模型（目录名 → 显示名）
MODELS = {
    'basicvsr++':       'BasicVSR++',
    'srno':             'SRNO',
    'edsr':             'EDSR',
    'rvrt':             'RVRT',
    'resshift50M':      'ResShift',
    'WRD':              'WRD',
    'WRD_sgf_KCS+WSDF': 'WRD_sgf_KCS+WSDF',
}

def model_style(dir_name):
    if dir_name.startswith('WRD'):
        colors = {
            'WRD':              '#e6550d',
            'WRD_sgf_KCS+WSDF': '#7f2704',
        }
        return dict(color=colors.get(dir_name, '#e6550d'), ls='-.', lw=2.0, alpha=0.95)
    palette = {
        'basicvsr++':  '#1f77b4',
        'srno':        '#17becf',
        'edsr':        '#ff7f0e',
        'rvrt':        '#e377c2',
        'resshift50M': '#7b2d8b',
    }
    return dict(color=palette.get(dir_name, 'gray'), ls='-', lw=1.6, alpha=0.92)


def compute_error_spec_frame(pred_frame: np.ndarray, gt_frame: np.ndarray,
                             err_mean: np.ndarray, smooth: int = 3) -> np.ndarray:
    """单帧 1D x 方向误差能量谱（减去序列时均误差场，k 空间平滑）。
    pred_frame, gt_frame, err_mean: (Nx, Nz)
    返回: (Nx//2+1,)
    """
    from scipy.ndimage import uniform_filter1d
    err = (pred_frame - gt_frame) - err_mean       # 去时均误差，与参考图一致
    F   = np.fft.rfft(err, axis=0) / NX           # (Nx//2+1, Nz)
    E   = (np.abs(F) ** 2).mean(axis=1)           # (Nx//2+1,)
    return uniform_filter1d(E, size=smooth, mode='nearest')  # k 空间平滑降噪


def main():
    # ── 加载所有模型数据 ──────────────────────────────────────────────────────
    print('Loading model predictions ...')
    model_data = {}   # dir_name → (n_seqs, T, Nx, Nz) float32
    gt = None

    for dir_name in MODELS:
        p = PRED_ROOT / dir_name / 'pred.npz'
        g = PRED_ROOT / dir_name / 'gt.npz'
        if not p.exists():
            print(f'  [SKIP] {dir_name}')
            continue
        d = np.load(p)['data'][..., 0].astype(np.float32)   # (N*T, Nx, Nz)
        n = d.shape[0] // T_PER_SEQ
        model_data[dir_name] = d.reshape(n, T_PER_SEQ, d.shape[1], d.shape[2])
        if gt is None and g.exists():
            gd = np.load(g)['data'][..., 0].astype(np.float32)
            gt = gd.reshape(gd.shape[0] // T_PER_SEQ, T_PER_SEQ, gd.shape[1], gd.shape[2])
        print(f'  {dir_name:20s}  shape={model_data[dir_name].shape}')

    if gt is None:
        raise RuntimeError('GT not found')

    n_seqs = gt.shape[0]
    kx     = np.fft.rfftfreq(NX) * NX              # [0, 1, ..., 64]
    kx_plot = kx[1: K_MAX + 1]                     # k=1..56

    plt.rcParams.update({'font.family': 'serif', 'axes.linewidth': 1.2})

    # ── 预计算每个 (model, seq) 的误差时均场 ────────────────────────────────
    # err_means[dir_name][seq_i] = (Nx, Nz)，与参考图的时均减法一致
    err_means = {}
    for dir_name, arr in model_data.items():
        err_means[dir_name] = [
            (arr[s] - gt[s]).mean(axis=0)   # (T, Nx, Nz) → (Nx, Nz)
            for s in range(arr.shape[0])
        ]

    # ── 逐 seq / 逐帧绘图 ─────────────────────────────────────────────────────
    for seq_i in range(n_seqs):
        out_dir = OUT_ROOT / f'seq_{seq_i:02d}'
        out_dir.mkdir(parents=True, exist_ok=True)

        for t in range(T_PER_SEQ):
            fig, ax = plt.subplots(figsize=(9, 4.5))

            for dir_name, label in MODELS.items():
                if dir_name not in model_data:
                    continue
                pred_arr = model_data[dir_name]
                if seq_i >= pred_arr.shape[0]:
                    continue
                E = compute_error_spec_frame(
                    pred_arr[seq_i, t],
                    gt[seq_i, t],
                    err_means[dir_name][seq_i],
                )
                st = model_style(dir_name)
                ax.semilogy(kx_plot, E[1: K_MAX + 1],
                            label=label, **st)

            # 竖线
            for kv, lbl in [(16, 'LR Nyquist\n(k=16)'), (32, 'k=32')]:
                ax.axvline(kv, color='#666666', linestyle='--', linewidth=1.0, alpha=0.7)
                ax.text(kv + 0.4, 1.0, lbl,
                        transform=blended_transform_factory(ax.transData, ax.transAxes),
                        fontsize=6.5, color='#555555', va='top', ha='left')

            # 频段标签
            btrans = blended_transform_factory(ax.transData, ax.transAxes)
            for x_c, seg in [(8, 'LR band'), (24, 'Mid band'), (44, 'HR band')]:
                ax.text(x_c, 0.03, seg, transform=btrans,
                        fontsize=7.5, color='#888888', ha='center', va='bottom')

            ax.set_xlabel('FREQUENCY', fontsize=11, labelpad=6, fontfamily='serif')
            ax.set_ylabel('ERROR ENERGY', fontsize=11, labelpad=6, fontfamily='serif')
            ax.set_title(
                f'Error-Energy Spectrum — seq {seq_i:02d} / frame {t:03d}',
                fontsize=11, fontweight='bold', pad=8
            )
            ax.set_xlim([1, K_MAX])
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(8))
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(4))
            ax.grid(True, which='major', alpha=0.25, color='gray', linewidth=0.7)
            ax.grid(True, which='minor', alpha=0.10, color='gray', linewidth=0.4)

            leg = ax.legend(fontsize=6.5, ncol=3, loc='upper right',
                            framealpha=0.9, edgecolor='#aaaaaa',
                            borderpad=0.5, labelspacing=0.25)

            fig.tight_layout()
            fig.savefig(out_dir / f'frame_{t:03d}.png', dpi=120, bbox_inches='tight')
            plt.close(fig)

        print(f'  seq {seq_i:02d} done → {out_dir}')

    print(f'\n所有帧已保存 → {OUT_ROOT}')


if __name__ == '__main__':
    main()
