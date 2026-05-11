"""plot_zoom_figures.py
放大图格式：
  左：整张 LR 场（虚线框标出选取区域）
  右：2×3 模型 patch 对比（上行 SRNO/EDSR/GT，下行 RVRT/ResShift-50M/WRD）

单数据集输出：paper_figures/{DS}_zoom.{png,pdf}
组合输出（默认）：paper_figures/RBC_SW_zoom.{png,pdf}, paper_figures/KF256_ERA5_zoom.{png,pdf}

Usage:
    conda run -n 25cvpr python "/home/gjx/Qualitative comparison and zoomed view figure/plot_zoom_figures.py" --dataset ERA5_72
    conda run -n 25cvpr python "/home/gjx/Qualitative comparison and zoomed view figure/plot_zoom_figures.py" --dataset RBC
    conda run -n 25cvpr python "/home/gjx/Qualitative comparison and zoomed view figure/plot_zoom_figures.py" --dataset all
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── 修改这三行以适配本地环境 ──────────────────────────────────────────────────
PRED_BASE  = Path('/data_new/FluidVSR_data/expriments')        # 推理结果根目录
OUT_DIR    = Path('/home/gjx/cvpr_figure1_4')                  # 图片输出目录
PAPER_DIR  = Path('/home/gjx/cvpr_figure1_4/paper_figures')    # zoom 图输出子目录

DATASET_CONFIGS = {
    'RBC':     dict(fps=100, scale=4, n_seqs=12,
                    pred_root=PRED_BASE / 'RBC' / '×4' / 'test_predictions'),
    'ERA5':    dict(fps=24,  scale=4, n_seqs=6,
                    pred_root=PRED_BASE / 'ERA5' / '×4' / 'test_predictions'),
    'ERA5_72': dict(fps=72,  scale=4, n_seqs=10,
                    pred_root=Path('/data_new/FluidVSR_data/expriments/ERA5_72frames/x4/test_predictions')),
    'KF256':   dict(fps=180, scale=4, n_seqs=7,
                    pred_root=PRED_BASE / 'KF256' / '×4' / 'test_predictions'),
    'SW':      dict(fps=72,  scale=4, n_seqs=20,
                    pred_root=PRED_BASE / 'SW' / '×4' / 'test_predictions'),
}

DEFAULT_FRAMES = {
    'RBC':     (4,  24),
    'ERA5':    (4,  14),
    'ERA5_72': (8,  60),
    'KF256':   (5, 146),
    'SW':      (17, 28),
}

DEFAULT_PATCHES = {
    'RBC':     dict(x0=10,  x1=62,  z0=4,   z1=56),
    'ERA5':    dict(x0=113, x1=209, z0=160, z1=256),
    'ERA5_72': dict(x0=0,   x1=96,  z0=160, z1=256),
    'KF256':   dict(x0=108, x1=204, z0=160, z1=256),
    'SW':      dict(x0=1,   x1=49,  z0=40,  z1=88),
}

# 默认模型列表（RBC/SW/KF256/旧ERA5）
MODELS_DEFAULT = [
    ('srno',             'SRNO',         False),
    ('edsr',             'EDSR',         False),
    ('__GT__',           'GT',           False),
    ('rvrt',             'RVRT',         False),
    ('resshift50M',      'ResShift',     False),
    ('WRD_sgf_KCS+WSDF', 'WRD (ours)',   True),
]

# ERA5_72frames 专用模型列表（路径下模型名全小写，WRD用wrd_baseline_attn）
MODELS_ERA5_72 = [
    ('srno',              'SRNO',         False),
    ('edsr',              'EDSR',         False),
    ('__GT__',            'GT',           False),
    ('rvrt',              'RVRT',         False),
    ('resshift50m',       'ResShift',     False),
    ('wrd_baseline_attn', 'WRD (ours)',   True),
]

DS_MODELS = {
    'ERA5_72': MODELS_ERA5_72,
}

CMAP_FIELD = 'RdBu_r'

# highlight 框：只画在 GT 和 WRD 面板上，坐标为 patch 内像素索引
HIGHLIGHT_BOXES = {
    'RBC':     [dict(x0=32, x1=40, z0=10, z1=32),
                dict(x0=0,  x1=14, z0=12, z1=46)],
    'SW':      [dict(x0=25, x1=34, z0=36, z1=45),
                dict(x0=10, x1=22, z0=0,  z1=16)],
    'ERA5':    [],
    'ERA5_72': [dict(x0=72, x1=96, z0=40, z1=96)],
    'KF256':   [],
}

# ── 布局参数（英寸） ─────────────────────────────────────────────────────────
LR_SIZE    = 3.2
PATCH_W    = 1.55
COL_GAP    = 0.06
ROW_GAP    = 0.32
LR_PAD     = 0.30
LABEL_H    = 0.28
DS_LABEL_H = 0.22
LEFT_PAD   = 0.15
BOT_PAD    = 0.20
TOP_PAD    = 0.15
BLOCK_GAP  = 0.45
CBAR_W     = 0.18
CBAR_GAP   = 0.10

N_COLS = 3
N_ROWS = 2


def _panels_size():
    w = N_COLS * PATCH_W + (N_COLS - 1) * COL_GAP
    h = N_ROWS * PATCH_W + (N_ROWS - 1) * ROW_GAP
    return w, h


def _fig_w():
    pw, _ = _panels_size()
    return LEFT_PAD + LR_SIZE + LR_PAD + pw + CBAR_GAP + CBAR_W + 0.20


def find_npz(d: Path, stem: str):
    p = d / f'{stem}.npz'
    if p.exists(): return p
    cands = sorted(d.glob(f'{stem}_*.npz'))
    return cands[0] if cands else None


def load_seq(path: Path, fps: int):
    d = np.load(path)['data'][..., 0].astype(np.float32)
    n = d.shape[0] // fps
    return d.reshape(n, fps, d.shape[1], d.shape[2])


def make_lr(frame: np.ndarray, scale: int) -> np.ndarray:
    h, w = frame.shape
    return frame.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))


def load_zoom_data(ds_name, seq_idx, t_idx, patch):
    cfg   = DATASET_CONFIGS[ds_name]
    fps, scale = cfg['fps'], cfg['scale']
    pred_root  = cfg['pred_root']
    models     = DS_MODELS.get(ds_name, MODELS_DEFAULT)
    x0, x1, z0, z1 = patch['x0'], patch['x1'], patch['z0'], patch['z1']

    print(f'\n[{ds_name}] seq={seq_idx} t={t_idx}  patch x[{x0}:{x1}] z[{z0}:{z1}]')

    gt_path = None
    for dir_name, _, _ in models:
        if dir_name == '__GT__': continue
        p = find_npz(pred_root / dir_name, 'gt')
        if p: gt_path = p; break
    if gt_path is None:
        print('  [ERROR] GT not found'); return None

    gt_seqs  = load_seq(gt_path, fps)
    gt_frame = gt_seqs[seq_idx, t_idx]
    lr_frame = make_lr(gt_frame, scale)
    vmin, vmax = float(gt_frame.min()), float(gt_frame.max())

    pred_patches = {'GT': gt_frame[x0:x1, z0:z1]}
    for dir_name, label, _ in models:
        if dir_name == '__GT__': continue
        pp = find_npz(pred_root / dir_name, 'pred')
        if pp is None: print(f'  [SKIP] {dir_name}'); continue
        p_seqs = load_seq(pp, fps)
        n = min(p_seqs.shape[0], gt_seqs.shape[0])
        if seq_idx >= n: continue
        pred_patches[label] = p_seqs[seq_idx, t_idx][x0:x1, z0:z1]
        print(f'  OK  {label}')

    return dict(lr_frame=lr_frame, pred_patches=pred_patches,
                vmin=vmin, vmax=vmax, scale=scale, models=models)


def draw_zoom_block(fig, fig_w, fig_h, block_bottom, ds_name, patch, data,
                    hl_boxes=None):
    _, panels_h = _panels_size()
    panels_left = LEFT_PAD + LR_SIZE + LR_PAD
    cbar_left   = panels_left + _panels_size()[0] + CBAR_GAP

    x0, x1, z0, z1 = patch['x0'], patch['x1'], patch['z0'], patch['z1']
    lr_frame     = data['lr_frame']
    pred_patches = data['pred_patches']
    vmin, vmax   = data['vmin'], data['vmax']
    scale        = data['scale']
    models       = data['models']

    def norm(x, y, w, h):
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    patches_bottom = block_bottom + LABEL_H

    # ── LR 面板 ───────────────────────────────────────────────────────────────
    lr_y = patches_bottom + (panels_h - LR_SIZE) / 2
    ax_lr = fig.add_axes(norm(LEFT_PAD, lr_y, LR_SIZE, LR_SIZE))
    ax_lr.imshow(lr_frame, cmap=CMAP_FIELD,
                 vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
    rect = mpatches.Rectangle(
        (z0 / scale - 0.5, x0 / scale - 0.5),
        (z1 - z0) / scale, (x1 - x0) / scale,
        linewidth=1.5, edgecolor='black', facecolor='none',
        linestyle='--', zorder=5)
    ax_lr.add_patch(rect)
    ax_lr.set_xticks([]); ax_lr.set_yticks([])
    fig.text((LEFT_PAD + LR_SIZE / 2) / fig_w,
             (lr_y - 0.06) / fig_h,
             'Input LR', ha='center', va='top',
             fontsize=8.5, style='italic', color='black')

    # 数据集名标签（ERA5_72 显示为 ERA5）
    ds_label = ds_name.replace('_72', '')
    fig.text((LEFT_PAD + LR_SIZE / 2) / fig_w,
             (block_bottom + LABEL_H * 0.5) / fig_h,
             ds_label, ha='center', va='center',
             fontsize=9, style='italic', fontweight='bold')

    # ── 模型 patch 面板（2×3） ────────────────────────────────────────────────
    im_ref = None
    for mi, (dir_name, label, is_ours) in enumerate(models):
        row = mi // N_COLS
        col = mi % N_COLS
        col_left   = panels_left + col * (PATCH_W + COL_GAP)
        row_bottom = patches_bottom + (N_ROWS - 1 - row) * (PATCH_W + ROW_GAP)

        pf = pred_patches.get(label)
        ax = fig.add_axes(norm(col_left, row_bottom, PATCH_W, PATCH_W))
        if pf is not None:
            im_ref = ax.imshow(pf, cmap=CMAP_FIELD,
                               vmin=vmin, vmax=vmax, aspect='auto',
                               interpolation='nearest')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_xticks([]); ax.set_yticks([])

        if hl_boxes and pf is not None and (label == 'GT' or is_ours):
            for hl in hl_boxes:
                ax.add_patch(mpatches.Rectangle(
                    (hl['z0'] - 0.5, hl['x0'] - 0.5),
                    hl['z1'] - hl['z0'], hl['x1'] - hl['x0'],
                    linewidth=1.5, edgecolor='black', facecolor='none', zorder=6))

        lc = '#cc0000' if is_ours else 'black'
        lw = 'bold'    if is_ours else 'normal'
        fig.text((col_left + PATCH_W / 2) / fig_w,
                 (row_bottom - 0.06) / fig_h,
                 label, ha='center', va='top',
                 fontsize=8.5, style='italic', color=lc, fontweight=lw)

    # ── 色条 ─────────────────────────────────────────────────────────────────
    if im_ref is not None:
        ax_cb = fig.add_axes(norm(cbar_left, patches_bottom, CBAR_W, panels_h))
        cb = fig.colorbar(im_ref, cax=ax_cb)
        cb.ax.tick_params(labelsize=7)
        cb.set_ticks(np.linspace(vmin, vmax, 6))


# ── 单数据集图 ────────────────────────────────────────────────────────────────
def run_dataset(ds_name, seq_idx, t_idx, patch):
    data = load_zoom_data(ds_name, seq_idx, t_idx, patch)
    if data is None: return

    _, panels_h = _panels_size()
    block_h = LABEL_H + panels_h + DS_LABEL_H
    fig_w   = _fig_w()
    fig_h   = TOP_PAD + block_h + BOT_PAD

    fig = plt.figure(figsize=(fig_w, fig_h))
    draw_zoom_block(fig, fig_w, fig_h, BOT_PAD, ds_name, patch, data,
                    hl_boxes=HIGHLIGHT_BOXES.get(ds_name))

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    stem = f'{ds_name}_zoom'
    for ext in ('png', 'pdf'):
        out = PAPER_DIR / f'{stem}.{ext}'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  → {out}')
    plt.close()


# ── 多数据集组合图 ─────────────────────────────────────────────────────────────
def run_combined(ds_list, frame_overrides, patch_overrides, out_name):
    _, panels_h = _panels_size()
    block_h = LABEL_H + panels_h + DS_LABEL_H
    n_ds    = len(ds_list)
    fig_w   = _fig_w()
    fig_h   = TOP_PAD + n_ds * block_h + (n_ds - 1) * BLOCK_GAP + BOT_PAD

    fig = plt.figure(figsize=(fig_w, fig_h))

    for bi, ds_name in enumerate(ds_list):
        seq_idx, t_idx = frame_overrides.get(ds_name, DEFAULT_FRAMES[ds_name])
        patch = patch_overrides.get(ds_name, DEFAULT_PATCHES[ds_name])
        data  = load_zoom_data(ds_name, seq_idx, t_idx, patch)
        if data is None: continue

        block_bottom = fig_h - TOP_PAD - (bi + 1) * block_h - bi * BLOCK_GAP
        draw_zoom_block(fig, fig_w, fig_h, block_bottom, ds_name, patch, data,
                        hl_boxes=HIGHLIGHT_BOXES.get(ds_name))

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    stem = out_name.replace('.png', '')
    for ext in ('png', 'pdf'):
        out = PAPER_DIR / f'{stem}.{ext}'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f'\n→ {out}')
    plt.close()


# ── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['RBC', 'ERA5', 'ERA5_72', 'KF256', 'SW', 'all'],
                        default='all')
    parser.add_argument('--seq',   type=int, default=None)
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--x0', type=int, default=None)
    parser.add_argument('--x1', type=int, default=None)
    parser.add_argument('--z0', type=int, default=None)
    parser.add_argument('--z1', type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overrides = {ds: DEFAULT_FRAMES[ds] for ds in DATASET_CONFIGS}
    patches   = {ds: dict(DEFAULT_PATCHES[ds]) for ds in DATASET_CONFIGS}
    for ds in overrides:
        s, t = overrides[ds]
        if args.seq   is not None: s = args.seq
        if args.frame is not None: t = args.frame
        overrides[ds] = (s, t)
    for ds in patches:
        if args.x0 is not None: patches[ds]['x0'] = args.x0
        if args.x1 is not None: patches[ds]['x1'] = args.x1
        if args.z0 is not None: patches[ds]['z0'] = args.z0
        if args.z1 is not None: patches[ds]['z1'] = args.z1

    if args.dataset == 'all':
        for ds in DATASET_CONFIGS:
            s, t = overrides[ds]
            run_dataset(ds, s, t, patches[ds])
        run_combined(['RBC',   'SW'],      overrides, patches, 'RBC_SW_zoom.png')
        run_combined(['KF256', 'ERA5_72'], overrides, patches, 'KF256_ERA5_zoom.png')
    else:
        ds = args.dataset
        s, t = overrides[ds]
        run_dataset(ds, s, t, patches[ds])
