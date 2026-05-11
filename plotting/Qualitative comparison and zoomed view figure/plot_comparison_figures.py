"""plot_comparison_figures.py
参考格式：左侧大 LR 面板（占两行），右侧模型列（上行预测 / 下行误差图），4 数据集垂直堆叠。

Usage:
    conda run -n 25cvpr python "/home/gjx/Qualitative comparison and zoomed view figure/plot_comparison_figures.py"
    conda run -n 25cvpr python "/home/gjx/Qualitative comparison and zoomed view figure/plot_comparison_figures.py" --dataset RBC --seq 2 --frame 30
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path

# ── 修改这两行以适配本地环境 ──────────────────────────────────────────────────
PRED_BASE = Path('/data_new/FluidVSR_data/expriments')   # 推理结果根目录
OUT_DIR   = Path('/home/gjx/cvpr_figure1_4')             # 图片输出目录

DATASET_CONFIGS = {
    'RBC':     dict(fps=100, scale=4, n_seqs=12,
                    pred_root=PRED_BASE / 'RBC'   / '×4' / 'test_predictions'),
    'ERA5':    dict(fps=24,  scale=4, n_seqs=6,
                    pred_root=PRED_BASE / 'ERA5'  / '×4' / 'test_predictions'),
    'ERA5_72': dict(fps=72,  scale=4, n_seqs=10,
                    pred_root=Path('/data_new/FluidVSR_data/expriments/ERA5_72frames/x4/test_predictions')),
    'KF256':   dict(fps=180, scale=4, n_seqs=7,
                    pred_root=PRED_BASE / 'KF256' / '×4' / 'test_predictions'),
    'SW':      dict(fps=72,  scale=4, n_seqs=20,
                    pred_root=PRED_BASE / 'SW'    / '×4' / 'test_predictions'),
}

DEFAULT_FRAMES = {
    'RBC':     (4,  24),
    'ERA5':    (4,  14),
    'ERA5_72': (8,  60),
    'KF256':   (5, 146),
    'SW':      (17, 28),
}

MODELS_DEFAULT = [
    ('srno',             'SRNO',         False),
    ('edsr',             'EDSR',         False),
    ('basicvsr++',       'BasicVSR++',   False),
    ('rvrt',             'RVRT',         False),
    ('resshift50M',      'ResShift',     False),
    ('WRD_sgf_KCS+WSDF', 'WRD (ours)',   True),
]

MODELS_ERA5_72 = [
    ('srno',              'SRNO',         False),
    ('edsr',              'EDSR',         False),
    ('basicvsr++',        'BasicVSR++',   False),
    ('rvrt',              'RVRT',         False),
    ('resshift50m',       'ResShift',     False),
    ('wrd_baseline_attn', 'WRD (ours)',   True),
]

DS_MODELS = {
    'ERA5_72': MODELS_ERA5_72,
}

CMAP_FIELD = 'RdBu_r'
CMAP_ERROR = 'hot'

ERR_VMAX_OVERRIDE = {'RBC': 0.06}

# ── 布局参数（单位：英寸） ────────────────────────────────────────────────────
PANEL_W    = 1.15   # 每个模型面板宽
PANEL_H    = 1.15   # 每个模型面板高
LR_W       = 2.55   # 正方形：= 2*PANEL_H + ROW_GAP = lr_h
ROW_GAP    = 0.25   # 预测行与误差行之间的间距（色条分隔）
COL_GAP    = 0.04   # 列间距
BLOCK_GAP  = 0.90   # 数据集 block 之间的间距
LABEL_H    = 0.30   # 标签行高
LR_MARGIN  = 0.90   # LR 面板左侧（用于数据集文字标签）
RIGHT_PAD  = 0.50   # 右边距（给色条）
TOP_PAD    = 0.20
BOT_PAD    = 0.20
CBAR_W     = 0.16
CBAR_GAP   = 0.10


def find_npz(d: Path, stem: str):
    p = d / f'{stem}.npz'
    if p.exists():
        return p
    cands = sorted(d.glob(f'{stem}_*.npz'))
    return cands[0] if cands else None


def load_seq(path: Path, fps: int):
    d = np.load(path)['data'][..., 0].astype(np.float32)
    n = d.shape[0] // fps
    return d.reshape(n, fps, d.shape[1], d.shape[2])


def make_lr(frame: np.ndarray, scale: int) -> np.ndarray:
    h, w = frame.shape
    return frame.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))


def ax_pos(left_in, bottom_in, w_in, h_in, fig_w, fig_h):
    """英寸坐标 → 归一化 figure 坐标 [left, bottom, width, height]"""
    return [left_in / fig_w, bottom_in / fig_h, w_in / fig_w, h_in / fig_h]


def run(datasets, frame_overrides, out_name='comparison.png'):
    n_ds = len(datasets)
    n_models = len(MODELS_DEFAULT)

    lr_h = 2 * PANEL_H + ROW_GAP
    models_total_w = n_models * PANEL_W + (n_models - 1) * COL_GAP
    block_h = LABEL_H + lr_h

    fig_w = LR_MARGIN + LR_W + COL_GAP + models_total_w + CBAR_GAP + CBAR_W + RIGHT_PAD
    fig_h = TOP_PAD + n_ds * block_h + (n_ds - 1) * BLOCK_GAP + BOT_PAD

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── 固定 x 坐标 ────────────────────────────────────────────────────────────
    lr_left   = LR_MARGIN
    mod_left0 = LR_MARGIN + LR_W + COL_GAP     # 第0个模型列的左边
    cbar_left = mod_left0 + models_total_w + CBAR_GAP

    for bi, ds_name in enumerate(datasets):
        cfg = DATASET_CONFIGS[ds_name]
        fps, scale = cfg['fps'], cfg['scale']
        pred_root = cfg['pred_root']
        models = DS_MODELS.get(ds_name, MODELS_DEFAULT)

        seq_idx, t_idx = frame_overrides.get(ds_name, DEFAULT_FRAMES[ds_name])
        print(f'\n[{ds_name}] seq={seq_idx} t={t_idx}')

        # GT
        gt_path = None
        for dir_name, _, _ in models:
            p = find_npz(pred_root / dir_name, 'gt')
            if p:
                gt_path = p; break
        if gt_path is None:
            print(f'  [ERROR] GT not found'); continue
        gt_seqs  = load_seq(gt_path, fps)
        gt_frame = gt_seqs[seq_idx, t_idx]
        lr_frame = make_lr(gt_frame, scale)
        vmin, vmax = float(gt_frame.min()), float(gt_frame.max())

        # 预测
        pred_frames = {}
        for dir_name, label, _ in models:
            pp = find_npz(pred_root / dir_name, 'pred')
            if pp is None:
                print(f'  [SKIP] {dir_name}'); continue
            p_seqs = load_seq(pp, fps)
            n = min(p_seqs.shape[0], gt_seqs.shape[0])
            if seq_idx < n:
                pred_frames[label] = p_seqs[seq_idx, t_idx]
                print(f'  OK  {label}')

        # 误差上限：所有模型误差的 95 百分位
        errs = [np.abs(pred_frames[l] - gt_frame)
                for _, l, _ in models if l in pred_frames]
        err_vmax = ERR_VMAX_OVERRIDE.get(ds_name,
                   float(np.percentile(np.concatenate([e.ravel() for e in errs]), 95))
                   if errs else 1.0)

        # ── block 的 y 坐标（从图底部算起） ─────────────────────────────────
        # bi=0 是最顶端
        block_bottom = fig_h - TOP_PAD - (bi + 1) * block_h - bi * BLOCK_GAP
        pred_bottom  = block_bottom + PANEL_H + ROW_GAP
        err_bottom   = block_bottom

        # ── LR 面板 ──────────────────────────────────────────────────────────
        ax_lr = fig.add_axes(ax_pos(lr_left, block_bottom, LR_W, lr_h, fig_w, fig_h))
        ax_lr.imshow(lr_frame, cmap=CMAP_FIELD,
                     vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
        ax_lr.set_xticks([]); ax_lr.set_yticks([])
        ds_label = ds_name.replace('_72', '')
        fig.text((lr_left + LR_W / 2) / fig_w,
                 (block_bottom + lr_h + LABEL_H * 0.55) / fig_h,
                 f'{ds_label} – LR Input',
                 ha='center', va='center', fontsize=9, style='italic', fontweight='bold')

        # ── 模型列 ────────────────────────────────────────────────────────────
        for mi, (dir_name, label, is_ours) in enumerate(models):
            col_left = mod_left0 + mi * (PANEL_W + COL_GAP)
            pf = pred_frames.get(label)

            # 列标签（仅 bi==0 且在最上方 block 时显示，或每个 block 都显示）
            label_color = '#cc0000' if is_ours else 'black'
            label_weight = 'bold' if is_ours else 'normal'
            fig.text((col_left + PANEL_W / 2) / fig_w,
                     (block_bottom + lr_h + LABEL_H * 0.55) / fig_h,
                     label, ha='center', va='center',
                     fontsize=8, style='italic',
                     color=label_color, fontweight=label_weight)

            # 预测面板
            ax_p = fig.add_axes(ax_pos(col_left, pred_bottom, PANEL_W, PANEL_H, fig_w, fig_h))
            if pf is not None:
                im_f = ax_p.imshow(pf, cmap=CMAP_FIELD,
                                   vmin=vmin, vmax=vmax, aspect='auto',
                                   interpolation='nearest')
            else:
                ax_p.text(0.5, 0.5, 'N/A', ha='center', va='center',
                          transform=ax_p.transAxes, fontsize=8, color='gray')
            ax_p.set_xticks([]); ax_p.set_yticks([])

            # 误差面板
            ax_e = fig.add_axes(ax_pos(col_left, err_bottom, PANEL_W, PANEL_H, fig_w, fig_h))
            if pf is not None:
                err = np.abs(pf - gt_frame)
                im_e = ax_e.imshow(err, cmap=CMAP_ERROR,
                                   vmin=0, vmax=err_vmax, aspect='auto',
                                   interpolation='nearest')
            else:
                ax_e.text(0.5, 0.5, 'N/A', ha='center', va='center',
                          transform=ax_e.transAxes, fontsize=8, color='gray')
            ax_e.set_xticks([]); ax_e.set_yticks([])

        # ── 色条（预测 + 误差，分别贴右侧） ─────────────────────────────────
        # 预测色条（上半段，对应 pred 行）
        ax_cbf = fig.add_axes(ax_pos(cbar_left, pred_bottom, CBAR_W, PANEL_H, fig_w, fig_h))
        cb_f = fig.colorbar(
            cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CMAP_FIELD),
            cax=ax_cbf)
        cb_f.ax.tick_params(labelsize=6.5)
        cb_f.set_ticks(np.linspace(vmin, vmax, 5))

        # 误差色条（下半段，对应 error 行）
        ax_cbe = fig.add_axes(ax_pos(cbar_left, err_bottom, CBAR_W, PANEL_H, fig_w, fig_h))
        cb_e = fig.colorbar(
            cm.ScalarMappable(norm=Normalize(vmin=0, vmax=err_vmax), cmap=CMAP_ERROR),
            cax=ax_cbe)
        cb_e.ax.tick_params(labelsize=6.5)
        cb_e.set_ticks(np.linspace(0, err_vmax, 5))

    stem = out_name.replace('.png', '')
    for ext in ('png', 'pdf'):
        out_path = OUT_DIR / f'{stem}.{ext}'
        plt.savefig(out_path, dpi=180, bbox_inches='tight')
        print(f'\n→ {out_path}')
    plt.close()


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['RBC', 'ERA5', 'ERA5_72', 'KF256', 'SW', 'all'],
                        default='all', help='单独跑某数据集，或 all 跑全部组合')
    parser.add_argument('--seq',   type=int, default=None)
    parser.add_argument('--frame', type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overrides = {ds: DEFAULT_FRAMES[ds] for ds in DATASET_CONFIGS}
    for ds in overrides:
        s, t = overrides[ds]
        if args.seq   is not None: s = args.seq
        if args.frame is not None: t = args.frame
        overrides[ds] = (s, t)

    if args.dataset == 'all':
        run(['RBC', 'SW'],                      overrides, out_name='comparison_fig1.png')
        run(['RBC'],                            overrides, out_name='comparison_RBC.png')
        run(['SW'],                             overrides, out_name='comparison_SW.png')
        run(['ERA5_72'],                        overrides, out_name='comparison_fig2.png')
        run(['RBC', 'SW', 'KF256', 'ERA5_72'], overrides, out_name='comparison_all.png')
    else:
        run([args.dataset], overrides, out_name=f'comparison_{args.dataset}.png')
