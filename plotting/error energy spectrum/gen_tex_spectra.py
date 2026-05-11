#!/usr/bin/env python3
"""Generate pgfplots .tex files following example.tex style exactly.
Only the coordinate data and axis ranges change per frame.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.transforms import blended_transform_factory
from scipy.ndimage import uniform_filter1d
from pathlib import Path

# ── 修改这三行以适配本地环境 ──────────────────────────────────────────────────
PRED_BASE    = Path('/data_new/FluidVSR_data/expriments')                                    # 推理结果根目录
PRED_ERA5_72 = Path('/data_new/FluidVSR_data/expriments/ERA5_72frames/x4/test_predictions') # ERA5_72 单独路径（注意小写 x4）
OUT_DIR      = Path('/home/gjx/VSR_temporal_metrics/figures/selected_frames')                # 图片和 .tex 输出目录

# ── Model definitions (hex color, pgfstyle, thickness, legend label) ─────────
MODELS = [
    ('basicvsr++',       '1f77b4', 'solid',      'very thick', 'BasicVSR++'),
    ('srno',             '17becf', 'solid',      'thick',      'SRNO'),
    ('edsr',             'ff7f0e', 'dashed',     'thick',      'EDSR'),
    ('rvrt',             'e377c2', 'solid',      'thick',      'RVRT'),
    ('resshift50M',      '7b2d8b', 'dashed',     'thick',      'ResShift'),
    ('WRD',              'e6550d', 'dashdotted', 'very thick', 'WRD'),
    ('WRD_sgf_KCS+WSDF', '7f2704', 'dashdotted', 'very thick',
     r'WRD$_{\rm sgf}$+KCS+WSDF'),
]

# ERA5_72 专用模型列表（路径全小写，WRD目录=wrd_baseline_attn）
MODELS_ERA5_72 = [
    ('basicvsr++',        '1f77b4', 'solid',      'very thick', 'BasicVSR++'),
    ('srno',              '17becf', 'solid',      'thick',      'SRNO'),
    ('edsr',              'ff7f0e', 'dashed',     'thick',      'EDSR'),
    ('rvrt',              'e377c2', 'solid',      'thick',      'RVRT'),
    ('resshift50m',       '7b2d8b', 'dashed',     'thick',      'ResShift'),
    ('wrd_baseline_attn', 'e6550d', 'dashdotted', 'very thick', 'WRD (ours)'),
]

# color name in LaTeX: strip non-alphanumeric
def cname(key):
    return 'col' + ''.join(c for c in key if c.isalnum())

# ── Frames ────────────────────────────────────────────────────────────────────
FRAMES = [
    dict(name='ERA5_seq08_frame060',  dataset='ERA5_72', seq_i=8,  t=60,
         nx=256, fps=72,  k_max=110, lr_nyq=32, mid=64,
         pred_root=PRED_ERA5_72, models=MODELS_ERA5_72),
    dict(name='KF256_seq01_frame054', dataset='KF256', seq_i=1,  t=54,
         nx=256, fps=180, k_max=110, lr_nyq=32, mid=64),
    dict(name='RBC_seq01_frame044',   dataset='RBC',   seq_i=1,  t=44,
         nx=128, fps=100, k_max=56,  lr_nyq=16, mid=32),
    dict(name='SW_seq17_frame028',    dataset='SW',    seq_i=17, t=28,
         nx=128, fps=72,  k_max=56,  lr_nyq=16, mid=32),
]

# ── Data loading ──────────────────────────────────────────────────────────────
def find_npz(d: Path, stem: str):
    p = d / f'{stem}.npz'
    if p.exists():
        return p
    cands = sorted(d.glob(f'{stem}_*.npz'))
    return cands[0] if cands else None

def load_seq(path: Path, fps: int, seq_i: int):
    d = np.load(path)['data'][..., 0].astype(np.float32)
    n = d.shape[0] // fps
    return d.reshape(n, fps, d.shape[1], d.shape[2])[seq_i]

def error_spec(pred_seq, gt_seq, t, nx, smooth=3):
    em  = (pred_seq - gt_seq).mean(axis=0)
    err = (pred_seq[t] - gt_seq[t]) - em
    F   = np.fft.rfft(err, axis=0) / nx
    E   = (np.abs(F)**2).mean(axis=1)
    return uniform_filter1d(E, size=smooth, mode='nearest')

# ── PNG preview ───────────────────────────────────────────────────────────────
_LS_MAP = {'solid': '-', 'dashed': '--', 'dashdotted': '-.', 'dotted': ':'}
_LW_MAP = {'very thick': 2.2, 'thick': 1.7}

def _save_png_preview(fc, spectra, kx, frame_models, out_path):
    k_max  = fc['k_max']
    lr_nyq = fc['lr_nyq']
    mid    = fc['mid']
    ds     = fc['dataset']
    seq_i  = fc['seq_i']
    t      = fc['t']

    plt.rcParams.update({'font.family': 'serif', 'axes.linewidth': 1.2})
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for m, hex_, ls_pgf, lw_pgf, label in frame_models:
        if m not in spectra:
            continue
        color = f'#{hex_}'
        ls    = _LS_MAP.get(ls_pgf, '-')
        lw    = _LW_MAP.get(lw_pgf, 1.7)
        kx_plot = kx[1:k_max + 1]
        ax.semilogy(kx_plot, spectra[m][1:k_max + 1],
                    label=label, color=color, ls=ls, lw=lw, alpha=0.92)

    for kv, lbl in [(lr_nyq, f'LR Nyquist\n(k={lr_nyq})'), (mid, f'k={mid}')]:
        ax.axvline(kv, color='#666', ls='--', lw=1.0, alpha=0.7)
        ax.text(kv + 0.5, 1.0, lbl,
                transform=blended_transform_factory(ax.transData, ax.transAxes),
                fontsize=6.5, color='#555', va='top')

    btrans = blended_transform_factory(ax.transData, ax.transAxes)
    for x_c, seg in [(lr_nyq // 2, 'LR band'), ((lr_nyq + mid) // 2, 'Mid band'),
                     ((mid + k_max) // 2, 'HR band')]:
        ax.text(x_c, 0.03, seg, transform=btrans, fontsize=7.5,
                color='#888', ha='center', va='bottom')

    ax.set_xlabel('FREQUENCY', fontsize=11, labelpad=6)
    ax.set_ylabel('ERROR ENERGY', fontsize=11, labelpad=6)
    ds_display = 'ERA5' if ds == 'ERA5_72' else ds
    ax.set_title(f'{ds_display} — Error-Energy Spectrum  seq {seq_i:02d} / frame {t:03d}',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim([1, k_max])
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(lr_nyq // 2))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(lr_nyq // 4))
    ax.grid(True, which='major', alpha=0.25, color='gray', lw=0.7)
    ax.grid(True, which='minor', alpha=0.10, color='gray', lw=0.4)
    ax.legend(fontsize=7, ncol=3, loc='upper right',
              framealpha=0.9, edgecolor='#aaa', borderpad=0.5, labelspacing=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path}')


# ── TeX generation ────────────────────────────────────────────────────────────
def coords(kx, E, k0, k1):
    return '\n'.join(f'        ({int(k)},{e:.8e})'
                     for k, e in zip(kx, E) if k0 <= k <= k1)

def ylim(vals):
    lo = 10 ** np.floor(np.log10(vals.min()) - 0.5)
    hi = 10 ** np.ceil (np.log10(vals.max()) + 0.5)
    return lo, hi

def make_tex(fc, spectra, kx, frame_models=None):
    if frame_models is None:
        frame_models = MODELS
    k_max  = fc['k_max']
    lr_nyq = fc['lr_nyq']
    mid    = fc['mid']
    ds     = fc['dataset']
    seq_i  = fc['seq_i']
    t      = fc['t']
    name   = fc['name']

    # inset zoom: high-freq band
    ins_k0, ins_k1 = mid, k_max

    all_e    = np.concatenate([E[1:k_max+1] for E in spectra.values()])
    ins_e    = np.concatenate([E[ins_k0:k_max+1] for E in spectra.values()])
    ymin, ymax = ylim(all_e)
    iymin, iymax = ylim(ins_e)

    # color definitions
    cdefs = '\n'.join(
        f'\\definecolor{{{cname(m)}}}{{HTML}}{{{hex_}}}'
        for m, hex_, *_ in frame_models
    )

    # main addplot blocks
    main_plots = []
    for m, hex_, ls, lw, label in frame_models:
        if m not in spectra:
            continue
        cn = cname(m)
        main_plots.append(
            f'      \\addplot+[mark=none, {lw}, color={cn}, {ls}] coordinates {{\n'
            f'{coords(kx, spectra[m], 1, k_max)}\n'
            f'      }};\n'
            f'      \\addlegendentry{{{label}}}'
        )

    # inset addplot blocks (no legend entries)
    ins_plots = []
    for m, hex_, ls, lw, label in frame_models:
        if m not in spectra:
            continue
        cn = cname(m)
        ins_plots.append(
            f'      \\addplot+[mark=none, {lw}, color={cn}, {ls}] coordinates {{\n'
            f'{coords(kx, spectra[m], ins_k0, ins_k1)}\n'
            f'      }};'
        )

    main_block = '\n\n'.join(main_plots)
    ins_block  = '\n\n'.join(ins_plots)

    return f"""{cdefs}
\\begin{{figure}}[t]
  \\centering
  \\begin{{tikzpicture}}

    %========================
    % Main axis
    %========================
    \\begin{{axis}}[
      name=main,
      width=0.8\\linewidth,
      height=0.45\\linewidth,
      xmode=log,
      ymode=log,
      xlabel={{Wavenumber $k$}},
      ylabel={{Error energy spectrum $E_{{\\rm err}}(k)$}},
      xmin=0.9, xmax={k_max + 5},
      ymin={ymin:.2e}, ymax={ymax:.2e},
      grid=both,
      grid style={{line width=0.1pt, draw=gray!25}},
      major grid style={{line width=0.2pt, draw=gray!35}},
      legend style={{
        at={{(0.02,0.05)}},
        anchor=south west,
        draw=none,
        fill=white,
        fill opacity=0.9,
        text opacity=1,
        font=\\footnotesize,
        legend columns=2,
      }},
      legend cell align=left,
      mark size=1.5pt,
      set layers,
    ]

{main_block}

      % ---- zoom region rectangle (match inset limits) ----
      \\draw[gray!60, very thin] (axis cs:{ins_k0},{iymin:.2e}) rectangle (axis cs:{ins_k1},{iymax:.2e});

    \\end{{axis}}

    %========================
    % Inset axis (top-right)
    %========================
    \\begin{{axis}}[
      name=inset,
      at={{(main.north east)}},
      anchor=north east,
      xshift=-2.5mm,
      yshift=-1.6mm,
      width=0.32\\linewidth,
      height=0.10\\linewidth,
      xmode=log,
      ymode=log,
      xmin={ins_k0}, xmax={ins_k1 + 2},
      ymin={iymin:.2e}, ymax={iymax:.2e},
      grid=both,
      grid style={{line width=0.1pt, draw=gray!25}},
      major grid style={{line width=0.2pt, draw=gray!35}},
      ticklabel style={{font=\\scriptsize}},
      label style={{font=\\scriptsize}},
      axis background/.style={{fill=white}},
      xlabel={{}},
      ylabel={{}},
      legend style={{
        at={{(0.02,0.98)}},
        anchor=north west,
        draw=none,
        fill=white,
        fill opacity=0.9,
        text opacity=1,
        font=\\scriptsize,
      }},
      legend cell align=left,
    ]

{ins_block}

    \\end{{axis}}

  \\end{{tikzpicture}}
  % \\vspace{{-23pt}}
  \\caption{{{ds} error-energy spectrum (log--log), seq~{seq_i:02d} / frame~{t:03d}.}}
  \\label{{fig:errspec-{name.lower().replace('_', '-')}}}
\\end{{figure}}
"""

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all',
                    help='dataset to process: ERA5_72 / RBC / KF256 / SW / all')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fc in FRAMES:
        if args.dataset != 'all' and fc['dataset'] != args.dataset:
            continue
        ds, seq_i, t = fc['dataset'], fc['seq_i'], fc['t']
        nx, fps      = fc['nx'], fc['fps']
        name         = fc['name']
        pred_root    = fc.get('pred_root', PRED_BASE / ds / '×4' / 'test_predictions')
        frame_models = fc.get('models', MODELS)
        kx           = np.arange(nx // 2 + 1, dtype=float)

        print(f'\n=== {name} ===')
        gt_seq, spectra = None, {}

        for m, *_ in frame_models:
            pp = find_npz(pred_root / m, 'pred')
            gp = find_npz(pred_root / m, 'gt')
            if pp is None:
                print(f'  [SKIP] {m}'); continue
            pred_seq = load_seq(pp, fps, seq_i)
            if gt_seq is None and gp is not None:
                gt_seq = load_seq(gp, fps, seq_i)
            spectra[m] = error_spec(pred_seq, gt_seq, t, nx)
            print(f'  {m}: ok')

        tex = make_tex(fc, spectra, kx, frame_models)
        out = OUT_DIR / f'{name}.tex'
        out.write_text(tex, encoding='utf-8')
        print(f'  -> {out}')

        # PNG preview
        ds_label = 'ERA5' if fc['dataset'] == 'ERA5_72' else ds
        png_dir  = OUT_DIR / ds_label
        png_dir.mkdir(parents=True, exist_ok=True)
        _save_png_preview(fc, spectra, kx, frame_models, png_dir / f'{name}.png')

    print('\nAll done.')

if __name__ == '__main__':
    main()
