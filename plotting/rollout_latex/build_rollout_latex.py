#!/usr/bin/env python3
"""Build per-dataset rollout CSVs and LaTeX plots for selected models.

Outputs are written under this directory by default.

For each dataset, this script generates:
    - one CSV for `Error over time` (mean absolute error)
    - one CSV for `RMSE over time`
    - one standalone LaTeX file for each metric
    - a manifest JSON describing the source directories used

The metric definitions are aligned with the notebook logic:
    mae_per_frame  = mean_seq(mean_space(abs(pred - gt)))
    rmse_per_frame = mean_seq(sqrt(mean_space((pred - gt)^2)))
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PRED_ROOT = Path("/home/yc/transfer_to_a100/x4_test_predictions")
DEFAULT_OUT_ROOT = THIS_DIR

DATASETS = {
    "ERA5_24frames": {
        "folder": "ERA5V",
        "title": "ERA5_24frames×4",
        "frames_per_seq": 24,
        "force_frames_per_seq": 24,
    },
    "ERA5": {
        "title": "ERA5×4",
        "frames_per_seq": 72,
        "force_frames_per_seq": 72,
        "test_predictions_dir": "/home/yc/transfer_to_a100/formal_predictions/x4/era5/test_predictions",
        "model_alias_overrides": {
            "wrd_kcs_wsdf": ["wrd_baseline_attn"],
        },
    },
    "KF256": {
        "folder": "KF256",
        "title": "KF256×4",
        "frames_per_seq": 180,
    },
    "KF256_video": {
        "title": "KF256×4",
        "frames_per_seq": 180,
        "force_frames_per_seq": 180,
        "test_predictions_dir": "/home/yc/transfer_to_a100/formal_predictions/x4/KF256/test_predictions",
        "model_keys": ["basicvsrpp", "vrt", "rvrt", "sdift", "wdno", "wrd_kcs_wsdf"],
        "model_alias_overrides": {
            "wrd_kcs_wsdf": ["wrd_sgf"],
        },
    },
    "RBC": {
        "folder": "RBC",
        "title": "RBC×4",
        "frames_per_seq": 100,
    },
    "SW": {
        "folder": "SW",
        "title": "SW×4",
        "frames_per_seq": 72,
    },
}

MODEL_SPECS = [
    {
        "key": "srno",
        "display": "SRNO",
        "aliases": ["srno"],
        "style": "rolloutSRNO",
    },
    {
        "key": "edsr",
        "display": "EDSR",
        "aliases": ["edsr"],
        "style": "rolloutEDSR",
    },
    {
        "key": "basicvsrpp",
        "display": "BasicVSR++",
        "aliases": ["basicvsr++"],
        "style": "rolloutBasicVSRPP",
    },
    {
        "key": "rvrt",
        "display": "RVRT",
        "aliases": ["rvrt"],
        "style": "rolloutRVRT",
    },
    {
        "key": "vrt",
        "display": "VRT",
        "aliases": ["vrt"],
        "style": "rolloutVRT",
    },
    {
        "key": "sdift",
        "display": "SDIFT",
        "aliases": ["sdift"],
        "style": "rolloutSDIFT",
    },
    {
        "key": "wdno",
        "display": "WDNO",
        "aliases": ["wdno"],
        "style": "rolloutWDNO",
    },
    {
        "key": "resshift50m",
        "display": "ResShift",
        "aliases": ["resshift50m"],
        "style": "rolloutResShift50M",
    },
    {
        "key": "wrd_kcs_wsdf",
        "display": "WRD",
        "aliases": ["wrd_sgf_kcs+wsdf", "wrd_kcs+wsdf"],
        "style": "rolloutWRDKCSWSDF",
    },
]

METRICS = [
    {
        "key": "error",
        "ylabel": "Mean absolute error",
        "title": "Error over time",
    },
    {
        "key": "rmse",
        "ylabel": "RMSE",
        "title": "RMSE over time",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Subset of datasets to generate",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile all generated LaTeX files into PDFs after generation.",
    )
    return parser.parse_args()


def ensure_dirs(out_root: Path) -> dict[str, Path]:
    paths = {
        "root": out_root,
        "data": out_root / "data",
        "tex": out_root / "tex",
        "pdf": out_root / "pdf",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_model_specs(dataset_cfg: dict[str, object]) -> list[dict[str, object]]:
    model_keys = dataset_cfg.get("model_keys")
    if not model_keys:
        return MODEL_SPECS
    spec_by_key = {spec["key"]: spec for spec in MODEL_SPECS}
    return [spec_by_key[key] for key in model_keys if key in spec_by_key]


def load_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        if "data" in npz:
            arr = npz["data"]
        else:
            arr = npz[npz.files[0]]
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    return arr.astype(np.float64, copy=False)


def load_frames_per_seq(meta_path: Path, fallback: int) -> int:
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        fps = meta.get("frames_per_seq")
        if isinstance(fps, int) and fps > 0:
            return fps
    return fallback


def choose_model_dirs(
    test_predictions_dir: Path,
    model_specs: list[dict[str, object]],
    model_alias_overrides: dict[str, list[str]] | None = None,
) -> dict[str, Path]:
    available = {
        child.name.lower(): child
        for child in sorted(test_predictions_dir.iterdir())
        if child.is_dir()
    }

    chosen: dict[str, Path] = {}
    for spec in model_specs:
        aliases = spec["aliases"]
        if model_alias_overrides and spec["key"] in model_alias_overrides:
            aliases = model_alias_overrides[spec["key"]]
        for alias in aliases:
            alias_l = alias.lower()
            if alias_l in available:
                chosen[spec["key"]] = available[alias_l]
                break
    return chosen


def resolve_test_predictions_dir(pred_root: Path, dataset_cfg: dict[str, object]) -> Path:
    explicit_dir = dataset_cfg.get("test_predictions_dir")
    if explicit_dir:
        return Path(str(explicit_dir))
    return pred_root / str(dataset_cfg["folder"]) / "test_predictions"


def get_frames_per_seq(meta_path: Path, dataset_cfg: dict[str, object]) -> int:
    forced = dataset_cfg.get("force_frames_per_seq")
    if isinstance(forced, int) and forced > 0:
        return forced
    return load_frames_per_seq(meta_path, int(dataset_cfg["frames_per_seq"]))


def compute_curves(pred: np.ndarray, gt: np.ndarray, frames_per_seq: int) -> dict[str, np.ndarray]:
    n_frames = min(pred.shape[0], gt.shape[0])
    n_seqs = n_frames // frames_per_seq
    if n_seqs <= 0:
        raise ValueError(f"Not enough frames for frames_per_seq={frames_per_seq}: {n_frames}")

    use_frames = n_seqs * frames_per_seq
    pred_all = pred[:use_frames].squeeze(-1).reshape(n_seqs, frames_per_seq, -1)
    gt_all = gt[:use_frames].squeeze(-1).reshape(n_seqs, frames_per_seq, -1)
    diff_all = pred_all - gt_all

    mae_per_frame = np.abs(diff_all).mean(axis=2).mean(axis=0)
    rmse_per_frame = np.sqrt((diff_all ** 2).mean(axis=2)).mean(axis=0)

    return {
        "error": mae_per_frame,
        "rmse": rmse_per_frame,
    }


def write_metric_csv(
    csv_path: Path,
    model_curves: dict[str, np.ndarray],
    model_specs: list[dict[str, object]],
) -> None:
    keys_in_order = [spec["key"] for spec in model_specs if spec["key"] in model_curves]
    if not keys_in_order:
        raise ValueError(f"No selected models available for {csv_path.name}")

    n_frames = len(model_curves[keys_in_order[0]])
    for key in keys_in_order[1:]:
        if len(model_curves[key]) != n_frames:
            raise ValueError(f"Curve length mismatch in {csv_path.name}: {keys_in_order[0]} vs {key}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", *keys_in_order])
        for frame_idx in range(n_frames):
            row = [frame_idx]
            for key in keys_in_order:
                row.append(f"{model_curves[key][frame_idx]:.10f}")
            writer.writerow(row)


def latex_escape(text: str) -> str:
    escaped = text
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        escaped = escaped.replace(old, new)
    return escaped


def build_addplot_lines(
    csv_relpath: str,
    available_keys: list[str],
    model_specs: list[dict[str, object]],
) -> str:
    parts: list[str] = []
    for spec in model_specs:
        key = spec["key"]
        if key not in available_keys:
            continue
        parts.append(
            "\\addplot+[{}] table[x=frame_idx,y={},col sep=comma] {{{}}};\n"
            "\\addlegendentry{{{}}}".format(
                spec["style"],
                key,
                csv_relpath,
                latex_escape(spec["display"]),
            )
        )
    return "\n".join(parts)


def latex_text(text: str) -> str:
    return latex_escape(text).replace("×", r"$\times$")


def write_latex_plot(
    tex_path: Path,
    dataset_title: str,
    metric_title: str,
    ylabel: str,
    csv_filename: str,
    available_keys: list[str],
    model_specs: list[dict[str, object]],
    ymax: float,
) -> None:
    addplot_lines = build_addplot_lines(f"../data/{csv_filename}", available_keys, model_specs)
    content = f"""\\documentclass[tikz,border=3pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\input{{rollout_plot_styles.tex}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
    rolloutAxis,
    ylabel={{{latex_text(ylabel)}}},
    ymax={ymax:.10f},
]
{addplot_lines}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
    tex_path.write_text(content, encoding="utf-8")


def write_styles_file(style_path: Path) -> None:
    content = r"""\definecolor{rollBlue}{HTML}{1F77B4}
\definecolor{rollTeal}{HTML}{0F766E}
\definecolor{rollOrange}{HTML}{FF7F0E}
\definecolor{rollPurple}{HTML}{7E57C2}
\definecolor{rollBrown}{HTML}{8C564B}
\definecolor{rollRed}{HTML}{D62728}
\definecolor{rollBlack}{HTML}{111111}

\pgfplotsset{
  rolloutAxis/.style={
    width=15.0cm,
    height=7.9cm,
    xlabel={Frame index},
    ymin=0,
    tick align=outside,
    tick pos=left,
    grid=both,
    major grid style={draw=black!24, dashed},
    minor grid style={draw=black!10, dotted},
    axis line style={line width=0.95pt},
    every axis plot/.append style={line join=round},
    label style={font=\bfseries\normalsize},
    tick label style={font=\bfseries\small},
    legend style={
      at={(0.985,0.985)},
      anchor=north east,
      draw=black!18,
      fill=white,
      fill opacity=0.92,
      text opacity=1,
      rounded corners=1pt,
      font=\bfseries\footnotesize,
      legend columns=2,
      /tikz/every even column/.append style={column sep=0.18cm},
      row sep=1.5pt,
      inner xsep=5pt,
      inner ysep=4pt,
    },
    legend cell align=left,
    clip mode=individual,
    unbounded coords=discard,
    scaled y ticks=false,
  },
  rolloutSRNO/.style={color=rollBlue, solid, semithick},
  rolloutEDSR/.style={color=rollTeal, dashed, semithick},
  rolloutBasicVSRPP/.style={color=rollOrange, dashdotted, semithick},
  rolloutRVRT/.style={color=rollPurple, densely dashed, semithick},
  rolloutVRT/.style={color=green!55!black, densely dashdotted, semithick},
  rolloutSDIFT/.style={color=gray!70!black, solid, line width=1.0pt},
  rolloutWDNO/.style={color=magenta!75!black, dashed, line width=1.0pt},
  rolloutResShift50M/.style={color=rollBrown, solid, line width=1.05pt},
  rolloutWRD/.style={color=rollRed, densely dashdotted, line width=1.05pt},
  rolloutWRDKCSWSDF/.style={color=rollBlack, solid, line width=1.35pt},
}
"""
    style_path.write_text(content, encoding="utf-8")


def write_compile_script(script_path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_DIR="$ROOT/tex"
PDF_DIR="$ROOT/pdf"

mkdir -p "$PDF_DIR"
cd "$TEX_DIR"

if command -v latexmk >/dev/null 2>&1; then
  for tex in *.tex; do
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$PDF_DIR" "$tex"
  done
elif command -v pdflatex >/dev/null 2>&1; then
  for tex in *.tex; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$PDF_DIR" "$tex"
  done
else
  echo "Neither latexmk nor pdflatex was found in PATH." >&2
  exit 1
fi
"""
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)


def write_readme(readme_path: Path) -> None:
    content = """# Rollout LaTeX and Qualitative Figures

This directory stores rollout curve assets and qualitative figure metadata for the project paper.

Main scripts:
- `build_rollout_latex.py`
- `render_preview_pngs.py`
- `build_focus_variants.py`
- `export_self_contained_rollout.py`
- `build_qualitative_comparison.py`
- `build_qualitative_comparison_era5_72.py`
- `compile_all.sh`

Rollout outputs:
- `data/*_error_over_time.csv`
- `data/*_rmse_over_time.csv`
- `tex/*.tex`
- `manifest.json`

Comparison outputs and metadata:
- `comparison/candidates/*.csv`
- `comparison/manifests/*.json`
- `comparison/tex/*.tex`

Regenerate rollout CSV/LaTeX files:

```bash
cd /data/yc/FluidVSR
source /home/yc/miniconda3/etc/profile.d/conda.sh
conda activate WDNO
python plotting/rollout_latex/build_rollout_latex.py --pred-root /path/to/test_predictions_root
```

Build qualitative figures:

```bash
cd /data/yc/FluidVSR
python plotting/rollout_latex/build_qualitative_comparison.py
```

Compile all LaTeX figures into PDFs:

```bash
cd /data/yc/FluidVSR
bash plotting/rollout_latex/compile_all.sh
```

Use in Overleaf:

Option 1 (recommended, simplest):
- run `compile_all.sh`
- upload the generated PDFs from `pdf/` into your Overleaf project
- include them in the main text with `\\includegraphics`

```tex
\\begin{figure}[t]
  \\centering
  \\includegraphics[width=\\linewidth]{figures/era5_rmse_over_time.pdf}
  \\caption{RMSE over time on ERA5$\\times$4.}
  \\label{fig:era5x4-rmse}
\\end{figure}
```

Option 2 (keep them as editable LaTeX/TikZ):
- upload `tex/*.tex`, `tex/rollout_plot_styles.tex`, and `data/*.csv`
- keep the same relative structure in Overleaf, for example `figures/tex/` and `figures/data/`
- in the paper preamble add:

```tex
\\usepackage{standalone}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
```

- then include a figure in the body with:

```tex
\\begin{figure}[t]
  \\centering
  \\includestandalone[width=\\linewidth]{figures/tex/era5_rmse_over_time}
  \\caption{RMSE over time on ERA5$\\times$4.}
  \\label{fig:era5x4-rmse}
\\end{figure}
```

Notes:
- `Error over time` matches the notebook's `mean absolute error` definition.
- `RMSE over time` matches the notebook's per-frame rollout RMSE definition.
- For the true `ERA5` rollout set, `WRD` is taken from `wrd_baseline_attn`.
"""
    readme_path.write_text(content, encoding="utf-8")


def maybe_compile(out_root: Path) -> None:
    compile_script = out_root / "compile_all.sh"
    subprocess.run([str(compile_script)], check=True, cwd=out_root)


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(args.out_root)
    manifest_path = paths["root"] / "manifest.json"

    write_styles_file(paths["tex"] / "rollout_plot_styles.tex")
    write_compile_script(paths["root"] / "compile_all.sh")
    write_readme(paths["root"] / "README.md")

    manifest: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for dataset_name in args.datasets:
        dataset_cfg = DATASETS[dataset_name]
        model_specs = get_model_specs(dataset_cfg)
        test_predictions_dir = resolve_test_predictions_dir(args.pred_root, dataset_cfg)
        if not test_predictions_dir.exists():
            raise FileNotFoundError(f"Missing test_predictions dir: {test_predictions_dir}")

        selected_dirs = choose_model_dirs(
            test_predictions_dir,
            model_specs=model_specs,
            model_alias_overrides=dataset_cfg.get("model_alias_overrides"),
        )
        available_keys = [spec["key"] for spec in model_specs if spec["key"] in selected_dirs]
        missing_models = [spec["display"] for spec in model_specs if spec["key"] not in selected_dirs]

        metric_curves = {metric["key"]: {} for metric in METRICS}
        source_dirs: dict[str, str] = {}
        frames_per_seq_seen: int | None = None

        for spec in model_specs:
            key = spec["key"]
            if key not in selected_dirs:
                continue

            model_dir = selected_dirs[key]
            pred_path = model_dir / "pred.npz"
            gt_path = model_dir / "gt.npz"
            meta_path = model_dir / "meta.json"

            pred = load_npz_array(pred_path)
            gt = load_npz_array(gt_path)
            frames_per_seq = get_frames_per_seq(meta_path, dataset_cfg)
            curves = compute_curves(pred, gt, frames_per_seq)

            frames_per_seq_seen = frames_per_seq
            metric_curves["error"][key] = curves["error"]
            metric_curves["rmse"][key] = curves["rmse"]
            source_dirs[spec["display"]] = str(model_dir)

        if not available_keys:
            raise RuntimeError(f"No selected model results found for dataset {dataset_name}")

        for metric in METRICS:
            metric_key = metric["key"]
            csv_filename = f"{dataset_name.lower()}_{metric_key}_over_time.csv"
            tex_filename = f"{dataset_name.lower()}_{metric_key}_over_time.tex"
            max_val = max(float(np.max(curve)) for curve in metric_curves[metric_key].values())
            pad_factor = 1.30 if str(dataset_cfg["title"]).startswith(("ERA5", "KF256")) else 1.16
            ymax = max_val * pad_factor

            write_metric_csv(paths["data"] / csv_filename, metric_curves[metric_key], model_specs)
            write_latex_plot(
                tex_path=paths["tex"] / tex_filename,
                dataset_title=dataset_cfg["title"],
                metric_title=metric["title"],
                ylabel=metric["ylabel"],
                csv_filename=csv_filename,
                available_keys=available_keys,
                model_specs=model_specs,
                ymax=ymax,
            )

        manifest[dataset_name] = {
            "dataset_title": dataset_cfg["title"],
            "frames_per_seq": frames_per_seq_seen or dataset_cfg["frames_per_seq"],
            "available_models": [spec["display"] for spec in model_specs if spec["key"] in selected_dirs],
            "missing_models": missing_models,
            "source_dirs": source_dirs,
            "csv_outputs": {
                "error": f"data/{dataset_name.lower()}_error_over_time.csv",
                "rmse": f"data/{dataset_name.lower()}_rmse_over_time.csv",
            },
            "tex_outputs": {
                "error": f"tex/{dataset_name.lower()}_error_over_time.tex",
                "rmse": f"tex/{dataset_name.lower()}_rmse_over_time.tex",
            },
        }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if args.compile:
        maybe_compile(paths["root"])
        print(f"Compiled PDFs under: {paths['pdf']}")


if __name__ == "__main__":
    main()
