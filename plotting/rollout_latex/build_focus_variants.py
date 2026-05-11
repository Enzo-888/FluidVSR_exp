#!/usr/bin/env python3
"""Build focused LaTeX/PNG figure variants without touching the current versions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TEX_DIR = ROOT / "tex"
PNG_DIR = ROOT / "png"
MANIFEST_PATH = ROOT / "manifest.json"
TITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 12
LEGEND_SIZE = 11

STYLE_MAP = {
    "srno": {
        "label": "SRNO",
        "color": "#1F77B4",
        "linestyle": "-",
        "linewidth": 2.0,
        "tex_style": "rolloutSRNO",
    },
    "edsr": {
        "label": "EDSR",
        "color": "#0F766E",
        "linestyle": "--",
        "linewidth": 2.0,
        "tex_style": "rolloutEDSR",
    },
    "basicvsrpp": {
        "label": "BasicVSR++",
        "color": "#FF7F0E",
        "linestyle": "-.",
        "linewidth": 2.0,
        "tex_style": "rolloutBasicVSRPP",
    },
    "rvrt": {
        "label": "RVRT",
        "color": "#7E57C2",
        "linestyle": (0, (4, 2)),
        "linewidth": 2.0,
        "tex_style": "rolloutRVRT",
    },
    "vrt": {
        "label": "VRT",
        "color": "#2CA02C",
        "linestyle": (0, (4, 1.5, 1, 1.5)),
        "linewidth": 2.0,
        "tex_style": "rolloutVRT",
    },
    "sdift": {
        "label": "SDIFT",
        "color": "#6B7280",
        "linestyle": "-",
        "linewidth": 2.0,
        "tex_style": "rolloutSDIFT",
    },
    "wdno": {
        "label": "WDNO",
        "color": "#C2185B",
        "linestyle": "--",
        "linewidth": 2.0,
        "tex_style": "rolloutWDNO",
    },
    "resshift50m": {
        "label": "ResShift",
        "color": "#8C564B",
        "linestyle": "-",
        "linewidth": 2.2,
        "tex_style": "rolloutResShift50M",
    },
    "wrd": {
        "label": "WRD",
        "color": "#D62728",
        "linestyle": (0, (4, 1.5, 1, 1.5)),
        "linewidth": 2.2,
        "tex_style": "rolloutWRD",
    },
    "wrd_kcs_wsdf": {
        "label": "WRD",
        "color": "#111111",
        "linestyle": "-",
        "linewidth": 2.8,
        "tex_style": "rolloutWRDKCSWSDF",
    },
}

METRIC_META = {
    "error": {
        "ylabel": "Mean absolute error",
        "title": "Error over time",
    },
    "rmse": {
        "ylabel": "RMSE",
        "title": "RMSE over time",
    },
}

VARIANTS = [
    {
        "dataset": "ERA5_24frames",
        "metric": "error",
        "suffix": "zoom",
        "exclude": [],
        "title_suffix": "",
    },
    {
        "dataset": "ERA5_24frames",
        "metric": "rmse",
        "suffix": "zoom",
        "exclude": [],
        "title_suffix": "",
    },
    {
        "dataset": "ERA5",
        "metric": "error",
        "suffix": "zoom",
        "exclude": [],
        "title_suffix": "",
    },
    {
        "dataset": "ERA5",
        "metric": "rmse",
        "suffix": "zoom",
        "exclude": [],
        "title_suffix": "",
    },
    {
        "dataset": "KF256",
        "metric": "error",
        "suffix": "no_basicvsrpp_zoom",
        "exclude": ["basicvsrpp"],
        "title_suffix": "",
    },
    {
        "dataset": "KF256",
        "metric": "rmse",
        "suffix": "no_basicvsrpp_zoom",
        "exclude": ["basicvsrpp"],
        "title_suffix": "",
    },
    {
        "dataset": "KF256_video",
        "metric": "error",
        "suffix": "no_sdift_zoom",
        "exclude": ["sdift"],
        "title_suffix": "",
    },
    {
        "dataset": "KF256_video",
        "metric": "rmse",
        "suffix": "no_sdift_zoom",
        "exclude": ["sdift"],
        "title_suffix": "",
    },
]


def read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[int], dict[str, list[float]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        frame_idx: list[int] = []
        series: dict[str, list[float]] = {k: [] for k in reader.fieldnames if k != "frame_idx"}
        for row in reader:
            frame_idx.append(int(row["frame_idx"]))
            for key in series:
                series[key].append(float(row[key]))
    return frame_idx, series


def compute_tight_ylim(series: dict[str, list[float]]) -> tuple[float, float]:
    vals = [v for curve in series.values() for v in curve]
    vmin = min(vals)
    vmax = max(vals)
    span = max(vmax - vmin, 1e-6)
    pad = max(span * 0.18, 0.015 * vmax if vmax > 0 else 0.01)
    ymin = max(0.0, vmin - pad)
    ymax = vmax + pad
    return ymin, ymax


def latex_text(text: str) -> str:
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
    return escaped.replace("×", r"$\times$")


def write_tex(
    tex_path: Path,
    dataset_title: str,
    metric_key: str,
    csv_relpath: str,
    selected_keys: list[str],
    ymin: float,
    ymax: float,
    title_suffix: str,
) -> None:
    meta = METRIC_META[metric_key]
    lines = []
    for key in selected_keys:
        style = STYLE_MAP[key]
        lines.append(
            "\\addplot+[{}] table[x=frame_idx,y={},col sep=comma] {{{}}};\n"
            "\\addlegendentry{{{}}}".format(
                style["tex_style"], key, csv_relpath, style["label"]
            )
        )
    content = f"""\\documentclass[tikz,border=3pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\input{{rollout_plot_styles.tex}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
    rolloutAxis,
    ylabel={{{latex_text(meta["ylabel"])}}},
    ymin={ymin:.6f},
    ymax={ymax:.6f},
]
{chr(10).join(lines)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
    tex_path.write_text(content, encoding="utf-8")


def render_png(
    png_path: Path,
    dataset_title: str,
    metric_key: str,
    frame_idx: list[int],
    selected_series: dict[str, list[float]],
    ymin: float,
    ymax: float,
    title_suffix: str,
) -> None:
    meta = METRIC_META[metric_key]
    fig, ax = plt.subplots(figsize=(11.4, 6.3))
    for key, values in selected_series.items():
        style = STYLE_MAP[key]
        ax.plot(
            frame_idx,
            values,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    ax.set_xlabel("Frame index", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(meta["ylabel"], fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_xlim(frame_idx[0], frame_idx[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.65, alpha=0.18)
    ax.minorticks_on()
    ax.tick_params(labelsize=TICK_SIZE, width=1.0, length=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.92,
        fontsize=LEGEND_SIZE,
        prop={"weight": "bold", "size": LEGEND_SIZE},
        handlelength=2.6,
        columnspacing=0.9,
        borderpad=0.45,
        labelspacing=0.45,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    manifest = read_manifest()
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        dataset = variant["dataset"]
        metric = variant["metric"]
        suffix = variant["suffix"]
        exclude = set(variant["exclude"])
        title_suffix = variant["title_suffix"]

        csv_path = DATA_DIR / f"{dataset.lower()}_{metric}_over_time.csv"
        frame_idx, series = read_csv(csv_path)
        selected_keys = [key for key in series if key not in exclude]
        selected_series = {key: series[key] for key in selected_keys}
        ymin, ymax = compute_tight_ylim(selected_series)

        dataset_title = manifest[dataset]["dataset_title"]
        tex_name = f"{dataset.lower()}_{metric}_over_time_{suffix}.tex"
        png_name = f"{dataset.lower()}_{metric}_over_time_{suffix}.png"

        write_tex(
            tex_path=TEX_DIR / tex_name,
            dataset_title=dataset_title,
            metric_key=metric,
            csv_relpath=f"../data/{csv_path.name}",
            selected_keys=selected_keys,
            ymin=ymin,
            ymax=ymax,
            title_suffix=title_suffix,
        )
        render_png(
            png_path=PNG_DIR / png_name,
            dataset_title=dataset_title,
            metric_key=metric,
            frame_idx=frame_idx,
            selected_series=selected_series,
            ymin=ymin,
            ymax=ymax,
            title_suffix=title_suffix,
        )
        print(f"Saved: {TEX_DIR / tex_name}")
        print(f"Saved: {PNG_DIR / png_name}")


if __name__ == "__main__":
    main()
