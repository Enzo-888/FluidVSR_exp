#!/usr/bin/env python3
"""Export rollout figures as self-contained TeX snippets plus PNG previews.

Each generated TeX file is a single figure snippet that can be uploaded to
Overleaf without any external CSV or shared style file dependency.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = ROOT / "manifest.json"
SELF_TEX_DIR = ROOT / "self_contained_tex"
SELF_PNG_DIR = ROOT / "self_contained_png"

PNG_LABEL_SIZE = 16
PNG_TICK_SIZE = 12
PNG_LEGEND_SIZE = 11

STYLE_MAP = {
    "srno": {
        "label": "SRNO",
        "color": "#1F77B4",
        "tex_color": "rollBlue",
        "linestyle": "-",
        "tex_style": "solid",
        "linewidth": 2.0,
        "tex_width": "semithick",
    },
    "edsr": {
        "label": "EDSR",
        "color": "#0F766E",
        "tex_color": "rollTeal",
        "linestyle": "--",
        "tex_style": "dashed",
        "linewidth": 2.0,
        "tex_width": "semithick",
    },
    "basicvsrpp": {
        "label": "BasicVSR++",
        "color": "#FF7F0E",
        "tex_color": "rollOrange",
        "linestyle": "-.",
        "tex_style": "dashdotted",
        "linewidth": 2.0,
        "tex_width": "semithick",
    },
    "rvrt": {
        "label": "RVRT",
        "color": "#7E57C2",
        "tex_color": "rollPurple",
        "linestyle": (0, (4, 2)),
        "tex_style": "densely dashed",
        "linewidth": 2.0,
        "tex_width": "semithick",
    },
    "vrt": {
        "label": "VRT",
        "color": "#2CA02C",
        "tex_color": "rollGreen",
        "linestyle": (0, (4, 1.5, 1, 1.5)),
        "tex_style": "densely dashdotted",
        "linewidth": 2.0,
        "tex_width": "semithick",
    },
    "sdift": {
        "label": "SDIFT",
        "color": "#6B7280",
        "tex_color": "rollGray",
        "linestyle": "-",
        "tex_style": "solid",
        "linewidth": 2.0,
        "tex_width": "line width=1.00pt",
    },
    "wdno": {
        "label": "WDNO",
        "color": "#C2185B",
        "tex_color": "rollMagenta",
        "linestyle": "--",
        "tex_style": "dashed",
        "linewidth": 2.0,
        "tex_width": "line width=1.00pt",
    },
    "resshift50m": {
        "label": "ResShift",
        "color": "#8C564B",
        "tex_color": "rollBrown",
        "linestyle": "-",
        "tex_style": "solid",
        "linewidth": 2.2,
        "tex_width": "line width=1.05pt",
    },
    "wrd": {
        "label": "WRD",
        "color": "#D62728",
        "tex_color": "rollRed",
        "linestyle": (0, (4, 1.5, 1, 1.5)),
        "tex_style": "densely dashdotted",
        "linewidth": 2.2,
        "tex_width": "line width=1.05pt",
    },
    "wrd_kcs_wsdf": {
        "label": "WRD",
        "color": "#111111",
        "tex_color": "rollBlack",
        "linestyle": "-",
        "tex_style": "solid",
        "linewidth": 2.8,
        "tex_width": "line width=1.35pt",
    },
}

METRIC_META = {
    "error": {
        "ylabel": "Mean absolute error",
        "caption": "Error over time",
    },
    "rmse": {
        "ylabel": "RMSE",
        "caption": "RMSE over time",
    },
}

FOCUS_VARIANTS = [
    {
        "dataset": "ERA5_24frames",
        "metric": "error",
        "suffix": "zoom",
        "exclude": [],
        "caption_suffix": " (focused range)",
    },
    {
        "dataset": "ERA5_24frames",
        "metric": "rmse",
        "suffix": "zoom",
        "exclude": [],
        "caption_suffix": " (focused range)",
    },
    {
        "dataset": "ERA5",
        "metric": "error",
        "suffix": "zoom",
        "exclude": [],
        "caption_suffix": " (focused range)",
    },
    {
        "dataset": "ERA5",
        "metric": "rmse",
        "suffix": "zoom",
        "exclude": [],
        "caption_suffix": " (focused range)",
    },
    {
        "dataset": "KF256",
        "metric": "error",
        "suffix": "no_basicvsrpp_zoom",
        "exclude": ["basicvsrpp"],
        "caption_suffix": " (focused range without BasicVSR++)",
    },
    {
        "dataset": "KF256",
        "metric": "rmse",
        "suffix": "no_basicvsrpp_zoom",
        "exclude": ["basicvsrpp"],
        "caption_suffix": " (focused range without BasicVSR++)",
    },
    {
        "dataset": "KF256_video",
        "metric": "error",
        "suffix": "no_sdift_zoom",
        "exclude": ["sdift"],
        "caption_suffix": " (focused range without SDIFT)",
    },
    {
        "dataset": "KF256_video",
        "metric": "rmse",
        "suffix": "no_sdift_zoom",
        "exclude": ["sdift"],
        "caption_suffix": " (focused range without SDIFT)",
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


def latex_text(text: str) -> str:
    return latex_escape(text).replace("×", r"$\times$")


def compute_main_ylim(dataset_title: str, series: dict[str, list[float]]) -> tuple[float, float]:
    all_values = [v for curve in series.values() for v in curve]
    max_val = max(all_values)
    pad_factor = 1.30 if dataset_title.startswith(("ERA5", "KF256")) else 1.16
    return 0.0, max_val * pad_factor


def compute_tight_ylim(series: dict[str, list[float]]) -> tuple[float, float]:
    vals = [v for curve in series.values() for v in curve]
    vmin = min(vals)
    vmax = max(vals)
    span = max(vmax - vmin, 1e-6)
    pad = max(span * 0.18, 0.015 * vmax if vmax > 0 else 0.01)
    ymin = max(0.0, vmin - pad)
    ymax = vmax + pad
    return ymin, ymax


def format_coordinates(frame_idx: list[int], values: list[float]) -> str:
    lines = [f"({x},{y:.10f})" for x, y in zip(frame_idx, values)]
    return "\n        ".join(lines)


def build_definecolors() -> str:
    color_lines = []
    for key in STYLE_MAP:
        style = STYLE_MAP[key]
        color_lines.append(
            f"\\definecolor{{{style['tex_color']}}}{{HTML}}{{{style['color'].lstrip('#')}}}"
        )
    return "\n".join(color_lines)


def build_addplot_block(frame_idx: list[int], key: str, values: list[float]) -> str:
    style = STYLE_MAP[key]
    coordinates = format_coordinates(frame_idx, values)
    return (
        f"      \\addplot+[mark=none, {style['tex_width']}, color={style['tex_color']}, {style['tex_style']}] coordinates {{\n"
        f"        {coordinates}\n"
        f"      }};\n"
        f"      \\addlegendentry{{{latex_text(style['label'])}}}"
    )


def build_axis_block(
    frame_idx: list[int],
    series: dict[str, list[float]],
    ylabel: str,
    ymin: float,
    ymax: float,
) -> str:
    blocks = [build_addplot_block(frame_idx, key, values) for key, values in series.items()]
    return f"""    \\begin{{axis}}[
      width=0.92\\linewidth,
      height=0.52\\linewidth,
      xlabel={{Frame index}},
      ylabel={{{latex_text(ylabel)}}},
      xmin={frame_idx[0]}, xmax={frame_idx[-1]},
      ymin={ymin:.10f}, ymax={ymax:.10f},
      tick align=outside,
      tick pos=left,
      grid=both,
      major grid style={{draw=black!24, dashed}},
      minor grid style={{draw=black!10, dotted}},
      axis line style={{line width=0.95pt}},
      label style={{font=\\bfseries\\normalsize}},
      tick label style={{font=\\bfseries\\small}},
      legend style={{
        at={{(0.985,0.985)}},
        anchor=north east,
        draw=black!18,
        fill=white,
        fill opacity=0.92,
        text opacity=1,
        rounded corners=1pt,
        font=\\bfseries\\footnotesize,
        legend columns=2,
        /tikz/every even column/.append style={{column sep=0.18cm}},
        row sep=1.5pt,
        inner xsep=5pt,
        inner ysep=4pt,
      }},
      legend cell align=left,
      clip mode=individual,
      unbounded coords=discard,
      scaled y ticks=false,
    ]
{chr(10).join(blocks)}
    \\end{{axis}}"""


def write_self_contained_tex(
    tex_path: Path,
    frame_idx: list[int],
    series: dict[str, list[float]],
    dataset_title: str,
    metric_key: str,
    ymin: float,
    ymax: float,
    caption_suffix: str = "",
) -> None:
    axis_block = build_axis_block(frame_idx, series, METRIC_META[metric_key]["ylabel"], ymin, ymax)
    caption = f"{METRIC_META[metric_key]['caption']} on {dataset_title}{caption_suffix}."
    label = f"fig:rollout-{tex_path.stem.replace('_', '-')}"
    content = f"""%% Auto-generated by export_self_contained_rollout.py
{build_definecolors()}
\\begin{{figure}}[t]
  \\centering
  \\begin{{tikzpicture}}
{axis_block}
  \\end{{tikzpicture}}
  \\caption{{{latex_text(caption)}}}
  \\label{{{label}}}
\\end{{figure}}
"""
    tex_path.write_text(content, encoding="utf-8")


def render_png(
    png_path: Path,
    frame_idx: list[int],
    series: dict[str, list[float]],
    metric_key: str,
    ymin: float,
    ymax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 6.3))
    for key, values in series.items():
        style = STYLE_MAP[key]
        ax.plot(
            frame_idx,
            values,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    ax.set_xlabel("Frame index", fontsize=PNG_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(METRIC_META[metric_key]["ylabel"], fontsize=PNG_LABEL_SIZE, fontweight="bold")
    ax.set_xlim(frame_idx[0], frame_idx[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.65, alpha=0.18)
    ax.minorticks_on()
    ax.tick_params(labelsize=PNG_TICK_SIZE, width=1.0, length=5)
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
        fontsize=PNG_LEGEND_SIZE,
        prop={"weight": "bold", "size": PNG_LEGEND_SIZE},
        handlelength=2.6,
        columnspacing=0.9,
        borderpad=0.45,
        labelspacing=0.45,
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    SELF_TEX_DIR.mkdir(parents=True, exist_ok=True)
    SELF_PNG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()

    for dataset_name, dataset_meta in manifest.items():
        dataset_title = dataset_meta["dataset_title"]
        for metric_key in METRIC_META:
            csv_path = DATA_DIR / f"{dataset_name.lower()}_{metric_key}_over_time.csv"
            frame_idx, series = read_csv(csv_path)
            ymin, ymax = compute_main_ylim(dataset_title, series)
            base_name = f"{dataset_name.lower()}_{metric_key}_over_time"
            write_self_contained_tex(
                tex_path=SELF_TEX_DIR / f"{base_name}.tex",
                frame_idx=frame_idx,
                series=series,
                dataset_title=dataset_title,
                metric_key=metric_key,
                ymin=ymin,
                ymax=ymax,
            )
            render_png(
                png_path=SELF_PNG_DIR / f"{base_name}.png",
                frame_idx=frame_idx,
                series=series,
                metric_key=metric_key,
                ymin=ymin,
                ymax=ymax,
            )
            print(f"Saved: {SELF_TEX_DIR / f'{base_name}.tex'}")
            print(f"Saved: {SELF_PNG_DIR / f'{base_name}.png'}")

    for variant in FOCUS_VARIANTS:
        dataset = variant["dataset"]
        metric = variant["metric"]
        suffix = variant["suffix"]
        exclude = set(variant["exclude"])
        dataset_title = manifest[dataset]["dataset_title"]
        csv_path = DATA_DIR / f"{dataset.lower()}_{metric}_over_time.csv"
        frame_idx, raw_series = read_csv(csv_path)
        series = {key: values for key, values in raw_series.items() if key not in exclude}
        ymin, ymax = compute_tight_ylim(series)
        base_name = f"{dataset.lower()}_{metric}_over_time_{suffix}"
        write_self_contained_tex(
            tex_path=SELF_TEX_DIR / f"{base_name}.tex",
            frame_idx=frame_idx,
            series=series,
            dataset_title=dataset_title,
            metric_key=metric,
            ymin=ymin,
            ymax=ymax,
            caption_suffix=variant["caption_suffix"],
        )
        render_png(
            png_path=SELF_PNG_DIR / f"{base_name}.png",
            frame_idx=frame_idx,
            series=series,
            metric_key=metric,
            ymin=ymin,
            ymax=ymax,
        )
        print(f"Saved: {SELF_TEX_DIR / f'{base_name}.tex'}")
        print(f"Saved: {SELF_PNG_DIR / f'{base_name}.png'}")


if __name__ == "__main__":
    main()
