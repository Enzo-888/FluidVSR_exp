#!/usr/bin/env python3
"""Select and render qualitative comparison panels for paper figures.

Outputs under:
  ./comparison

For each dataset:
  - ranked top20 candidate CSV
  - best panel PNG in comparison/panels/{dataset}_best.png
  - individual candidate PNGs in comparison/panels/{dataset}/
  - LaTeX wrapper for the best PNG

Panel layout:
  - left: ground-truth HR spanning two rows
  - top row: bilinear, ResShift, WRD(full)
  - bottom row: pointwise absolute error maps for the three methods
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR / "comparison"
ROLLOUT_ROOT = THIS_DIR
CANDIDATE_DIR = ROOT / "candidates"
PANEL_DIR = ROOT / "panels"
TEX_DIR = ROOT / "tex"
MANIFEST_DIR = ROOT / "manifests"
FONT_FAMILY = "DejaVu Serif"
TITLE_SIZE_GT = 18
TITLE_SIZE_PRED = 16
CBAR_LABEL_SIZE = 12
CBAR_TICK_SIZE = 11
ERR_PERCENTILE = 98.0

CANDIDATE_COLUMNS = [
    "rank",
    "global_frame",
    "seq_id",
    "t",
    "score",
    "selection_mode",
    "rmse_bilinear",
    "rmse_resshift50m",
    "rmse_wrd_full",
    "mae_bilinear",
    "mae_resshift50m",
    "mae_wrd_full",
    "rel_improve_vs_resshift",
    "rel_improve_vs_bilinear",
    "structure",
    "panel_png",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    short_caption: str
    priority: int
    resshift_dir: Path
    wrd_dir: Path
    frames_per_seq: int


DATASETS = [
    DatasetSpec(
        key="rbc",
        display="RBC",
        short_caption="RBC",
        priority=1,
        resshift_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/RBC/test_predictions/resshift50M"),
        wrd_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/RBC/test_predictions/WRD_KCS+WSDF"),
        frames_per_seq=100,
    ),
    DatasetSpec(
        key="sw",
        display="SW",
        short_caption="SW",
        priority=2,
        resshift_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/SW/test_predictions/resshift50M"),
        wrd_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/SW/test_predictions/WRD_sgf_KCS+WSDF"),
        frames_per_seq=72,
    ),
    DatasetSpec(
        key="era5",
        display="ERA5",
        short_caption="ERA5",
        priority=3,
        resshift_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/ERA5V/test_predictions/ResShift50M"),
        wrd_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/ERA5V/test_predictions/WRD_KCS+WSDF"),
        frames_per_seq=24,
    ),
    DatasetSpec(
        key="kf256",
        display="KF256",
        short_caption="KF256",
        priority=4,
        resshift_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/KF256/test_predictions/resshift50M"),
        wrd_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/KF256/test_predictions/WRD_sgf_KCS+WSDF"),
        frames_per_seq=180,
    ),
]


def ensure_dirs() -> None:
    for path in [ROOT, CANDIDATE_DIR, PANEL_DIR, TEX_DIR, MANIFEST_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for spec in DATASETS:
        (PANEL_DIR / spec.key).mkdir(parents=True, exist_ok=True)


def load_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        arr = npz["data"] if "data" in npz else npz[npz.files[0]]
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    return arr.astype(np.float32, copy=False)


def bilinear_upsample(lr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(lr).permute(0, 3, 1, 2)
    upsampled = F.interpolate(tensor, size=out_hw, mode="bilinear", align_corners=False)
    return upsampled.permute(0, 2, 3, 1).numpy()


def framewise_rmse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.sqrt(((pred - gt) ** 2).mean(axis=(1, 2, 3)))


def framewise_mae(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.abs(pred - gt).mean(axis=(1, 2, 3))


def framewise_structure(gt: np.ndarray) -> np.ndarray:
    frames = gt[..., 0]
    dx = np.diff(frames, axis=2, append=frames[:, :, -1:])
    dy = np.diff(frames, axis=1, append=frames[:, -1:, :])
    return np.mean(np.abs(dx), axis=(1, 2)) + np.mean(np.abs(dy), axis=(1, 2))


def normalize01(x: np.ndarray) -> np.ndarray:
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


def select_diverse_candidates(
    fps: int,
    score: np.ndarray,
    allow_mask: np.ndarray,
    structure: np.ndarray,
    top_k: int = 20,
    min_time_gap: int = 4,
    max_per_seq: int = 4,
) -> list[int]:
    idx_sorted = np.argsort(-score)
    structure_thresh = float(np.quantile(structure, 0.50))
    selected: list[int] = []
    per_seq_count: dict[int, int] = {}
    selected_times: dict[int, list[int]] = {}

    for idx in idx_sorted:
        if not allow_mask[idx]:
            continue
        if structure[idx] < structure_thresh:
            continue

        seq_id = int(idx // fps)
        t = int(idx % fps)

        if per_seq_count.get(seq_id, 0) >= max_per_seq:
            continue
        if any(abs(t - prev_t) < min_time_gap for prev_t in selected_times.get(seq_id, [])):
            continue

        selected.append(int(idx))
        per_seq_count[seq_id] = per_seq_count.get(seq_id, 0) + 1
        selected_times.setdefault(seq_id, []).append(t)
        if len(selected) >= top_k:
            break

    return selected


def write_candidate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def rel_to_rollout_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROLLOUT_ROOT))
    except ValueError:
        return str(path)


def resolve_from_rollout_root(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROLLOUT_ROOT / path


def render_panel(
    png_path: Path,
    state_vmin: float,
    state_vmax: float,
    err_vmax: float,
    gt_frame: np.ndarray,
    bilinear_frame: np.ndarray,
    resshift_frame: np.ndarray,
    wrd_frame: np.ndarray,
) -> None:
    gt_2d = np.asarray(gt_frame, dtype=np.float32)
    bilinear_2d = np.asarray(bilinear_frame, dtype=np.float32)
    resshift_2d = np.asarray(resshift_frame, dtype=np.float32)
    wrd_2d = np.asarray(wrd_frame, dtype=np.float32)

    err_bilinear = np.abs(bilinear_2d - gt_2d)
    err_resshift = np.abs(resshift_2d - gt_2d)
    err_wrd = np.abs(wrd_2d - gt_2d)

    fig = plt.figure(figsize=(15.2, 6.0))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.28, 1.0, 1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0],
        wspace=0.06,
        hspace=0.08,
    )

    ax_gt = fig.add_subplot(grid[:, 0])
    axes_pred = [fig.add_subplot(grid[0, col]) for col in range(1, 4)]
    axes_err = [fig.add_subplot(grid[1, col]) for col in range(1, 4)]
    cax_state = fig.add_subplot(grid[0, 4])
    cax_err = fig.add_subplot(grid[1, 4])

    im_state = ax_gt.imshow(gt_2d, cmap="RdBu_r", vmin=state_vmin, vmax=state_vmax)
    ax_gt.set_title(
        "Ground truth HR",
        fontsize=TITLE_SIZE_GT,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
        pad=12,
    )
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])

    pred_titles = ["Bilinear upsampling", "ResShift\u2020", "WRD (full)"]
    pred_frames = [bilinear_2d, resshift_2d, wrd_2d]
    for axis, title, frame in zip(axes_pred, pred_titles, pred_frames):
        axis.imshow(frame, cmap="RdBu_r", vmin=state_vmin, vmax=state_vmax)
        axis.set_title(
            title,
            fontsize=TITLE_SIZE_PRED,
            fontweight="bold",
            fontfamily=FONT_FAMILY,
            pad=12,
        )
        axis.set_xticks([])
        axis.set_yticks([])

    err_frames = [err_bilinear, err_resshift, err_wrd]
    err_image = None
    for axis, frame in zip(axes_err, err_frames):
        err_image = axis.imshow(frame, cmap="hot", vmin=0.0, vmax=err_vmax)
        axis.set_xticks([])
        axis.set_yticks([])

    state_cbar = fig.colorbar(im_state, cax=cax_state)
    state_cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    state_cbar.set_label(
        "Physical value",
        fontsize=CBAR_LABEL_SIZE,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
    )

    err_cbar = fig.colorbar(err_image, cax=cax_err)
    err_cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    err_cbar.set_label(
        "Absolute error",
        fontsize=CBAR_LABEL_SIZE,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
    )

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_tex_wrapper(
    tex_path: Path,
    dataset_title: str,
    panel_relpath: str,
    seq_id: int,
    t: int,
) -> None:
    content = f"""% Figure snippet for direct inclusion in LaTeX
\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.98\\linewidth]{{{panel_relpath}}}
  \\caption{{Qualitative comparison on a representative {dataset_title} test sample ($t={t}$, seq {seq_id}). From left to right: ground truth HR, bilinear upsampling, ResShift$^\\dagger$, and WRD (full). The bottom row shows pointwise absolute error.}}
  \\label{{fig:{dataset_title.lower()}_qualitative_best}}
\\end{{figure}}
"""
    tex_path.write_text(content, encoding="utf-8")


def process_dataset(spec: DatasetSpec) -> dict[str, object]:
    gt = load_npz_array(spec.resshift_dir / "gt.npz")
    lr = load_npz_array(spec.resshift_dir / "lr.npz")
    resshift = load_npz_array(spec.resshift_dir / "pred.npz")
    wrd = load_npz_array(spec.wrd_dir / "pred.npz")
    bilinear = bilinear_upsample(lr, out_hw=gt.shape[1:3])

    rmse_bilinear = framewise_rmse(bilinear, gt)
    rmse_resshift = framewise_rmse(resshift, gt)
    rmse_wrd = framewise_rmse(wrd, gt)

    mae_bilinear = framewise_mae(bilinear, gt)
    mae_resshift = framewise_mae(resshift, gt)
    mae_wrd = framewise_mae(wrd, gt)

    structure = framewise_structure(gt)
    structure_norm = normalize01(structure)

    improve_res = (rmse_resshift - rmse_wrd) / np.maximum(rmse_resshift, 1e-8)
    improve_bi = (rmse_bilinear - rmse_wrd) / np.maximum(rmse_bilinear, 1e-8)

    strict_score = improve_res + 0.45 * improve_bi + 0.15 * structure_norm
    relaxed_score = 1.25 * improve_res + 0.20 * improve_bi + 0.20 * structure_norm

    strict_mask = (rmse_wrd < rmse_resshift) & (rmse_wrd < rmse_bilinear)
    better_than_res_mask = rmse_wrd < rmse_resshift
    better_than_bi_mask = rmse_wrd < rmse_bilinear

    selection_mode = "strict_wrd_best"
    status = "ok"
    selected_idx = select_diverse_candidates(
        fps=spec.frames_per_seq,
        score=strict_score,
        allow_mask=strict_mask,
        structure=structure,
        top_k=20,
        min_time_gap=max(2, spec.frames_per_seq // 24),
        max_per_seq=4,
    )

    if not selected_idx:
        selection_mode = "relaxed_better_than_bilinear"
        status = "relaxed_no_strict_candidate"
        selected_idx = select_diverse_candidates(
            fps=spec.frames_per_seq,
            score=relaxed_score,
            allow_mask=better_than_bi_mask,
            structure=structure,
            top_k=20,
            min_time_gap=max(2, spec.frames_per_seq // 24),
            max_per_seq=4,
        )

    if not selected_idx:
        selection_mode = "relaxed_any_frame"
        status = "relaxed_no_bilinear_margin"
        selected_idx = select_diverse_candidates(
            fps=spec.frames_per_seq,
            score=relaxed_score,
            allow_mask=np.ones_like(strict_mask, dtype=bool),
            structure=structure,
            top_k=20,
            min_time_gap=max(2, spec.frames_per_seq // 24),
            max_per_seq=4,
        )

    dataset_panel_dir = PANEL_DIR / spec.key
    seq_scale_cache: dict[int, tuple[float, float, float]] = {}
    rows: list[dict[str, object]] = []
    for rank, idx in enumerate(selected_idx, start=1):
        seq_id = int(idx // spec.frames_per_seq)
        t = int(idx % spec.frames_per_seq)
        candidate_png = dataset_panel_dir / f"rank{rank:02d}_seq{seq_id:03d}_t{t:03d}.png"

        rows.append(
            {
                "rank": rank,
                "global_frame": int(idx),
                "seq_id": seq_id,
                "t": t,
                "score": f"{strict_score[idx] if selection_mode == 'strict_wrd_best' else relaxed_score[idx]:.8f}",
                "selection_mode": selection_mode,
                "rmse_bilinear": f"{rmse_bilinear[idx]:.8f}",
                "rmse_resshift50m": f"{rmse_resshift[idx]:.8f}",
                "rmse_wrd_full": f"{rmse_wrd[idx]:.8f}",
                "mae_bilinear": f"{mae_bilinear[idx]:.8f}",
                "mae_resshift50m": f"{mae_resshift[idx]:.8f}",
                "mae_wrd_full": f"{mae_wrd[idx]:.8f}",
                "rel_improve_vs_resshift": f"{improve_res[idx]:.8f}",
                "rel_improve_vs_bilinear": f"{improve_bi[idx]:.8f}",
                "structure": f"{structure[idx]:.8f}",
                "panel_png": rel_to_rollout_root(candidate_png),
            }
        )

    csv_path = CANDIDATE_DIR / f"{spec.key}_top20.csv"
    write_candidate_csv(csv_path, rows)

    best_png = None
    best_tex = None
    if rows:
        for row in rows:
            idx = int(row["global_frame"])
            seq_id = int(row["seq_id"])
            if seq_id not in seq_scale_cache:
                seq_start = seq_id * spec.frames_per_seq
                seq_end = seq_start + spec.frames_per_seq
                gt_seq = gt[seq_start:seq_end, ..., 0]
                bi_seq = bilinear[seq_start:seq_end, ..., 0]
                rs_seq = resshift[seq_start:seq_end, ..., 0]
                wrd_seq = wrd[seq_start:seq_end, ..., 0]
                err_values = np.concatenate(
                    [
                        np.abs(bi_seq - gt_seq).ravel(),
                        np.abs(rs_seq - gt_seq).ravel(),
                        np.abs(wrd_seq - gt_seq).ravel(),
                    ]
                )
                seq_scale_cache[seq_id] = (
                    float(gt_seq.min()),
                    float(gt_seq.max()),
                    float(max(np.percentile(err_values, ERR_PERCENTILE), 1e-8)),
                )
            state_vmin, state_vmax, err_vmax = seq_scale_cache[seq_id]
            render_panel(
                png_path=resolve_from_rollout_root(str(row["panel_png"])),
                state_vmin=state_vmin,
                state_vmax=state_vmax,
                err_vmax=err_vmax,
                gt_frame=gt[idx, ..., 0],
                bilinear_frame=bilinear[idx, ..., 0],
                resshift_frame=resshift[idx, ..., 0],
                wrd_frame=wrd[idx, ..., 0],
            )

        best = rows[0]
        best_idx = int(best["global_frame"])
        best_seq = int(best["seq_id"])
        best_t = int(best["t"])
        state_vmin, state_vmax, err_vmax = seq_scale_cache[best_seq]

        best_png = PANEL_DIR / f"{spec.key}_best.png"
        render_panel(
            png_path=best_png,
            state_vmin=state_vmin,
            state_vmax=state_vmax,
            err_vmax=err_vmax,
            gt_frame=gt[best_idx, ..., 0],
            bilinear_frame=bilinear[best_idx, ..., 0],
            resshift_frame=resshift[best_idx, ..., 0],
            wrd_frame=wrd[best_idx, ..., 0],
        )

        best_tex = TEX_DIR / f"{spec.key}_best.tex"
        write_tex_wrapper(
            tex_path=best_tex,
            dataset_title=spec.short_caption,
            panel_relpath=f"../panels/{best_png.name}",
            seq_id=best_seq,
            t=best_t,
        )

    manifest = {
        "dataset": spec.display,
        "priority": spec.priority,
        "status": status,
        "selection_mode": selection_mode,
        "strict_candidate_count": int(strict_mask.sum()),
        "better_than_resshift_count": int(better_than_res_mask.sum()),
        "better_than_bilinear_count": int(better_than_bi_mask.sum()),
        "resshift_dir": str(spec.resshift_dir),
        "wrd_dir": str(spec.wrd_dir),
        "candidate_csv": rel_to_rollout_root(csv_path),
        "candidate_panel_dir": rel_to_rollout_root(dataset_panel_dir),
        "best_png": rel_to_rollout_root(best_png) if best_png is not None else None,
        "best_tex": rel_to_rollout_root(best_tex) if best_tex is not None else None,
        "selected_candidates": rows[:20],
    }
    manifest_path = MANIFEST_DIR / f"{spec.key}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ensure_dirs()
    combined: list[dict[str, object]] = []
    for spec in sorted(DATASETS, key=lambda item: item.priority):
        combined.append(process_dataset(spec))
        print(f"Processed {spec.display}")

    summary_path = MANIFEST_DIR / "summary.json"
    summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
