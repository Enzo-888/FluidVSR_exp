#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DATASET_CONFIGS = {
    "RBC": {
        4: {
            "test_predictions_root": Path("/data_new/FluidVSR_data/expriments/RBC/×4/test_predictions"),
            "reference_dir": "WRD",
        },
        8: {
            "test_predictions_root": Path("/data_new/FluidVSR_data/expriments/RBC/×8/test_prections"),
            "reference_dir": "wrd",
        },
    },
    "SW": {
        4: {
            "test_predictions_root": Path("/data_new/FluidVSR_data/expriments/SW/×4/test_predictions"),
            "reference_dir": "WRD",
        },
        8: {
            "test_predictions_root": Path("/data_new/FluidVSR_data/expriments/SW/×8/test_predictions"),
            "reference_dir": "wrd",
        },
    },
    "KF256": {
        4: {
            "test_predictions_root": Path("/data_new/FluidVSR_data/expriments/KF256/×4/test_predictions"),
            "reference_dir": "WRD",
        },
        8: {
            "test_predictions_root": Path("/data_new/FluidVSR_data/expriments/KF256/×8/test_predictions"),
            "reference_dir": "wrd",
        },
    },
    "ERA5": {
        4: {
            "test_predictions_root": Path("/data_new/FluidVSR_data/expriments/ERA5_72frames/x4/test_predictions"),
            "reference_dir": "wrd",
        },
        8: {
            "test_predictions_root": Path("/data_new/FluidVSR_data/expriments/ERA5_72frames/x8/test_predictions"),
            "reference_dir": "wrd_baseline",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create bilinear baseline predictions from WRD-format test_predictions folders.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=sorted(DATASET_CONFIGS.keys()),
        default=sorted(DATASET_CONFIGS.keys()),
        help="Subset of datasets to process.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=[4, 8],
        default=4,
        help="Super-resolution scale whose test_predictions roots should be used.",
    )
    parser.add_argument(
        "--output_dir_name",
        default="bilinear",
        help="Name of the output folder under each test_predictions root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folders.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Frame batch size for bilinear interpolation.",
    )
    return parser.parse_args()


def load_npz_data(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    try:
        if "data" in payload:
            arr = payload["data"]
        else:
            first_key = list(payload.keys())[0]
            arr = payload[first_key]
    finally:
        payload.close()
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D NHWC array in {path}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def bilinear_upsample_nhwc(lr_arr: np.ndarray, out_hw: tuple[int, int], batch_size: int) -> np.ndarray:
    n, _, _, c = lr_arr.shape
    out_h, out_w = out_hw
    out_chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        chunk = torch.from_numpy(lr_arr[start:stop]).permute(0, 3, 1, 2)
        up = F.interpolate(
            chunk,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        out_chunks.append(up.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(out_chunks, axis=0)


def ensure_reference_files(reference_dir: Path) -> tuple[Path, Path, Path]:
    gt_path = reference_dir / "gt.npz"
    lr_path = reference_dir / "lr.npz"
    meta_path = reference_dir / "meta.json"
    missing = [str(p) for p in (gt_path, lr_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reference files in {reference_dir}: {missing}")
    return gt_path, lr_path, meta_path


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_one(dataset_name: str, root: Path, reference_dir_name: str, output_dir_name: str, overwrite: bool, batch_size: int) -> Path:
    reference_dir = root / reference_dir_name
    output_dir = root / output_dir_name
    gt_path, lr_path, meta_path = ensure_reference_files(reference_dir)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_if_needed(gt_path, output_dir / "gt.npz")
    copy_if_needed(lr_path, output_dir / "lr.npz")
    copy_if_needed(meta_path, output_dir / "meta.json")

    gt_arr = load_npz_data(gt_path)
    lr_arr = load_npz_data(lr_path)
    if gt_arr.shape[0] != lr_arr.shape[0]:
        raise ValueError(f"Frame count mismatch for {dataset_name}: gt {gt_arr.shape}, lr {lr_arr.shape}")

    pred_arr = bilinear_upsample_nhwc(
        lr_arr,
        out_hw=(gt_arr.shape[1], gt_arr.shape[2]),
        batch_size=batch_size,
    )
    np.savez_compressed(output_dir / "pred.npz", data=pred_arr)

    summary = {
        "source_dataset": dataset_name,
        "reference_dir": str(reference_dir),
        "output_dir": str(output_dir),
        "reference_gt_shape": list(gt_arr.shape),
        "reference_lr_shape": list(lr_arr.shape),
        "pred_shape": list(pred_arr.shape),
        "method": "framewise bilinear upsampling from WRD reference lr.npz",
    }
    (output_dir / "bilinear_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    args = parse_args()
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIGS[dataset_name][args.scale]
        out_dir = build_one(
            dataset_name=f"{dataset_name}x{args.scale}",
            root=cfg["test_predictions_root"],
            reference_dir_name=cfg["reference_dir"],
            output_dir_name=args.output_dir_name,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
        )
        print(f"[done] {dataset_name}: {out_dir}")


if __name__ == "__main__":
    main()
