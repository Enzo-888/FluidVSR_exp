from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_extra_metrics import compute_rollout_extra_metrics


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "RBC": {
        "root": Path("/data_new/FluidVSR_data/expriments/RBC/×4/test_predictions"),
        "metric_dataset_name": "RBC",
        "frames_per_seq": 100,
    },
    "KF256": {
        "root": Path("/data_new/FluidVSR_data/expriments/KF256/×4/test_predictions"),
        "metric_dataset_name": "KF256",
        "frames_per_seq": 180,
    },
    "SW": {
        "root": Path("/data_new/FluidVSR_data/expriments/SW/×4/test_predictions"),
        "metric_dataset_name": "SW",
        "frames_per_seq": 72,
    },
    "ERA5_72frames": {
        "root": Path("/data_new/FluidVSR_data/expriments/ERA5_72frames/x4/test_predictions"),
        "metric_dataset_name": "ERA5_72frames",
        "frames_per_seq": 72,
    },
}

SUMMARY_ORDER_GROUPS: list[list[str]] = [
    ["bilinear"],
    ["fno"],
    ["srno"],
    ["swinir"],
    ["edsr"],
    ["resshift112M"],
    ["remd112M"],
    ["resshift50M", "resshift50m"],
    ["remd50M", "remd50m"],
    ["basicvsr++"],
    ["vrt"],
    ["rvrt"],
    ["sdift", "sdift_rbc_x4", "sdift_sw_x4", "sdift_era5_x4"],
    ["wdno"],
    ["WRD", "wrd"],
    ["WRD_sobo"],
    ["WRD_KCS"],
    ["WRD_sgf"],
    ["WRD_sgf_KCS"],
    ["WRD_sgf_WSDF"],
    ["WRD_sgf_KCS+WSDF", "wrd_baseline_attn"],
]

SUMMARY_ORDER_RANK: dict[str, int] = {
    alias: idx for idx, aliases in enumerate(SUMMARY_ORDER_GROUPS) for alias in aliases
}


def format_scientific(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value == 0:
        return "0.000×10^0"
    mantissa, exp = f"{value:.3e}".split("e")
    return f"{mantissa}×10^{int(exp)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-compute ε_spec / ε_lo / ε_hi / L_Diff L1 from pred/gt npz folders."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Datasets to process. Default: all.",
    )
    parser.add_argument(
        "--summary_root",
        type=Path,
        default=Path("/data/yc/Fluid_VSR/spectrum_metric_reports"),
        help="Directory for per-dataset markdown/csv summaries.",
    )
    parser.add_argument(
        "--per_model_json_name",
        default="spectrum_metrics.json",
        help="Filename to write inside each model folder.",
    )
    parser.add_argument(
        "--merge_into_results",
        action="store_true",
        help="Also merge the four metrics into existing results.json in each model folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-model spectrum_metrics.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N model folders per dataset.",
    )
    return parser.parse_args()


def load_npz_array(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    try:
        if "data" in payload:
            arr = payload["data"]
        else:
            first_key = list(payload.keys())[0]
            arr = payload[first_key]
    finally:
        payload.close()

    arr = np.asarray(arr)
    if arr.ndim == 5:
        arr = arr.reshape(-1, *arr.shape[2:])
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D NHWC or 5D NTHWC array in {path}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_model_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "pred.npz").exists() and (child / "gt.npz").exists() and (child / "meta.json").exists():
            dirs.append(child)
    return dirs


def resolve_frames_per_seq(meta: dict[str, Any], expected: int, model_dir: Path) -> tuple[int, int | None, str]:
    if expected <= 1:
        raise ValueError(f"Invalid dataset frames_per_seq={expected}")

    meta_value = meta.get("frames_per_seq")
    if meta_value is None:
        return expected, None, "dataset_default"

    actual = int(meta_value)
    if actual <= 1:
        raise ValueError(f"Invalid meta.json frames_per_seq={actual} in {model_dir}")
    if actual != expected:
        print(
            f"  warning: {model_dir.name} overrides frames_per_seq "
            f"from dataset default {expected} to meta.json value {actual}"
        )
        return actual, actual, "meta_override"
    return expected, actual, "meta_match"


def compute_one_model(
    *,
    dataset_key: str,
    model_dir: Path,
    metric_dataset_name: str,
    frames_per_seq: int,
) -> dict[str, Any]:
    pred = load_npz_array(model_dir / "pred.npz")
    gt = load_npz_array(model_dir / "gt.npz")
    lr_exists = (model_dir / "lr.npz").exists()
    meta = load_json(model_dir / "meta.json")
    expected_frames_per_seq = frames_per_seq

    frames_per_seq, meta_frames_per_seq, frames_per_seq_source = resolve_frames_per_seq(
        meta, frames_per_seq, model_dir
    )
    usable_frames = (min(len(pred), len(gt)) // frames_per_seq) * frames_per_seq
    if usable_frames == 0:
        raise ValueError(
            f"Not enough frames for one full sequence in {model_dir}: "
            f"pred={len(pred)}, gt={len(gt)}, frames_per_seq={frames_per_seq}"
        )

    metrics = compute_rollout_extra_metrics(
        pred,
        gt,
        frames_per_seq=frames_per_seq,
        dataset_name=metric_dataset_name,
    )
    return {
        "dataset": dataset_key,
        "metric_dataset_name": metric_dataset_name,
        "model_name": model_dir.name,
        "model_dir": str(model_dir),
        "pred_shape": list(pred.shape),
        "gt_shape": list(gt.shape),
        "lr_present": lr_exists,
        "frames_per_seq": frames_per_seq,
        "expected_frames_per_seq": expected_frames_per_seq,
        "meta_frames_per_seq": meta_frames_per_seq,
        "frames_per_seq_source": frames_per_seq_source,
        "usable_frames": int(usable_frames),
        "usable_sequences": int(usable_frames // frames_per_seq),
        **metrics,
    }


def write_model_json(
    *,
    model_dir: Path,
    output_name: str,
    payload: dict[str, Any],
    overwrite: bool,
    merge_into_results: bool,
) -> None:
    output_path = model_dir / output_name
    if output_path.exists() and not overwrite:
        existing = load_json(output_path)
        merged = dict(existing)
        merged.update(payload)
        output_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    else:
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if merge_into_results:
        results_path = model_dir / "results.json"
        results = load_json(results_path) if results_path.exists() else {}
        for key in ("eps_spec", "eps_lo", "eps_hi", "ldiff_l1"):
            results[key] = payload[key]
        results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def write_dataset_summary(dataset_key: str, rows: list[dict[str, Any]], summary_root: Path) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            SUMMARY_ORDER_RANK.get(str(row["model_name"]), len(SUMMARY_ORDER_RANK)),
            str(row["model_name"]).lower(),
        ),
    )

    out_dir = summary_root / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{dataset_key}_spectrum_metrics.csv"
    md_path = out_dir / f"{dataset_key}_spectrum_metrics.md"

    fieldnames = [
        "model_name",
        "eps_spec",
        "eps_lo",
        "eps_hi",
        "ldiff_l1",
        "frames_per_seq",
        "usable_sequences",
        "model_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": row["model_name"],
                    "eps_spec": format_scientific(float(row["eps_spec"])),
                    "eps_lo": format_scientific(float(row["eps_lo"])),
                    "eps_hi": format_scientific(float(row["eps_hi"])),
                    "ldiff_l1": format_scientific(float(row["ldiff_l1"])),
                    "frames_per_seq": row["frames_per_seq"],
                    "usable_sequences": row["usable_sequences"],
                    "model_dir": row["model_dir"],
                }
            )

    lines = [
        f"# {dataset_key} Spectrum Metrics",
        "",
        "| Model | ε_spec | ε_lo | ε_hi | L_Diff L1 | frames/seq | n_seq |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | "
            f"{format_scientific(float(row['eps_spec']))} | "
            f"{format_scientific(float(row['eps_lo']))} | "
            f"{format_scientific(float(row['eps_hi']))} | "
            f"{format_scientific(float(row['ldiff_l1']))} | "
            f"{row['frames_per_seq']} | "
            f"{row['usable_sequences']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(
    *,
    dataset_key: str,
    summary_root: Path,
    per_model_json_name: str,
    merge_into_results: bool,
    overwrite: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    cfg = DATASET_CONFIGS[dataset_key]
    root = cfg["root"]
    metric_dataset_name = cfg["metric_dataset_name"]
    frames_per_seq = int(cfg["frames_per_seq"])

    if not root.exists():
        raise FileNotFoundError(root)

    model_dirs = find_model_dirs(root)
    if limit is not None:
        model_dirs = model_dirs[:limit]

    print(f"\n=== {dataset_key} ===")
    print(f"root: {root}")
    print(f"models: {len(model_dirs)}")

    rows: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        row = compute_one_model(
            dataset_key=dataset_key,
            model_dir=model_dir,
            metric_dataset_name=metric_dataset_name,
            frames_per_seq=frames_per_seq,
        )
        write_model_json(
            model_dir=model_dir,
            output_name=per_model_json_name,
            payload=row,
            overwrite=overwrite,
            merge_into_results=merge_into_results,
        )
        rows.append(row)
        print(
            f"  {model_dir.name:24s} "
            f"eps_spec={row['eps_spec']:.6f} "
            f"eps_lo={row['eps_lo']:.6f} "
            f"eps_hi={row['eps_hi']:.6f} "
            f"ldiff_l1={row['ldiff_l1']:.6f}"
        )

    write_dataset_summary(dataset_key, rows, summary_root)
    return rows


def main() -> None:
    args = parse_args()
    datasets = list(DATASET_CONFIGS.keys()) if args.datasets == ["all"] else args.datasets

    for dataset_key in datasets:
        run_dataset(
            dataset_key=dataset_key,
            summary_root=args.summary_root,
            per_model_json_name=args.per_model_json_name,
            merge_into_results=args.merge_into_results,
            overwrite=args.overwrite,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
