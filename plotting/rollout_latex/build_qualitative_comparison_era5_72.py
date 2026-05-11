#!/usr/bin/env python3
"""Generate qualitative comparison panels for the ERA5_72 setup.

Inputs:
  - ResShift50M predictions
  - WRD(full) predictions

Outputs are written into:
  ./qualitative

This keeps the original multi-dataset script unchanged and provides a
single-dataset entry point for the 72-frame ERA5 experiment.
"""

from __future__ import annotations

from pathlib import Path

import build_qualitative_comparison as base


SPEC = base.DatasetSpec(
    key="era5_72",
    display="ERA5_72",
    short_caption="ERA5_72",
    priority=1,
    resshift_dir=Path("/home/yc/transfer_to_a100/x4_test_predictions/ERA5_72frames/test_predictions/resshift50m"),
    wrd_dir=Path("/home/yc/WRD-main/flow_sr_resshift_wavelet/_runs/era5v100_diffusion_resshift_swin3d_align_physnorm_mc64_kappa1p5_850hpa3day_nonoverlap_bilinearx4_attn32_64_128_win16_det0_ddp4_a100_bs1_20260429T130026Z/wrd_baseline_attn"),
    frames_per_seq=72,
)


def main() -> None:
    for path in [base.ROOT, base.CANDIDATE_DIR, base.PANEL_DIR, base.TEX_DIR, base.MANIFEST_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    (base.PANEL_DIR / SPEC.key).mkdir(parents=True, exist_ok=True)

    manifest = base.process_dataset(SPEC)
    print(f"Processed {SPEC.display}")
    print(f"best_png={manifest['best_png']}")
    print(f"best_tex={manifest['best_tex']}")
    print(f"candidate_csv={manifest['candidate_csv']}")


if __name__ == "__main__":
    main()
