#!/usr/bin/env python3
"""
Batch-run brain_ct_to_jwave_acoustics_hounsfield2density.py on CQ500-style DICOM trees.

CT series selection matches ct_hu_clean_top_head_view.py:
  build_series_candidates → choose_best_series_per_subject → sort by subject_id.
By default, series with fewer than 150 slices are excluded (--min-slices).

Outputs (with --comparison-layout, default):
  OUT_ROOT/acoustic_maps/<subject_id>/
      hu_xyz.npy, sound_speed_xyz.npy, density_xyz.npy,
      attenuation_db_cm_mhz_xyz.npy, tissue_mask_xyz.npy,
      tissue_stats.csv, metadata.json
      plus aliases expected by ct_scan_comparison_analysis.py:
      <subject_id>_sound_speed_cleaned.npy, ..._density_cleaned.npy,
      ..._attenuation_cleaned.npy, <subject_id>_body_mask.npy
  OUT_ROOT/processed/<subject_id>/tissue_stats.csv

Later, point the comparison script at these dirs, e.g.:
  export CT_ACOUSTIC_DIR=".../OUT_ROOT/acoustic_maps"
  export CT_PROC_DIR=".../OUT_ROOT/processed"
  export CT_COMPARISON_OUT=".../OUT_ROOT/ct_comparison_analysis"
  python3 ct_scan_comparison_analysis.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent


def _safe_dir_name(subject_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", subject_id)


def _discover_series(dicom_root: Path, min_slices: int, max_subjects: int):
    sys.path.insert(0, str(_CODE_DIR))
    from ct_hu_clean_top_head_view import (  # noqa: E402
        build_series_candidates,
        choose_best_series_per_subject,
    )

    cands = build_series_candidates(dicom_root, min_slices=min_slices)
    chosen = choose_best_series_per_subject(cands)
    chosen.sort(key=lambda c: c.subject_id)
    if max_subjects > 0:
        chosen = chosen[:max_subjects]
    return chosen


def _write_comparison_aliases(pdir: Path, pid: str) -> None:
    """Duplicate / derive files under pdir so ct_scan_comparison_analysis discover_patients() succeeds."""
    shutil.copy2(pdir / "sound_speed_xyz.npy", pdir / f"{pid}_sound_speed_cleaned.npy")
    shutil.copy2(pdir / "density_xyz.npy", pdir / f"{pid}_density_cleaned.npy")
    shutil.copy2(pdir / "attenuation_db_cm_mhz_xyz.npy", pdir / f"{pid}_attenuation_cleaned.npy")
    tm = np.load(pdir / "tissue_mask_xyz.npy")
    np.save(pdir / f"{pid}_body_mask.npy", (tm > 0).astype(np.uint8))


def _prune_to_minimal_outputs(pdir: Path, pid: str) -> None:
    """Keep only the minimal per-patient files requested by user."""
    keep = {
        "acoustic_property_slices.png",
        "metadata.json",
        f"{pid}_sound_speed_cleaned.npy",
        f"{pid}_density_cleaned.npy",
        f"{pid}_attenuation_cleaned.npy",
        f"{pid}_body_mask.npy",
    }
    for fp in pdir.iterdir():
        if not fp.is_file():
            continue
        if fp.name not in keep:
            fp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch HU→acoustic maps (hounsfield2density mapping) for many DICOM series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dicom-root",
        type=Path,
        default=Path("/Users/shanmukasadhu/Downloads/500CTScans/qure.headct.study"),
        help="Same root as ct_hu_clean_top_head_view.py --dicom-root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density"
        ),
        help="Writes acoustic_maps/ and processed/ under this directory.",
    )
    parser.add_argument("--max-scans", type=int, default=50, help="0 = all subjects after sort.")
    parser.add_argument(
        "--min-slices",
        type=int,
        default=150,
        help="Skip CT series with fewer than this many slices (default 150).",
    )
    parser.add_argument(
        "--target-shape",
        default=None,
        help="Optional Nx,Ny,Nz resampling (e.g. 256,256,256) passed to each single-scan run.",
    )
    parser.add_argument("--resample-order", type=int, default=1, choices=[0, 1, 3])
    parser.add_argument(
        "--clean-head-mask",
        type=int,
        default=1,
        choices=[0, 1],
        help="Apply head cleaning/masking before HU->acoustic mapping.",
    )
    parser.add_argument(
        "--head-center-ellipse-frac",
        type=float,
        default=0.35,
        help="Head-mask center ellipse half-axis fraction (passed through).",
    )
    parser.add_argument(
        "--comparison-layout",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: acoustic_maps/<id>/ + processed/<id>/tissue_stats.csv + comparison aliases.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if acoustic_maps/<id>/metadata.json already exists.",
    )
    parser.add_argument(
        "--minimal-outputs",
        type=int,
        default=1,
        choices=[0, 1],
        help="Keep only prefixed cleaned arrays + body mask + slices PNG + metadata in acoustic_maps/<id>/.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dicom_root = args.dicom_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    acoustic_root = out_root / "acoustic_maps"
    proc_root = out_root / "processed"

    chosen = _discover_series(dicom_root, args.min_slices, args.max_scans)
    if not chosen:
        raise SystemExit("No CT series found. Check --dicom-root and --min-slices.")

    script = _CODE_DIR / "brain_ct_to_jwave_acoustics_hounsfield2density.py"
    if not script.is_file():
        raise SystemExit(f"Missing {script}")

    print(f"[info] DICOM root: {dicom_root}")
    print(f"[info] Subjects: {len(chosen)} (max_scans={args.max_scans})")
    print(f"[info] Out root: {out_root}")

    ok, failed = 0, []
    for c in chosen:
        pid = c.subject_id
        # Folder name and {id}_prefixed.npy must match (ct_scan_comparison uses d.name as patient id).
        sid = _safe_dir_name(pid)
        pdir = acoustic_root / sid
        if args.skip_existing and (pdir / "metadata.json").exists():
            print(f"[skip] {pid}")
            continue
        if args.comparison_layout:
            pdir.parent.mkdir(parents=True, exist_ok=True)
        else:
            pdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(c.series_dir),
            "--out",
            str(pdir),
            "--resample-order",
            str(args.resample_order),
            "--clean-head-mask",
            str(args.clean_head_mask),
            "--head-center-ellipse-frac",
            str(args.head_center_ellipse_frac),
        ]
        if args.target_shape:
            cmd += ["--target-shape", args.target_shape]

        print(f"[run] {pid} -> {pdir}")
        if args.dry_run:
            print("       ", " ".join(cmd))
            continue

        r = subprocess.run(cmd, cwd=str(_CODE_DIR))
        if r.returncode != 0:
            failed.append(pid)
            print(f"[err] {pid} exit {r.returncode}")
            continue

        if args.comparison_layout:
            _write_comparison_aliases(pdir, sid)
            proc_dir = proc_root / sid
            proc_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdir / "tissue_stats.csv", proc_dir / "tissue_stats.csv")
        if args.minimal_outputs:
            _prune_to_minimal_outputs(pdir, sid)
        ok += 1

    print(f"[done] succeeded={ok} failed={len(failed)}")
    if failed:
        print("[done] failed ids:", ", ".join(failed))


if __name__ == "__main__":
    main()
