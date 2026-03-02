"""
Prepare true neuro CT sources (CQ500 + RSNA ICH) for j-wave.

This script can:
1) Download RSNA ICH (Kaggle competition) if requested.
2) Download CQ500 from a Kaggle dataset slug if available, or use local path.
3) Find DICOM series directories.
4) Run:
   - brain_ct_to_jwave_acoustics.py
   - plot_brain_ct_jwave_maps.py
   for top K series from each source.

Important:
- Kaggle downloads require Kaggle API credentials configured on your machine.
- RSNA competition access requires you to have accepted competition rules.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_cmd(cmd: List[str], log_path: Path) -> Tuple[bool, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode == 0, proc.returncode


def _kaggle_cmd_base() -> List[str]:
    """
    Kaggle 2.x may not support `python -m kaggle` (no __main__).
    Prefer CLI executable, then fallback to kaggle.cli module.
    """
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    return [sys.executable, "-m", "kaggle.cli"]


def unzip_all_zips(root: Path) -> None:
    for zf in root.rglob("*.zip"):
        out_dir = zf.with_suffix("")
        if out_dir.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(out_dir)
        except zipfile.BadZipFile:
            continue


def download_rsna_competition(target_dir: Path) -> Tuple[bool, str]:
    cmd = _kaggle_cmd_base() + [
        "competitions",
        "download",
        "-c",
        "rsna-intracranial-hemorrhage-detection",
        "-p",
        str(target_dir),
    ]
    ok, rc = run_cmd(cmd, target_dir / "download_rsna.log")
    if not ok:
        return False, f"RSNA download failed (rc={rc}). See {target_dir / 'download_rsna.log'}"
    unzip_all_zips(target_dir)
    return True, "ok"


def download_cq500_dataset(target_dir: Path, kaggle_slug: str) -> Tuple[bool, str]:
    cmd = _kaggle_cmd_base() + [
        "datasets",
        "download",
        "-d",
        kaggle_slug,
        "-p",
        str(target_dir),
    ]
    ok, rc = run_cmd(cmd, target_dir / "download_cq500.log")
    if not ok:
        return False, (
            f"CQ500 download failed for slug '{kaggle_slug}' (rc={rc}). "
            f"See {target_dir / 'download_cq500.log'}. "
            "Try a different --cq500-kaggle-slug or use --cq500-local-dir."
        )
    unzip_all_zips(target_dir)
    return True, "ok"


def find_series_dirs(root: Path, min_slices: int = 20) -> List[Path]:
    """
    Find directories that look like DICOM series folders (contain many .dcm files).
    """
    counts: Dict[Path, int] = {}
    for dcm in root.rglob("*.dcm"):
        p = dcm.parent
        counts[p] = counts.get(p, 0) + 1
    series = [p for p, n in counts.items() if n >= min_slices]
    series.sort(key=lambda p: counts[p], reverse=True)
    return series


def process_series(
    source_name: str,
    series_dirs: List[Path],
    out_root: Path,
    top_k: int,
    target_shape: Optional[str],
    resample_order: int,
) -> List[Dict[str, str]]:
    code_dir = Path(__file__).resolve().parent
    convert_script = code_dir / "brain_ct_to_jwave_acoustics.py"
    plot_script = code_dir / "plot_brain_ct_jwave_maps.py"
    rows: List[Dict[str, str]] = []

    for i, sdir in enumerate(series_dirs[:top_k], start=1):
        out_dir = out_root / source_name / f"{i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        convert_cmd = [
            sys.executable,
            str(convert_script),
            "--input",
            str(sdir),
            "--out",
            str(out_dir),
            "--resample-order",
            str(resample_order),
        ]
        if target_shape:
            convert_cmd.extend(["--target-shape", target_shape])

        ok_conv, rc_conv = run_cmd(convert_cmd, out_dir / "convert.log")
        ok_plot, rc_plot = False, -1
        if ok_conv:
            plot_cmd = [sys.executable, str(plot_script), "--input-dir", str(out_dir)]
            ok_plot, rc_plot = run_cmd(plot_cmd, out_dir / "plot.log")

        rows.append(
            {
                "source": source_name,
                "rank_in_source": str(i),
                "series_dir": str(sdir),
                "out_dir": str(out_dir),
                "convert_ok": str(int(ok_conv)),
                "plot_ok": str(int(ok_plot)),
                "convert_rc": str(rc_conv),
                "plot_rc": str(rc_plot),
            }
        )
    return rows


def write_summary(rows: List[Dict[str, str]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "rank_in_source",
                "series_dir",
                "out_dir",
                "convert_ok",
                "plot_ok",
                "convert_rc",
                "plot_rc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CQ500 + RSNA brain CT for j-wave maps/plots.")
    parser.add_argument("--out-root", default="Results/brain_ct_cq500_rsna", help="Output root directory.")
    parser.add_argument("--top-k-per-source", type=int, default=10, help="How many series to process per source.")
    parser.add_argument("--target-shape", default=None, help="Optional output shape Nx,Ny,Nz (e.g. 128,128,128).")
    parser.add_argument("--resample-order", type=int, default=1, choices=[0, 1, 3])

    # RSNA options
    parser.add_argument("--rsna-download", type=int, default=1, choices=[0, 1], help="Download RSNA via Kaggle.")
    parser.add_argument("--rsna-local-dir", default=None, help="Use existing RSNA directory instead of download.")

    # CQ500 options
    parser.add_argument("--cq500-download", type=int, default=1, choices=[0, 1], help="Download CQ500 via Kaggle.")
    parser.add_argument("--cq500-local-dir", default=None, help="Use existing CQ500 directory instead of download.")
    parser.add_argument(
        "--cq500-kaggle-slug",
        default="fabiendaniel/cq500",
        help="Kaggle dataset slug for CQ500 (override if your preferred slug differs).",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    downloads_root = out_root / "_downloads"
    downloads_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []

    # ---------- RSNA ----------
    if args.rsna_local_dir:
        rsna_root = Path(args.rsna_local_dir).expanduser().resolve()
    else:
        rsna_root = downloads_root / "rsna"
        rsna_root.mkdir(parents=True, exist_ok=True)
        if args.rsna_download:
            ok, msg = download_rsna_competition(rsna_root)
            print(f"[RSNA] download: {msg}")
            if not ok:
                print("[RSNA] skipping due to download failure")
                rsna_root = None

    if rsna_root and rsna_root.exists():
        rsna_series = find_series_dirs(rsna_root)
        print(f"[RSNA] found {len(rsna_series)} candidate series dirs")
        rows = process_series(
            "rsna_ich",
            rsna_series,
            out_root,
            args.top_k_per_source,
            args.target_shape,
            args.resample_order,
        )
        all_rows.extend(rows)

    # ---------- CQ500 ----------
    if args.cq500_local_dir:
        cq_root = Path(args.cq500_local_dir).expanduser().resolve()
    else:
        cq_root = downloads_root / "cq500"
        cq_root.mkdir(parents=True, exist_ok=True)
        if args.cq500_download:
            ok, msg = download_cq500_dataset(cq_root, args.cq500_kaggle_slug)
            print(f"[CQ500] download: {msg}")
            if not ok:
                print("[CQ500] skipping due to download failure")
                cq_root = None

    if cq_root and cq_root.exists():
        cq_series = find_series_dirs(cq_root)
        print(f"[CQ500] found {len(cq_series)} candidate series dirs")
        rows = process_series(
            "cq500",
            cq_series,
            out_root,
            args.top_k_per_source,
            args.target_shape,
            args.resample_order,
        )
        all_rows.extend(rows)

    summary_csv = out_root / "summary.csv"
    write_summary(all_rows, summary_csv)
    n_ok = sum(1 for r in all_rows if r["convert_ok"] == "1" and r["plot_ok"] == "1")
    print(f"[done] processed rows: {len(all_rows)} | full success: {n_ok}")
    print(f"[done] summary: {summary_csv}")


if __name__ == "__main__":
    main()

