"""
Create a multimodal CT/MRI + acoustic report for one patient from a DICOM root.

Given a patient ID (e.g. ACRIN-FMISO-Brain-001), this script:
1) Finds CT and MR series for that patient
2) Loads CT as HU volume and MR as intensity volume
3) Builds acoustic maps from CT HU (sound speed, density, attenuation)
4) Saves requested plots and additional sanity plots

Example:
  python3 Code/plot_kaggle_patient_ct_mri_report.py \
    --patient-id "ACRIN-FMISO-Brain-001" \
    --dicom-root "Results/kaggle_brain_tumor_mri_ct/unzipped/dicom"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from brain_ct_to_jwave_acoustics import map_acoustic_properties


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.upper()
    return any(k in t for k in keywords)


def _series_blob(meta: Dict) -> str:
    return " | ".join(
        [
            str(meta.get("series_desc", "")),
            str(meta.get("study_desc", "")),
            str(meta.get("protocol", "")),
            str(meta.get("modality", "")),
            str(meta.get("body_part", "")),
        ]
    ).upper()


def discover_patient_series(dicom_root: Path, patient_id: str) -> Dict[str, Dict]:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required. Install with: pip install pydicom") from exc

    series: Dict[str, Dict] = defaultdict(
        lambda: {
            "modality": "",
            "count": 0,
            "files": [],
            "series_desc": "",
            "study_desc": "",
            "protocol": "",
            "body_part": "",
        }
    )

    for fp in dicom_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(ds, "PatientID", "")) != patient_id:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if not uid:
            continue
        rec = series[uid]
        rec["count"] += 1
        rec["files"].append(fp)
        if not rec["modality"]:
            rec["modality"] = str(getattr(ds, "Modality", ""))
        if not rec["series_desc"]:
            rec["series_desc"] = str(getattr(ds, "SeriesDescription", ""))
        if not rec["study_desc"]:
            rec["study_desc"] = str(getattr(ds, "StudyDescription", ""))
        if not rec["protocol"]:
            rec["protocol"] = str(getattr(ds, "ProtocolName", ""))
        if not rec["body_part"]:
            rec["body_part"] = str(getattr(ds, "BodyPartExamined", ""))

    if not series:
        raise ValueError(f"No DICOM series found for patient '{patient_id}' under {dicom_root}")
    return series


def select_best_series(series: Dict[str, Dict], modality: str) -> Tuple[str, Dict]:
    modality = modality.upper()
    candidates = [(uid, meta) for uid, meta in series.items() if str(meta.get("modality", "")).upper() == modality]
    if not candidates:
        raise ValueError(f"No {modality} series found for this patient.")

    head_kw = ["HEAD", "BRAIN", "AX", "AXIAL", "T1", "T2", "FLAIR", "WO", "W/O"]
    penalty_kw = ["SCOUT", "LOCALIZER", "MPR", "LOC", "CALIB"]

    ranked = []
    for uid, meta in candidates:
        blob = _series_blob(meta)
        score = float(meta.get("count", 0))
        if _contains_any(blob, head_kw):
            score += 50.0
        if _contains_any(blob, penalty_kw):
            score -= 80.0
        ranked.append((score, uid, meta))
    ranked.sort(key=lambda t: t[0], reverse=True)
    _, uid, meta = ranked[0]
    return uid, meta


def load_volume_from_files(files: List[Path], modality: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    import pydicom

    slices = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
        except Exception:
            continue
        if not hasattr(ds, "pixel_array"):
            continue

        arr = ds.pixel_array.astype(np.float32)
        if modality.upper() == "CT":
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

        z = None
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            z = float(ipp[2])
        if z is None:
            z = float(getattr(ds, "InstanceNumber", len(slices)))

        px = getattr(ds, "PixelSpacing", [1.0, 1.0])
        sy, sx = float(px[0]), float(px[1])
        st = float(getattr(ds, "SliceThickness", 1.0))
        slices.append((z, arr, (sx, sy, st)))

    if not slices:
        raise ValueError(f"No readable image slices for modality {modality}.")

    slices.sort(key=lambda t: t[0])
    vol = np.stack([s[1] for s in slices], axis=2)
    vol = np.transpose(vol, (1, 0, 2))

    sx, sy, st = slices[0][2]
    if len(slices) > 1:
        dz = np.median(np.diff([s[0] for s in slices]))
        if np.isfinite(dz) and dz > 0:
            st = float(abs(dz))
    return vol, (sx, sy, st)


def _slice_triplet(vol_xyz: np.ndarray, xyz: Tuple[int, int, int]):
    x, y, z = xyz
    xy = vol_xyz[:, :, z].T
    xz = vol_xyz[:, y, :].T
    yz = vol_xyz[x, :, :].T
    return xy, xz, yz


def _robust_limits(vol: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> Tuple[float, float]:
    vals = vol[np.isfinite(vol)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(vals))
        hi = float(np.max(vals) + 1e-6)
    return lo, hi


def _mri_normalized(mri_xyz: np.ndarray) -> np.ndarray:
    vals = mri_xyz[np.isfinite(mri_xyz)]
    if vals.size == 0:
        return np.zeros_like(mri_xyz, dtype=np.float32)
    lo, hi = np.percentile(vals, [1.0, 99.0])
    out = (mri_xyz - lo) / max(1e-6, hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def plot_slices(
    arr_xyz: np.ndarray,
    title: str,
    unit: str,
    cmap: str,
    out_path: Path,
    spacing_xyz_mm: Tuple[float, float, float],
) -> None:
    nx, ny, nz = arr_xyz.shape
    center = (nx // 2, ny // 2, nz // 2)
    xy, xz, yz = _slice_triplet(arr_xyz, center)
    sx, sy, sz = spacing_xyz_mm
    extent_xy = [0, nx * sx, 0, ny * sy]
    extent_xz = [0, nx * sx, 0, nz * sz]
    extent_yz = [0, ny * sy, 0, nz * sz]
    vmin, vmax = _robust_limits(arr_xyz)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im = axes[0].imshow(xy, origin="lower", cmap=cmap, extent=extent_xy, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title} XY")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, label=unit)

    im = axes[1].imshow(xz, origin="lower", cmap=cmap, extent=extent_xz, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title} XZ")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, label=unit)

    im = axes[2].imshow(yz, origin="lower", cmap=cmap, extent=extent_yz, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"{title} YZ")
    axes[2].set_xlabel("y (mm)")
    axes[2].set_ylabel("z (mm)")
    plt.colorbar(im, ax=axes[2], fraction=0.046, label=unit)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_joint_overview(
    ct_hu_xyz: np.ndarray,
    mri_xyz: np.ndarray,
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    att_xyz: np.ndarray,
    out_path: Path,
) -> None:
    ct_center = (ct_hu_xyz.shape[0] // 2, ct_hu_xyz.shape[1] // 2, ct_hu_xyz.shape[2] // 2)
    mri_center = (mri_xyz.shape[0] // 2, mri_xyz.shape[1] // 2, mri_xyz.shape[2] // 2)
    ct_xy, _, _ = _slice_triplet(ct_hu_xyz, ct_center)
    mri_xy, _, _ = _slice_triplet(mri_xyz, mri_center)
    c_xy, _, _ = _slice_triplet(c_xyz, ct_center)
    rho_xy, _, _ = _slice_triplet(rho_xyz, ct_center)
    att_xy, _, _ = _slice_triplet(att_xyz, ct_center)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    items = [
        ("CT HU (center XY)", ct_xy, "gray"),
        ("MRI normalized (center XY)", mri_xy, "gray"),
        ("Sound speed (m/s)", c_xy, "viridis"),
        ("Density (kg/m^3)", rho_xy, "viridis"),
        ("Attenuation (dB/cm/MHz)", att_xy, "magma"),
    ]
    for ax, (title, img, cmap) in zip(axes.reshape(-1), items):
        vmin, vmax = _robust_limits(img)
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Last panel: HU->acoustics scatter sample
    ax = axes.reshape(-1)[5]
    n = ct_hu_xyz.size
    idx = np.random.default_rng(0).choice(n, size=min(30000, n), replace=False)
    hu_s = ct_hu_xyz.reshape(-1)[idx]
    c_s = c_xyz.reshape(-1)[idx]
    ax.scatter(hu_s, c_s, s=2, alpha=0.2)
    ax.set_title("HU vs sound speed")
    ax.set_xlabel("HU")
    ax.set_ylabel("m/s")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(
    ct_hu_xyz: np.ndarray,
    mri_xyz: np.ndarray,
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    att_xyz: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    items = [
        ("CT HU", ct_hu_xyz, "HU"),
        ("MRI normalized", mri_xyz, "a.u."),
        ("Sound speed", c_xyz, "m/s"),
        ("Density", rho_xyz, "kg/m^3"),
        ("Attenuation", att_xyz, "dB/cm/MHz"),
    ]
    for ax, (title, arr, unit) in zip(axes.reshape(-1), items):
        vals = arr[np.isfinite(arr)].ravel()
        if vals.size == 0:
            ax.set_title(f"{title} (no finite data)")
            continue
        lo, hi = np.percentile(vals, [0.5, 99.5])
        vals = vals[(vals >= lo) & (vals <= hi)]
        ax.hist(vals, bins=100, alpha=0.85)
        ax.set_title(f"{title} histogram")
        ax.set_xlabel(unit)
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
    axes.reshape(-1)[5].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ct_3d(ct_hu_xyz: np.ndarray, spacing_xyz_mm: Tuple[float, float, float], out_path: Path) -> None:
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise ImportError("scikit-image is required for 3D mesh plots. Install with: pip install scikit-image") from exc

    sx, sy, sz = spacing_xyz_mm
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    levels = [
        (35.0, "#5DADE2", 0.20, "brain-ish"),
        (700.0, "#F5B041", 0.35, "skull"),
    ]
    for level, color, alpha, _ in levels:
        try:
            verts, faces, _, _ = measure.marching_cubes(ct_hu_xyz, level=level)
        except (ValueError, RuntimeError):
            continue
        verts_mm = verts * np.array([sx, sy, sz])[None, :]
        mesh = Poly3DCollection(verts_mm[faces], alpha=alpha, facecolor=color, edgecolor="none")
        ax.add_collection3d(mesh)

    ax.set_xlim(0, ct_hu_xyz.shape[0] * sx)
    ax.set_ylim(0, ct_hu_xyz.shape[1] * sy)
    ax.set_zlim(0, ct_hu_xyz.shape[2] * sz)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("CT 3D isosurfaces (brain-ish + skull)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mri_3d(mri_xyz: np.ndarray, spacing_xyz_mm: Tuple[float, float, float], out_path: Path) -> None:
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise ImportError("scikit-image is required for 3D mesh plots. Install with: pip install scikit-image") from exc

    sx, sy, sz = spacing_xyz_mm
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    finite = mri_xyz[np.isfinite(mri_xyz)]
    if finite.size == 0:
        raise ValueError("MRI volume has no finite values for 3D plotting.")
    p75 = float(np.percentile(finite, 75.0))
    p90 = float(np.percentile(finite, 90.0))
    levels = [
        (p75, "#85C1E9", 0.12),
        (p90, "#1F618D", 0.30),
    ]
    for level, color, alpha in levels:
        try:
            verts, faces, _, _ = measure.marching_cubes(mri_xyz, level=level)
        except (ValueError, RuntimeError):
            continue
        verts_mm = verts * np.array([sx, sy, sz])[None, :]
        mesh = Poly3DCollection(verts_mm[faces], alpha=alpha, facecolor=color, edgecolor="none")
        ax.add_collection3d(mesh)

    ax.set_xlim(0, mri_xyz.shape[0] * sx)
    ax.set_ylim(0, mri_xyz.shape[1] * sy)
    ax.set_zlim(0, mri_xyz.shape[2] * sz)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("MRI 3D isosurfaces (p75/p90)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CT/MRI/acoustic report for one Kaggle DICOM patient.")
    parser.add_argument("--patient-id", required=True, help='Patient ID, e.g. "ACRIN-FMISO-Brain-001".')
    parser.add_argument(
        "--dicom-root",
        default="Results/kaggle_brain_tumor_mri_ct/unzipped/dicom",
        help="Root directory containing patient DICOM files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: Results/kaggle_patient_reports/<patient-id>/).",
    )
    args = parser.parse_args()

    dicom_root = Path(args.dicom_root).expanduser().resolve()
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root does not exist: {dicom_root}")

    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.patient_id)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (Path("Results/kaggle_patient_reports").resolve() / safe_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    series = discover_patient_series(dicom_root, args.patient_id)
    ct_uid, ct_meta = select_best_series(series, modality="CT")
    mr_uid, mr_meta = select_best_series(series, modality="MR")

    ct_hu_xyz, ct_spacing = load_volume_from_files(ct_meta["files"], modality="CT")
    mr_xyz_raw, mr_spacing = load_volume_from_files(mr_meta["files"], modality="MR")
    mr_xyz = _mri_normalized(mr_xyz_raw)

    c_xyz, rho_xyz, att_xyz, _ = map_acoustic_properties(ct_hu_xyz)

    # Save arrays for reuse.
    np.save(out_dir / "ct_hu_xyz.npy", ct_hu_xyz.astype(np.float32))
    np.save(out_dir / "mri_norm_xyz.npy", mr_xyz.astype(np.float32))
    np.save(out_dir / "sound_speed_xyz.npy", c_xyz.astype(np.float32))
    np.save(out_dir / "density_xyz.npy", rho_xyz.astype(np.float32))
    np.save(out_dir / "attenuation_db_cm_mhz_xyz.npy", att_xyz.astype(np.float32))

    # Requested plots.
    plot_slices(ct_hu_xyz, "CT scan (HU)", "HU", "gray", out_dir / "ct_scan_slices.png", ct_spacing)
    plot_slices(mr_xyz, "MRI (normalized)", "a.u.", "gray", out_dir / "mri_scan_slices.png", mr_spacing)
    plot_slices(ct_hu_xyz, "HU", "HU", "gray", out_dir / "hu_slices.png", ct_spacing)
    plot_slices(
        att_xyz,
        "Attenuation",
        "dB/cm/MHz",
        "magma",
        out_dir / "attenuation_slices.png",
        ct_spacing,
    )
    plot_slices(rho_xyz, "Density", "kg/m^3", "viridis", out_dir / "density_slices.png", ct_spacing)
    plot_slices(c_xyz, "Sound speed", "m/s", "viridis", out_dir / "sound_speed_slices.png", ct_spacing)

    # Additional relevant plots.
    plot_histograms(ct_hu_xyz, mr_xyz, c_xyz, rho_xyz, att_xyz, out_dir / "histograms.png")
    plot_joint_overview(ct_hu_xyz, mr_xyz, c_xyz, rho_xyz, att_xyz, out_dir / "joint_overview.png")
    plot_ct_3d(ct_hu_xyz, ct_spacing, out_dir / "ct_3d_isosurfaces.png")
    plot_mri_3d(mr_xyz, mr_spacing, out_dir / "mri_3d_isosurfaces.png")

    summary = {
        "patient_id": args.patient_id,
        "dicom_root": str(dicom_root),
        "ct_series_uid": ct_uid,
        "ct_series_desc": ct_meta.get("series_desc", ""),
        "ct_slice_count": int(ct_meta.get("count", 0)),
        "mr_series_uid": mr_uid,
        "mr_series_desc": mr_meta.get("series_desc", ""),
        "mr_slice_count": int(mr_meta.get("count", 0)),
        "ct_shape_xyz": list(ct_hu_xyz.shape),
        "ct_spacing_xyz_mm": list(ct_spacing),
        "mr_shape_xyz": list(mr_xyz.shape),
        "mr_spacing_xyz_mm": list(mr_spacing),
        "outputs": [
            "ct_scan_slices.png",
            "mri_scan_slices.png",
            "hu_slices.png",
            "attenuation_slices.png",
            "density_slices.png",
            "sound_speed_slices.png",
            "histograms.png",
            "joint_overview.png",
            "ct_3d_isosurfaces.png",
            "mri_3d_isosurfaces.png",
        ],
    }
    (out_dir / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[ok] Generated patient report.")
    print(f"[ok] Output directory: {out_dir}")
    print(f"[info] CT series: {ct_uid} | {ct_meta.get('series_desc', '')}")
    print(f"[info] MR series: {mr_uid} | {mr_meta.get('series_desc', '')}")


if __name__ == "__main__":
    main()

