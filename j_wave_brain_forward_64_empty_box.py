"""
Step-1 CT visualization script:
- Load one patient's CT volume from DICOM
- Use CT as the full scene (no extra domain, no lens box)
- Convert HU to acoustic property maps
- Save one image containing XY / XZ / YZ slices for:
    * Sound speed
    * Density
    * Attenuation

Example:
  python3 Code/j_wave_brain_forward_64_empty_box.py \
    --patient-id "ACRIN-FMISO-Brain-001" \
    --dicom-root "Results/kaggle_brain_tumor_mri_ct/unzipped/dicom"
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from brain_ct_to_jwave_acoustics import map_acoustic_properties


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.upper()
    return any(k in t for k in keywords)


def discover_ct_series_for_patient(dicom_root: Path, patient_id: str) -> Dict[str, Dict]:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required. Install with: pip install pydicom") from exc

    records: Dict[str, Dict] = defaultdict(
        lambda: {
            "count": 0,
            "desc": "",
            "study_desc": "",
            "protocol": "",
            "files": [],
        }
    )

    for fp in dicom_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(ds, "PatientID", "")) != patient_id:
            continue
        if str(getattr(ds, "Modality", "")).upper() != "CT":
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if not uid:
            continue
        rec = records[uid]
        rec["count"] += 1
        rec["files"].append(fp)
        if not rec["desc"]:
            rec["desc"] = str(getattr(ds, "SeriesDescription", ""))
        if not rec["study_desc"]:
            rec["study_desc"] = str(getattr(ds, "StudyDescription", ""))
        if not rec["protocol"]:
            rec["protocol"] = str(getattr(ds, "ProtocolName", ""))

    if not records:
        raise ValueError(f"No CT series found for patient '{patient_id}' under {dicom_root}")
    return records


def pick_best_ct_series(records: Dict[str, Dict]) -> Tuple[str, Dict]:
    head_kw = ["HEAD", "BRAIN", "AX", "AXIAL", "ROUTINE", "STD", "WO", "W/O", "NEURO", "CRANI"]
    bad_kw = ["SCOUT", "LOCALIZER", "MPR", "LOC", "LUNG", "CHEST", "SPINE"]

    ranked = []
    for uid, rec in records.items():
        blob = " | ".join([rec.get("desc", ""), rec.get("study_desc", ""), rec.get("protocol", "")]).upper()
        score = float(rec["count"])
        if _contains_any(blob, head_kw):
            score += 60.0
        if _contains_any(blob, bad_kw):
            score -= 100.0
        ranked.append((score, uid))
    ranked.sort(reverse=True)
    best_uid = ranked[0][1]
    return best_uid, records[best_uid]


def load_hu_volume(files: List[Path]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    import pydicom

    slices = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
        except Exception:
            continue
        if not hasattr(ds, "pixel_array"):
            continue

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = ds.pixel_array.astype(np.float32) * slope + intercept

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
        raise ValueError("Could not load any DICOM slices with pixel data.")

    slices.sort(key=lambda t: t[0])
    vol = np.stack([s[1] for s in slices], axis=2)
    vol = np.transpose(vol, (1, 0, 2))  # (x, y, z)

    sx, sy, st = slices[0][2]
    if len(slices) > 1:
        dz = np.median(np.diff([s[0] for s in slices]))
        if np.isfinite(dz) and dz > 0:
            st = float(abs(dz))
    return vol.astype(np.float32), (sx, sy, st)


def _slice_triplet(vol_xyz: np.ndarray):
    cx, cy, cz = vol_xyz.shape[0] // 2, vol_xyz.shape[1] // 2, vol_xyz.shape[2] // 2
    xy = vol_xyz[:, :, cz].T
    xz = vol_xyz[:, cy, :].T
    yz = vol_xyz[cx, :, :].T
    return xy, xz, yz


def make_body_mask_from_hu(hu_xyz: np.ndarray, hu_threshold: float = -300.0) -> np.ndarray:
    """
    Body mask = largest connected component of voxels above HU threshold.
    This keeps body/head intact and only identifies outside background.
    """
    try:
        from scipy.ndimage import label
    except ImportError as exc:
        raise ImportError("scipy is required. Install with: pip install scipy") from exc

    candidate = hu_xyz > hu_threshold
    labeled, n_components = label(candidate)
    if n_components == 0:
        return np.zeros_like(candidate, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labeled == largest


def force_air_outside_body(
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    att_xyz: np.ndarray,
    body_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IMPORTANT: This does not modify values inside body_mask.
    It only sets values outside body_mask to air.
    """
    c_out = c_xyz.copy()
    rho_out = rho_xyz.copy()
    att_out = att_xyz.copy()
    outside = ~body_mask
    c_out[outside] = 343.0
    rho_out[outside] = 1.2
    att_out[outside] = 1.0
    return c_out, rho_out, att_out


def _robust_limits(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    # Exclude background air — voxels below water speed or near-zero density
    # are outside the head and dominate the colormap otherwise
    if arr.mean() > 500:  # sound speed map
        vals = arr[arr > 400.0]
    else:
        vals = arr[arr > 10.0]  # density map
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def plot_acoustic_slices(
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    att_xyz: np.ndarray,
    spacing_xyz_mm: Tuple[float, float, float],
    out_path: Path,
) -> None:
    sx, sy, sz = spacing_xyz_mm
    nx, ny, nz = c_xyz.shape
    extent_xy = [0, nx * sx, 0, ny * sy]
    extent_xz = [0, nx * sx, 0, nz * sz]
    extent_yz = [0, ny * sy, 0, nz * sz]

    c_xy, c_xz, c_yz = _slice_triplet(c_xyz)
    r_xy, r_xz, r_yz = _slice_triplet(rho_xyz)
    a_xy, a_xz, a_yz = _slice_triplet(att_xyz)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    rows = [
        ("Sound speed (m/s)", (c_xy, c_xz, c_yz), "viridis", c_xyz),
        ("Density (kg/m^3)", (r_xy, r_xz, r_yz), "viridis", rho_xyz),
        ("Attenuation (dB/cm/MHz)", (a_xy, a_xz, a_yz), "magma", att_xyz),
    ]
    extents = [extent_xy, extent_xz, extent_yz]
    plane_names = ["XY", "XZ", "YZ"]

    for row_i, (row_title, imgs, cmap, full_vol) in enumerate(rows):
        vmin, vmax = _robust_limits(full_vol)
        for col_i, img in enumerate(imgs):
            ax = axes[row_i, col_i]
            im = ax.imshow(
                img,
                origin="lower",
                cmap=cmap,
                extent=extents[col_i],
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{row_title} | {plane_names[col_i]}")
            if col_i == 0:
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("y (mm)")
            elif col_i == 1:
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("z (mm)")
            else:
                ax.set_xlabel("y (mm)")
                ax.set_ylabel("z (mm)")
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("CT full scene: acoustic property slices (XY / XZ / YZ)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_3d_speed_isosurfaces(
    c_xyz: np.ndarray,
    spacing_xyz_mm: Tuple[float, float, float],
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plotted = False
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        levels = [
            (1520.0, "#5DADE2", 0.20),  # soft-tissue-ish
            (2200.0, "#F5B041", 0.35),  # skull-ish
        ]
        sx, sy, sz = spacing_xyz_mm
        for level, color, alpha in levels:
            try:
                verts, faces, _, _ = measure.marching_cubes(c_xyz, level=level)
            except (ValueError, RuntimeError):
                continue
            verts_mm = verts * np.array([sx, sy, sz])[None, :]
            mesh = Poly3DCollection(
                verts_mm[faces],
                alpha=alpha,
                facecolor=color,
                edgecolor="none",
            )
            ax.add_collection3d(mesh)
            plotted = True
    except ImportError:
        pass

    if not plotted:
        # Fallback: sparse scatter for high-speed voxels.
        idx = np.argwhere(c_xyz > 1600.0)
        if idx.size > 0:
            idx = idx[:: max(1, len(idx) // 5000)]
            sx, sy, sz = spacing_xyz_mm
            ax.scatter(idx[:, 0] * sx, idx[:, 1] * sy, idx[:, 2] * sz, s=2, alpha=0.25)

    sx, sy, sz = spacing_xyz_mm
    ax.set_xlim(0, c_xyz.shape[0] * sx)
    ax.set_ylim(0, c_xyz.shape[1] * sy)
    ax.set_zlim(0, c_xyz.shape[2] * sz)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-1: full CT scene acoustic slice visualization.")
    parser.add_argument("--patient-id", required=True, help='Patient ID, e.g. "ACRIN-FMISO-Brain-001".')
    parser.add_argument(
        "--dicom-root",
        default="Results/kaggle_brain_tumor_mri_ct/unzipped/dicom",
        help="Root directory containing DICOM files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: Results/jwave_brain_forward_64/<patient>_acoustic_slices.png).",
    )
    parser.add_argument(
        "--body-mask-hu-threshold",
        type=float,
        default=-300.0,
        help="HU threshold to build body candidate mask (default: -300).",
    )
    parser.add_argument(
        "--save-cleaned-maps",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, save cleaned maps and body mask next to output image (default: 1).",
    )
    args = parser.parse_args()

    dicom_root = Path(args.dicom_root).expanduser().resolve()
    records = discover_ct_series_for_patient(dicom_root, args.patient_id)
    uid, rec = pick_best_ct_series(records)
    hu_xyz, spacing_xyz_mm = load_hu_volume(rec["files"])
    c_before, rho_before, att_before, _ = map_acoustic_properties(hu_xyz)
    body_mask = make_body_mask_from_hu(hu_xyz, hu_threshold=float(args.body_mask_hu_threshold))
    c_xyz, rho_xyz, att_xyz = force_air_outside_body(c_before, rho_before, att_before, body_mask)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.patient_id)
        out_path = Path("Results/jwave_brain_forward_64").resolve() / f"{safe_id}_acoustic_slices.png"

    plot_acoustic_slices(c_xyz, rho_xyz, att_xyz, spacing_xyz_mm, out_path)
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.patient_id)
    out_dir = out_path.parent
    before_3d = out_dir / f"{safe_id}_3d_before_cleanup.png"
    after_3d = out_dir / f"{safe_id}_3d_after_cleanup.png"
    _plot_3d_speed_isosurfaces(c_before, spacing_xyz_mm, before_3d, "3D before cleanup (sound speed)")
    _plot_3d_speed_isosurfaces(c_xyz, spacing_xyz_mm, after_3d, "3D after cleanup (sound speed)")

    if args.save_cleaned_maps == 1:
        np.save(out_dir / f"{safe_id}_sound_speed_cleaned.npy", c_xyz.astype(np.float32))
        np.save(out_dir / f"{safe_id}_density_cleaned.npy", rho_xyz.astype(np.float32))
        np.save(out_dir / f"{safe_id}_attenuation_cleaned.npy", att_xyz.astype(np.float32))
        np.save(out_dir / f"{safe_id}_body_mask.npy", body_mask.astype(np.uint8))

    print(f"[ok] Saved image: {out_path}")
    print(f"[info] patient={args.patient_id}")
    print(f"[info] series_uid={uid}")
    print(f"[info] series_desc={rec.get('desc', '')}")
    print(f"[info] shape_xyz={hu_xyz.shape}, spacing_mm={spacing_xyz_mm}")
    print(f"[info] body-mask fraction={float(np.mean(body_mask)):.4f} | threshold={args.body_mask_hu_threshold}")
    print(f"[ok] Saved 3D before: {before_3d}")
    print(f"[ok] Saved 3D after:  {after_3d}")
    import json
    meta = {
        "patient_id": args.patient_id,
        "series_uid": uid,
        "spacing_xyz_mm": list(spacing_xyz_mm),
        "shape_xyz": list(hu_xyz.shape),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] Saved metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

