"""
Generate j-wave acoustic property maps from a brain CT scan.

This variant uses:
  1) HU -> density mapping copied exactly from `hounsfield2density.m`
     (piece-wise linear polynomials; density in kg/m^3).
  2) Sound speed computed from the density via:
        sound_speed = 1.33 * density + 167
  3) Attenuation mapping kept the same as in `brain_ct_to_jwave_acoustics.py`
     (HU -> attenuation in dB/cm/MHz).

Outputs (all in j-wave x,y,z axis order, matching the base script):
  - hu_xyz.npy
  - sound_speed_xyz.npy
  - density_xyz.npy
  - attenuation_db_cm_mhz_xyz.npy
  - tissue_mask_xyz.npy
  - tissue_stats.csv
  - metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brain_ct_to_jwave_acoustics import (
    TISSUE_LABELS,
    _attenuation_from_hu,
    _classify_tissues,
    _sanity_check_hu,
    load_ct_volume,
    maybe_download_tcia_series,
    list_tcia_candidates,
    parse_target_shape,
    resample_volume,
)
from ct_hu_clean_top_head_view import clean_hu_volume, make_head_mask

MERGED_TISSUE_LABELS = {
    "air": 0,
    "csf": 1,
    "brain_soft": 2,
    "trabecular_skull": 3,
    "cortical_skull": 4,
}

_BRAIN_SOFT_SRC = (
    "fat",
    "brain_wm",
    "brain_gm",
    "blood",
    "soft_tissue",
)


def _density_from_hu_hounsfield2density(hu_xyz: np.ndarray) -> np.ndarray:
    """
    HU -> density (kg/m^3) using the exact piece-wise linear polynomials in:
        Code/hounsfield2density.m
    """
    # The MATLAB `hounsfield2density.m` model expects *shifted* HU where:
    #   air ≈ 0 and water ≈ 1000
    # Typical DICOM CT HU uses water ≈ 0 and air ≈ -1000, so we shift here.
    hu = np.asarray(hu_xyz, dtype=np.float32) + 1000.0
    density = np.zeros_like(hu, dtype=np.float32)

    # Part 1: Less than 930 Hounsfield Units
    m1 = hu < 930.0
    density[m1] = (
        1.025793065681423 * hu[m1]
        + (-5.680404011488714)
    )

    # Part 2: Between 930 and 1098 (soft tissue region, inclusive)
    m2 = (hu >= 930.0) & (hu <= 1098.0)
    density[m2] = (
        0.9082709691264 * hu[m2]
        + 103.6151457847139
    )

    # Part 3: Between 1098 and 1260 (between soft tissue and bone; strict)
    m3 = (hu > 1098.0) & (hu < 1260.0)
    density[m3] = (
        0.5108369316599 * hu[m3]
        + 539.9977189228704
    )

    # Part 4: Greater than or equal to 1260 (bone region)
    m4 = hu >= 1260.0
    density[m4] = (
        0.6625370912451 * hu[m4]
        + 348.8555178455294
    )

    return density


def _sound_speed_from_density(density_kg_m3: np.ndarray) -> np.ndarray:
    """
    Sound speed (m/s) from density (kg/m^3), as requested:
        speed = 1.33*density + 167
    """
    return (1.33 * density_kg_m3 + 167.0).astype(np.float32)


def _prepare_hu_for_mapping(
    hu_xyz: np.ndarray,
    hu_clip_min: float,
    hu_clip_max: float,
) -> Tuple[np.ndarray, int]:
    """Replace non-finite HU and clip to a physically valid range."""
    hu = np.asarray(hu_xyz, dtype=np.float32)
    hu = np.nan_to_num(hu, nan=hu_clip_min, neginf=hu_clip_min, posinf=hu_clip_max)
    n_clipped = int(np.count_nonzero((hu < hu_clip_min) | (hu > hu_clip_max)))
    hu = np.clip(hu, hu_clip_min, hu_clip_max)
    return hu, n_clipped


def map_acoustic_properties(
    hu_xyz: np.ndarray,
    hu_clip_min: float = -1000.0,
    hu_clip_max: float = 3000.0,
    rho_clip_min: float = 1000.0,
    rho_clip_max: float = 2500.0,
    c_clip_min: float = 1400.0,
    c_clip_max: float = 3500.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Convert a 3-D HU volume (x, y, z) to continuous acoustic property maps.
    """
    _sanity_check_hu(hu_xyz)
    hu_mapped_xyz, n_clipped = _prepare_hu_for_mapping(hu_xyz, hu_clip_min, hu_clip_max)

    # Use the HU-based classifier to keep "air" from being forced to tissue density.
    mask_xyz = _classify_tissues(hu_mapped_xyz)
    is_air = mask_xyz == TISSUE_LABELS["air"]

    rho_xyz = _density_from_hu_hounsfield2density(hu_mapped_xyz)
    rho_xyz = rho_xyz.astype(np.float32, copy=False)
    rho_xyz[~is_air] = np.clip(rho_xyz[~is_air], rho_clip_min, rho_clip_max).astype(np.float32)
    rho_xyz[is_air] = 0.0
    c_xyz = _sound_speed_from_density(rho_xyz)
    c_xyz = c_xyz.astype(np.float32, copy=False)
    c_xyz[~is_air] = np.clip(c_xyz[~is_air], c_clip_min, c_clip_max).astype(np.float32)
    c_xyz[is_air] = 0.0
    att_xyz = _attenuation_from_hu(hu_mapped_xyz)  # keep attenuation mapping identical
    return c_xyz, rho_xyz, att_xyz, mask_xyz, hu_mapped_xyz, n_clipped


def merge_tissue_mask(mask_xyz: np.ndarray) -> np.ndarray:
    """Merge multiple soft classes into brain_soft while keeping CSF and skull split."""
    out = np.zeros_like(mask_xyz, dtype=np.uint8)
    ids = TISSUE_LABELS
    out[mask_xyz == ids["csf"]] = MERGED_TISSUE_LABELS["csf"]
    out[mask_xyz == ids["trabecular_skull"]] = MERGED_TISSUE_LABELS["trabecular_skull"]
    out[mask_xyz == ids["cortical_skull"]] = MERGED_TISSUE_LABELS["cortical_skull"]
    for name in _BRAIN_SOFT_SRC:
        out[mask_xyz == ids[name]] = MERGED_TISSUE_LABELS["brain_soft"]
    out[mask_xyz == ids["air"]] = MERGED_TISSUE_LABELS["air"]
    return out


def write_stats_csv_bodymask(
    path: Path,
    merged_mask_xyz: np.ndarray,
    body_mask_xyz: np.ndarray,
    hu_xyz: np.ndarray,
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    att_xyz: np.ndarray,
) -> None:
    """Write tissue stats using body-mask voxels only as denominator."""
    body = np.asarray(body_mask_xyz, dtype=bool)
    n_total = int(body.sum())
    if n_total <= 0:
        raise ValueError("Body mask is empty; cannot compute body-normalized fractions.")

    label_to_name = {v: k for k, v in MERGED_TISSUE_LABELS.items()}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tissue", "label_id", "voxel_count", "fraction",
            "hu_mean", "hu_std", "hu_min", "hu_max",
            "c_mean", "rho_mean", "att_mean",
        ])
        for label_id, name in sorted(label_to_name.items()):
            m = (merged_mask_xyz == label_id) & body
            n = int(m.sum())
            if n == 0:
                writer.writerow([name, label_id, 0, 0.0, "", "", "", "", "", "", ""])
                continue
            hu_v = hu_xyz[m]
            writer.writerow([
                name, label_id, n, round(n / n_total, 6),
                round(float(hu_v.mean()), 2), round(float(hu_v.std()), 2),
                round(float(hu_v.min()), 2), round(float(hu_v.max()), 2),
                round(float(c_xyz[m].mean()), 2),
                round(float(rho_xyz[m].mean()), 2),
                round(float(att_xyz[m].mean()), 4),
            ])


def plot_property_slices(
    out_png: Path,
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    att_xyz: np.ndarray,
) -> None:
    """Save mid-axial 2D slices for sound speed, density, and attenuation."""
    z = c_xyz.shape[2] // 2
    c2d = c_xyz[:, :, z]
    rho2d = rho_xyz[:, :, z]
    att2d = att_xyz[:, :, z]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=140)
    items = [
        ("Sound speed (m/s)", c2d, "viridis"),
        ("Density (kg/m^3)", rho2d, "magma"),
        ("Attenuation (dB/cm/MHz)", att2d, "inferno"),
    ]
    for ax, (title, img, cmap) in zip(axes, items):
        lo, hi = np.percentile(img, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(img)), float(np.max(img))
        im = ax.imshow(img.T, cmap=cmap, origin="lower", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Mid-axial slices (z={z})", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HU->density via hounsfield2density.m and density->sound_speed via c=1.33*rho+167.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a DICOM series directory, a NIfTI file (.nii/.nii.gz), "
             "or a TCIA specifier like tcia:ACRIN-FMISO-Brain.",
    )
    parser.add_argument("--out", required=False, help="Output directory.")
    parser.add_argument(
        "--target-shape", default=None,
        help="Resample to this shape before mapping, as Nx,Ny,Nz (e.g. 256,256,256).",
    )
    parser.add_argument(
        "--resample-order", type=int, default=1, choices=[0, 1, 3],
        help="Interpolation order for resampling: 0=nearest, 1=linear, 3=cubic.",
    )
    parser.add_argument("--hu-clip-min", type=float, default=-1000.0,
                        help="Lower HU bound before mapping (air floor).")
    parser.add_argument("--hu-clip-max", type=float, default=3000.0,
                        help="Upper HU bound before mapping.")
    parser.add_argument("--rho-clip-min", type=float, default=1000.0,
                        help="Lower density bound after mapping (kg/m^3).")
    parser.add_argument("--rho-clip-max", type=float, default=2500.0,
                        help="Upper density bound after mapping (kg/m^3).")
    parser.add_argument("--c-clip-min", type=float, default=1400.0,
                        help="Lower sound-speed bound after mapping (m/s).")
    parser.add_argument("--c-clip-max", type=float, default=3500.0,
                        help="Upper sound-speed bound after mapping (m/s).")
    parser.add_argument("--plot-slices", type=int, default=1, choices=[0, 1],
                        help="Save mid-axial 2D slices for speed/density/attenuation.")
    parser.add_argument("--clean-head-mask", type=int, default=1, choices=[0, 1],
                        help="Run head cleaning first and zero everything outside head.")
    parser.add_argument("--head-center-ellipse-frac", type=float, default=0.35,
                        help="Center ellipse half-axis fraction used by head masker.")
    parser.add_argument("--save-clean-mask", type=int, default=1, choices=[0, 1],
                        help="Save head mask + cleaned HU arrays used for mapping.")

    parser.add_argument("--tcia-series-uid", default=None, help="Force a specific TCIA SeriesInstanceUID.")
    parser.add_argument("--tcia-patient-id", default=None, help="Filter TCIA query by PatientID.")
    parser.add_argument("--tcia-bodypart", default="HEAD", help="BodyPartExamined filter for TCIA auto-selection.")
    parser.add_argument(
        "--tcia-require-head-keywords", type=int, default=1, choices=[0, 1],
        help="Reject non-head-like series when auto-selecting (1=yes).",
    )
    parser.add_argument("--tcia-list-candidates", type=int, default=0, choices=[0, 1],
                        help="Print ranked candidate series and exit (1=yes).")
    parser.add_argument("--tcia-max-candidates", type=int, default=20,
                        help="How many candidates to print with --tcia-list-candidates.")
    args = parser.parse_args()

    if args.tcia_list_candidates:
        list_tcia_candidates(
            args.input, args.tcia_patient_id, args.tcia_bodypart,
            bool(args.tcia_require_head_keywords), args.tcia_max_candidates,
        )
        return

    if not args.out:
        raise ValueError("--out is required (unless --tcia-list-candidates 1).")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_dicom_dir = maybe_download_tcia_series(
        args.input,
        out_dir,
        args.tcia_series_uid,
        args.tcia_patient_id,
        args.tcia_bodypart,
        bool(args.tcia_require_head_keywords),
    )
    in_path = maybe_dicom_dir if maybe_dicom_dir else Path(args.input).expanduser().resolve()

    target_shape = parse_target_shape(args.target_shape)

    hu_xyz, spacing_xyz_mm, source_kind = load_ct_volume(in_path)

    if target_shape is not None:
        print(f"[info] Resampling {hu_xyz.shape} → {target_shape} (order={args.resample_order})")
        hu_xyz, spacing_xyz_mm = resample_volume(
            hu_xyz, spacing_xyz_mm, target_shape, args.resample_order
        )
        print(f"[info] Resampled spacing_mm={spacing_xyz_mm}")

    head_mask_xyz = None
    hu_for_mapping_xyz = hu_xyz
    if args.clean_head_mask:
        print("[info] Cleaning CT and masking outside head …")
        head_mask_xyz = make_head_mask(hu_xyz, center_frac=args.head_center_ellipse_frac)
        hu_for_mapping_xyz = clean_hu_volume(hu_xyz, head_mask_xyz)
        # Guarantee strict outside-head masking before acoustic mapping.
        hu_for_mapping_xyz[~head_mask_xyz] = -1024.0

    print("[info] Mapping acoustic properties …")
    c_xyz, rho_xyz, att_xyz, mask_xyz, hu_mapped_xyz, n_hu_clipped = map_acoustic_properties(
        hu_for_mapping_xyz,
        hu_clip_min=args.hu_clip_min,
        hu_clip_max=args.hu_clip_max,
        rho_clip_min=args.rho_clip_min,
        rho_clip_max=args.rho_clip_max,
        c_clip_min=args.c_clip_min,
        c_clip_max=args.c_clip_max,
    )

    print(f"[info] Sound speed : {c_xyz.min():.1f} – {c_xyz.max():.1f} m/s")
    print(f"[info] Density     : {rho_xyz.min():.1f} – {rho_xyz.max():.1f} kg/m^3")
    print(f"[info] Attenuation : {att_xyz.min():.4f} – {att_xyz.max():.4f} dB/cm/MHz")
    print(
        f"[info] HU used for mapping: {hu_mapped_xyz.min():.1f} – {hu_mapped_xyz.max():.1f} "
        f"(clipped voxels: {n_hu_clipped:,})"
    )

    np.save(out_dir / "hu_xyz.npy", hu_xyz.astype(np.float32))
    if args.save_clean_mask and head_mask_xyz is not None:
        np.save(out_dir / "head_mask_xyz.npy", head_mask_xyz.astype(np.uint8))
        np.save(out_dir / "hu_head_cleaned_xyz.npy", hu_for_mapping_xyz.astype(np.float32))
    np.save(out_dir / "hu_mapped_xyz.npy", hu_mapped_xyz.astype(np.float32))
    np.save(out_dir / "sound_speed_xyz.npy", c_xyz.astype(np.float32))
    np.save(out_dir / "density_xyz.npy", rho_xyz.astype(np.float32))
    np.save(out_dir / "attenuation_db_cm_mhz_xyz.npy", att_xyz.astype(np.float32))
    merged_mask_xyz = merge_tissue_mask(mask_xyz)
    np.save(out_dir / "tissue_mask_xyz.npy", merged_mask_xyz.astype(np.uint8))
    if args.plot_slices:
        plot_property_slices(out_dir / "acoustic_property_slices.png", c_xyz, rho_xyz, att_xyz)

    body_mask_xyz = merged_mask_xyz > 0
    write_stats_csv_bodymask(
        out_dir / "tissue_stats.csv",
        merged_mask_xyz,
        body_mask_xyz,
        hu_mapped_xyz,
        c_xyz,
        rho_xyz,
        att_xyz,
    )

    meta = {
        "source_kind": source_kind,
        "input_path": str(in_path),
        "shape_xyz": list(hu_xyz.shape),
        "spacing_xyz_mm": list(spacing_xyz_mm),
        "mapping_input_hu_file": "hu_mapped_xyz.npy",
        "head_cleaning": {
            "enabled": bool(args.clean_head_mask),
            "head_center_ellipse_frac": float(args.head_center_ellipse_frac),
            "saved_head_mask": bool(args.save_clean_mask and head_mask_xyz is not None),
            "cleaned_hu_file": "hu_head_cleaned_xyz.npy" if (args.save_clean_mask and head_mask_xyz is not None) else None,
            "head_mask_file": "head_mask_xyz.npy" if (args.save_clean_mask and head_mask_xyz is not None) else None,
        },
        "mapping_safety": {
            "hu_clip_min": float(args.hu_clip_min),
            "hu_clip_max": float(args.hu_clip_max),
            "rho_clip_min_kg_m3": float(args.rho_clip_min),
            "rho_clip_max_kg_m3": float(args.rho_clip_max),
            "c_clip_min_m_s": float(args.c_clip_min),
            "c_clip_max_m_s": float(args.c_clip_max),
            "n_hu_voxels_clipped": int(n_hu_clipped),
        },
        "acoustic_mapping": (
            "density: HU->density via piece-wise linear polynomials from Code/hounsfield2density.m "
            "(density in kg/m^3); sound_speed: c=1.33*density+167; "
            "attenuation: kept identical to brain_ct_to_jwave_acoustics._attenuation_from_hu (dB/cm/MHz)."
        ),
        "tissue_labels": MERGED_TISSUE_LABELS,
        "for_jwave": {
            "sound_speed_file": "sound_speed_xyz.npy",
            "density_file": "density_xyz.npy",
            "attenuation_file": "attenuation_db_cm_mhz_xyz.npy",
            "property_slice_plot": "acoustic_property_slices.png" if args.plot_slices else None,
            "attenuation_note": (
                "Units are dB/cm/MHz. Downstream may convert to Np/m at simulation frequency."
            ),
        },
        "tcia_auto_download": {
            "enabled": maybe_dicom_dir is not None,
            "input_spec": args.input,
            "downloaded_dicom_dir": str(maybe_dicom_dir) if maybe_dicom_dir else None,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[ok] Saved:")
    for fname in [
        "hu_xyz.npy",
        "head_mask_xyz.npy" if (args.save_clean_mask and head_mask_xyz is not None) else None,
        "hu_head_cleaned_xyz.npy" if (args.save_clean_mask and head_mask_xyz is not None) else None,
        "hu_mapped_xyz.npy",
        "sound_speed_xyz.npy",
        "density_xyz.npy",
        "attenuation_db_cm_mhz_xyz.npy",
        "tissue_mask_xyz.npy",
        "acoustic_property_slices.png" if args.plot_slices else None,
        "tissue_stats.csv",
        "metadata.json",
    ]:
        if fname is not None:
            print(f"  {out_dir / fname}")
    print("[ok] Done.")


if __name__ == "__main__":
    main()

