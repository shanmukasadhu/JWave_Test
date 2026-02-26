"""
Load a brain CT scan and generate acoustic property volumes for j-wave.

Supported input:
- DICOM series directory
- NIfTI file (.nii / .nii.gz)

Outputs:
- hu_xyz.npy
- sound_speed_xyz.npy (m/s)
- density_xyz.npy (kg/m^3)
- attenuation_db_cm_mhz_xyz.npy (dB/cm/MHz)
- metadata.json
- tissue_stats.csv

Notes:
- The tissue-to-acoustic mappings here are pragmatic defaults and should be
  calibrated for your scanner/protocol/frequency before final studies.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TissueAcoustics:
    sound_speed: float  # m/s
    density: float      # kg/m^3
    attenuation_db_cm_mhz: float  # dB/cm/MHz


DEFAULT_TISSUE_TABLE: Dict[str, TissueAcoustics] = {
    # Approximate values around MHz ultrasound; tune as needed.
    "air": TissueAcoustics(sound_speed=343.0, density=1.2, attenuation_db_cm_mhz=120.0),
    "csf": TissueAcoustics(sound_speed=1505.0, density=1007.0, attenuation_db_cm_mhz=0.05),
    "brain_soft": TissueAcoustics(sound_speed=1540.0, density=1040.0, attenuation_db_cm_mhz=0.6),
    "trabecular_skull": TissueAcoustics(sound_speed=2200.0, density=1300.0, attenuation_db_cm_mhz=15.0),
    "cortical_skull": TissueAcoustics(sound_speed=2800.0, density=1900.0, attenuation_db_cm_mhz=35.0),
}


def parse_target_shape(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("target shape must be 'Nx,Ny,Nz' (e.g. 128,128,128)")
    nx, ny, nz = (int(v) for v in parts)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("all target-shape dimensions must be positive")
    return nx, ny, nz


def _contains_any(text: str, keywords) -> bool:
    t = text.upper()
    return any(k in t for k in keywords)


def _series_text_blob(row) -> str:
    fields = [
        "BodyPartExamined",
        "SeriesDescription",
        "StudyDescription",
        "ProtocolName",
        "Collection",
        "Modality",
    ]
    vals = []
    for f in fields:
        if f in row.index:
            vals.append(str(row[f]))
    return " | ".join(vals).upper()


def _rank_series_candidates(series_df, tcia_bodypart: str, require_head_keywords: bool):
    if series_df is None or len(series_df) == 0:
        return []
    filtered = series_df
    head_kw = ["HEAD", "BRAIN", "SKULL", "CRANI", "NEURO", "CEREB"]
    chest_kw = ["CHEST", "LUNG", "THORAX", "ABDOMEN", "PELVIS", "CARDIAC"]

    if require_head_keywords:
        keep_idx = []
        for idx, row in filtered.iterrows():
            blob = _series_text_blob(row)
            if _contains_any(blob, head_kw) and (not _contains_any(blob, chest_kw)):
                keep_idx.append(idx)
        if keep_idx:
            filtered = filtered.loc[keep_idx]
        else:
            return []

    ranked = []
    for idx, row in filtered.iterrows():
        blob = _series_text_blob(row)
        score = 0.0
        if _contains_any(blob, head_kw):
            score += 5.0
        if _contains_any(blob, chest_kw):
            score -= 10.0
        if tcia_bodypart and _contains_any(blob, [tcia_bodypart.upper()]):
            score += 3.0
        img_count = float(row["ImageCount"]) if "ImageCount" in row.index and str(row["ImageCount"]) != "nan" else 0.0
        score += 0.001 * img_count
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    return ranked


def _choose_series_row(
    series_df,
    tcia_series_uid: Optional[str],
    tcia_bodypart: str,
    require_head_keywords: bool,
):
    if series_df is None or len(series_df) == 0:
        return None
    if tcia_series_uid:
        pick = series_df[series_df["SeriesInstanceUID"] == tcia_series_uid]
        if len(pick) == 0:
            raise ValueError(f"SeriesInstanceUID not found: {tcia_series_uid}")
        return pick.iloc[0]

    ranked = _rank_series_candidates(series_df, tcia_bodypart, require_head_keywords)
    if not ranked:
        raise ValueError(
            "No head/brain-like CT series found after keyword filtering. "
            "Try a specific --tcia-series-uid or set --tcia-require-head-keywords 0."
        )
    return series_df.loc[ranked[0][1]]


def list_tcia_candidates(
    input_spec: str,
    tcia_patient_id: Optional[str],
    tcia_bodypart: str,
    tcia_require_head_keywords: bool,
    max_candidates: int,
) -> None:
    spec = input_spec.strip()
    if not (spec.startswith("tcia:") or spec.startswith("tcia://")):
        raise ValueError("Candidate listing requires --input tcia:<collection> or tcia:auto-head-ct")
    collection = spec.split(":", 1)[1].replace("//", "").strip()
    auto_head_mode = collection.lower() in {"auto", "auto-head-ct", "head-ct", "ct-head"}

    tcia_backend = None
    try:
        from tcia_utils import nbia as tcia_nbia  # type: ignore
        tcia_backend = "tcia_utils"
    except ImportError:
        tcia_nbia = None
    if tcia_backend is None:
        try:
            from nbiatoolkit import NBIAClient  # type: ignore
            tcia_backend = "nbiatoolkit"
        except ImportError as exc:
            raise ImportError(
                "TCIA query requires either tcia_utils or nbiatoolkit.\n"
                "Install one of:\n"
                "  pip install tcia_utils\n"
                "  pip install nbiatoolkit"
            ) from exc

    if tcia_backend == "tcia_utils":
        if auto_head_mode:
            series_df = tcia_nbia.getSeries(
                modality="CT",
                bodyPart=tcia_bodypart or "HEAD",
                patientId=tcia_patient_id or "",
                format="df",
            )
            if series_df is None or len(series_df) == 0:
                series_df = tcia_nbia.getSeries(modality="CT", patientId=tcia_patient_id or "", format="df")
        else:
            series_df = tcia_nbia.getSeries(
                collection=collection, modality="CT", patientId=tcia_patient_id or "", format="df"
            )
            if series_df is None or len(series_df) == 0:
                series_df = tcia_nbia.getSeries(collection=collection, patientId=tcia_patient_id or "", format="df")
    else:
        with NBIAClient(return_type="dataframe") as client:
            query_kwargs = {"Modality": "CT"}
            if not auto_head_mode:
                query_kwargs["Collection"] = collection
            if tcia_patient_id:
                query_kwargs["PatientID"] = tcia_patient_id
            series_df = client.getSeries(**query_kwargs)
            if series_df is None or len(series_df) == 0:
                query_kwargs = {}
                if not auto_head_mode:
                    query_kwargs["Collection"] = collection
                if tcia_patient_id:
                    query_kwargs["PatientID"] = tcia_patient_id
                series_df = client.getSeries(**query_kwargs)

    if series_df is None or len(series_df) == 0:
        raise ValueError("No TCIA series found for this query.")

    ranked = _rank_series_candidates(series_df, tcia_bodypart, tcia_require_head_keywords)
    if not ranked:
        raise ValueError("No candidates left after head/brain filtering. Try --tcia-require-head-keywords 0")

    print("\nTop TCIA candidate series (use --tcia-series-uid):")
    n = min(max_candidates, len(ranked))
    for i in range(n):
        score, idx = ranked[i]
        row = series_df.loc[idx]
        uid = str(row.get("SeriesInstanceUID", ""))
        coll = str(row.get("Collection", ""))
        pid = str(row.get("PatientID", ""))
        body = str(row.get("BodyPartExamined", ""))
        img_count = str(row.get("ImageCount", ""))
        sdesc = str(row.get("SeriesDescription", ""))
        print(
            f"{i+1:2d}. score={score:6.3f}  imgs={img_count:>5}  uid={uid}\n"
            f"    collection={coll}  patient={pid}  body={body}\n"
            f"    series={sdesc}"
        )


def load_ct_volume(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float], str]:
    """
    Returns:
      hu_xyz: array in (x, y, z)
      spacing_xyz_mm: (sx, sy, sz) in mm
      source_kind: string label
    """
    if path.is_dir():
        return _load_dicom_series(path)

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        return _load_nifti(path)

    raise ValueError(
        f"Unsupported input: {path}. Provide DICOM folder or NIfTI file (.nii/.nii.gz)."
    )


def maybe_download_tcia_series(
    input_spec: str,
    work_dir: Path,
    tcia_series_uid: Optional[str],
    tcia_patient_id: Optional[str],
    tcia_bodypart: str,
    tcia_require_head_keywords: bool,
) -> Optional[Path]:
    """
    Supports input like:
      tcia:CT-ORG
      tcia://CT-ORG
      tcia:auto-head-ct

    Downloads one CT series via NBIA API and returns a local DICOM directory path.
    """
    spec = input_spec.strip()
    if not (spec.startswith("tcia:") or spec.startswith("tcia://")):
        return None

    collection = spec.split(":", 1)[1].replace("//", "").strip()
    if not collection:
        raise ValueError("TCIA input must include collection, e.g. tcia:CT-ORG")
    auto_head_mode = collection.lower() in {"auto", "auto-head-ct", "head-ct", "ct-head"}

    # Prefer tcia_utils (simple stateless API calls), then fall back to nbiatoolkit.
    tcia_backend = None
    try:
        from tcia_utils import nbia as tcia_nbia  # type: ignore
        tcia_backend = "tcia_utils"
    except ImportError:
        tcia_nbia = None

    if tcia_backend is None:
        try:
            from nbiatoolkit import NBIAClient  # type: ignore
            tcia_backend = "nbiatoolkit"
        except ImportError as exc:
            raise ImportError(
                "TCIA auto-download requires either tcia_utils or nbiatoolkit.\n"
                "Install one of:\n"
                "  pip install tcia_utils\n"
                "  pip install nbiatoolkit"
            ) from exc

    download_dir = work_dir / "_tcia_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Querying TCIA collection: {collection}")
    series_uid = None
    if tcia_backend == "tcia_utils":
        # tcia_utils expects lowercase function args.
        if auto_head_mode:
            series_df = tcia_nbia.getSeries(
                modality="CT",
                bodyPart=tcia_bodypart or "HEAD",
                patientId=tcia_patient_id or "",
                format="df",
            )
            if series_df is None or len(series_df) == 0:
                series_df = tcia_nbia.getSeries(
                    modality="CT",
                    patientId=tcia_patient_id or "",
                    format="df",
                )
        else:
            series_df = tcia_nbia.getSeries(
                collection=collection,
                modality="CT",
                patientId=tcia_patient_id or "",
                format="df",
            )
            if series_df is None or len(series_df) == 0:
                # Retry without modality filter in case metadata is inconsistent.
                series_df = tcia_nbia.getSeries(
                    collection=collection,
                    patientId=tcia_patient_id or "",
                    format="df",
                )
        if series_df is None or len(series_df) == 0:
            if not auto_head_mode:
                # Helpful suggestion list for likely collection naming mismatches.
                try:
                    cols = tcia_nbia.getCollections(format="df")
                    if cols is not None and len(cols) > 0 and "Collection" in cols.columns:
                        cand = cols["Collection"].astype(str).tolist()
                        guess = [c for c in cand if collection.lower() in c.lower()][:8]
                        if guess:
                            raise ValueError(
                                f"No series found for collection={collection}. Did you mean one of: {guess} ? "
                                "Or use --input tcia:auto-head-ct"
                            )
                except Exception:
                    pass
            raise ValueError(
                f"No series found for collection={collection}. "
                "Try --input tcia:auto-head-ct, a different collection, or provide --tcia-series-uid."
            )
        chosen = _choose_series_row(
            series_df, tcia_series_uid, tcia_bodypart, tcia_require_head_keywords
        )

        series_uid = str(chosen["SeriesInstanceUID"])
        print(f"[info] Downloading SeriesInstanceUID: {series_uid} using tcia_utils")
        tcia_nbia.downloadSeries(
            [series_uid],
            input_type="list",
            path=str(download_dir),
            as_zip=False,
            max_workers=4,
        )
    else:
        with NBIAClient(return_type="dataframe") as client:
            query_kwargs = {"Modality": "CT"}
            if not auto_head_mode:
                query_kwargs["Collection"] = collection
            if tcia_patient_id:
                query_kwargs["PatientID"] = tcia_patient_id
            series_df = client.getSeries(**query_kwargs)
            if series_df is None or len(series_df) == 0:
                # Retry without modality.
                query_kwargs = {}
                if not auto_head_mode:
                    query_kwargs["Collection"] = collection
                if tcia_patient_id:
                    query_kwargs["PatientID"] = tcia_patient_id
                series_df = client.getSeries(**query_kwargs)
            if series_df is None or len(series_df) == 0:
                raise ValueError(
                    f"No series found for collection={collection}. "
                    "Try --input tcia:auto-head-ct, a different collection, or provide --tcia-series-uid."
                )
            chosen = _choose_series_row(
                series_df, tcia_series_uid, tcia_bodypart, tcia_require_head_keywords
            )

            series_uid = str(chosen["SeriesInstanceUID"])
            print(f"[info] Downloading SeriesInstanceUID: {series_uid} using nbiatoolkit")
            client.downloadSeries(
                series_uid,
                downloadDir=str(download_dir),
                filePattern="%Collection/%PatientID/%StudyInstanceUID/%SeriesInstanceUID/%InstanceNumber.dcm",
                nParallel=4,
                Progressbar=True,
            )

    # Find directory containing downloaded DICOM slices.
    dicom_files = list(download_dir.rglob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"Downloaded TCIA data but found no .dcm files under {download_dir}")

    # Pick parent directory with most DICOM files (usually the selected series folder).
    counts: Dict[Path, int] = {}
    for fp in dicom_files:
        counts[fp.parent] = counts.get(fp.parent, 0) + 1
    best_dir = max(counts.items(), key=lambda kv: kv[1])[0]
    print(f"[info] Using downloaded DICOM directory: {best_dir} ({counts[best_dir]} files)")
    return best_dir


def _load_dicom_series(dicom_dir: Path) -> Tuple[np.ndarray, Tuple[float, float, float], str]:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for DICOM loading. Install with: pip install SimpleITK"
        ) from exc

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")

    # Use first series by default.
    series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(series_files)
    image = reader.Execute()

    # SITK array shape is (z, y, x); transpose to (x, y, z) for j-wave style.
    arr_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    hu_xyz = np.transpose(arr_zyx, (2, 1, 0))
    spacing_xyz_mm = tuple(float(v) for v in image.GetSpacing())  # (x, y, z)
    return hu_xyz, spacing_xyz_mm, "dicom"


def _load_nifti(nifti_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float], str]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for NIfTI loading. Install with: pip install nibabel"
        ) from exc

    img = nib.load(str(nifti_path))
    hu_xyz = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if hu_xyz.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {hu_xyz.shape}")

    hdr = img.header
    zooms = hdr.get_zooms()
    spacing_xyz_mm = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return hu_xyz, spacing_xyz_mm, "nifti"


def resample_volume(
    vol_xyz: np.ndarray, spacing_xyz_mm: Tuple[float, float, float], target_shape: Tuple[int, int, int], order: int
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise ImportError(
            "scipy is required for resampling. Install with: pip install scipy"
        ) from exc

    src_shape = vol_xyz.shape
    zoom_factors = (
        target_shape[0] / src_shape[0],
        target_shape[1] / src_shape[1],
        target_shape[2] / src_shape[2],
    )
    out = zoom(vol_xyz, zoom=zoom_factors, order=order)

    # Preserve physical FOV approximately.
    sx, sy, sz = spacing_xyz_mm
    new_spacing = (
        sx * (src_shape[0] / target_shape[0]),
        sy * (src_shape[1] / target_shape[1]),
        sz * (src_shape[2] / target_shape[2]),
    )
    return out.astype(np.float32), new_spacing


def classify_tissues(hu_xyz: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Basic HU-threshold segmentation.
    """
    air = hu_xyz < -500.0
    csf = (hu_xyz >= -10.0) & (hu_xyz < 20.0)
    brain_soft = (hu_xyz >= 20.0) & (hu_xyz < 120.0)
    trabecular = (hu_xyz >= 120.0) & (hu_xyz < 700.0)
    cortical = hu_xyz >= 700.0

    unknown = ~(air | csf | brain_soft | trabecular | cortical)
    # Fold unknown into brain_soft by default (conservative soft-tissue assumption).
    brain_soft = brain_soft | unknown

    return {
        "air": air,
        "csf": csf,
        "brain_soft": brain_soft,
        "trabecular_skull": trabecular,
        "cortical_skull": cortical,
    }


def map_acoustic_properties(hu_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    masks = classify_tissues(hu_xyz)

    c = np.zeros_like(hu_xyz, dtype=np.float32)
    rho = np.zeros_like(hu_xyz, dtype=np.float32)
    att = np.zeros_like(hu_xyz, dtype=np.float32)

    for label, mask in masks.items():
        props = DEFAULT_TISSUE_TABLE[label]
        c[mask] = props.sound_speed
        rho[mask] = props.density
        att[mask] = props.attenuation_db_cm_mhz

    return c, rho, att, masks


def write_stats_csv(path: Path, masks: Dict[str, np.ndarray], hu_xyz: np.ndarray) -> None:
    n_total = hu_xyz.size
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tissue", "voxel_count", "fraction", "hu_mean", "hu_std"])
        for label, mask in masks.items():
            vals = hu_xyz[mask]
            if vals.size == 0:
                writer.writerow([label, 0, 0.0, "", ""])
            else:
                writer.writerow(
                    [label, int(vals.size), float(vals.size / n_total), float(np.mean(vals)), float(np.std(vals))]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate j-wave acoustic property maps from brain CT.")
    parser.add_argument("--input", required=True, help="Path to DICOM directory or NIfTI file.")
    parser.add_argument("--out", required=False, help="Output directory for generated maps.")
    parser.add_argument(
        "--target-shape",
        default=None,
        help="Optional output shape as Nx,Ny,Nz (e.g., 128,128,128).",
    )
    parser.add_argument(
        "--resample-order",
        type=int,
        default=1,
        choices=[0, 1, 3],
        help="Interpolation order for resampling (0 nearest, 1 linear, 3 cubic).",
    )
    parser.add_argument(
        "--tcia-series-uid",
        default=None,
        help="Optional SeriesInstanceUID when using tcia:<collection> input.",
    )
    parser.add_argument(
        "--tcia-patient-id",
        default=None,
        help="Optional PatientID filter when using tcia:<collection> input.",
    )
    parser.add_argument(
        "--tcia-bodypart",
        default="HEAD",
        help="BodyPartExamined filter for auto-selecting CT series (default: HEAD).",
    )
    parser.add_argument(
        "--tcia-require-head-keywords",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, require head/brain-like keywords and reject chest/lung-like series (default: 1).",
    )
    parser.add_argument(
        "--tcia-list-candidates",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, print ranked TCIA candidate CT series and exit (no download/conversion).",
    )
    parser.add_argument(
        "--tcia-max-candidates",
        type=int,
        default=20,
        help="How many candidate series to print with --tcia-list-candidates (default: 20).",
    )
    args = parser.parse_args()

    if args.tcia_list_candidates:
        list_tcia_candidates(
            args.input,
            args.tcia_patient_id,
            args.tcia_bodypart,
            bool(args.tcia_require_head_keywords),
            args.tcia_max_candidates,
        )
        return

    if not args.out:
        raise ValueError("--out is required unless --tcia-list-candidates 1 is used.")

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
    in_path = maybe_dicom_dir if maybe_dicom_dir is not None else Path(args.input).expanduser().resolve()

    target_shape = parse_target_shape(args.target_shape)

    print(f"[info] Loading CT from: {in_path}")
    hu_xyz, spacing_xyz_mm, source_kind = load_ct_volume(in_path)
    print(f"[info] Loaded {source_kind} volume, shape={hu_xyz.shape}, spacing_mm={spacing_xyz_mm}")
    print(f"[info] HU range: min={hu_xyz.min():.1f}, max={hu_xyz.max():.1f}")

    if target_shape is not None:
        print(f"[info] Resampling to target shape: {target_shape} (order={args.resample_order})")
        hu_xyz, spacing_xyz_mm = resample_volume(hu_xyz, spacing_xyz_mm, target_shape, args.resample_order)
        print(f"[info] Resampled shape={hu_xyz.shape}, spacing_mm={spacing_xyz_mm}")

    c_xyz, rho_xyz, att_xyz, masks = map_acoustic_properties(hu_xyz)

    np.save(out_dir / "hu_xyz.npy", hu_xyz.astype(np.float32))
    np.save(out_dir / "sound_speed_xyz.npy", c_xyz.astype(np.float32))
    np.save(out_dir / "density_xyz.npy", rho_xyz.astype(np.float32))
    np.save(out_dir / "attenuation_db_cm_mhz_xyz.npy", att_xyz.astype(np.float32))

    write_stats_csv(out_dir / "tissue_stats.csv", masks, hu_xyz)

    meta = {
        "source_kind": source_kind,
        "input_path": str(in_path),
        "shape_xyz": list(hu_xyz.shape),
        "spacing_xyz_mm": list(spacing_xyz_mm),
        "tissue_thresholds_hu": {
            "air": "HU < -500",
            "csf": "-10 <= HU < 20",
            "brain_soft": "20 <= HU < 120 plus unknown fallback",
            "trabecular_skull": "120 <= HU < 700",
            "cortical_skull": "HU >= 700",
        },
        "acoustic_table": {
            k: {
                "sound_speed_m_per_s": v.sound_speed,
                "density_kg_per_m3": v.density,
                "attenuation_db_per_cm_per_mhz": v.attenuation_db_cm_mhz,
            }
            for k, v in DEFAULT_TISSUE_TABLE.items()
        },
        "for_jwave": {
            "sound_speed_file": "sound_speed_xyz.npy",
            "density_file": "density_xyz.npy",
            "attenuation_file": "attenuation_db_cm_mhz_xyz.npy",
            "note": "Use sound_speed and density directly for Medium; attenuation optional depending on solver config.",
        },
        "tcia_auto_download": {
            "enabled": maybe_dicom_dir is not None,
            "input_spec": args.input,
            "downloaded_dicom_dir": str(maybe_dicom_dir) if maybe_dicom_dir is not None else None,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[ok] Saved files:")
    print(f"  - {out_dir / 'hu_xyz.npy'}")
    print(f"  - {out_dir / 'sound_speed_xyz.npy'}")
    print(f"  - {out_dir / 'density_xyz.npy'}")
    print(f"  - {out_dir / 'attenuation_db_cm_mhz_xyz.npy'}")
    print(f"  - {out_dir / 'tissue_stats.csv'}")
    print(f"  - {out_dir / 'metadata.json'}")
    print("[ok] Done.")


if __name__ == "__main__":
    main()

