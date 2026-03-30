#!/usr/bin/env python3
"""
CT Scan Structural Comparison Analysis
=======================================
Geometrical and anatomical structural comparisons across all CT scans.
Complements ct_scan_comparison_analysis.py (which focuses on acoustic metrics
and pairwise distribution distances).

Structural analyses included
----------------------------
A) Volume-based  (tissue_stats.csv + metadata.json — no large arrays needed):
   01 – Intracranial volume (ICV) and skull volume per patient (stacked bar)
   02 – Skull-to-brain volume ratio per patient
   03 – Skull porosity (trabecular / total-skull voxel fraction) per patient
   04 – Skull HU: weighted mean ± std and coefficient of variation
   05 – GM / WM HU difference per patient (tissue differentiation proxy)
   06 – CSF fraction within ICV per patient

B) Geometry-based  (density_xyz + body_mask arrays — sub-sampled for speed):
   07 – 3-D shape descriptors: sphericity, elongation, body volume
   08 – Axial cross-section area profiles (body area vs z-slice)
   09 – Left–right structural symmetry (density correlation & asymmetry index)
   10 – Skull wall thickness distributions (distance-transform estimate)
   11 – Skull surface roughness (surface-normal angular deviation)

C) Pairwise & multivariate:
   12 – Pairwise structural Euclidean distance matrix (heatmap)
   13 – Hierarchical clustering dendrogram (Ward linkage)
   14 – Structural feature radar chart (group means)
   15 – Skull thickness vs ICV scatter
   16 – Skull porosity vs skull HU scatter

D) Shape comparisons (body-mask geometry):
   17 – Multi-level Dice shape similarity matrix (z = 25 / 50 / 75 %)
   18 – Pairwise Hausdorff distance at mid-axial level (surface shape)
   19 – 3-D shape PCA scatter (patients embedded in shape space)
   20 – Cross-section contour overlay at 3 z-levels (shape cloud)
   21 – Cross-section aspect ratio profiles (width/depth vs z-slice)
   21 – Skull surface Chamfer distance matrix (ICP-aligned, mm)
   22 – Skull surface P95 distance matrix (ICP-aligned, mm)
   23 – Chamfer pipeline debug (best pair)
   24 – Chamfer pipeline debug (worst pair)
   25 – Skull volume IoU matrix (ICP-aligned)
   26 – Skull volume Dice matrix (ICP-aligned)
   27 – Volume IoU pipeline debug (highest IoU pair)
   28 – Volume IoU pipeline debug (lowest IoU pair)

Outputs
-------
  ct_structural_comparison/
    01_icv_skull_volumes.png  …  28_volume_iou_pipeline_debug_worst_pair.png
    structural_features.csv   — all per-patient structural scalars

Environment overrides (same convention as ct_scan_comparison_analysis.py):
  CT_ACOUSTIC_DIR   – acoustic_maps/<pid>/   (density + body_mask)
  CT_PROC_DIR       – processed/<pid>/tissue_stats.csv
  CT_STRUCTURAL_OUT – output directory

Also prints cohort mean ± std for structural metrics, and Chamfer / IoU pair stats
(upper triangle), to the console — useful for slide copy.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import binary_erosion, distance_transform_edt, zoom
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff, pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths (same env-var override convention as ct_scan_comparison_analysis) ───
_BASE_DEFAULT = Path(
    "/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results"
)
BASE_DIR = Path(os.environ.get("CT_COMPARISON_BASE", str(_BASE_DEFAULT)))
ACOUSTIC_DIR = Path(
    os.environ.get("CT_ACOUSTIC_DIR", str(BASE_DIR / "acoustic_maps"))
)
PROC_DIR = Path(
    os.environ.get(
        "CT_PROC_DIR",
        str(BASE_DIR / "kaggle_brain_tumor_mri_ct" / "processed_all_42_patients_ct"),
    )
)
OUT_DIR = Path(
    os.environ.get("CT_STRUCTURAL_OUT", str(BASE_DIR / "ct_structural_comparison"))
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# Sub-sampling stride applied to large 3-D arrays.
# Stride 4 on a 512³ scan → 128³ (~2 M voxels) — cheap but representative.
SPATIAL_STRIDE = 4

# Density threshold (kg/m³) that separates skull from soft tissue.
# After the corrected HU+1000 mapping: brain ~1042, trabecular ~1316, cortical ~1734.
# 1200 sits cleanly in the gap.
SKULL_DENSITY_THR = 1200.0

# Tissues that contribute to intracranial volume.
# Works for both the 4-label CQ500 layout (brain_soft) and the
# 9-label original layout (brain_gm, brain_wm, csf, …).
_ICV_TISSUES = (
    "brain_soft", "brain_gm", "brain_wm",
    "csf", "blood", "fat", "soft_tissue",
)
_SKULL_TISSUES = ("cortical_skull", "trabecular_skull")

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


# ── Label helpers ─────────────────────────────────────────────────────────────

def short(pid: str) -> str:
    return (
        pid
        .replace("ACRIN-FMISO-Brain-", "AF-")
        .replace("TCGA-14-", "T14-")
        .replace("CQ500-CT-", "CQ-")
    )


def patient_group_color(pid: str) -> str:
    if "ACRIN" in pid:  return "steelblue"
    if "TCGA"  in pid:  return "tomato"
    if "C3L"   in pid:  return "forestgreen"
    if "C3N"   in pid:  return "darkorchid"
    if "CQ500" in pid:  return "darkorange"
    return "gray"


_LEGEND_PATCHES = [
    Patch(color="steelblue",   label="ACRIN-FMISO-Brain"),
    Patch(color="tomato",      label="TCGA-14"),
    Patch(color="forestgreen", label="C3L"),
    Patch(color="darkorchid",  label="C3N"),
    Patch(color="darkorange",  label="CQ500"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def discover_patients() -> list[str]:
    """
    Find patients with all required files for structural analysis:
      acoustic_maps/<pid>/<pid>_density_cleaned.npy
      acoustic_maps/<pid>/<pid>_body_mask.npy
      acoustic_maps/<pid>/metadata.json
      processed/<pid>/tissue_stats.csv
    """
    patients = sorted([
        d.name for d in ACOUSTIC_DIR.iterdir()
        if d.is_dir()
        and (d / f"{d.name}_density_cleaned.npy").exists()
        and (d / f"{d.name}_body_mask.npy").exists()
        and (d / "metadata.json").exists()
        and (PROC_DIR / d.name / "tissue_stats.csv").exists()
    ])
    print(f"  Found {len(patients)} patients with complete structural data.")
    return patients


def load_tissue_stats(patients: list[str]) -> pd.DataFrame:
    """Wide DataFrame: per-tissue voxel counts, fractions, HU stats."""
    records = []
    for pid in patients:
        csv_p = PROC_DIR / pid / "tissue_stats.csv"
        df = pd.read_csv(csv_p)
        row: dict = {"patient_id": pid}
        for _, r in df.iterrows():
            t = str(r["tissue"])
            row[f"count_{t}"]   = float(r.get("voxel_count", 0.0))
            row[f"frac_{t}"]    = float(r.get("fraction", 0.0))
            row[f"hu_mean_{t}"] = float(r.get("hu_mean",  np.nan))
            row[f"hu_std_{t}"]  = float(r.get("hu_std",   np.nan))
        records.append(row)
    return pd.DataFrame(records).set_index("patient_id")


def load_metadata(patients: list[str]) -> pd.DataFrame:
    """shape_xyz and spacing_xyz_mm from metadata.json."""
    records = []
    for pid in patients:
        row: dict = {"patient_id": pid}
        meta_p = ACOUSTIC_DIR / pid / "metadata.json"
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            shape   = m.get("shape_xyz",     [np.nan, np.nan, np.nan])
            spacing = m.get("spacing_xyz_mm", [np.nan, np.nan, np.nan])
            row.update({
                "shape_x":        float(shape[0]),
                "shape_y":        float(shape[1]),
                "shape_z":        float(shape[2]),
                "spacing_x_mm":   float(spacing[0]),
                "spacing_y_mm":   float(spacing[1]),
                "spacing_z_mm":   float(spacing[2]),
                "voxel_vol_mm3":  float(spacing[0]) * float(spacing[1]) * float(spacing[2]),
            })
        records.append(row)
    return pd.DataFrame(records).set_index("patient_id")


def _load_density_body(pid: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    pdir   = ACOUSTIC_DIR / pid
    d_path = pdir / f"{pid}_density_cleaned.npy"
    b_path = pdir / f"{pid}_body_mask.npy"
    d = np.load(d_path, mmap_mode="r") if d_path.exists() else None
    b = np.load(b_path, mmap_mode="r") if b_path.exists() else None
    return d, b


def _effective_spacing(meta_df: pd.DataFrame, pid: str, stride: int) -> list[float]:
    if pid in meta_df.index:
        sp = meta_df.loc[pid]
        return [
            float(sp["spacing_x_mm"]) * stride,
            float(sp["spacing_y_mm"]) * stride,
            float(sp["spacing_z_mm"]) * stride,
        ]
    return [float(stride)] * 3


# ── A: Volume-based structural features ──────────────────────────────────────

def compute_volume_features(
    stats_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    patients: list[str],
) -> pd.DataFrame:
    """
    Derive structural scalars from tissue_stats + metadata only.
    No large-array loading required.
    """
    records = []
    for pid in patients:
        if pid not in stats_df.index or pid not in meta_df.index:
            continue
        row     = stats_df.loc[pid]
        vox_vol = float(meta_df.loc[pid].get("voxel_vol_mm3", np.nan))

        def cnt(tissue: str) -> float:
            v = row.get(f"count_{tissue}", 0.0)
            return float(v) if np.isfinite(float(v)) else 0.0

        cortical    = cnt("cortical_skull")
        trabecular  = cnt("trabecular_skull")
        skull_total = cortical + trabecular
        # Fallback for merged-label runs that only have "skull"
        if skull_total == 0:
            skull_total = cnt("skull")

        icv_voxels = sum(cnt(t) for t in _ICV_TISSUES)

        # Volumes in cm³ (1 cm³ = 1 000 mm³)
        icv_cm3  = icv_voxels  * vox_vol / 1000.0
        skl_cm3  = skull_total * vox_vol / 1000.0
        head_cm3 = (icv_voxels + skull_total) * vox_vol / 1000.0

        skull_brain_ratio = skl_cm3 / icv_cm3  if icv_cm3  > 0 else np.nan
        skull_porosity    = trabecular / skull_total if skull_total > 0 else np.nan

        # Weighted skull HU (mean and std, weighted by voxel count)
        wsum_mu, wsum_sd, total_w = 0.0, 0.0, 0.0
        for t in _SKULL_TISSUES:
            c  = cnt(t)
            mu = float(row.get(f"hu_mean_{t}", np.nan))
            sd = float(row.get(f"hu_std_{t}",  np.nan))
            if c > 0 and np.isfinite(mu):
                wsum_mu += c * mu
                wsum_sd += c * (sd if np.isfinite(sd) else 0.0)
                total_w += c
        skull_hu_mean = wsum_mu / total_w if total_w > 0 else np.nan
        skull_hu_std  = wsum_sd / total_w if total_w > 0 else np.nan
        skull_hu_cv   = skull_hu_std / abs(skull_hu_mean) if skull_hu_mean else np.nan

        # GM – WM HU difference: proxy for tissue contrast / scan quality
        gm_hu = float(row.get("hu_mean_brain_gm",  np.nan))
        wm_hu = float(row.get("hu_mean_brain_wm",  np.nan))
        gm_wm_diff = gm_hu - wm_hu if np.isfinite(gm_hu) and np.isfinite(wm_hu) else np.nan

        # CSF fraction within ICV (ventricular load proxy)
        csf_cnt      = cnt("csf")
        csf_frac_icv = csf_cnt / icv_voxels if icv_voxels > 0 else np.nan

        records.append({
            "patient_id":        pid,
            "icv_cm3":           round(icv_cm3,  1),
            "skull_cm3":         round(skl_cm3,  1),
            "head_cm3":          round(head_cm3, 1),
            "skull_brain_ratio": skull_brain_ratio,
            "skull_porosity":    skull_porosity,
            "skull_hu_mean":     skull_hu_mean,
            "skull_hu_std":      skull_hu_std,
            "skull_hu_cv":       skull_hu_cv,
            "gm_wm_hu_diff":     gm_wm_diff,
            "csf_frac_icv":      csf_frac_icv,
        })
    return pd.DataFrame(records).set_index("patient_id")


# ── B: Geometry-based structural features ────────────────────────────────────

def compute_shape_descriptors(
    patients: list[str],
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    3-D shape descriptors from the binary body mask (sub-sampled):
      sphericity  = π^(1/3) * (6V)^(2/3) / A   (sphere=1, <1 otherwise)
      elongation  = λ₁ / λ₃  (ratio of PCA eigenvalues — 1=isotropic, >1=elongated)
      body_vol_cm3 — estimated body-mask volume
    """
    s = SPATIAL_STRIDE
    records = []
    for pid in patients:
        _, bm = _load_density_body(pid)
        if bm is None:
            continue
        sp = _effective_spacing(meta_df, pid, s)
        b  = np.asarray(bm[::s, ::s, ::s]).astype(bool)
        coords = np.argwhere(b).astype(float)
        if coords.shape[0] < 200:
            continue

        vox_vol_mm3 = sp[0] * sp[1] * sp[2]
        vol_mm3     = float(b.sum()) * vox_vol_mm3
        vol_cm3     = vol_mm3 / 1000.0

        # Surface voxel count via erosion; average face area in mm²
        surface_n = int((b & ~binary_erosion(b)).sum())
        avg_face_mm2 = (sp[0]*sp[1] + sp[1]*sp[2] + sp[0]*sp[2]) / 3.0
        sa_mm2 = surface_n * avg_face_mm2

        sphericity = (
            (np.pi ** (1.0 / 3.0)) * (6.0 * vol_mm3) ** (2.0 / 3.0) / sa_mm2
            if sa_mm2 > 0 else np.nan
        )

        # PCA elongation from physical voxel coordinates (mm)
        coords_mm = coords * np.array(sp)
        cov       = np.cov(coords_mm.T)
        eigvals   = np.sort(np.linalg.eigvalsh(cov))[::-1]   # descending
        elongation = float(eigvals[0] / eigvals[2]) if eigvals[2] > 1e-6 else np.nan

        records.append({
            "patient_id":  pid,
            "body_vol_cm3": round(vol_cm3, 1),
            "sphericity":   sphericity,
            "elongation":   elongation,
        })
    return pd.DataFrame(records).set_index("patient_id")


def compute_axial_area_profiles(
    patients: list[str],
    meta_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Per patient: body-mask cross-section area (mm²) as a function of z-slice.
    x,y are sub-sampled; full z-resolution is kept for a smooth profile.
    Returns {pid: 1-D float array of length nz}.
    """
    s = SPATIAL_STRIDE
    profiles: dict[str, np.ndarray] = {}
    for pid in patients:
        _, bm = _load_density_body(pid)
        if bm is None:
            continue
        meta = meta_df.loc[pid] if pid in meta_df.index else None
        sx   = float(meta["spacing_x_mm"]) * s if meta is not None else float(s)
        sy   = float(meta["spacing_y_mm"]) * s if meta is not None else float(s)
        b    = np.asarray(bm[::s, ::s, :])         # (nx_sub, ny_sub, nz)
        area_per_z = b.sum(axis=(0, 1)).astype(float) * (sx * sy)   # mm²
        profiles[pid] = area_per_z
    return profiles


def compute_lr_symmetry(patients: list[str]) -> pd.DataFrame:
    """
    Left–right structural symmetry estimated from the density array.
    The volume is split at x-midpoint; sorted density distributions of each
    half are correlated — a proxy for bilateral anatomical symmetry.

    Returns:
      lr_correlation      (1 = perfectly symmetric)
      lr_asymmetry_index  (0 = symmetric; higher = asymmetric mean density)
    """
    s = SPATIAL_STRIDE
    rng = np.random.default_rng(42)
    records = []
    for pid in patients:
        d, bm = _load_density_body(pid)
        if d is None or bm is None:
            continue
        den = np.asarray(d[::s, ::s, ::s]).astype(np.float32)
        b   = np.asarray(bm[::s, ::s, ::s]).astype(bool)
        mid = den.shape[0] // 2

        left  = den[:mid, :, :][b[:mid, :, :]]
        right = den[mid:, :, :][b[mid:, :, :]]
        n = min(left.size, right.size, 60_000)
        if n < 200:
            continue

        left_s  = np.sort(rng.choice(left,  n, replace=False))[::-1]
        right_s = np.sort(rng.choice(right, n, replace=False))[::-1]

        corr = float(np.corrcoef(left_s, right_s)[0, 1])
        asym = float(
            abs(float(left_s.mean()) - float(right_s.mean()))
            / (float(left_s.mean()) + float(right_s.mean()) + 1e-6)
        )
        records.append({
            "patient_id":         pid,
            "lr_correlation":     corr,
            "lr_asymmetry_index": asym,
        })
    return pd.DataFrame(records).set_index("patient_id")


def compute_skull_thickness(
    patients: list[str],
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate skull wall thickness (mm) via Euclidean distance transform (EDT):
      1. Build skull mask: density ≥ SKULL_DENSITY_THR within body mask
      2. Compute EDT of the complement (distance of each voxel to nearest non-skull)
      3. Skull voxels' EDT values ≈ distance to inner/outer skull surface → thickness proxy
      4. Clamp to 30 mm to remove segmentation artefacts

    Summary statistics per patient: mean, std, P25, P75, P95.
    """
    s = SPATIAL_STRIDE
    records = []
    for pid in patients:
        d, bm = _load_density_body(pid)
        if d is None or bm is None:
            continue
        sp  = _effective_spacing(meta_df, pid, s)
        den = np.asarray(d[::s, ::s, ::s]).astype(np.float32)
        b   = np.asarray(bm[::s, ::s, ::s]).astype(bool)

        skull = (den >= SKULL_DENSITY_THR) & b
        if skull.sum() < 50:
            continue
        try:
            # EDT in mm; sampling=(sz_x, sz_y, sz_z) converts voxel dist to mm
            edt        = distance_transform_edt(~skull, sampling=sp)
            thick_vals = edt[skull]
            thick_vals = thick_vals[thick_vals <= 30.0]
            if thick_vals.size < 10:
                continue
            records.append({
                "patient_id":       pid,
                "skull_thick_mean": float(np.mean(thick_vals)),
                "skull_thick_std":  float(np.std(thick_vals)),
                "skull_thick_p25":  float(np.percentile(thick_vals, 25)),
                "skull_thick_p75":  float(np.percentile(thick_vals, 75)),
                "skull_thick_p95":  float(np.percentile(thick_vals, 95)),
            })
        except Exception:
            pass
    return pd.DataFrame(records).set_index("patient_id")


def compute_skull_roughness(
    patients: list[str],
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Skull surface roughness: angular standard deviation of outer-surface normals.
    A high value means the skull shape is locally irregular (relevant for transducer fit).

    Strategy:
      1. EDT of body mask → smooth distance field
      2. Narrow-band shell (EDT ≤ 2.5 × voxel width) = outer surface
      3. Gradient of the distance field ≈ outward surface normal
      4. Angle of each normal relative to the mean normal → std = roughness
    """
    s = SPATIAL_STRIDE
    records = []
    for pid in patients:
        _, bm = _load_density_body(pid)
        if bm is None:
            continue
        sp = _effective_spacing(meta_df, pid, s)
        b  = np.asarray(bm[::s, ::s, ::s]).astype(bool)
        if b.sum() < 200:
            continue
        try:
            sdf     = distance_transform_edt(b, sampling=sp).astype(np.float32)
            surface = (sdf > 0) & (sdf <= max(sp) * 2.5)
            if surface.sum() < 50:
                continue
            # Gradients of the distance field at surface voxels = outward normals
            gx = np.gradient(sdf, sp[0], axis=0)[surface]
            gy = np.gradient(sdf, sp[1], axis=1)[surface]
            gz = np.gradient(sdf, sp[2], axis=2)[surface]
            mag = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-9
            nx_n, ny_n, nz_n = gx / mag, gy / mag, gz / mag

            mean_n = np.array([nx_n.mean(), ny_n.mean(), nz_n.mean()])
            mean_n /= np.linalg.norm(mean_n) + 1e-9
            dots   = np.clip(nx_n*mean_n[0] + ny_n*mean_n[1] + nz_n*mean_n[2], -1.0, 1.0)
            angles = np.degrees(np.arccos(dots))
            records.append({
                "patient_id":              pid,
                "skull_roughness_deg_mean": float(np.mean(angles)),
                "skull_roughness_deg_std":  float(np.std(angles)),
            })
        except Exception:
            pass
    return pd.DataFrame(records).set_index("patient_id")


# ── D: Shape comparison functions ────────────────────────────────────────────

_SHAPE_LEVELS: tuple[float, ...] = (0.25, 0.50, 0.75)
_SHAPE_SLICE_SIZE: int = 64          # pixels per side after resizing
_SHAPE_PCA_VOL:    tuple = (32, 32, 32)  # 3-D grid for full-volume PCA

_LEVEL_LABELS = {0.25: "Inferior (z=25%)", 0.50: "Mid-axial (z=50%)", 0.75: "Superior (z=75%)"}


def extract_multilevel_slices(
    patients: list[str],
    levels: tuple[float, ...] = _SHAPE_LEVELS,
    target_size: int = _SHAPE_SLICE_SIZE,
) -> dict[str, dict[float, np.ndarray]]:
    """
    For each patient and z-fraction level, extract a 2D binary body-mask
    cross-section and resize it to target_size × target_size.

    Sub-samples x and y by SPATIAL_STRIDE before choosing the z-slice so that
    the slice is taken from the same sub-sampled volume used everywhere else.

    Returns {pid: {level_fraction: binary_float32_array}}.
    """
    s = SPATIAL_STRIDE
    result: dict[str, dict[float, np.ndarray]] = {}
    for pid in patients:
        _, bm = _load_density_body(pid)
        if bm is None:
            continue
        b  = np.asarray(bm[::s, ::s, ::s]).astype(bool)
        nz = b.shape[2]
        slices: dict[float, np.ndarray] = {}
        for lev in levels:
            zi  = max(0, min(nz - 1, int(round(lev * (nz - 1)))))
            sl  = b[:, :, zi].astype(np.float32)
            if sl.sum() < 20:
                continue
            if sl.shape[0] != target_size or sl.shape[1] != target_size:
                sx = target_size / sl.shape[0]
                sy = target_size / sl.shape[1]
                sl = zoom(sl, (sx, sy), order=1)
            slices[lev] = (sl > 0.5).astype(np.float32)
        if slices:
            result[pid] = slices
    return result


def pairwise_multilevel_dice(
    multilevel_slices: dict[str, dict[float, np.ndarray]],
    patients: list[str],
    levels: tuple[float, ...] = _SHAPE_LEVELS,
) -> pd.DataFrame:
    """
    N×N mean Dice coefficient averaged across all z-fraction levels.
    Higher = more similar 2-D cross-section shapes.
    """
    n   = len(patients)
    mat = np.full((n, n), np.nan)
    for i, p1 in enumerate(patients):
        for j, p2 in enumerate(patients):
            s1 = multilevel_slices.get(p1)
            s2 = multilevel_slices.get(p2)
            if s1 is None or s2 is None:
                continue
            dices = []
            for lev in levels:
                a = s1.get(lev)
                b = s2.get(lev)
                if a is None or b is None:
                    continue
                b1 = a > 0.5
                b2 = b > 0.5
                denom = float(b1.sum() + b2.sum())
                if denom > 0:
                    dices.append(2.0 * float((b1 & b2).sum()) / denom)
            if dices:
                mat[i, j] = float(np.mean(dices))
    return pd.DataFrame(mat, index=patients, columns=patients)


def _boundary_points(mask2d: np.ndarray, n_pts: int = 400) -> np.ndarray | None:
    """
    Extract ~n_pts boundary pixels of a binary 2D mask, centred at their centroid.
    Returns array of shape (N, 2) in (row, col) order, or None if the mask is empty.
    """
    b = mask2d > 0.5
    if b.sum() < 20:
        return None
    boundary = b & ~binary_erosion(b)
    pts = np.argwhere(boundary).astype(float)
    if pts.shape[0] < 5:
        pts = np.argwhere(b).astype(float)   # fallback: all voxels
    if pts.shape[0] > n_pts:
        idx = np.random.default_rng(0).choice(len(pts), n_pts, replace=False)
        pts = pts[idx]
    pts -= pts.mean(axis=0)   # centre at centroid
    return pts


def pairwise_hausdorff_2d(
    multilevel_slices: dict[str, dict[float, np.ndarray]],
    patients: list[str],
    level: float = 0.50,
) -> pd.DataFrame:
    """
    N×N symmetric Hausdorff distance (in pixels of the resized slice) at
    the specified z-fraction level, after centroid alignment.
    Lower = more similar cross-section shapes.
    """
    n        = len(patients)
    mat      = np.full((n, n), np.nan)
    contours = {
        pid: _boundary_points(multilevel_slices.get(pid, {}).get(level, np.zeros((1, 1))))
        for pid in patients
    }
    for i, p1 in enumerate(patients):
        mat[i, i] = 0.0
        for j, p2 in enumerate(patients):
            if i == j:
                continue
            c1, c2 = contours.get(p1), contours.get(p2)
            if c1 is None or c2 is None:
                continue
            d12 = directed_hausdorff(c1, c2)[0]
            d21 = directed_hausdorff(c2, c1)[0]
            mat[i, j] = max(d12, d21)
    return pd.DataFrame(mat, index=patients, columns=patients)


def compute_shape_pca(
    patients: list[str],
    target_vol: tuple[int, int, int] = _SHAPE_PCA_VOL,
) -> tuple[np.ndarray, PCA | None, list[str]]:
    """
    Embed each patient's 3-D body mask into a common shape space via PCA.

    Each mask is downsampled to target_vol (e.g. 32×32×32) and flattened to a
    vector, then normalised.  PCA is run on the resulting (N_patients × V)
    matrix.  Returns (pc_coords, pca_object, available_pids).
    """
    s = SPATIAL_STRIDE
    vectors: list[np.ndarray] = []
    pids_avail: list[str]     = []
    for pid in patients:
        _, bm = _load_density_body(pid)
        if bm is None:
            continue
        b = np.asarray(bm[::s, ::s, ::s]).astype(np.float32)
        if b.sum() < 200:
            continue
        scale  = tuple(t / c for t, c in zip(target_vol, b.shape))
        b_small = zoom(b, scale, order=1)
        b_small = (b_small > 0.5).astype(np.float32)
        vectors.append(b_small.flatten())
        pids_avail.append(pid)

    if len(vectors) < 3:
        return np.empty((0, 2)), None, pids_avail

    X      = np.array(vectors)
    n_comp = min(3, X.shape[0] - 1, X.shape[1])
    pca    = PCA(n_components=n_comp)
    coords = pca.fit_transform(X)
    return coords, pca, pids_avail


def compute_cross_section_aspect_ratios(
    patients: list[str],
) -> dict[str, np.ndarray]:
    """
    For every z-slice, fit an ellipse to the body-mask cross-section using
    image moments (PCA of 2-D voxel coordinates) and return the
    major/minor axis ratio (λ₁/λ₂).

    Ratio = 1 → circular cross-section; >1 → elongated / elliptical.

    Returns {pid: float array of length nz_subsampled}.
    """
    s = SPATIAL_STRIDE
    result: dict[str, np.ndarray] = {}
    for pid in patients:
        _, bm = _load_density_body(pid)
        if bm is None:
            continue
        b  = np.asarray(bm[::s, ::s, ::s]).astype(bool)
        nz = b.shape[2]
        ratios = np.full(nz, np.nan)
        for zi in range(nz):
            sl = b[:, :, zi]
            if sl.sum() < 10:
                continue
            coords = np.argwhere(sl).astype(float)
            cov    = np.cov(coords.T)
            eigs   = np.sort(np.linalg.eigvalsh(cov))[::-1]
            if eigs[1] > 1e-6:
                ratios[zi] = float(eigs[0] / eigs[1])
        result[pid] = ratios
    return result


# ── D2: Surface-based 3-D skull comparison (point cloud + ICP + Chamfer) ───────

def _skull_surface_point_cloud_mm(
    pid: str,
    meta_df: pd.DataFrame,
    *,
    max_points: int = 40_000,
    rng_seed: int = 0,
) -> np.ndarray | None:
    """
    Build a skull *surface* point cloud (mm) from density + body_mask.

    We avoid mesh dependencies by extracting boundary voxels:
      skull = (density >= SKULL_DENSITY_THR) & body_mask
      surface = skull & ~erode(skull)
    Points are returned in physical space using scan spacing and subsampling stride.
    """
    s = SPATIAL_STRIDE
    d, bm = _load_density_body(pid)
    if d is None or bm is None:
        return None
    den = np.asarray(d[::s, ::s, ::s]).astype(np.float32)
    b   = np.asarray(bm[::s, ::s, ::s]).astype(bool)
    skull = (den >= SKULL_DENSITY_THR) & b
    if skull.sum() < 200:
        return None
    surf = skull & ~binary_erosion(skull)
    pts  = np.argwhere(surf).astype(np.float32)  # voxel coords in sub-sampled grid
    if pts.shape[0] < 200:
        return None

    sp = _effective_spacing(meta_df, pid, s)
    pts_mm = pts * np.array(sp, dtype=np.float32)

    if pts_mm.shape[0] > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(pts_mm.shape[0], max_points, replace=False)
        pts_mm = pts_mm[idx]
    return pts_mm


def _skull_volume_point_cloud_mm(
    pid: str,
    meta_df: pd.DataFrame,
    *,
    max_points: int = 120_000,
    rng_seed: int = 0,
) -> np.ndarray | None:
    """
    Build skull *volume* point cloud (mm) from density + body_mask.
    Uses all skull voxels (not only boundary), then optional random subsampling.
    """
    s = SPATIAL_STRIDE
    d, bm = _load_density_body(pid)
    if d is None or bm is None:
        return None
    den = np.asarray(d[::s, ::s, ::s]).astype(np.float32)
    b   = np.asarray(bm[::s, ::s, ::s]).astype(bool)
    skull = (den >= SKULL_DENSITY_THR) & b
    if skull.sum() < 200:
        return None
    pts = np.argwhere(skull).astype(np.float32)
    sp = _effective_spacing(meta_df, pid, s)
    pts_mm = pts * np.array(sp, dtype=np.float32)
    if pts_mm.shape[0] > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(pts_mm.shape[0], max_points, replace=False)
        pts_mm = pts_mm[idx]
    return pts_mm


def _rigid_kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find rotation R and translation t that best aligns A to B (least squares):
      A @ R + t ≈ B
    A and B are (N,3) with correspondences.
    """
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    H  = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Proper rotation (det=+1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = B.mean(axis=0) - A.mean(axis=0) @ R
    return R.astype(np.float32), t.astype(np.float32)


def _icp_point_to_point(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    iters: int = 20,
    max_corr_dist_mm: float = 10.0,
    sample_n: int = 12_000,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lightweight point-to-point ICP aligning src to tgt.
    Returns (aligned_src, R_total, t_total).
    """
    if src.shape[0] < 200 or tgt.shape[0] < 200:
        return src, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    rng = np.random.default_rng(rng_seed)
    s_idx = rng.choice(src.shape[0], min(sample_n, src.shape[0]), replace=False)
    t_idx = rng.choice(tgt.shape[0], min(sample_n, tgt.shape[0]), replace=False)
    S = src[s_idx].astype(np.float32)
    T = tgt[t_idx].astype(np.float32)

    # Initial: centroid align
    S = S - S.mean(axis=0, keepdims=True)
    T = T - T.mean(axis=0, keepdims=True)

    R_total = np.eye(3, dtype=np.float32)
    t_total = np.zeros(3, dtype=np.float32)

    tree = cKDTree(T)
    aligned = S.copy()

    for _ in range(iters):
        dists, nn = tree.query(aligned, k=1)
        ok = dists <= max_corr_dist_mm
        if int(ok.sum()) < 200:
            break
        A = aligned[ok]
        B = T[nn[ok]]
        R, t = _rigid_kabsch(A, B)
        aligned = aligned @ R + t
        R_total = R_total @ R
        t_total = t_total @ R + t

    return aligned, R_total, t_total


def _icp_rigid_transform_raw(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    iters: int = 20,
    max_corr_dist_mm: float = 10.0,
    sample_n: int = 12_000,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ICP on raw coordinates; returns global transform (R, t) such that:
      src @ R + t ≈ tgt
    """
    if src.shape[0] < 200 or tgt.shape[0] < 200:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    rng = np.random.default_rng(rng_seed)
    s_idx = rng.choice(src.shape[0], min(sample_n, src.shape[0]), replace=False)
    t_idx = rng.choice(tgt.shape[0], min(sample_n, tgt.shape[0]), replace=False)
    S = src[s_idx].astype(np.float32)
    T = tgt[t_idx].astype(np.float32)

    # Initial centroid translation
    t_total = (T.mean(axis=0) - S.mean(axis=0)).astype(np.float32)
    R_total = np.eye(3, dtype=np.float32)
    aligned = S + t_total
    tree = cKDTree(T)

    for _ in range(iters):
        dists, nn = tree.query(aligned, k=1)
        ok = dists <= max_corr_dist_mm
        if int(ok.sum()) < 200:
            break
        A = aligned[ok]
        B = T[nn[ok]]
        R_up, t_up = _rigid_kabsch(A, B)
        aligned = aligned @ R_up + t_up
        R_total = R_total @ R_up
        t_total = t_total @ R_up + t_up

    return R_total, t_total


def _voxelized_overlap_iou_dice(
    A_mm: np.ndarray,
    B_mm: np.ndarray,
    *,
    voxel_size_mm: float = 2.5,
) -> tuple[float, float]:
    """
    Voxelize two point clouds onto a shared regular grid and compute IoU/Dice.
    """
    if A_mm.shape[0] < 50 or B_mm.shape[0] < 50:
        return float("nan"), float("nan")

    all_pts = np.vstack([A_mm, B_mm])
    lo = np.min(all_pts, axis=0) - voxel_size_mm
    A_idx = np.floor((A_mm - lo) / voxel_size_mm).astype(np.int32)
    B_idx = np.floor((B_mm - lo) / voxel_size_mm).astype(np.int32)

    A_set = set(map(tuple, A_idx.tolist()))
    B_set = set(map(tuple, B_idx.tolist()))
    inter = len(A_set & B_set)
    union = len(A_set | B_set)
    if union == 0:
        return float("nan"), float("nan")
    iou = inter / union
    dice = (2 * inter) / (len(A_set) + len(B_set) + 1e-12)
    return float(iou), float(dice)


def chamfer_and_p95_mm(
    A: np.ndarray,
    B: np.ndarray,
    *,
    sample_n: int = 20_000,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """
    Symmetric surface distance metrics between two point clouds:
      chamfer = mean_nn(A->B) + mean_nn(B->A)
      p95     = 95th percentile of concatenated NN distances
    Units: mm.
    """
    if A.shape[0] < 200 or B.shape[0] < 200:
        return float("nan"), float("nan")

    rng = np.random.default_rng(rng_seed)
    if A.shape[0] > sample_n:
        A = A[rng.choice(A.shape[0], sample_n, replace=False)]
    if B.shape[0] > sample_n:
        B = B[rng.choice(B.shape[0], sample_n, replace=False)]

    treeB = cKDTree(B)
    dA, _ = treeB.query(A, k=1)
    treeA = cKDTree(A)
    dB, _ = treeA.query(B, k=1)
    chamfer = float(np.mean(dA) + np.mean(dB))
    p95 = float(np.percentile(np.r_[dA, dB], 95.0))
    return chamfer, p95


def pairwise_skull_surface_distances(
    patients: list[str],
    meta_df: pd.DataFrame,
    *,
    icp_iters: int = 20,
    max_corr_dist_mm: float = 10.0,
) -> dict[str, pd.DataFrame]:
    """
    Pairwise skull shape comparison:
      1) Extract skull surface point clouds
      2) ICP-align A->B (rigid, point-to-point)
      3) Compute Chamfer distance + P95 surface distance (mm)

    Returns dict with keys: 'chamfer_mm', 'p95_mm'.
    """
    pcs: dict[str, np.ndarray] = {}
    for pid in patients:
        pts = _skull_surface_point_cloud_mm(pid, meta_df)
        if pts is not None:
            pcs[pid] = pts

    avail = [p for p in patients if p in pcs]
    n = len(avail)
    chamfer = np.full((n, n), np.nan, dtype=float)
    p95     = np.full((n, n), np.nan, dtype=float)

    for i, pa in enumerate(avail):
        chamfer[i, i] = 0.0
        p95[i, i] = 0.0
        for j, pb in enumerate(avail):
            if i == j:
                continue
            A = pcs[pa]
            B = pcs[pb]
            A_aligned, _, _ = _icp_point_to_point(
                A, B, iters=icp_iters, max_corr_dist_mm=max_corr_dist_mm,
                rng_seed=(i * 1337 + j),
            )
            c, q = chamfer_and_p95_mm(A_aligned, B, rng_seed=(i * 733 + j))
            chamfer[i, j] = c
            p95[i, j] = q

    return {
        "chamfer_mm": pd.DataFrame(chamfer, index=avail, columns=avail),
        "p95_mm":     pd.DataFrame(p95,     index=avail, columns=avail),
    }


def pairwise_skull_volume_overlap(
    patients: list[str],
    meta_df: pd.DataFrame,
    *,
    icp_iters: int = 20,
    max_corr_dist_mm: float = 10.0,
    voxel_size_mm: float = 2.5,
) -> dict[str, pd.DataFrame]:
    """
    Pairwise volumetric skull comparison after rigid ICP alignment:
      1) Use skull surface points to estimate rigid transform A->B
      2) Transform skull volume points of A into B frame
      3) Voxelize both clouds on shared grid and compute IoU + Dice
    """
    surf_pcs: dict[str, np.ndarray] = {}
    vol_pcs: dict[str, np.ndarray] = {}
    for pid in patients:
        spts = _skull_surface_point_cloud_mm(pid, meta_df)
        vpts = _skull_volume_point_cloud_mm(pid, meta_df)
        if spts is not None and vpts is not None:
            surf_pcs[pid] = spts
            vol_pcs[pid] = vpts

    avail = [p for p in patients if p in surf_pcs and p in vol_pcs]
    n = len(avail)
    iou = np.full((n, n), np.nan, dtype=float)
    dice = np.full((n, n), np.nan, dtype=float)

    for i, pa in enumerate(avail):
        iou[i, i] = 1.0
        dice[i, i] = 1.0
        for j, pb in enumerate(avail):
            if i == j:
                continue
            R, t = _icp_rigid_transform_raw(
                surf_pcs[pa],
                surf_pcs[pb],
                iters=icp_iters,
                max_corr_dist_mm=max_corr_dist_mm,
                rng_seed=(i * 4001 + j),
            )
            A_vol_aligned = vol_pcs[pa] @ R + t
            B_vol = vol_pcs[pb]
            ov_iou, ov_dice = _voxelized_overlap_iou_dice(
                A_vol_aligned,
                B_vol,
                voxel_size_mm=voxel_size_mm,
            )
            iou[i, j] = ov_iou
            dice[i, j] = ov_dice

    return {
        "volume_iou": pd.DataFrame(iou, index=avail, columns=avail),
        "volume_dice": pd.DataFrame(dice, index=avail, columns=avail),
    }


def plot_surface_distance_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    off_diag = df.values[~np.eye(len(df), dtype=bool)]
    vmax = float(np.nanpercentile(off_diag[np.isfinite(off_diag)], 90)) if off_diag.size else 10.0
    _heatmap(
        df,
        title,
        out_path,
        vmin=0.0,
        vmax=max(vmax, 1e-3),
        cmap="YlOrRd",
        fmt=".2f",
    )


def plot_chamfer_pipeline_debug(
    source_id: str,
    target_id: str,
    meta_df: pd.DataFrame,
    out_path: Path,
    *,
    icp_iters: int = 20,
    max_corr_dist_mm: float = 10.0,
    sample_n: int = 12_000,
) -> None:
    """
    Visualize the Chamfer pipeline for one source->target pair:
      1) raw skull surface point clouds
      2) centroid-normalized sampled clouds
      3) ICP-aligned source against target
      4) NN distance map on aligned source
      5) NN distance map on target

    All geometry is in mm.
    """
    src = _skull_surface_point_cloud_mm(source_id, meta_df, max_points=40_000, rng_seed=11)
    tgt = _skull_surface_point_cloud_mm(target_id, meta_df, max_points=40_000, rng_seed=17)
    if src is None or tgt is None:
        return

    rng = np.random.default_rng(123)
    s_idx = rng.choice(src.shape[0], min(sample_n, src.shape[0]), replace=False)
    t_idx = rng.choice(tgt.shape[0], min(sample_n, tgt.shape[0]), replace=False)
    S0 = src[s_idx].astype(np.float32)
    T0 = tgt[t_idx].astype(np.float32)

    # Step 1: centroid normalization
    S = S0 - S0.mean(axis=0, keepdims=True)
    T = T0 - T0.mean(axis=0, keepdims=True)

    # Step 2: ICP alignment (same algorithm as production path)
    tree = cKDTree(T)
    aligned = S.copy()
    for _ in range(icp_iters):
        dists, nn = tree.query(aligned, k=1)
        ok = dists <= max_corr_dist_mm
        if int(ok.sum()) < 200:
            break
        A = aligned[ok]
        B = T[nn[ok]]
        R, t = _rigid_kabsch(A, B)
        aligned = aligned @ R + t

    # Step 3: final pairwise surface distances
    tree_t = cKDTree(T)
    d_src, _ = tree_t.query(aligned, k=1)
    tree_s = cKDTree(aligned)
    d_tgt, _ = tree_s.query(T, k=1)
    chamfer = float(np.mean(d_src) + np.mean(d_tgt))
    p95 = float(np.percentile(np.r_[d_src, d_tgt], 95.0))

    # 2D projections: (x,y), (y,z), (x,z)
    proj_pairs = [
        (0, 1, "X (mm)", "Y (mm)"),
        (1, 2, "Y (mm)", "Z (mm)"),
        (0, 2, "X (mm)", "Z (mm)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes.ravel()

    # Raw surfaces
    for k, (a, b, xl, yl) in enumerate(proj_pairs):
        ax[k].scatter(S0[:, a], S0[:, b], s=1, alpha=0.25, color="tab:blue", label=f"{source_id} raw")
        ax[k].scatter(T0[:, a], T0[:, b], s=1, alpha=0.25, color="tab:orange", label=f"{target_id} raw")
        ax[k].set_title(f"Step 1: Raw surface clouds ({xl},{yl})")
        ax[k].set_xlabel(xl)
        ax[k].set_ylabel(yl)
        ax[k].grid(alpha=0.2)
        if k == 0:
            ax[k].legend(markerscale=6, fontsize=7, loc="upper right")

    # Centroid-normalized
    for k, (a, b, xl, yl) in enumerate(proj_pairs):
        idx = 3 + k
        if k < 2:
            ax[idx].scatter(S[:, a], S[:, b], s=1, alpha=0.25, color="tab:blue", label="source centered")
            ax[idx].scatter(T[:, a], T[:, b], s=1, alpha=0.25, color="tab:orange", label="target centered")
            ax[idx].set_title(f"Step 2: Center-normalized ({xl},{yl})")
            ax[idx].set_xlabel(xl)
            ax[idx].set_ylabel(yl)
            ax[idx].grid(alpha=0.2)
            if k == 0:
                ax[idx].legend(markerscale=6, fontsize=7, loc="upper right")
        else:
            # Final aligned with distance coloring on source
            p = ax[idx].scatter(aligned[:, a], aligned[:, b], c=d_src, s=2, cmap="magma", alpha=0.7)
            ax[idx].scatter(T[:, a], T[:, b], s=1, alpha=0.15, color="cyan", label="target centered")
            ax[idx].set_title(
                "Step 3: ICP-aligned source + NN distance map\n"
                f"Chamfer={chamfer:.2f} mm, P95={p95:.2f} mm"
            )
            ax[idx].set_xlabel(xl)
            ax[idx].set_ylabel(yl)
            ax[idx].grid(alpha=0.2)
            ax[idx].legend(markerscale=6, fontsize=7, loc="upper right")
            cb = fig.colorbar(p, ax=ax[idx], fraction=0.046, pad=0.02)
            cb.set_label("source->target NN distance (mm)", fontsize=8)

    plt.suptitle(
        "Chamfer Pipeline Debug View\n"
        f"Source -> Target: {source_id} -> {target_id}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_volume_iou_pipeline_debug(
    source_id: str,
    target_id: str,
    meta_df: pd.DataFrame,
    out_path: Path,
    *,
    icp_iters: int = 20,
    max_corr_dist_mm: float = 10.0,
    voxel_size_mm: float = 2.5,
    display_sample_n: int = 12_000,
    icp_rng_seed: int = 0,
) -> None:
    """
    Visualize the volumetric IoU/Dice pipeline for one source→target pair:
      1) Raw skull volume point clouds (mm), before alignment
      2) Same clouds after ICP rigid transform estimated from *surface* points
      3) Voxelized overlap (2.5 mm grid): intersection vs A-only vs B-only

    Uses the same skull definition and ICP/voxelization as production
    `pairwise_skull_volume_overlap`.
    """
    # Same subsampling seeds as pairwise_skull_volume_overlap (default rng_seed=0).
    surf_src = _skull_surface_point_cloud_mm(source_id, meta_df, max_points=40_000, rng_seed=0)
    surf_tgt = _skull_surface_point_cloud_mm(target_id, meta_df, max_points=40_000, rng_seed=0)
    vol_src = _skull_volume_point_cloud_mm(source_id, meta_df, max_points=120_000, rng_seed=0)
    vol_tgt = _skull_volume_point_cloud_mm(target_id, meta_df, max_points=120_000, rng_seed=0)

    if surf_src is None or surf_tgt is None or vol_src is None or vol_tgt is None:
        return

    R, t = _icp_rigid_transform_raw(
        surf_src,
        surf_tgt,
        iters=icp_iters,
        max_corr_dist_mm=max_corr_dist_mm,
        rng_seed=icp_rng_seed,
    )
    A_aligned = vol_src @ R + t
    B = vol_tgt

    iou, dice = _voxelized_overlap_iou_dice(
        A_aligned, B, voxel_size_mm=voxel_size_mm,
    )

    rng = np.random.default_rng(42)
    n_src = min(display_sample_n, vol_src.shape[0])
    n_tgt = min(display_sample_n, vol_tgt.shape[0])
    n_aln = min(display_sample_n, A_aligned.shape[0])
    n_b = min(display_sample_n, B.shape[0])
    idx_s = rng.choice(vol_src.shape[0], n_src, replace=False)
    idx_t = rng.choice(vol_tgt.shape[0], n_tgt, replace=False)
    idx_a = rng.choice(A_aligned.shape[0], n_aln, replace=False)
    idx_b = rng.choice(B.shape[0], n_b, replace=False)

    proj_pairs = [
        (0, 1, "X (mm)", "Y (mm)"),
        (1, 2, "Y (mm)", "Z (mm)"),
        (0, 2, "X (mm)", "Z (mm)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    ax = axes.ravel()

    # Row 0: raw volumes (native coordinates — different head positions)
    for k, (a, b, xl, yl) in enumerate(proj_pairs):
        ax[k].scatter(vol_src[idx_s, a], vol_src[idx_s, b], s=1, alpha=0.2,
                      color="tab:blue", label=f"{source_id} vol")
        ax[k].scatter(vol_tgt[idx_t, a], vol_tgt[idx_t, b], s=1, alpha=0.2,
                      color="tab:orange", label=f"{target_id} vol")
        ax[k].set_title(f"Step 1: Raw volume clouds ({xl},{yl})")
        ax[k].set_xlabel(xl)
        ax[k].set_ylabel(yl)
        ax[k].grid(alpha=0.2)
        if k == 0:
            ax[k].legend(markerscale=6, fontsize=6, loc="upper right")

    # Row 1: after ICP (same R,t as production)
    for k, (a, b, xl, yl) in enumerate(proj_pairs):
        idx = 3 + k
        ax[idx].scatter(A_aligned[idx_a, a], A_aligned[idx_a, b], s=1, alpha=0.22,
                        color="tab:blue", label="source vol aligned")
        ax[idx].scatter(B[idx_b, a], B[idx_b, b], s=1, alpha=0.22,
                        color="tab:orange", label="target vol")
        ax[idx].set_title(f"Step 2: ICP-aligned source vol → target ({xl},{yl})")
        ax[idx].set_xlabel(xl)
        ax[idx].set_ylabel(yl)
        ax[idx].grid(alpha=0.2)
        if k == 0:
            ax[idx].legend(markerscale=6, fontsize=6, loc="upper right")

    # Row 3: voxel overlap — subsample voxel centers by category
    all_pts = np.vstack([A_aligned, B])
    lo = np.min(all_pts, axis=0) - voxel_size_mm
    A_idx = np.floor((A_aligned - lo) / voxel_size_mm).astype(np.int32)
    B_idx = np.floor((B - lo) / voxel_size_mm).astype(np.int32)
    A_set = set(map(tuple, A_idx.tolist()))
    B_set = set(map(tuple, B_idx.tolist()))
    inter = A_set & B_set
    a_only = A_set - B_set
    b_only = B_set - A_set

    def _voxel_centers_mm(idxs: set[tuple[int, int, int]], cap: int) -> np.ndarray:
        lst_all = list(idxs)
        if not lst_all:
            return np.empty((0, 3), dtype=np.float32)
        if len(lst_all) > cap:
            pick = rng.choice(len(lst_all), cap, replace=False)
            lst_all = [lst_all[i] for i in pick]
        arr = np.array(lst_all, dtype=np.float32)
        return lo.astype(np.float32) + (arr + 0.5) * float(voxel_size_mm)

    pts_inter = _voxel_centers_mm(inter, 10_000)
    pts_aonly = _voxel_centers_mm(a_only, 8_000)
    pts_bonly = _voxel_centers_mm(b_only, 8_000)

    for k, (a, b, xl, yl) in enumerate(proj_pairs):
        idx = 6 + k
        if pts_inter.shape[0]:
            ax[idx].scatter(pts_inter[:, a], pts_inter[:, b], s=3, alpha=0.35,
                            color="green", label="intersection")
        if pts_aonly.shape[0]:
            ax[idx].scatter(pts_aonly[:, a], pts_aonly[:, b], s=2, alpha=0.25,
                            color="tab:blue", label="A only")
        if pts_bonly.shape[0]:
            ax[idx].scatter(pts_bonly[:, a], pts_bonly[:, b], s=2, alpha=0.25,
                            color="tab:orange", label="B only")
        ax[idx].set_title(
            f"Step 3: Voxel overlap ({voxel_size_mm} mm bins) ({xl},{yl})\n"
            f"IoU={iou:.3f}  Dice={dice:.3f}"
        )
        ax[idx].set_xlabel(xl)
        ax[idx].set_ylabel(yl)
        ax[idx].grid(alpha=0.2)
        if k == 0:
            ax[idx].legend(markerscale=4, fontsize=6, loc="upper right")

    plt.suptitle(
        "Volume IoU / Dice Pipeline Debug\n"
        f"Source → Target: {source_id} → {target_id}\n"
        f"IoU={iou:.3f}  Dice={dice:.3f}  (ICP from surface, overlap on voxelized skull volume)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ── C: Pairwise distance matrix ───────────────────────────────────────────────

def print_pitch_deck_per_scan_structural(
    patients: list[str],
    all_scalars: pd.DataFrame,
) -> None:
    """
    Cohort-level pitch deck: mean ± std for each structural scalar across scans.
    """
    cols_want = [
        ("icv_cm3", "ICV_cm3"),
        ("skull_cm3", "skull_cm3"),
        ("skull_brain_ratio", "skull/ICV"),
        ("skull_porosity", "skull_porosity"),
        ("skull_hu_mean", "skull_HU_mean"),
        ("body_vol_cm3", "head_mask_cm3"),
        ("sphericity", "sphericity"),
        ("elongation", "elongation"),
        ("skull_thick_mean", "skull_thick_mm_mean"),
        ("skull_thick_p95", "skull_thick_mm_p95"),
        ("skull_roughness_deg_mean", "skull_roughness_deg"),
        ("lr_correlation", "LR_density_symmetry_r"),
        ("lr_asymmetry_index", "LR_asymmetry_idx"),
        ("csf_frac_icv", "CSF_frac_of_ICV"),
        ("gm_wm_hu_diff", "GM_WM_HU_diff"),
    ]
    tracked: dict[str, list[float]] = {}

    for pid in patients:
        if pid not in all_scalars.index:
            continue
        row = all_scalars.loc[pid]
        for col, label in cols_want:
            if col not in all_scalars.columns:
                continue
            v = float(row[col])
            if np.isfinite(v):
                tracked.setdefault(label, []).append(v)

    n_scans = len([p for p in patients if p in all_scalars.index])
    print("\n" + "=" * 72)
    print(
        "PITCH DECK — cohort summary (structural: mean ± std across scans; "
        f"N_scans={n_scans})"
    )
    print("=" * 72)
    for label in sorted(tracked.keys()):
        a = np.asarray(tracked[label], dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        mu = float(a.mean())
        sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
        print(
            f"  {label}:  mean={mu:.6g}  std={sd:.6g}  "
            f"(N={a.size}, min={a.min():.6g}, max={a.max():.6g})"
        )
    print("=" * 72)


def print_pairwise_matrix_cohort_stats(df: pd.DataFrame, title: str) -> None:
    """
    Mean, std, min, max over **upper triangle** (i<j) finite entries — one value per
    unordered patient pair; directed A→B metrics use matrix row/col order.
    """
    v = np.asarray(df.values, dtype=float)
    n = v.shape[0]
    if n < 2:
        print(f"  {title}:  (need ≥2 patients)")
        return
    iu = np.triu_indices(n, k=1)
    x = v[iu]
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"  {title}:  (no finite upper-triangle values)")
        return
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
    print(
        f"  {title}:  mean={mu:.6g}  std={sd:.6g}  "
        f"min={x.min():.6g}  max={x.max():.6g}  (N_pairs={x.size})"
    )


def pairwise_structural_distance(
    feat_df: pd.DataFrame,
    patients: list[str],
) -> pd.DataFrame:
    """
    N×N Euclidean distance in z-scored structural feature space.
    Lower = more similar anatomy.
    """
    avail = [p for p in patients if p in feat_df.index]
    X     = StandardScaler().fit_transform(feat_df.loc[avail].fillna(0.0).values)
    dists = squareform(pdist(X, metric="euclidean"))
    return pd.DataFrame(dists, index=avail, columns=avail)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "YlGn",
    fmt: str = ".2f",
) -> None:
    n      = len(df)
    fig_w  = max(10, n * 0.38)
    fig_h  = max(9,  n * 0.36)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    labels  = [short(p) for p in df.index]
    do_annot = n <= 30
    sns.heatmap(
        df.values.astype(float), ax=ax,
        xticklabels=labels, yticklabels=labels,
        vmin=vmin, vmax=vmax, cmap=cmap,
        annot=do_annot, fmt=fmt if do_annot else "",
        annot_kws={"size": max(4, 8 - n // 10)},
        linewidths=0.3 if n <= 30 else 0, linecolor="lightgray",
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def _bar_with_median(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    colors: list[str],
    ylabel: str,
    title: str,
    ylim: tuple | None = None,
) -> None:
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5, width=0.75)
    med = float(np.nanmedian(values))
    ax.axhline(med, color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {med:.3f}")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


# ── Individual plot functions ──────────────────────────────────────────────────

def plot_icv_skull_volumes(
    vol_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail  = [p for p in patients if p in vol_df.index]
    df     = vol_df.loc[avail]
    labels = [short(p) for p in avail]
    colors = [patient_group_color(p) for p in avail]
    x      = np.arange(len(avail))

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(avail) * 0.42), 11))

    ax = axes[0]
    ax.bar(x, df["icv_cm3"].values + df["skull_cm3"].values,
           color="lightsteelblue", label="Skull",   edgecolor="none", width=0.75)
    ax.bar(x, df["icv_cm3"].values,
           color="steelblue",      label="ICV",     edgecolor="none", width=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Volume (cm³)")
    ax.set_title("Intracranial Volume (ICV) and Skull Volume per Patient",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    _bar_with_median(ax, x, df["skull_brain_ratio"].values, colors,
                     ylabel="Skull Volume / ICV",
                     title="Skull-to-Brain Volume Ratio per Patient")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)

    plt.suptitle("Volume-Based Structural Summary", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_skull_porosity(
    vol_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail  = [p for p in patients if p in vol_df.index]
    df     = vol_df.loc[avail]
    labels = [short(p) for p in avail]
    colors = [patient_group_color(p) for p in avail]
    x      = np.arange(len(avail))

    fig, ax = plt.subplots(figsize=(max(12, len(avail) * 0.42), 5))
    _bar_with_median(ax, x, df["skull_porosity"].values, colors,
                     ylabel="Trabecular / (Trabecular + Cortical) voxels",
                     title="Skull Porosity Index per Patient\n"
                           "(higher = more trabecular bone → more porous skull)",
                     ylim=(0.0, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_skull_hu_statistics(
    vol_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail  = [p for p in patients if p in vol_df.index]
    df     = vol_df.loc[avail]
    labels = [short(p) for p in avail]
    x      = np.arange(len(avail))

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(avail) * 0.42), 11))

    ax = axes[0]
    ax.bar(x, df["skull_hu_mean"].values,
           yerr=df["skull_hu_std"].values, capsize=3,
           color="peru", alpha=0.85, edgecolor="white", linewidth=0.5,
           error_kw=dict(ecolor="black", elinewidth=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Weighted Mean HU")
    ax.set_title("Skull HU — Weighted Mean ± Std per Patient\n"
                 "(weighted by cortical + trabecular voxel counts)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    _bar_with_median(ax, x, df["skull_hu_cv"].values,
                     ["saddlebrown"] * len(avail),
                     ylabel="HU Coefficient of Variation (σ / |μ|)",
                     title="Skull HU Inhomogeneity (CV) per Patient\n"
                           "(higher = more heterogeneous skull)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)

    plt.suptitle("Skull Hounsfield-Unit Statistics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_gm_wm_diff(
    vol_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail  = [p for p in patients if p in vol_df.index]
    df     = vol_df.loc[avail].dropna(subset=["gm_wm_hu_diff"])
    if df.empty:
        print(f"    Skipped: {out_path.name} (no GM/WM HU data)")
        return
    pids   = list(df.index)
    labels = [short(p) for p in pids]
    colors = [patient_group_color(p) for p in pids]
    x      = np.arange(len(pids))

    fig, ax = plt.subplots(figsize=(max(12, len(pids) * 0.42), 5))
    ax.bar(x, df["gm_wm_hu_diff"].values, color=colors,
           edgecolor="white", linewidth=0.5, width=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("HU(GM) − HU(WM)")
    ax.set_title("Grey Matter – White Matter HU Difference per Patient\n"
                 "(proxy for tissue contrast; typically positive ~6–10 HU)",
                 fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=_LEGEND_PATCHES, fontsize=7, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_csf_fraction(
    vol_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail  = [p for p in patients if p in vol_df.index]
    df     = vol_df.loc[avail].dropna(subset=["csf_frac_icv"])
    if df.empty:
        print(f"    Skipped: {out_path.name} (no CSF data)")
        return
    pids   = list(df.index)
    labels = [short(p) for p in pids]
    colors = [patient_group_color(p) for p in pids]
    x      = np.arange(len(pids))

    fig, ax = plt.subplots(figsize=(max(12, len(pids) * 0.42), 5))
    _bar_with_median(ax, x, df["csf_frac_icv"].values, colors,
                     ylabel="CSF voxels / ICV voxels",
                     title="CSF Fraction within Intracranial Volume per Patient\n"
                           "(elevated → more ventricular space / cerebral atrophy)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_shape_descriptors(
    shape_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in shape_df.index]
    if not avail:
        print(f"    Skipped: {out_path.name} (no shape data)")
        return
    df     = shape_df.loc[avail]
    labels = [short(p) for p in avail]
    colors = [patient_group_color(p) for p in avail]
    x      = np.arange(len(avail))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.scatter(df["sphericity"], df["elongation"],
               c=colors, s=60, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=5)
    for lbl, sp_v, el_v in zip(labels, df["sphericity"], df["elongation"]):
        ax.annotate(lbl, (sp_v, el_v), fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Sphericity  (1 = perfect sphere)")
    ax.set_ylabel("Elongation  (PCA λ₁/λ₃ — 1 = isotropic)")
    ax.set_title("Sphericity vs Elongation")
    ax.grid(alpha=0.3)
    ax.legend(handles=_LEGEND_PATCHES, fontsize=7, loc="lower right")

    ax = axes[1]
    ax.bar(x, df["sphericity"].values, color=colors,
           edgecolor="white", linewidth=0.5, width=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Sphericity")
    ax.set_title("Head Sphericity per Patient\n(closer to 1 = more sphere-like skull)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(x, df["body_vol_cm3"].values, color=colors,
           edgecolor="white", linewidth=0.5, width=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Body-mask volume (cm³)")
    ax.set_title("Total Head Volume (body mask) per Patient")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("3-D Shape Descriptors from Body Mask",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_axial_area_profiles(
    profiles: dict[str, np.ndarray],
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in profiles]
    if not avail:
        print(f"    Skipped: {out_path.name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel — each patient as a normalised curve
    ax = axes[0]
    cmap_fn = plt.cm.tab20
    for i, pid in enumerate(avail):
        prof  = profiles[pid].astype(float)
        z_n   = np.linspace(0.0, 1.0, len(prof))
        a_n   = (prof - prof.min()) / (prof.max() - prof.min() + 1e-6)
        ax.plot(z_n, a_n, alpha=0.6, linewidth=0.9,
                color=cmap_fn(i % 20), label=short(pid))
    ax.set_xlabel("Normalised z  (0 = inferior, 1 = superior)")
    ax.set_ylabel("Normalised cross-section area")
    ax.set_title("Axial Area Profiles — All Patients\n(normalised per patient)")
    ax.grid(alpha=0.3)
    if len(avail) <= 20:
        ax.legend(fontsize=5, ncol=2, loc="lower center")

    # Right panel — mean ± 1 SD envelope
    ax = axes[1]
    n_pts = 100
    resampled = []
    for pid in avail:
        prof  = profiles[pid].astype(float)
        z_old = np.linspace(0.0, 1.0, len(prof))
        z_new = np.linspace(0.0, 1.0, n_pts)
        resampled.append(np.interp(z_new, z_old, prof))
    mat  = np.array(resampled)
    mu   = mat.mean(axis=0)
    sd   = mat.std(axis=0)
    z_n  = np.linspace(0.0, 1.0, n_pts)
    ax.fill_between(z_n, mu - sd, mu + sd, alpha=0.3,
                    color="steelblue", label="Mean ± 1 SD")
    ax.plot(z_n, mu, color="steelblue", linewidth=2, label="Mean")
    ax.set_xlabel("Normalised z  (0 = inferior, 1 = superior)")
    ax.set_ylabel("Cross-section area (mm²)")
    ax.set_title(f"Mean ± 1 SD Axial Area Profile\n(N = {len(avail)} patients)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle("Axial Cross-Section Area Profiles",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_lr_symmetry(
    sym_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in sym_df.index]
    if not avail:
        print(f"    Skipped: {out_path.name}")
        return
    df     = sym_df.loc[avail]
    labels = [short(p) for p in avail]
    colors = [patient_group_color(p) for p in avail]
    x      = np.arange(len(avail))

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(avail) * 0.42), 5))

    ax = axes[0]
    ax.bar(x, df["lr_correlation"].values, color=colors,
           edgecolor="white", linewidth=0.5, width=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r (sorted density distributions)")
    ax.set_title("L–R Density Symmetry Correlation\n(1 = perfectly symmetric)",
                 fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    _bar_with_median(ax, x, df["lr_asymmetry_index"].values, colors,
                     ylabel="|mean_L − mean_R| / (mean_L + mean_R)",
                     title="L–R Density Asymmetry Index\n(0 = symmetric; higher = asymmetric)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)

    plt.suptitle("Left–Right Structural Symmetry",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_skull_thickness(
    thick_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in thick_df.index]
    if not avail:
        print(f"    Skipped: {out_path.name}")
        return
    df     = thick_df.loc[avail]
    labels = [short(p) for p in avail]
    colors = [patient_group_color(p) for p in avail]
    x      = np.arange(len(avail))

    means = df["skull_thick_mean"].values
    p25   = df["skull_thick_p25"].values
    p75   = df["skull_thick_p75"].values
    p95   = df["skull_thick_p95"].values

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(avail) * 0.42), 5))

    ax = axes[0]
    ax.errorbar(x, means, yerr=[means - p25, p75 - means],
                fmt="none", ecolor="gray", elinewidth=1.5, capsize=4, zorder=3)
    ax.scatter(x, means, c=colors, s=60, zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Skull thickness (mm)")
    ax.set_title("Skull Thickness — Mean with P25–P75 range",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(x, p95, marker="^", c=colors, s=55, zorder=5, label="P95")
    ax.scatter(x, means, c=colors, s=55, zorder=5,
               edgecolors="white", linewidths=0.5, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Skull thickness (mm)")
    ax.set_title("Skull Thickness — Mean and P95",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle("Skull Wall Thickness (Euclidean Distance Transform Estimate)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_skull_roughness(
    rough_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in rough_df.index]
    if not avail:
        print(f"    Skipped: {out_path.name}")
        return
    df     = rough_df.loc[avail]
    labels = [short(p) for p in avail]
    colors = [patient_group_color(p) for p in avail]
    x      = np.arange(len(avail))

    fig, ax = plt.subplots(figsize=(max(12, len(avail) * 0.42), 5))
    ax.bar(x, df["skull_roughness_deg_mean"].values,
           yerr=df["skull_roughness_deg_std"].values,
           capsize=3, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.5,
           error_kw=dict(ecolor="black", elinewidth=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Angular deviation from mean normal (degrees)")
    ax.set_title("Skull Surface Roughness per Patient\n"
                 "(std of outer-surface normal directions — higher = more irregular skull)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=_LEGEND_PATCHES, fontsize=7, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_pairwise_structural_distance(
    dist_df: pd.DataFrame,
    out_path: Path,
) -> None:
    off_diag = dist_df.values[~np.eye(len(dist_df), dtype=bool)]
    vmax = float(np.nanpercentile(off_diag[np.isfinite(off_diag)], 90))
    _heatmap(
        dist_df,
        "Pairwise Structural Feature Distance\n"
        "(Euclidean in z-scored feature space — lower = more similar anatomy)",
        out_path,
        vmin=0.0, vmax=max(vmax, 1.0), cmap="YlOrRd", fmt=".2f",
    )


def plot_hierarchical_clustering(
    feat_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in feat_df.index]
    if len(avail) < 4:
        print(f"    Skipped: {out_path.name} (fewer than 4 patients)")
        return
    X      = StandardScaler().fit_transform(feat_df.loc[avail].fillna(0.0).values)
    labels = [short(p) for p in avail]
    colors = [patient_group_color(p) for p in avail]
    Z      = linkage(X, method="ward")

    fig, ax = plt.subplots(figsize=(max(14, len(avail) * 0.42), 7))
    dendrogram(
        Z, ax=ax, labels=labels,
        leaf_rotation=55, leaf_font_size=7,
        color_threshold=0.7 * float(Z[:, 2].max()),
    )
    # Colour leaf labels by patient group
    lbl_map = {l.get_text(): c for l, c in zip(ax.get_xticklabels(), colors)}
    for lbl_obj in ax.get_xticklabels():
        lbl_obj.set_color(lbl_map.get(lbl_obj.get_text(), "black"))

    ax.set_title(
        "Hierarchical Clustering (Ward linkage) of Structural Features",
        fontweight="bold",
    )
    ax.set_ylabel("Linkage Distance")
    ax.legend(handles=_LEGEND_PATCHES, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_radar_chart(
    feat_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    """
    Spider / radar chart of normalised structural features, one polygon per
    patient group (mean of all patients in that group).
    Features shown: ICV, skull volume, skull-brain ratio, porosity, skull HU mean, skull HU CV.
    """
    avail = [p for p in patients if p in feat_df.index]
    if len(avail) < 3:
        print(f"    Skipped: {out_path.name}")
        return

    radar_feats = [
        "icv_cm3", "skull_cm3", "skull_brain_ratio",
        "skull_porosity", "skull_hu_mean", "skull_hu_cv",
    ]
    present = [f for f in radar_feats if f in feat_df.columns]
    if len(present) < 3:
        print(f"    Skipped: {out_path.name} (too few features)")
        return

    df   = feat_df.loc[avail, present].fillna(0.0)
    df_n = (df - df.min()) / (df.max() - df.min() + 1e-9)

    # Group patients
    group_map = {
        "ACRIN": "steelblue",  "TCGA":  "tomato",
        "C3L":   "forestgreen","C3N":   "darkorchid",
        "CQ500": "darkorange", "Other": "gray",
    }
    groups: dict[str, list[str]] = {}
    for pid in avail:
        key = next((k for k in group_map if k in pid), "Other")
        groups.setdefault(key, []).append(pid)

    group_means = {
        g: df_n.loc[pids].mean()
        for g, pids in groups.items() if pids
    }
    if not group_means:
        return

    feat_labels = [
        f.replace("skull_hu_mean", "sk_HU_mean")
         .replace("skull_hu_cv",   "sk_HU_cv")
         .replace("skull_brain_ratio", "sk/brain_ratio")
         .replace("skull_porosity", "sk_porosity")
         .replace("skull_cm3",  "skull_vol")
         .replace("icv_cm3",    "ICV")
        for f in present
    ]
    N      = len(present)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for g, vals in group_means.items():
        v = vals.tolist() + vals.tolist()[:1]
        ax.plot(angles, v, linewidth=2, label=g, color=group_map[g])
        ax.fill(angles, v, alpha=0.10, color=group_map[g])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_title(
        "Structural Feature Radar — Group Means\n(features normalised 0–1 across patients)",
        fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_skull_thickness_vs_icv(
    vol_df: pd.DataFrame,
    thick_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in vol_df.index and p in thick_df.index]
    if not avail:
        print(f"    Skipped: {out_path.name}")
        return
    icv    = vol_df.loc[avail, "icv_cm3"].values
    thick  = thick_df.loc[avail, "skull_thick_mean"].values
    colors = [patient_group_color(p) for p in avail]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(icv, thick, c=colors, s=65, alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=5)
    for pid, xv, yv in zip(avail, icv, thick):
        ax.annotate(short(pid), (xv, yv), fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Intracranial Volume (cm³)")
    ax.set_ylabel("Mean Skull Thickness (mm)")
    ax.set_title("Skull Thickness vs Intracranial Volume\n"
                 "(larger heads do not necessarily have thicker skulls)",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(handles=_LEGEND_PATCHES, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_porosity_vs_skull_hu(
    vol_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in vol_df.index]
    df    = vol_df.loc[avail].dropna(subset=["skull_porosity", "skull_hu_mean"])
    if df.empty:
        print(f"    Skipped: {out_path.name}")
        return
    colors = [patient_group_color(p) for p in df.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["skull_porosity"], df["skull_hu_mean"],
               c=colors, s=65, alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=5)
    for pid, xv, yv in zip(df.index, df["skull_porosity"], df["skull_hu_mean"]):
        ax.annotate(short(pid), (xv, yv), fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Skull Porosity  (trabecular / total skull)")
    ax.set_ylabel("Skull Weighted Mean HU")
    ax.set_title("Skull Porosity vs Skull HU\n"
                 "(expected negative correlation: more trabecular → lower mean HU)",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(handles=_LEGEND_PATCHES, fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ── D: Shape comparison plots ─────────────────────────────────────────────────

def plot_shape_similarity_matrix(
    dice_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Heatmap of mean Dice similarity across three z-fraction levels.
    Higher = more similar skull cross-section shapes.
    """
    _heatmap(
        dice_df,
        "Multi-Level Shape Dice Similarity (z = 25 / 50 / 75 %)\n"
        "Higher = more similar 2-D cross-section shapes across axial levels",
        out_path,
        vmin=0.0, vmax=1.0, cmap="YlGn", fmt=".2f",
    )


def plot_hausdorff_heatmap(
    h_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Heatmap of pairwise symmetric Hausdorff distance at mid-axial level.
    Units: pixels of the resized 64×64 slice (lower = more similar surface shape).
    """
    off_diag = h_df.values[~np.eye(len(h_df), dtype=bool)]
    vmax = float(np.nanpercentile(off_diag[np.isfinite(off_diag)], 90)) if off_diag.size else 20.0
    _heatmap(
        h_df,
        "Pairwise Hausdorff Distance — Mid-Axial Shape (pixels, centroid-aligned)\n"
        "Lower = more similar skull surface shape at z = 50 %",
        out_path,
        vmin=0.0, vmax=max(vmax, 1.0), cmap="YlOrRd", fmt=".1f",
    )


def plot_shape_pca(
    coords: np.ndarray,
    pids_avail: list[str],
    pca: PCA | None,
    out_path: Path,
) -> None:
    """
    Scatter of patients in 3-D shape PCA space (PC1 vs PC2, with optional
    PC3 colour-coded).  Each point represents the entire 3-D head shape.
    Patients that cluster together have globally similar skull geometry.
    """
    if coords.shape[0] < 3 or pca is None:
        print(f"    Skipped: {out_path.name} (too few patients for PCA)")
        return

    colors = [patient_group_color(p) for p in pids_avail]
    ev     = pca.explained_variance_ratio_ * 100

    n_comp = coords.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=65, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=5,
    )
    for pid, xy in zip(pids_avail, coords[:, :2]):
        ax.annotate(short(pid), xy, fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1  ({ev[0]:.1f} % variance)")
    ax.set_ylabel(f"PC2  ({ev[1]:.1f} % variance)")
    ax.set_title("3-D Shape PCA — PC1 vs PC2\n"
                 "(each point = one patient's full 3-D body-mask shape)")
    ax.grid(alpha=0.3)
    ax.legend(handles=_LEGEND_PATCHES, fontsize=7, loc="best")

    ax = axes[1]
    cum_ev = np.cumsum(ev)
    ax.bar(range(1, n_comp + 1), ev, color="steelblue",
           edgecolor="navy", alpha=0.85)
    ax.step(range(1, n_comp + 1), cum_ev, color="red",
            where="post", linewidth=2, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Shape PCA — Explained Variance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("3-D Body-Mask Shape PCA\n"
                 "(32³ resampled body mask, all orientations)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_contour_gallery(
    multilevel_slices: dict[str, dict[float, np.ndarray]],
    patients: list[str],
    levels: tuple[float, ...],
    out_path: Path,
) -> None:
    """
    One panel per z-level.  Each panel shows ALL patients' skull contours
    (boundary pixels) overlaid after centroid centering — a 'shape cloud'.
    Colour per patient; overlapping contours reveal inter-patient variation.

    A second row shows a per-patient gallery of their individual slices at
    the mid-axial level only, for easy per-patient inspection.
    """
    avail = [p for p in patients if p in multilevel_slices]
    if not avail:
        print(f"    Skipped: {out_path.name}")
        return

    n_levels = len(levels)
    fig = plt.figure(figsize=(5 * n_levels, 5))

    cmap_fn = plt.cm.tab20
    axes_row = [fig.add_subplot(1, n_levels, k + 1) for k in range(n_levels)]

    for ax, lev in zip(axes_row, levels):
        for i, pid in enumerate(avail):
            sl = multilevel_slices[pid].get(lev)
            if sl is None:
                continue
            pts = _boundary_points(sl)
            if pts is None:
                continue
            color = cmap_fn(i % 20)
            ax.scatter(pts[:, 1], pts[:, 0], s=1, color=color,
                       alpha=0.55, linewidths=0)

        sz = _SHAPE_SLICE_SIZE
        ax.set_xlim(-sz // 2, sz // 2)
        ax.set_ylim(-sz // 2, sz // 2)
        ax.set_aspect("equal")
        ax.set_title(_LEVEL_LABELS.get(lev, f"z={lev:.0%}"), fontsize=9)
        ax.set_xlabel("← left  /  right →", fontsize=7)
        ax.set_ylabel("← ant  /  post →", fontsize=7)
        ax.grid(alpha=0.2)

    plt.suptitle(
        "Cross-Section Shape Cloud — All Patients Overlaid (centroid-centred)\n"
        "Spread of points = inter-patient shape variability at that axial level",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_aspect_ratio_profiles(
    aspect_ratios: dict[str, np.ndarray],
    patients: list[str],
    out_path: Path,
) -> None:
    """
    Two panels:
      Left  — normalised z vs aspect ratio (λ₁/λ₂) for every patient,
              colour-coded by group.  Shows where the head is most elongated.
      Right — mean ± 1 SD envelope across patients + per-patient scatter
              at z = 25 / 50 / 75 % shown as box-and-whisker.
    Ratio = 1 → circular; >1 → elliptical (more elongated) cross-section.
    """
    avail = [p for p in patients if p in aspect_ratios]
    if not avail:
        print(f"    Skipped: {out_path.name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap_fn = plt.cm.tab20

    # Left: individual patient profiles
    ax = axes[0]
    n_pts = 100
    resampled = []
    for i, pid in enumerate(avail):
        vals = aspect_ratios[pid].astype(float)
        finite = np.isfinite(vals)
        if finite.sum() < 5:
            continue
        z_old = np.linspace(0.0, 1.0, len(vals))
        z_new = np.linspace(0.0, 1.0, n_pts)
        # Interpolate only over finite regions
        interped = np.interp(z_new, z_old[finite], vals[finite])
        resampled.append(interped)
        ax.plot(z_new, interped, alpha=0.55, linewidth=0.9,
                color=cmap_fn(i % 20), label=short(pid))

    ax.set_xlabel("Normalised z  (0 = inferior, 1 = superior)")
    ax.set_ylabel("Aspect ratio  λ₁/λ₂  (1 = circular)")
    ax.set_title("Cross-Section Aspect Ratio Profiles\n(all patients)")
    ax.grid(alpha=0.3)
    if len(avail) <= 20:
        ax.legend(fontsize=5, ncol=2, loc="lower center")

    # Right: mean ± SD envelope
    ax = axes[1]
    if resampled:
        mat = np.array(resampled)
        mu  = mat.mean(axis=0)
        sd  = mat.std(axis=0)
        z_new = np.linspace(0.0, 1.0, n_pts)
        ax.fill_between(z_new, mu - sd, mu + sd, alpha=0.3,
                        color="steelblue", label="Mean ± 1 SD")
        ax.plot(z_new, mu, color="steelblue", linewidth=2, label="Mean")

    # Box-and-whisker at the three standard levels
    level_keys = [0.25, 0.50, 0.75]
    level_vals = []
    for lev in level_keys:
        vals_at = []
        for pid in avail:
            arr = aspect_ratios[pid]
            nz  = len(arr)
            zi  = max(0, min(nz - 1, int(round(lev * (nz - 1)))))
            v   = arr[zi]
            if np.isfinite(v):
                vals_at.append(v)
        level_vals.append(vals_at)

    bp = ax.boxplot(
        level_vals,
        positions=[0.25, 0.50, 0.75],
        widths=0.06,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightskyblue", color="navy"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="navy"),
        capprops=dict(color="navy"),
    )
    ax.set_xticks([0.25, 0.50, 0.75])
    ax.set_xticklabels(["z=25%\n(inferior)", "z=50%\n(mid)", "z=75%\n(superior)"])
    ax.set_xlabel("z-fraction level")
    ax.set_ylabel("Aspect ratio  λ₁/λ₂")
    ax.set_title(f"Mean ± 1 SD + Level Box-plots\n(N = {len(resampled)} patients)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Cross-Section Aspect Ratio (Ellipse Major/Minor Axis Ratio)\n"
                 "from Body-Mask PCA — 1 = circular, higher = more elongated skull shape",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 70
    print(sep)
    print("CT Scan Structural Comparison Analysis")
    print(sep)

    # ── 1. Discover patients ──────────────────────────────────────────────────
    print("\n[1/5] Discovering patients …")
    patients = discover_patients()
    if not patients:
        raise SystemExit(
            "No patients found.\n"
            "Set CT_ACOUSTIC_DIR and CT_PROC_DIR env vars, or update the "
            "default paths at the top of this script."
        )

    # ── 2. Load tissue stats + metadata ──────────────────────────────────────
    print("\n[2/5] Loading tissue statistics and metadata …")
    stats_df = load_tissue_stats(patients)
    meta_df  = load_metadata(patients)

    # ── 3. Volume-based features ──────────────────────────────────────────────
    print("\n[3/5] Computing volume-based structural features …")
    vol_df = compute_volume_features(stats_df, meta_df, patients)
    print(f"  Features computed for {len(vol_df)} patients.")

    # ── 4. Geometry-based features ────────────────────────────────────────────
    print("\n[4/5] Computing geometry-based features from density arrays …")

    print("  3-D shape descriptors …")
    shape_df = compute_shape_descriptors(patients, meta_df)
    print(f"  Done: {len(shape_df)} / {len(patients)}")

    print("  Axial area profiles …")
    axial_profiles = compute_axial_area_profiles(patients, meta_df)
    print(f"  Done: {len(axial_profiles)} / {len(patients)}")

    print("  L–R symmetry …")
    sym_df = compute_lr_symmetry(patients)
    print(f"  Done: {len(sym_df)} / {len(patients)}")

    print("  Skull thickness (EDT) …")
    thick_df = compute_skull_thickness(patients, meta_df)
    print(f"  Done: {len(thick_df)} / {len(patients)}")

    print("  Skull surface roughness …")
    rough_df = compute_skull_roughness(patients, meta_df)
    print(f"  Done: {len(rough_df)} / {len(patients)}")

    print("  Multi-level 2-D slices …")
    multilevel_slices = extract_multilevel_slices(patients)
    print(f"  Done: {len(multilevel_slices)} / {len(patients)}")

    print("  Cross-section aspect ratios …")
    aspect_ratios = compute_cross_section_aspect_ratios(patients)
    print(f"  Done: {len(aspect_ratios)} / {len(patients)}")

    print("  3-D shape PCA …")
    shape_pca_coords, shape_pca_obj, shape_pca_pids = compute_shape_pca(patients)
    print(f"  Done: {len(shape_pca_pids)} / {len(patients)} patients in PCA")

    # Merge all structural scalars; drop raw voxel-count columns
    all_scalars = (
        vol_df
        .join(shape_df,  how="outer", rsuffix="_shp")
        .join(sym_df,    how="outer", rsuffix="_sym")
        .join(thick_df,  how="outer", rsuffix="_thk")
        .join(rough_df,  how="outer", rsuffix="_rgh")
    )
    all_scalars.to_csv(OUT_DIR / "structural_features.csv")
    print("\n  Saved: structural_features.csv")

    print_pitch_deck_per_scan_structural(patients, all_scalars)

    # ── 5. Generate plots ─────────────────────────────────────────────────────
    print("\n[5/5] Generating plots …")
    print("\n  A — Volume-based:")

    plot_icv_skull_volumes(vol_df, patients,
                           OUT_DIR / "01_icv_skull_volumes.png")
    plot_skull_porosity(vol_df, patients,
                        OUT_DIR / "02_skull_porosity.png")
    plot_skull_hu_statistics(vol_df, patients,
                             OUT_DIR / "03_skull_hu_statistics.png")
    plot_gm_wm_diff(vol_df, patients,
                    OUT_DIR / "04_gm_wm_hu_diff.png")
    plot_csf_fraction(vol_df, patients,
                      OUT_DIR / "05_csf_fraction_icv.png")

    print("\n  B — Geometry-based:")

    plot_shape_descriptors(shape_df, patients,
                           OUT_DIR / "06_shape_descriptors.png")
    plot_axial_area_profiles(axial_profiles, patients,
                             OUT_DIR / "07_axial_area_profiles.png")
    plot_lr_symmetry(sym_df, patients,
                     OUT_DIR / "08_lr_symmetry.png")
    plot_skull_thickness(thick_df, patients,
                         OUT_DIR / "09_skull_thickness.png")
    plot_skull_roughness(rough_df, patients,
                         OUT_DIR / "10_skull_roughness.png")

    print("\n  C — Pairwise & multivariate:")

    # Distance matrix over volume features (always available)
    numeric_cols = [
        c for c in all_scalars.columns
        if all_scalars[c].dtype in (np.float64, np.float32, float)
        and "vox" not in c
    ]
    if numeric_cols and len(all_scalars) >= 2:
        dist_df = pairwise_structural_distance(all_scalars[numeric_cols], patients)
        plot_pairwise_structural_distance(
            dist_df, OUT_DIR / "11_structural_distance_matrix.png"
        )
    else:
        print("    Skipped: 11_structural_distance_matrix.png")

    vol_feat_cols = [
        c for c in vol_df.columns
        if vol_df[c].dtype in (np.float64, float)
    ]
    plot_hierarchical_clustering(
        vol_df[vol_feat_cols] if vol_feat_cols else vol_df,
        patients,
        OUT_DIR / "12_hierarchical_clustering.png",
    )

    plot_radar_chart(vol_df, patients,
                     OUT_DIR / "13_structural_radar_chart.png")

    plot_skull_thickness_vs_icv(vol_df, thick_df, patients,
                                OUT_DIR / "14_skull_thickness_vs_icv.png")

    plot_porosity_vs_skull_hu(vol_df, patients,
                              OUT_DIR / "15_porosity_vs_skull_hu.png")

    print("\n  D — Shape comparisons:")

    print("  Multi-level Dice matrix …")
    dice_df = pairwise_multilevel_dice(multilevel_slices, patients)
    plot_shape_similarity_matrix(dice_df,
                                 OUT_DIR / "16_shape_dice_similarity.png")

    print("  Hausdorff distance matrix …")
    hausdorff_df = pairwise_hausdorff_2d(multilevel_slices, patients, level=0.50)
    plot_hausdorff_heatmap(hausdorff_df,
                           OUT_DIR / "17_hausdorff_distance.png")

    plot_shape_pca(shape_pca_coords, shape_pca_pids, shape_pca_obj,
                   OUT_DIR / "18_shape_pca.png")

    plot_contour_gallery(multilevel_slices, patients, _SHAPE_LEVELS,
                         OUT_DIR / "19_contour_overlay.png")

    plot_aspect_ratio_profiles(aspect_ratios, patients,
                               OUT_DIR / "20_aspect_ratio_profiles.png")

    print("  Skull surface ICP + Chamfer …")
    surf = pairwise_skull_surface_distances(patients, meta_df)
    if "chamfer_mm" in surf and len(surf["chamfer_mm"]) >= 2:
        plot_surface_distance_heatmap(
            surf["chamfer_mm"],
            "Skull Surface Chamfer Distance (mm, ICP-aligned, A→B)\n"
            "Lower = more similar 3-D skull geometry",
            OUT_DIR / "21_skull_surface_chamfer_mm.png",
        )
        plot_surface_distance_heatmap(
            surf["p95_mm"],
            "Skull Surface P95 Distance (mm, ICP-aligned, A→B)\n"
            "Robust worst-case mismatch (95th percentile of surface distances)",
            OUT_DIR / "22_skull_surface_p95_mm.png",
        )
        chamfer_df = surf["chamfer_mm"].copy()
        vals = chamfer_df.values.copy()
        np.fill_diagonal(vals, np.nan)
        if np.isfinite(vals).any():
            bi, bj = np.unravel_index(np.nanargmin(vals), vals.shape)
            wi, wj = np.unravel_index(np.nanargmax(vals), vals.shape)
            src_best, tgt_best = chamfer_df.index[bi], chamfer_df.columns[bj]
            src_worst, tgt_worst = chamfer_df.index[wi], chamfer_df.columns[wj]
            plot_chamfer_pipeline_debug(
                src_best,
                tgt_best,
                meta_df,
                OUT_DIR / "23_chamfer_pipeline_debug_best_pair.png",
            )
            plot_chamfer_pipeline_debug(
                src_worst,
                tgt_worst,
                meta_df,
                OUT_DIR / "24_chamfer_pipeline_debug_worst_pair.png",
            )
        print("\n" + "=" * 72)
        print("PAIRWISE SKULL CHAMFER (mm, ICP A→B) — upper-triangle pair stats")
        print("=" * 72)
        print_pairwise_matrix_cohort_stats(
            surf["chamfer_mm"],
            "Chamfer distance (mm)",
        )
        print("=" * 72)
    else:
        print("    Skipped: skull surface distance heatmaps (too few valid point clouds)")

    print("  Volumetric overlap (IoU/Dice) …")
    vol_overlap = pairwise_skull_volume_overlap(patients, meta_df)
    if "volume_iou" in vol_overlap and len(vol_overlap["volume_iou"]) >= 2:
        _heatmap(
            vol_overlap["volume_iou"],
            "Skull Volume Overlap IoU (ICP-aligned, A→B)\n"
            "Higher = more volumetrically similar skulls",
            OUT_DIR / "25_skull_volume_iou.png",
            vmin=0.0,
            vmax=1.0,
            cmap="YlGn",
            fmt=".3f",
        )
        _heatmap(
            vol_overlap["volume_dice"],
            "Skull Volume Overlap Dice (ICP-aligned, A→B)\n"
            "Higher = more volumetrically similar skulls",
            OUT_DIR / "26_skull_volume_dice.png",
            vmin=0.0,
            vmax=1.0,
            cmap="YlGn",
            fmt=".3f",
        )
        iou_df = vol_overlap["volume_iou"].copy()
        vvals = iou_df.values.copy()
        np.fill_diagonal(vvals, np.nan)
        if np.isfinite(vvals).any():
            vbi, vbj = np.unravel_index(np.nanargmax(vvals), vvals.shape)
            vwi, vwj = np.unravel_index(np.nanargmin(vvals), vvals.shape)
            src_iou_best = iou_df.index[vbi]
            tgt_iou_best = iou_df.columns[vbj]
            src_iou_worst = iou_df.index[vwi]
            tgt_iou_worst = iou_df.columns[vwj]
            avail_v = list(iou_df.index)
            plot_volume_iou_pipeline_debug(
                src_iou_best,
                tgt_iou_best,
                meta_df,
                OUT_DIR / "27_volume_iou_pipeline_debug_best_pair.png",
                icp_rng_seed=avail_v.index(src_iou_best) * 4001 + avail_v.index(tgt_iou_best),
            )
            plot_volume_iou_pipeline_debug(
                src_iou_worst,
                tgt_iou_worst,
                meta_df,
                OUT_DIR / "28_volume_iou_pipeline_debug_worst_pair.png",
                icp_rng_seed=avail_v.index(src_iou_worst) * 4001 + avail_v.index(tgt_iou_worst),
            )
        print("\n" + "=" * 72)
        print(
            "PAIRWISE SKULL VOLUME IoU (ICP-aligned A→B) — upper-triangle pair stats"
        )
        print("=" * 72)
        print_pairwise_matrix_cohort_stats(
            vol_overlap["volume_iou"],
            "Volume IoU",
        )
        print("=" * 72)
    else:
        print("    Skipped: skull volume IoU/Dice heatmaps (too few valid point clouds)")

    print(f"\n{sep}")
    print("Structural comparison complete.")
    print(f"  28 figures + structural_features.csv  →  {OUT_DIR}")
    print(f"\nTo run on the CQ500 batch:")
    print(f"  export CT_ACOUSTIC_DIR='.../cq500_50_hounsfield2density/acoustic_maps'")
    print(f"  export CT_PROC_DIR='.../cq500_50_hounsfield2density/processed'")
    print(f"  export CT_STRUCTURAL_OUT='.../cq500_50_hounsfield2density/ct_structural_comparison'")
    print(f"  python3 ct_scan_structural_comparison.py")
    print(sep)


if __name__ == "__main__":
    main()
