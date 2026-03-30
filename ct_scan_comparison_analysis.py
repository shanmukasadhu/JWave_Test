#!/usr/bin/env python3
"""
CT Scan Pairwise Comparison Analysis
=====================================
Quantifies and visualises structural and acoustic differences between all CT
scans in the dataset.

Outputs (saved to Results/ct_comparison_analysis/):
  01 – Tissue composition similarity matrix   (cosine similarity)
  02 – Sound-speed distribution distance      (Jensen–Shannon divergence)
  03 – Density distribution distance          (Jensen–Shannon divergence)
  04 – Attenuation distribution distance      (Jensen–Shannon divergence)
  05 – Hounsfield-unit distribution distance  (Jensen–Shannon divergence)
  06 – Top-of-head shape similarity           (Dice coefficient)
  07 – Tissue composition stacked bar chart
  08 – Acoustic property boxplots per patient
  09 – PCA clustering of patients
  10 – Top-of-head cross-section gallery
  11 – Mean HU per tissue per patient heatmap
  12 – Structural tissue-fraction scatter plots
  13 – Per-region acoustic property heatmap
  14 – Feature inter-correlation matrix
  15 – Acoustic property mean ± std bar chart
  patient_feature_summary.csv – aggregated per-patient features

When run, prints cohort-level blocks: pitch-deck summary plus tissue_stats-derived
skull / brain soft / CSF acoustics and fractions (mean, std, min, max).

Data sources:
  acoustic_maps/<pid>/          – full-res tissue_mask, metadata
  processed_all_42_patients_ct/<pid>/ – 128³ acoustic arrays + tissue_stats.csv
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from scipy.ndimage import zoom
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Override for CQ500 / other batches, e.g. after batch_cq500_hounsfield2density.py:
#   export CT_ACOUSTIC_DIR=".../cq500_50_hounsfield2density/acoustic_maps"
#   export CT_PROC_DIR=".../cq500_50_hounsfield2density/processed"
#   export CT_COMPARISON_OUT=".../cq500_50_hounsfield2density/ct_comparison_analysis"
_BASE_DEFAULT = Path("/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results")
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
OUT_DIR = Path(os.environ.get("CT_COMPARISON_OUT", str(BASE_DIR / "ct_comparison_analysis")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TISSUE_LABELS = {
    "air": 0, "fat": 1, "csf": 2, "brain_wm": 3, "brain_gm": 4,
    "blood": 5, "soft_tissue": 6, "trabecular_skull": 7, "cortical_skull": 8,
}

# Spatial sub-sampling stride applied to full-resolution arrays.
# Stride 4 reduces a 625³ volume to ~157³ (~2.4 M voxels) — plenty for
# histograms and statistics while keeping each read < 10 MB.
SPATIAL_STRIDE = 4
SURFACE_YZ_STRIDE = 4
CONTACT_TOL_VOX = 1.0
TOP_BOWL_PERCENTILE = 70.0

# Histogram ranges (min, max, n_bins) for body-region acoustic properties
HIST_RANGES = {
    "sound_speed": (1400.0, 2800.0, 25),
    "density":     (900.0,  1900.0, 25),
    "attenuation": (0.0,    50.0,   25),
    "hu":          (-300.0, 1600.0, 25),
}

# File names in acoustic_maps/<pid>/ — sound/density/attenuation are prefixed with the patient ID
# "hu" and "body_mask" use fixed names; the rest use f"{pid}_{stem}"
ARRAY_STEMS = {
    "sound_speed": "sound_speed_cleaned",
    "density":     "density_cleaned",
    "attenuation": "attenuation_cleaned",
}
ARRAY_FIXED = {
    "hu":        "hu_xyz.npy",
    "body_mask": "{pid}_body_mask.npy",   # placeholder resolved at load time
}

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
    """Compact axis label for a patient ID."""
    return (pid
            .replace("ACRIN-FMISO-Brain-", "AF-")
            .replace("TCGA-14-", "T14-"))


def patient_group_color(pid: str) -> str:
    if "ACRIN" in pid:  return "steelblue"
    if "TCGA"  in pid:  return "tomato"
    if "C3L"   in pid:  return "forestgreen"
    if "C3N"   in pid:  return "darkorchid"
    return "gray"


def print_pitch_deck_per_scan(
    patients: list[str],
    stats_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    acoustic_data: dict[str, dict[str, np.ndarray]],
    head_shapes: dict[str, np.ndarray],
) -> None:
    """
    Cohort-level pitch deck: mean ± std across scans (one summary line per metric).
    """
    s = SPATIAL_STRIDE
    collected: dict[str, list[float]] = {}

    def _track(key: str, val: float) -> None:
        if np.isfinite(val):
            collected.setdefault(key, []).append(float(val))

    for pid in patients:
        if pid in meta_df.index:
            m = meta_df.loc[pid]
            sx = float(m.get("shape_x", np.nan))
            sy = float(m.get("shape_y", np.nan))
            sz = float(m.get("shape_z", np.nan))
            if np.isfinite(sx) and np.isfinite(sy) and np.isfinite(sz):
                nvox = int(sx * sy * sz)
                _track("nvox_millions", nvox / 1e6)
            bf = float(m.get("body_fraction", np.nan))
            if np.isfinite(bf):
                _track("body_fraction", bf)

        if pid in stats_df.index:
            r = stats_df.loc[pid]
            f_cort = float(r.get("frac_cortical_skull", 0.0) or 0.0)
            f_trab = float(r.get("frac_trabecular_skull", 0.0) or 0.0)
            f_skull = f_cort + f_trab
            if f_skull > 0:
                _track("skull_voxel_frac", f_skull)
            f_bs = float(r.get("frac_brain_soft", np.nan))
            if not np.isfinite(f_bs) or f_bs <= 0:
                f_bs = float(r.get("frac_brain_gm", 0.0) or 0.0) + float(
                    r.get("frac_brain_wm", 0.0) or 0.0
                )
            if np.isfinite(f_bs) and f_bs > 0:
                _track("brain_voxel_frac", f_bs)
            f_csf = float(r.get("frac_csf", np.nan))
            if np.isfinite(f_csf) and f_csf > 0:
                _track("csf_voxel_frac", f_csf)

        arrays = acoustic_data.get(pid, {})
        bm, hu = arrays.get("body_mask"), arrays.get("hu")
        if bm is not None:
            body = np.asarray(bm[::s, ::s, ::s]).astype(bool)
        elif hu is not None:
            body = np.asarray(hu[::s, ::s, ::s]) > -500.0
        else:
            body = None

        if body is not None and body.any():
            for key, lo, hi, tag in [
                ("sound_speed", 1400.0, 2800.0, "c_body_mean_mps"),
                ("density", 900.0, 2000.0, "rho_body_mean_kgm3"),
                ("attenuation", 0.0, 50.0, "att_body_mean_dbcmMHz"),
            ]:
                arr = arrays.get(key)
                if arr is None:
                    continue
                vals = np.asarray(arr[::s, ::s, ::s])[body].astype(float)
                vals = vals[(vals >= lo) & (vals <= hi)]
                if vals.size:
                    _track(tag, float(vals.mean()))
            if hu is not None:
                h = np.asarray(hu[::s, ::s, ::s])[body].astype(float)
                h = h[np.isfinite(h) & (h > -500)]
                if h.size:
                    _track("HU_body_mean", float(h.mean()))

        if pid in head_shapes:
            sil = float((head_shapes[pid] > 0.5).sum())
            _track("superior_outline_px", sil)

    print("\n" + "=" * 72)
    print(
        "PITCH DECK — cohort summary (mean ± std across scans; "
        f"N_scans={len(patients)})"
    )
    print("=" * 72)
    for key in sorted(collected.keys()):
        a = np.asarray(collected[key], dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        mu = float(a.mean())
        sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
        print(
            f"  {key}:  mean={mu:.6g}  std={sd:.6g}  "
            f"(N={a.size}, min={a.min():.6g}, max={a.max():.6g})"
        )
    print("=" * 72)


def _finite_float(x: object) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _weighted_skull_acoustic(row: pd.Series, prop: str) -> float:
    """prop: c_mean | rho_mean | att_mean — cortical+trabecular by count, else merged skull."""
    c_c = _finite_float(row.get("count_cortical_skull", 0))
    c_t = _finite_float(row.get("count_trabecular_skull", 0))
    if c_c + c_t > 0:
        v_c = _finite_float(row.get(f"{prop}_cortical_skull", np.nan))
        v_t = _finite_float(row.get(f"{prop}_trabecular_skull", np.nan))
        num, den = 0.0, 0.0
        if c_c > 0 and np.isfinite(v_c):
            num += v_c * c_c
            den += c_c
        if c_t > 0 and np.isfinite(v_t):
            num += v_t * c_t
            den += c_t
        return float(num / den) if den > 0 else float("nan")
    c_s = _finite_float(row.get("count_skull", 0))
    if c_s > 0:
        return _finite_float(row.get(f"{prop}_skull", np.nan))
    return float("nan")


def _tissue_acoustic(row: pd.Series, tissue: str, prop: str) -> float:
    c = _finite_float(row.get(f"count_{tissue}", 0))
    if c <= 0:
        return float("nan")
    return _finite_float(row.get(f"{prop}_{tissue}", np.nan))


def _brain_soft_acoustic(row: pd.Series, prop: str) -> float:
    v = _tissue_acoustic(row, "brain_soft", prop)
    if np.isfinite(v):
        return v
    c_gm = _finite_float(row.get("count_brain_gm", 0))
    c_wm = _finite_float(row.get("count_brain_wm", 0))
    if c_gm + c_wm <= 0:
        return float("nan")
    v_gm = _finite_float(row.get(f"{prop}_brain_gm", np.nan))
    v_wm = _finite_float(row.get(f"{prop}_brain_wm", np.nan))
    num, den = 0.0, 0.0
    if c_gm > 0 and np.isfinite(v_gm):
        num += v_gm * c_gm
        den += c_gm
    if c_wm > 0 and np.isfinite(v_wm):
        num += v_wm * c_wm
        den += c_wm
    return float(num / den) if den > 0 else float("nan")


def _frac_brain_soft(row: pd.Series) -> float:
    f = _finite_float(row.get("frac_brain_soft", np.nan))
    if np.isfinite(f) and f > 0:
        return f
    g = _finite_float(row.get("frac_brain_gm", 0))
    w = _finite_float(row.get("frac_brain_wm", 0))
    s = g + w
    return s if s > 0 else float("nan")


def _print_cohort_msmm(label: str, values: list[float]) -> None:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"  {label}:  (no finite values)")
        return
    mu = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    print(
        f"  {label}:  mean={mu:.6g}  std={sd:.6g}  "
        f"min={a.min():.6g}  max={a.max():.6g}  (N={a.size})"
    )


def print_tissue_acoustic_cohort_stats(stats_df: pd.DataFrame, patients: list[str]) -> None:
    """
    Cohort mean/std/min/max from tissue_stats.csv wide rows: skull (combined),
    brain soft, CSF, tissue fractions, trabecular & cortical skull acoustics.
    """
    skull_c, skull_rho, skull_att = [], [], []
    brain_c, brain_rho, brain_att = [], [], []
    csf_c, csf_rho, csf_att = [], [], []
    frac_bs: list[float] = []
    frac_trab: list[float] = []
    frac_cort: list[float] = []
    trab_c, trab_rho, trab_att = [], [], []
    cort_c, cort_rho, cort_att = [], [], []

    for pid in patients:
        if pid not in stats_df.index:
            continue
        row = stats_df.loc[pid]
        for lst, prop in (
            (skull_c, "c_mean"),
            (skull_rho, "rho_mean"),
            (skull_att, "att_mean"),
        ):
            lst.append(_weighted_skull_acoustic(row, prop))
        for lst, prop in (
            (brain_c, "c_mean"),
            (brain_rho, "rho_mean"),
            (brain_att, "att_mean"),
        ):
            lst.append(_brain_soft_acoustic(row, prop))
        for lst, prop in (
            (csf_c, "c_mean"),
            (csf_rho, "rho_mean"),
            (csf_att, "att_mean"),
        ):
            lst.append(_tissue_acoustic(row, "csf", prop))
        frac_bs.append(_frac_brain_soft(row))
        frac_trab.append(_finite_float(row.get("frac_trabecular_skull", np.nan)))
        frac_cort.append(_finite_float(row.get("frac_cortical_skull", np.nan)))
        for lst, prop in (
            (trab_c, "c_mean"),
            (trab_rho, "rho_mean"),
            (trab_att, "att_mean"),
        ):
            lst.append(_tissue_acoustic(row, "trabecular_skull", prop))
        for lst, prop in (
            (cort_c, "c_mean"),
            (cort_rho, "rho_mean"),
            (cort_att, "att_mean"),
        ):
            lst.append(_tissue_acoustic(row, "cortical_skull", prop))

    print("\n" + "=" * 72)
    print(
        "TISSUE / REGION COHORT STATS (from tissue_stats.csv; mean, std, min, max over scans)"
    )
    print("=" * 72)
    print("  Skull (cortical+trabecular weighted, or merged skull if no split):")
    _print_cohort_msmm("    sound_speed_mps", skull_c)
    _print_cohort_msmm("    density_kg_m3", skull_rho)
    _print_cohort_msmm("    attenuation_db_cm_MHz", skull_att)
    print("  Brain soft tissue (brain_soft row, else GM+WM weighted):")
    _print_cohort_msmm("    sound_speed_mps", brain_c)
    _print_cohort_msmm("    density_kg_m3", brain_rho)
    _print_cohort_msmm("    attenuation_db_cm_MHz", brain_att)
    print("  CSF (tissue mean c, rho, att per scan):")
    _print_cohort_msmm("    sound_speed_mps", csf_c)
    _print_cohort_msmm("    density_kg_m3", csf_rho)
    _print_cohort_msmm("    attenuation_db_cm_MHz", csf_att)
    print("  Voxel fraction brain_soft (or GM+WM sum if no brain_soft):")
    _print_cohort_msmm("    frac_brain_soft", frac_bs)
    print("  Voxel fraction trabecular_skull:")
    _print_cohort_msmm("    frac_trabecular_skull", frac_trab)
    print("  Voxel fraction cortical_skull:")
    _print_cohort_msmm("    frac_cortical_skull", frac_cort)
    print("  Trabecular skull (tissue mean c, rho, att per scan):")
    _print_cohort_msmm("    sound_speed_mps", trab_c)
    _print_cohort_msmm("    density_kg_m3", trab_rho)
    _print_cohort_msmm("    attenuation_db_cm_MHz", trab_att)
    print("  Cortical skull (tissue mean c, rho, att per scan):")
    _print_cohort_msmm("    sound_speed_mps", cort_c)
    _print_cohort_msmm("    density_kg_m3", cort_rho)
    _print_cohort_msmm("    attenuation_db_cm_MHz", cort_att)
    print("=" * 72)


# ── Data loading ──────────────────────────────────────────────────────────────

def discover_patients() -> list[str]:
    """Sorted list of patient IDs that have cleaned acoustic arrays in
    acoustic_maps AND tissue_stats.csv in processed_all_42."""
    patients = sorted([
        d.name for d in ACOUSTIC_DIR.iterdir()
        if d.is_dir()
        and (d / f"{d.name}_sound_speed_cleaned.npy").exists()
        and (d / f"{d.name}_density_cleaned.npy").exists()
        and (d / f"{d.name}_attenuation_cleaned.npy").exists()
        and (d / f"{d.name}_body_mask.npy").exists()
        and (PROC_DIR / d.name / "tissue_stats.csv").exists()
    ])
    print(f"  Found {len(patients)} patients with complete data.")
    return patients


def load_tissue_stats(patients: list[str]) -> pd.DataFrame:
    """Load tissue_stats.csv for every patient into a wide DataFrame."""
    records = []
    for pid in patients:
        csv_p = PROC_DIR / pid / "tissue_stats.csv"
        df = pd.read_csv(csv_p)
        row: dict = {"patient_id": pid}
        for _, r in df.iterrows():
            t = str(r["tissue"])
            row[f"count_{t}"]   = float(r.get("voxel_count", 0.0))
            row[f"frac_{t}"]    = float(r.get("fraction", 0.0))
            row[f"hu_mean_{t}"] = float(r.get("hu_mean",  0.0))
            row[f"hu_std_{t}"]  = float(r.get("hu_std",   0.0))
            row[f"c_mean_{t}"]  = float(r.get("c_mean",   np.nan))
            row[f"rho_mean_{t}"]= float(r.get("rho_mean", np.nan))
            row[f"att_mean_{t}"]= float(r.get("att_mean", np.nan))
        records.append(row)
    return pd.DataFrame(records).set_index("patient_id")


def load_metadata(patients: list[str]) -> pd.DataFrame:
    """Load metadata.json from acoustic_maps for each patient."""
    records = []
    for pid in patients:
        row: dict = {"patient_id": pid}
        meta_p = ACOUSTIC_DIR / pid / "metadata.json"
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            shape = m.get("shape_xyz", [np.nan, np.nan, np.nan])
            spacing = m.get("spacing_xyz_mm", [np.nan, np.nan, np.nan])
            hu_range = m.get("hu_range", None)
            if hu_range is None:
                ms = m.get("mapping_safety", {})
                hu_range = [ms.get("hu_clip_min", np.nan), ms.get("hu_clip_max", np.nan)]
            row.update({
                "shape_x": shape[0],
                "shape_y": shape[1],
                "shape_z": shape[2],
                "spacing_mm": spacing[0],
                "hu_min": hu_range[0],
                "hu_max": hu_range[1],
                "body_fraction": m.get("body_fraction", np.nan),
            })
        records.append(row)
    return pd.DataFrame(records).set_index("patient_id")


def load_acoustic_arrays(patients: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """
    Load full-resolution cleaned acoustic arrays from acoustic_maps/<pid>/ using
    memory-mapping (arrays can be 600³+ voxels — never copied into RAM).

    Keys per patient: "hu", "sound_speed", "density", "attenuation", "body_mask".
    File layout:
        hu_xyz.npy                          (fixed name)
        {pid}_sound_speed_cleaned.npy       (prefixed)
        {pid}_density_cleaned.npy           (prefixed)
        {pid}_attenuation_cleaned.npy       (prefixed)
        {pid}_body_mask.npy                 (prefixed)
    """
    data: dict = {}
    for i, pid in enumerate(patients):
        pdir = ACOUSTIC_DIR / pid
        arrays: dict = {}

        # Fixed-name files
        hu_path = pdir / "hu_xyz.npy"
        if hu_path.exists():
            arrays["hu"] = np.load(hu_path, mmap_mode="r")

        # Patient-ID-prefixed files
        for key, stem in ARRAY_STEMS.items():
            fpath = pdir / f"{pid}_{stem}.npy"
            if fpath.exists():
                arrays[key] = np.load(fpath, mmap_mode="r")

        bm_path = pdir / f"{pid}_body_mask.npy"
        if bm_path.exists():
            arrays["body_mask"] = np.load(bm_path, mmap_mode="r")

        data[pid] = arrays
        if (i + 1) % 10 == 0:
            print(f"    Loaded {i+1}/{len(patients)}")
    return data


def compute_body_histograms(
    acoustic_data: dict[str, dict[str, np.ndarray]],
    patients: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    For each patient compute normalised histograms of each acoustic property
    over the body region (HU > -500, i.e. excluding open air).
    Returns {prop: {pid: histogram_array}}.
    """
    hists: dict = {prop: {} for prop in HIST_RANGES}

    s = SPATIAL_STRIDE
    for pid in patients:
        arrays = acoustic_data.get(pid, {})

        # Sub-sample spatial grid before any masking to keep reads small
        bm = arrays.get("body_mask")
        if bm is not None:
            body_mask = np.asarray(bm[::s, ::s, ::s]).astype(bool)
        else:
            hu = arrays.get("hu")
            if hu is None:
                continue
            body_mask = np.asarray(hu[::s, ::s, ::s]) > -500.0

        for prop, (lo, hi, nbins) in HIST_RANGES.items():
            arr = arrays.get(prop)
            if arr is None:
                continue
            vals = np.asarray(arr[::s, ::s, ::s])[body_mask].astype(float)
            vals = vals[(vals >= lo) & (vals <= hi)]
            if vals.size == 0:
                hists[prop][pid] = np.zeros(nbins)
                continue
            h, _ = np.histogram(vals, bins=nbins, range=(lo, hi))
            total = h.sum()
            hists[prop][pid] = h / total if total > 0 else h.astype(float)

    return hists


def extract_head_shapes(
    acoustic_data: dict[str, dict[str, np.ndarray]],
    patients: list[str],
    target_size: int = 64,
) -> dict[str, np.ndarray]:
    """
    Extract a 2D binary top-of-head cross-section from the full-resolution data.

    Uses body_mask (preferred) or falls back to HU > -200.
    Strategy:
      1. Sum body_mask along x and y to get body-presence per z-slice
      2. Find skull-containing z range (body present)
      3. Take slices in the top 25 % of that range (superior aspect)
      4. Max-project those slices → binary head silhouette
      5. Resize to target_size × target_size

    Arrays are accessed via mmap slicing to avoid loading the full volume.
    """
    shapes: dict = {}
    for pid in patients:
        arrays = acoustic_data.get(pid, {})
        bm = arrays.get("body_mask")

        s = SPATIAL_STRIDE
        if bm is not None:
            # Sub-sample x,y to locate the z-range cheaply, then read top slices
            bm_sub_z = np.asarray(bm[::s, ::s, :])        # (nx//s, ny//s, nz)
            body_per_z = bm_sub_z.sum(axis=(0, 1))        # (nz,)
            body_z = np.where(body_per_z > 5)[0]
            if body_z.size == 0:
                continue
            z_lo, z_hi = int(body_z.min()), int(body_z.max())
            top_start = int(z_lo + 0.75 * (z_hi - z_lo))
            top_start = max(top_start, z_lo)

            # Project top slices at full x-y resolution (only reads a thin z-slab)
            proj = np.asarray(bm[:, :, top_start : z_hi + 1]).any(axis=2).astype(np.float32)
        else:
            hu = arrays.get("hu")
            if hu is None:
                continue
            hu_sub_z = np.asarray(hu[::s, ::s, :])        # (nx//s, ny//s, nz)
            skull_per_z = (hu_sub_z > 200.0).sum(axis=(0, 1))
            skull_z = np.where(skull_per_z > 3)[0]
            if skull_z.size == 0:
                continue
            z_lo, z_hi = int(skull_z.min()), int(skull_z.max())
            top_start = int(z_lo + 0.75 * (z_hi - z_lo))
            top_start = max(top_start, z_lo)
            slc = np.asarray(hu[:, :, top_start : z_hi + 1])
            proj = (slc > -200.0).any(axis=2).astype(np.float32)

        # Resize to target_size × target_size
        if proj.shape[0] != target_size:
            scale = target_size / proj.shape[0]
            proj = zoom(proj, scale, order=1)
        shapes[pid] = (proj > 0.5).astype(np.float32)

    return shapes


# ── Pairwise similarity / distance matrices ───────────────────────────────────

def pairwise_cosine_similarity(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """N×N cosine similarity matrix from selected feature columns."""
    mat = df[cols].fillna(0.0).values.astype(float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_n = mat / norms
    sim = np.clip(mat_n @ mat_n.T, 0.0, 1.0)
    return pd.DataFrame(sim, index=df.index, columns=df.index)


def pairwise_jsdiv(
    hists: dict[str, np.ndarray],
    patients: list[str],
) -> pd.DataFrame:
    """N×N Jensen–Shannon divergence matrix from histogram dicts."""
    n = len(patients)
    mat = np.full((n, n), np.nan)
    for i, p1 in enumerate(patients):
        for j, p2 in enumerate(patients):
            h1, h2 = hists.get(p1), hists.get(p2)
            if h1 is not None and h2 is not None and h1.sum() > 0 and h2.sum() > 0:
                mat[i, j] = float(jensenshannon(h1, h2))
            else:
                mat[i, j] = np.nan
    return pd.DataFrame(mat, index=patients, columns=patients)


def pairwise_dice(
    shapes: dict[str, np.ndarray],
    patients: list[str],
) -> pd.DataFrame:
    """N×N Dice coefficient matrix on binary head-shape masks."""
    n = len(patients)
    mat = np.full((n, n), np.nan)
    for i, p1 in enumerate(patients):
        for j, p2 in enumerate(patients):
            s1, s2 = shapes.get(p1), shapes.get(p2)
            if s1 is None or s2 is None:
                continue
            b1 = s1 > 0.5
            b2 = s2 > 0.5
            denom = b1.sum() + b2.sum()
            mat[i, j] = 2.0 * float((b1 & b2).sum()) / denom if denom > 0 else 0.0
    return pd.DataFrame(mat, index=patients, columns=patients)


def extract_body_surface_maps(
    acoustic_data: dict[str, dict[str, np.ndarray]],
    patients: list[str],
) -> dict[str, np.ndarray]:
    """
    Per patient, extract first-tissue index along x for each (y,z) column from
    body_mask (or fallback HU threshold), then downsample in y/z.
    Returned arrays are shape (ny_sub, nz_sub) with NaN where no tissue exists.
    """
    surf_maps: dict[str, np.ndarray] = {}
    s = SURFACE_YZ_STRIDE

    for pid in patients:
        arrays = acoustic_data.get(pid, {})
        bm = arrays.get("body_mask")
        if bm is not None:
            # Keep full x resolution, downsample only y/z.
            b = np.asarray(bm[:, ::s, ::s]).astype(bool)  # (nx, ny_sub, nz_sub)
        else:
            hu = arrays.get("hu")
            if hu is None:
                continue
            b = np.asarray(hu[:, ::s, ::s]) > -500.0

        has_tissue = b.any(axis=0)  # (ny_sub, nz_sub)
        first = np.argmax(b.astype(np.uint8), axis=0).astype(np.float32)
        first[~has_tissue] = np.nan
        surf_maps[pid] = first

    return surf_maps


def _rim_mask(valid_mask: np.ndarray, frac: float = 0.12) -> np.ndarray:
    """Ring mask around the valid domain bounding box."""
    ys, zs = np.where(valid_mask)
    if ys.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max())
    z0, z1 = int(zs.min()), int(zs.max())
    h = max(1, y1 - y0 + 1)
    w = max(1, z1 - z0 + 1)
    by = max(1, int(round(h * frac)))
    bz = max(1, int(round(w * frac)))

    rim = np.zeros_like(valid_mask, dtype=bool)
    rim[y0:y0 + by, z0:z1 + 1] = True
    rim[y1 - by + 1:y1 + 1, z0:z1 + 1] = True
    rim[y0:y1 + 1, z0:z0 + bz] = True
    rim[y0:y1 + 1, z1 - bz + 1:z1 + 1] = True
    return rim & valid_mask


def compute_external_seating_matrices(
    surf_maps: dict[str, np.ndarray],
    patients: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Pairwise external-seating metrics using first-tissue x(y,z) maps.
    A->B compares source patient A surface template against target B after
    global x-shift alignment (median offset), matching lens-transplant behavior.
    """
    n = len(patients)
    names = [
        "contact_fraction",
        "mean_gap_vox",
        "p95_gap_vox",
        "penetration_fraction",
        "rim_gap_fraction",
    ]
    mats = {k: np.full((n, n), np.nan, dtype=float) for k in names}

    for i, pa in enumerate(patients):
        sa = surf_maps.get(pa)
        for j, pb in enumerate(patients):
            sb = surf_maps.get(pb)
            if sa is None or sb is None:
                continue
            # Center-crop both maps to common size for pairwise comparison.
            h = min(sa.shape[0], sb.shape[0])
            w = min(sa.shape[1], sb.shape[1])
            if h < 10 or w < 10:
                continue
            a0 = (sa.shape[0] - h) // 2
            a1 = (sa.shape[1] - w) // 2
            b0 = (sb.shape[0] - h) // 2
            b1 = (sb.shape[1] - w) // 2
            sa_c = sa[a0:a0 + h, a1:a1 + w]
            sb_c = sb[b0:b0 + h, b1:b1 + w]
            valid = np.isfinite(sa_c) & np.isfinite(sb_c)
            if np.sum(valid) < 200:
                continue

            delta = sb_c[valid] - sa_c[valid]
            # Align global x offset, keep local mismatch as "gap".
            gap = delta - np.median(delta)
            abs_gap = np.abs(gap)

            mats["contact_fraction"][i, j] = float(np.mean(abs_gap <= CONTACT_TOL_VOX))
            mats["mean_gap_vox"][i, j] = float(np.mean(abs_gap))
            mats["p95_gap_vox"][i, j] = float(np.percentile(abs_gap, 95.0))
            mats["penetration_fraction"][i, j] = float(np.mean(gap < -CONTACT_TOL_VOX))

            rim = _rim_mask(valid)
            if np.any(rim):
                rim_gap = np.abs((sb_c - sa_c) - np.nanmedian(sb_c[rim] - sa_c[rim]))
                mats["rim_gap_fraction"][i, j] = float(np.mean(rim_gap[rim] > CONTACT_TOL_VOX))

    # Fill diagonals with identity-like values.
    for k, v in mats.items():
        np.fill_diagonal(v, 1.0 if k == "contact_fraction" else 0.0)

    # Build total ranking score (lower is better).
    def _norm(m: np.ndarray, inverse: bool = False) -> np.ndarray:
        x = m.copy()
        valid = np.isfinite(x)
        if not np.any(valid):
            return x
        mn, mx = np.nanmin(x), np.nanmax(x)
        if mx - mn < 1e-12:
            z = np.zeros_like(x)
        else:
            z = (x - mn) / (mx - mn)
        return 1.0 - z if inverse else z

    contact_cost = _norm(mats["contact_fraction"], inverse=True)  # higher contact is better
    mean_gap_n = _norm(mats["mean_gap_vox"])
    p95_gap_n = _norm(mats["p95_gap_vox"])
    pen_n = _norm(mats["penetration_fraction"])
    rim_n = _norm(mats["rim_gap_fraction"])

    total = (
        0.30 * contact_cost +
        0.25 * mean_gap_n +
        0.20 * p95_gap_n +
        0.15 * pen_n +
        0.10 * rim_n
    )
    np.fill_diagonal(total, 0.0)

    out = {k: pd.DataFrame(v, index=patients, columns=patients) for k, v in mats.items()}
    out["total_external_fit_score"] = pd.DataFrame(total, index=patients, columns=patients)
    return out


def plot_external_mean_gap_debug(
    surf_maps: dict[str, np.ndarray],
    source_id: str,
    target_id: str,
    out_path: Path,
) -> None:
    """
    Step-by-step visualisation for external mean-gap calculation (source -> target).
    Shows raw maps, center-cropped maps, valid overlap, delta/alignment, and final
    residual gap map used for mean-gap metric.
    """
    sa = surf_maps.get(source_id)
    sb = surf_maps.get(target_id)
    if sa is None or sb is None:
        return

    # Center-crop to common size
    h = min(sa.shape[0], sb.shape[0])
    w = min(sa.shape[1], sb.shape[1])
    if h < 10 or w < 10:
        return
    a0 = (sa.shape[0] - h) // 2
    a1 = (sa.shape[1] - w) // 2
    b0 = (sb.shape[0] - h) // 2
    b1 = (sb.shape[1] - w) // 2
    sa_c = sa[a0:a0 + h, a1:a1 + w]
    sb_c = sb[b0:b0 + h, b1:b1 + w]

    valid_all = np.isfinite(sa_c) & np.isfinite(sb_c)
    if np.sum(valid_all) < 200:
        return

    # Restrict to "top bowl" region: high first-tissue-x values on both maps.
    qa = float(np.nanpercentile(sa_c[valid_all], TOP_BOWL_PERCENTILE))
    qb = float(np.nanpercentile(sb_c[valid_all], TOP_BOWL_PERCENTILE))
    top_bowl = (sa_c >= qa) & (sb_c >= qb) & valid_all
    valid = top_bowl
    if np.sum(valid) < 120:
        # Fallback if percentile mask is too sparse.
        valid = valid_all

    delta = sb_c - sa_c
    med_shift = float(np.nanmedian(delta[valid]))
    gap = delta - med_shift
    abs_gap = np.abs(gap)
    mean_gap = float(np.nanmean(abs_gap[valid]))
    contact = float(np.mean(abs_gap[valid] <= CONTACT_TOL_VOX))
    p95_gap = float(np.nanpercentile(abs_gap[valid], 95.0))

    # Build aligned "source on target" surface for visual comparison
    sa_aligned = sa_c + med_shift

    vmax_raw = float(np.nanpercentile(np.r_[sa_c[valid], sb_c[valid]], 99))
    vmin_raw = float(np.nanpercentile(np.r_[sa_c[valid], sb_c[valid]], 1))

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    ax = axes.ravel()

    im0 = ax[0].imshow(sa, cmap="viridis", origin="lower")
    ax[0].set_title(f"Step 1: Source raw surface map\n{source_id}")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, label="first tissue x (vox)")

    im1 = ax[1].imshow(sb, cmap="viridis", origin="lower")
    ax[1].set_title(f"Step 1: Target raw surface map\n{target_id}")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, label="first tissue x (vox)")

    im2 = ax[2].imshow(sa_c, cmap="viridis", origin="lower", vmin=vmin_raw, vmax=vmax_raw)
    ax[2].set_title(f"Step 2: Source center-cropped\nshape={sa_c.shape}")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    im3 = ax[3].imshow(sb_c, cmap="viridis", origin="lower", vmin=vmin_raw, vmax=vmax_raw)
    ax[3].set_title(f"Step 2: Target center-cropped\nshape={sb_c.shape}")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    show_mask = np.zeros_like(sa_c, dtype=float)
    show_mask[valid_all] = 0.35
    show_mask[top_bowl] = 1.0
    im4 = ax[4].imshow(show_mask, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax[4].set_title(
        "Step 3: Overlap + top-bowl mask\n"
        f"all={int(np.sum(valid_all))}, top_bowl={int(np.sum(top_bowl))}"
    )
    plt.colorbar(im4, ax=ax[4], fraction=0.046, label="valid")

    im5 = ax[5].imshow(delta, cmap="coolwarm", origin="lower", vmin=-20, vmax=20)
    ax[5].set_title(f"Step 4: Raw delta = target - source\nmedian(delta)={med_shift:.2f} vox")
    plt.colorbar(im5, ax=ax[5], fraction=0.046, label="vox")

    im6 = ax[6].imshow(sa_aligned - sb_c, cmap="coolwarm", origin="lower", vmin=-20, vmax=20)
    ax[6].set_title("Step 5: Source after median x-alignment\n(aligned source - target)")
    plt.colorbar(im6, ax=ax[6], fraction=0.046, label="vox")

    show_gap = np.where(valid, abs_gap, np.nan)
    im7 = ax[7].imshow(show_gap, cmap="magma", origin="lower", vmin=0, vmax=15)
    ax[7].set_title(
        "Step 6: Residual |gap| (top-bowl only)\n"
        f"mean={mean_gap:.2f} vox, p95={p95_gap:.2f} vox, contact={contact:.3f}"
    )
    plt.colorbar(im7, ax=ax[7], fraction=0.046, label="|gap| vox")

    for a in ax:
        a.set_xlabel("z (downsampled)")
        a.set_ylabel("y (downsampled)")

    plt.suptitle(
        "External Seating Mean-Gap Debug View (Source -> Target)\n"
        f"{source_id} -> {target_id}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ── Plotting helpers ──────────────────────────────────────────────────────────

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
    """Generic N×N heatmap plot and save."""
    n = len(df)
    fig_w = max(10, n * 0.38)
    fig_h = max(9,  n * 0.36)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    labels = [short(p) for p in df.index]
    do_annot = (n <= 30)

    sns.heatmap(
        df.values.astype(float),
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        annot=do_annot,
        fmt=fmt if do_annot else "",
        annot_kws={"size": max(4, 8 - n // 10)},
        linewidths=0.3 if n <= 30 else 0,
        linecolor="lightgray",
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ── Individual plot functions ──────────────────────────────────────────────────

def plot_tissue_composition(
    stats_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    frac_cols = [c for c in stats_df.columns if c.startswith("frac_")]
    tissues = [c.replace("frac_", "") for c in frac_cols]

    avail = [p for p in patients if p in stats_df.index]
    df_p = stats_df.loc[avail, frac_cols].copy()
    df_p.columns = tissues
    df_p.index = [short(p) for p in df_p.index]

    colors = plt.cm.Set3(np.linspace(0, 1, len(tissues)))
    fig, ax = plt.subplots(figsize=(max(14, len(avail) * 0.38), 6))
    df_p.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.85, edgecolor="none")
    ax.set_title("Tissue Composition per Patient (Voxel Fractions)", fontweight="bold")
    ax.set_xlabel("Patient")
    ax.set_ylabel("Fraction of total voxels")
    ax.legend(title="Tissue", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=55, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_acoustic_boxplots(
    acoustic_data: dict[str, dict[str, np.ndarray]],
    patients: list[str],
    out_path: Path,
    rng: np.random.Generator | None = None,
) -> None:
    """Boxplot of body-region acoustic properties per patient."""
    if rng is None:
        rng = np.random.default_rng(42)

    prop_specs = [
        ("sound_speed", "Sound Speed (m/s)",        (1400, 2800)),
        ("density",     "Density (kg/m³)",           (900,  2000)),
        ("attenuation", "Attenuation (dB/cm/MHz)",   (0,    50)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(patients) * 0.38), 15))
    max_pts = 4000  # subsample per patient for speed

    s = SPATIAL_STRIDE
    for ax, (prop_key, ylabel, (lo, hi)) in zip(axes, prop_specs):
        series_list, labels = [], []
        for pid in patients:
            arrays = acoustic_data.get(pid, {})
            arr  = arrays.get(prop_key)
            if arr is None:
                continue
            bm = arrays.get("body_mask")
            if bm is not None:
                body = np.asarray(bm[::s, ::s, ::s]).astype(bool)
            else:
                hu = arrays.get("hu")
                if hu is None:
                    continue
                body = np.asarray(hu[::s, ::s, ::s]) > -500.0
            vals = np.asarray(arr[::s, ::s, ::s])[body].flatten().astype(float)
            vals = vals[(vals >= lo) & (vals <= hi)]
            if vals.size > max_pts:
                vals = rng.choice(vals, max_pts, replace=False)
            series_list.append(vals)
            labels.append(short(pid))

        if not series_list:
            continue

        bplot = ax.boxplot(
            series_list, labels=labels, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue", color="navy"),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="navy"),
            capprops=dict(color="navy"),
        )
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — Body Region Distribution per Patient")
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Acoustic Property Distributions per Patient", fontsize=12,
                 fontweight="bold", y=1.002)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_pca(
    stats_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in stats_df.index]
    feat_cols = [c for c in stats_df.columns if c.startswith(("frac_", "hu_mean_"))]
    X_df = stats_df.loc[avail, feat_cols].fillna(0.0)

    if len(X_df) < 4:
        return

    X = StandardScaler().fit_transform(X_df.values)
    n_comp = min(3, X.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X)

    colors = [patient_group_color(p) for p in avail]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for pid, c, xy in zip(avail, colors, coords[:, :2]):
        ax.scatter(*xy, color=c, s=55, zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(short(pid), xy, fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    ax.set_title("PCA — PC1 vs PC2")
    ax.grid(alpha=0.3)
    legend_handles = [
        Patch(color="steelblue",  label="ACRIN-FMISO-Brain"),
        Patch(color="tomato",     label="TCGA-14"),
        Patch(color="forestgreen",label="C3L"),
        Patch(color="darkorchid", label="C3N"),
    ]
    ax.legend(handles=legend_handles, fontsize=8)

    ax = axes[1]
    ev = pca.explained_variance_ratio_ * 100
    cum_ev = np.cumsum(ev)
    ax.bar(range(1, n_comp + 1), ev, color="steelblue", edgecolor="navy", alpha=0.85)
    ax.step(range(1, n_comp + 1), cum_ev, color="red", where="post", linewidth=2,
            label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Explained Variance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("PCA of CT-Scan Features (Tissue Fractions + Mean HU)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_acoustic_region_means_from_stats(
    stats_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    """
    Plot raw mean acoustic properties for brain_soft and skull regions using tissue_stats.csv
    (body-mask normalised; does not require HU arrays).
    """
    avail = [p for p in patients if p in stats_df.index]
    if not avail:
        return

    def _wmean(pid: str, key: str) -> float:
        # skull = weighted mean of cortical + trabecular by voxel count
        c_c = float(stats_df.loc[pid].get("count_cortical_skull", 0.0))
        c_t = float(stats_df.loc[pid].get("count_trabecular_skull", 0.0))
        v_c = float(stats_df.loc[pid].get(f"{key}_cortical_skull", np.nan))
        v_t = float(stats_df.loc[pid].get(f"{key}_trabecular_skull", np.nan))
        denom = c_c + c_t
        if denom <= 0:
            return np.nan
        parts = []
        if np.isfinite(v_c) and c_c > 0:
            parts.append(v_c * c_c)
        if np.isfinite(v_t) and c_t > 0:
            parts.append(v_t * c_t)
        return float(np.sum(parts) / denom) if parts else np.nan

    records = []
    for pid in avail:
        row = {"patient_id": short(pid)}
        row["brain_soft c (m/s)"] = float(stats_df.loc[pid].get("c_mean_brain_soft", np.nan))
        row["brain_soft ρ (kg/m³)"] = float(stats_df.loc[pid].get("rho_mean_brain_soft", np.nan))
        row["brain_soft α (dB/cm/MHz)"] = float(stats_df.loc[pid].get("att_mean_brain_soft", np.nan))
        row["skull c (m/s)"] = _wmean(pid, "c_mean")
        row["skull ρ (kg/m³)"] = _wmean(pid, "rho_mean")
        row["skull α (dB/cm/MHz)"] = _wmean(pid, "att_mean")
        records.append(row)

    df = pd.DataFrame(records).set_index("patient_id")
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(10, len(records) * 0.36)))
    sns.heatmap(df, ax=axes[0], cmap="plasma", annot=True, fmt=".1f",
                annot_kws={"size": 6}, linewidths=0.3, linecolor="lightgray")
    axes[0].set_title("Raw Mean Acoustic Properties (Body Mask)\nBrain Soft & Skull",
                      fontweight="bold")

    positive_vals = df.values[np.isfinite(df.values) & (df.values > 0)]
    if positive_vals.size > 0:
        vmin = float(np.min(positive_vals))
        vmax = float(np.max(positive_vals))
        norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin + 1e-6))
    else:
        norm = None
    sns.heatmap(df, ax=axes[1], cmap="plasma", annot=True, fmt=".1f",
                annot_kws={"size": 6}, linewidths=0.3, linecolor="lightgray",
                norm=norm)
    axes[1].set_title("Raw Mean Acoustic Properties (log color scale)",
                      fontweight="bold")

    plt.suptitle("Per-Region Acoustic Property Comparison Across Patients",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")

def plot_head_shape_grid(
    shapes: dict[str, np.ndarray],
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in shapes]
    n = len(avail)
    if n == 0:
        return
    ncols = 8
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 1.9, nrows * 2.1))
    axes = np.array(axes).flatten()

    for i, pid in enumerate(avail):
        axes[i].imshow(shapes[pid], cmap="bone", origin="lower", aspect="equal",
                       interpolation="nearest")
        axes[i].set_title(short(pid), fontsize=6, pad=2)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Top-of-Head Cross-Section Gallery (Superior View)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_hu_per_tissue(
    stats_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    hu_cols_all = [c for c in stats_df.columns if c.startswith("hu_mean_")]
    tissues_all = [c.replace("hu_mean_", "") for c in hu_cols_all]
    # Remove air column (it dominates the HU range and isn't useful here)
    keep = [t != "air" for t in tissues_all]
    hu_cols = [c for c, k in zip(hu_cols_all, keep) if k]
    tissues = [t for t in tissues_all if t != "air"]

    avail = [p for p in patients if p in stats_df.index]
    df_p = stats_df.loc[avail, hu_cols].copy()
    df_p.columns = tissues
    df_p.index = [short(p) for p in df_p.index]
    df_p = df_p.replace(0.0, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(tissues) * 1.5),
                                     max(10, len(avail) * 0.36)))
    sns.heatmap(df_p, ax=ax, cmap="RdYlBu_r", annot=True, fmt=".0f",
                annot_kws={"size": 7}, linewidths=0.5, linecolor="lightgray",
                cbar_kws={"label": "Mean HU"})
    ax.set_title("Mean Hounsfield Unit per Tissue per Patient", fontweight="bold")
    ax.set_xlabel("Tissue type")
    ax.set_ylabel("Patient")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_structural_scatter(
    stats_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    avail = [p for p in patients if p in stats_df.index]
    df = stats_df.loc[avail].copy()
    df.index = [short(p) for p in df.index]
    colors = [patient_group_color(p) for p in avail]

    comparisons = [
        ("frac_brain_soft",      "frac_cortical_skull",    "Brain Soft",     "Cortical Skull"),
        ("frac_brain_soft",      "frac_trabecular_skull",  "Brain Soft",     "Trabecular Skull"),
        ("frac_cortical_skull",  "frac_trabecular_skull",  "Cortical Skull", "Trabecular Skull"),
        # If present (e.g. CQ500 hounsfield2density), include CSF explicitly.
        ("frac_csf",             "frac_brain_soft",        "CSF",            "Brain Soft"),
        ("frac_csf",             "frac_cortical_skull",    "CSF",            "Cortical Skull"),
        ("frac_csf",             "frac_trabecular_skull",  "CSF",            "Trabecular Skull"),
    ]

    ncols = len(comparisons)
    fig, axes = plt.subplots(1, ncols, figsize=(max(15, 5 * ncols), 5))
    axes = axes.flatten()

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, comparisons):
        if xcol not in df.columns or ycol not in df.columns:
            ax.axis("off")
            continue
        ax.scatter(df[xcol], df[ycol], c=colors, s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=5)
        for pid_s, x, y in zip(df.index, df[xcol], df[ycol]):
            ax.annotate(pid_s, (x, y), fontsize=5, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel(f"{xlabel} Fraction")
        ax.set_ylabel(f"{ylabel} Fraction")
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.grid(alpha=0.3)

    legend_handles = [
        Patch(color="steelblue",  label="ACRIN-FMISO-Brain"),
        Patch(color="tomato",     label="TCGA-14"),
        Patch(color="forestgreen",label="C3L"),
        Patch(color="darkorchid", label="C3N"),
    ]
    axes[-1].legend(handles=legend_handles, fontsize=9)
    axes[-1].axis("off")

    plt.suptitle("Pairwise Tissue-Fraction Structural Relationships",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_acoustic_region_heatmap(
    acoustic_data: dict[str, dict[str, np.ndarray]],
    patients: list[str],
    out_path: Path,
) -> None:
    """Per-region (skull vs brain) mean acoustic property heatmap."""
    region_defs = {
        "skull": lambda hu: hu > 200.0,
        "brain": lambda hu: (hu >= -100.0) & (hu < 200.0),
    }
    prop_keys = [
        ("sound_speed", "c (m/s)"),
        ("density",     "ρ (kg/m³)"),
        ("attenuation", "α (dB/cm/MHz)"),
    ]

    records = []
    for pid in patients:
        arrays = acoustic_data.get(pid, {})
        hu = arrays.get("hu")
        if hu is None:
            continue
        s = SPATIAL_STRIDE
        hu_arr = np.asarray(hu[::s, ::s, ::s])
        bm = arrays.get("body_mask")
        body = np.asarray(bm[::s, ::s, ::s]).astype(bool) if bm is not None else (hu_arr > -500.0)
        row: dict = {"patient_id": short(pid)}
        for reg_name, reg_fn in region_defs.items():
            mask = reg_fn(hu_arr) & body
            for key, lbl in prop_keys:
                arr = arrays.get(key)
                if arr is None:
                    continue
                vals = np.asarray(arr[::s, ::s, ::s])[mask]
                row[f"{reg_name} {lbl}"] = float(vals.mean()) if vals.size > 0 else np.nan
        records.append(row)

    if not records:
        print("    Skipped: 13_acoustic_region_heatmap.png (HU arrays not available)")
        return

    df = pd.DataFrame(records).set_index("patient_id")
    if df.empty:
        return

    # Normalise 0-1 per column for the right panel
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(10, len(records) * 0.36)))

    sns.heatmap(df, ax=axes[0], cmap="plasma", annot=True, fmt=".1f",
                annot_kws={"size": 6}, linewidths=0.3, linecolor="lightgray")
    axes[0].set_title("Raw Mean Acoustic Properties\n(Skull & Brain Regions)",
                      fontweight="bold")

    sns.heatmap(df_norm, ax=axes[1], cmap="plasma", annot=True, fmt=".2f",
                annot_kws={"size": 6}, linewidths=0.3, linecolor="lightgray",
                vmin=0, vmax=1)
    axes[1].set_title("Normalised (0 = min patient, 1 = max patient)",
                      fontweight="bold")

    plt.suptitle("Per-Region Acoustic Property Comparison Across Patients",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_feature_correlation(
    stats_df: pd.DataFrame,
    patients: list[str],
    out_path: Path,
) -> None:
    cols = [c for c in stats_df.columns if c.startswith(("frac_", "hu_mean_"))]
    avail = [p for p in patients if p in stats_df.index]
    df = stats_df.loc[avail, cols].fillna(0.0)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                mask=mask, annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, linecolor="lightgray",
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Feature Inter-Correlation Matrix\n"
                 "(Tissue Fractions + Mean HU per Tissue)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_acoustic_mean_bars(
    acoustic_data: dict[str, dict[str, np.ndarray]],
    patients: list[str],
    out_path: Path,
) -> None:
    prop_specs = [
        ("sound_speed", "Sound Speed (m/s)",      (1400, 2800)),
        ("density",     "Density (kg/m³)",         (900,  2000)),
        ("attenuation", "Attenuation (dB/cm/MHz)", (0,    50)),
    ]

    records = []
    for pid in patients:
        arrays = acoustic_data.get(pid, {})
        s = SPATIAL_STRIDE
        bm = arrays.get("body_mask")
        if bm is not None:
            body = np.asarray(bm[::s, ::s, ::s]).astype(bool)
        else:
            hu = arrays.get("hu")
            if hu is None:
                continue
            body = np.asarray(hu[::s, ::s, ::s]) > -500.0
        row: dict = {"patient_id": short(pid)}
        for key, _, (lo, hi) in prop_specs:
            arr = arrays.get(key)
            if arr is None:
                continue
            vals = np.asarray(arr[::s, ::s, ::s])[body].astype(float)
            vals = vals[(vals >= lo) & (vals <= hi)]
            row[f"{key}_mean"] = float(vals.mean()) if vals.size > 0 else np.nan
            row[f"{key}_std"]  = float(vals.std())  if vals.size > 0 else np.nan
        records.append(row)

    df = pd.DataFrame(records).set_index("patient_id")
    colors = [patient_group_color(p) for p in patients
              if short(p) in df.index]

    fig, axes = plt.subplots(len(prop_specs), 1,
                              figsize=(max(14, len(records) * 0.4), 13))

    for ax, (key, ylabel, (lo, hi)) in zip(axes, prop_specs):
        mean_c = f"{key}_mean"
        std_c  = f"{key}_std"
        if mean_c not in df.columns:
            continue
        sub = df[[mean_c, std_c]].dropna()
        row_colors = [patient_group_color(
            next((p for p in patients if short(p) == idx), idx))
            for idx in sub.index]

        bars = ax.bar(range(len(sub)), sub[mean_c],
                      yerr=sub[std_c], capsize=3,
                      color=row_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5,
                      error_kw=dict(ecolor="black", elinewidth=0.8))
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(list(sub.index), rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Mean ± Std {ylabel} per Patient (Body Region)")
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        Patch(color="steelblue",  label="ACRIN-FMISO-Brain"),
        Patch(color="tomato",     label="TCGA-14"),
        Patch(color="forestgreen",label="C3L"),
        Patch(color="darkorchid", label="C3N"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")
    plt.suptitle("Acoustic Property Summary Statistics per Patient",
                 fontsize=12, fontweight="bold", y=1.002)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 70
    print(sep)
    print("CT Scan Pairwise Comparison Analysis")
    print(sep)

    # ── 1. Discover patients ──────────────────────────────────────────────────
    print("\n[1/6] Discovering patients …")
    patients = discover_patients()

    # ── 2. Load tissue stats & metadata ──────────────────────────────────────
    print("\n[2/6] Loading tissue statistics and metadata …")
    stats_df = load_tissue_stats(patients)
    meta_df  = load_metadata(patients)
    print(f"  Tissue-stat features: {list(stats_df.columns)}")
    stats_df.to_csv(OUT_DIR / "patient_feature_summary.csv")
    print(f"  Saved patient_feature_summary.csv")

    # ── 3. Load acoustic arrays ───────────────────────────────────────────────
    print("\n[3/6] Loading 128³ acoustic arrays …")
    acoustic_data = load_acoustic_arrays(patients)
    has_hu_data = any(arrays.get("hu") is not None for arrays in acoustic_data.values())

    # ── 4. Compute histograms & head shapes ───────────────────────────────────
    print("\n[4/6] Computing body-region histograms …")
    hists = compute_body_histograms(acoustic_data, patients)

    print("\n[5/6] Extracting top-of-head shapes …")
    head_shapes = extract_head_shapes(acoustic_data, patients)
    print(f"  Extracted shapes for {len(head_shapes)}/{len(patients)} patients.")

    print_pitch_deck_per_scan(
        patients, stats_df, meta_df, acoustic_data, head_shapes
    )
    print_tissue_acoustic_cohort_stats(stats_df, patients)

    print("\n[5b/6] Extracting body-surface maps for external seating …")
    surf_maps = extract_body_surface_maps(acoustic_data, patients)
    seating = compute_external_seating_matrices(surf_maps, patients)

    # ── 5. Generate plots ──────────────────────────────────────────────────────
    print("\n[6/6] Generating plots …")

    # --- Pairwise comparison matrices -----------------------------------------
    print("\n  Pairwise matrices:")

    frac_cols = [c for c in stats_df.columns if c.startswith("frac_")]
    tissue_sim = pairwise_cosine_similarity(stats_df, frac_cols)
    _heatmap(
        tissue_sim,
        "Tissue Composition Similarity (Cosine)\n"
        "Higher = more similar tissue-fraction profiles",
        OUT_DIR / "01_tissue_composition_similarity.png",
        vmin=0.88, vmax=1.0, cmap="YlGn",
    )

    pairwise_specs = [
        ("sound_speed", "02",
         "Sound Speed Distribution Distance (Jensen–Shannon)\n"
         "Lower = more similar distributions across patients"),
        ("density", "03",
         "Density Distribution Distance (Jensen–Shannon)\n"
         "Lower = more similar distributions"),
        ("attenuation", "04",
         "Attenuation Distribution Distance (Jensen–Shannon)\n"
         "Lower = more similar distributions"),
    ]
    if has_hu_data:
        pairwise_specs.append(
            ("hu", "05",
             "Hounsfield-Unit Distribution Distance (Jensen–Shannon)\n"
             "Lower = more similar HU profiles")
        )
    else:
        print("    Skipped: 05_hu_jsdiv.png (HU arrays not available)")

    for prop, num, title in pairwise_specs:
        mat = pairwise_jsdiv(hists[prop], patients)
        _heatmap(mat, title,
                 OUT_DIR / f"{num}_{prop}_jsdiv.png",
                 vmin=0.0, vmax=0.5, cmap="YlOrRd",
                 fmt=".3f")

    shape_sim = pairwise_dice(head_shapes, patients)
    _heatmap(
        shape_sim,
        "Top-of-Head Shape Similarity (Dice Coefficient)\n"
        "Higher = more similar superior-view head geometry",
        OUT_DIR / "06_head_shape_similarity.png",
        vmin=0.0, vmax=1.0, cmap="YlGn",
    )

    _heatmap(
        seating["contact_fraction"],
        "External Seating Contact Fraction (A->B)\n"
        "Higher = more lens-template contact after median x-alignment",
        OUT_DIR / "16_external_contact_fraction.png",
        vmin=0.0, vmax=1.0, cmap="YlGn",
    )
    _heatmap(
        seating["mean_gap_vox"],
        "External Seating Mean Gap (voxels, A->B)\n"
        "Lower = better physical conformity",
        OUT_DIR / "17_external_mean_gap_vox.png",
        vmin=0.0, vmax=20.0, cmap="YlOrRd", fmt=".2f",
    )
    _heatmap(
        seating["p95_gap_vox"],
        "External Seating P95 Gap (voxels, A->B)\n"
        "Lower = fewer large local mismatches",
        OUT_DIR / "18_external_p95_gap_vox.png",
        vmin=0.0, vmax=40.0, cmap="YlOrRd", fmt=".2f",
    )
    _heatmap(
        seating["penetration_fraction"],
        "External Seating Penetration Fraction (A->B)\n"
        "Lower = less body-mask intersection risk",
        OUT_DIR / "19_external_penetration_fraction.png",
        vmin=0.0, vmax=1.0, cmap="YlOrRd", fmt=".3f",
    )
    _heatmap(
        seating["rim_gap_fraction"],
        "External Seating Rim Gap Fraction (A->B)\n"
        "Lower = better edge/rim continuity",
        OUT_DIR / "20_external_rim_gap_fraction.png",
        vmin=0.0, vmax=1.0, cmap="YlOrRd", fmt=".3f",
    )
    _heatmap(
        seating["total_external_fit_score"],
        "Total External Seating Fit Score (A->B)\n"
        "Lower = better overall physical fit",
        OUT_DIR / "21_external_total_fit_score.png",
        vmin=0.0, vmax=1.0, cmap="YlOrRd", fmt=".3f",
    )

    # Step-by-step visual debug for one pair (best and worst mean-gap example).
    mean_gap_df = seating["mean_gap_vox"].copy()
    mg = mean_gap_df.values.copy()
    np.fill_diagonal(mg, np.nan)
    if np.isfinite(mg).any():
        bi, bj = np.unravel_index(np.nanargmin(mg), mg.shape)
        wi, wj = np.unravel_index(np.nanargmax(mg), mg.shape)
        src_best, tgt_best = mean_gap_df.index[bi], mean_gap_df.columns[bj]
        src_worst, tgt_worst = mean_gap_df.index[wi], mean_gap_df.columns[wj]
        plot_external_mean_gap_debug(
            surf_maps, src_best, tgt_best,
            OUT_DIR / "22_external_mean_gap_debug_best_pair.png",
        )
        plot_external_mean_gap_debug(
            surf_maps, src_worst, tgt_worst,
            OUT_DIR / "23_external_mean_gap_debug_worst_pair.png",
        )

    # --- Additional analyses --------------------------------------------------
    print("\n  Additional plots:")

    plot_tissue_composition(
        stats_df, patients,
        OUT_DIR / "07_tissue_composition_bars.png",
    )
    plot_acoustic_boxplots(
        acoustic_data, patients,
        OUT_DIR / "08_acoustic_boxplots.png",
    )
    # PCA removed (not useful for this workflow)
    plot_head_shape_grid(
        head_shapes, patients,
        OUT_DIR / "10_head_shape_gallery.png",
    )
    # HU-per-tissue plot removed for CQ500 minimal outputs.
    plot_structural_scatter(
        stats_df, patients,
        OUT_DIR / "12_structural_tissue_scatter.png",
    )
    plot_acoustic_region_means_from_stats(
        stats_df, patients,
        OUT_DIR / "13_acoustic_region_means_heatmap.png",
    )
    plot_feature_correlation(
        stats_df, patients,
        OUT_DIR / "14_feature_correlation.png",
    )
    plot_acoustic_mean_bars(
        acoustic_data, patients,
        OUT_DIR / "15_acoustic_mean_bars.png",
    )

    # Mean "how well this source lens fits all targets" ranking.
    total_fit = seating["total_external_fit_score"].copy()
    np.fill_diagonal(total_fit.values, np.nan)
    rank = pd.DataFrame({
        "source_patient": total_fit.index,
        "mean_total_fit_score_to_others": np.nanmean(total_fit.values, axis=1),
        "median_total_fit_score_to_others": np.nanmedian(total_fit.values, axis=1),
    }).sort_values("mean_total_fit_score_to_others", ascending=True)
    rank.to_csv(OUT_DIR / "external_seating_source_ranking.csv", index=False)
    print("    Saved: external_seating_source_ranking.csv")

    print(f"\n{sep}")
    print(f"Analysis complete. 21 figures + 2 CSVs saved to:")
    print(f"  {OUT_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
