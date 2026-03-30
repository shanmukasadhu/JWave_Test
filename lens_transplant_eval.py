"""
lens_transplant_eval.py
=======================
Transplants an optimized acoustic holographic lens onto every patient in a
batch_dicom_to_acoustic_maps.py output directory and runs one forward
Helmholtz solve per patient.

Reports:
  - Max-pressure location (mm) in each patient's domain
  - Focus error = distance from max-pressure to the mapped training focal
    (skull-aligned x, lateral offset vs volume centre as in reference; training
    focal voxels match pol6: nx/1.5, ny/1.5, nz/2) or domain centre if reference
    cannot be loaded
  - Per-patient pressure slice plots with the max-pressure dot and focal
    target circle overlaid
  - A summary bar chart of all focus errors
  - summary.json

Configuration — edit the block below:
  LENS_DIR   : output folder from pol6wattenuation_trainingv2.py
               (must contain lens_sound_speed_ms.npy, lens_density_kgm3.npy,
                lens_meta.json, summary.json; optional lens_orange_mask.npy)
  If lens_orange_mask.npy is missing (old pol6 run), the script can rebuild it from
  the training patient's body mask: include that patient under PATIENT_DIR, or set
  REFERENCE_DIR_FOR_ORANGE_MASK to its folder (needs body_mask.npy).
  PATIENT_DIR: output folder from batch_dicom_to_acoustic_maps.py
               (sub-folders per patient, each with *_sound_speed_cleaned.npy,
                *_density_cleaned.npy, metadata.json)
  OUT_DIR    : where results are written (created if absent)
  SOLVER settings (method, tol, maxiter, restart) — match training run
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.ndimage import zoom

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp

try:
    from jwave import Domain, FourierSeries, helmholtz_solver
    from jwave.geometry import Medium
except Exception as _exc:
    raise ImportError(
        "jwave not found. Activate the correct conda env before running."
    ) from _exc

# ---------------------------------------------------------------------------
# ── CONFIGURATION ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Directory produced by pol6wattenuation_trainingv2.py for the reference patient.
LENS_DIR = Path(
    "/nfshomes/ssadhu1/Results/jwave_brain_forward_64/CQ500-CT-0/helmholtz_3d/"
)

# Directory produced by batch_dicom_to_acoustic_maps.py  (one sub-folder per patient).
PATIENT_DIR = Path(
    "/nfshomes/ssadhu1/ct_scans"
)

# Where to write evaluation results.
OUT_DIR = Path(
    "/nfshomes/ssadhu1/ct_scans/lens_transplant_eval"
)

# If ``lens_orange_mask.npy`` is missing in LENS_DIR (old pol6 run), set this to the
# batch sub-folder of the *training/reference* patient, OR leave None to auto-pick
# the folder whose name matches ``reference_id`` from training ``summary.json``.
REFERENCE_DIR_FOR_ORANGE_MASK = Path(
    "/nfshomes/ssadhu1/Results/jwave_brain_forward_64/CQ500-CT-0/helmholtz_3d/"
)

# When True, a rebuilt orange mask (and updated lens_meta skull_surface_x_train) is
# written into LENS_DIR so the next run loads files directly.
SAVE_REBUILT_ORANGE_TO_LENS_DIR = True

# Helmholtz solver settings — should match the training run.
SOLVER_METHOD  = "gmres"
SOLVER_TOL     = 2e-3
SOLVER_MAXITER = 400
SOLVER_RESTART = 10

# Focal voxel on the **preprocessed** training grid — must match
# ``pol6wattenuation_trainingv2`` (``plot_slices`` / focus loss), not geometric centre.
TRAINING_FOCAL_X_DIVISOR = 1.5
TRAINING_FOCAL_Y_DIVISOR = 1.5
TRAINING_FOCAL_Z_DIVISOR = 2.0


def training_focal_voxels(nx: int, ny: int, nz: int) -> Tuple[int, int, int]:
    """Same indices as pol6: ``int(nx/1.5), int(ny/1.5), int(nz/2)``."""
    cx = int(nx / TRAINING_FOCAL_X_DIVISOR)
    cy = int(ny / TRAINING_FOCAL_Y_DIVISOR)
    cz = int(nz / TRAINING_FOCAL_Z_DIVISOR)
    cx = int(np.clip(cx, 0, max(0, nx - 1)))
    cy = int(np.clip(cy, 0, max(0, ny - 1)))
    cz = int(np.clip(cz, 0, max(0, nz - 1)))
    return cx, cy, cz


# ---------------------------------------------------------------------------
# ── Transducer source (Jimenez-Gambin 2019, same as pol6) ──────────────────
# ---------------------------------------------------------------------------
_TRANSDUCER_RADIUS_MM = 50.0


def _make_transducer_source(nxyz, dx_mm, domain, frequency_hz, source_x_slice=0):
    nx, ny, nz = nxyz
    radius_vox = _TRANSDUCER_RADIUS_MM / dx_mm
    cy, cz = ny / 2.0, nz / 2.0
    yy = jnp.arange(ny, dtype=jnp.float32) - cy
    zz = jnp.arange(nz, dtype=jnp.float32) - cz
    YY, ZZ = jnp.meshgrid(yy, zz, indexing="ij")
    aperture = (YY ** 2 + ZZ ** 2 <= radius_vox ** 2).astype(jnp.float32)
    print(
        f"[source] PZT-26: radius={_TRANSDUCER_RADIUS_MM:.0f} mm={radius_vox:.1f} vox, "
        f"active={int(jnp.sum(aperture))} px, x_slice={source_x_slice}"
    )
    src = jnp.zeros((nx, ny, nz, 1), dtype=jnp.complex64)
    src = src.at[source_x_slice, :, :, 0].set(aperture.astype(jnp.complex64))
    return FourierSeries(src, domain)


# ---------------------------------------------------------------------------
# ── Preprocessing helpers (mirrors pol6 internal functions) ────────────────
# ---------------------------------------------------------------------------

def _resample_to_dx(vol: np.ndarray, spacing_xyz: Tuple[float, float, float],
                    target_dx: float, order: int = 1) -> np.ndarray:
    """Resample vol from spacing_xyz to isotropic target_dx mm."""
    sx, sy, sz = spacing_xyz
    factors = (sx / target_dx, sy / target_dx, sz / target_dx)
    out = zoom(vol, zoom=factors, order=order, prefilter=False)
    return out.astype(np.float32)


def _crop_x_both(arr: np.ndarray, dx: float, mm: float) -> np.ndarray:
    vox = int(round(mm / dx))
    nx = arr.shape[0]
    if vox <= 0 or 2 * vox >= nx:
        return arr
    return arr[vox:nx - vox]


def _crop_start(arr: np.ndarray, dx: float, mm: float, axis: int) -> np.ndarray:
    vox = int(round(mm / dx))
    if vox <= 0:
        return arr
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(vox, None)
    return arr[tuple(slices)]


def _crop_end(arr: np.ndarray, dx: float, mm: float, axis: int) -> np.ndarray:
    vox = int(round(mm / dx))
    if vox <= 0:
        return arr
    n = arr.shape[axis]
    if vox >= n:
        return arr
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None, n - vox)
    return arr[tuple(slices)]


def preprocess_patient(
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    att_xyz: Optional[np.ndarray],
    spacing_xyz: Tuple[float, float, float],
    target_dx: float,
    crop_x_mm: float,
    crop_y_start_mm: float,
    crop_z_start_mm: float,
    crop_z_end_mm: float,
    crop_x_end_mm: float,
    flip_propagation: bool,
    propagation_axis: int,
    water_bath_vox: int,
    body_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Replicate pol6's preprocessing pipeline:
      resample → crop → moveaxis → flip → prepend water bath
    Returns (c, rho, att, body_mask) all in the final internal (x, y, z) layout.
    """
    def _proc(v, order=1):
        v = _resample_to_dx(v, spacing_xyz, target_dx, order=order)
        v = _crop_x_both(v, target_dx, crop_x_mm)
        v = _crop_start(v, target_dx, crop_y_start_mm, axis=1)
        v = _crop_start(v, target_dx, crop_z_start_mm, axis=2)
        v = _crop_end(v, target_dx, crop_z_end_mm, axis=1)
        v = _crop_end(v, target_dx, crop_x_end_mm, axis=2)
        return v

    c   = _proc(c_xyz)
    rho = _proc(rho_xyz)
    att = _proc(att_xyz) if att_xyz is not None else None
    bm  = _proc(body_mask.astype(np.float32), order=0).astype(bool) if body_mask is not None else None

    # Move chosen propagation axis to position 0
    c   = np.moveaxis(c,   propagation_axis, 0)
    rho = np.moveaxis(rho, propagation_axis, 0)
    if att is not None:
        att = np.moveaxis(att, propagation_axis, 0)
    if bm is not None:
        bm = np.moveaxis(bm, propagation_axis, 0)

    if flip_propagation:
        c   = np.flip(c,   axis=0).copy()
        rho = np.flip(rho, axis=0).copy()
        if att is not None:
            att = np.flip(att, axis=0).copy()
        if bm is not None:
            bm = np.flip(bm, axis=0).copy()

    # Prepend water bath
    if water_bath_vox > 0:
        ny, nz = c.shape[1], c.shape[2]
        bath_c   = np.full((water_bath_vox, ny, nz), 1480.0, dtype=np.float32)
        bath_rho = np.full((water_bath_vox, ny, nz), 1000.0, dtype=np.float32)
        c   = np.concatenate([bath_c,   c],   axis=0)
        rho = np.concatenate([bath_rho, rho], axis=0)
        if att is not None:
            bath_att = np.zeros((water_bath_vox, ny, nz), dtype=np.float32)
            att = np.concatenate([bath_att, att], axis=0)
        if bm is not None:
            bath_bm = np.zeros((water_bath_vox, ny, nz), dtype=bool)
            bm = np.concatenate([bath_bm, bm], axis=0)

    return c, rho, att, bm


# ---------------------------------------------------------------------------
# ── Skull surface detection ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def find_skull_surface_x(body_mask: Optional[np.ndarray], water_bath_vox: int) -> int:
    """
    Return the x-index of the first tissue voxel along the propagation axis
    (axis 0), averaged across the y-z face.

    This is how pol6 locates the skull surface to define the orange design
    polygon.  By anchoring the lens just in front of this surface we ensure
    the lens is always in the water-coupling layer, never inside bone.

    Falls back to water_bath_vox if no body mask is available.
    """
    if body_mask is None:
        return water_bath_vox

    # body_mask shape: (nx, ny, nz)  — True inside body
    # argmax along axis-0 gives the first True index per (y,z) pixel.
    # Pixels with no body voxels at all return 0 from argmax, so mask those out.
    has_tissue = body_mask.any(axis=0)          # (ny, nz) bool
    first_tissue = np.argmax(body_mask.astype(np.int32), axis=0)  # (ny, nz)
    valid = first_tissue[has_tissue]
    if valid.size == 0:
        return water_bath_vox
    # Use the median to be robust to stray voxels
    return int(np.median(valid))


def map_reference_focal_to_patient(
    cx_ref: int,
    cy_ref: int,
    cz_ref: int,
    skull_x_ref: int,
    ny_ref: int,
    nz_ref: int,
    skull_x_pat: int,
    nx: int,
    ny: int,
    nz: int,
) -> Tuple[int, int, int]:
    """
    Map the training focal voxel (reference preprocessed grid) onto another
    patient's preprocessed grid.

    - **x (propagation):** preserve voxel offset from the detected skull surface:
      ``x_pat = skull_x_pat + (cx_ref - skull_x_ref)``.
    - **y, z:** preserve offset from the lateral volume centre (reference focal
      minus ``(ny_ref//2, nz_ref//2)``), same as when training focal is not at centre.

    Indices are clamped to the patient volume.  This is a geometric surrogate,
    not full deformable registration.
    """
    fx = skull_x_pat + (cx_ref - skull_x_ref)
    fy = ny // 2 + (cy_ref - ny_ref // 2)
    fz = nz // 2 + (cz_ref - nz_ref // 2)
    fx = int(np.clip(fx, 0, nx - 1))
    fy = int(np.clip(fy, 0, ny - 1))
    fz = int(np.clip(fz, 0, nz - 1))
    return fx, fy, fz


def load_reference_focal_geometry(
    batch_root: Path,
    reference_id: str,
    explicit_ref_dir: Optional[Path],
    *,
    dx_mm: float,
    crop_x_mm: float,
    crop_y_mm: float,
    crop_z_mm: float,
    crop_z_end_mm: float,
    crop_x_end_mm: float,
    flip_prop: bool,
    prop_axis: int,
    water_bath_vox: int,
) -> Optional[dict]:
    """
    Preprocess the reference (training) patient like pol6/eval; focal target
    on that grid matches pol6: ``training_focal_voxels(nx, ny, nz)`` (default
    ``int(nx/1.5), int(ny/1.5), int(nz/2)``).

    Returns a dict with ``nx_ref, ny_ref, nz_ref, skull_x_ref, cx_ref, cy_ref, cz_ref,
    reference_folder`` or None if maps cannot be loaded.
    """
    # Try explicit path first (orange-mask hint), then batch folder matching reference_id.
    # If explicit points at e.g. helmholtz_3d (no metadata.json), fall through — otherwise
    # focal would silently fall back to domain centre and the red marker sits at mid-volume.
    ref_candidates: list[Path] = []
    _seen: set[Path] = set()

    def _push_ref_dir(p: Optional[Path]) -> None:
        if p is None or not p.is_dir():
            return
        try:
            key = p.expanduser().resolve()
        except OSError:
            key = p
        if key in _seen:
            return
        _seen.add(key)
        ref_candidates.append(key)

    _push_ref_dir(explicit_ref_dir)
    _push_ref_dir(_resolve_reference_patient_dir(batch_root, reference_id, None))

    if not ref_candidates:
        print(
            f"[focal] Reference folder not found (id={reference_id!r}) under {batch_root} — "
            "using each patient's domain centre as focal target."
        )
        return None

    last_exc: Optional[Exception] = None
    ref_dir_used: Optional[Path] = None
    c: Optional[np.ndarray] = None
    bm: Optional[np.ndarray] = None
    for ref_dir in ref_candidates:
        ref_load_id = _infer_patient_id_for_dir(ref_dir)
        try:
            c_xyz, rho_xyz, att_xyz, bm0, spacing = _load_patient_maps(ref_dir, ref_load_id)
            c, rho, att, bm = preprocess_patient(
                c_xyz, rho_xyz, att_xyz, spacing,
                target_dx=dx_mm,
                crop_x_mm=crop_x_mm,
                crop_y_start_mm=crop_y_mm,
                crop_z_start_mm=crop_z_mm,
                crop_z_end_mm=crop_z_end_mm,
                crop_x_end_mm=crop_x_end_mm,
                flip_propagation=flip_prop,
                propagation_axis=prop_axis,
                water_bath_vox=water_bath_vox,
                body_mask=bm0,
            )
            ref_dir_used = ref_dir
            break
        except Exception as exc:
            last_exc = exc
            print(f"[focal] Skip reference candidate {ref_dir!s}: {exc}")
            continue

    if ref_dir_used is None or c is None or bm is None:
        print(
            f"[focal] Could not load reference patient maps from any candidate "
            f"(id={reference_id!r}) — using domain centre as focal target. Last error: {last_exc}"
        )
        return None

    nx_r, ny_r, nz_r = c.shape
    skull_x_ref = find_skull_surface_x(bm, water_bath_vox)
    cx_r, cy_r, cz_r = training_focal_voxels(nx_r, ny_r, nz_r)
    print(
        f"[focal] Reference grid {ref_dir_used.name!r}: shape=({nx_r},{ny_r},{nz_r}), "
        f"skull_x={skull_x_ref}, training focal (vox)=({cx_r},{cy_r},{cz_r}) "
        f"[pol6: nx/{TRAINING_FOCAL_X_DIVISOR}, ny/{TRAINING_FOCAL_Y_DIVISOR}, nz/{TRAINING_FOCAL_Z_DIVISOR}]"
    )
    return {
        "nx_ref": nx_r,
        "ny_ref": ny_r,
        "nz_ref": nz_r,
        "skull_x_ref": skull_x_ref,
        "cx_ref": cx_r,
        "cy_ref": cy_r,
        "cz_ref": cz_r,
        "reference_folder": ref_dir_used.name,
    }


def _x_surf_from_body_mask(body_mask: np.ndarray) -> np.ndarray:
    """Per-(y,z) first tissue index along x; columns with no tissue → nx (sentinel)."""
    nx_post, _, _ = body_mask.shape
    no_tissue = ~np.any(body_mask.astype(np.int32), axis=0)
    x_surf = np.argmax(body_mask.astype(np.int32), axis=0)
    return np.where(no_tissue, nx_post, x_surf).astype(np.float64)


def build_orange_mask_like_pol6(
    body_mask: np.ndarray,
    lens_x0: int,
    lens_x1: int,
    design_j_lo: int,
    design_j_hi: int,
    design_k_lo: int,
    design_k_hi: int,
) -> Tuple[np.ndarray, int]:
    """
    Same geometry as pol6wattenuation_trainingv2._orange_region_mask.
    Returns (orange_mask shaped like the lens .npy box, median_skull_surface_x).
    """
    nx_post = int(body_mask.shape[0])
    x_surf = _x_surf_from_body_mask(body_mask)
    valid = x_surf[x_surf < nx_post]
    skull_med = int(np.median(valid)) if valid.size > 0 else int(lens_x0)

    lx = lens_x1 - lens_x0
    ly_box = design_j_hi - design_j_lo
    lz_box = design_k_hi - design_k_lo
    m = np.zeros((lx, ly_box, lz_box), dtype=bool)
    x_surf_slice = x_surf[design_j_lo:design_j_hi, design_k_lo:design_k_hi]
    ii = np.arange(lens_x0, lens_x1, dtype=np.float64)[:, None, None]
    has_tissue = x_surf_slice < nx_post
    in_front = ii < x_surf_slice
    left_edge = ii == lens_x0
    m[:] = (in_front | left_edge) & has_tissue[None, :, :]
    return m, skull_med


def _resolve_reference_patient_dir(
    batch_root: Path,
    reference_id: str,
    explicit: Optional[Path],
) -> Optional[Path]:
    if explicit is not None and explicit.is_dir():
        return explicit
    for pid, pdir in _find_patient_dirs(batch_root):
        if pid == reference_id:
            return pdir
    # Loose match (folder names sometimes differ in punctuation)
    ref_norm = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in reference_id)
    for pid, pdir in _find_patient_dirs(batch_root):
        if "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pid) == ref_norm:
            return pdir
    return None


def try_rebuild_orange_mask_from_reference(
    lens_dir: Path,
    lens_c: np.ndarray,
    lens_meta: dict,
    train_summ: dict,
    batch_root: Path,
    reference_id: str,
    explicit_ref_dir: Optional[Path],
    save_to_lens_dir: bool,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Rebuild ``lens_orange_mask.npy`` geometry using the reference patient's
    body mask after the same preprocessing as training.  Does not require
    re-running pol6.
    """
    lens_x0 = int(lens_meta["lens_x0"])
    lens_x1 = int(lens_meta["lens_x1"])
    dx_mm = float(lens_meta["dx_mm"])

    dj_lo = lens_meta.get("design_j_lo")
    dj_hi = lens_meta.get("design_j_hi")
    dk_lo = lens_meta.get("design_k_lo")
    dk_hi = lens_meta.get("design_k_hi")
    if None in (dj_lo, dj_hi, dk_lo, dk_hi):
        yz = train_summ.get("lens_design_yz_range") or []
        if len(yz) >= 4:
            dj_lo, dj_hi, dk_lo, dk_hi = (int(yz[0]), int(yz[1]), int(yz[2]), int(yz[3]))
        else:
            print("[lens] Cannot rebuild orange mask: lens_meta / summary lack design y/z bounds.")
            return None, int(lens_meta.get("skull_surface_x_train", lens_x0))

    dj_lo, dj_hi, dk_lo, dk_hi = int(dj_lo), int(dj_hi), int(dk_lo), int(dk_hi)

    ref_dir = _resolve_reference_patient_dir(batch_root, reference_id, explicit_ref_dir)
    if ref_dir is None:
        print(
            f"[lens] Cannot rebuild orange mask: reference patient {reference_id!r} not found under "
            f"{batch_root}. Set REFERENCE_DIR_FOR_ORANGE_MASK to that patient's folder."
        )
        return None, int(lens_meta.get("skull_surface_x_train", lens_x0))

    ref_load_id = _infer_patient_id_for_dir(ref_dir)
    if ref_load_id != ref_dir.name:
        print(
            f"[lens] Reference folder {ref_dir.name!r} → map prefix patient_id {ref_load_id!r} "
            f"(from metadata.json or *_sound_speed_cleaned.npy)"
        )

    crop_x_mm     = 40 #float(train_summ.get("crop_x_mm_each_side", 40.0))
    crop_y_mm     = 30 #float(train_summ.get("crop_y_start_mm", 30.0))
    crop_z_mm     = 50 #float(train_summ.get("crop_z_start_mm", 100.0))
    crop_z_end_mm = 5 #float(train_summ.get("crop_z_end_mm", 5.0))
    crop_x_end_mm = 5 #float(train_summ.get("crop_x_end_mm", 5.0))
    flip_prop     = bool(train_summ.get("flip_propagation_axis", False))
    prop_axis     = int(train_summ.get("propagation_axis", 2))
    water_bath_vox = int(train_summ.get("water_bath_vox", 25))

    try:
        c_xyz, rho_xyz, att_xyz, bm0, spacing = _load_patient_maps(ref_dir, ref_load_id)
    except Exception as exc:
        print(f"[lens] Cannot rebuild orange mask: failed to load reference maps: {exc}")
        return None, int(lens_meta.get("skull_surface_x_train", lens_x0))

    if bm0 is None:
        print("[lens] Cannot rebuild orange mask: reference patient has no body_mask.npy.")
        return None, int(lens_meta.get("skull_surface_x_train", lens_x0))

    _, _, _, bm = preprocess_patient(
        c_xyz, rho_xyz, att_xyz, spacing,
        target_dx=dx_mm,
        crop_x_mm=crop_x_mm,
        crop_y_start_mm=crop_y_mm,
        crop_z_start_mm=crop_z_mm,
        crop_z_end_mm=crop_z_end_mm,
        crop_x_end_mm=crop_x_end_mm,
        flip_propagation=flip_prop,
        propagation_axis=prop_axis,
        water_bath_vox=water_bath_vox,
        body_mask=bm0,
    )

    orange, skull_x_train = build_orange_mask_like_pol6(
        bm, lens_x0, lens_x1, dj_lo, dj_hi, dk_lo, dk_hi,
    )

    exp = (lens_x1 - lens_x0, dj_hi - dj_lo, dk_hi - dk_lo)
    if orange.shape != exp or lens_c.shape != exp:
        print(
            f"[lens] Orange rebuild shape {orange.shape} / lens {lens_c.shape} "
            f"≠ expected {exp} — check lens_meta vs training summary."
        )
        return None, int(lens_meta.get("skull_surface_x_train", lens_x0))

    print(
        f"[lens] Rebuilt orange mask from {ref_dir.name!r} (patient_id={ref_load_id!r}), "
        f"{int(np.sum(orange)):,} vox, skull_surface_x_train={skull_x_train}"
    )

    if save_to_lens_dir:
        try:
            np.save(lens_dir / "lens_orange_mask.npy", orange.astype(np.bool_))
            meta_out = dict(lens_meta)
            meta_out["skull_surface_x_train"] = skull_x_train
            meta_out.setdefault("design_j_lo", dj_lo)
            meta_out.setdefault("design_j_hi", dj_hi)
            meta_out.setdefault("design_k_lo", dk_lo)
            meta_out.setdefault("design_k_hi", dk_hi)
            (lens_dir / "lens_meta.json").write_text(
                json.dumps(meta_out, indent=2), encoding="utf-8"
            )
            print(f"[lens] Wrote {lens_dir / 'lens_orange_mask.npy'} and updated lens_meta.json")
        except OSError as exc:
            print(f"[lens] Could not save rebuilt mask to lens dir: {exc}")

    return orange, skull_x_train


# ---------------------------------------------------------------------------
# ── Lens transplant ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def transplant_lens(
    c: np.ndarray,
    rho: np.ndarray,
    lens_c: np.ndarray,
    lens_rho: np.ndarray,
    lens_x0_train: int,
    lens_x1_train: int,
    skull_surface_x: int,
    water_bath_vox: int,
    orange_mask: Optional[np.ndarray] = None,
    skull_surface_x_train: Optional[int] = None,
    body_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Paste only the pre-skull holographic lens layer into (c, rho).

    Never writes lens material into target ``body_mask`` tissue: shifts the
    slab toward lower x (water) until paste avoids tissue, then clips any
    remaining overlap.

    Returns (c, rho, actual_x0, actual_x1, retreat_x_vox, tissue_clipped_vox).
    """
    nx, ny, nz = c.shape
    lx, ly, lz = lens_c.shape

    # ── skull-surface alignment ────────────────────────────────────────────
    skull_x_ref = skull_surface_x_train if skull_surface_x_train is not None else lens_x0_train
    delta_x = skull_surface_x - skull_x_ref
    x0 = lens_x0_train + delta_x        # shift training anchor to match this patient
    x0 = max(0, x0)

    x1 = x0 + lx
    x1_clamped = min(x1, nx)
    lx_used = x1_clamped - x0
    if lx_used <= 0:
        print(f"[warn] Transplant x-range [{x0}:{x1_clamped}] is empty (nx={nx}); skipping.")
        return c, rho, x0, x0, 0, 0

    # ── y / z centering ────────────────────────────────────────────────────
    cy, cz = ny // 2, nz // 2
    y0 = max(0, cy - ly // 2);  y1 = min(ny, y0 + ly);  y0 = max(0, y1 - ly)
    z0 = max(0, cz - lz // 2);  z1 = min(nz, z0 + lz);  z0 = max(0, z1 - lz)
    ly_used = y1 - y0
    lz_used = z1 - z0

    retreat_x_vox = 0
    x_floor = max(0, int(water_bath_vox))

    # Move lens toward water until orange/full-box paste does not hit tissue.
    if body_mask is not None and body_mask.shape == (nx, ny, nz):
        while True:
            x1_clamped = min(x0 + lx, nx)
            lx_used = x1_clamped - x0
            if lx_used <= 0:
                break
            if orange_mask is not None:
                om = orange_mask[:lx_used, :ly_used, :lz_used]
            else:
                om = np.ones((lx_used, ly_used, lz_used), dtype=bool)
            sub_b = body_mask[x0:x1_clamped, y0:y1, z0:z1]
            if not np.any(om & sub_b):
                break
            if x0 <= x_floor:
                break
            x0 -= 1
            retreat_x_vox += 1

    x1_clamped = min(x0 + lx, nx)
    lx_used = x1_clamped - x0
    if lx_used <= 0:
        print(f"[warn] Transplant x-range empty after retreat (nx={nx}); skipping.")
        return c, rho, x0, x0, retreat_x_vox, 0

    lc   = lens_c[:lx_used,  :ly_used, :lz_used]
    lrho = lens_rho[:lx_used, :ly_used, :lz_used]
    if orange_mask is not None:
        paste_mask = np.asarray(orange_mask[:lx_used, :ly_used, :lz_used], dtype=bool)
    else:
        paste_mask = np.ones((lx_used, ly_used, lz_used), dtype=bool)

    tissue_clipped_vox = 0
    if body_mask is not None and body_mask.shape == (nx, ny, nz):
        sub_b = body_mask[x0:x1_clamped, y0:y1, z0:z1]
        overlap = paste_mask & sub_b
        if np.any(overlap):
            tissue_clipped_vox = int(np.sum(overlap))
            paste_mask = paste_mask & (~sub_b)
            print(
                f"[lens] Clipped {tissue_clipped_vox} paste voxels inside body_mask "
                f"(after {retreat_x_vox} x-step retreat toward water)."
            )

    c   = c.copy()
    rho = rho.copy()
    region_c   = c[x0:x1_clamped, y0:y1, z0:z1]
    region_rho = rho[x0:x1_clamped, y0:y1, z0:z1]
    c[x0:x1_clamped,   y0:y1, z0:z1] = np.where(paste_mask, lc,   region_c)
    rho[x0:x1_clamped, y0:y1, z0:z1] = np.where(paste_mask, lrho, region_rho)

    n_pasted = int(np.sum(paste_mask))
    if orange_mask is None:
        mode_str = f"full-box → tissue-safe paste ({n_pasted} vox)"
    else:
        mode_str = f"orange-mask ({n_pasted} vox = {100*n_pasted/max(paste_mask.size,1):.1f}%)"

    retreat_note = f", retreated Δx={retreat_x_vox} vox" if retreat_x_vox else ""
    print(
        f"[lens] Skull ref x={skull_x_ref} → this patient x={skull_surface_x} "
        f"(shift={delta_x:+d} vox)\n"
        f"       Placed at x=[{x0}:{x1_clamped}], y=[{y0}:{y1}], z=[{z0}:{z1}]  "
        f"mode={mode_str}{retreat_note}"
    )
    return c, rho, x0, x1_clamped, retreat_x_vox, tissue_clipped_vox


# ---------------------------------------------------------------------------
# ── Helmholtz forward solve ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def run_forward_solve(
    c: np.ndarray,
    rho: np.ndarray,
    att: Optional[np.ndarray],
    dx_mm: float,
    frequency_hz: float,
    water_bath_vox: int,
    method: str = "gmres",
    tol: float = 2e-3,
    maxiter: int = 400,
    restart: int = 10,
) -> np.ndarray:
    """Run one Helmholtz solve and return |p| (float32 3D array)."""
    # Clamp unphysical values
    c   = np.where(c   < 1480.0, 1480.0, c).astype(np.float32)
    rho = np.where(rho < 1000.0, 1000.0, rho).astype(np.float32)

    nx, ny, nz = c.shape
    dx_m = dx_mm * 1e-3
    domain = Domain((nx, ny, nz), (dx_m, dx_m, dx_m))

    source_x = max(0, water_bath_vox - 2)
    source = _make_transducer_source(
        nxyz=(nx, ny, nz),
        dx_mm=dx_mm,
        domain=domain,
        frequency_hz=frequency_hz,
        source_x_slice=source_x,
    )

    omega = 2.0 * np.pi * frequency_hz
    c_j   = jnp.asarray(c,   dtype=jnp.float32)
    rho_j = jnp.asarray(rho, dtype=jnp.float32)

    if att is not None:
        att_np_m = att * frequency_hz / 1e6 / (20 / np.log(10)) * 100  # dB/cm/MHz → Np/m
        alpha_j = jnp.asarray(att_np_m.astype(np.float32))
        c_complex = c_j / (1.0 + 1j * alpha_j * c_j / omega)
        medium = Medium(domain=domain, sound_speed=FourierSeries(
            jnp.expand_dims(c_complex.real, -1), domain),
            density=FourierSeries(jnp.expand_dims(rho_j, -1), domain))
    else:
        medium = Medium(
            domain=domain,
            sound_speed=FourierSeries(jnp.expand_dims(c_j, -1), domain),
            density=FourierSeries(jnp.expand_dims(rho_j, -1), domain),
        )

    settings = {
        "method": method,
        "tol": tol,
        "maxiter": maxiter,
    }
    if method == "gmres":
        settings["restart"] = restart

    print(f"[solve] shape={c.shape}, dx={dx_mm} mm, freq={frequency_hz/1e3:.0f} kHz")
    t0 = time.time()
    pressure = helmholtz_solver(medium, omega, source, **settings)
    elapsed = time.time() - t0
    print(f"[solve] Done in {elapsed:.1f} s")

    p_abs = np.abs(np.array(pressure.on_grid[..., 0], dtype=np.complex64))
    return p_abs.astype(np.float32)


# ---------------------------------------------------------------------------
# ── Metrics ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def compute_focus_metrics(
    p_abs: np.ndarray,
    body_mask: Optional[np.ndarray],
    dx_mm: float,
    water_bath_vox: int,
    focus_radius_mm: float = 8.0,
    focal_center_vox: Optional[Tuple[int, int, int]] = None,
) -> dict:
    """
    Compute max-pressure location and its distance to the intended focal voxel.

    If ``focal_center_vox`` is None, the target is the domain centre (legacy).
    Otherwise it should be the reference training focal mapped onto this grid
    (see ``map_reference_focal_to_patient``).

    All positions reported in mm from domain origin.
    """
    nx, ny, nz = p_abs.shape
    if focal_center_vox is None:
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        focal_mapping = "domain_centre"
    else:
        cx = int(np.clip(focal_center_vox[0], 0, nx - 1))
        cy = int(np.clip(focal_center_vox[1], 0, ny - 1))
        cz = int(np.clip(focal_center_vox[2], 0, nz - 1))
        focal_mapping = "reference_mapped"
    focal_mm = np.array([cx, cy, cz], dtype=float) * dx_mm

    # Build a focus sphere mask (center = domain center) in voxel coordinates.
    focus_r_vox = float(focus_radius_mm) / float(dx_mm)
    xi = np.arange(nx, dtype=np.float32)[:, None, None]
    yi = np.arange(ny, dtype=np.float32)[None, :, None]
    zi = np.arange(nz, dtype=np.float32)[None, None, :]
    dist2 = (xi - float(cx)) ** 2 + (yi - float(cy)) ** 2 + (zi - float(cz)) ** 2
    focus_mask = dist2 <= (focus_r_vox ** 2)

    # "Inside actual CT-scan": use body_mask if present (water bath is 0 there),
    # otherwise just exclude water-bath region by x index.
    if body_mask is not None:
        valid_mask = focus_mask & body_mask
    else:
        valid_mask = focus_mask & (np.arange(nx)[:, None, None] >= water_bath_vox)

    if not np.any(valid_mask):
        # Fallback: search inside focus sphere but ignore body_mask.
        valid_mask = focus_mask
        print("[warn] focus+body_mask ROI empty; falling back to focus sphere only.")

    # Max pressure restricted to the focus region only.
    roi = np.where(valid_mask, p_abs, -np.inf)
    flat_idx = int(np.argmax(roi))
    xi_m, yi_m, zi_m = np.unravel_index(flat_idx, roi.shape)

    max_loc_vox = np.array([xi_m, yi_m, zi_m], dtype=float)
    max_loc_mm = max_loc_vox * dx_mm
    focus_err_mm = float(np.linalg.norm(max_loc_mm - focal_mm))

    return {
        "focal_center_vox":  [int(cx), int(cy), int(cz)],
        "focal_center_mm":   [float(v) for v in focal_mm],
        "focal_mapping":     focal_mapping,
        "max_pressure_vox":  [int(xi_m), int(yi_m), int(zi_m)],
        "max_pressure_mm":   [float(v) for v in max_loc_mm],
        "focus_error_mm":    focus_err_mm,
        "max_pressure_val":  float(np.max(roi[np.isfinite(roi)])),
        "focus_radius_mm":   float(focus_radius_mm),
    }


# ---------------------------------------------------------------------------
# ── Visualisation ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _pol6_cmap_bad(name: str):
    cm = plt.get_cmap(name).copy()
    cm.set_bad("black")
    return cm


def _pol6_add_boundary_full_domain(
    ax, water_bath_boundary_mm: float, ct_end_mm: float, col_idx: int,
) -> None:
    """Match pol6 plot_slices: water|CT and CT end on XY/XZ only."""
    if col_idx >= 2:
        return
    ax.axvline(water_bath_boundary_mm, color="blue", linewidth=2.0, linestyle="--", alpha=1.0)
    ax.axvline(ct_end_mm, color="red", linewidth=2.0, linestyle="--", alpha=1.0)
    ylo, yhi = ax.get_ylim()
    ymid_ax = (ylo + yhi) * 0.5
    ax.text(
        water_bath_boundary_mm / 2.0, ymid_ax, "water",
        color="blue", ha="center", va="center", fontsize=8,
    )
    ax.text(
        water_bath_boundary_mm + (ct_end_mm - water_bath_boundary_mm) / 2.0, ymid_ax, "CT",
        color="gray", ha="center", va="center", fontsize=8,
    )


def _pol6_add_design_subbox(
    ax,
    col_idx: int,
    design_region_vox: Tuple[int, int, int, int, int, int],
    dx_mm: float,
) -> None:
    """Lens design region as orange rectangle (paste / design vox bounds), pol6-style."""
    dx0, dx1, dj_lo, dj_hi, dk_lo, dk_hi = design_region_vox
    design_start_x = dx0 * dx_mm
    design_end_x = dx1 * dx_mm
    design_len_x = design_end_x - design_start_x
    design_y_lo = dj_lo * dx_mm
    design_y_hi = dj_hi * dx_mm
    design_z_lo = dk_lo * dx_mm
    design_z_hi = dk_hi * dx_mm

    if col_idx == 0:
        patch = Rectangle(
            (design_start_x, design_y_lo), design_len_x, design_y_hi - design_y_lo,
            fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0,
        )
    elif col_idx == 1:
        patch = Rectangle(
            (design_start_x, design_z_lo), design_len_x, design_z_hi - design_z_lo,
            fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0,
        )
    else:
        patch = Rectangle(
            (design_y_lo, design_z_lo), design_y_hi - design_y_lo, design_z_hi - design_z_lo,
            fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0,
        )
    ax.add_patch(patch)


def _pol6_add_reference_point(ax, col_idx: int, dx_mm: float, xmid_full: int, ymid: int, zmid: int) -> None:
    """Red circle at focal voxel (pol6-style indices: nx/1.5, ny/1.5, nz/2 on training grid)."""
    center_x_mm = (xmid_full + 0.5) * dx_mm
    center_y_mm = (ymid + 0.5) * dx_mm
    center_z_mm = (zmid + 0.5) * dx_mm
    if col_idx == 0:
        cx, cy = center_x_mm, center_y_mm
    elif col_idx == 1:
        cx, cy = center_x_mm, center_z_mm
    else:
        cx, cy = center_y_mm, center_z_mm
    ax.add_patch(
        Circle((cx, cy), 4.0, fill=False, edgecolor="red", linewidth=2.0, transform=ax.transData)
    )


def plot_patient_result(
    p_abs: np.ndarray,
    c: np.ndarray,
    metrics: dict,
    dx_mm: float,
    water_bath_vox: int,
    patient_id: str,
    out_path: Path,
    reference_id: str = "",
    skull_surface_x: int = 0,
    actual_lens_x0: int = 0,
    actual_lens_x1: int = 0,
    design_region_vox: Optional[Tuple[int, int, int, int, int, int]] = None,
):
    nx, ny, nz = p_abs.shape
    cx, cy, cz = metrics["focal_center_vox"]
    mx, my, mz = metrics["max_pressure_vox"]

    if design_region_vox is None:
        dj_lo = max(0, int(round(0.125 * ny)))
        dj_hi = min(ny, int(round(0.875 * ny)))
        dk_lo = max(0, int(round(0.125 * nz)))
        dk_hi = min(nz, int(round(0.875 * nz)))
        design_region_vox = (actual_lens_x0, actual_lens_x1, dj_lo, dj_hi, dk_lo, dk_hi)

    # Slices through the focal centre
    sl_xy = p_abs[cx, :, :]     # fix x=focal_cx, show y vs z
    sl_xz = p_abs[:, cy, :]     # fix y=focal_cy, show x vs z
    sl_yz = p_abs[:, :, cz]     # fix z=focal_cz, show x vs y

    # Matching sound-speed slices for lens visualization
    sl_xy_c = c[cx, :, :]
    sl_xz_c = c[:, cy, :]
    sl_yz_c = c[:, :, cz]

    vmax_p = float(np.nanpercentile(p_abs, 99.5))
    vmin_c = float(np.nanpercentile(c, 1.0))
    vmax_c = float(np.nanpercentile(c, 99.0))

    # Rows 0–1: focal |p| and c. Rows 2–4: pol6 figure rows 2, 4, 5 (1-based):
    #   |P| full-domain log, P_rms full-domain log, c full-domain (mid-planes).
    fig, axes = plt.subplots(5, 3, figsize=(18, 26))
    focal_note = ""
    if metrics.get("focal_mapping") == "domain_centre":
        focal_note = (
            "\nReference focal geometry not loaded — green/red markers use volume centre "
            "(check summary input_dir basename vs PATIENT_DIR; batch folder must load for pol6 focal)."
        )
    fig.suptitle(
        f"Patient: {patient_id}  |  Focus error: {metrics['focus_error_mm']:.1f} mm  "
        f"(reference: {reference_id})"
        f"{focal_note}",
        fontsize=12,
        fontweight="bold",
    )

    focus_r_mm = float(metrics.get("focus_radius_mm", 8.0))
    slices_info = [
        (sl_xy,   sl_xy_c, "YZ — slice at x=focal",    my, mz, cy, cz, "y (mm)", "z (mm)"),
        (sl_xz,   sl_xz_c, "XZ — slice at y=focal",    mx, mz, cx, cz, "x (mm)", "z (mm)"),
        (sl_yz,   sl_yz_c, "XY — slice at z=focal",    mx, my, cx, cy, "x (mm)", "y (mm)"),
    ]

    for col, (sl_p, sl_c, title, dot_a, dot_b, tgt_a, tgt_b, xlabel, ylabel) in enumerate(slices_info):
        extent_a = sl_p.shape[0] * dx_mm
        extent_b = sl_p.shape[1] * dx_mm

        # --- Row 0: |p| ---
        axp = axes[0, col]
        im_p = axp.imshow(
            sl_p.T, origin="lower", cmap="hot", vmin=0, vmax=vmax_p,
            extent=[0, extent_a, 0, extent_b], aspect="equal",
        )
        plt.colorbar(im_p, ax=axp, fraction=0.046, pad=0.04, label="|p| (Pa)")

        # Focal target circle (green dashed)
        focal_circle = plt.Circle(
            (tgt_a * dx_mm, tgt_b * dx_mm),
            radius=focus_r_mm,
            color="lime", fill=False, linestyle="--", linewidth=1.5, label="focal target",
        )
        axp.add_patch(focal_circle)

        # Max pressure dot (blue)
        axp.plot(dot_a * dx_mm, dot_b * dx_mm, "bo", markersize=8, label="max |p|")

        # Water bath boundary, skull surface, and lens region (only meaningful on x-axis)
        if xlabel.startswith("x"):
            axp.axvline(water_bath_vox * dx_mm, color="cyan", linestyle="--",
                        linewidth=1, label="water|CT")
            if skull_surface_x > water_bath_vox:
                axp.axvline(skull_surface_x * dx_mm, color="orange", linestyle=":",
                            linewidth=1.2, label="skull surface")
            axp.axvspan(actual_lens_x0 * dx_mm, actual_lens_x1 * dx_mm,
                        alpha=0.12, color="yellow", label="lens region")

        axp.set_title(title + "  |p|", fontsize=9)
        axp.set_xlabel(xlabel)
        axp.set_ylabel(ylabel)
        axp.legend(fontsize=7, loc="upper right")

        # --- Row 1: c (m/s) ---
        axc = axes[1, col]
        im_c = axc.imshow(
            sl_c.T, origin="lower", cmap="plasma", vmin=vmin_c, vmax=vmax_c,
            extent=[0, extent_a, 0, extent_b], aspect="equal",
        )
        plt.colorbar(im_c, ax=axc, fraction=0.046, pad=0.04, label="c (m/s)")
        axc.set_title(title + "  c", fontsize=9)
        axc.set_xlabel(xlabel)
        axc.set_ylabel(ylabel)

        if xlabel.startswith("x"):
            axc.axvline(water_bath_vox * dx_mm, color="cyan", linestyle="--",
                        linewidth=1)
            if skull_surface_x > water_bath_vox:
                axc.axvline(skull_surface_x * dx_mm, color="orange", linestyle=":")
            # Thin lens box edges only on c row — avoids yellow wash overlapping plasma.
            axc.axvline(actual_lens_x0 * dx_mm, color="gold", linestyle="-",
                        linewidth=1.0, alpha=0.85)
            axc.axvline(actual_lens_x1 * dx_mm, color="gold", linestyle="-",
                        linewidth=1.0, alpha=0.85)

    # --- pol6-style mid-plane rows: slices through the focal target (mapped), like pol6 focal indices ---
    fx = int(np.clip(cx, 0, nx - 1))
    fy = int(np.clip(cy, 0, ny - 1))
    fz = int(np.clip(cz, 0, nz - 1))
    names = ["XY", "XZ", "YZ"]
    extent_full_xy = [0, nx * dx_mm, 0, ny * dx_mm]
    extent_full_xz = [0, nx * dx_mm, 0, nz * dx_mm]
    extent_full_yz = [0, ny * dx_mm, 0, nz * dx_mm]
    extents_full = [extent_full_xy, extent_full_xz, extent_full_yz]

    planes_full = [
        p_abs[:, :, fz].T,
        p_abs[:, fy, :].T,
        p_abs[fx, :, :].T,
    ]
    planes_full_rms = [
        (p_abs[:, :, fz] / np.sqrt(2.0)).T,
        (p_abs[:, fy, :] / np.sqrt(2.0)).T,
        (p_abs[fx, :, :] / np.sqrt(2.0)).T,
    ]
    planes_c_full = [
        c[:, :, fz].T,
        c[:, fy, :].T,
        c[fx, :, :].T,
    ]

    finite_full = p_abs[np.isfinite(p_abs)]
    vmax_full = float(np.percentile(finite_full, 99.5)) if finite_full.size > 0 else 1.0
    vmax_full = max(vmax_full, 1e-12)
    vmin_log_full = max(1e-12, vmax_full / 1e5)
    norm_full = LogNorm(vmin=vmin_log_full, vmax=vmax_full)
    norm_rms_full = LogNorm(vmin=vmin_log_full / np.sqrt(2.0), vmax=vmax_full / np.sqrt(2.0))

    cmap_mag = _pol6_cmap_bad("inferno")
    cmap_rms = _pol6_cmap_bad("hot")

    water_bath_boundary_mm = float(water_bath_vox * dx_mm)
    ct_end_mm = float(nx * dx_mm)

    for i in range(3):
        # Row 2 (pol6 1-based): |P| full domain, bath|CT
        ax_m = axes[2, i]
        im_m = ax_m.imshow(
            np.where(planes_full[i] > 0, planes_full[i], np.nan),
            origin="lower", cmap=cmap_mag, norm=norm_full,
            extent=extents_full[i],
        )
        plt.colorbar(im_m, ax=ax_m, fraction=0.046, label="|P| (Pa)")
        ax_m.set_title(f"|P| Pa (log) {names[i]}  (full domain, bath|CT)", fontsize=9)
        ax_m.set_xlabel("mm")
        ax_m.set_ylabel("mm")
        _pol6_add_boundary_full_domain(ax_m, water_bath_boundary_mm, ct_end_mm, i)
        _pol6_add_design_subbox(ax_m, i, design_region_vox, dx_mm)
        _pol6_add_reference_point(ax_m, i, dx_mm, fx, fy, fz)

        # Row 4 (pol6 1-based): P_rms full domain, lens
        ax_r = axes[3, i]
        im_r = ax_r.imshow(
            np.where(planes_full_rms[i] > 0, planes_full_rms[i], np.nan),
            origin="lower", cmap=cmap_rms, norm=norm_rms_full,
            extent=extents_full[i],
        )
        plt.colorbar(im_r, ax=ax_r, fraction=0.046, label="P_rms (Pa)")
        ax_r.set_title(f"P_rms Pa (log) {names[i]}  (full domain, lens)", fontsize=9)
        ax_r.set_xlabel("mm")
        ax_r.set_ylabel("mm")
        _pol6_add_boundary_full_domain(ax_r, water_bath_boundary_mm, ct_end_mm, i)
        _pol6_add_design_subbox(ax_r, i, design_region_vox, dx_mm)
        _pol6_add_reference_point(ax_r, i, dx_mm, fx, fy, fz)

        # Row 5 (pol6 1-based): c m/s
        ax_cf = axes[4, i]
        c_min, c_max = float(np.min(c)), float(np.max(c))
        c_water = 1480.0
        norm_c = plt.Normalize(vmin=c_water - 100, vmax=min(c_max + 50, 4500.0))
        cmap_c = plt.get_cmap("viridis").copy()
        cmap_c.set_bad("black")
        im_cf = ax_cf.imshow(
            planes_c_full[i], origin="lower", cmap=cmap_c, norm=norm_c,
            extent=extents_full[i],
        )
        plt.colorbar(im_cf, ax=ax_cf, fraction=0.046, label="c (m/s)")
        ax_cf.set_title(f"c m/s {names[i]}  (lens = higher c)", fontsize=9)
        ax_cf.set_xlabel("mm")
        ax_cf.set_ylabel("mm")
        _pol6_add_boundary_full_domain(ax_cf, water_bath_boundary_mm, ct_end_mm, i)
        _pol6_add_design_subbox(ax_cf, i, design_region_vox, dx_mm)
        _pol6_add_reference_point(ax_cf, i, dx_mm, fx, fy, fz)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out_path.name}")


def plot_summary_chart(results: list[dict], out_path: Path, reference_id: str):
    """Bar chart of focus error for all patients."""
    ids    = [r["patient_id"] for r in results]
    errors = [r["focus_error_mm"] for r in results]
    colors = ["green" if pid == reference_id else "steelblue" for pid in ids]

    fig, ax = plt.subplots(figsize=(max(10, len(ids) * 0.6), 5))
    bars = ax.bar(range(len(ids)), errors, color=colors)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Focus error (mm)")
    ax.set_title(f"Focus error — lens optimised on {reference_id}")
    ax.axhline(0, color="black", linewidth=0.5)

    legend_handles = [
        mpatches.Patch(color="green",     label=f"Reference ({reference_id})"),
        mpatches.Patch(color="steelblue", label="Other patients"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)

    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary chart → {out_path.name}")


# ---------------------------------------------------------------------------
# ── Patient directory discovery ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _find_patient_dirs(batch_root: Path) -> list[Tuple[str, Path]]:
    """Return (patient_id, patient_dir) for every sub-folder with metadata.json."""
    patients = []
    for sub in sorted(batch_root.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "metadata.json").exists():
            continue
        patients.append((sub.name, sub))
    return patients


def _infer_patient_id_for_dir(patient_dir: Path) -> str:
    """
    File prefix for cleaned .npy maps (e.g. C3L-03260_) may differ from the folder
    name (e.g. C3L_npy_files).  Prefer metadata.json patient_id, else a single
    *_sound_speed_cleaned.npy match, else fall back to folder name.
    """
    meta_path = patient_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pid = meta.get("patient_id")
            if pid:
                return str(pid)
        except (json.JSONDecodeError, OSError):
            pass
    matches = sorted(patient_dir.glob("*_sound_speed_cleaned.npy"))
    if len(matches) == 1:
        return matches[0].name[: -len("_sound_speed_cleaned.npy")]
    if len(matches) > 1:
        print(
            f"[warn] Multiple *_sound_speed_cleaned.npy in {patient_dir}; "
            f"using {matches[0].name}"
        )
        return matches[0].name[: -len("_sound_speed_cleaned.npy")]
    return patient_dir.name


def _load_patient_maps(
    patient_dir: Path,
    patient_id: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
           Tuple[float, float, float]]:
    """Load cleaned maps and metadata from a batch output directory."""
    meta = json.loads((patient_dir / "metadata.json").read_text())
    spacing = tuple(float(v) for v in meta["spacing_xyz_mm"])

    # Prefer cleaned maps; fall back to raw
    def _load_field(primary: Path, fallback: Path) -> np.ndarray:
        p = patient_dir / primary
        if p.exists():
            return np.load(p).astype(np.float32)
        f = patient_dir / fallback
        if f.exists():
            return np.load(f).astype(np.float32)
        raise FileNotFoundError(f"Neither {primary} nor {fallback} found in {patient_dir}")

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in patient_id)

    c_xyz   = _load_field(f"{safe}_sound_speed_cleaned.npy",   "sound_speed_xyz.npy")
    rho_xyz = _load_field(f"{safe}_density_cleaned.npy",       "density_xyz.npy")

    att_xyz = None
    for name in (f"{safe}_attenuation_cleaned.npy", "attenuation_db_cm_mhz_xyz.npy"):
        p = patient_dir / name
        if p.exists():
            att_xyz = np.load(p).astype(np.float32)
            break

    bm = None
    for name in (f"{safe}_body_mask.npy", "body_mask.npy"):
        p = patient_dir / name
        if p.exists():
            bm = np.load(p).astype(bool)
            break

    return c_xyz, rho_xyz, att_xyz, bm, spacing  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ── Main ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Lens Transplant Evaluation")
    print(f"  Lens dir   : {LENS_DIR}")
    print(f"  Patient dir: {PATIENT_DIR}")
    print(f"  Output dir : {OUT_DIR}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load lens artefacts
    # ------------------------------------------------------------------
    lens_c   = np.load(LENS_DIR / "lens_sound_speed_ms.npy").astype(np.float32)
    lens_rho = np.load(LENS_DIR / "lens_density_kgm3.npy").astype(np.float32)
    lens_meta  = json.loads((LENS_DIR / "lens_meta.json").read_text())
    train_summ = json.loads((LENS_DIR / "summary.json").read_text())

    lens_x0 = int(lens_meta["lens_x0"])
    lens_x1 = int(lens_meta["lens_x1"])
    dx_mm   = float(lens_meta["dx_mm"])

    # Training summary stores ``input_dir`` as the *patient folder* (maps + metadata),
    # not its parent — use .name so it matches batch sub-folder names under PATIENT_DIR.
    _input_dir = Path(train_summ.get("input_dir", "unknown"))
    reference_id = _input_dir.name or _input_dir.parent.name or "reference"

    # Median skull-surface x-position in the training domain (updated if mask rebuilt).
    skull_surface_x_train = int(lens_meta.get("skull_surface_x_train", lens_x0))

    # Orange polygon mask: pre-skull coupling layer only (matches pol6 geometry).
    orange_mask_path = LENS_DIR / "lens_orange_mask.npy"
    if orange_mask_path.exists():
        orange_mask = np.load(orange_mask_path).astype(bool)
        print(f"[lens] Orange mask loaded: shape={orange_mask.shape}, "
              f"active={int(np.sum(orange_mask))} vox "
              f"({100*np.sum(orange_mask)/max(orange_mask.size,1):.1f}%)")
    else:
        rebuilt, sk_tr = try_rebuild_orange_mask_from_reference(
            LENS_DIR,
            lens_c,
            lens_meta,
            train_summ,
            PATIENT_DIR,
            reference_id,
            REFERENCE_DIR_FOR_ORANGE_MASK,
            SAVE_REBUILT_ORANGE_TO_LENS_DIR,
        )
        orange_mask = rebuilt
        if rebuilt is not None:
            skull_surface_x_train = sk_tr
        if orange_mask is None:
            print(
                "[lens] lens_orange_mask.npy missing and rebuild failed — "
                "full-box paste (include reference patient in PATIENT_DIR or set "
                "REFERENCE_DIR_FOR_ORANGE_MASK, with body_mask.npy)."
            )

    # Extract all preprocessing parameters from the training summary.
    # Defaults match pol6wattenuation_trainingv2.py argparse defaults exactly.
    crop_x_mm     = 40 #float(train_summ.get("crop_x_mm_each_side", 40.0))
    crop_y_mm     = 30 #float(train_summ.get("crop_y_start_mm", 30.0))
    crop_z_mm     = 50 #float(train_summ.get("crop_z_start_mm", 100.0))
    crop_z_end_mm = 5 #float(train_summ.get("crop_z_end_mm", 5.0))
    crop_x_end_mm = 5 #float(train_summ.get("crop_x_end_mm", 5.0))
    flip_prop        = bool(train_summ.get("flip_propagation_axis",  False))
    prop_axis        = int(train_summ.get("propagation_axis",            2))
    water_bath_vox   = int(train_summ.get("water_bath_vox",            25))
    frequency_hz     = float(train_summ.get("frequency_hz",        250_000))
    # Read solver settings from summary so they always match the training run.
    solver_method    = str(train_summ.get("method",  SOLVER_METHOD))
    solver_tol       = float(train_summ.get("tol",   SOLVER_TOL))
    solver_maxiter   = int(train_summ.get("maxiter", SOLVER_MAXITER))
    solver_restart   = int(train_summ.get("restart", SOLVER_RESTART))

    print(f"\nLens shape    : {lens_c.shape}")
    print(f"dx            : {dx_mm} mm")
    print(f"Frequency     : {frequency_hz/1e3:.0f} kHz")
    print(f"Water bath    : {water_bath_vox} vox")
    print(f"Skull ref x   : {skull_surface_x_train} vox (training)")
    print(f"Reference     : {reference_id}")
    print(f"Prop axis     : {prop_axis}, flip={flip_prop}")
    print(f"Crops (mm)    : x±{crop_x_mm}, y_start={crop_y_mm}, z_start={crop_z_mm}, "
          f"z_end={crop_z_end_mm}, x_end={crop_x_end_mm}")
    print(f"Solver        : {solver_method}, tol={solver_tol}, "
          f"maxiter={solver_maxiter}, restart={solver_restart}")

    # ------------------------------------------------------------------
    # 2. Discover patients
    # ------------------------------------------------------------------
    patients = _find_patient_dirs(PATIENT_DIR)
    if not patients:
        print(f"\n[!] No patient sub-folders with metadata.json found in {PATIENT_DIR}")
        sys.exit(1)

    print(f"\nFound {len(patients)} patient(s):")
    for pid, _ in patients:
        print(f"  {pid}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2b. Reference focal (training grid) for cross-patient mapping
    # ------------------------------------------------------------------
    ref_geom = load_reference_focal_geometry(
        PATIENT_DIR,
        reference_id,
        REFERENCE_DIR_FOR_ORANGE_MASK,
        dx_mm=dx_mm,
        crop_x_mm=crop_x_mm,
        crop_y_mm=crop_y_mm,
        crop_z_mm=crop_z_mm,
        crop_z_end_mm=crop_z_end_mm,
        crop_x_end_mm=crop_x_end_mm,
        flip_prop=flip_prop,
        prop_axis=prop_axis,
        water_bath_vox=water_bath_vox,
    )

    # ------------------------------------------------------------------
    # 3. Evaluate each patient
    # ------------------------------------------------------------------
    all_results = []
    failed = []

    for patient_id, patient_dir in patients:
        print(f"\n{'─'*60}")
        print(f"Patient: {patient_id}")

        pat_out = OUT_DIR / patient_id
        pat_out.mkdir(parents=True, exist_ok=True)

        try:
            # 3a. Load
            c_xyz, rho_xyz, att_xyz, body_mask, spacing = _load_patient_maps(
                patient_dir, patient_id
            )
            print(f"  Loaded: shape={c_xyz.shape}, spacing={spacing} mm")

            # 3b. Preprocess (same pipeline as training run)
            c, rho, att, bm = preprocess_patient(
                c_xyz, rho_xyz, att_xyz, spacing,
                target_dx=dx_mm,
                crop_x_mm=crop_x_mm,
                crop_y_start_mm=crop_y_mm,
                crop_z_start_mm=crop_z_mm,
                crop_z_end_mm=crop_z_end_mm,
                crop_x_end_mm=crop_x_end_mm,
                flip_propagation=flip_prop,
                propagation_axis=prop_axis,
                water_bath_vox=water_bath_vox,
                body_mask=body_mask,
            )
            print(f"  After preprocessing: shape={c.shape}")

            # 3c. Detect skull surface for this patient, then transplant lens
            skull_x = find_skull_surface_x(bm, water_bath_vox)
            skull_shift = skull_x - water_bath_vox
            print(
                f"  Skull surface at x={skull_x} "
                f"({'same as training' if skull_shift == 0 else f'shift={skull_shift:+d} vox = {skull_shift*dx_mm:+.1f} mm'})"
            )
            c, rho, actual_lens_x0, actual_lens_x1, lens_retreat_x, lens_tissue_clip = transplant_lens(
                c, rho, lens_c, lens_rho, lens_x0, lens_x1,
                skull_surface_x=skull_x,
                water_bath_vox=water_bath_vox,
                orange_mask=orange_mask,
                skull_surface_x_train=skull_surface_x_train,
                body_mask=bm,
            )

            # 3d. Forward solve
            p_abs = run_forward_solve(
                c, rho, att, dx_mm, frequency_hz, water_bath_vox,
                method=solver_method, tol=solver_tol,
                maxiter=solver_maxiter, restart=solver_restart,
            )

            # 3e. Metrics — focal target mapped from reference (or domain centre)
            if ref_geom is not None:
                fx, fy, fz = map_reference_focal_to_patient(
                    ref_geom["cx_ref"],
                    ref_geom["cy_ref"],
                    ref_geom["cz_ref"],
                    ref_geom["skull_x_ref"],
                    ref_geom["ny_ref"],
                    ref_geom["nz_ref"],
                    skull_x,
                    c.shape[0],
                    c.shape[1],
                    c.shape[2],
                )
                focal_vox: Optional[Tuple[int, int, int]] = (fx, fy, fz)
            else:
                focal_vox = None

            metrics = compute_focus_metrics(
                p_abs, bm, dx_mm, water_bath_vox,
                focal_center_vox=focal_vox,
            )
            metrics["patient_id"]        = patient_id
            metrics["skull_surface_x"]   = skull_x
            metrics["skull_shift_vox"]   = skull_shift
            metrics["skull_shift_mm"]    = round(skull_shift * dx_mm, 2)
            metrics["actual_lens_x0"]          = actual_lens_x0
            metrics["actual_lens_x1"]          = actual_lens_x1
            metrics["lens_retreat_x_vox"]      = lens_retreat_x
            metrics["lens_tissue_clipped_vox"] = lens_tissue_clip
            if ref_geom is not None:
                metrics["reference_focal_vox"]   = [
                    ref_geom["cx_ref"], ref_geom["cy_ref"], ref_geom["cz_ref"],
                ]
                metrics["reference_skull_x_vox"] = ref_geom["skull_x_ref"]
                metrics["reference_folder"]      = ref_geom["reference_folder"]
            print(
                f"  Max |p| @ vox {metrics['max_pressure_vox']}  "
                f"= {metrics['max_pressure_mm']} mm\n"
                f"  Focal target  @ vox {metrics['focal_center_vox']}  "
                f"= {metrics['focal_center_mm']} mm  ({metrics['focal_mapping']})\n"
                f"  Focus error   : {metrics['focus_error_mm']:.2f} mm"
            )

            # 3f. Plot (focal rows + pol6 rows 2/4/5: |P| full, P_rms full, c full)
            dj_lo = lens_meta.get("design_j_lo")
            dj_hi = lens_meta.get("design_j_hi")
            dk_lo = lens_meta.get("design_k_lo")
            dk_hi = lens_meta.get("design_k_hi")
            _ny, _nz = c.shape[1], c.shape[2]
            if None in (dj_lo, dj_hi, dk_lo, dk_hi):
                dj_lo = max(0, int(round(0.125 * _ny)))
                dj_hi = min(_ny, int(round(0.875 * _ny)))
                dk_lo = max(0, int(round(0.125 * _nz)))
                dk_hi = min(_nz, int(round(0.875 * _nz)))
            design_vox = (
                actual_lens_x0,
                actual_lens_x1,
                int(dj_lo),
                int(dj_hi),
                int(dk_lo),
                int(dk_hi),
            )
            plot_patient_result(
                p_abs, c, metrics, dx_mm, water_bath_vox,
                patient_id=patient_id,
                out_path=pat_out / "pressure_slices.png",
                reference_id=reference_id,
                skull_surface_x=skull_x,
                actual_lens_x0=actual_lens_x0,
                actual_lens_x1=actual_lens_x1,
                design_region_vox=design_vox,
            )

            # Save pressure field for later use
            np.save(pat_out / "p_abs.npy", p_abs)

            # 3g. Save per-patient JSON
            (pat_out / "results.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            all_results.append(metrics)

        except Exception as exc:
            import traceback
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            failed.append(patient_id)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Finished: {len(all_results)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed: {failed}")

    if all_results:
        plot_summary_chart(all_results, OUT_DIR / "focus_error_summary.png", reference_id)

        # Sort by focus error
        all_results.sort(key=lambda r: r["focus_error_mm"])
        print("\nFocus errors (sorted):")
        print(f"  {'Patient':<35} {'Error (mm)':>10}  {'Max |p|':>10}")
        print("  " + "-" * 60)
        for r in all_results:
            star = " ★" if r["patient_id"] == reference_id else ""
            print(
                f"  {r['patient_id']:<35} {r['focus_error_mm']:>10.2f}  "
                f"{r['max_pressure_val']:>10.4f}{star}"
            )

    summary = {
        "reference_patient": reference_id,
        "lens_dir":   str(LENS_DIR),
        "dx_mm":      dx_mm,
        "frequency_hz": frequency_hz,
        "n_patients": len(all_results),
        "failed":     failed,
        "results":    all_results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
