"""
3D frequency-domain (Helmholtz) planar-wave simulation through CT-brain maps.

This is a memory-light alternative to full time-domain propagation.
It solves a single-frequency complex pressure field using j-wave's Helmholtz solver.

Expected input maps:
  - sound_speed_xyz.npy
  - density_xyz.npy

Run (auto-discover latest cleaned maps):
  /Users/shanmukasadhu/miniconda3/envs/jwave/bin/python Code/j_wave_planar_ct_brain_3d_helmholtz.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

_J_WAVE_IMPORT_ERROR = None
try:
    from jwave import Domain, FourierSeries, helmholtz_solver
    from jwave.geometry import Medium
except Exception as _exc:  # pragma: no cover
    _J_WAVE_IMPORT_ERROR = _exc
    Domain = None
    FourierSeries = None
    helmholtz_solver = None
    Medium = None

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# ---------------------------------------------------------------------------
# Memory detection helpers
# ---------------------------------------------------------------------------

def _available_memory_bytes() -> int:
    """Return a conservative estimate of available memory (GPU VRAM if present, else CPU RAM)."""
    # GPU VRAM via nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
            if lines:
                free_mib = max(int(v) for v in lines)
                free_bytes = int(free_mib * 1024 * 1024 * 0.70)
                print(f"[info] GPU VRAM free: {free_mib} MiB → budget {free_bytes / 1e9:.2f} GB")
                return free_bytes
    except Exception:
        pass

    # psutil
    try:
        import psutil
        avail = psutil.virtual_memory().available
        print(f"[info] RAM available (psutil): {avail / 1e9:.2f} GB")
        return int(avail * 0.7)
    except Exception:
        pass

    # macOS vm_stat
    try:
        import subprocess, re
        out = subprocess.check_output(["vm_stat"], text=True, timeout=5)
        page_size = 4096
        m = re.search(r"page size of (\d+)", out)
        if m:
            page_size = int(m.group(1))
        free_pages = 0
        for label in ("Pages free", "Pages inactive", "Pages speculative"):
            for line in out.splitlines():
                if line.startswith(label):
                    mm = re.search(r":\s+(\d+)", line)
                    if mm:
                        free_pages += int(mm.group(1))
        avail = free_pages * page_size
        print(f"[info] RAM available (vm_stat): {avail / 1e9:.2f} GB")
        return int(avail * 0.7)
    except Exception:
        pass

    # os.sysconf fallback
    try:
        page_size = os.sysconf("SC_PAGESIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total = page_size * phys_pages
        avail = int(total * 0.4)
        print(f"[info] RAM total (os.sysconf): {total / 1e9:.2f} GB → budget {avail / 1e9:.2f} GB")
        return avail
    except Exception:
        pass

    print("[warn] Could not detect memory; assuming 8 GB budget.")
    return int(8e9)


# Helmholtz memory estimate: complex64 field + BiCGStab/GMRES work vectors + FFT buffers.
# Empirically calibrated: 171M voxels OOM'd on a 43 GB budget → actual peak > 251 B/vox.
# 350 B/vox gives a safe margin; XLA compilation buffers add several extra GB on top.
_HELMHOLTZ_BYTES_PER_VOXEL = 350


def _auto_voxel_budget() -> int:
    """Compute max_voxels from available memory using Helmholtz memory model."""
    avail = _available_memory_bytes()
    max_vox = int(avail / _HELMHOLTZ_BYTES_PER_VOXEL)
    print(f"[info] Auto voxel budget: {max_vox:,} ({avail / 1e9:.2f} GB / {_HELMHOLTZ_BYTES_PER_VOXEL} B/vox)")
    return max_vox


def _discover_input_maps(
    input_dir: Path | None,
    auto_root: Path,
    patient_id: str | None,
) -> Tuple[Path, Path, Path]:
    def _pair_from_cleaned(c_path: Path) -> Tuple[Path, Path]:
        stem = c_path.name.replace("_sound_speed_cleaned.npy", "")
        rho = c_path.with_name(f"{stem}_density_cleaned.npy")
        if not rho.exists():
            raise FileNotFoundError(f"Matching density file not found for {c_path}")
        return c_path, rho

    if input_dir is not None:
        c_std = input_dir / "sound_speed_xyz.npy"
        rho_std = input_dir / "density_xyz.npy"
        if c_std.exists() and rho_std.exists():
            return c_std, rho_std, input_dir

        if patient_id:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in patient_id)
            c_named = input_dir / f"{safe}_sound_speed_cleaned.npy"
            if c_named.exists():
                c_path, rho_path = _pair_from_cleaned(c_named)
                return c_path, rho_path, input_dir

        cleaned = sorted(input_dir.glob("*_sound_speed_cleaned.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cleaned:
            c_path, rho_path = _pair_from_cleaned(cleaned[0])
            return c_path, rho_path, input_dir

        raise FileNotFoundError(f"Could not find maps in {input_dir}")

    if not auto_root.exists():
        raise FileNotFoundError(f"Auto root does not exist: {auto_root}")

    if patient_id:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in patient_id)
        cand = sorted(auto_root.rglob(f"{safe}_sound_speed_cleaned.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cand:
            c_path, rho_path = _pair_from_cleaned(cand[0])
            return c_path, rho_path, cand[0].parent

    cleaned = sorted(auto_root.rglob("*_sound_speed_cleaned.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cleaned:
        c_path, rho_path = _pair_from_cleaned(cleaned[0])
        return c_path, rho_path, cleaned[0].parent

    std = sorted(auto_root.rglob("sound_speed_xyz.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c_path in std:
        rho_path = c_path.with_name("density_xyz.npy")
        if rho_path.exists():
            return c_path, rho_path, c_path.parent
    raise FileNotFoundError(f"No suitable maps found under {auto_root}")


def _discover_body_mask(c_path: Path) -> Optional[Path]:
    """Find body mask .npy next to the sound speed file."""
    stem = c_path.name.replace("_sound_speed_cleaned.npy", "")
    mask_path = c_path.with_name(f"{stem}_body_mask.npy")
    if mask_path.exists():
        return mask_path
    return None


def _load_spacing_from_metadata(input_dir: Path) -> Tuple[float, float, float]:
    meta_path = input_dir / "metadata.json"
    if meta_path.exists():
        print("Metadata file found...")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            spacing = meta.get("spacing_xyz_mm")
            if isinstance(spacing, list) and len(spacing) == 3:
                return float(spacing[0]), float(spacing[1]), float(spacing[2])
        except Exception:
            pass
    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        print("Summary file found...")
        try:
            meta = json.loads(summary_path.read_text(encoding="utf-8"))
            spacing = meta.get("ct_spacing_xyz_mm")
            if isinstance(spacing, list) and len(spacing) == 3:
                return float(spacing[0]), float(spacing[1]), float(spacing[2])
        except Exception:
            pass
    print("No metadata file found, falling back to 1mm isotropic.")
    return 1.0, 1.0, 1.0


# This function is used to resample the CT-brain map to an isotropic grid, 
# which means the spacing in all three dimensions is the same.
def _resample_to_isotropic(
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    spacing_xyz_mm: Tuple[float, float, float],
    target_dx_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import zoom
    sx, sy, sz = spacing_xyz_mm
    zf = (sx / target_dx_mm, sy / target_dx_mm, sz / target_dx_mm)
    c_iso = zoom(c_xyz, zf, order=1).astype(np.float32)
    rho_iso = zoom(rho_xyz, zf, order=1).astype(np.float32)
    return c_iso, rho_iso


def _resample_mask(
    mask: np.ndarray,
    spacing_xyz_mm: Tuple[float, float, float],
    target_dx_mm: float,
) -> np.ndarray:
    from scipy.ndimage import zoom
    sx, sy, sz = spacing_xyz_mm
    zf = (sx / target_dx_mm, sy / target_dx_mm, sz / target_dx_mm)
    # order=0 = nearest neighbour so mask stays binary
    mask_iso = zoom(mask.astype(np.float32), zf, order=0)
    return mask_iso > 0.5



def _crop_x_both_sides(
    arr_xyz: np.ndarray,
    dx_mm: float,
    crop_x_mm_each_side: float,
) -> Tuple[np.ndarray, int]:
    """
    Crop both sides along x by crop_x_mm_each_side.
    Returns cropped array and voxels cropped on each side.
    """
    if crop_x_mm_each_side <= 0:
        return arr_xyz, 0
    crop_vox = int(round(crop_x_mm_each_side / dx_mm))
    if crop_vox <= 0:
        return arr_xyz, 0
    nx = arr_xyz.shape[0]
    if 2 * crop_vox >= nx:
        raise ValueError(
            f"Requested x-crop too large: crop_vox={crop_vox} each side for nx={nx}."
        )
    return arr_xyz[crop_vox : nx - crop_vox, :, :], crop_vox


def _crop_axis_start(
    arr_xyz: np.ndarray,
    dx_mm: float,
    crop_mm: float,
    axis: int,
) -> Tuple[np.ndarray, int]:
    """
    Crop only from the start of a given axis.
    Returns cropped array and number of cropped voxels.
    """
    if crop_mm <= 0:
        return arr_xyz, 0
    crop_vox = int(round(crop_mm / dx_mm))
    if crop_vox <= 0:
        return arr_xyz, 0
    shape = arr_xyz.shape
    if crop_vox >= shape[axis]:
        raise ValueError(
            f"Requested start-crop too large: axis={axis}, crop_vox={crop_vox}, shape={shape}."
        )
    if axis == 0:
        return arr_xyz[crop_vox:, :, :], crop_vox
    if axis == 1:
        return arr_xyz[:, crop_vox:, :], crop_vox
    if axis == 2:
        return arr_xyz[:, :, crop_vox:], crop_vox
    raise ValueError(f"Invalid axis for cropping: {axis}")


def solve_helmholtz(
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    dx_mm: float,
    frequency_hz: float,
    source_x_slices: int,
    method: str,
    tol: float,
    restart: int,
    maxiter: int,
    checkpoint: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    # Clamp background/air voxels to water-like values.
    # CT-derived maps have c≈0 and ρ≈0 for air/background; these produce
    # k²=(ω/c)²→∞ which makes the Helmholtz system degenerate and causes
    # BiCGStab to "converge" immediately to a trivially near-zero solution.
    # Floor values: water coupling medium (c=1480 m/s, ρ=1000 kg/m³).
    C_WATER = 1480.0   # m/s
    RHO_WATER = 1000.0  # kg/m³
    n_low_c = int(np.sum(c_xyz < C_WATER))
    n_low_rho = int(np.sum(rho_xyz < RHO_WATER))
    if n_low_c > 0:
        print(f"[info] Clamping {n_low_c:,} voxels with c < {C_WATER} m/s to water ({C_WATER} m/s).")
        c_xyz = np.where(c_xyz < C_WATER, C_WATER, c_xyz)
    if n_low_rho > 0:
        print(f"[info] Clamping {n_low_rho:,} voxels with ρ < {RHO_WATER} kg/m³ to water ({RHO_WATER} kg/m³).")
        rho_xyz = np.where(rho_xyz < RHO_WATER, RHO_WATER, rho_xyz)

    nxyz = c_xyz.shape
    dx_m = dx_mm * 1e-3
    domain = Domain(nxyz, (dx_m, dx_m, dx_m))

    c_j = jnp.expand_dims(jnp.asarray(c_xyz, dtype=jnp.float32), -1)
    rho_j = jnp.expand_dims(jnp.asarray(rho_xyz, dtype=jnp.float32), -1)

    # PML: scale with domain; min 10, max 32, ~1/8 of smallest dimension.
    # Using the computed value (not hardcoded 12) is critical for proper absorption.
    pml = max(10, min(32, min(nxyz) // 8))
    medium = Medium(domain=domain, sound_speed=c_j, density=rho_j, pml_size=pml)

    # Real-valued unit source across the first source_x_slices planes.
    # Using -1.0j (pure imaginary) previously produced a 90° phase-shifted
    # forcing that confused the solver; a real source with PML boundaries
    # produces a coherent pseudo-traveling wave in the +x direction.
    src = jnp.zeros((*nxyz, 1), dtype=jnp.complex64)
    src = src.at[:source_x_slices, :, :, 0].set(1.0 + 0.0j)
    source = FourierSeries(src, domain)

    omega = 2.0 * jnp.pi * frequency_hz
    c_mean = float(jnp.mean(c_j))
    lam_mm = (c_mean / frequency_hz) * 1e3
    ppw = lam_mm / dx_mm
    print(f"[info] shape={nxyz}, dx={dx_mm:.3f} mm, freq={frequency_hz/1e3:.0f} kHz")
    print(f"[info] mean c={c_mean:.0f} m/s → λ={lam_mm:.2f} mm, PPW={ppw:.1f}")
    print(f"[info] PML={pml}, method={method}, tol={tol}, maxiter={maxiter}, restart={restart}")

    time_start = time.time()
    p_complex = helmholtz_solver(
        medium,
        omega,
        source,
        method=method,
        checkpoint=checkpoint,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
    )

    # Extract the scalar pressure field and immediately free the FourierSeries
    # wrapper (p_complex) — it holds the full complex64 field (~780 MB for 97 M vox)
    # and is no longer needed once we have p_on_grid.
    p_on_grid = p_complex.on_grid[..., 0]
    del p_complex

    # Compute abs and phase while p_on_grid is still on GPU, then free it.
    p_abs_jax   = jnp.abs(p_on_grid)
    p_phase_jax = jnp.angle(p_on_grid)
    del p_on_grid

    # Block until both results are ready, then free the GPU-side JAX arrays one
    # at a time so the peak VRAM spike is only one float32 field at a time.
    p_abs_jax.block_until_ready()
    p_phase_jax.block_until_ready()
    print(f"[ok] Solver done in {time.time() - time_start:.1f}s")

    # Transfer to CPU numpy and immediately release GPU memory.
    # With XLA_PYTHON_CLIENT_ALLOCATOR=platform, del + gc.collect() returns
    # memory to CUDA immediately rather than keeping it in JAX's cache.
    import gc
    p_abs = np.array(p_abs_jax, dtype=np.float32)
    del p_abs_jax
    p_phase = np.array(p_phase_jax, dtype=np.float32)
    del p_phase_jax
    # Free medium/source arrays that are no longer needed.
    del c_j, rho_j, medium, source
    gc.collect()

    print(f"[debug] p_abs min={p_abs.min():.3e}, max={p_abs.max():.3e}, mean={p_abs.mean():.3e}")
    return p_abs, p_phase


def _add_max_circle(ax, plane: np.ndarray, extent: list, radius_mm: float = 8.0) -> None:
    """Draw a circle around the peak pressure location on a 2D slice."""
    from matplotlib.patches import Circle
    finite = np.where(np.isfinite(plane), plane, -np.inf)
    idx = np.unravel_index(np.argmax(finite), finite.shape)
    # plane is (row, col) = (dim1, dim0) after .T, extent = [x0,x1,y0,y1]
    x0, x1, y0, y1 = extent
    col, row = idx  # after .T: row=col axis, col=row axis
    rows, cols = plane.shape
    cx = x0 + (row + 0.5) * (x1 - x0) / cols
    cy = y0 + (col + 0.5) * (y1 - y0) / rows
    circle = Circle((cx, cy), radius_mm, fill=False, edgecolor="cyan", linewidth=2.0, linestyle="--")
    ax.add_patch(circle)
    ax.plot(cx, cy, "+", color="cyan", markersize=10, markeredgewidth=2)
    ax.set_title(ax.get_title() + f"\npeak ({cx:.1f}, {cy:.1f}) mm", fontsize=8)


def plot_slices(
    p_abs: np.ndarray,
    p_phase: np.ndarray,
    dx_mm: float,
    out_path: Path,
    body_mask: Optional[np.ndarray] = None,
    source_region_x: int = 0,
    viz_full_field: bool = False,
) -> None:
    nx, ny, nz = p_abs.shape

    # Crop off the source/water-bath x-region for display and for vmax stats.
    # The source slices have field amplitude orders of magnitude higher than the
    # propagating wave and would otherwise collapse the entire colormap to black.
    x0 = min(source_region_x, nx - 1)
    p_abs_disp  = p_abs[x0:]
    p_phase_disp = p_phase[x0:]
    bm_disp = body_mask[x0:] if body_mask is not None else None

    nd_x = p_abs_disp.shape[0]
    xmid = nd_x // 2
    ymid = ny // 2
    zmid = nz // 2

    # Extents in mm: x-axis starts at x0*dx_mm offset from domain origin.
    x_off = x0 * dx_mm
    extent_xy = [x_off, x_off + nd_x * dx_mm, 0, ny * dx_mm]
    extent_xz = [x_off, x_off + nd_x * dx_mm, 0, nz * dx_mm]
    extent_yz = [0, ny * dx_mm, 0, nz * dx_mm]
    extents = [extent_xy, extent_xz, extent_yz]
    names = ["XY", "XZ", "YZ"]

    # Swap working arrays to the cropped versions.
    p_abs_work  = p_abs_disp.copy()
    p_phase_work = p_phase_disp.copy()
    body_mask = bm_disp  # local alias

    if body_mask is not None:
        if body_mask.shape != p_abs_work.shape:
            print(f"[warn] body_mask shape {body_mask.shape} != display shape {p_abs_work.shape}, ignoring mask.")
            body_mask = None

    finite_abs = p_abs_work[np.isfinite(p_abs_work)]
    vmax_global = float(np.percentile(finite_abs, 99.5)) if finite_abs.size > 0 else 1.0

    use_mask = False
    if viz_full_field:
        vmax = vmax_global
        print(f"[info] Pressure colormap vmax (full-field, 99.5th pct): {vmax:.4e} Pa")
    elif body_mask is not None:
        in_head = p_abs_work[body_mask]
        finite_in_head = in_head[np.isfinite(in_head)]
        vmax_in_head_99 = float(np.percentile(finite_in_head, 99.5)) if finite_in_head.size > 0 else 0.0
        vmax_in_head_90 = float(np.percentile(finite_in_head, 90.0)) if finite_in_head.size > 0 else 0.0
        # Use 90th percentile — 99.5th is dominated by skull hotspots and makes
        # the brain interior black.  Floor at 30 % of 99.5th to avoid too-dark plots.
        vmax_in_head = max(vmax_in_head_90, vmax_in_head_99 * 0.3)
        # Only apply mask when in-head signal is at least 1 % of global peak.
        if vmax_global > 0 and (vmax_in_head / vmax_global) >= 0.01:
            use_mask = True
        else:
            print(
                f"[warn] In-head vmax ({vmax_in_head:.2e}) is < 1 % of global vmax ({vmax_global:.2e}). "
                "Body mask suppressed — showing full field."
            )

    if use_mask:
        p_abs_work[~body_mask] = np.nan
        p_phase_work[~body_mask] = np.nan
        vmax = vmax_in_head
        print(f"[info] Pressure colormap vmax (90th pct in-head): {vmax:.4e} Pa")
    else:
        vmax = vmax_global
        print(f"[info] Pressure colormap vmax (99.5th pct global): {vmax:.4e} Pa")

    # RMS pressure = |P| / sqrt(2)  (for a time-harmonic field)
    p_rms_work = p_abs_work / np.sqrt(2.0)
    vmax_rms = vmax / np.sqrt(2.0)

    planes_mag  = [p_abs_work[:, :, zmid].T, p_abs_work[:, ymid, :].T, p_abs_work[xmid, :, :].T]
    planes_ph   = [p_phase_work[:, :, zmid].T, p_phase_work[:, ymid, :].T, p_phase_work[xmid, :, :].T]
    planes_rms  = [p_rms_work[:, :, zmid].T, p_rms_work[:, ymid, :].T, p_rms_work[xmid, :, :].T]

    # Log scale floor — pressure spans 4–6 orders of magnitude; linear makes interior black.
    vmin_log = max(1e-12, vmax / 1e5)

    # dB scale (re global peak): reveals deep-brain field hidden by linear scale.
    # Phase is only meaningful where amplitude is above the noise floor; mask low-amplitude
    # voxels in the phase plot to avoid showing random noise as structure.
    p_ref = float(np.nanmax(p_abs_work)) if np.any(np.isfinite(p_abs_work)) else 1.0
    p_ref = max(p_ref, 1e-30)
    DB_FLOOR = -60.0  # dB floor — 60 dB below peak
    def _to_db(plane):
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 20.0 * np.log10(np.where(plane > 0, plane, np.nan) / p_ref)
        return np.where(np.isfinite(db), np.clip(db, DB_FLOOR, 0.0), np.nan)

    planes_db = [_to_db(pl) for pl in planes_mag]

    # Phase: blank voxels where |P| is below the log floor (noise / no signal)
    amp_threshold = vmin_log
    planes_ph_masked = []
    for pm, pa in zip(planes_ph, planes_mag):
        ph_m = pm.copy()
        ph_m[pa < amp_threshold] = np.nan
        planes_ph_masked.append(ph_m)

    # Colourmap with NaN rendered as black (outside head / below threshold)
    def _cmap(name):
        cm = plt.get_cmap(name).copy()
        cm.set_bad("black")
        return cm
    cmap_mag   = _cmap("inferno")
    cmap_phase = _cmap("twilight")
    cmap_rms   = _cmap("hot")
    cmap_db    = _cmap("viridis")

    from matplotlib.colors import LogNorm
    norm_mag = LogNorm(vmin=vmin_log, vmax=vmax)
    norm_rms = LogNorm(vmin=vmin_log / np.sqrt(2.0), vmax=vmax_rms)

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    # --- Row 0: |P| log scale (Pa) ---
    for i in range(3):
        im = axes[0, i].imshow(
            np.where(planes_mag[i] > 0, planes_mag[i], np.nan),
            origin="lower", cmap=cmap_mag, norm=norm_mag,
            extent=extents[i],
        )
        axes[0, i].set_title(f"|P| Pa (log) {names[i]}")
        axes[0, i].set_xlabel("mm"); axes[0, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, label="|P| (Pa)")

    # --- Row 1: |P| in dB re peak (shows full dynamic range) ---
    for i in range(3):
        im = axes[1, i].imshow(
            planes_db[i], origin="lower", cmap=cmap_db,
            extent=extents[i], vmin=DB_FLOOR, vmax=0.0,
        )
        axes[1, i].set_title(f"|P| dB re peak {names[i]}")
        axes[1, i].set_xlabel("mm"); axes[1, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, label="dB re peak")

    # --- Row 2: RMS pressure log scale ---
    for i in range(3):
        im = axes[2, i].imshow(
            np.where(planes_rms[i] > 0, planes_rms[i], np.nan),
            origin="lower", cmap=cmap_rms, norm=norm_rms,
            extent=extents[i],
        )
        axes[2, i].set_title(f"P_rms Pa (log) {names[i]}")
        axes[2, i].set_xlabel("mm"); axes[2, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, label="P_rms (Pa)")

    # --- Row 3: Phase (amplitude-gated) ---
    for i in range(3):
        im = axes[3, i].imshow(
            planes_ph_masked[i], origin="lower", cmap=cmap_phase,
            extent=extents[i], vmin=-np.pi, vmax=np.pi,
        )
        axes[3, i].set_title(f"phase(P) {names[i]}  (|P|>log floor)")
        axes[3, i].set_xlabel("mm"); axes[3, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[3, i], fraction=0.046, label="phase (rad)")

    if vmax <= 1e-12:
        fig.text(
            0.5, 0.01,
            "Warning: pressure magnitude is near zero everywhere; check source/solver settings.",
            ha="center", color="crimson",
        )

    mask_note = " (masked to head)" if use_mask else ""
    plt.suptitle(
        f"Helmholtz CT-brain @ {p_ref:.2e} Pa peak  |  linear, dB, RMS, phase{mask_note}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Saved slices: {out_path}")


def plot_3d_magnitude(
    p_abs: np.ndarray,
    dx_mm: float,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plotted = False
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        vals = p_abs[np.isfinite(p_abs)]
        if vals.size > 0:
            levels = np.unique(np.percentile(vals, [95.0, 98.0]))
            for lv, col, a in zip(levels, ["#f39c12", "#e74c3c"], [0.15, 0.30]):
                try:
                    verts, faces, _, _ = measure.marching_cubes(p_abs, level=float(lv))
                except Exception:
                    continue
                verts_mm = verts * dx_mm
                mesh = Poly3DCollection(verts_mm[faces], alpha=a, facecolor=col, edgecolor="none")
                ax.add_collection3d(mesh)
                plotted = True
    except Exception:
        pass

    if not plotted:
        idx = np.argwhere(p_abs > np.percentile(p_abs[np.isfinite(p_abs)], 98.0))
        if idx.size > 0:
            idx = idx[:: max(1, len(idx) // 5000)]
            ax.scatter(idx[:, 0] * dx_mm, idx[:, 1] * dx_mm, idx[:, 2] * dx_mm, s=2, alpha=0.25)

    ax.set_xlim(0, p_abs.shape[0] * dx_mm)
    ax.set_ylim(0, p_abs.shape[1] * dx_mm)
    ax.set_zlim(0, p_abs.shape[2] * dx_mm)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D Helmholtz pressure magnitude isosurfaces")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if _J_WAVE_IMPORT_ERROR is not None:
        py = sys.executable
        ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "Failed to import jwave stack.\n"
            f"Current python: {py} (version {ver})\n"
            "Run with your jwave env python, e.g.:\n"
            "  /Users/shanmukasadhu/miniconda3/envs/jwave/bin/python Code/j_wave_planar_ct_brain_3d_helmholtz.py\n\n"
            f"Original import error:\n{repr(_J_WAVE_IMPORT_ERROR)}"
        )

    parser = argparse.ArgumentParser(description="3D CT-brain frequency-domain Helmholtz solver.")
    parser.add_argument("--input-dir", default=None, help="Directory containing acoustic maps.")
    parser.add_argument("--auto-root", default="Results/jwave_brain_forward_64", help="Auto-discovery root.")
    parser.add_argument("--patient-id", default=None, help="Prefer this patient in auto-discovery.")
    parser.add_argument("--out-dir", default=None, help="Output dir (default: <resolved_input_dir>/helmholtz_3d).")
    parser.add_argument("--dx-mm", type=float, default=0.2, help="Target isotropic dx in mm.")
    parser.add_argument("--auto-relax-dx", type=int, default=1, choices=[0, 1], help="Auto-increase dx to fit max-voxels.")
    parser.add_argument("--max-voxels", type=int, default=300_000_000, help="Voxel cap after resampling.")
    parser.add_argument("--frequency", type=float, default=1.0e6, help="Helmholtz frequency (Hz).")
    parser.add_argument("--source-x-slices", type=int, default=6, help="Planar source thickness in x slices.")
    parser.add_argument("--method", default="gmres", choices=["gmres", "bicgstab"], help="Helmholtz linear solver method.")
    parser.add_argument("--tol", type=float, default=2e-3, help="Solver tolerance.")
    parser.add_argument("--restart", type=int, default=30, help="GMRES restart.")
    parser.add_argument("--maxiter", type=int, default=300, help="Linear-solver max iterations.")
    parser.add_argument("--checkpoint", type=int, default=1, choices=[0, 1], help="Enable checkpoint mode.")
    parser.add_argument(
        "--crop-x-mm-each-side",
        type=float,
        default=45.0,
        help="Crop this many mm from each x-side after isotropic resampling (default: 0 = no crop).",
    )
    parser.add_argument(
        "--crop-y-start-mm",
        type=float,
        default=20.0,
        help="Crop this many mm from start of y-axis after isotropic resampling (default: 0 = no crop).",
    )
    parser.add_argument(
        "--crop-z-start-mm",
        type=float,
        default=20.0,
        help="Crop this many mm from start of z-axis after isotropic resampling (default: 0 = no crop).",
    )
    args = parser.parse_args()

    in_dir_in = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
    auto_root = Path(args.auto_root).expanduser().resolve()
    c_path, rho_path, resolved_dir = _discover_input_maps(in_dir_in, auto_root, args.patient_id)
    in_dir = resolved_dir
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (in_dir / "helmholtz_3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    c_xyz = np.load(c_path).astype(np.float32)
    rho_xyz = np.load(rho_path).astype(np.float32)
    if c_xyz.shape != rho_xyz.shape:
        raise ValueError(f"Shape mismatch: c={c_xyz.shape}, rho={rho_xyz.shape}")

    spacing_xyz_mm = _load_spacing_from_metadata(in_dir)
    print(f"[info] Raw shape={c_xyz.shape}, spacing={spacing_xyz_mm} mm")
    dx_mm = float(args.dx_mm)

    # Voxel budget: auto-detect from memory or use explicit override.
    max_voxels = _auto_voxel_budget() if args.max_voxels <= 0 else args.max_voxels
    print(f"[info] Voxel budget: {max_voxels:,}")

    # Load raw body mask before any resampling so it can follow auto-relax.
    raw_mask = None
    mask_path = _discover_body_mask(c_path)
    if mask_path is not None:
        raw_mask = np.load(mask_path)
        print(f"[info] Found body mask: {mask_path}")
    else:
        print("[info] No body mask found, plotting full volume.")

    # Crop parameters in mm — stored here so they can be reused inside the
    # auto-relax loop each time dx changes.
    crop_x_mm = float(args.crop_x_mm_each_side)
    crop_y_mm = float(args.crop_y_start_mm)
    crop_z_mm = float(args.crop_z_start_mm)

    def _resample_and_crop(target_dx: float):
        """Resample c/rho to target_dx and apply all crops.  Returns (c, rho)."""
        c, rho = _resample_to_isotropic(c_xyz, rho_xyz, spacing_xyz_mm, target_dx_mm=target_dx)
        c,   _ = _crop_x_both_sides(c,   dx_mm=target_dx, crop_x_mm_each_side=crop_x_mm)
        rho, _ = _crop_x_both_sides(rho, dx_mm=target_dx, crop_x_mm_each_side=crop_x_mm)
        c,   _ = _crop_axis_start(c,   dx_mm=target_dx, crop_mm=crop_y_mm, axis=1)
        rho, _ = _crop_axis_start(rho, dx_mm=target_dx, crop_mm=crop_y_mm, axis=1)
        c,   _ = _crop_axis_start(c,   dx_mm=target_dx, crop_mm=crop_z_mm, axis=2)
        rho, _ = _crop_axis_start(rho, dx_mm=target_dx, crop_mm=crop_z_mm, axis=2)
        return c, rho

    # Resample + crop at the requested dx first, then check the budget.
    # Doing crops BEFORE auto-relax means the budget check sees the reduced
    # domain, so the auto-relax can stay at a finer dx (or skip entirely).
    c_iso, rho_iso = _resample_and_crop(dx_mm)
    print(f"[info] After initial resample+crop: shape={c_iso.shape}, voxels={np.prod(c_iso.shape):,}")

    # Auto-relax dx upward (15 % per step) until the CROPPED domain fits budget.
    # Hard floor: 4 PPW at the target frequency in soft tissue (c≈1400 m/s).
    if args.auto_relax_dx:
        freq = float(args.frequency)
        lam_min_mm = 1400.0 / freq * 1e3
        dx_ppw_floor = lam_min_mm / 4.0
        while np.prod(c_iso.shape) > max_voxels:
            dx_mm_new = dx_mm * 1.15
            if dx_mm_new > dx_ppw_floor:
                print(
                    f"[warn] Auto-relax would reach {dx_mm_new:.3f} mm > 4 PPW floor "
                    f"({dx_ppw_floor:.3f} mm at {freq/1e3:.0f} kHz). Clamping."
                )
                dx_mm = dx_ppw_floor
                c_iso, rho_iso = _resample_and_crop(dx_mm)
                break
            dx_mm = dx_mm_new
            c_iso, rho_iso = _resample_and_crop(dx_mm)
            print(f"[info] Auto-relaxed dx → {dx_mm:.4f} mm, shape={c_iso.shape}, voxels={np.prod(c_iso.shape):,}")
            if np.prod(c_iso.shape) <= max_voxels:
                break

    # Log the crops that were applied at the final dx.
    crop_x_vox = int(round(crop_x_mm / dx_mm)) if crop_x_mm > 0 else 0
    crop_y_vox = int(round(crop_y_mm / dx_mm)) if crop_y_mm > 0 else 0
    crop_z_vox = int(round(crop_z_mm / dx_mm)) if crop_z_mm > 0 else 0
    if crop_x_vox > 0:
        print(f"[info] x-crop: {crop_x_mm:.1f} mm per side ({crop_x_vox} vox at dx={dx_mm:.3f} mm).")
    if crop_y_vox > 0:
        print(f"[info] y-start crop: {crop_y_mm:.1f} mm ({crop_y_vox} vox).")
    if crop_z_vox > 0:
        print(f"[info] z-start crop: {crop_z_mm:.1f} mm ({crop_z_vox} vox).")

    # Resample + crop body mask at the final dx_mm.
    body_mask = None
    if raw_mask is not None:
        body_mask = _resample_mask(raw_mask, spacing_xyz_mm, target_dx_mm=dx_mm)
        body_mask, _ = _crop_x_both_sides(body_mask, dx_mm=dx_mm, crop_x_mm_each_side=crop_x_mm)
        body_mask, _ = _crop_axis_start(body_mask, dx_mm=dx_mm, crop_mm=crop_y_mm, axis=1)
        body_mask, _ = _crop_axis_start(body_mask, dx_mm=dx_mm, crop_mm=crop_z_mm, axis=2)
        print(f"[info] Body mask: shape={body_mask.shape}, coverage={body_mask.mean():.3f}")

    # Rotate the chosen propagation axis to position 0, then optionally flip it
    # so the source is placed at the HIGH-index face (e.g. vertex for z-axis HIFU).
    # Crops above were applied in the original CT coordinate system, so this step
    # always comes after cropping.
    prop_axis = int(args.propagation_axis)
    ax_names = {0: "CT-row (x)", 1: "CT-col (y)", 2: "CT-slice (z)"}
    if prop_axis != 0:
        c_iso   = np.moveaxis(c_iso,   prop_axis, 0)
        rho_iso = np.moveaxis(rho_iso, prop_axis, 0)
        if body_mask is not None:
            body_mask = np.moveaxis(body_mask, prop_axis, 0)

    if bool(args.flip_propagation_axis):
        # np.flip returns a view; ascontiguousarray makes it a proper C array
        # so downstream concatenation (water bath) works without issues.
        c_iso   = np.ascontiguousarray(np.flip(c_iso,   axis=0))
        rho_iso = np.ascontiguousarray(np.flip(rho_iso, axis=0))
        if body_mask is not None:
            body_mask = np.ascontiguousarray(np.flip(body_mask, axis=0))
        flip_note = ", flipped (source at high-index face)"
    else:
        flip_note = ""

    print(
        f"[info] Propagation axis: {ax_names.get(prop_axis, prop_axis)} → internal axis 0"
        f"{flip_note}.  Shape: {c_iso.shape}"
    )

    # Prepend a water coupling bath at x=0.
    # A CT volume typically fills its full FOV, so x=0 is right at the scalp/skull.
    # Without a water gap, the unit-source at x=0:source_x_slices fires directly
    # into bone, making wave injection completely inefficient.  We prepend water
    # so the source launches a plane wave that then impinges on the skull at normal
    # incidence with proper far-field development.
    water_bath_vox = max(int(args.source_x_slices) + 4,
                         int(round(float(args.water_bath_mm) / dx_mm)))
    if water_bath_vox > 0:
        bath_c   = np.full((water_bath_vox, c_iso.shape[1],   c_iso.shape[2]),   1480.0, dtype=np.float32)
        bath_rho = np.full((water_bath_vox, rho_iso.shape[1], rho_iso.shape[2]), 1000.0, dtype=np.float32)
        c_iso   = np.concatenate([bath_c,   c_iso],   axis=0)
        rho_iso = np.concatenate([bath_rho, rho_iso], axis=0)
        if body_mask is not None:
            bath_mask = np.zeros((water_bath_vox, body_mask.shape[1], body_mask.shape[2]), dtype=bool)
            body_mask = np.concatenate([bath_mask, body_mask], axis=0)
        print(f"[info] Prepended {water_bath_vox} vox ({water_bath_vox * dx_mm:.1f} mm) water coupling bath at x=0.")
        print(f"[info] Domain after bath: shape={c_iso.shape}, voxels={np.prod(c_iso.shape):,}")

    # Report final resolution quality before committing to the solve.
    lam_tissue_mm = 1540.0 / float(args.frequency) * 1e3
    ppw_final = lam_tissue_mm / dx_mm
    print(f"[info] Final: shape={c_iso.shape}, voxels={np.prod(c_iso.shape):,}, dx={dx_mm:.4f} mm")
    print(f"[info] λ(tissue,1540 m/s)={lam_tissue_mm:.2f} mm, PPW={ppw_final:.1f}")
    if ppw_final < 5.0:
        print(
            f"[warn] PPW={ppw_final:.1f} < 5.0; dispersion (graininess) artifacts likely. "
            "To improve: (1) crop domain with --crop-* flags to a smaller ROI, "
            "(2) set --max-voxels to a smaller number to force finer dx on a cropped domain, "
            "or (3) use a lower frequency."
        )
    p_abs, p_phase = solve_helmholtz(
        c_iso,
        rho_iso,
        dx_mm=dx_mm,
        frequency_hz=float(args.frequency),
        source_x_slices=int(args.source_x_slices),
        method=args.method,
        tol=float(args.tol),
        restart=int(args.restart),
        maxiter=int(args.maxiter),
        checkpoint=bool(args.checkpoint),
    )

    fig_slices = out_dir / "helmholtz_pressure_slices.png"
    # Skip water bath + optional interface region so the left of the plot isn't dominated by interface artifacts.
    viz_skip_vox = int(round(float(args.viz_skip_interface_mm) / dx_mm))
    display_start_x = min(water_bath_vox + viz_skip_vox, p_abs.shape[0] - 1)
    plot_slices(
        p_abs, p_phase, dx_mm=dx_mm, out_path=fig_slices,
        body_mask=body_mask,
        source_region_x=display_start_x,
        viz_full_field=bool(args.viz_full_field),
    )

    summary = {
        "input_dir": str(in_dir),
        "out_dir": str(out_dir),
        "dx_mm_requested": float(args.dx_mm),
        "dx_mm_used": float(dx_mm),
        "frequency_hz": float(args.frequency),
        "shape_before": list(c_xyz.shape),
        "shape_after_isotropic": list(c_iso.shape),
        "crop_x_mm_each_side": float(crop_x_mm),
        "crop_x_vox_each_side": int(crop_x_vox),
        "crop_y_start_mm": float(crop_y_mm),
        "crop_y_start_vox": int(crop_y_vox),
        "crop_z_start_mm": float(crop_z_mm),
        "crop_z_start_vox": int(crop_z_vox),
        "propagation_axis": int(args.propagation_axis),
        "flip_propagation_axis": bool(args.flip_propagation_axis),
        "spacing_before_mm": list(spacing_xyz_mm),
        "method": args.method,
        "tol": float(args.tol),
        "restart": int(args.restart),
        "maxiter": int(args.maxiter),
        "checkpoint": bool(args.checkpoint),
        "body_mask_used": mask_path is not None,
        "max_pressure_abs": float(np.max(p_abs[np.isfinite(p_abs)])) if np.any(np.isfinite(p_abs)) else 0.0,
        "outputs": [
            "helmholtz_pressure_slices.png",
            "summary.json",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[info] Isotropic shape={c_iso.shape}, dx={dx_mm:.4f} mm")
    print(f"[info] Body mask used: {mask_path is not None}")


if __name__ == "__main__":
    main()