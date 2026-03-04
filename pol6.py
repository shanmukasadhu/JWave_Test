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


# ---------------------------------------------------------------------------
# SIREN MLP for lens alpha (forward-only, no backprop)
# ---------------------------------------------------------------------------

def _siren_forward(weights: list[np.ndarray], biases: list[np.ndarray], x: np.ndarray, w0: float = 60.0, alpha_lo: float = 0.5) -> np.ndarray:
    """
    Forward pass of a SIREN MLP. Input (N, 3), output (N,) alpha in [alpha_lo, 1].
    Uses sin(w0 * z) activations with high w0 for strong spatial variation.
    """
    h = x.astype(np.float32)
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W + b
        if i < len(weights) - 1:
            h = np.sin(w0 * h)
    sig = 1.0 / (1.0 + np.exp(-np.clip(h.ravel(), -50, 50)))
    return alpha_lo + (1.0 - alpha_lo) * sig


def _build_siren_lens_alpha(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    hidden: int = 32,
    w0: float = 60.0,
    coord_scale: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Build a SIREN with random weights and return alpha for given (x,y,z) in mm.
    Layout: 3 -> hidden -> hidden -> hidden -> 1 for more expressivity.
    coord_scale multiplies coords so SIREN sees finer spatial structure.
    """
    rng = np.random.default_rng(seed)
    xs = x_mm.ravel().astype(np.float32) * coord_scale
    ys = y_mm.ravel().astype(np.float32) * coord_scale
    zs = z_mm.ravel().astype(np.float32) * coord_scale
    pts = np.stack([xs, ys, zs], axis=1)
    W1 = rng.uniform(-1.0 / 3.0, 1.0 / 3.0, (3, hidden)).astype(np.float32)
    b1 = rng.uniform(-1.0 / hidden, 1.0 / hidden, (hidden,)).astype(np.float32)
    W2 = rng.uniform(-1.0 / hidden, 1.0 / hidden, (hidden, hidden)).astype(np.float32)
    b2 = rng.uniform(-1.0 / hidden, 1.0 / hidden, (hidden,)).astype(np.float32)
    W3 = rng.uniform(-1.0 / hidden, 1.0 / hidden, (hidden, hidden)).astype(np.float32)
    b3 = rng.uniform(-1.0 / hidden, 1.0 / hidden, (hidden,)).astype(np.float32)
    W4 = rng.uniform(-1.0 / hidden, 1.0 / hidden, (hidden, 1)).astype(np.float32)
    b4 = rng.uniform(-0.5, 0.5, (1,)).astype(np.float32)
    weights = [W1, W2, W3, W4]
    biases = [b1, b2, b3, b4]
    return _siren_forward(weights, biases, pts, w0=w0, alpha_lo=0.5)


def _apply_siren_lens_to_bath(
    bath_c: np.ndarray,
    bath_rho: np.ndarray,
    dx_mm: float,
    bath_fracs: Tuple[float, float, float],
    rho_solid: float = 2500.0,
    c_solid: float = 4000.0,
) -> None:
    """
    In-place: fill the entire lens design partition with SIREN-modulated material.
    bath_c, bath_rho are (nx, ny, nz). Lens partition is the middle fraction.
    Uses alpha in [0.5, 1] -> c in [2000, 4000], rho in [1250, 2500] for strong variation.
    """
    n1 = int(bath_c.shape[0] * bath_fracs[0])
    n2 = int(bath_c.shape[0] * bath_fracs[1])
    x0, x1 = n1, n1 + n2
    if x1 <= x0:
        return
    ny, nz = bath_c.shape[1], bath_c.shape[2]
    ix = np.arange(x0, x1, dtype=np.float32) * dx_mm
    iy = np.arange(ny, dtype=np.float32) * dx_mm
    iz = np.arange(nz, dtype=np.float32) * dx_mm
    xg, yg, zg = np.meshgrid(ix, iy, iz, indexing="ij")
    alpha = _build_siren_lens_alpha(xg, yg, zg)
    alpha = alpha.reshape(xg.shape)
    bath_rho[x0:x1, :, :] = alpha * rho_solid
    bath_c[x0:x1, :, :] = alpha * c_solid
    c_lens_min, c_lens_max = float(np.min(alpha) * c_solid), float(np.max(alpha) * c_solid)
    rho_lens_min, rho_lens_max = float(np.min(alpha) * rho_solid), float(np.max(alpha) * rho_solid)
    print(f"[info] SIREN lens: vox [{x0}:{x1}, 0:{ny}, 0:{nz}] ({np.prod(alpha.shape):,} voxels) c=[{c_lens_min:.0f},{c_lens_max:.0f}] m/s rho=[{rho_lens_min:.0f},{rho_lens_max:.0f}] kg/m³")


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


def _crop_axis_end(
    arr_xyz: np.ndarray,
    dx_mm: float,
    crop_mm: float,
    axis: int,
) -> Tuple[np.ndarray, int]:
    """Crop only from the end (high-index) of a given axis."""
    if crop_mm <= 0:
        return arr_xyz, 0
    crop_vox = int(round(crop_mm / dx_mm))
    if crop_vox <= 0:
        return arr_xyz, 0
    shape = arr_xyz.shape
    if crop_vox >= shape[axis]:
        raise ValueError(
            f"Requested end-crop too large: axis={axis}, crop_vox={crop_vox}, shape={shape}."
        )
    if axis == 0:
        return arr_xyz[:-crop_vox, :, :], crop_vox
    if axis == 1:
        return arr_xyz[:, :-crop_vox, :], crop_vox
    if axis == 2:
        return arr_xyz[:, :, :-crop_vox], crop_vox
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



def plot_slices(
    p_abs: np.ndarray,
    p_phase: np.ndarray,
    dx_mm: float,
    out_path: Path,
    body_mask: Optional[np.ndarray] = None,
    source_region_x: int = 0,
    viz_full_field: bool = False,
    bath_partition_fracs: Tuple[float, float, float] = (1.0/3.0, 1.0/3.0, 1.0/3.0),
    c_full: Optional[np.ndarray] = None,
) -> None:
    nx, ny, nz = p_abs.shape

    # Debug: where is the peak pressure along the propagation axis?
    finite = np.isfinite(p_abs)
    if np.any(finite):
        peak_flat = np.argmax(np.where(finite, p_abs, -np.inf))
        peak_idx_3d = np.unravel_index(peak_flat, p_abs.shape)
        peak_prop_idx = peak_idx_3d[0]
        peak_prop_mm = peak_prop_idx * dx_mm
        prop_len_mm = nx * dx_mm
        print(
            f"[debug] PEAK PRESSURE: at propagation index {peak_prop_idx} / {nx - 1} "
            f"= {peak_prop_mm:.1f} mm from LEFT (source end). "
            f"Domain length {prop_len_mm:.1f} mm. "
            f"Peak is {(100.0 * peak_prop_idx / max(1, nx - 1)):.0f}% along propagation."
        )

    # Crop off the source/water-bath x-region for display and for vmax stats.
    # The source slices have field amplitude orders of magnitude higher than the
    # propagating wave and would otherwise collapse the entire colormap to black.
    x0 = min(source_region_x, nx - 1)
    p_abs_disp  = p_abs[x0:]
    p_phase_disp = p_phase[x0:]
    bm_disp = body_mask[x0:] if body_mask is not None else None

    nd_x = p_abs_disp.shape[0]
    # Debug: peak within DISPLAYED region (what the user sees in cropped plots)
    finite_disp = np.isfinite(p_abs_disp)
    if np.any(finite_disp):
        peak_disp_flat = np.argmax(np.where(finite_disp, p_abs_disp, -np.inf))
        peak_disp_3d = np.unravel_index(peak_disp_flat, p_abs_disp.shape)
        peak_disp_prop_idx = peak_disp_3d[0]
        peak_disp_prop_mm = x0 * dx_mm + peak_disp_prop_idx * dx_mm
        disp_len_mm = nd_x * dx_mm
        pct_disp = 100.0 * peak_disp_prop_idx / max(1, nd_x - 1)
        print(
            f"[debug] PEAK IN DISPLAYED REGION (cropped plots): at displayed index {peak_disp_prop_idx} / {nd_x - 1} "
            f"= {peak_disp_prop_mm:.1f} mm from left of full domain. "
            f"Displayed span: {x0 * dx_mm:.1f}..{x0 * dx_mm + disp_len_mm:.1f} mm. "
            f"Peak is {pct_disp:.0f}% along displayed region "
            f"({'near LEFT/source' if pct_disp < 20 else 'near RIGHT/far end' if pct_disp > 80 else 'middle'})."
        )
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

    # Swap working arrays to the cropped versions. Avoid extra copies to reduce OOM risk.
    p_abs_work  = p_abs_disp.copy()
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
        vmax = vmax_in_head
        print(f"[info] Pressure colormap vmax (90th pct in-head): {vmax:.4e} Pa")
    else:
        vmax = vmax_global
        print(f"[info] Pressure colormap vmax (99.5th pct global): {vmax:.4e} Pa")

    # RMS pressure = |P| / sqrt(2)  (for a time-harmonic field)
    p_rms_work = p_abs_work / np.sqrt(2.0)
    vmax_rms = vmax / np.sqrt(2.0)

    planes_mag  = [p_abs_work[:, :, zmid].T, p_abs_work[:, ymid, :].T, p_abs_work[xmid, :, :].T]
    planes_rms  = [p_rms_work[:, :, zmid].T, p_rms_work[:, ymid, :].T, p_rms_work[xmid, :, :].T]

    # Log scale floor — pressure spans 4–6 orders of magnitude; linear makes interior black.
    vmin_log = max(1e-12, vmax / 1e5)

    # Full-domain planes (including water bath) so boundary lines are visible inside the plot
    xmid_full = nx // 2
    planes_full = [p_abs[:, :, zmid].T, p_abs[:, ymid, :].T, p_abs[xmid_full, :, :].T]
    planes_full_rms = [
        (p_abs[:, :, zmid] / np.sqrt(2.0)).T,
        (p_abs[:, ymid, :] / np.sqrt(2.0)).T,
        (p_abs[xmid_full, :, :] / np.sqrt(2.0)).T,
    ]
    extent_full_xy = [0, nx * dx_mm, 0, ny * dx_mm]
    extent_full_xz = [0, nx * dx_mm, 0, nz * dx_mm]
    extent_full_yz = [0, ny * dx_mm, 0, nz * dx_mm]
    extents_full = [extent_full_xy, extent_full_xz, extent_full_yz]

    # Colourmap with NaN rendered as black (outside head / below threshold)
    def _cmap(name):
        cm = plt.get_cmap(name).copy()
        cm.set_bad("black")
        return cm
    cmap_mag   = _cmap("inferno")
    cmap_rms   = _cmap("hot")

    from matplotlib.colors import LogNorm
    norm_mag = LogNorm(vmin=vmin_log, vmax=vmax)
    norm_rms = LogNorm(vmin=vmin_log / np.sqrt(2.0), vmax=vmax_rms)
    finite_full = p_abs[np.isfinite(p_abs)]
    vmax_full = float(np.percentile(finite_full, 99.5)) if finite_full.size > 0 else vmax
    vmax_full = max(vmax_full, 1e-12)
    vmin_log_full = max(1e-12, vmax_full / 1e5)
    norm_full = LogNorm(vmin=vmin_log_full, vmax=vmax_full)

    n_rows = 5 if c_full is not None else 4
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 22 if c_full is not None else 18))

    def _add_boundary_lines(ax, extent, col_idx: int):
        """Draw vertical lines at water bath|CT and CT end on XY and XZ (propagation on x-axis)."""
        if col_idx >= 2:
            return
        water_bath_boundary_mm = x_off
        ct_end_mm = x_off + nd_x * dx_mm
        ax.axvline(water_bath_boundary_mm, color="cyan", linewidth=1.2, linestyle="--", alpha=0.9)
        ax.axvline(ct_end_mm, color="red", linewidth=1.2, linestyle="--", alpha=0.9)

    f1, f2, f3 = bath_partition_fracs

    def _add_boundary_lines_full_domain(ax, col_idx: int):
        """Draw boundary lines: water bath split into waterbath1 | lens | waterbath2, then CT end."""
        if col_idx >= 2:
            return
        water_bath_boundary_mm = x0 * dx_mm
        ct_end_mm = nx * dx_mm
        # 3 partitions: waterbath1 (0-f1), lens (f1-f1+f2), waterbath2 (f1+f2-1)
        pos_w1_lens = water_bath_boundary_mm * f1
        pos_lens_w2 = water_bath_boundary_mm * (f1 + f2)
        ax.axvline(pos_w1_lens, color="blue", linewidth=2.0, linestyle="--", alpha=1.0, label="waterbath1|lens")
        ax.axvline(pos_lens_w2, color="lime", linewidth=2.0, linestyle="--", alpha=1.0, label="lens|waterbath2")
        ax.axvline(water_bath_boundary_mm, color="cyan", linewidth=2.0, linestyle="--", alpha=1.0, label="waterbath2|CT")
        ax.axvline(ct_end_mm, color="red", linewidth=2.0, linestyle="--", alpha=1.0, label="CT end")
        # Labels in the middle of each partition (all water for now)
        ylo, yhi = ax.get_ylim()
        ymid = (ylo + yhi) * 0.5
        ax.text(water_bath_boundary_mm * (f1 / 2.0), ymid, "waterbath1", color="blue", ha="center", va="center", fontsize=8)
        ax.text(water_bath_boundary_mm * (f1 + f2 / 2.0), ymid, "lens", color="lime", ha="center", va="center", fontsize=8)
        ax.text(water_bath_boundary_mm * (f1 + f2 + f3 / 2.0), ymid, "waterbath2", color="cyan", ha="center", va="center", fontsize=8)

    # Red circle marking the center point (xmid_full, ymid, zmid) visible in all 3 slice planes
    from matplotlib.patches import Circle
    center_x_mm = (xmid_full + 0.5) * dx_mm
    center_y_mm = (ymid + 0.5) * dx_mm
    center_z_mm = (zmid + 0.5) * dx_mm
    ref_circle_radius_mm = 4.0

    def _add_reference_point(ax, col_idx: int):
        """Draw a red circle at the center point (visible in XY, XZ, YZ)."""
        if col_idx == 0:
            cx, cy = center_x_mm, center_y_mm
        elif col_idx == 1:
            cx, cy = center_x_mm, center_z_mm
        else:
            cx, cy = center_y_mm, center_z_mm
        circle = Circle((cx, cy), ref_circle_radius_mm, fill=False, edgecolor="red", linewidth=2.0)
        ax.add_patch(circle)

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
        _add_boundary_lines(axes[0, i], extents[i], i)
        _add_reference_point(axes[0, i], i)

    # --- Row 1: |P| full domain (water bath + CT) with visible boundary lines ---
    for i in range(3):
        im = axes[1, i].imshow(
            np.where(planes_full[i] > 0, planes_full[i], np.nan),
            origin="lower", cmap=cmap_mag, norm=norm_full,
            extent=extents_full[i],
        )
        axes[1, i].set_title(f"|P| Pa (log) {names[i]}  (full domain, bath|CT)")
        axes[1, i].set_xlabel("mm"); axes[1, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, label="|P| (Pa)")
        _add_boundary_lines_full_domain(axes[1, i], i)
        _add_reference_point(axes[1, i], i)

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
        _add_boundary_lines(axes[2, i], extents[i], i)
        _add_reference_point(axes[2, i], i)

    # --- Row 3: Pressure RMS full domain (with lens) ---
    norm_rms_full = LogNorm(vmin=vmin_log_full / np.sqrt(2.0), vmax=vmax_full / np.sqrt(2.0))
    for i in range(3):
        im = axes[3, i].imshow(
            np.where(planes_full_rms[i] > 0, planes_full_rms[i], np.nan),
            origin="lower", cmap=cmap_rms, norm=norm_rms_full,
            extent=extents_full[i],
        )
        axes[3, i].set_title(f"P_rms Pa (log) {names[i]}  (full domain, lens)")
        axes[3, i].set_xlabel("mm"); axes[3, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[3, i], fraction=0.046, label="P_rms (Pa)")
        _add_boundary_lines_full_domain(axes[3, i], i)
        _add_reference_point(axes[3, i], i)

    # --- Row 4: Sound speed c (m/s) — lens visible as higher c vs water ---
    if c_full is not None:
        planes_c = [
            c_full[:, :, zmid].T,
            c_full[:, ymid, :].T,
            c_full[xmid_full, :, :].T,
        ]
        c_min, c_max = float(np.min(c_full)), float(np.max(c_full))
        c_water = 1480.0
        norm_c = plt.Normalize(vmin=c_water - 100, vmax=min(c_max + 50, 4500.0))
        cmap_c = plt.get_cmap("viridis").copy()
        cmap_c.set_bad("black")
        for i in range(3):
            im = axes[4, i].imshow(
                planes_c[i], origin="lower", cmap=cmap_c, norm=norm_c,
                extent=extents_full[i],
            )
            axes[4, i].set_title(f"c m/s {names[i]}  (lens = higher c)")
            axes[4, i].set_xlabel("mm"); axes[4, i].set_ylabel("mm")
            plt.colorbar(im, ax=axes[4, i], fraction=0.046, label="c (m/s)")
            _add_boundary_lines_full_domain(axes[4, i], i)
            _add_reference_point(axes[4, i], i)
        print(f"[info] Material viz: c min={c_min:.0f}, max={c_max:.0f} m/s (lens ~2000–4000 vs water 1480)")

    if vmax <= 1e-12:
        fig.text(
            0.5, 0.01,
            "Warning: pressure magnitude is near zero everywhere; check source/solver settings.",
            ha="center", color="crimson",
        )

    p_ref = float(np.nanmax(p_abs_work)) if np.any(np.isfinite(p_abs_work)) else 1.0
    p_ref = max(p_ref, 1e-30)
    mask_note = " (masked to head)" if use_mask else ""
    plt.suptitle(
        f"Helmholtz CT-brain @ {p_ref:.2e} Pa peak  |  linear, full-domain, RMS{mask_note}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")  # dpi 100 reduces memory vs 150
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
    parser.add_argument("--dx-mm", type=float, default=0.4, help="Target isotropic dx in mm (default 0.4 mm = ~6 PPW at 650 kHz).")
    parser.add_argument("--auto-relax-dx", type=int, default=1, choices=[0, 1], help="Auto-increase dx to fit max-voxels.")
    parser.add_argument("--max-voxels", type=int, default=70000000, help="Voxel cap after resampling; -1 = auto-detect from available memory.")
    parser.add_argument("--frequency", type=float, default=5.0e5, help="Helmholtz frequency (Hz). Default: 650 kHz.")
    parser.add_argument("--source-x-slices", type=int, default=4, help="Planar source thickness in x slices.")
    parser.add_argument("--water-bath-mm", type=float, default=20.0, help="Water coupling layer prepended at x=0 in mm. Ensures the source fires into water before hitting the skull (default: 20 mm).")
    parser.add_argument(
        "--waterbath1-frac", type=float, default=1.0/3.0,
        help="Fraction of water bath for waterbath1 (0-1). Default 1/3. Must sum with lens-frac and waterbath2-frac to 1.",
    )
    parser.add_argument(
        "--lens-frac", type=float, default=1.0/3.0,
        help="Fraction of water bath for lens (0-1). Default 1/3.",
    )
    parser.add_argument(
        "--waterbath2-frac", type=float, default=1.0/3.0,
        help="Fraction of water bath for waterbath2 (0-1). Default 1/3.",
    )
    parser.add_argument(
        "--propagation-axis", type=int, default=2, choices=[0, 1, 2],
        help=(
            "Axis along which the planar wave propagates (0=CT-row, 1=CT-col, 2=CT-slice). "
            "Default: 2 = CT slice (z), the superior-inferior axis. "
            "For vertex HIFU (Clement & Hynynen), the transducer sits at the parietal vertex "
            "(superior skull, z-max in most head CTs) and the beam travels inferiorly."
        ),
    )
    parser.add_argument(
        "--viz-full-field", type=int, default=0, choices=[0, 1],
        help="If 1, show full field (no body mask) and use global vmax. Use to diagnose propagation direction.",
    )
    parser.add_argument(
        "--flip-propagation-axis", type=int, default=0, choices=[0, 1],
        help=(
            "After moving the chosen axis to position 0, flip so the source is at the "
            "HIGH-index face. Default: 0. Try 1 if pressure appears at the wrong end "
            "(e.g. concentrated at base instead of vertex). CT z-convention varies."
        ),
    )
    parser.add_argument("--method", default="bicgstab", choices=["gmres", "bicgstab"], help="Helmholtz linear solver. BiCGStab is faster per iteration for 3D Helmholtz.")
    parser.add_argument("--tol", type=float, default=2e-3, help="Solver tolerance.")
    parser.add_argument("--restart", type=int, default=100, help="GMRES restart window (only used when method=gmres).")
    parser.add_argument("--maxiter", type=int, default=2000, help="Linear-solver max iterations.")
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
    parser.add_argument(
        "--crop-z-end-mm",
        type=float,
        default=35.0,
        help="Crop this many mm from end (high-index) of z-axis in plots (default: 35).",
    )
    parser.add_argument(
        "--crop-x-end-mm",
        type=float,
        default=10.0,
        help="Crop this many mm from end (high-index) of horizontal axis in XY/XZ plots (default: 10).",
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
    crop_z_end_mm = float(args.crop_z_end_mm)
    crop_x_end_mm = float(args.crop_x_end_mm)

    def _resample_and_crop(target_dx: float):
        """Resample c/rho to target_dx and apply all crops.  Returns (c, rho)."""
        c, rho = _resample_to_isotropic(c_xyz, rho_xyz, spacing_xyz_mm, target_dx_mm=target_dx)
        c,   _ = _crop_x_both_sides(c,   dx_mm=target_dx, crop_x_mm_each_side=crop_x_mm)
        rho, _ = _crop_x_both_sides(rho, dx_mm=target_dx, crop_x_mm_each_side=crop_x_mm)
        c,   _ = _crop_axis_start(c,   dx_mm=target_dx, crop_mm=crop_y_mm, axis=1)
        rho, _ = _crop_axis_start(rho, dx_mm=target_dx, crop_mm=crop_y_mm, axis=1)
        c,   _ = _crop_axis_start(c,   dx_mm=target_dx, crop_mm=crop_z_mm, axis=2)
        rho, _ = _crop_axis_start(rho, dx_mm=target_dx, crop_mm=crop_z_mm, axis=2)
        # Crop the axis that becomes the vertical (z) in XZ and YZ plots (axis 1 = y in CT frame)
        c,   _ = _crop_axis_end(c,   dx_mm=target_dx, crop_mm=crop_z_end_mm, axis=1)
        rho, _ = _crop_axis_end(rho, dx_mm=target_dx, crop_mm=crop_z_end_mm, axis=1)
        # Crop the axis that becomes the horizontal (x) in XY and XZ plots (axis 2 = z/propagation in CT frame)
        c,   _ = _crop_axis_end(c,   dx_mm=target_dx, crop_mm=crop_x_end_mm, axis=2)
        rho, _ = _crop_axis_end(rho, dx_mm=target_dx, crop_mm=crop_x_end_mm, axis=2)
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
    crop_z_end_vox = int(round(crop_z_end_mm / dx_mm)) if crop_z_end_mm > 0 else 0
    if crop_z_end_vox > 0:
        print(f"[info] z-end crop: {crop_z_end_mm:.1f} mm ({crop_z_end_vox} vox).")
    crop_x_end_vox = int(round(crop_x_end_mm / dx_mm)) if crop_x_end_mm > 0 else 0
    if crop_x_end_vox > 0:
        print(f"[info] x-end crop (horizontal in XY/XZ): {crop_x_end_mm:.1f} mm ({crop_x_end_vox} vox).")

    # Resample + crop body mask at the final dx_mm.
    body_mask = None
    if raw_mask is not None:
        body_mask = _resample_mask(raw_mask, spacing_xyz_mm, target_dx_mm=dx_mm)
        body_mask, _ = _crop_x_both_sides(body_mask, dx_mm=dx_mm, crop_x_mm_each_side=crop_x_mm)
        body_mask, _ = _crop_axis_start(body_mask, dx_mm=dx_mm, crop_mm=crop_y_mm, axis=1)
        body_mask, _ = _crop_axis_start(body_mask, dx_mm=dx_mm, crop_mm=crop_z_mm, axis=2)
        body_mask, _ = _crop_axis_end(body_mask, dx_mm=dx_mm, crop_mm=crop_z_end_mm, axis=1)
        body_mask, _ = _crop_axis_end(body_mask, dx_mm=dx_mm, crop_mm=crop_x_end_mm, axis=2)
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
    print(
        "[debug] PROPAGATION DIRECTION: After moveaxis, internal axis 0 = propagation. "
        "Index 0 = LOW end, index N-1 = HIGH end. "
        f"flip_propagation_axis={bool(args.flip_propagation_axis)} means source at "
        f"{'HIGH-index' if args.flip_propagation_axis else 'LOW-index'} face."
    )

    # Prepend a water coupling bath at x=0.
    # A CT volume typically fills its full FOV, so x=0 is right at the scalp/skull.
    # Without a water gap, the unit-source at x=0:source_x_slices fires directly
    # into bone, making wave injection completely inefficient.  We prepend water
    # so the source launches a plane wave that then impinges on the skull at normal
    # incidence with proper far-field development.
    water_bath_vox = max(int(args.source_x_slices) + 4,
                         int(round(float(args.water_bath_mm) / dx_mm)))
    bath_fracs = (
        float(args.waterbath1_frac),
        float(args.lens_frac),
        float(args.waterbath2_frac),
    )
    frac_sum = sum(bath_fracs)
    if abs(frac_sum - 1.0) > 0.001:
        bath_fracs = tuple(f / frac_sum for f in bath_fracs)
    if water_bath_vox > 0:
        bath_c   = np.full((water_bath_vox, c_iso.shape[1],   c_iso.shape[2]),   1480.0, dtype=np.float32)
        bath_rho = np.full((water_bath_vox, rho_iso.shape[1], rho_iso.shape[2]), 1000.0, dtype=np.float32)
        _apply_siren_lens_to_bath(bath_c, bath_rho, dx_mm, bath_fracs)
        c_iso   = np.concatenate([bath_c,   c_iso],   axis=0)
        rho_iso = np.concatenate([bath_rho, rho_iso], axis=0)
        if body_mask is not None:
            bath_mask = np.zeros((water_bath_vox, body_mask.shape[1], body_mask.shape[2]), dtype=bool)
            body_mask = np.concatenate([bath_mask, body_mask], axis=0)
        print(f"[info] Prepended {water_bath_vox} vox ({water_bath_vox * dx_mm:.1f} mm) water coupling bath at x=0.")
        print(f"[info] Domain after bath: shape={c_iso.shape}, voxels={np.prod(c_iso.shape):,}")
    # Debug: propagation layout for plotting
    nx_post = c_iso.shape[0]
    prop_len_mm = nx_post * dx_mm
    print(
        "[debug] PLOT LAYOUT (XY and XZ): horizontal axis = propagation (internal axis 0). "
        "Source fires at index 0. Left of plot = index 0 = source end. Right = index N-1 = far end."
    )
    if water_bath_vox > 0:
        print(
            f"[debug] Water bath: indices 0..{water_bath_vox - 1} "
            f"(0..{water_bath_vox * dx_mm - dx_mm:.1f} mm). "
            f"CT/skull: indices {water_bath_vox}..{nx_post - 1} "
            f"({water_bath_vox * dx_mm:.1f}..{prop_len_mm:.1f} mm)."
        )
    else:
        print(f"[debug] No water bath. CT spans 0..{nx_post - 1} (0..{prop_len_mm:.1f} mm).")
    print(
        "[debug] Cyan line = water|CT boundary. Red line = end of domain. "
        "Wave propagates LEFT→RIGHT in the plots."
    )
    print(
        "[debug] CT CONVENTION: propagation_axis=2 uses CT slice (z) direction. "
        "Whether LEFT=top or LEFT=bottom of skull depends on your CT's z-ordering. "
        "flip_propagation_axis=1 swaps which end becomes the source (LEFT)."
    )

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
    bath_fracs = (
        float(args.waterbath1_frac),
        float(args.lens_frac),
        float(args.waterbath2_frac),
    )
    frac_sum = sum(bath_fracs)
    if abs(frac_sum - 1.0) > 0.001:
        bath_fracs = tuple(f / frac_sum for f in bath_fracs)
        print(f"[info] Bath partition fractions normalized to sum 1: {bath_fracs}")
    plot_slices(
        p_abs, p_phase, dx_mm=dx_mm, out_path=fig_slices,
        body_mask=body_mask,
        source_region_x=water_bath_vox,
        viz_full_field=bool(args.viz_full_field),
        bath_partition_fracs=bath_fracs,
        c_full=c_iso,
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
        "siren_lens_applied": water_bath_vox > 0,
        "water_bath_vox": water_bath_vox,
        "bath_partition_fracs": list(bath_fracs),
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
