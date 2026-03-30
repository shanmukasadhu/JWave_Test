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
import jax
from jax import numpy as jnp
from jax import value_and_grad
import optax

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


def _design_band_indices_from_fracs(n: int, lo_frac: float, hi_frac: float) -> Tuple[int, int]:
    """Clamp fractional band [lo_frac, hi_frac) to valid voxel indices in [0, n]."""
    lo = max(0.0, min(1.0, float(lo_frac)))
    hi = max(0.0, min(1.0, float(hi_frac)))
    if hi <= lo:
        hi = min(1.0, lo + 1e-6)
    j_lo = max(0, int(lo * n))
    j_hi = min(n, int(hi * n))
    if j_hi <= j_lo:
        j_hi = min(n, j_lo + 1)
    return j_lo, j_hi


def _orange_mask_full_volume(
    nx: int,
    ny: int,
    nz: int,
    x_surf: Optional[np.ndarray],
    nx_post: int,
    design_j_lo: int,
    design_j_hi: int,
    design_k_lo: int,
    design_k_hi: int,
    source_x_slices_int: int,
    lens_depth_vox: int,
    fallback_x0: int,
    fallback_x1: int,
) -> np.ndarray:
    """Boolean (nx,ny,nz): True = lens / SIREN design voxels (coupling slab in front of skull).

    Lateral extent is limited by the design band [design_j_lo:design_j_hi] etc.
    When ``x_surf`` is None, fills the fallback x-slab in that band (legacy no-segmentation).
    """
    m = np.zeros((nx, ny, nz), dtype=bool)
    if x_surf is not None:
        ii = np.arange(nx, dtype=np.float64)[:, None, None]
        left_b = np.maximum(
            float(source_x_slices_int),
            x_surf[None, :, :] - float(lens_depth_vox),
        )
        in_lens = (ii >= left_b) & (ii < x_surf[None, :, :])
        has_tissue = x_surf[None, :, :] < float(nx_post)
        m = (in_lens & has_tissue)
        m[:, :design_j_lo, :] = False
        m[:, design_j_hi:, :] = False
        m[:, :, :design_k_lo] = False
        m[:, :, design_k_hi:] = False
    else:
        if fallback_x1 > fallback_x0:
            m[fallback_x0:fallback_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi] = True
    return m


def _tight_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return (i0, i1, j0, j1, k0, k1) exclusive upper bounds, or None if mask is empty."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    i0, j0, k0 = coords.min(axis=0)
    i1, j1, k1 = coords.max(axis=0)
    return int(i0), int(i1) + 1, int(j0), int(j1) + 1, int(k0), int(k1) + 1


# ---------------------------------------------------------------------------
# Transducer specifications — Jimenez-Gambin et al. 2019
# Single-element flat circular PZT-26 piezoceramic, 50 mm aperture, 1.112 MHz.
# The nominal frequency is stored only for a warning when the simulation
# frequency deviates >10%; dx and frequency are always taken from the CLI args.
# ---------------------------------------------------------------------------
_TRANSDUCER_RADIUS_MM = 25.0    # 50 mm diameter → 25 mm radius
_TRANSDUCER_FREQ_HZ   = 1.112e6 # nominal spec (warning reference only)


def _make_transducer_source(
    nxyz: Tuple[int, int, int],
    dx_mm: float,
    domain,
    frequency_hz: float,
    source_x_slice: int = 0,
) -> "FourierSeries":
    """
    Single-element flat circular PZT-26 transducer.
    Jimenez-Gambin et al. 2019 — 50 mm aperture, uniform phase.
    Focusing is done entirely by the external acoustic holographic lens (SIREN).
    dx and frequency are taken from the simulation arguments, not hardcoded.
    """
    nx, ny, nz = nxyz

    if abs(frequency_hz - _TRANSDUCER_FREQ_HZ) / _TRANSDUCER_FREQ_HZ > 0.1:
        print(
            f"[warn] Simulation frequency {frequency_hz / 1e6:.3f} MHz differs >10% "
            f"from transducer spec {_TRANSDUCER_FREQ_HZ / 1e6:.3f} MHz "
            f"(Jimenez-Gambin 2019 PZT-26)."
        )

    radius_vox    = _TRANSDUCER_RADIUS_MM / dx_mm
    domain_half_y = (ny / 2.0) * dx_mm
    domain_half_z = (nz / 2.0) * dx_mm

    if _TRANSDUCER_RADIUS_MM > min(domain_half_y, domain_half_z):
        print(
            f"[warn] Transducer radius {_TRANSDUCER_RADIUS_MM:.1f} mm exceeds domain "
            f"half-width ({min(domain_half_y, domain_half_z):.1f} mm). "
            f"Aperture will be clipped by domain boundary."
        )

    cy = ny / 2.0
    cz = nz / 2.0
    yy = jnp.arange(ny, dtype=jnp.float32) - cy
    zz = jnp.arange(nz, dtype=jnp.float32) - cz
    YY, ZZ = jnp.meshgrid(yy, zz, indexing="ij")
    r2       = YY ** 2 + ZZ ** 2
    aperture = (r2 <= radius_vox ** 2).astype(jnp.float32)

    n_active = int(jnp.sum(aperture))
    print(
        f"[source] PZT-26 transducer (Jimenez-Gambin 2019): "
        f"radius={_TRANSDUCER_RADIUS_MM:.1f} mm = {radius_vox:.1f} vox, "
        f"active={n_active} of {ny * nz} px ({100 * n_active / (ny * nz):.1f}%), "
        f"x_slice={source_x_slice}, freq={frequency_hz/1e3:.0f} kHz, dx={dx_mm:.3f} mm"
    )

    src = jnp.zeros((nx, ny, nz, 1), dtype=jnp.complex64)
    src = src.at[source_x_slice, :, :, 0].set(aperture.astype(jnp.complex64))
    return FourierSeries(src, domain)


# ---------------------------------------------------------------------------
# Trainable SIREN MLP + optimization helpers
# ---------------------------------------------------------------------------

def _siren_init_params(
    key: jax.Array,
    layer_sizes: Tuple[int, ...],
    omega0: float,
) -> list[Tuple[jax.Array, jax.Array]]:
    """SIREN init (Sitzmann-style bounds)."""
    params: list[Tuple[jax.Array, jax.Array]] = []
    k = key
    for i in range(len(layer_sizes) - 1):
        k, ki = jax.random.split(k)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        if i == 0:
            bound = 1.0 / fan_in
        elif i == len(layer_sizes) - 2:
            bound = 1.0 / fan_in
        else:
            bound = float(np.sqrt(6.0 / fan_in) / omega0)
        w = jax.random.uniform(ki, (fan_in, fan_out), minval=-bound, maxval=bound, dtype=jnp.float32)
        b = jnp.zeros((fan_out,), dtype=jnp.float32)
        params.append((w, b))
    return params


def _siren_apply(params: list[Tuple[jax.Array, jax.Array]], xyz: jax.Array, omega0: float) -> jax.Array:
    """Apply SIREN on xyz (..., 3) -> alpha (...,) in [0,1]."""
    h = xyz
    for w, b in params[:-1]:
        h = jnp.sin(omega0 * (h @ w + b))
    wl, bl = params[-1]
    out = h @ wl + bl
    return jax.nn.sigmoid(jnp.squeeze(out, axis=-1))


def _focus_sphere_mask(shape: Tuple[int, int, int], radius_vox: int) -> np.ndarray:
    """3D sphere mask around volume center (matches red-circle center in slice plots)."""
    nx, ny, nz = shape
    # Shift the sphere centre away from the exact mid-plane.
    # Use /2.2 (floored to int) for safe integer voxel indexing.
    cx, cy, cz = int(nx / 1.5), int(ny / 1.5), int(nz / 2.0)
    x = np.arange(nx, dtype=np.int32)[:, None, None]
    y = np.arange(ny, dtype=np.int32)[None, :, None]
    z = np.arange(nz, dtype=np.int32)[None, None, :]
    d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return d2 <= int(radius_vox) ** 2


def optimize_lens_with_siren(
    c_base: np.ndarray,
    rho_base: np.ndarray,
    orange_mask_full: np.ndarray,
    dx_mm: float,
    frequency_hz: float,
    source_x_slices: int,
    method: str,
    tol: float,
    restart: int,
    maxiter: int,
    checkpoint: bool,
    n_steps: int,
    lr: float,
    omega0: float,
    hidden: int,
    depth: int,
    seed: int,
    c_solid: float,
    rho_solid: float,
    focus_radius_mm: float,
    use_jit: bool,
    use_transducer_source: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list[float]]:
    """
    End-to-end differentiable pipeline: SIREN only edits voxels where ``orange_mask_full`` is True
    (the orange coupling slab). No separate logical bounding box — indices come from this mask.
    """
    nx, ny, nz = c_base.shape
    if orange_mask_full.shape != (nx, ny, nz):
        print(f"[warn] orange_mask_full shape {orange_mask_full.shape} != c_base {c_base.shape}; skipping optimization.")
        return c_base, rho_base, []
    if not np.any(orange_mask_full):
        print("[warn] Empty orange mask; skipping optimization.")
        return c_base, rho_base, []

    ii, jj, kk = np.nonzero(orange_mask_full)
    n_orange = int(ii.size)
    # Normalized coords in [-1,1] from tight bbox of orange (conditions SIREN).
    i_min, i_max = float(ii.min()), float(ii.max())
    j_min, j_max = float(jj.min()), float(jj.max())
    k_min, k_max = float(kk.min()), float(kk.max())
    den_i = max(i_max - i_min, 1.0)
    den_j = max(j_max - j_min, 1.0)
    den_k = max(k_max - k_min, 1.0)
    xi = (2.0 * (ii.astype(np.float64) - i_min) / den_i - 1.0).astype(np.float32)
    yj = (2.0 * (jj.astype(np.float64) - j_min) / den_j - 1.0).astype(np.float32)
    zk = (2.0 * (kk.astype(np.float64) - k_min) / den_k - 1.0).astype(np.float32)
    lens_xyz = np.stack([xi, yj, zk], axis=1)

    c_base_j = jnp.asarray(c_base, dtype=jnp.float32)
    rho_base_j = jnp.asarray(rho_base, dtype=jnp.float32)

    lens_xyz_j = jnp.asarray(lens_xyz, dtype=jnp.float32)
    ii_i = jnp.asarray(ii.astype(np.int32))
    jj_i = jnp.asarray(jj.astype(np.int32))
    kk_i = jnp.asarray(kk.astype(np.int32))

    print(
        f"[opt] SIREN design: {n_orange:,} orange voxels only (no separate logical box). "
        f"Skull/brain CT frozen outside orange."
    )

    clamp_exempt_j = jnp.asarray(orange_mask_full.astype(np.bool_))

    focus_radius_vox = max(1, int(round(focus_radius_mm / dx_mm)))
    focus_mask = _focus_sphere_mask((nx, ny, nz), radius_vox=focus_radius_vox).astype(np.float32)
    focus_mask_j = jnp.asarray(focus_mask, dtype=jnp.float32)
    focus_mask_sum = jnp.maximum(jnp.sum(focus_mask_j), 1.0)

    nxyz = (nx, ny, nz)
    dx_m = dx_mm * 1e-3
    domain = Domain(nxyz, (dx_m, dx_m, dx_m))
    if use_transducer_source:
        source = _make_transducer_source(
            nxyz=nxyz,
            dx_mm=dx_mm,
            domain=domain,
            frequency_hz=frequency_hz,
            source_x_slice=source_x_slices - 1,
        )
    else:
        src = jnp.zeros((nx, ny, nz, 1), dtype=jnp.complex64)
        src = src.at[:source_x_slices, :, :, 0].set(1.0 + 0.0j)
        source = FourierSeries(src, domain)
    omega = 2.0 * jnp.pi * frequency_hz
    pml = max(10, min(32, min(nxyz) // 8))

    layer_sizes = tuple([3] + [hidden] * max(1, depth) + [1])
    params = _siren_init_params(jax.random.PRNGKey(seed), layer_sizes, omega0=omega0)

    c_water = 1480.0
    rho_water = 1000.0
    history: list[float] = []

    def _loss_fn(p):
        alpha = _siren_apply(p, lens_xyz_j, omega0=omega0)
        c_orange = c_water + alpha * (c_solid - c_water)
        rho_orange = rho_water + alpha * (rho_solid - rho_water)
        c_now = c_base_j.at[ii_i, jj_i, kk_i].set(c_orange)
        rho_now = rho_base_j.at[ii_i, jj_i, kk_i].set(rho_orange)
        c_now = jnp.where((c_now < c_water) & (~clamp_exempt_j), c_water, c_now)
        rho_now = jnp.where((rho_now < rho_water) & (~clamp_exempt_j), rho_water, rho_now)

        medium = Medium(
            domain=domain,
            sound_speed=jnp.expand_dims(c_now, -1),
            density=jnp.expand_dims(rho_now, -1),
            pml_size=pml,
        )
        import time
        t0 = time.time()
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
        print(f"[opt] Helmholtz solve time: {time.time() - t0:.3f}s")
        p_abs = jnp.abs(p_complex.on_grid[..., 0])
        rms = jnp.sqrt(jnp.sum((p_abs ** 2) * focus_mask_j) / (2.0 * focus_mask_sum) + 1e-20)
        loss = -rms
        alpha_min = jnp.min(alpha)
        alpha_max = jnp.max(alpha)
        return loss, (rms, alpha_min, alpha_max)

    step_fn = value_and_grad(_loss_fn, has_aux=True)
    if use_jit:
        step_fn = jax.jit(step_fn)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    print(f"[opt] Lens optimization start: steps={n_steps}, lr={lr:.3e} (Adam), hidden={hidden}, depth={depth}, omega0={omega0}")
    print(
        f"[opt] Orange mask only (tight): x=[{int(ii.min())}:{int(ii.max()) + 1}], "
        f"y=[{int(jj.min())}:{int(jj.max()) + 1}], z=[{int(kk.min())}:{int(kk.max()) + 1}], "
        f"focus radius={focus_radius_vox} vox"
    )
    for step in range(1, n_steps + 1):
        t0 = time.time()
        (loss, aux), grads = step_fn(params)
        print(f"[opt] Step time time: {time.time() - t0:.3f}s")
        rms, a_min, a_max = aux
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss_f = float(loss)
        rms_f = float(rms)
        history.append(loss_f)
        print(
            f"[opt] step {step:03d}/{n_steps:03d} "
            f"loss={loss_f:.4e} focus_rms={rms_f:.4e} alpha=[{float(a_min):.3f},{float(a_max):.3f}] "
            f"dt={time.time() - t0:.1f}s"
        )

    alpha_final = _siren_apply(params, lens_xyz_j, omega0=omega0)
    c_final_orange = c_water + alpha_final * (c_solid - c_water)
    rho_final_orange = rho_water + alpha_final * (rho_solid - rho_water)
    c_final = c_base_j.at[ii_i, jj_i, kk_i].set(c_final_orange)
    rho_final = rho_base_j.at[ii_i, jj_i, kk_i].set(rho_final_orange)
    c_final_np = np.array(c_final, dtype=np.float32)
    rho_final_np = np.array(rho_final, dtype=np.float32)

    # Stats on orange polygon voxels only
    c_orange_np   = float(c_water)   + np.array(alpha_final) * (c_solid   - float(c_water))
    rho_orange_np = float(rho_water) + np.array(alpha_final) * (rho_solid - float(rho_water))
    print(
        f"[opt] Final lens material range (orange polygon, {n_orange:,} vox): "
        f"c=[{float(np.min(c_orange_np)):.1f}, {float(np.max(c_orange_np)):.1f}] m/s, "
        f"rho=[{float(np.min(rho_orange_np)):.1f}, {float(np.max(rho_orange_np)):.1f}] kg/m^3"
    )
    return c_final_np, rho_final_np, history


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


def _discover_attenuation(c_path: Path) -> Optional[Path]:
    """Find attenuation .npy next to the sound speed file (e.g. {stem}_attenuation_cleaned.npy)."""
    stem = c_path.name.replace("_sound_speed_cleaned.npy", "")
    att_path = c_path.with_name(f"{stem}_attenuation_cleaned.npy")
    if att_path.exists():
        return att_path
    std = c_path.with_name("attenuation_db_cm_mhz_xyz.npy")
    if std.exists():
        return std
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
    att_xyz: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    from scipy.ndimage import zoom
    sx, sy, sz = spacing_xyz_mm
    zf = (sx / target_dx_mm, sy / target_dx_mm, sz / target_dx_mm)
    c_iso = zoom(c_xyz, zf, order=1).astype(np.float32)
    rho_iso = zoom(rho_xyz, zf, order=1).astype(np.float32)
    att_iso = zoom(att_xyz, zf, order=1).astype(np.float32) if att_xyz is not None else None
    return c_iso, rho_iso, att_iso


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
    clamp_exempt_mask: Optional[np.ndarray] = None,
    att_xyz: Optional[np.ndarray] = None,
    use_transducer_source: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    # Clamp background/air voxels to water-like values.
    # CT-derived maps have c≈0 and ρ≈0 for air/background; these produce
    # k²=(ω/c)²→∞ which makes the Helmholtz system degenerate and causes
    # BiCGStab to "converge" immediately to a trivially near-zero solution.
    # Floor values: water coupling medium (c=1480 m/s, ρ=1000 kg/m³).
    C_WATER = 1480.0   # m/s
    RHO_WATER = 1000.0  # kg/m³
    if clamp_exempt_mask is not None:
        if clamp_exempt_mask.shape != c_xyz.shape:
            raise ValueError(
                f"clamp_exempt_mask shape mismatch: {clamp_exempt_mask.shape} vs {c_xyz.shape}"
            )
        clamp_where = ~clamp_exempt_mask
    else:
        clamp_where = np.ones_like(c_xyz, dtype=bool)
    n_low_c = int(np.sum((c_xyz < C_WATER) & clamp_where))
    n_low_rho = int(np.sum((rho_xyz < RHO_WATER) & clamp_where))
    if n_low_c > 0:
        print(f"[info] Clamping {n_low_c:,} voxels with c < {C_WATER} m/s to water ({C_WATER} m/s).")
        c_xyz = np.where((c_xyz < C_WATER) & clamp_where, C_WATER, c_xyz)
    if n_low_rho > 0:
        print(f"[info] Clamping {n_low_rho:,} voxels with ρ < {RHO_WATER} kg/m³ to water ({RHO_WATER} kg/m³).")
        rho_xyz = np.where((rho_xyz < RHO_WATER) & clamp_where, RHO_WATER, rho_xyz)

    nxyz = c_xyz.shape
    dx_m = dx_mm * 1e-3
    domain = Domain(nxyz, (dx_m, dx_m, dx_m))

    c_j = jnp.expand_dims(jnp.asarray(c_xyz, dtype=jnp.float32), -1)
    rho_j = jnp.expand_dims(jnp.asarray(rho_xyz, dtype=jnp.float32), -1)

    # PML: scale with domain; min 10, max 32, ~1/8 of smallest dimension.
    # Using the computed value (not hardcoded 12) is critical for proper absorption.
    pml = max(10, min(32, min(nxyz) // 8))
    # Attenuation: optional. Input att_xyz is in dB/cm/MHz; j-wave expects Np/m at the simulation frequency.
    if att_xyz is not None:
        freq_mhz = frequency_hz / 1e6
        att_np_per_m = np.asarray(att_xyz * freq_mhz * (100.0 / 8.686), dtype=np.float32)
        att_j = jnp.expand_dims(jnp.asarray(att_np_per_m, dtype=jnp.float32), -1)
        medium = Medium(domain=domain, sound_speed=c_j, density=rho_j, attenuation=att_j, pml_size=pml)
        print(f"[info] Attenuation in use: max {float(np.max(att_np_per_m)):.4f} Np/m at {frequency_hz/1e3:.0f} kHz")
    else:
        medium = Medium(domain=domain, sound_speed=c_j, density=rho_j, pml_size=pml)

    if use_transducer_source:
        source = _make_transducer_source(
            nxyz=nxyz,
            dx_mm=dx_mm,
            domain=domain,
            frequency_hz=frequency_hz,
            source_x_slice=source_x_slices - 1,
        )
    else:
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
    bath_partition_fracs: Tuple[float, float] = (1.0/3.0, 2.0/3.0),
    c_full: Optional[np.ndarray] = None,
    design_region_vox: Optional[Tuple[int, int, int, int, int, int]] = None,
    lens_depth_vox: int = 0,
    lens_left_min_x: int = 0,
) -> None:
    nx, ny, nz = p_abs.shape
    body_mask_full = body_mask  # full domain for design region / skull surface (before crop alias)

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
    # Match the focus sphere center shift (/2.2) used for focus_mask/metrics.
    # Note: xmid is in the *cropped display* coordinates (after p_abs[x0:]).
    xmid_full = int(nx / 1.5)
    xmid = int(xmid_full - x0)
    xmid = int(np.clip(xmid, 0, nd_x - 1))
    ymid = int(ny / 1.5)
    zmid = int(nz / 2.0)

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

    f1, f2 = bath_partition_fracs  # waterbath1 (0-f1), lens (f1-1) extends to head

    # Skull surface (first tissue along propagation) for curved design region in XY/XZ
    x_surf = None
    if body_mask_full is not None and body_mask_full.shape == (nx, ny, nz):
        no_tissue = ~np.any(body_mask_full.astype(np.int32), axis=0)
        x_surf = np.argmax(body_mask_full.astype(np.int32), axis=0)
        x_surf = np.where(no_tissue, nx, x_surf).astype(np.float64)

    # Lens design region: from bath_partition_fracs or from design_region_vox (CT design box)
    water_bath_boundary_mm = x0 * dx_mm
    if design_region_vox is not None:
        dx0, dx1, dj_lo, dj_hi, dk_lo, dk_hi = design_region_vox
        design_start_x = dx0 * dx_mm
        design_end_x = dx1 * dx_mm
        design_len_x = design_end_x - design_start_x
        design_y_lo = dj_lo * dx_mm
        design_y_hi = dj_hi * dx_mm
        design_z_lo = dk_lo * dx_mm
        design_z_hi = dk_hi * dx_mm
        # When x_surf available, orange = curved polygon (skull); else rectangle.
        use_design_rect = x_surf is None
    else:
        lens_start_mm = water_bath_boundary_mm * f1
        lens_len_mm = water_bath_boundary_mm * (1.0 - f1)
        design_len_x = 0.75 * lens_len_mm
        design_start_x = lens_start_mm + 0.5 * (lens_len_mm - design_len_x)
        design_y_lo = 0.125 * ny * dx_mm
        design_y_hi = 0.875 * ny * dx_mm
        design_z_lo = 0.125 * nz * dx_mm
        design_z_hi = 0.875 * nz * dx_mm
        use_design_rect = False

    def _add_boundary_lines_full_domain(ax, col_idx: int):
        """Draw boundary lines: water|CT and CT end. Lens design is in the CT (orange box)."""
        if col_idx >= 2:
            return
        ct_end_mm = nx * dx_mm
        ax.axvline(water_bath_boundary_mm, color="blue", linewidth=2.0, linestyle="--", alpha=1.0, label="water|CT")
        ax.axvline(ct_end_mm, color="red", linewidth=2.0, linestyle="--", alpha=1.0, label="CT end")
        ylo, yhi = ax.get_ylim()
        ymid_ax = (ylo + yhi) * 0.5
        ax.text(water_bath_boundary_mm / 2.0, ymid_ax, "water", color="blue", ha="center", va="center", fontsize=8)
        ax.text(water_bath_boundary_mm + (ct_end_mm - water_bath_boundary_mm) / 2.0, ymid_ax, "CT", color="gray", ha="center", va="center", fontsize=8)

    def _add_design_subbox(ax, col_idx: int):
        """Draw the lens design region.

        XY / XZ: coupling layer with **left** edge ``max(lens_left_min_x, x_surf−depth)`` per column
        (matches the orange mask), **right** edge ``x_surf``. Not a single vertical at ``lens_x0``,
        so top/bottom left corners close correctly when the skull curve is oblique.

        YZ: unchanged rectangle over the y/z design band at ``x = xmid_full`` (as requested).
        """
        from matplotlib.patches import Rectangle, Polygon

        def _contiguous_runs_1d(j_lo: int, j_hi: int, valid: np.ndarray) -> list[tuple[int, int]]:
            """Indices inclusive [j_lo, j_hi] where valid[j] is True; return list of (ja, jb)."""
            runs: list[tuple[int, int]] = []
            ja = None
            jb = None
            for j in range(j_lo, j_hi + 1):
                if bool(valid[j]):
                    if ja is None:
                        ja = j
                    jb = j
                else:
                    if ja is not None and jb is not None:
                        runs.append((ja, jb))
                        ja = None
                        jb = None
            if ja is not None and jb is not None:
                runs.append((ja, jb))
            return runs

        def _poly_xy(ja: int, jb: int):
            # Left boundary varies per j (matches orange_mask): max(source, skull−depth).
            # A single vertical at design_start_x leaves empty wedges at top/bottom "left corners".
            if lens_depth_vox > 0:
                left_x = np.maximum(
                    float(lens_left_min_x),
                    x_surf[:, zmid].astype(np.float64) - float(lens_depth_vox),
                )
                verts = []
                for j in range(ja, jb + 1):
                    verts.append((float(left_x[j]) * dx_mm, j * dx_mm))
                for j in range(jb, ja - 1, -1):
                    verts.append((float(x_surf[j, zmid]) * dx_mm, j * dx_mm))
                return verts
            verts = [
                (design_start_x, ja * dx_mm),
                (design_start_x, jb * dx_mm),
                (float(x_surf[jb, zmid]) * dx_mm, jb * dx_mm),
            ]
            for j in range(jb - 1, ja - 1, -1):
                verts.append((float(x_surf[j, zmid]) * dx_mm, j * dx_mm))
            verts.append((float(x_surf[ja, zmid]) * dx_mm, ja * dx_mm))
            verts.append((design_start_x, ja * dx_mm))
            return verts

        def _poly_xz(ka: int, kb: int):
            if lens_depth_vox > 0:
                left_z = np.maximum(
                    float(lens_left_min_x),
                    x_surf[ymid, :].astype(np.float64) - float(lens_depth_vox),
                )
                verts = []
                for k in range(ka, kb + 1):
                    verts.append((float(left_z[k]) * dx_mm, k * dx_mm))
                for k in range(kb, ka - 1, -1):
                    verts.append((float(x_surf[ymid, k]) * dx_mm, k * dx_mm))
                return verts
            verts = [
                (design_start_x, ka * dx_mm),
                (design_start_x, kb * dx_mm),
                (float(x_surf[ymid, kb]) * dx_mm, kb * dx_mm),
            ]
            for k in range(kb - 1, ka - 1, -1):
                verts.append((float(x_surf[ymid, k]) * dx_mm, k * dx_mm))
            verts.append((float(x_surf[ymid, ka]) * dx_mm, ka * dx_mm))
            verts.append((design_start_x, ka * dx_mm))
            return verts

        patches: list = []

        if col_idx == 0:
            if use_design_rect:
                patches.append(
                    Rectangle(
                        (design_start_x, design_y_lo),
                        design_len_x,
                        design_y_hi - design_y_lo,
                        fill=False,
                        edgecolor="orange",
                        linewidth=2.0,
                        linestyle="-",
                        alpha=1.0,
                        label="design (75%)",
                    )
                )
            elif x_surf is not None:
                j_lo = max(0, min(ny - 1, int(round(design_y_lo / dx_mm))))
                j_hi = max(0, min(ny - 1, int(round(design_y_hi / dx_mm))))
                j_lo, j_hi = min(j_lo, j_hi), max(j_lo, j_hi)
                valid_y = (x_surf[:, zmid] < nx).astype(bool)
                for ir, (ja, jb) in enumerate(_contiguous_runs_1d(j_lo, j_hi, valid_y)):
                    patches.append(
                        Polygon(
                            _poly_xy(ja, jb),
                            fill=False,
                            edgecolor="orange",
                            linewidth=1.6,
                            linestyle="-",
                            alpha=0.9,
                            zorder=6,
                            label="lens (in front of skull)" if ir == 0 else "_nolegend_",
                        )
                    )
            else:
                patches.append(
                    Rectangle(
                        (design_start_x, design_y_lo),
                        design_len_x,
                        design_y_hi - design_y_lo,
                        fill=False,
                        edgecolor="orange",
                        linewidth=2.0,
                        linestyle="-",
                        alpha=1.0,
                        label="design (75%)",
                    )
                )
        elif col_idx == 1:
            if use_design_rect:
                patches.append(
                    Rectangle(
                        (design_start_x, design_z_lo),
                        design_len_x,
                        design_z_hi - design_z_lo,
                        fill=False,
                        edgecolor="orange",
                        linewidth=2.0,
                        linestyle="-",
                        alpha=1.0,
                        label="design (75%)",
                    )
                )
            elif x_surf is not None:
                k_lo = max(0, min(nz - 1, int(round(design_z_lo / dx_mm))))
                k_hi = max(0, min(nz - 1, int(round(design_z_hi / dx_mm))))
                k_lo, k_hi = min(k_lo, k_hi), max(k_lo, k_hi)
                valid_z = (x_surf[ymid, :] < nx).astype(bool)
                for ir, (ka, kb) in enumerate(_contiguous_runs_1d(k_lo, k_hi, valid_z)):
                    patches.append(
                        Polygon(
                            _poly_xz(ka, kb),
                            fill=False,
                            edgecolor="orange",
                            linewidth=1.6,
                            linestyle="-",
                            alpha=0.9,
                            zorder=6,
                            label="lens (in front of skull)" if ir == 0 else "_nolegend_",
                        )
                    )
            else:
                patches.append(
                    Rectangle(
                        (design_start_x, design_z_lo),
                        design_len_x,
                        design_z_hi - design_z_lo,
                        fill=False,
                        edgecolor="orange",
                        linewidth=2.0,
                        linestyle="-",
                        alpha=1.0,
                        label="design (75%)",
                    )
                )
        else:
            patches.append(
                Rectangle(
                    (design_y_lo, design_z_lo),
                    design_y_hi - design_y_lo,
                    design_z_hi - design_z_lo,
                    fill=False,
                    edgecolor="orange",
                    linewidth=2.0,
                    linestyle="-",
                    alpha=1.0,
                    label="design (75%)",
                )
            )
        for p in patches:
            ax.add_patch(p)

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
        _add_design_subbox(axes[1, i], i)
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
        _add_design_subbox(axes[3, i], i)
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
            _add_design_subbox(axes[4, i], i)
            _add_reference_point(axes[4, i], i)
        print(f"[info] Material viz: c min={c_min:.0f}, max={c_max:.0f} m/s (water baseline ~1480)")

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
    parser.add_argument("--source-x-slices", type=int, default=4, help="Planar source thickness in x slices (sets transducer x position when --transducer-source=1).")
    parser.add_argument("--transducer-source", type=int, default=1, choices=[0, 1], help="If 1 (default), use circular PZT-26 transducer aperture (Jimenez-Gambin 2019, 50 mm). If 0, use planar wave source.")
    parser.add_argument("--water-bath-mm", type=float, default=20.0, help="Water coupling layer prepended at x=0 in mm. Ensures the source fires into water before hitting the skull (default: 20 mm).")
    parser.add_argument(
        "--waterbath1-frac", type=float, default=1.0/3.0,
        help="Fraction of water bath for waterbath1 (0-1). Default 1/3. Must sum with lens-frac to 1.",
    )
    parser.add_argument(
        "--lens-frac", type=float, default=2.0/3.0,
        help="Fraction of water bath for lens (0-1). Lens extends from waterbath1 to head. Default 2/3.",
    )
    parser.add_argument("--opt-steps", type=int, default=10, help="Number of SIREN optimization steps.")
    parser.add_argument("--opt-lr", type=float, default=5e-4, help="Learning rate for SIREN optimization.")
    parser.add_argument("--siren-hidden", type=int, default=16, help="SIREN hidden width.")
    parser.add_argument("--siren-depth", type=int, default=2, help="Number of SIREN hidden layers.")
    parser.add_argument("--siren-omega0", type=float, default=30.0, help="SIREN omega0.")
    parser.add_argument("--siren-seed", type=int, default=42, help="Random seed for SIREN init.")
    parser.add_argument("--lens-c-solid", type=float, default=2500.0, help="Reference solid speed (m/s), c=alpha*c_solid.")
    parser.add_argument("--lens-rho-solid", type=float, default=1200.0, help="Reference solid density (kg/m^3), rho=alpha*rho_solid.")
    parser.add_argument("--focus-radius-mm", type=float, default=4.0, help="Focus sphere radius in mm around red-circle center.")
    parser.add_argument(
        "--lens-ct-depth-mm",
        type=float,
        default=20.0,
        help=(
            "Lens thickness (mm) along propagation: extends LEFT from the skull surface (first CT tissue). "
            "The RIGHT edge stays on the head/skull. This many voxels are PREPENDED as water at x=0 (like --water-bath-mm) "
            "so the domain grows when depth increases; the lens slab is [max(source, skull−depth), skull) per (j,k); "
            "does not advance into the head."
        ),
    )
    parser.add_argument(
        "--design-j-lo-frac", type=float, default=0.125,
        help="Lens design band: start fraction along internal y (0..1), after water bath. Default 0.125 = central 75%% band.",
    )
    parser.add_argument(
        "--design-j-hi-frac", type=float, default=0.875,
        help="Lens design band: end fraction along internal y (0..1). Raise toward 1.0 to include more tissue on the high-y side of the slice.",
    )
    parser.add_argument(
        "--design-k-lo-frac", type=float, default=0.125,
        help="Lens design band: start fraction along internal z (0..1).",
    )
    parser.add_argument(
        "--design-k-hi-frac", type=float, default=0.875,
        help="Lens design band: end fraction along internal z (0..1). Raise toward 1.0 to extend the box toward the high-z edge (often 'top' of axial view; depends on CT orientation).",
    )
    parser.add_argument("--opt-jit", type=int, default=0, choices=[0, 1], help="JIT compile optimization step (faster, more memory).")
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
        default=50.0,
        help="Crop this many mm from each x-side after isotropic resampling (default: 0 = no crop).",
    )
    parser.add_argument(
        "--crop-y-start-mm",
        type=float,
        default=40.0,
        help="Crop this many mm from start of y-axis after isotropic resampling (default: 0 = no crop).",
    )
    parser.add_argument(
        "--crop-z-start-mm",
        type=float,
        default=40.0,
        help="Crop this many mm from start of z-axis after isotropic resampling (default: 0 = no crop).",
    )
    parser.add_argument(
        "--crop-z-end-mm",
        type=float,
        default=25.0,
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

    att_xyz = None
    att_path = _discover_attenuation(c_path)
    if att_path is not None:
        att_xyz = np.load(att_path).astype(np.float32)
        if att_xyz.shape != c_xyz.shape:
            print(f"[warn] Attenuation shape {att_xyz.shape} != c shape {c_xyz.shape}; ignoring attenuation.")
            att_xyz = None
        else:
            print(f"[info] Loaded attenuation: {att_path.name}")

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

    def _resample_and_crop(target_dx: float, att_xyz_in: Optional[np.ndarray] = None):
        """Resample c/rho (and optional att) to target_dx and apply all crops. Returns (c, rho, att)."""
        c, rho, att = _resample_to_isotropic(c_xyz, rho_xyz, spacing_xyz_mm, target_dx_mm=target_dx, att_xyz=att_xyz_in)
        c,   _ = _crop_x_both_sides(c,   dx_mm=target_dx, crop_x_mm_each_side=crop_x_mm)
        rho, _ = _crop_x_both_sides(rho, dx_mm=target_dx, crop_x_mm_each_side=crop_x_mm)
        if att is not None:
            att, _ = _crop_x_both_sides(att, dx_mm=target_dx, crop_x_mm_each_side=crop_x_mm)
        c,   _ = _crop_axis_start(c,   dx_mm=target_dx, crop_mm=crop_y_mm, axis=1)
        rho, _ = _crop_axis_start(rho, dx_mm=target_dx, crop_mm=crop_y_mm, axis=1)
        if att is not None:
            att, _ = _crop_axis_start(att, dx_mm=target_dx, crop_mm=crop_y_mm, axis=1)
        c,   _ = _crop_axis_start(c,   dx_mm=target_dx, crop_mm=crop_z_mm, axis=2)
        rho, _ = _crop_axis_start(rho, dx_mm=target_dx, crop_mm=crop_z_mm, axis=2)
        if att is not None:
            att, _ = _crop_axis_start(att, dx_mm=target_dx, crop_mm=crop_z_mm, axis=2)
        c,   _ = _crop_axis_end(c,   dx_mm=target_dx, crop_mm=crop_z_end_mm, axis=1)
        rho, _ = _crop_axis_end(rho, dx_mm=target_dx, crop_mm=crop_z_end_mm, axis=1)
        if att is not None:
            att, _ = _crop_axis_end(att, dx_mm=target_dx, crop_mm=crop_z_end_mm, axis=1)
        c,   _ = _crop_axis_end(c,   dx_mm=target_dx, crop_mm=crop_x_end_mm, axis=2)
        rho, _ = _crop_axis_end(rho, dx_mm=target_dx, crop_mm=crop_x_end_mm, axis=2)
        if att is not None:
            att, _ = _crop_axis_end(att, dx_mm=target_dx, crop_mm=crop_x_end_mm, axis=2)
        return c, rho, att

    # Resample + crop at the requested dx first, then check the budget.
    # Doing crops BEFORE auto-relax means the budget check sees the reduced
    # domain, so the auto-relax can stay at a finer dx (or skip entirely).
    c_iso, rho_iso, att_iso = _resample_and_crop(dx_mm, att_xyz)
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
                c_iso, rho_iso, att_iso = _resample_and_crop(dx_mm, att_xyz)
                break
            dx_mm = dx_mm_new
            c_iso, rho_iso, att_iso = _resample_and_crop(dx_mm, att_xyz)
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
        if att_iso is not None:
            att_iso = np.moveaxis(att_iso, prop_axis, 0)
        if body_mask is not None:
            body_mask = np.moveaxis(body_mask, prop_axis, 0)

    if bool(args.flip_propagation_axis):
        # np.flip returns a view; ascontiguousarray makes it a proper C array
        # so downstream concatenation (water bath) works without issues.
        c_iso   = np.ascontiguousarray(np.flip(c_iso,   axis=0))
        rho_iso = np.ascontiguousarray(np.flip(rho_iso, axis=0))
        if att_iso is not None:
            att_iso = np.ascontiguousarray(np.flip(att_iso, axis=0))
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

    # Lens thickness (mm) → extra prepend layers (same mechanism as water bath: grows the domain).
    lens_ct_depth_mm = float(args.lens_ct_depth_mm)
    lens_depth_vox = max(1, int(round(lens_ct_depth_mm / dx_mm)))
    lens_prepend_vox = lens_depth_vox

    # Prepend (1) lens coupling water, then (2) water coupling bath at x=0.
    # A CT volume typically fills its full FOV, so x=0 is right at the scalp/skull.
    # Without a water gap, the unit-source at x=0:source_x_slices fires directly
    # into bone, making wave injection completely inefficient.  We prepend water
    # so the source launches a plane wave that then impinges on the skull at normal
    # incidence with proper far-field development.
    water_bath_only_vox = max(int(args.source_x_slices) + 4,
                              int(round(float(args.water_bath_mm) / dx_mm)))
    bath_fracs = (
        float(args.waterbath1_frac),
        float(args.lens_frac),
    )
    frac_sum = sum(bath_fracs)
    if abs(frac_sum - 1.0) > 0.001:
        bath_fracs = tuple(f / frac_sum for f in bath_fracs)
    lens_x0 = 0
    lens_x1 = 0
    design_j_lo, design_j_hi = 0, 0
    design_k_lo, design_k_hi = 0, 0
    source_x_slices_int = int(args.source_x_slices)

    def _prepend_water(nx_pre: int, label: str) -> None:
        nonlocal c_iso, rho_iso, att_iso, body_mask
        if nx_pre <= 0:
            return
        wc = np.full((nx_pre, c_iso.shape[1], c_iso.shape[2]), 1480.0, dtype=np.float32)
        wr = np.full((nx_pre, rho_iso.shape[1], rho_iso.shape[2]), 1000.0, dtype=np.float32)
        c_iso = np.concatenate([wc, c_iso], axis=0)
        rho_iso = np.concatenate([wr, rho_iso], axis=0)
        if att_iso is not None:
            wa = np.zeros((nx_pre, att_iso.shape[1], att_iso.shape[2]), dtype=np.float32)
            att_iso = np.concatenate([wa, att_iso], axis=0)
        if body_mask is not None:
            wm = np.zeros((nx_pre, body_mask.shape[1], body_mask.shape[2]), dtype=bool)
            body_mask = np.concatenate([wm, body_mask], axis=0)
        print(f"[info] Prepended {nx_pre} vox ({nx_pre * dx_mm:.1f} mm) {label} at x=0.")

    if lens_prepend_vox > 0:
        _prepend_water(lens_prepend_vox, "lens coupling (water, expands domain with --lens-ct-depth-mm)")
    if water_bath_only_vox > 0:
        _prepend_water(water_bath_only_vox, "water coupling bath")
    if lens_prepend_vox > 0 or water_bath_only_vox > 0:
        print(f"[info] Domain after prepends: shape={c_iso.shape}, voxels={np.prod(c_iso.shape):,}")

    nx_post = c_iso.shape[0]
    ny_post = c_iso.shape[1]
    nz_post = c_iso.shape[2]
    # First index of original CT stack after both prepends (water|CT boundary for plots).
    ct_start_idx = lens_prepend_vox + water_bath_only_vox
    # Back-compat name: total "bath" before CT used to be water_bath_vox only; now CT starts after lens+bath.
    water_bath_vox = ct_start_idx

    # Lens: RIGHT edge = skull (first tissue). LEFT = skull − depth; domain grows by lens_prepend_vox so
    # increasing --lens-ct-depth-mm prepends more voxels (like water bath) instead of only stealing CT slab.
    design_j_lo, design_j_hi = _design_band_indices_from_fracs(
        ny_post, args.design_j_lo_frac, args.design_j_hi_frac
    )
    design_k_lo, design_k_hi = _design_band_indices_from_fracs(
        nz_post, args.design_k_lo_frac, args.design_k_hi_frac
    )
    # Skull surface: first x where body is True. Design region = orange_mask_full (no separate logical box).
    x_surf = None
    skull_surface_x_train = water_bath_vox  # fallback: skull starts at first CT index (after lens prepend + bath)
    if body_mask is not None and body_mask.shape == (nx_post, ny_post, nz_post):
        no_tissue = ~np.any(body_mask.astype(np.int32), axis=0)
        x_surf = np.argmax(body_mask.astype(np.int32), axis=0)
        x_surf = np.where(no_tissue, nx_post, x_surf).astype(np.float64)
        valid_surf = x_surf[x_surf < nx_post]
        if valid_surf.size > 0:
            skull_surface_x_train = int(np.median(valid_surf))

    orange_mask_full = _orange_mask_full_volume(
        nx_post,
        ny_post,
        nz_post,
        x_surf,
        nx_post,
        design_j_lo,
        design_j_hi,
        design_k_lo,
        design_k_hi,
        source_x_slices_int,
        lens_depth_vox,
        water_bath_vox,
        min(nx_post, water_bath_vox + lens_depth_vox),
    )
    if x_surf is None and np.any(orange_mask_full):
        print(
            "[warn] No body mask / skull surface — orange region is the fallback x-slab in the y/z band (not skull-attached)."
        )

    _tb = _tight_bbox_from_mask(orange_mask_full)
    lens_x0 = 0
    lens_x1 = 0
    design_j_lo = design_j_hi = design_k_lo = design_k_hi = 0
    if _tb is not None:
        lens_x0, lens_x1, design_j_lo, design_j_hi, design_k_lo, design_k_hi = _tb

    if lens_x1 > lens_x0:
        print(
            f"[info] Lens = orange mask (tight AABB): x=[{lens_x0}:{lens_x1}], y=[{design_j_lo}:{design_j_hi}], "
            f"z=[{design_k_lo}:{design_k_hi}], {int(np.sum(orange_mask_full)):,} voxels, "
            f"≤{lens_depth_vox} vox depth ({lens_ct_depth_mm:.2f} mm)."
        )
    # Debug: propagation layout for plotting
    prop_len_mm = nx_post * dx_mm
    print(
        "[debug] PLOT LAYOUT (XY and XZ): horizontal axis = propagation (internal axis 0). "
        "Source fires at index 0. Left of plot = index 0 = source end. Right = index N-1 = far end."
    )
    if ct_start_idx > 0:
        _pp = []
        if lens_prepend_vox > 0:
            _pp.append(f"lens prepend {lens_prepend_vox} vox [0..{lens_prepend_vox - 1}]")
        if water_bath_only_vox > 0:
            _pp.append(
                f"bath {water_bath_only_vox} vox [{lens_prepend_vox}..{ct_start_idx - 1}]"
            )
        print(
            f"[debug] Prepends: {', '.join(_pp)}; CT from index {ct_start_idx} "
            f"({ct_start_idx * dx_mm:.1f} mm) to {nx_post - 1}."
        )
    else:
        print(f"[debug] No prepends. CT spans 0..{nx_post - 1} (0..{prop_len_mm:.1f} mm).")
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

    opt_history: list[float] = []
    orange_mask = (
        orange_mask_full[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].copy()
        if lens_x1 > lens_x0
        else np.zeros((0, 0, 0), dtype=bool)
    )
    lens_clamp_exempt_mask = np.asarray(orange_mask_full, dtype=bool)
    if lens_x1 > lens_x0 and int(args.opt_steps) == 0:
        c_iso = np.asarray(c_iso, dtype=np.float32)
        rho_iso = np.asarray(rho_iso, dtype=np.float32)
        c_iso[orange_mask_full] = float(args.lens_c_solid)
        rho_iso[orange_mask_full] = float(args.lens_rho_solid)
        print(
            f"[info] 0 opt-steps: painted orange mask only ({int(np.sum(orange_mask_full))} voxels) for visual check."
        )
    elif lens_x1 > lens_x0 and int(args.opt_steps) > 0:
        c_iso, rho_iso, opt_history = optimize_lens_with_siren(
            c_base=c_iso,
            rho_base=rho_iso,
            orange_mask_full=np.asarray(orange_mask_full, dtype=bool),
            dx_mm=dx_mm,
            frequency_hz=float(args.frequency),
            source_x_slices=int(args.source_x_slices),
            method=args.method,
            tol=float(args.tol),
            restart=int(args.restart),
            maxiter=int(args.maxiter),
            checkpoint=bool(args.checkpoint),
            n_steps=int(args.opt_steps),
            lr=float(args.opt_lr),
            omega0=float(args.siren_omega0),
            hidden=int(args.siren_hidden),
            depth=int(args.siren_depth),
            seed=int(args.siren_seed),
            c_solid=float(args.lens_c_solid),
            rho_solid=float(args.lens_rho_solid),
            focus_radius_mm=float(args.focus_radius_mm),
            use_jit=bool(args.opt_jit),
            use_transducer_source=bool(args.transducer_source),
        )
    else:
        print("[warn] Lens optimization skipped (empty orange mask).")

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
        clamp_exempt_mask=lens_clamp_exempt_mask,
        att_xyz=att_iso,
        use_transducer_source=bool(args.transducer_source),
    )

    fig_slices = out_dir / "helmholtz_pressure_slices.png"
    bath_fracs = (
        float(args.waterbath1_frac),
        float(args.lens_frac),
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
        design_region_vox=(lens_x0, lens_x1, design_j_lo, design_j_hi, design_k_lo, design_k_hi) if lens_x1 > lens_x0 else None,
        lens_depth_vox=int(lens_depth_vox) if lens_x1 > lens_x0 else 0,
        lens_left_min_x=source_x_slices_int,
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
        "crop_z_end_mm": float(crop_z_end_mm),
        "crop_x_end_mm": float(crop_x_end_mm),
        "propagation_axis": int(args.propagation_axis),
        "flip_propagation_axis": bool(args.flip_propagation_axis),
        "spacing_before_mm": list(spacing_xyz_mm),
        "method": args.method,
        "tol": float(args.tol),
        "restart": int(args.restart),
        "maxiter": int(args.maxiter),
        "checkpoint": bool(args.checkpoint),
        "body_mask_used": mask_path is not None,
        "siren_lens_applied": lens_x1 > lens_x0,
        "water_bath_vox": water_bath_vox,
        "lens_prepend_vox": int(lens_prepend_vox),
        "water_bath_only_vox": int(water_bath_only_vox),
        "ct_start_idx": int(ct_start_idx),
        "bath_partition_fracs": list(bath_fracs),
        "lens_x_range": [int(lens_x0), int(lens_x1)],
        "lens_depth_mm": float(lens_ct_depth_mm),
        "lens_depth_vox": int(lens_depth_vox),
        "lens_right_attached_skull_surface": bool(x_surf is not None and lens_x1 > lens_x0),
        "lens_design_yz_range": [
            int(design_j_lo),
            int(design_j_hi),
            int(design_k_lo),
            int(design_k_hi),
        ],
        "design_band_fracs": {
            "j_lo": float(args.design_j_lo_frac),
            "j_hi": float(args.design_j_hi_frac),
            "k_lo": float(args.design_k_lo_frac),
            "k_hi": float(args.design_k_hi_frac),
        },
        "opt_steps": int(args.opt_steps),
        "opt_lr": float(args.opt_lr),
        "opt_history_loss": [float(v) for v in opt_history],
        "max_pressure_abs": float(np.max(p_abs[np.isfinite(p_abs)])) if np.any(np.isfinite(p_abs)) else 0.0,
        "outputs": [
            "helmholtz_pressure_slices.png",
            "summary.json",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if lens_x1 > lens_x0:
        lens_c = np.asarray(
            c_iso[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi],
            dtype=np.float32,
        )
        lens_rho = np.asarray(
            rho_iso[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi],
            dtype=np.float32,
        )
        np.save(out_dir / "lens_sound_speed_ms.npy", lens_c)
        np.save(out_dir / "lens_density_kgm3.npy", lens_rho)
        # Save the orange polygon mask so transplant_eval can selectively paste only
        # the pre-skull lens voxels, preserving each target patient's skull anatomy.
        np.save(out_dir / "lens_orange_mask.npy", orange_mask.astype(np.bool_))
        lens_npy = ["lens_sound_speed_ms.npy", "lens_density_kgm3.npy", "lens_orange_mask.npy"]
        print(
            f"[info] Saved lens material arrays ({lens_c.shape}): "
            f"{lens_npy[0]}, {lens_npy[1]}, {lens_npy[2]} in {out_dir}"
        )
        print(
            f"[info] Orange mask: {int(np.sum(orange_mask))} of "
            f"{orange_mask.size} voxels active "
            f"({100*np.sum(orange_mask)/max(orange_mask.size,1):.1f}%)"
        )
        summary["outputs"] = list(summary["outputs"]) + lens_npy
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        lens_meta = {
            "lens_right_attached_skull_surface": x_surf is not None,
            "lens_prepend_vox": int(lens_prepend_vox),
            "lens_depth_mm": float(lens_ct_depth_mm),
            "lens_depth_vox": int(lens_depth_vox),
            "lens_x0": lens_x0,
            "lens_x1": lens_x1,
            "design_j_lo": design_j_lo,
            "design_j_hi": design_j_hi,
            "design_k_lo": design_k_lo,
            "design_k_hi": design_k_hi,
            "design_j_lo_frac": float(args.design_j_lo_frac),
            "design_j_hi_frac": float(args.design_j_hi_frac),
            "design_k_lo_frac": float(args.design_k_lo_frac),
            "design_k_hi_frac": float(args.design_k_hi_frac),
            "dx_mm": dx_mm,
            # Median skull-surface x (in training domain) used to align the lens
            # to each target patient's skull surface during transplant.
            "skull_surface_x_train": skull_surface_x_train,
        }

        json.dump(lens_meta, open(out_dir / "lens_meta.json", "w"), indent=2)

    print(f"[info] Isotropic shape={c_iso.shape}, dx={dx_mm:.4f} mm")
    print(f"[info] Body mask used: {mask_path is not None}")
    print(f"Check?")


if __name__ == "__main__":
    main()

