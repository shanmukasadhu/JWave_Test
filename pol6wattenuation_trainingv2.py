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


def _focus_sphere_mask(
    shape: Tuple[int, int, int],
    radius_vox: int,
    center: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """3D sphere mask around a given voxel center (defaults to volume center)."""
    nx, ny, nz = shape
    cx, cy, cz = center if center is not None else (nx // 2, ny // 2, nz // 2)
    x = np.arange(nx, dtype=np.int32)[:, None, None]
    y = np.arange(ny, dtype=np.int32)[None, :, None]
    z = np.arange(nz, dtype=np.int32)[None, None, :]
    d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return d2 <= int(radius_vox) ** 2


def optimize_lens_with_siren(
    c_base: np.ndarray,
    rho_base: np.ndarray,
    lens_x0: int,
    lens_x1: int,
    design_j_lo: int,
    design_j_hi: int,
    design_k_lo: int,
    design_k_hi: int,
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
    x_surf: Optional[np.ndarray] = None,
    nx_post: int = 0,
    focus_centers_vox: Optional[list] = None,
    results_log_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, list[float]]:
    """
    End-to-end differentiable pipeline:
      Design region = [lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi] (e.g. CT + 75%% y/z).
      (x,y,z) -> SIREN -> alpha -> material in that box -> Helmholtz -> negative RMS at focus -> backprop.
    No clamp is applied inside the lens design space.
    """
    if lens_x1 <= lens_x0 or design_j_hi <= design_j_lo or design_k_hi <= design_k_lo:
        print("[warn] Empty lens design region; skipping optimization.")
        return c_base, rho_base, []

    nx, ny, nz = c_base.shape
    lx = lens_x1 - lens_x0
    ly = design_j_hi - design_j_lo
    lz = design_k_hi - design_k_lo
    xs = np.linspace(-1.0, 1.0, lx, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, ly, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, lz, dtype=np.float32)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    lens_xyz_full = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)

    c_base_j = jnp.asarray(c_base, dtype=jnp.float32)
    rho_base_j = jnp.asarray(rho_base, dtype=jnp.float32)

    # Orange polygon mask: i < x_surf(j,k) or i == lens_x0 (left edge) so corners are filled; exclude background.
    design_region_mask_j = None
    lens_xyz_j = jnp.asarray(lens_xyz_full, dtype=jnp.float32)
    orange_flat_indices = None
    if x_surf is not None and nx_post > 0:
        x_surf_slice = x_surf[design_j_lo:design_j_hi, design_k_lo:design_k_hi]
        ii_idx = np.arange(lx, dtype=np.float64) + lens_x0
        in_front = ii_idx[:, None, None] < x_surf_slice[None, :, :]
        left_edge = (ii_idx == lens_x0)[:, None, None]
        has_tissue = (x_surf_slice < nx_post)[None, :, :]
        design_region_mask = (in_front | left_edge) & has_tissue
        design_region_mask_j = jnp.asarray(design_region_mask)
        n_design = int(np.sum(design_region_mask))
        print(f"[opt] Design restricted to orange polygon: {n_design} voxels (of {lx*ly*lz} in box).")
        # SIREN inputs: only orange voxels
        orange_flat_indices = np.flatnonzero(design_region_mask)
        lens_xyz_j = jnp.asarray(lens_xyz_full[orange_flat_indices], dtype=jnp.float32)
    clamp_exempt_mask = np.zeros((nx, ny, nz), dtype=bool)
    if design_region_mask_j is not None:
        clamp_exempt_mask[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi] = np.array(design_region_mask_j)
    else:
        clamp_exempt_mask[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi] = True
    clamp_exempt_j = jnp.asarray(clamp_exempt_mask)

    focus_radius_vox = max(1, int(round(focus_radius_mm / dx_mm)))

    # Build per-focus masks. Default: single focus at volume center.
    if focus_centers_vox is None or len(focus_centers_vox) == 0:
        focus_centers_vox = [(nx // 2, ny // 2, nz // 2)]

    n_foci = len(focus_centers_vox)
    focus_masks_np = [
        _focus_sphere_mask((nx, ny, nz), radius_vox=focus_radius_vox, center=tuple(c)).astype(np.float32)
        for c in focus_centers_vox
    ]
    focus_masks_j = [jnp.asarray(m, dtype=jnp.float32) for m in focus_masks_np]
    focus_mask_sums_j = [jnp.maximum(jnp.sum(m), 1.0) for m in focus_masks_j]
    print(f"[opt] Foci: {n_foci}  centers={focus_centers_vox}  radius={focus_radius_vox} vox ({focus_radius_mm:.1f} mm)")
    for i, m in enumerate(focus_masks_np):
        print(f"[opt]   Focus {i+1}: {int(m.sum())} voxels")

    nxyz = (nx, ny, nz)
    dx_m = dx_mm * 1e-3
    domain = Domain(nxyz, (dx_m, dx_m, dx_m))
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
        alpha_masked = _siren_apply(p, lens_xyz_j, omega0=omega0)
        if design_region_mask_j is not None:
            # Scatter SIREN output into full box (unmasked positions ignored via jnp.where below)
            alpha_full_flat = jnp.zeros(lx * ly * lz).at[orange_flat_indices].set(alpha_masked)
            alpha = alpha_full_flat.reshape((lx, ly, lz))
        else:
            alpha = alpha_masked.reshape((lx, ly, lz))

        c_lens = c_water + alpha * (c_solid - c_water)
        rho_lens = rho_water + alpha * (rho_solid - rho_water)
        if design_region_mask_j is not None:
            c_slice = c_base_j[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi]
            rho_slice = rho_base_j[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi]
            c_new = jnp.where(design_region_mask_j, c_lens, c_slice)
            rho_new = jnp.where(design_region_mask_j, rho_lens, rho_slice)
            c_now = c_base_j.at[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].set(c_new)
            rho_now = rho_base_j.at[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].set(rho_new)
        else:
            c_now = c_base_j.at[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].set(c_lens)
            rho_now = rho_base_j.at[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].set(rho_lens)

        # Clamp only outside design space; keep design space unclamped.
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
        p_sq = p_abs ** 2
        # Per-focus RMS; loss = mean across all foci
        per_focus_rms = [
            jnp.sqrt(jnp.sum(p_sq * focus_masks_j[i]) / (2.0 * focus_mask_sums_j[i]) + 1e-20)
            for i in range(n_foci)
        ]
        avg_rms = sum(per_focus_rms) / n_foci
        loss = -avg_rms
        alpha_min = jnp.min(alpha_masked)
        alpha_max = jnp.max(alpha_masked)
        return loss, (avg_rms, alpha_min, alpha_max, *per_focus_rms)

    step_fn = value_and_grad(_loss_fn, has_aux=True)
    if use_jit:
        step_fn = jax.jit(step_fn)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    print(f"[opt] Lens optimization start: steps={n_steps}, lr={lr:.3e} (Adam), hidden={hidden}, depth={depth}, omega0={omega0}")
    print(f"[opt] Design region: x=[{lens_x0}:{lens_x1}] ({lx} vox), y=[{design_j_lo}:{design_j_hi}] ({ly}), z=[{design_k_lo}:{design_k_hi}] ({lz}), focus radius={focus_radius_vox} vox")

    # Open results log file (append mode so multiple runs accumulate).
    log_fh = None
    if results_log_path is not None:
        log_fh = open(results_log_path, "a", buffering=1)
        import datetime
        log_fh.write(f"\n{'='*70}\n")
        log_fh.write(f"RUN START: {datetime.datetime.now().isoformat()}\n")
        log_fh.write(f"frequency_hz: {frequency_hz}\n")
        log_fh.write(f"dx_mm: {dx_mm}\n")
        log_fh.write(f"domain_shape: ({nx}, {ny}, {nz})\n")
        log_fh.write(f"n_foci: {n_foci}\n")
        for i, c in enumerate(focus_centers_vox):
            log_fh.write(f"focus_{i+1}_vox: {c}\n")
        log_fh.write(f"focus_radius_mm: {focus_radius_mm}  radius_vox: {focus_radius_vox}\n")
        log_fh.write(f"n_steps: {n_steps}  lr: {lr}  hidden: {hidden}  depth: {depth}  omega0: {omega0}\n")
        log_fh.write(f"lens_x: [{lens_x0}:{lens_x1}]  y: [{design_j_lo}:{design_j_hi}]  z: [{design_k_lo}:{design_k_hi}]\n")
        log_fh.write(f"{'='*70}\n")
        # Header row
        focus_cols = "  ".join(f"rms_f{i+1}" for i in range(n_foci))
        log_fh.write(f"step  loss        avg_rms     {focus_cols}  alpha_min  alpha_max  elapsed_s\n")
        log_fh.flush()

    rms_first_run = None  # track for improvement %
    for step in range(1, n_steps + 1):
        t0 = time.time()
        (loss, aux), grads = step_fn(params)
        elapsed = time.time() - t0
        print(f"[opt] Step time time: {elapsed:.3f}s")
        avg_rms = aux[0]
        a_min, a_max = aux[1], aux[2]
        per_focus_rms_vals = [float(aux[3 + i]) for i in range(n_foci)]
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss_f = float(loss)
        rms_f = float(avg_rms)
        if rms_first_run is None:
            rms_first_run = rms_f
        history.append(loss_f)
        focus_rms_str = "  ".join(f"{v:.4e}" for v in per_focus_rms_vals)
        print(
            f"[opt] step {step:03d}/{n_steps:03d} "
            f"loss={loss_f:.4e} avg_rms={rms_f:.4e} "
            f"[{focus_rms_str}] "
            f"alpha=[{float(a_min):.3f},{float(a_max):.3f}] "
            f"dt={elapsed:.1f}s"
        )
        if log_fh is not None:
            focus_log_cols = "  ".join(f"{v:.6e}" for v in per_focus_rms_vals)
            log_fh.write(
                f"{step:04d}  {loss_f:+.6e}  {rms_f:.6e}  {focus_log_cols}"
                f"  {float(a_min):.4f}     {float(a_max):.4f}     {elapsed:.2f}\n"
            )
            log_fh.flush()

    if log_fh is not None:
        improvement_pct = (rms_f / max(rms_first_run, 1e-30) - 1.0) * 100.0 if rms_first_run else 0.0
        log_fh.write(f"{'='*70}\n")
        log_fh.write(f"OPTIMIZATION COMPLETE: {n_steps} steps\n")
        log_fh.write(f"initial_avg_rms: {rms_first_run:.6e}\n")
        log_fh.write(f"final_avg_rms:   {rms_f:.6e}\n")
        log_fh.write(f"improvement:     {improvement_pct:+.1f}%\n")
        log_fh.write(f"{'='*70}\n")
        log_fh.close()

    alpha_masked_final = _siren_apply(params, lens_xyz_j, omega0=omega0)
    if design_region_mask_j is not None:
        alpha_full_flat = jnp.zeros(lx * ly * lz).at[orange_flat_indices].set(alpha_masked_final)
        alpha_final = alpha_full_flat.reshape((lx, ly, lz))
    else:
        alpha_final = alpha_masked_final.reshape((lx, ly, lz))
    c_lens_final = c_water + alpha_final * (c_solid - c_water)
    rho_lens_final = rho_water + alpha_final * (rho_solid - rho_water)
    if design_region_mask_j is not None:
        c_slice = c_base_j[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi]
        rho_slice = rho_base_j[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi]
        c_final = c_base_j.at[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].set(
            jnp.where(design_region_mask_j, c_lens_final, c_slice)
        )
        rho_final = rho_base_j.at[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].set(
            jnp.where(design_region_mask_j, rho_lens_final, rho_slice)
        )
    else:
        c_final = c_base_j.at[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].set(c_lens_final)
        rho_final = rho_base_j.at[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].set(rho_lens_final)
    c_final_np = np.array(c_final, dtype=np.float32)
    rho_final_np = np.array(rho_final, dtype=np.float32)
    box_slice = (slice(lens_x0, lens_x1), slice(design_j_lo, design_j_hi), slice(design_k_lo, design_k_hi))
    if design_region_mask_j is not None:
        c_des = c_final_np[box_slice][np.array(design_region_mask_j)]
        rho_des = rho_final_np[box_slice][np.array(design_region_mask_j)]
    else:
        c_des = c_final_np[box_slice]
        rho_des = rho_final_np[box_slice]
    print(
        f"[opt] Final lens material range (orange region): "
        f"c=[{float(np.min(c_des)):.1f}, {float(np.max(c_des)):.1f}] m/s, "
        f"rho=[{float(np.min(rho_des)):.1f}, {float(np.max(rho_des)):.1f}] kg/m^3"
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
    bath_partition_fracs: Tuple[float, float] = (1.0/3.0, 2.0/3.0),
    c_full: Optional[np.ndarray] = None,
    design_region_vox: Optional[Tuple[int, int, int, int, int, int]] = None,
    focus_centers_vox: Optional[list] = None,
    focus_radius_mm: float = 4.0,
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
        """Draw the lens design region: rectangle when design_region_vox given, else curved right edge if x_surf available."""
        from matplotlib.patches import Rectangle, Polygon
        if col_idx == 0:
            if use_design_rect:
                patch = Rectangle(
                    (design_start_x, design_y_lo), design_len_x, design_y_hi - design_y_lo,
                    fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0, label="design (75%)",
                )
            elif x_surf is not None:
                j_lo = max(0, min(ny - 1, int(round(design_y_lo / dx_mm))))
                j_hi = max(0, min(ny - 1, int(round(design_y_hi / dx_mm))))
                j_lo, j_hi = min(j_lo, j_hi), max(j_lo, j_hi)
                verts = [
                    (design_start_x, design_y_lo),
                    (design_start_x, design_y_hi),
                    (float(x_surf[j_hi, zmid]) * dx_mm, design_y_hi),
                ]
                for j in range(j_hi - 1, j_lo - 1, -1):
                    verts.append((float(x_surf[j, zmid]) * dx_mm, j * dx_mm))
                verts.append((float(x_surf[j_lo, zmid]) * dx_mm, design_y_lo))
                verts.append((design_start_x, design_y_lo))
                patch = Polygon(verts, fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0, label="design (75%)")
            else:
                patch = Rectangle(
                    (design_start_x, design_y_lo), design_len_x, design_y_hi - design_y_lo,
                    fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0, label="design (75%)",
                )
        elif col_idx == 1:
            if use_design_rect:
                patch = Rectangle(
                    (design_start_x, design_z_lo), design_len_x, design_z_hi - design_z_lo,
                    fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0, label="design (75%)",
                )
            elif x_surf is not None:
                k_lo = max(0, min(nz - 1, int(round(design_z_lo / dx_mm))))
                k_hi = max(0, min(nz - 1, int(round(design_z_hi / dx_mm))))
                k_lo, k_hi = min(k_lo, k_hi), max(k_lo, k_hi)
                verts = [
                    (design_start_x, design_z_lo),
                    (design_start_x, design_z_hi),
                    (float(x_surf[ymid, k_hi]) * dx_mm, design_z_hi),
                ]
                for k in range(k_hi - 1, k_lo - 1, -1):
                    verts.append((float(x_surf[ymid, k]) * dx_mm, k * dx_mm))
                verts.append((float(x_surf[ymid, k_lo]) * dx_mm, design_z_lo))
                verts.append((design_start_x, design_z_lo))
                patch = Polygon(verts, fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0, label="design (75%)")
            else:
                patch = Rectangle(
                    (design_start_x, design_z_lo), design_len_x, design_z_hi - design_z_lo,
                    fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0, label="design (75%)",
                )
        else:
            # YZ: always rectangle
            patch = Rectangle(
                (design_y_lo, design_z_lo), design_y_hi - design_y_lo, design_z_hi - design_z_lo,
                fill=False, edgecolor="orange", linewidth=2.0, linestyle="-", alpha=1.0, label="design (75%)",
            )
        ax.add_patch(patch)

    # Build focus-circle data from focus_centers_vox (if provided) or fall back to
    # the single red circle at volume center that was previously always shown.
    from matplotlib.patches import Circle
    _focus_colors = ["red", "lime", "cyan", "yellow", "magenta"]
    if focus_centers_vox is not None and len(focus_centers_vox) > 0:
        _foci_mm = [
            ((c[0] + 0.5) * dx_mm, (c[1] + 0.5) * dx_mm, (c[2] + 0.5) * dx_mm)
            for c in focus_centers_vox
        ]
    else:
        # Legacy single-focus at volume center
        _foci_mm = [((xmid_full + 0.5) * dx_mm, (ymid + 0.5) * dx_mm, (zmid + 0.5) * dx_mm)]

    def _add_reference_point(ax, col_idx: int):
        """Draw one circle per focus center.  col_idx: 0=XY, 1=XZ, 2=YZ."""
        for fi, (fx_mm, fy_mm, fz_mm) in enumerate(_foci_mm):
            color = _focus_colors[fi % len(_focus_colors)]
            lw = 2.5 if fi == 0 else 2.0
            ls = "-" if fi == 0 else "--"
            if col_idx == 0:    # XY: x-axis=propagation, y-axis=y
                cx, cy = fx_mm, fy_mm
            elif col_idx == 1:  # XZ: x-axis=propagation, y-axis=z
                cx, cy = fx_mm, fz_mm
            else:               # YZ: x-axis=y, y-axis=z
                cx, cy = fy_mm, fz_mm
            circle = Circle(
                (cx, cy), focus_radius_mm,
                fill=False, edgecolor=color, linewidth=lw, linestyle=ls,
                label=f"focus {fi+1}" if col_idx == 0 else None,
            )
            ax.add_patch(circle)
            # Label each circle so the two foci are distinguishable
            ax.annotate(
                f"F{fi+1}", xy=(cx, cy + focus_radius_mm),
                color=color, fontsize=7, ha="center", va="bottom",
                fontweight="bold",
            )

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
    parser.add_argument("--source-x-slices", type=int, default=4, help="Planar source thickness in x slices.")
    parser.add_argument("--water-bath-mm", type=float, default=10.0, help="Water coupling layer prepended at x=0 in mm. Ensures the source fires into water before hitting the skull (default: 20 mm).")
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
    parser.add_argument("--focus2-delta-y-mm", type=float, default=15.0, help="Second focus center: Y offset in mm from first focus (default: +15 mm).")
    parser.add_argument("--focus2-delta-z-mm", type=float, default=0.0, help="Second focus center: Z offset in mm from first focus (default: 0 mm).")
    parser.add_argument("--focus2-delta-x-mm", type=float, default=0.0, help="Second focus center: X offset in mm from first focus (default: 0 mm).")
    parser.add_argument("--dual-focus", type=int, default=1, choices=[0, 1], help="Enable dual-focus optimization (default: 1). Set 0 for single focus.")
    parser.add_argument("--lens-ct-depth-mm", type=float, default=10.0, help="Depth (mm) into CT for lens design region. Design is in CT (orange box), not water bath.")
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
        default=100.0,
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
    )
    frac_sum = sum(bath_fracs)
    if abs(frac_sum - 1.0) > 0.001:
        bath_fracs = tuple(f / frac_sum for f in bath_fracs)
    lens_x0 = 0
    lens_x1 = 0
    design_j_lo, design_j_hi = 0, 0
    design_k_lo, design_k_hi = 0, 0
    if water_bath_vox > 0:
        bath_c   = np.full((water_bath_vox, c_iso.shape[1],   c_iso.shape[2]),   1480.0, dtype=np.float32)
        bath_rho = np.full((water_bath_vox, rho_iso.shape[1], rho_iso.shape[2]), 1000.0, dtype=np.float32)
        c_iso   = np.concatenate([bath_c,   c_iso],   axis=0)
        rho_iso = np.concatenate([bath_rho, rho_iso], axis=0)
        if att_iso is not None:
            bath_att = np.zeros((water_bath_vox, att_iso.shape[1], att_iso.shape[2]), dtype=np.float32)
            att_iso  = np.concatenate([bath_att, att_iso], axis=0)
        if body_mask is not None:
            bath_mask = np.zeros((water_bath_vox, body_mask.shape[1], body_mask.shape[2]), dtype=bool)
            body_mask = np.concatenate([bath_mask, body_mask], axis=0)
        print(f"[info] Prepended {water_bath_vox} vox ({water_bath_vox * dx_mm:.1f} mm) water coupling bath at x=0.")
        print(f"[info] Domain after bath: shape={c_iso.shape}, voxels={np.prod(c_iso.shape):,}")
    nx_post = c_iso.shape[0]
    ny_post = c_iso.shape[1]
    nz_post = c_iso.shape[2]
    # Lens design region = orange polygon: from water|CT boundary to skull surface, 75% y/z.
    # Bounding box in x: water_bath_vox to water_bath_vox + lens_ct_depth_mm (we mask to skull inside).
    lens_ct_depth_mm = float(args.lens_ct_depth_mm)
    lens_depth_vox = max(1, min(nx_post - water_bath_vox, int(round(lens_ct_depth_mm / dx_mm))))
    lens_x0 = water_bath_vox
    lens_x1 = lens_x0 + lens_depth_vox
    design_j_lo = max(0, int(0.125 * ny_post))
    design_j_hi = min(ny_post, int(0.875 * ny_post))
    design_k_lo = max(0, int(0.125 * nz_post))
    design_k_hi = min(nz_post, int(0.875 * nz_post))
    # Skull surface: first x where body is True. Design only in front of skull (orange polygon).
    x_surf = None
    if body_mask is not None and body_mask.shape == (nx_post, ny_post, nz_post):
        no_tissue = ~np.any(body_mask.astype(np.int32), axis=0)
        x_surf = np.argmax(body_mask.astype(np.int32), axis=0)
        x_surf = np.where(no_tissue, nx_post, x_surf).astype(np.float64)
        # Extend lens_x1 so the design box covers the full skull curve in the 75% band
        # (fixes top-right and bottom-right corners of orange in XY view).
        x_surf_band = x_surf[design_j_lo:design_j_hi, design_k_lo:design_k_hi]
        valid = x_surf_band < nx_post
        if np.any(valid):
            x_surf_max_in_band = int(np.max(x_surf_band[valid]))
            lens_x1 = min(nx_post, max(lens_x1, x_surf_max_in_band))
    if lens_x1 > lens_x0:
        print(f"[info] Lens design region (orange polygon): x=[{lens_x0}:{lens_x1}], y=[{design_j_lo}:{design_j_hi}], z=[{design_k_lo}:{design_k_hi}], mask=in front of skull.")
    # Debug: propagation layout for plotting
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

    # Mask for orange polygon only: in front of skull, and exclude background columns (no tissue).
    # Include left-edge voxel (i=lens_x0) for every (j,k) in band so corners of the orange are filled.
    def _orange_region_mask():
        m = np.zeros((lens_x1 - lens_x0, design_j_hi - design_j_lo, design_k_hi - design_k_lo), dtype=bool)
        if x_surf is None:
            m[:] = True
            return m
        x_surf_slice = x_surf[design_j_lo:design_j_hi, design_k_lo:design_k_hi]  # (ly, lz)
        ii = np.arange(lens_x0, lens_x1, dtype=np.float64)[:, None, None]
        has_tissue = (x_surf_slice < nx_post)
        in_front = (ii < x_surf_slice)
        left_edge = (ii == lens_x0)  # always include left edge so corners get at least one voxel
        m[:] = (in_front | left_edge) & has_tissue[None, :, :]
        return m

    # Compute focus centers in voxel coordinates.
    # Focus 1: volume center.
    cx1 = nx_post // 2
    cy1 = ny_post // 2
    cz1 = nz_post // 2
    focus_centers: list = [(cx1, cy1, cz1)]
    if bool(args.dual_focus):
        dx2 = int(round(float(args.focus2_delta_x_mm) / dx_mm))
        dy2 = int(round(float(args.focus2_delta_y_mm) / dx_mm))
        dz2 = int(round(float(args.focus2_delta_z_mm) / dx_mm))
        cx2 = max(0, min(nx_post - 1, cx1 + dx2))
        cy2 = max(0, min(ny_post - 1, cy1 + dy2))
        cz2 = max(0, min(nz_post - 1, cz1 + dz2))
        focus_centers.append((cx2, cy2, cz2))
        print(f"[info] Dual-focus enabled. Focus 1: {focus_centers[0]}, Focus 2: {focus_centers[1]}")
    else:
        print(f"[info] Single-focus mode. Focus: {focus_centers[0]}")

    results_log_path = out_dir / "results.txt"
    print(f"[info] Per-epoch stats will be appended to: {results_log_path}")

    opt_history: list[float] = []
    orange_mask = _orange_region_mask() if lens_x1 > lens_x0 else np.zeros((0, 0, 0), dtype=bool)
    lens_clamp_exempt_mask = np.zeros_like(c_iso, dtype=bool)
    if lens_x1 > lens_x0:
        lens_clamp_exempt_mask[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi] = orange_mask
    if lens_x1 > lens_x0 and int(args.opt_steps) == 0:
        # 0 opt-steps: paint only the orange polygon with c_solid/rho_solid.
        box_c = c_iso[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].copy()
        box_rho = rho_iso[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi].copy()
        box_c[orange_mask] = float(args.lens_c_solid)
        box_rho[orange_mask] = float(args.lens_rho_solid)
        c_iso[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi] = box_c
        rho_iso[lens_x0:lens_x1, design_j_lo:design_j_hi, design_k_lo:design_k_hi] = box_rho
        print(f"[info] 0 opt-steps: painted orange polygon only ({np.sum(orange_mask)} voxels) for visual check.")
    elif lens_x1 > lens_x0 and int(args.opt_steps) > 0:
        c_iso, rho_iso, opt_history = optimize_lens_with_siren(
            c_base=c_iso,
            rho_base=rho_iso,
            lens_x0=lens_x0,
            lens_x1=lens_x1,
            design_j_lo=design_j_lo,
            design_j_hi=design_j_hi,
            design_k_lo=design_k_lo,
            design_k_hi=design_k_hi,
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
            x_surf=x_surf,
            nx_post=nx_post,
            focus_centers_vox=focus_centers,
            results_log_path=results_log_path,
        )
    else:
        print("[warn] Lens optimization skipped (empty lens region or --opt-steps=0).")

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
        focus_centers_vox=focus_centers,
        focus_radius_mm=float(args.focus_radius_mm),
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
        "siren_lens_applied": lens_x1 > lens_x0,
        "water_bath_vox": water_bath_vox,
        "bath_partition_fracs": list(bath_fracs),
        "lens_x_range": [int(lens_x0), int(lens_x1)],
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

    print(f"[info] Isotropic shape={c_iso.shape}, dx={dx_mm:.4f} mm")
    print(f"[info] Body mask used: {mask_path is not None}")

    # Append post-solve summary to results.txt
    with open(results_log_path, "a", buffering=1) as log_fh:
        import datetime
        log_fh.write(f"\n{'='*70}\n")
        log_fh.write(f"POST-SOLVE SUMMARY: {datetime.datetime.now().isoformat()}\n")
        log_fh.write(f"final_domain_shape: {c_iso.shape}\n")
        log_fh.write(f"dx_mm_used: {dx_mm:.4f}\n")
        log_fh.write(f"max_pressure_abs: {float(np.max(p_abs[np.isfinite(p_abs)])) if np.any(np.isfinite(p_abs)) else 0.0:.6e}\n")
        log_fh.write(f"focus_centers_vox: {focus_centers}\n")
        focus_radius_vox_final = max(1, int(round(float(args.focus_radius_mm) / dx_mm)))
        for i, c in enumerate(focus_centers):
            mask_np = _focus_sphere_mask(c_iso.shape, radius_vox=focus_radius_vox_final, center=c)
            in_focus = p_abs[mask_np]
            in_focus = in_focus[np.isfinite(in_focus)]
            rms_focus = float(np.sqrt(np.mean(in_focus ** 2) / 2.0)) if in_focus.size > 0 else 0.0
            log_fh.write(f"focus_{i+1}_final_rms: {rms_focus:.6e} Pa  (center={c}, n_vox={int(mask_np.sum())})\n")
        log_fh.write(f"output_figure: {fig_slices}\n")
        log_fh.write(f"{'='*70}\n")


if __name__ == "__main__":
    main()

