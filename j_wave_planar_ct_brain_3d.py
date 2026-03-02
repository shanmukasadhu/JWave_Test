"""
Continuous planar-wave simulation through preprocessed CT-brain acoustic maps.

Outputs:
1) Stabilized pressure XY/YZ/XZ slices

Primary mode expects preprocessed maps:
  --input-dir containing:
    sound_speed_xyz.npy
    density_xyz.npy
    attenuation_db_cm_mhz_xyz.npy   (optional for plotting; not used by solver)
    metadata.json                    (optional; spacing read if present)

Example:
  python3 Code/j_wave_planar_ct_brain_3d.py \
    --input-dir "Results/jwave_brain_forward_64/body_only_air" \
    --out-dir "Results/jwave_planar_brain_3d"

Auto-discover latest cleaned maps from j_wave_brain_forward_64:
  python3 Code/j_wave_planar_ct_brain_3d.py
"""

from __future__ import annotations

# ── XLA memory settings MUST be set before JAX is imported ──────────────────
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# ────────────────────────────────────────────────────────────────────────────

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from jax import lax
from jax import numpy as jnp

_J_WAVE_IMPORT_ERROR = None
try:
    from jwave import FourierSeries
    from jwave.acoustics.time_varying import simulate_wave_propagation
    from jwave.geometry import Domain, Medium, TimeAxis
except Exception as _exc:  # pragma: no cover - runtime environment dependent
    _J_WAVE_IMPORT_ERROR = _exc
    FourierSeries = None
    simulate_wave_propagation = None
    Domain = None
    Medium = None
    TimeAxis = None

def _discover_input_maps(
    input_dir: Path | None,
    auto_root: Path,
    patient_id: str | None,
) -> Tuple[Path, Path, Path]:
    """
    Resolve sound_speed and density map paths.
    Priority:
      1) Explicit input_dir with standard filenames
      2) Explicit input_dir with <patient>_sound_speed_cleaned.npy naming
      3) Auto-discover latest *_sound_speed_cleaned.npy under auto_root
    """
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

        raise FileNotFoundError(
            f"Could not find acoustic maps in {input_dir}. Expected sound_speed_xyz.npy/density_xyz.npy "
            f"or *_sound_speed_cleaned.npy/*_density_cleaned.npy."
        )

    # Auto mode from root
    if not auto_root.exists():
        raise FileNotFoundError(f"Auto root does not exist: {auto_root}")

    pattern = f"{''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in (patient_id or ''))}_sound_speed_cleaned.npy"
    if patient_id:
        explicit = list(auto_root.rglob(pattern))
        explicit = sorted(explicit, key=lambda p: p.stat().st_mtime, reverse=True)
        if explicit:
            c_path, rho_path = _pair_from_cleaned(explicit[0])
            return c_path, rho_path, explicit[0].parent

    cleaned = sorted(auto_root.rglob("*_sound_speed_cleaned.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cleaned:
        c_path, rho_path = _pair_from_cleaned(cleaned[0])
        return c_path, rho_path, cleaned[0].parent

    # fallback to standard names anywhere under root
    std = sorted(auto_root.rglob("sound_speed_xyz.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c_path in std:
        rho_path = c_path.with_name("density_xyz.npy")
        if rho_path.exists():
            return c_path, rho_path, c_path.parent

    raise FileNotFoundError(
        f"No suitable acoustic maps found under auto root: {auto_root}"
    )


def _load_spacing_from_metadata(input_dir: Path) -> Tuple[float, float, float]:
    meta_path = input_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            spacing = meta.get("spacing_xyz_mm", None)
            if isinstance(spacing, list) and len(spacing) == 3:
                return float(spacing[0]), float(spacing[1]), float(spacing[2])
        except Exception:
            pass
    # Also support spacing in summary.json produced by prior scripts.
    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        try:
            meta = json.loads(summary_path.read_text(encoding="utf-8"))
            spacing = meta.get("ct_spacing_xyz_mm", None)
            if isinstance(spacing, list) and len(spacing) == 3:
                return float(spacing[0]), float(spacing[1]), float(spacing[2])
        except Exception:
            pass
    return 1.0, 1.0, 1.0


def _resample_to_isotropic(
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    spacing_xyz_mm: Tuple[float, float, float],
    target_dx_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import zoom

    sx, sy, sz = spacing_xyz_mm
    zx = sx / target_dx_mm
    zy = sy / target_dx_mm
    zz = sz / target_dx_mm
    c_iso = zoom(c_xyz, (zx, zy, zz), order=1).astype(np.float32)
    rho_iso = zoom(rho_xyz, (zx, zy, zz), order=1).astype(np.float32)
    return c_iso, rho_iso


def _estimate_bytes(n: int) -> int:
    # Peak device memory during one step of the online-accumulation scan.
    # Observed on A6000 (48 GB): 567M voxels → 65.6 GB XLA peak ≈ 116 bytes/voxel.
    # Breakdown:
    #   Carry (p, u, rho)                         =  7 × float32 = 28 bytes/voxel
    #   Static (pml_rho, pml_u)                   =  6 × float32 = 24 bytes/voxel
    #   momentum FFT intermediates (3 dims ×
    #     FFT + k-op + IFFT, complex64)            =               ≈ 36 bytes/voxel
    #   mass FFT intermediates (same)              =               ≈ 36 bytes/voxel
    #   misc / XLA allocator overhead              =               ≈ 12 bytes/voxel
    #   Observed total: ≈ 116 bytes/voxel → use 180 (1.55× safety factor)
    return int(n * 180)


def _estimate_sensor_history_gb(shape_xyz: Tuple[int, int, int], n_time_steps: int) -> float:
    """Estimate memory for storing 3 center-slice sensor outputs across all time steps."""
    nx, ny, nz = shape_xyz
    slice_voxels = nx * ny + nx * nz + ny * nz
    return float(slice_voxels * n_time_steps * 4 / 1e9)


def _available_memory_bytes() -> int:
    """
    Best-effort memory detection for auto memory budgets.

    Precedence:
      1. GPU VRAM (via nvidia-smi) — used when JAX detects a GPU backend,
         because the solver runs on the GPU and VRAM is the binding resource.
      2. psutil available RAM.
      3. POSIX sysconf (SC_AVPHYS_PAGES).
      4. POSIX sysconf total pages × 0.5.
      5. macOS vm_stat parsing.
    """
    # GPU VRAM — highest priority when a GPU is present.
    try:
        import jax as _jax
        if _jax.default_backend() == "gpu":
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
                if lines:
                    vram_mb = int(lines[0])  # free VRAM of first GPU, in MiB
                    # Reserve 15 % for JAX/CUDA allocator overhead and
                    # XLA intermediate buffers not captured by our estimate.
                    return int(vram_mb * 1024 * 1024 * 0.85)
    except Exception:
        pass

    try:
        import psutil  # type: ignore

        avail = int(psutil.virtual_memory().available)
        if avail > 0:
            return avail
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass

    try:
        # Fallback for systems that expose only total pages.
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages * page_size * 0.5)
    except (AttributeError, OSError, ValueError):
        pass

    try:
        # macOS fallback without extra dependencies.
        vm_stat_out = subprocess.check_output(["vm_stat"], text=True)
        page_size_match = re.search(r"page size of (\d+) bytes", vm_stat_out)
        if page_size_match is not None:
            page_size = int(page_size_match.group(1))
            page_counts = {
                "Pages free": 0,
                "Pages inactive": 0,
                "Pages speculative": 0,
            }
            for line in vm_stat_out.splitlines():
                for key in page_counts:
                    if line.startswith(key):
                        num = line.split(":", 1)[1].strip().rstrip(".")
                        page_counts[key] = int(num.replace(".", ""))
            avail_pages = int(sum(page_counts.values()))
            avail = avail_pages * page_size
            if avail > 0:
                return avail
    except Exception:
        pass

    raise RuntimeError(
        "Unable to detect available system memory automatically. "
        "Please set --max-voxels explicitly."
    )


def _auto_memory_budgets() -> Tuple[int, float]:
    available_bytes = _available_memory_bytes()
    max_voxels = max(1, available_bytes // _estimate_bytes(1))
    max_history_gb = float(available_bytes / 1e9)
    return int(max_voxels), float(max_history_gb)


def _recommended_dx_mm(
    shape_xyz: Tuple[int, int, int],
    spacing_xyz_mm: Tuple[float, float, float],
    max_voxels: int,
) -> float:
    sx, sy, sz = spacing_xyz_mm
    nx, ny, nz = shape_xyz
    # vox(dx) = (nx*sx/dx)*(ny*sy/dx)*(nz*sz/dx) = K / dx^3
    k = float(nx * sx * ny * sy * nz * sz)
    return float((k / max(max_voxels, 1)) ** (1.0 / 3.0))


def run_planar_wave_3d(
    c_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    dx_m: float,
    frequency_hz: float,
    cfl: float,
    source_x_slices: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """
    Run the simulation with true online accumulation — no scan output buffer
    is ever allocated on the device.  Instead of stacking field values across
    time steps, the scan carry holds three running-sum arrays (one per centre
    slice) that are updated in-place at every step.

    Peak device memory is O(N³) — the current solver state only — regardless
    of how many time steps are taken.  This makes the approach viable on GPU
    because XLA never needs to allocate nt × slice_size bytes upfront.

    Returns:
        p_stab_xy  – mean(|p|) at z=zmid, shape (nx, ny)
        p_stab_xz  – mean(|p|) at y=ymid, shape (nx, nz)
        p_stab_yz  – mean(|p|) at x=xmid, shape (ny, nz)
        t_arr      – time array, shape (nt,)
        stab_idx   – first time index used for stabilised average
        t_end      – actual simulation end time
        stab_time  – physical stabilisation time
    """
    from jwave.acoustics.time_varying import (
        momentum_conservation_rhs,
        mass_conservation_rhs,
        pressure_from_density,
    )
    from jwave.acoustics.spectral import kspace_op
    from jwave.acoustics.pml import td_pml_on_grid

    nxyz = c_xyz.shape
    xmid, ymid, zmid = nxyz[0] // 2, nxyz[1] // 2, nxyz[2] // 2

    domain = Domain(nxyz, (dx_m, dx_m, dx_m))
    c_j   = jnp.expand_dims(jnp.asarray(c_xyz),   -1)
    rho_j = jnp.expand_dims(jnp.asarray(rho_xyz), -1)
    pml_size = max(8, min(24, min(nxyz) // 6))
    medium = Medium(domain=domain, sound_speed=c_j, density=rho_j, pml_size=pml_size)

    c_min = float(np.min(c_xyz[np.isfinite(c_xyz)]))
    c_min = max(c_min, 343.0)
    lx = nxyz[0] * dx_m
    t_cross   = lx / c_min
    # 5 extra cycles after crossing is enough when combined with the source
    # ramp; previously 8 cycles — the 3-cycle savings compound over t_cross.
    t_end     = t_cross + 5.0 / frequency_hz
    stab_time = t_cross + 1.5 / frequency_hz

    time_axis = TimeAxis.from_medium(
        Medium(domain=domain, sound_speed=float(np.max(c_xyz))), cfl=cfl, t_end=t_end
    )
    t_arr    = np.asarray(time_axis.to_array())
    dt       = float(time_axis.dt)
    nt       = int(time_axis.Nt)
    stab_idx = int(np.argmin(np.abs(t_arr - stab_time)))

    # Pre-compute static parameters, mirroring jwave's fourier_wave_prop_params.
    c_ref        = float(np.max(c_xyz))
    pml_rho      = FourierSeries(td_pml_on_grid(medium, dt, c0=c_ref, dx=dx_m, coord_shift=0.0), domain)
    pml_u        = FourierSeries(td_pml_on_grid(medium, dt, c0=c_ref, dx=dx_m, coord_shift=0.5), domain)
    fourier_params = kspace_op(domain, c_ref, dt)

    # Zero initial conditions (p0 = 0 ⟹ u0 = 0, rho0 = 0 exactly).
    shape_1 = tuple(list(nxyz) + [1])
    shape_3 = tuple(list(nxyz) + [3])
    p0   = FourierSeries(jnp.zeros(shape_1, dtype=jnp.float32), domain)
    u0   = FourierSeries(jnp.zeros(shape_3, dtype=jnp.float32), domain)
    rho0 = FourierSeries(jnp.zeros(shape_3, dtype=jnp.float32), domain)

    class TimeVaryingSource:
        def __init__(self, ta):
            self.omega   = 2.0 * jnp.pi * frequency_hz
            self.ta_arr  = ta.to_array()
            # Hann-window ramp over the first 3 source cycles eliminates the
            # broadband startup transient, letting the field stabilise faster.
            self.t_ramp  = jnp.float32(3.0 / frequency_hz)
            src_mask = jnp.zeros(nxyz, dtype=jnp.float32).at[:source_x_slices, :, :].set(1.0)
            self.src_mask = jnp.expand_dims(src_mask, -1)

        def on_grid(self, ti):
            tidx     = lax.convert_element_type(ti, jnp.int32)
            t_scalar = jnp.squeeze(lax.dynamic_slice(self.ta_arr, (tidx,), (1,)))
            # Hann ramp: 0→1 over t_ramp, then 1 permanently.
            phase    = jnp.clip(t_scalar / self.t_ramp, 0.0, 1.0)
            ramp     = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
            return self.src_mask * ramp * jnp.sin(self.omega * t_scalar)

    src = TimeVaryingSource(time_axis)

    # Online-accumulation scan.
    # Carry: [p, u, rho, sum_xy, sum_xz, sum_yz, count]
    # Output: a dummy scalar — no per-step buffer allocated on device.
    sum_xy_init = jnp.zeros((nxyz[0], nxyz[1]), dtype=jnp.float32)
    sum_xz_init = jnp.zeros((nxyz[0], nxyz[2]), dtype=jnp.float32)
    sum_yz_init = jnp.zeros((nxyz[1], nxyz[2]), dtype=jnp.float32)
    count_init  = jnp.int32(0)

    def scan_fn(carry, n):
        p, u, rho, sum_xy, sum_xz, sum_yz, count = carry

        mass_src = src.on_grid(n)

        # Physics update — identical to jwave's internal scan_fun.
        du   = momentum_conservation_rhs(p, u, medium, c_ref=c_ref, dt=dt, params=fourier_params)
        u    = pml_u * (pml_u * u + dt * du)
        drho = mass_conservation_rhs(p, u, mass_src, medium, c_ref=c_ref, dt=dt, params=fourier_params)
        rho  = pml_rho * (pml_rho * rho + dt * drho)
        p    = pressure_from_density(rho, medium)

        # Accumulate mean |p| at the three centre planes, but only after
        # stabilisation.  Multiplying by a 0/1 mask avoids lax.cond and
        # keeps the XLA graph simple.
        p_3d   = p.params[..., 0]
        w      = (n >= stab_idx).astype(jnp.float32)
        sum_xy = sum_xy + w * jnp.abs(p_3d[:, :, zmid])
        sum_xz = sum_xz + w * jnp.abs(p_3d[:, ymid, :])
        sum_yz = sum_yz + w * jnp.abs(p_3d[xmid, :, :])
        count  = count + jnp.int32(w)

        return [p, u, rho, sum_xy, sum_xz, sum_yz, count], jnp.float32(0.0)

    output_steps = jnp.arange(0, nt, 1)
    init_carry   = [p0, u0, rho0, sum_xy_init, sum_xz_init, sum_yz_init, count_init]
    final_carry, _ = lax.scan(scan_fn, init_carry, output_steps)
    _, _, _, sum_xy, sum_xz, sum_yz, count = final_carry

    denom = jnp.maximum(count, jnp.int32(1)).astype(jnp.float32)
    p_stab_xy = np.nan_to_num(np.asarray(sum_xy / denom, dtype=np.float32))
    p_stab_xz = np.nan_to_num(np.asarray(sum_xz / denom, dtype=np.float32))
    p_stab_yz = np.nan_to_num(np.asarray(sum_yz / denom, dtype=np.float32))

    return p_stab_xy, p_stab_xz, p_stab_yz, t_arr, stab_idx, t_end, stab_time


def plot_stabilized_slices(
    p_stab_xy: np.ndarray,
    p_stab_xz: np.ndarray,
    p_stab_yz: np.ndarray,
    dx_mm: float,
    out_path: Path,
) -> None:
    """Plot stabilized mean |p| for the three center-plane slices.

    Args:
        p_stab_xy: mean(|p|) at z=zmid, shape (nx, ny)
        p_stab_xz: mean(|p|) at y=ymid, shape (nx, nz)
        p_stab_yz: mean(|p|) at x=xmid, shape (ny, nz)
    """
    nx, ny = p_stab_xy.shape
    nz = p_stab_xz.shape[1]
    extent_xy = [0, nx * dx_mm, 0, ny * dx_mm]
    extent_xz = [0, nx * dx_mm, 0, nz * dx_mm]
    extent_yz = [0, ny * dx_mm, 0, nz * dx_mm]

    all_vals = np.concatenate([p_stab_xy.ravel(), p_stab_xz.ravel(), p_stab_yz.ravel()])
    finite_vals = all_vals[np.isfinite(all_vals)]
    vmax = float(np.percentile(np.abs(finite_vals), 99.5)) if finite_vals.size else 1e-8
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im = axes[0].imshow(p_stab_xy.T, origin="lower", cmap="hot", extent=extent_xy, vmin=0, vmax=vmax)
    axes[0].set_title("Stabilized |P| XY (z=center)")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].imshow(p_stab_xz.T, origin="lower", cmap="hot", extent=extent_xz, vmin=0, vmax=vmax)
    axes[1].set_title("Stabilized |P| XZ (y=center)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    im = axes[2].imshow(p_stab_yz.T, origin="lower", cmap="hot", extent=extent_yz, vmin=0, vmax=vmax)
    axes[2].set_title("Stabilized |P| YZ (x=center)")
    axes[2].set_xlabel("y (mm)")
    axes[2].set_ylabel("z (mm)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle("Continuous planar wave after stabilization")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if _J_WAVE_IMPORT_ERROR is not None:
        py = sys.executable
        ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "Failed to import jwave stack. This is usually an environment mismatch.\n"
            f"Current python: {py} (version {ver})\n"
            "Your traceback indicates python 3.13, which is commonly incompatible with current "
            "jax/equinox/jaxtyping versions used by jwave.\n\n"
            "Run this script with your jwave conda env interpreter, e.g.:\n"
            "  /Users/shanmukasadhu/miniconda3/envs/jwave/bin/python "
            "Code/j_wave_planar_ct_brain_3d.py\n\n"
            f"Original import error:\n{repr(_J_WAVE_IMPORT_ERROR)}"
        )

    parser = argparse.ArgumentParser(description="Planar-wave simulation through preprocessed CT-brain maps.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with preprocessed sound_speed_xyz.npy and density_xyz.npy. "
             "If omitted, auto-discover from --auto-root.",
    )
    parser.add_argument(
        "--auto-root",
        default="Results/jwave_brain_forward_64",
        help="Root folder used for auto-discovery when --input-dir is omitted.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Optional patient id to prefer matching <patient>_sound_speed_cleaned.npy in auto mode.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <input-dir>/planar_wave_3d).")
    parser.add_argument("--dx-mm", type=float, default=0.4, help="Target isotropic dx in mm (default: 0.4; gives ~6 PPW at 650 kHz in tissue).")
    parser.add_argument(
        "--auto-relax-dx",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, automatically increase dx when max-voxels would be exceeded.",
    )
    parser.add_argument("--frequency", type=float, default=6.5e5, help="Wave frequency in Hz (default: 650 kHz).")
    parser.add_argument("--cfl", type=float, default=0.3, help="CFL number (default: 0.3; safe for PSTD at 650 kHz, 33%% fewer steps than 0.2).")
    parser.add_argument("--source-x-slices", type=int, default=6, help="Planar source thickness in x-grid slices.")
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=None,
        help="Safety cap on voxel count after isotropic resampling. "
             "If omitted, auto-uses currently available RAM.",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
    auto_root = Path(args.auto_root).expanduser().resolve()
    c_path, rho_path, resolved_dir = _discover_input_maps(in_dir, auto_root, args.patient_id)
    in_dir = resolved_dir
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (in_dir / "planar_wave_3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    c_xyz = np.load(c_path).astype(np.float32)
    rho_xyz = np.load(rho_path).astype(np.float32)
    if c_xyz.shape != rho_xyz.shape:
        raise ValueError(f"Shape mismatch: c={c_xyz.shape}, rho={rho_xyz.shape}")

    auto_max_voxels, _ = _auto_memory_budgets()
    max_voxels_budget = int(args.max_voxels) if args.max_voxels is not None else int(auto_max_voxels)

    spacing_xyz_mm = _load_spacing_from_metadata(in_dir)
    print("Loading...")
    target_dx_mm = float(args.dx_mm)
    c_iso, rho_iso = _resample_to_isotropic(c_xyz, rho_xyz, spacing_xyz_mm, target_dx_mm=target_dx_mm)
    vox = int(np.prod(c_iso.shape))
    if vox > max_voxels_budget:
        min_dx = _recommended_dx_mm(tuple(c_xyz.shape), spacing_xyz_mm, max_voxels_budget)
        if int(args.auto_relax_dx) == 1 and min_dx > target_dx_mm:
            target_dx_mm = min_dx * 1.02  # tiny safety margin
            print(
                f"[warn] Requested dx={args.dx_mm:.4f} mm exceeds max-voxels budget. "
                f"Auto-relaxing to dx={target_dx_mm:.4f} mm."
            )
            c_iso, rho_iso = _resample_to_isotropic(c_xyz, rho_xyz, spacing_xyz_mm, target_dx_mm=target_dx_mm)
            vox = int(np.prod(c_iso.shape))
        if vox > max_voxels_budget:
            est_gb = _estimate_bytes(vox) / 1e9
            raise RuntimeError(
                f"Resampled volume too large: shape={c_iso.shape} ({vox:,} voxels), est memory ~{est_gb:.1f} GB.\n"
                f"Requested dx={args.dx_mm:.4f} mm, spacing={spacing_xyz_mm}, source_shape={c_xyz.shape}\n"
                f"For this max-voxels budget, recommended dx >= {min_dx:.4f} mm.\n"
                "Options:\n"
                "  1) rerun with larger dx (e.g. --dx-mm {:.4f})\n"
                "  2) increase --max-voxels (if RAM allows)\n"
                "  3) use --auto-relax-dx 1".format(min_dx)
            )

    # Final resample with settled dx
    c_iso, rho_iso = _resample_to_isotropic(c_xyz, rho_xyz, spacing_xyz_mm, target_dx_mm=target_dx_mm)
    dx_m = float(target_dx_mm) * 1e-3
    p_stab_xy, p_stab_xz, p_stab_yz, t_arr, stab_idx, t_end_used, stab_time_used = run_planar_wave_3d(
        c_iso,
        rho_iso,
        dx_m=dx_m,
        frequency_hz=float(args.frequency),
        cfl=float(args.cfl),
        source_x_slices=int(args.source_x_slices),
    )

    stabilized_fig = out_dir / "pressure_stabilized_slices_xy_yz_xz.png"
    plot_stabilized_slices(
        p_stab_xy, p_stab_xz, p_stab_yz,
        dx_mm=float(target_dx_mm),
        out_path=stabilized_fig,
    )

    print(f"[ok] Saved stabilized slices: {stabilized_fig}")
    print(f"[info] max_voxels_budget={max_voxels_budget:,}")
    print(f"[info] isotropic shape={c_iso.shape}, steps={len(t_arr)}, stab_idx={stab_idx}")


if __name__ == "__main__":
    main()

