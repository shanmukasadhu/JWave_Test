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
    print("Creating domain...")
    nxyz = c_xyz.shape
    print(f"Domain shape: {nxyz}")
    print(f"Domain shape(90%): ({nxyz[0] * 0.9}, {nxyz[1] * 0.9}, {nxyz[2] * 0.9})")
    print(f"Domain shape(80%): ({nxyz[0] * 0.8}, {nxyz[1] * 0.8}, {nxyz[2] * 0.8})")
    print(f"Domain shape(70%): ({nxyz[0] * 0.7}, {nxyz[1] * 0.7}, {nxyz[2] * 0.7})")
    dx_m = dx_mm * 1e-3
    domain = Domain(nxyz, (dx_m, dx_m, dx_m))
    print("Creating medium...")
    c_j = jnp.expand_dims(jnp.asarray(c_xyz, dtype=jnp.float32), -1)
    rho_j = jnp.expand_dims(jnp.asarray(rho_xyz, dtype=jnp.float32), -1)
    pml = max(8, min(24, min(nxyz) // 6))
    medium = Medium(domain=domain, sound_speed=c_j, density=rho_j, pml_size=12)
    print("Creating source...")
    src = jnp.zeros((*nxyz, 1), dtype=jnp.complex64)
    src = src.at[:source_x_slices, :, :, 0].set(-1.0j)
    source = FourierSeries(src, domain)
    print("Solving Helmholtz equation...")
    time_start = time.time()
    omega = 2.0 * jnp.pi * frequency_hz
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
    # Compute abs and phase in JAX before transferring to avoid pulling the
    # full complex64 array to CPU.
    p_on_grid = p_complex.on_grid[..., 0]
    p_abs_jax = jnp.abs(p_on_grid)
    p_phase_jax = jnp.angle(p_on_grid)

    print("Waiting for JAX to finish computing...")
    p_abs_jax.block_until_ready()
    print(f"JAX solver done in {time.time() - time_start:.1f}s. Transferring to numpy...")

    p_abs = np.array(p_abs_jax, dtype=np.float32)
    p_phase = np.array(p_phase_jax, dtype=np.float32)
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
) -> None:
    nx, ny, nz = p_abs.shape
    xmid, ymid, zmid = nx // 2, ny // 2, nz // 2
    extent_xy = [0, nx * dx_mm, 0, ny * dx_mm]
    extent_xz = [0, nx * dx_mm, 0, nz * dx_mm]
    extent_yz = [0, ny * dx_mm, 0, nz * dx_mm]
    extents = [extent_xy, extent_xz, extent_yz]
    names = ["XY", "XZ", "YZ"]

    # --- Set up working arrays, mask outside head to NaN ---
    p_abs_work = p_abs.copy()
    p_phase_work = p_phase.copy()

    if body_mask is not None:
        if body_mask.shape != p_abs.shape:
            print(f"[warn] body_mask shape {body_mask.shape} != p_abs shape {p_abs.shape}, ignoring mask.")
            body_mask = None

    if body_mask is not None:
        p_abs_work[~body_mask] = np.nan
        p_phase_work[~body_mask] = np.nan
        in_head = p_abs[body_mask]
        finite_in_head = in_head[np.isfinite(in_head)]
        vmax = float(np.percentile(finite_in_head, 99.5)) if finite_in_head.size > 0 else 1.0
    else:
        finite_abs = p_abs[np.isfinite(p_abs)]
        vmax = float(np.percentile(finite_abs, 99.5)) if finite_abs.size > 0 else 1.0

    print(f"[info] Pressure colormap vmax (99.5th pct in-head): {vmax:.4e} Pa")

    # RMS pressure = |P| / sqrt(2)  (for a time-harmonic field)
    p_rms_work = p_abs_work / np.sqrt(2.0)
    vmax_rms = vmax / np.sqrt(2.0)

    planes_mag  = [p_abs_work[:, :, zmid].T, p_abs_work[:, ymid, :].T, p_abs_work[xmid, :, :].T]
    planes_ph   = [p_phase_work[:, :, zmid].T, p_phase_work[:, ymid, :].T, p_phase_work[xmid, :, :].T]
    planes_rms  = [p_rms_work[:, :, zmid].T, p_rms_work[:, ymid, :].T, p_rms_work[xmid, :, :].T]

    # Colourmap with NaN rendered as black (outside head)
    cmap_mag = plt.cm.inferno.copy()
    cmap_mag.set_bad("black")
    cmap_phase = plt.cm.twilight.copy()
    cmap_phase.set_bad("black")
    cmap_rms = plt.cm.hot.copy()
    cmap_rms.set_bad("black")

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    # --- Row 0: |P| in Pa ---
    for i in range(3):
        im = axes[0, i].imshow(
            planes_mag[i], origin="lower", cmap=cmap_mag,
            extent=extents[i], vmin=0.0, vmax=vmax,
        )
        axes[0, i].set_title(f"|P| Pa {names[i]}")
        axes[0, i].set_xlabel("mm")
        axes[0, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, label="|P| (Pa)")

    # --- Row 1: phase ---
    for i in range(3):
        im = axes[1, i].imshow(
            planes_ph[i], origin="lower", cmap=cmap_phase,
            extent=extents[i], vmin=-np.pi, vmax=np.pi,
        )
        axes[1, i].set_title(f"phase(P) {names[i]}")
        axes[1, i].set_xlabel("mm")
        axes[1, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, label="phase (rad)")

    # --- Row 2: RMS pressure ---
    for i in range(3):
        im = axes[2, i].imshow(
            planes_rms[i], origin="lower", cmap=cmap_rms,
            extent=extents[i], vmin=0.0, vmax=vmax_rms,
        )
        axes[2, i].set_title(f"P_rms Pa {names[i]}")
        axes[2, i].set_xlabel("mm")
        axes[2, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, label="P_rms (Pa)")

    # --- Row 3: RMS pressure + circle at peak ---
    for i in range(3):
        im = axes[3, i].imshow(
            planes_rms[i], origin="lower", cmap=cmap_rms,
            extent=extents[i], vmin=0.0, vmax=vmax_rms,
        )
        axes[3, i].set_title(f"P_rms + peak {names[i]}")
        axes[3, i].set_xlabel("mm")
        axes[3, i].set_ylabel("mm")
        plt.colorbar(im, ax=axes[3, i], fraction=0.046, label="P_rms (Pa)")
        _add_max_circle(axes[3, i], planes_rms[i], extents[i], radius_mm=8.0)

    if vmax <= 1e-12:
        fig.text(
            0.5, 0.01,
            "Warning: pressure magnitude is near zero everywhere; check source/solver settings.",
            ha="center", color="crimson",
        )

    mask_note = " (masked to head)" if body_mask is not None else ""
    plt.suptitle(f"Helmholtz solution through CT-brain (Pa + phase + RMS){mask_note}", fontsize=14)
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
        help="Crop this many mm from each x-side after isotropic resampling (default: 45 mm).",
    )
    parser.add_argument(
        "--crop-y-start-mm",
        type=float,
        default=20.0,
        help="Crop this many mm from start of y-axis after isotropic resampling (default: 20 mm).",
    )
    parser.add_argument(
        "--crop-z-start-mm",
        type=float,
        default=20.0,
        help="Crop this many mm from start of z-axis after isotropic resampling (default: 20 mm).",
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

    print("Loading spacing from metadata...")
    spacing_xyz_mm = _load_spacing_from_metadata(in_dir)
    print(f"Shape of c_xyz: {c_xyz.shape}")
    print(f"Shape of rho_xyz: {rho_xyz.shape}")
    print(f"Spacing from metadata: {spacing_xyz_mm}")
    dx_mm = float(args.dx_mm)

    print("Resampling to isotropic...")
    c_iso, rho_iso = _resample_to_isotropic(c_xyz, rho_xyz, spacing_xyz_mm, target_dx_mm=dx_mm)



    # Load and resample body mask if available
    body_mask = None
    mask_path = _discover_body_mask(c_path)
    if mask_path is not None:
        print(f"Loading body mask from {mask_path}...")
        raw_mask = np.load(mask_path)
        body_mask = _resample_mask(raw_mask, spacing_xyz_mm, target_dx_mm=dx_mm)
        print(f"[info] Body mask resampled to {body_mask.shape}, coverage={body_mask.mean():.3f}")
    else:
        print("[info] No body mask found, plotting full volume.")

    # Apply requested x-crop to both sides on all volumes used downstream.
    c_iso, crop_vox = _crop_x_both_sides(c_iso, dx_mm=dx_mm, crop_x_mm_each_side=float(args.crop_x_mm_each_side))
    rho_iso, _ = _crop_x_both_sides(rho_iso, dx_mm=dx_mm, crop_x_mm_each_side=float(args.crop_x_mm_each_side))
    if body_mask is not None:
        body_mask, _ = _crop_x_both_sides(body_mask, dx_mm=dx_mm, crop_x_mm_each_side=float(args.crop_x_mm_each_side))
    crop_mm_used = crop_vox * dx_mm
    if crop_vox > 0:
        print(f"[info] Applied x-crop: {crop_mm_used:.2f} mm per side ({crop_vox} voxels each side).")

    # Crop from start of y-axis
    c_iso, crop_y_vox = _crop_axis_start(c_iso, dx_mm=dx_mm, crop_mm=float(args.crop_y_start_mm), axis=1)
    rho_iso, _ = _crop_axis_start(rho_iso, dx_mm=dx_mm, crop_mm=float(args.crop_y_start_mm), axis=1)
    if body_mask is not None:
        body_mask, _ = _crop_axis_start(body_mask, dx_mm=dx_mm, crop_mm=float(args.crop_y_start_mm), axis=1)
    crop_y_mm_used = crop_y_vox * dx_mm
    if crop_y_vox > 0:
        print(f"[info] Applied y-start crop: {crop_y_mm_used:.2f} mm ({crop_y_vox} voxels).")

    # Crop from start of z-axis
    c_iso, crop_z_vox = _crop_axis_start(c_iso, dx_mm=dx_mm, crop_mm=float(args.crop_z_start_mm), axis=2)
    rho_iso, _ = _crop_axis_start(rho_iso, dx_mm=dx_mm, crop_mm=float(args.crop_z_start_mm), axis=2)
    if body_mask is not None:
        body_mask, _ = _crop_axis_start(body_mask, dx_mm=dx_mm, crop_mm=float(args.crop_z_start_mm), axis=2)
    crop_z_mm_used = crop_z_vox * dx_mm
    if crop_z_vox > 0:
        print(f"[info] Applied z-start crop: {crop_z_mm_used:.2f} mm ({crop_z_vox} voxels).")

    print("Solving Helmholtz equation...")
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
    plot_slices(p_abs, p_phase, dx_mm=dx_mm, out_path=fig_slices, body_mask=body_mask)

    summary = {
        "input_dir": str(in_dir),
        "out_dir": str(out_dir),
        "dx_mm_requested": float(args.dx_mm),
        "dx_mm_used": float(dx_mm),
        "frequency_hz": float(args.frequency),
        "shape_before": list(c_xyz.shape),
        "shape_after_isotropic": list(c_iso.shape),
        "crop_x_mm_each_side_requested": float(args.crop_x_mm_each_side),
        "crop_x_mm_each_side_used": float(crop_mm_used),
        "crop_x_vox_each_side": int(crop_vox),
        "crop_y_start_mm_requested": float(args.crop_y_start_mm),
        "crop_y_start_mm_used": float(crop_y_mm_used),
        "crop_y_start_vox": int(crop_y_vox),
        "crop_z_start_mm_requested": float(args.crop_z_start_mm),
        "crop_z_start_mm_used": float(crop_z_mm_used),
        "crop_z_start_vox": int(crop_z_vox),
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