"""
Hard-coded lens baselines in the same 3D subwindow environment.

Runs multiple hand-crafted lenses (basic + creative), simulates wave propagation,
and reports focal-sphere RMS pressure for comparison.
"""

import os
import time as pytime
import numpy as np
import jax
from jax import numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle

from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# ------------------ Same environment as training script ------------------ #
N = (64, 64, 64)
c_water = 1500.0
c_cylinder = 2500.0
rho_water = 1000.0
rho_cylinder = 1200.0
frequency = 1e6
wavelength = c_water / frequency
dx = (wavelength / 10, wavelength / 10, wavelength / 10)
domain = Domain(N, dx)

design_boundary_x = 35
focus_boundary_x = 45
subwindow_fraction = 0.50
subwindow_x_start = int(0.5 * (1.0 - subwindow_fraction) * design_boundary_x)
subwindow_x_end = int(0.5 * (1.0 + subwindow_fraction) * design_boundary_x)
subwindow_y_start = int(0.5 * (1.0 - subwindow_fraction) * N[1])
subwindow_y_end = int(0.5 * (1.0 + subwindow_fraction) * N[1])
subwindow_z_start = int(0.5 * (1.0 - subwindow_fraction) * N[2])
subwindow_z_end = int(0.5 * (1.0 + subwindow_fraction) * N[2])

target_focal_center = (50, 32, 32)
focal_sphere_radius = 4

time_end = float(os.environ.get("BASELINE_TIME_END", "8.0e-05"))
stabilization_time = float(os.environ.get("BASELINE_STAB_TIME", "5.0e-05"))
time_axis = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=max(c_water, c_cylinder)),
    cfl=0.3,
    t_end=time_end,
)
time_array = time_axis.to_array()
stabilization_index = int(jnp.argmin(jnp.abs(time_array - stabilization_time)))

run_id = os.environ.get("AUTODIFF_RUN_ID", "")
results_dir = (
    os.path.join("Results/lens_baselines_subwindow_3d", run_id)
    if run_id
    else "Results/lens_baselines_subwindow_3d"
)
os.makedirs(results_dir, exist_ok=True)

print("=" * 70)
print("3D SUBWINDOW LENS BASELINES")
print("=" * 70)
print(f"Grid: {N}, dx={dx[0]:.2e} m, time steps={len(time_array)}")
print(f"Subwindow x=[{subwindow_x_start},{subwindow_x_end}] y=[{subwindow_y_start},{subwindow_y_end}] z=[{subwindow_z_start},{subwindow_z_end}]")
print(f"Focal sphere center={target_focal_center}, radius={focal_sphere_radius}")
print("=" * 70)


def create_sphere_mask_3d_smooth(N_grid, center, radius, steepness=2.0):
    x = jnp.arange(N_grid[0], dtype=jnp.float32)
    y = jnp.arange(N_grid[1], dtype=jnp.float32)
    z = jnp.arange(N_grid[2], dtype=jnp.float32)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    cx, cy, cz = center
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return jax.nn.sigmoid(steepness * (radius**2 - dist_sq))


def create_medium_from_alpha(alpha_full):
    sound_speed = c_water + (c_cylinder - c_water) * alpha_full
    density = rho_water + (rho_cylinder - rho_water) * alpha_full
    sound_speed = jnp.expand_dims(sound_speed, -1)
    density = jnp.expand_dims(density, -1)
    return Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=20)


class TimeVaryingSource:
    def __init__(self, ta):
        self.omega = 2.0 * jnp.pi * frequency
        self.time_array = ta.to_array()
        sm = jnp.zeros(N).at[:12, :, :].set(1.0)
        sm = jnp.expand_dims(sm, -1)
        self.source_fields = jnp.stack(
            [sm * jnp.sin(self.omega * t) for t in self.time_array], axis=0
        )

    def on_grid(self, ti):
        return jnp.squeeze(
            lax.dynamic_slice(
                self.source_fields,
                (lax.convert_element_type(ti, jnp.int32), 0, 0, 0, 0),
                (1, N[0], N[1], N[2], 1),
            ),
            axis=0,
        )


def run_simulation(alpha_full):
    p0 = FourierSeries(jnp.expand_dims(jnp.zeros(N), -1), domain)
    src = TimeVaryingSource(time_axis)
    medium = create_medium_from_alpha(alpha_full)
    return simulate_wave_propagation(medium, time_axis, p0=p0, sources=src)


def eval_focal_rms(pressure_field, sphere_mask):
    p = pressure_field.on_grid[..., 0]
    p_stab = p[stabilization_index:, ...]
    total_sq = jnp.sum((p_stab * sphere_mask[None, ...]) ** 2)
    n_time = p_stab.shape[0]
    n_pts = jnp.sum(sphere_mask)
    mean_sq = total_sq / jnp.maximum(n_time * n_pts, 1.0)
    return float(jnp.sqrt(jnp.maximum(mean_sq, 1e-20)))


def _subwindow_shape():
    nx = subwindow_x_end - subwindow_x_start + 1
    ny = subwindow_y_end - subwindow_y_start + 1
    nz = subwindow_z_end - subwindow_z_start + 1
    return nx, ny, nz


def _make_coords(nx, ny, nz):
    x = jnp.linspace(0.0, 1.0, nx)[:, None, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :, None]
    z = jnp.linspace(-1.0, 1.0, nz)[None, None, :]
    return x, y, z


def _embed_subwindow(alpha_sw):
    nx, _, _ = _subwindow_shape()
    design_mask_sw = ((subwindow_x_start + jnp.arange(nx)) < design_boundary_x)[:, None, None]
    alpha_sw = jnp.clip(alpha_sw * design_mask_sw, 0.0, 1.0)
    alpha_full = jnp.zeros(N, dtype=jnp.float32)
    return alpha_full.at[
        subwindow_x_start : subwindow_x_end + 1,
        subwindow_y_start : subwindow_y_end + 1,
        subwindow_z_start : subwindow_z_end + 1,
    ].set(alpha_sw)


def lens_water():
    nx, ny, nz = _subwindow_shape()
    return _embed_subwindow(jnp.zeros((nx, ny, nz), dtype=jnp.float32))


def lens_solid_block():
    nx, ny, nz = _subwindow_shape()
    return _embed_subwindow(jnp.ones((nx, ny, nz), dtype=jnp.float32))


def lens_plano_convex():
    nx, ny, nz = _subwindow_shape()
    x, y, z = _make_coords(nx, ny, nz)

    r2 = y**2 + z**2
    R = 1.2                     # radius of curvature
    xc = 0.15                   # flat face position

    # spherical surface
    sag = R - jnp.sqrt(jnp.clip(R**2 - r2, 0.0))
    surface = xc + sag

    alpha = (x <= surface).astype(jnp.float32)
    return _embed_subwindow(alpha)


def lens_biconvex():
    nx, ny, nz = _subwindow_shape()
    x, y, z = _make_coords(nx, ny, nz)

    r2 = y**2 + z**2
    R = 1.3
    x0 = 0.5

    sag = R - jnp.sqrt(jnp.clip(R**2 - r2, 0.0))

    left_surface  = x0 - sag
    right_surface = x0 + sag

    alpha = ((x >= left_surface) & (x <= right_surface)).astype(jnp.float32)
    return _embed_subwindow(alpha)


def lens_fresnel_rings(lam=0.1, f=1.5):
    nx, ny, nz = _subwindow_shape()
    x, y, z = _make_coords(nx, ny, nz)

    r = jnp.sqrt(y**2 + z**2)

    k = 2 * jnp.pi / lam
    delta_L = jnp.sqrt(r**2 + f**2) - f
    phase = k * delta_L

    zones = (jnp.mod(phase, 2*jnp.pi) < jnp.pi).astype(jnp.float32)

    thickness_mask = (x < 0.25).astype(jnp.float32)
    alpha = zones * thickness_mask

    return _embed_subwindow(alpha)


def lens_axicon_cone(angle=0.6):
    nx, ny, nz = _subwindow_shape()
    x, y, z = _make_coords(nx, ny, nz)

    r = jnp.sqrt(y**2 + z**2)

    base = 0.15
    slope = jnp.tan(angle)

    surface = base + slope * r
    alpha = (x <= surface).astype(jnp.float32)

    return _embed_subwindow(alpha)


def lens_grin_gaussian():
    nx, ny, nz = _subwindow_shape()
    x, y, z = _make_coords(nx, ny, nz)
    r2 = y**2 + z**2
    radial = jnp.exp(-3.0 * r2)
    axial = 0.65 + 0.35 * jnp.exp(-8.0 * (x - 0.45) ** 2)
    alpha = 0.85 * radial * axial
    return _embed_subwindow(alpha.astype(jnp.float32))




def run_baselines():
    baselines = [
        ("water_reference", lens_water),
        ("solid_block", lens_solid_block),
        ("plano_convex", lens_plano_convex),
        ("biconvex", lens_biconvex),
        ("fresnel_rings", lens_fresnel_rings),
        ("axicon_cone", lens_axicon_cone),
        ("grin_gaussian", lens_grin_gaussian),
    ]

    sphere_mask = create_sphere_mask_3d_smooth(N, target_focal_center, focal_sphere_radius)
    records = []
    pressure_summaries = {}
    alpha_fields = {}

    for name, make_lens in baselines:
        t0 = pytime.time()
        print(f"\nRunning baseline: {name}")
        alpha = make_lens()
        pf = run_simulation(alpha)
        rms = eval_focal_rms(pf, sphere_mask)
        elapsed = pytime.time() - t0
        p_mean = np.mean(np.abs(np.asarray(pf.on_grid[stabilization_index:, :, :, :, 0])), axis=0)
        pressure_summaries[name] = p_mean
        alpha_fields[name] = np.asarray(alpha)
        records.append(
            {
                "name": name,
                "rms": rms,
                "time_s": elapsed,
                "alpha_mean": float(np.asarray(alpha).mean()),
            }
        )
        print(f"  RMS @ sphere: {rms:.6f} Pa | time: {elapsed:.1f} s")

    records = sorted(records, key=lambda r: r["rms"], reverse=True)
    return records, pressure_summaries, alpha_fields


def plot_2d_slices_for_lens(name, alpha_data, p_mean, rms, out_dir):
    slice_x = int(target_focal_center[0])
    slice_y = int(target_focal_center[1])
    slice_z = int(target_focal_center[2])
    extent_xy = [0, N[0] * dx[0] * 1e3, 0, N[1] * dx[1] * 1e3]
    extent_xz = [0, N[0] * dx[0] * 1e3, 0, N[2] * dx[2] * 1e3]
    extent_yz = [0, N[1] * dx[1] * 1e3, 0, N[2] * dx[2] * 1e3]
    fc_x_mm = target_focal_center[0] * dx[0] * 1e3
    fc_y_mm = target_focal_center[1] * dx[1] * 1e3
    fc_z_mm = target_focal_center[2] * dx[2] * 1e3
    focal_r_mm = focal_sphere_radius * dx[0] * 1e3
    z_slice_mm = slice_z * dx[2] * 1e3
    y_slice_mm = slice_y * dx[1] * 1e3
    x_slice_mm = slice_x * dx[0] * 1e3
    r_xy = np.sqrt(max(0.0, focal_r_mm**2 - (z_slice_mm - fc_z_mm) ** 2))
    r_xz = np.sqrt(max(0.0, focal_r_mm**2 - (y_slice_mm - fc_y_mm) ** 2))
    r_yz = np.sqrt(max(0.0, focal_r_mm**2 - (x_slice_mm - fc_x_mm) ** 2))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    # Pressure slices
    im = axes[0, 0].imshow(p_mean[:, :, slice_z].T, cmap="hot", origin="lower", extent=extent_xy)
    if r_xy > 0:
        axes[0, 0].add_patch(MplCircle((fc_x_mm, fc_y_mm), r_xy, fill=False, edgecolor="lime", linewidth=2))
    axes[0, 0].set_title("Pressure XY")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    im = axes[0, 1].imshow(p_mean[:, slice_y, :].T, cmap="hot", origin="lower", extent=extent_xz)
    if r_xz > 0:
        axes[0, 1].add_patch(MplCircle((fc_x_mm, fc_z_mm), r_xz, fill=False, edgecolor="lime", linewidth=2))
    axes[0, 1].set_title("Pressure XZ")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = axes[0, 2].imshow(p_mean[slice_x, :, :].T, cmap="hot", origin="lower", extent=extent_yz)
    if r_yz > 0:
        axes[0, 2].add_patch(MplCircle((fc_y_mm, fc_z_mm), r_yz, fill=False, edgecolor="lime", linewidth=2))
    axes[0, 2].set_title("Pressure YZ")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Alpha slices
    im = axes[1, 0].imshow(alpha_data[:, :, slice_z].T, cmap="viridis", origin="lower", extent=extent_xy, vmin=0, vmax=1)
    axes[1, 0].set_title("Material alpha XY")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = axes[1, 1].imshow(alpha_data[:, slice_y, :].T, cmap="viridis", origin="lower", extent=extent_xz, vmin=0, vmax=1)
    axes[1, 1].set_title("Material alpha XZ")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    im = axes[1, 2].imshow(alpha_data[slice_x, :, :].T, cmap="viridis", origin="lower", extent=extent_yz, vmin=0, vmax=1)
    axes[1, 2].set_title("Material alpha YZ")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    for ax in axes.reshape(-1):
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")

    plt.suptitle(f"{name} | RMS={rms:.5f} Pa")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}_2d_slices.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_3d_structure_for_lens(name, alpha_data, out_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    x0 = subwindow_x_start * dx[0] * 1e3
    x1 = subwindow_x_end * dx[0] * 1e3
    y0 = subwindow_y_start * dx[1] * 1e3
    y1 = subwindow_y_end * dx[1] * 1e3
    z0 = subwindow_z_start * dx[2] * 1e3
    z1 = subwindow_z_end * dx[2] * 1e3

    def _edge(p1, p2):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k-", linewidth=1.5)

    _edge([x0, y0, z0], [x1, y0, z0]); _edge([x1, y0, z0], [x1, y1, z0])
    _edge([x1, y1, z0], [x0, y1, z0]); _edge([x0, y1, z0], [x0, y0, z0])
    _edge([x0, y0, z1], [x1, y0, z1]); _edge([x1, y0, z1], [x1, y1, z1])
    _edge([x1, y1, z1], [x0, y1, z1]); _edge([x0, y1, z1], [x0, y0, z1])
    _edge([x0, y0, z0], [x0, y0, z1]); _edge([x1, y0, z0], [x1, y0, z1])
    _edge([x1, y1, z0], [x1, y1, z1]); _edge([x0, y1, z0], [x0, y1, z1])

    alpha_sub = alpha_data[
        subwindow_x_start:subwindow_x_end,
        subwindow_y_start:subwindow_y_end,
        subwindow_z_start:subwindow_z_end,
    ]
    sm = None
    plotted = False
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib import cm
        nz = alpha_sub[alpha_sub > 1e-6]
        if nz.size > 0:
            levels = np.unique(np.clip(np.quantile(nz, [0.35, 0.55, 0.75]), 1e-4, 0.999))
        else:
            levels = np.array([0.1, 0.2, 0.3])
        for level in levels:
            try:
                verts, faces, _, _ = measure.marching_cubes(alpha_sub, level=float(level))
                verts_mm = (
                    verts + np.array([subwindow_x_start, subwindow_y_start, subwindow_z_start])
                ) * np.array([dx[0], dx[1], dx[2]]) * 1e3
                mesh = Poly3DCollection(verts_mm[faces], alpha=0.6, facecolor=plt.cm.viridis(float(level)), edgecolor="none")
                ax.add_collection3d(mesh)
                plotted = True
            except (ValueError, RuntimeError):
                continue
        sm = cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
    except ImportError:
        pass

    if not plotted:
        idx = np.argwhere(alpha_sub > max(0.02, float(np.max(alpha_sub) * 0.2)))
        if idx.size > 0:
            idx = idx[:: max(1, len(idx) // 4000)]
            vals = alpha_sub[idx[:, 0], idx[:, 1], idx[:, 2]]
            xs = (idx[:, 0] + subwindow_x_start) * dx[0] * 1e3
            ys = (idx[:, 1] + subwindow_y_start) * dx[1] * 1e3
            zs = (idx[:, 2] + subwindow_z_start) * dx[2] * 1e3
            ax.scatter(xs, ys, zs, c=vals, cmap="viridis", s=4, alpha=0.6)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_xlim(0, N[0] * dx[0] * 1e3)
    ax.set_ylim(0, N[1] * dx[1] * 1e3)
    ax.set_zlim(0, N[2] * dx[2] * 1e3)
    ax.set_title(f"{name} | 3D material structure")
    if sm is not None:
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label("alpha")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}_3d_structure.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_outputs(records, pressure_summaries, alpha_fields):
    table_path = os.path.join(results_dir, "baseline_scores.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("name,rms_pa,time_s,alpha_mean\n")
        for r in records:
            f.write(f"{r['name']},{r['rms']:.8f},{r['time_s']:.3f},{r['alpha_mean']:.6f}\n")

    print("\n" + "=" * 70)
    print("BASELINE RANKING (higher RMS is better)")
    for i, r in enumerate(records, 1):
        print(f"{i:2d}. {r['name']:<16} RMS={r['rms']:.6f} Pa  time={r['time_s']:.1f}s")
    print("=" * 70)
    print(f"Saved scores: {table_path}")

    # Plot pressure mean XY slice at focal z for each lens.
    slice_z = int(target_focal_center[2])
    n = len(records)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    vmax = max(np.max(pressure_summaries[r["name"]]) for r in records)
    extent_xy = [0, N[0] * dx[0] * 1e3, 0, N[1] * dx[1] * 1e3]

    for i, r in enumerate(records):
        name = r["name"]
        ax = axes[i]
        img = pressure_summaries[name][:, :, slice_z].T
        im = ax.imshow(img, cmap="hot", origin="lower", extent=extent_xy, vmin=0, vmax=vmax)
        ax.set_title(f"{name}\nRMS={r['rms']:.4f} Pa")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Hard-coded Lens Baselines | Mean |P| XY Slice", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "baseline_pressure_slices_xy.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    per_lens_dir = os.path.join(results_dir, "per_lens")
    os.makedirs(per_lens_dir, exist_ok=True)
    for r in records:
        name = r["name"]
        plot_2d_slices_for_lens(name, alpha_fields[name], pressure_summaries[name], r["rms"], per_lens_dir)
        plot_3d_structure_for_lens(name, alpha_fields[name], per_lens_dir)
    print(f"Saved per-lens 2D/3D visualizations: {per_lens_dir}")


if __name__ == "__main__":
    recs, summaries, alphas = run_baselines()
    save_outputs(recs, summaries, alphas)
    print("\nDone.")

