"""
3D FINER + Multiresolution Hash Material Field - Subwindow Optimization

HashEncoding(x,y,z) -> FINER MLP -> alpha in [0,1].
Replaces sinusoidal pos enc with Instant-NGP-style multiresolution hash encoding.
Variable-periodic FINER activation: sin(omega_0 * (|z|+1) * z).
Optimizes to maximize RMS pressure in a sphere using JAX autodiff through j-wave 3D simulation.

Combines:
- j_wave_autodiff_mlp_256_FINER_subwindow_3d_posenc.py (FINER MLP, subwindow, 3D sim)
- Multiresolution hash encoding from Instant-NGP (j_wave_autodiff_hash_subwindow_3d.py)
"""

import numpy as np
from jax import numpy as jnp
from jax import grad, jit, lax, value_and_grad, vmap
import jax
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import os
import time as pytime

from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis

_run_id = os.environ.get('AUTODIFF_RUN_ID', '')
results_dir = os.path.join('Results/autodiff_finer_hash_subwindow_3d', _run_id) if _run_id else 'Results/autodiff_finer_hash_subwindow_3d'
os.makedirs('Results/autodiff_finer_hash_subwindow_3d', exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("="*70)
print("3D FINER + MULTIRESOLUTION HASH - SUBWINDOW")
print("="*70)
print("\nHashEncoding(x,y,z) -> FINER -> alpha in [0,1]. Material only in subwindow.")
print("Instant-NGP multiresolution hash + variable-periodic FINER activation.")
print("Maximize RMS pressure in focal sphere. 3D j-wave simulation.")
print("="*70)

# 3D grid
N = (64, 64, 64)
dx = (0.4e-3, 0.4e-3, 0.4e-3)
domain = Domain(N, dx)

view_x_start, view_x_end = 0, N[0]
view_y_start, view_y_end = 0, N[1]
view_z_start, view_z_end = 0, N[2]

c_water = 1500.0
c_cylinder = 2500.0
rho_water = 1000.0
rho_cylinder = 1200.0

frequency = 1.0e6
wavelength = c_water / frequency

design_boundary_x = 35
focus_boundary_x = 45

# Subwindow: 50% of design space (centered)
subwindow_fraction = 0.50
subwindow_x_start = int(0.5 * (1.0 - subwindow_fraction) * design_boundary_x)
subwindow_x_end = int(0.5 * (1.0 + subwindow_fraction) * design_boundary_x)
subwindow_y_start = int(0.5 * (1.0 - subwindow_fraction) * N[1])
subwindow_y_end = int(0.5 * (1.0 + subwindow_fraction) * N[1])
subwindow_z_start = int(0.5 * (1.0 - subwindow_fraction) * N[2])
subwindow_z_end = int(0.5 * (1.0 + subwindow_fraction) * N[2])

# Target focal sphere (3D)
target_focal_center = (50, 32, 32)
focal_sphere_radius = 4

training_time_end = 5.0e-05
stabilization_time = 3.0e-05

time_axis = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3,
    t_end=training_time_end
)
time_array = time_axis.to_array()
stabilization_index = int(jnp.argmin(jnp.abs(time_array - stabilization_time)))

# ============================================================================
# Multiresolution Hash Encoding (Instant-NGP style)
# ============================================================================
hash_n_levels = int(os.environ.get('AUTODIFF_HASH_L', '16'))
hash_n_features_per_level = int(os.environ.get('AUTODIFF_HASH_F', '2'))
hash_log2_t = int(os.environ.get('AUTODIFF_HASH_LOG2T', '14'))
hash_base_resolution = 16
hash_finest_resolution = 512
HASH_PRIMES = (1, 2654435761, 805459861, 3674653429)

# FINER after hash: input dim = L * F
hash_output_dim = hash_n_levels * hash_n_features_per_level
finer_layer_sizes = [hash_output_dim, 64, 64, 32, 1]
finer_omega0 = 30.0
finer_first_bias_scale = float(os.environ.get('AUTODIFF_FINER_FIRST_BIAS_SCALE', '1.0'))
seed = 42

print(f"\nSimulation Configuration:")
print(f"  Grid: {N[0]} x {N[1]} x {N[2]}")
print(f"  Simulation time: {training_time_end*1e6:.1f} us")
print(f"  Target sphere: center {target_focal_center}, radius {focal_sphere_radius} pts")
print(f"  Hash: L={hash_n_levels}, F={hash_n_features_per_level}, log2(T)={hash_log2_t}")
print(f"  Hash output dim: {hash_output_dim}")
print(f"  FINER: {finer_layer_sizes}, omega0={finer_omega0}, first_bias_scale={finer_first_bias_scale}")
print(f"  Subwindow: x=[{subwindow_x_start},{subwindow_x_end}], y=[{subwindow_y_start},{subwindow_y_end}], z=[{subwindow_z_start},{subwindow_z_end}]")
print("="*70)


def _hash_func(coords, log2_t):
    T = 1 << log2_t
    mask = T - 1
    xr = jnp.asarray(coords[..., 0], dtype=jnp.uint32)
    yr = jnp.asarray(coords[..., 1], dtype=jnp.uint32)
    zr = jnp.asarray(coords[..., 2], dtype=jnp.uint32)
    h = (xr * jnp.uint32(HASH_PRIMES[0]) ^
         yr * jnp.uint32(HASH_PRIMES[1]) ^
         zr * jnp.uint32(HASH_PRIMES[2]))
    return jnp.asarray(h & mask, dtype=jnp.int32)


def _trilinear_interp(x, vmin, vmax, corner_features):
    eps = 1e-7
    w = (x - vmin) / jnp.maximum(vmax - vmin, eps)
    w = jnp.clip(w, 0.0, 1.0)
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    c00 = (1 - wx)[..., None] * corner_features[0] + wx[..., None] * corner_features[1]
    c01 = (1 - wx)[..., None] * corner_features[2] + wx[..., None] * corner_features[3]
    c10 = (1 - wx)[..., None] * corner_features[4] + wx[..., None] * corner_features[5]
    c11 = (1 - wx)[..., None] * corner_features[6] + wx[..., None] * corner_features[7]
    c0 = (1 - wy)[..., None] * c00 + wy[..., None] * c01
    c1 = (1 - wy)[..., None] * c10 + wy[..., None] * c11
    c = (1 - wz)[..., None] * c0 + wz[..., None] * c1
    return c


def hash_encode_single(xyz, hash_tables, b, base_res, finest_res):
    box_min = jnp.array([0.0, 0.0, 0.0])
    box_max = jnp.array([1.0, 1.0, 1.0])
    xyz_clip = jnp.clip(xyz, box_min, box_max)

    features_per_level = []
    for l in range(hash_n_levels):
        res_f = jnp.maximum(jnp.floor(base_res * (b ** l)), 1.0)
        res_int = lax.convert_element_type(res_f, jnp.int32)
        grid_size = (box_max - box_min) / res_f

        idx_f = (xyz_clip - box_min) / grid_size
        idx_lo = jnp.floor(idx_f).astype(jnp.int32)
        idx_lo = jnp.clip(idx_lo, 0, res_int - 1)

        voxel_min = idx_lo.astype(jnp.float32) * grid_size + box_min
        voxel_max = voxel_min + grid_size

        offsets = jnp.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=jnp.int32)
        corner_indices = jnp.clip(idx_lo + offsets, 0, res_int)

        hashes = _hash_func(corner_indices, hash_log2_t)
        table = hash_tables[l]
        corner_feats = table[hashes]
        interp = _trilinear_interp(xyz_clip, voxel_min, voxel_max, corner_feats)
        features_per_level.append(interp)

    return jnp.concatenate(features_per_level, axis=-1)


def hash_encode_init(key):
    T = 1 << hash_log2_t
    key, *subkeys = jax.random.split(key, hash_n_levels + 1)
    tables = []
    for i in range(hash_n_levels):
        t = jax.random.uniform(subkeys[i], (T, hash_n_features_per_level),
                               minval=-1e-4, maxval=1e-4)
        tables.append(t)
    return tables


# ============================================================================
# FINER: variable-periodic activation sin(omega_0 * (|z|+1) * z)
# ============================================================================

def finer_init(key, layer_sizes, omega_0=30.0, first_bias_scale=1.0):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, k1 = jax.random.split(key)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        if i == 0:
            k1a, k1b = jax.random.split(k1, 2)
            bound = 1.0 / fan_in
            W = jax.random.uniform(k1a, (fan_in, fan_out), minval=-bound, maxval=bound)
            b = jax.random.uniform(k1b, (fan_out,), minval=-first_bias_scale, maxval=first_bias_scale)
        elif i == len(layer_sizes) - 2:
            bound = 1.0 / fan_in
            W = jax.random.uniform(k1, (fan_in, fan_out), minval=-bound, maxval=bound)
            b = jnp.zeros(fan_out)
        else:
            bound = jnp.sqrt(6.0 / fan_in) / omega_0
            W = jax.random.uniform(k1, (fan_in, fan_out), minval=-bound, maxval=bound)
            b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def finer_activation(z, omega_0):
    scale = jnp.abs(z) + 1.0
    return jnp.sin(omega_0 * scale * z)


def finer_apply(params, feats, omega_0=30.0):
    h = feats
    for (W, b) in params[:-1]:
        z = h @ W + b
        h = finer_activation(z, omega_0)
    Wlast, blast = params[-1]
    h = h @ Wlast + blast
    return jax.nn.sigmoid(jnp.squeeze(h))


def hash_finer_apply(hash_tables, finer_params, xyz, b, base_res, finest_res):
    feats = hash_encode_single(xyz, hash_tables, b, base_res, finest_res)
    return finer_apply(finer_params, feats, finer_omega0)


def material_field_from_params(params):
    hash_tables, finer_params = params
    b = jnp.exp((jnp.log(float(hash_finest_resolution)) - jnp.log(float(hash_base_resolution)))
                / max(hash_n_levels - 1, 1))

    xs = jnp.linspace(0.0, 1.0, N[0])
    ys = jnp.linspace(0.0, 1.0, N[1])
    zs = jnp.linspace(0.0, 1.0, N[2])
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing='ij')
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def single_forward(p):
        return hash_finer_apply(hash_tables, finer_params, p, b, hash_base_resolution, hash_finest_resolution)

    alpha = vmap(single_forward, in_axes=(0,))(grid_points)
    alpha = alpha.reshape(N[0], N[1], N[2])

    x_norm_design = design_boundary_x / (N[0] - 1.0)
    X_norm = jnp.linspace(0.0, 1.0, N[0])
    design_mask = (X_norm < x_norm_design)[:, jnp.newaxis, jnp.newaxis]

    x_lo = subwindow_x_start / (N[0] - 1.0)
    x_hi = subwindow_x_end / (N[0] - 1.0)
    y_lo = subwindow_y_start / (N[1] - 1.0)
    y_hi = subwindow_y_end / (N[1] - 1.0)
    z_lo = subwindow_z_start / (N[2] - 1.0)
    z_hi = subwindow_z_end / (N[2] - 1.0)
    subwindow_mask = (
        (X >= x_lo) & (X <= x_hi) &
        (Y >= y_lo) & (Y <= y_hi) &
        (Z >= z_lo) & (Z <= z_hi)
    )

    alpha = alpha * design_mask * subwindow_mask

    sound_speed = c_water + (c_cylinder - c_water) * alpha
    density = rho_water + (rho_cylinder - rho_water) * alpha
    return sound_speed, density


def create_sphere_mask_3d_smooth(N, center, radius, steepness=2.0):
    x = jnp.arange(N[0], dtype=jnp.float32)
    y = jnp.arange(N[1], dtype=jnp.float32)
    z = jnp.arange(N[2], dtype=jnp.float32)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    cx, cy, cz = center
    dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    return jax.nn.sigmoid(steepness * (radius**2 - dist_sq))


def create_medium_from_material_field(domain, sound_speed, density):
    sound_speed = jnp.expand_dims(sound_speed, -1)
    density = jnp.expand_dims(density, -1)
    return Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=20)


def run_simulation_from_medium(medium, sim_time_axis=None):
    if sim_time_axis is None:
        sim_time_axis = time_axis
    p0_array = jnp.zeros(N)
    p0_array = jnp.expand_dims(p0_array, -1)
    p0 = FourierSeries(p0_array, domain)

    class TimeVaryingSource:
        def __init__(self, ta, domain):
            self.omega = 2 * jnp.pi * frequency
            self.time_array = ta.to_array()
            sm = jnp.zeros(N).at[:12, :, :].set(1.0)
            sm = jnp.expand_dims(sm, -1)
            self.source_fields = jnp.stack([sm * jnp.sin(self.omega * t) for t in self.time_array], axis=0)

        def on_grid(self, ti):
            return jnp.squeeze(lax.dynamic_slice(
                self.source_fields, (lax.convert_element_type(ti, jnp.int32), 0, 0, 0, 0),
                (1, N[0], N[1], N[2], 1)), axis=0)

    src = TimeVaryingSource(sim_time_axis, domain)
    return simulate_wave_propagation(medium, sim_time_axis, p0=p0, sources=src)


def compute_objective(params):
    sound_speed, density = material_field_from_params(params)
    medium = create_medium_from_material_field(domain, sound_speed, density)
    pressure_field = run_simulation_from_medium(medium)

    pressure_on_grid = pressure_field.on_grid
    sphere_mask = create_sphere_mask_3d_smooth(N, target_focal_center, focal_sphere_radius)
    stabilized_pressure = pressure_on_grid[stabilization_index:, :, :, :, 0]
    mask_expanded = jnp.expand_dims(sphere_mask, axis=0)
    pressure_in_sphere = stabilized_pressure * mask_expanded

    n_points = jnp.sum(sphere_mask)
    total_sq = jnp.sum(pressure_in_sphere**2)
    n_time = stabilized_pressure.shape[0]
    mean_sq = total_sq / jnp.maximum(n_time * n_points, 1.0)
    rms = jnp.sqrt(mean_sq)
    return -rms


# ============================================================================
# Optimization
# ============================================================================

key = jax.random.PRNGKey(seed)
hash_tables = hash_encode_init(key)
key, k2 = jax.random.split(key)
finer_params = finer_init(k2, finer_layer_sizes, finer_omega0, finer_first_bias_scale)
params = (hash_tables, finer_params)

learning_rate = float(os.environ.get('AUTODIFF_LR', '0.005'))
n_iterations = int(os.environ.get('AUTODIFF_N_ITER', '200'))

best_objective = float('inf')
best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
best_iteration = 0

optimization_history = {
    'objectives': [], 'pressures': [], 'gradient_norms': [], 'iteration_times': []
}

print("\n" + "="*70)
print("OPTIMIZATION: FINER + HASH MATERIAL FIELD (3D)")
print("="*70)
print(f"  Iterations: {n_iterations}, LR: {learning_rate}\n")

for iteration in range(n_iterations):
    iter_start = pytime.time()
    print(f"Iteration {iteration + 1}/{n_iterations}")

    obj_val, grad_params = value_and_grad(compute_objective)(params)
    obj_float = float(obj_val)

    grad_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad_params)
    ))

    print(f"  Loss: {obj_val:.6f}, RMS: {-obj_val:.6f} Pa")

    if obj_float < best_objective:
        best_objective = obj_float
        best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
        best_iteration = iteration + 1
        print(f"  *** New best! ***")

    def update(p, g):
        return p - learning_rate * g
    params = jax.tree_util.tree_map(update, params, grad_params)

    optimization_history['objectives'].append(obj_float)
    optimization_history['pressures'].append(float(-obj_val))
    optimization_history['gradient_norms'].append(float(grad_norm))
    optimization_history['iteration_times'].append(pytime.time() - iter_start)
    print(f"  Time: {optimization_history['iteration_times'][-1]:.2f} s")

final_params = best_params
first_rms = optimization_history['pressures'][0]
print(f"\nBest iter {best_iteration}, RMS {-best_objective:.6f} Pa")
print(f"Improvement: {(-best_objective / max(first_rms, 1e-10) - 1) * 100:.1f}%")


# ============================================================================
# Validation
# ============================================================================

print("\n" + "="*70)
print("VALIDATION")
print("="*70)

time_axis_full = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3,
    t_end=8.0e-05
)
time_array_full = time_axis_full.to_array()
stab_idx_full = int(jnp.argmin(jnp.abs(time_array_full - 5.0e-05)))
sphere_mask_np = np.array(create_sphere_mask_3d_smooth(N, target_focal_center, focal_sphere_radius))
n_sphere_pts = np.sum(sphere_mask_np)
n_time_val = len(time_array_full) - stab_idx_full


def run_val(pp):
    ss, dd = material_field_from_params(pp)
    med = create_medium_from_material_field(domain, ss, dd)
    pf = run_simulation_from_medium(med, time_axis_full)
    return np.array(pf.on_grid)[:, :, :, :, 0]


initial_params = (hash_tables, finer_params)
print("Running initial config...")
p_init = run_val(initial_params)
rms_init_val = np.sqrt(np.sum((p_init[stab_idx_full:] * sphere_mask_np)**2) / (n_time_val * n_sphere_pts))
print(f"  RMS: {rms_init_val:.6f} Pa")

print("Running optimized config...")
p_opt = run_val(final_params)
rms_opt_val = np.sqrt(np.sum((p_opt[stab_idx_full:] * sphere_mask_np)**2) / (n_time_val * n_sphere_pts))
print(f"  RMS: {rms_opt_val:.6f} Pa")

improvement_full = (rms_opt_val / max(rms_init_val, 1e-10) - 1) * 100
print(f"  Improvement: {improvement_full:.1f}%")


# ============================================================================
# Visualization helpers
# ============================================================================

slice_x = int(target_focal_center[0])
slice_y = int(target_focal_center[1])
slice_z = int(target_focal_center[2])
extent_xy = [0, N[0]*dx[0]*1e3, 0, N[1]*dx[1]*1e3]
extent_xz = [0, N[0]*dx[0]*1e3, 0, N[2]*dx[2]*1e3]
extent_yz = [0, N[1]*dx[1]*1e3, 0, N[2]*dx[2]*1e3]
design_x_mm = design_boundary_x * dx[0] * 1e3
focus_x_mm = focus_boundary_x * dx[0] * 1e3
fc_x_mm = target_focal_center[0] * dx[0] * 1e3
fc_y_mm = target_focal_center[1] * dx[1] * 1e3
fc_z_mm = target_focal_center[2] * dx[2] * 1e3


def add_boundary_plane(ax, x_grid_mm):
    y_plane = np.linspace(0, N[1] * dx[1] * 1e3, 12)
    z_plane = np.linspace(0, N[2] * dx[2] * 1e3, 12)
    Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
    X_plane = np.ones_like(Y_plane) * x_grid_mm
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.15, color='gray', shade=False)


p_mean_init = np.mean(np.abs(p_init[stab_idx_full:]), axis=0)
p_mean_opt = np.mean(np.abs(p_opt[stab_idx_full:]), axis=0)
vmax_slice = max(np.max(p_mean_init), np.max(p_mean_opt))

alpha_init = np.array(material_field_from_params(initial_params)[0])
alpha_init = np.clip((alpha_init - c_water) / (c_cylinder - c_water), 0, 1)
alpha_opt = np.array(material_field_from_params(final_params)[0])
alpha_opt = np.clip((alpha_opt - c_water) / (c_cylinder - c_water), 0, 1)

# -------------------------------------------------------------------------
# Figure 1: 2x4 layout
# -------------------------------------------------------------------------

fig1, axes1 = plt.subplots(2, 4, figsize=(24, 12))
extent_2d = [view_x_start * dx[0] * 1e3, view_x_end * dx[0] * 1e3,
             view_y_start * dx[1] * 1e3, view_y_end * dx[1] * 1e3]

ax = axes1[0, 0]
ax.plot(optimization_history['pressures'], 'b-o', linewidth=2, markersize=6)
ax.axhline(rms_init_val, color='r', linestyle='--', alpha=0.5, label='Initial (val)')
ax.axhline(rms_opt_val, color='g', linestyle='--', alpha=0.5, label='Optimized (val)')
ax.set_xlabel('Iteration')
ax.set_ylabel('RMS Pressure (Pa)')
ax.set_title('FINER+Hash Optimization Progress')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes1[0, 1]
ax.semilogy(optimization_history['gradient_norms'], 'b-o', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Gradient Norm')
ax.set_title('FINER+Hash Gradient Magnitude')
ax.grid(True, alpha=0.3)

ax = axes1[0, 2]
im = ax.imshow(alpha_init[:, :, slice_z].T, cmap='viridis', origin='lower', extent=extent_xy, vmin=0, vmax=1)
ax.axvline(design_x_mm, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_x_mm, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
ax.add_patch(Rectangle((subwindow_x_start*dx[0]*1e3, subwindow_y_start*dx[1]*1e3),
    (subwindow_x_end-subwindow_x_start)*dx[0]*1e3, (subwindow_y_end-subwindow_y_start)*dx[1]*1e3,
    fill=False, edgecolor='white', linewidth=2, linestyle='--'))
ax.set_title('Initial Material (XY slice)')
plt.colorbar(im, ax=ax, label='alpha', fraction=0.046)

ax = axes1[0, 3]
im = ax.imshow(alpha_opt[:, :, slice_z].T, cmap='viridis', origin='lower', extent=extent_xy, vmin=0, vmax=1)
ax.axvline(design_x_mm, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_x_mm, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
ax.add_patch(Rectangle((subwindow_x_start*dx[0]*1e3, subwindow_y_start*dx[1]*1e3),
    (subwindow_x_end-subwindow_x_start)*dx[0]*1e3, (subwindow_y_end-subwindow_y_start)*dx[1]*1e3,
    fill=False, edgecolor='white', linewidth=2, linestyle='--'))
ax.set_title('Optimized Material (XY slice)')
plt.colorbar(im, ax=ax, label='alpha', fraction=0.046)

ax = axes1[1, 0]
im = ax.imshow(p_mean_init[:, :, slice_z].T, cmap='hot', origin='lower', extent=extent_xy, vmin=0, vmax=vmax_slice)
ax.axvline(design_x_mm, color='gray', linewidth=1, linestyle='--', alpha=0.6)
ax.axvline(focus_x_mm, color='cyan', linewidth=1, linestyle='--', alpha=0.6)
focal_r_mm = focal_sphere_radius * dx[0] * 1e3
ax.add_patch(MplCircle((fc_x_mm, fc_y_mm), focal_r_mm, fill=False, edgecolor='lime', linewidth=2))
ax.set_title(f'Initial |P| - RMS: {rms_init_val:.4f} Pa')
plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)

ax = axes1[1, 1]
im = ax.imshow(p_mean_opt[:, :, slice_z].T, cmap='hot', origin='lower', extent=extent_xy, vmin=0, vmax=vmax_slice)
ax.axvline(design_x_mm, color='gray', linewidth=1, linestyle='--', alpha=0.6)
ax.axvline(focus_x_mm, color='cyan', linewidth=1, linestyle='--', alpha=0.6)
ax.add_patch(MplCircle((fc_x_mm, fc_y_mm), focal_r_mm, fill=False, edgecolor='lime', linewidth=2))
ax.set_title(f'Optimized |P| - RMS: {rms_opt_val:.4f} Pa (+{improvement_full:.1f}%)')
plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)

ax = axes1[1, 2]
ax.plot(optimization_history['iteration_times'], 'g-o', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Time (s)')
ax.set_title('Iteration Time')
ax.grid(True, alpha=0.3)

ax = axes1[1, 3]
ax.axis('off')
summary = f"""
3D FINER + HASH SUBWINDOW SUMMARY

Pipeline:
  - HashEncoding(x,y,z) -> L*F features
  - FINER(features) -> alpha in [0,1]
  - Variable-periodic: sin(omega0*(|z|+1)*z)
  - alpha only in subwindow; rest = water
  - Loss = -RMS(pressure in focal sphere)

Hash: L={hash_n_levels}, F={hash_n_features_per_level}, log2(T)={hash_log2_t}
FINER: {finer_layer_sizes}, omega0={finer_omega0}

Results:
  Initial RMS: {rms_init_val:.6f} Pa
  Optimized RMS: {rms_opt_val:.6f} Pa
  Improvement: {improvement_full:.1f}%

Iterations: {n_iterations}
Avg time/iter: {np.mean(optimization_history['iteration_times']):.1f} s
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('3D FINER + Multiresolution Hash Material Field - Subwindow (GINN + J-Wave)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'finer_hash_subwindow_3d_main.png'), dpi=150, bbox_inches='tight')
print(f"Saved: finer_hash_subwindow_3d_main.png")
plt.close(fig1)

# -------------------------------------------------------------------------
# Figure 2 & 3: Pressure and Material slices
# -------------------------------------------------------------------------

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))


def plot_slice_row(axes_row, p_mean, rms_val, is_opt=False):
    focal_r_mm = focal_sphere_radius * dx[0] * 1e3
    z_slice_mm = slice_z * dx[2] * 1e3
    r_xy = np.sqrt(max(0, focal_r_mm**2 - (z_slice_mm - fc_z_mm)**2))
    y_slice_mm = slice_y * dx[1] * 1e3
    r_xz = np.sqrt(max(0, focal_r_mm**2 - (y_slice_mm - fc_y_mm)**2))
    x_slice_mm = slice_x * dx[0] * 1e3
    r_yz = np.sqrt(max(0, focal_r_mm**2 - (x_slice_mm - fc_x_mm)**2))

    im = axes_row[0].imshow(p_mean[:, :, slice_z].T, cmap='hot', origin='lower',
        extent=extent_xy, vmin=0, vmax=vmax_slice)
    axes_row[0].axvline(design_x_mm, color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
    axes_row[0].axvline(focus_x_mm, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)
    axes_row[0].add_patch(MplCircle((fc_x_mm, fc_y_mm), r_xy, fill=False, edgecolor='lime', linewidth=2))
    axes_row[0].set_xlabel('x (mm)')
    axes_row[0].set_ylabel('y (mm)')
    axes_row[0].set_title(f'{"Optimized" if is_opt else "Initial"} - XY slice')
    plt.colorbar(im, ax=axes_row[0], label='|P| (Pa)', fraction=0.046)

    im = axes_row[1].imshow(p_mean[:, slice_y, :].T, cmap='hot', origin='lower',
        extent=extent_xz, vmin=0, vmax=vmax_slice)
    axes_row[1].axvline(design_x_mm, color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
    axes_row[1].axvline(focus_x_mm, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)
    axes_row[1].add_patch(MplCircle((fc_x_mm, fc_z_mm), r_xz, fill=False, edgecolor='lime', linewidth=2))
    axes_row[1].set_xlabel('x (mm)')
    axes_row[1].set_ylabel('z (mm)')
    axes_row[1].set_title(f'{"Optimized" if is_opt else "Initial"} - XZ slice')
    plt.colorbar(im, ax=axes_row[1], label='|P| (Pa)', fraction=0.046)

    im = axes_row[2].imshow(p_mean[slice_x, :, :].T, cmap='hot', origin='lower',
        extent=extent_yz, vmin=0, vmax=vmax_slice)
    axes_row[2].add_patch(MplCircle((fc_y_mm, fc_z_mm), r_yz, fill=False, edgecolor='lime', linewidth=2))
    axes_row[2].set_xlabel('y (mm)')
    axes_row[2].set_ylabel('z (mm)')
    axes_row[2].set_title(f'{"Optimized" if is_opt else "Initial"} - YZ slice')
    plt.colorbar(im, ax=axes_row[2], label='|P| (Pa)', fraction=0.046)


plot_slice_row(axes2[0], p_mean_init, rms_init_val, False)
plot_slice_row(axes2[1], p_mean_opt, rms_opt_val, True)

plt.suptitle('Initial vs Optimized - All 3 Planes (XY, XZ, YZ) | Circle = focus sphere', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'finer_hash_subwindow_3d_pressure_slices.png'), dpi=150, bbox_inches='tight')
print(f"Saved: finer_hash_subwindow_3d_pressure_slices.png")
plt.close(fig2)

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))


def plot_alpha_row(axes_row, alpha_data, label):
    focal_r_mm = focal_sphere_radius * dx[0] * 1e3
    z_slice_mm = slice_z * dx[2] * 1e3
    r_xy = np.sqrt(max(0, focal_r_mm**2 - (z_slice_mm - fc_z_mm)**2))
    y_slice_mm = slice_y * dx[1] * 1e3
    r_xz = np.sqrt(max(0, focal_r_mm**2 - (y_slice_mm - fc_y_mm)**2))
    x_slice_mm = slice_x * dx[0] * 1e3
    r_yz = np.sqrt(max(0, focal_r_mm**2 - (x_slice_mm - fc_x_mm)**2))
    circles = [(fc_x_mm, fc_y_mm, r_xy), (fc_x_mm, fc_z_mm, r_xz), (fc_y_mm, fc_z_mm, r_yz)]

    for j, (slice_data, ext, xlabel, ylabel, title) in enumerate([
        (alpha_data[:, :, slice_z].T, extent_xy, 'x (mm)', 'y (mm)', 'XY'),
        (alpha_data[:, slice_y, :].T, extent_xz, 'x (mm)', 'z (mm)', 'XZ'),
        (alpha_data[slice_x, :, :].T, extent_yz, 'y (mm)', 'z (mm)', 'YZ'),
    ]):
        im = axes_row[j].imshow(slice_data, cmap='viridis', origin='lower', extent=ext, vmin=0, vmax=1)
        if j < 2:
            axes_row[j].axvline(design_x_mm, color='red', linewidth=1, linestyle=':', alpha=0.5)
            axes_row[j].axvline(focus_x_mm, color='cyan', linewidth=1, linestyle=':', alpha=0.5)
        cx, cy, r = circles[j]
        if r > 0:
            axes_row[j].add_patch(MplCircle((cx, cy), r, fill=False, edgecolor='lime', linewidth=2))
        axes_row[j].set_xlabel(xlabel)
        axes_row[j].set_ylabel(ylabel)
        axes_row[j].set_title(f'{label} - {title} slice')
        plt.colorbar(im, ax=axes_row[j], label='alpha', fraction=0.046)


plot_alpha_row(axes3[0], alpha_init, 'Initial')
plot_alpha_row(axes3[1], alpha_opt, 'Optimized')

plt.suptitle('Material Field - All 3 Planes', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'finer_hash_subwindow_3d_material_slices.png'), dpi=150, bbox_inches='tight')
print(f"Saved: finer_hash_subwindow_3d_material_slices.png")
plt.close(fig3)

# -------------------------------------------------------------------------
# Figure 4 & 5: 3D views
# -------------------------------------------------------------------------

def plot_3d_structure_view(alpha_data, title, fpath):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    add_boundary_plane(ax, design_x_mm)
    add_boundary_plane(ax, focus_x_mm)

    x0 = subwindow_x_start * dx[0] * 1e3
    x1 = subwindow_x_end * dx[0] * 1e3
    y0 = subwindow_y_start * dx[1] * 1e3
    y1 = subwindow_y_end * dx[1] * 1e3
    z0 = subwindow_z_start * dx[2] * 1e3
    z1 = subwindow_z_end * dx[2] * 1e3

    def draw_edge(p1, p2):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=2)

    for e in [
        ([x0,y0,z0],[x1,y0,z0]), ([x1,y0,z0],[x1,y1,z0]), ([x1,y1,z0],[x0,y1,z0]), ([x0,y1,z0],[x0,y0,z0]),
        ([x0,y0,z1],[x1,y0,z1]), ([x1,y0,z1],[x1,y1,z1]), ([x1,y1,z1],[x0,y1,z1]), ([x0,y1,z1],[x0,y0,z1]),
        ([x0,y0,z0],[x0,y0,z1]), ([x1,y0,z0],[x1,y0,z1]), ([x1,y1,z0],[x1,y1,z1]), ([x0,y1,z0],[x0,y1,z1]),
    ]:
        draw_edge(e[0], e[1])

    alpha_sub = alpha_data[subwindow_x_start:subwindow_x_end,
                          subwindow_y_start:subwindow_y_end,
                          subwindow_z_start:subwindow_z_end]
    sm = None
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib import cm

        marching_cubes_fn = getattr(measure, 'marching_cubes', None) or getattr(measure, 'marching_cubes_lewiner', None)
        if marching_cubes_fn is not None:
            a_min, a_max = float(np.nanmin(alpha_sub)), float(np.nanmax(alpha_sub))
            levels = [0.2, 0.4, 0.6, 0.8]
            if a_max - a_min < 0.01:
                levels = [max(0.01, min(0.99, (a_min + a_max) / 2))]
            else:
                levels = [max(a_min + 0.01, min(a_max - 0.01, L)) for L in levels]
                levels = sorted(set(levels))
            for level in levels:
                try:
                    result = marching_cubes_fn(alpha_sub, level=float(level))
                    verts, faces = result[0], result[1]
                    if verts.size == 0 or faces.size == 0:
                        continue
                    verts_mm = (verts + np.array([subwindow_x_start, subwindow_y_start, subwindow_z_start])) * np.array([dx[0], dx[1], dx[2]]) * 1e3
                    color = plt.cm.viridis((level - a_min) / max(a_max - a_min, 0.01))
                    mesh = Poly3DCollection(verts_mm[faces], alpha=0.6, facecolor=color, edgecolor='none')
                    ax.add_collection3d(mesh)
                except (ValueError, RuntimeError, TypeError):
                    continue

        sm = cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
    except ImportError:
        pass

    focal_r_mm = focal_sphere_radius * dx[0] * 1e3
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 16)
    x_sph = fc_x_mm + focal_r_mm * np.outer(np.cos(u), np.sin(v))
    y_sph = fc_y_mm + focal_r_mm * np.outer(np.sin(u), np.sin(v))
    z_sph = fc_z_mm + focal_r_mm * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_sph, y_sph, z_sph, alpha=0.25, color='lime', edgecolor='none')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title(title)
    ax.set_xlim(0, N[0] * dx[0] * 1e3)
    ax.set_ylim(0, N[1] * dx[1] * 1e3)
    ax.set_zlim(0, N[2] * dx[2] * 1e3)
    if sm is not None:
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('alpha (material fraction)', fontsize=10)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


plot_3d_structure_view(alpha_init, '3D View - Initial Structure\n(FINER+Hash, subwindow, focus sphere)', os.path.join(results_dir, 'finer_hash_subwindow_3d_material_initial.png'))
plot_3d_structure_view(alpha_opt, '3D View - Optimized Structure', os.path.join(results_dir, 'finer_hash_subwindow_3d_material_optimized.png'))
print(f"Saved: finer_hash_subwindow_3d_material_initial.png, finer_hash_subwindow_3d_material_optimized.png")

print("\nDone!")
