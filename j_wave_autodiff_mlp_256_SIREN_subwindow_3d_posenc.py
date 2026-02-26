"""
3D SIREN Material Field — Subwindow Optimization (with sinusoidal position encoding)

Adds NeRF-style sinusoidal positional embeddings to (x, y, z) input coordinates:
gamma(p) = [p, sin(2^k pi p), cos(2^k pi p)] for k=0..L-1, for each coordinate.
"""

import numpy as np
from jax import numpy as jnp
from jax import lax, value_and_grad, vmap
import jax
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import os
import time as pytime

from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
_run_id = os.environ.get("AUTODIFF_RUN_ID", "")
results_dir = (
    os.path.join("Results/autodiff_siren_subwindow_3d_posenc", _run_id)
    if _run_id
    else "Results/autodiff_siren_subwindow_3d_posenc"
)
os.makedirs("Results/autodiff_siren_subwindow_3d_posenc", exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("=" * 70)
print("3D SIREN + POSITION ENCODING — SUBWINDOW")
print("=" * 70)
print("\nSIREN(gamma(x,y,z)) → α ∈ [0,1]. Material only in 3D subwindow.")
print("Maximize RMS pressure in focal sphere. 3D j-wave simulation.")
print("=" * 70)

# 3D grid
N = (64, 64, 64)



view_x_start, view_x_end = 0, N[0]
view_y_start, view_y_end = 0, N[1]
view_z_start, view_z_end = 0, N[2]

c_water = 1500.0
c_cylinder = 2500.0
rho_water = 1000.0
rho_cylinder = 1200.0

frequency = 1e6
wavelength = c_water / frequency

dx = (wavelength/10, wavelength/10, wavelength/10)
#dx = (0.4e-3, 0.4e-3, 0.4e-3)

print(f"Frequency: {frequency} Hz")
print(f"Wavelength: {wavelength} m")
print(f"Grid size: {N}")
print(f"Grid spacing: {dx}")
print(f"Domain(real size): {dx[0]*N[0]} m x {dx[1]*N[1]} m x {dx[2]*N[2]} m")
domain = Domain(N, dx)




design_boundary_x = 35
focus_boundary_x = 45

# Subwindow in 3D
subwindow_fraction = 0.50
subwindow_x_start = int(0.5 * (1.0 - subwindow_fraction) * design_boundary_x)
subwindow_x_end = int(0.5 * (1.0 + subwindow_fraction) * design_boundary_x)
subwindow_y_start = int(0.5 * (1.0 - subwindow_fraction) * N[1])
subwindow_y_end = int(0.5 * (1.0 + subwindow_fraction) * N[1])
subwindow_z_start = int(0.5 * (1.0 - subwindow_fraction) * N[2])
subwindow_z_end = int(0.5 * (1.0 + subwindow_fraction) * N[2])

# Focal sphere
target_focal_center = (50, 32, 32)
focal_sphere_radius = 4

training_time_end = 5.0e-05
stabilization_time = 3.0e-05

time_axis = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3, # cfl is defined as the ratio of the speed of sound to the grid spacing
    t_end=training_time_end,
)
time_array = time_axis.to_array()
# This creates the stabilization index which is the time index of the stabilization time
# We use jnp.argmin to find the index of the time array that is closest to the stabilization time
stabilization_index = int(jnp.argmin(jnp.abs(time_array - stabilization_time)))

# Positional encoding config
posenc_num_frequencies = int(os.environ.get("AUTODIFF_POSENC_FREQS", "4"))
siren_input_dim = 3 + 3 * 2 * posenc_num_frequencies

# Model option: "siren" (default) or "relu"
model_type = os.environ.get("AUTODIFF_MLP_TYPE", "siren").strip().lower()
if model_type not in ("siren", "relu"):
    print(f'WARNING: Unknown AUTODIFF_MLP_TYPE="{model_type}", falling back to "siren".')
    model_type = "siren"

siren_layer_sizes = [siren_input_dim, 64, 64, 32, 1]
siren_omega0 = 30.0
siren_seed = 42

print(f"\nSimulation Configuration:")
print(f"  Grid: {N[0]} × {N[1]} × {N[2]}")
print(f"  Simulation time: {training_time_end*1e6:.1f} μs")
print(f"  Target sphere: center {target_focal_center}, radius {focal_sphere_radius} pts")
print(f"  Position encoding frequencies: {posenc_num_frequencies}")
print(f"  SIREN input dim: {siren_input_dim}")
print(f"  SIREN layers: {siren_layer_sizes}, ω₀={siren_omega0}")
print(f"  MLP type: {model_type}")
print(
    f"  Subwindow: x=[{subwindow_x_start},{subwindow_x_end}], "
    f"y=[{subwindow_y_start},{subwindow_y_end}], "
    f"z=[{subwindow_z_start},{subwindow_z_end}]"
)
print("=" * 70)


def positional_encode_3d(xyz, num_frequencies):
    """xyz shape (..., 3) -> (..., 3 + 6*num_frequencies)."""
    freqs = 2.0 ** jnp.arange(num_frequencies)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    def enc_1d(v):
        s = jnp.sin(jnp.pi * v[..., None] * freqs)
        c = jnp.cos(jnp.pi * v[..., None] * freqs)
        return jnp.concatenate([s, c], axis=-1)

    return jnp.concatenate([xyz, enc_1d(x), enc_1d(y), enc_1d(z)], axis=-1)


def siren_init(key, layer_sizes, omega_0=30.0):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, k1 = jax.random.split(key)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        if i == 0:
            bound = 1.0 / fan_in
            W = jax.random.uniform(k1, (fan_in, fan_out), minval=-bound, maxval=bound)
        elif i == len(layer_sizes) - 2:
            bound = 1.0 / fan_in
            W = jax.random.uniform(k1, (fan_in, fan_out), minval=-bound, maxval=bound)
        else:
            bound = jnp.sqrt(6.0 / fan_in) / omega_0
            W = jax.random.uniform(k1, (fan_in, fan_out), minval=-bound, maxval=bound)
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def siren_apply(params, xyz_encoded, omega_0=30.0):
    h = xyz_encoded
    for (W, b) in params[:-1]:
        h = jnp.sin(omega_0 * (h @ W + b))
    Wlast, blast = params[-1]
    h = h @ Wlast + blast
    # This returns the 
    return jax.nn.sigmoid(jnp.squeeze(h))


def siren_apply_batch(params, grid_points, omega_0=siren_omega0):
    encoded = positional_encode_3d(grid_points, posenc_num_frequencies)
    
    return vmap(lambda p: siren_apply(params, p, omega_0), in_axes=(0,))(encoded)


def relu_mlp_init(key, layer_sizes):
    """Initialize standard ReLU MLP params (Xavier-style bounds)."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, k1 = jax.random.split(key)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        bound = jnp.sqrt(6.0 / (fan_in + fan_out))
        W = jax.random.uniform(k1, (fan_in, fan_out), minval=-bound, maxval=bound)
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def relu_mlp_apply(params, xyz_encoded):
    """Apply ReLU MLP, with sigmoid output so alpha stays in [0, 1]."""
    h = xyz_encoded
    for (W, b) in params[:-1]:
        h = jax.nn.relu(h @ W + b)
    Wlast, blast = params[-1]
    h = h @ Wlast + blast
    return jax.nn.sigmoid(jnp.squeeze(h))


def relu_mlp_apply_batch(params, grid_points):
    encoded = positional_encode_3d(grid_points, posenc_num_frequencies)
    return vmap(lambda p: relu_mlp_apply(params, p), in_axes=(0,))(encoded)


def material_field_from_mlp(params):
    # Build normalized coordinates only for the subwindow (MLP evaluated here only).
    xs_full = jnp.linspace(0.0, 1.0, N[0])
    ys_full = jnp.linspace(0.0, 1.0, N[1])
    zs_full = jnp.linspace(0.0, 1.0, N[2])
    xs_sw = xs_full[subwindow_x_start : subwindow_x_end + 1]
    ys_sw = ys_full[subwindow_y_start : subwindow_y_end + 1]
    zs_sw = zs_full[subwindow_z_start : subwindow_z_end + 1]
    X_sw, Y_sw, Z_sw = jnp.meshgrid(xs_sw, ys_sw, zs_sw, indexing="ij")
    grid_points_sw = jnp.stack([X_sw.ravel(), Y_sw.ravel(), Z_sw.ravel()], axis=1)

    nx_sw = subwindow_x_end - subwindow_x_start + 1
    ny_sw = subwindow_y_end - subwindow_y_start + 1
    nz_sw = subwindow_z_end - subwindow_z_start + 1

    if model_type == "relu":
        alpha_sw = relu_mlp_apply_batch(params, grid_points_sw).reshape(nx_sw, ny_sw, nz_sw)
    else:
        alpha_sw = siren_apply_batch(params, grid_points_sw).reshape(nx_sw, ny_sw, nz_sw)

    # Design mask within subwindow (material only for x < design_boundary_x)
    design_mask_sw = (
        (subwindow_x_start + jnp.arange(nx_sw)) < design_boundary_x
    )[:, jnp.newaxis, jnp.newaxis]
    alpha_sw = alpha_sw * design_mask_sw

    # Full grid: water everywhere, then write subwindow from MLP
    sound_speed = jnp.full(N, c_water).at[
        subwindow_x_start : subwindow_x_end + 1,
        subwindow_y_start : subwindow_y_end + 1,
        subwindow_z_start : subwindow_z_end + 1,
    ].set(c_water + (c_cylinder - c_water) * alpha_sw)
    density = jnp.full(N, rho_water).at[
        subwindow_x_start : subwindow_x_end + 1,
        subwindow_y_start : subwindow_y_end + 1,
        subwindow_z_start : subwindow_z_end + 1,
    ].set(rho_water + (rho_cylinder - rho_water) * alpha_sw)
    return sound_speed, density


def create_sphere_mask_3d_smooth(N, center, radius, steepness=2.0):
    x = jnp.arange(N[0], dtype=jnp.float32)
    y = jnp.arange(N[1], dtype=jnp.float32)
    z = jnp.arange(N[2], dtype=jnp.float32)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    cx, cy, cz = center
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return jax.nn.sigmoid(steepness * (radius**2 - dist_sq))


def create_medium_from_material_field(domain, sound_speed, density):
    # This expands the dimensions of the sound speed and density fields to match the shape of the medium
    sound_speed = jnp.expand_dims(sound_speed, -1)
    density = jnp.expand_dims(density, -1)
    # This creates the medium from the sound speed and density fields
    return Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=20)


def run_simulation_from_medium(medium, sim_time_axis=None):
    if sim_time_axis is None:
        sim_time_axis = time_axis
    # This creates the initial condition
    p0_array = jnp.zeros(N)
    p0_array = jnp.expand_dims(p0_array, -1)
    p0 = FourierSeries(p0_array, domain)
    # TimeVaryingSource: drives the wave equation with a time-harmonic plane source at the
    # left boundary. The solver (simulate_wave_propagation) expects a source object with
    # on_grid(ti) returning the source field at time index ti. Here we precompute the full
    # 4D field (time, x, y, z, 1): only the first 12 x-slices are non-zero because of the subwindow, forming a
    # source plane, and each time step t is sin(omega*t) so the source oscillates at
    # `frequency` Hz. on_grid(ti) then returns that precomputed slice via dynamic_slice,
    # which keeps the simulation JAX-differentiable and avoids Python callbacks per step.
    class TimeVaryingSource:
        def __init__(self, ta, domain):
            self.omega = 2 * jnp.pi * frequency
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

    src = TimeVaryingSource(sim_time_axis, domain)
    return simulate_wave_propagation(medium, sim_time_axis, p0=p0, sources=src)


def compute_objective_mlp(params):
    # This creates the sound speed and density fields from the MLP of just the design space
    sound_speed, density = material_field_from_mlp(params)
    # This creates the medium from the sound speed and density fields given by the MLP
    medium = create_medium_from_material_field(domain, sound_speed, density)
    pressure_field = run_simulation_from_medium(medium)
    pressure_on_grid = pressure_field.on_grid
    sphere_mask = create_sphere_mask_3d_smooth(N, target_focal_center, focal_sphere_radius)
    stabilized_pressure = pressure_on_grid[stabilization_index:, :, :, :, 0]
    pressure_in_sphere = stabilized_pressure * jnp.expand_dims(sphere_mask, axis=0)
    n_points = jnp.sum(sphere_mask)
    total_sq = jnp.sum(pressure_in_sphere**2)
    n_time = stabilized_pressure.shape[0]
    mean_sq = total_sq / jnp.maximum(n_time * n_points, 1.0)
    # This returns the negative of the root mean square pressure in the sphere
    return -jnp.sqrt(mean_sq)


key = jax.random.PRNGKey(siren_seed)
if model_type == "relu":
    initial_params = relu_mlp_init(key, siren_layer_sizes)
else:
    initial_params = siren_init(key, siren_layer_sizes, siren_omega0)

learning_rate = float(os.environ.get("AUTODIFF_LR", "0.0005"))
n_iterations = int(os.environ.get("AUTODIFF_N_ITER", "200"))

params = initial_params
best_objective = float("inf")
best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), initial_params)
best_iteration = 0

optimization_history = {
    "objectives": [],
    "pressures": [],
    "gradient_norms": [],
    "iteration_times": [],
}

print("\n" + "=" * 70)
print("OPTIMIZATION: SIREN + POSITION ENCODING (3D)")
print("=" * 70)
print(f"  Iterations: {n_iterations}, LR: {learning_rate}\n")

@jax.jit
def objective_and_grad(params):
    return value_and_grad(compute_objective_mlp)(params)

for iteration in range(n_iterations):
    iter_start = pytime.time()
    print(f"Iteration {iteration + 1}/{n_iterations}")
    obj_val, grad_params = objective_and_grad(params)
    obj_float = float(obj_val)
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for (W, b) in grad_params for g in [W, b])
    )
    print(f"  Loss: {obj_val:.6f}, RMS: {-obj_val:.6f} Pa")

    if obj_float < best_objective:
        best_objective = obj_float
        best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
        best_iteration = iteration + 1
        print("  *** New best! ***")

    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grad_params)
    optimization_history["objectives"].append(obj_float)
    optimization_history["pressures"].append(float(-obj_val))
    optimization_history["gradient_norms"].append(float(grad_norm))
    optimization_history["iteration_times"].append(pytime.time() - iter_start)
    print(f"  Time: {optimization_history['iteration_times'][-1]:.2f} s")

final_params = best_params
first_rms = optimization_history["pressures"][0]
print(f"\nBest iter {best_iteration}, RMS {-best_objective:.6f} Pa")
print(f"Improvement: {(-best_objective / max(first_rms, 1e-10) - 1) * 100:.1f}%")

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

time_axis_full = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water), cfl=0.3, t_end=8.0e-05
)
time_array_full = time_axis_full.to_array()
stab_idx_full = int(jnp.argmin(jnp.abs(time_array_full - 5.0e-05)))
sphere_mask_np = np.array(
    create_sphere_mask_3d_smooth(N, target_focal_center, focal_sphere_radius)
)
n_sphere_pts = np.sum(sphere_mask_np)
n_time_val = len(time_array_full) - stab_idx_full


def run_val(pp):
    ss, dd = material_field_from_mlp(pp)
    med = create_medium_from_material_field(domain, ss, dd)
    pf = run_simulation_from_medium(med, time_axis_full)
    return np.array(pf.on_grid)[:, :, :, :, 0]


print("Running initial config...")
p_init = run_val(initial_params)
rms_init_val = np.sqrt(
    np.sum((p_init[stab_idx_full:] * sphere_mask_np) ** 2) / (n_time_val * n_sphere_pts)
)
print(f"  RMS: {rms_init_val:.6f} Pa")

print("Running optimized config...")
p_opt = run_val(final_params)
rms_opt_val = np.sqrt(
    np.sum((p_opt[stab_idx_full:] * sphere_mask_np) ** 2) / (n_time_val * n_sphere_pts)
)
print(f"  RMS: {rms_opt_val:.6f} Pa")

improvement_full = (rms_opt_val / max(rms_init_val, 1e-10) - 1) * 100
print(f"  Improvement: {improvement_full:.1f}%")

# Visualization helpers
# Slice indices aligned with focal sphere center so all planes intersect sphere
slice_z = int(target_focal_center[2])
slice_y = int(target_focal_center[1])
slice_x = int(target_focal_center[0])
extent_xy = [0, N[0] * dx[0] * 1e3, 0, N[1] * dx[1] * 1e3]
extent_xz = [0, N[0] * dx[0] * 1e3, 0, N[2] * dx[2] * 1e3]
extent_yz = [0, N[1] * dx[1] * 1e3, 0, N[2] * dx[2] * 1e3]
design_x_mm = design_boundary_x * dx[0] * 1e3
focus_x_mm = focus_boundary_x * dx[0] * 1e3
fc_x_mm = target_focal_center[0] * dx[0] * 1e3
fc_y_mm = target_focal_center[1] * dx[1] * 1e3
fc_z_mm = target_focal_center[2] * dx[2] * 1e3
focal_r_mm = focal_sphere_radius * dx[0] * 1e3


def add_boundary_plane(ax, x_grid_mm):
    y_plane = np.linspace(0, N[1] * dx[1] * 1e3, 12)
    z_plane = np.linspace(0, N[2] * dx[2] * 1e3, 12)
    Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
    X_plane = np.ones_like(Y_plane) * x_grid_mm
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.15, color="gray", shade=False)


def draw_focus_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 16)
    x_sph = fc_x_mm + focal_r_mm * np.outer(np.cos(u), np.sin(v))
    y_sph = fc_y_mm + focal_r_mm * np.outer(np.sin(u), np.sin(v))
    z_sph = fc_z_mm + focal_r_mm * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_sph, y_sph, z_sph, alpha=0.25, color="lime", edgecolor="none")


p_mean_init = np.mean(np.abs(p_init[stab_idx_full:]), axis=0)
p_mean_opt = np.mean(np.abs(p_opt[stab_idx_full:]), axis=0)
vmax_slice = max(np.max(p_mean_init), np.max(p_mean_opt))

alpha_init = np.array(material_field_from_mlp(initial_params)[0])
alpha_init = np.clip((alpha_init - c_water) / (c_cylinder - c_water), 0, 1)
alpha_opt = np.array(material_field_from_mlp(final_params)[0])
alpha_opt = np.clip((alpha_opt - c_water) / (c_cylinder - c_water), 0, 1)

# Keep same primary output name family and style
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
ax = axes[0, 0]
ax.plot(optimization_history["pressures"], "b-o", linewidth=2, markersize=6)
ax.axhline(rms_init_val, color="r", linestyle="--", alpha=0.5, label="Initial (val)")
ax.axhline(rms_opt_val, color="g", linestyle="--", alpha=0.5, label="Optimized (val)")
ax.set_xlabel("Iteration")
ax.set_ylabel("RMS Pressure (Pa)")
ax.set_title("SIREN+PosEnc Optimization Progress")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.semilogy(optimization_history["gradient_norms"], "b-o", linewidth=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Gradient Norm")
ax.set_title("Gradient Magnitude")
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
im = ax.imshow(alpha_init[:, :, slice_z].T, cmap="viridis", origin="lower", extent=extent_xy, vmin=0, vmax=1)
ax.axvline(design_x_mm, color="red", linewidth=1.5, linestyle=":", alpha=0.5)
ax.axvline(focus_x_mm, color="cyan", linewidth=1.5, linestyle=":", alpha=0.5)
ax.add_patch(
    Rectangle(
        (subwindow_x_start * dx[0] * 1e3, subwindow_y_start * dx[1] * 1e3),
        (subwindow_x_end - subwindow_x_start) * dx[0] * 1e3,
        (subwindow_y_end - subwindow_y_start) * dx[1] * 1e3,
        fill=False,
        edgecolor="white",
        linewidth=2,
        linestyle="--",
    )
)
ax.set_title(f"Initial Material α (XY @ z={slice_z})")
plt.colorbar(im, ax=ax, label="α", fraction=0.046)

ax = axes[0, 3]
im = ax.imshow(alpha_opt[:, :, slice_z].T, cmap="viridis", origin="lower", extent=extent_xy, vmin=0, vmax=1)
ax.axvline(design_x_mm, color="red", linewidth=1.5, linestyle=":", alpha=0.5)
ax.axvline(focus_x_mm, color="cyan", linewidth=1.5, linestyle=":", alpha=0.5)
ax.add_patch(
    Rectangle(
        (subwindow_x_start * dx[0] * 1e3, subwindow_y_start * dx[1] * 1e3),
        (subwindow_x_end - subwindow_x_start) * dx[0] * 1e3,
        (subwindow_y_end - subwindow_y_start) * dx[1] * 1e3,
        fill=False,
        edgecolor="white",
        linewidth=2,
        linestyle="--",
    )
)
ax.set_title(f"Optimized Material α (XY @ z={slice_z})")
plt.colorbar(im, ax=ax, label="α", fraction=0.046)

ax = axes[1, 0]
im = ax.imshow(p_mean_init[:, :, slice_z].T, cmap="hot", origin="lower", extent=extent_xy, vmin=0, vmax=vmax_slice)
ax.axvline(design_x_mm, color="gray", linewidth=1, linestyle="--", alpha=0.6)
ax.axvline(focus_x_mm, color="cyan", linewidth=1, linestyle="--", alpha=0.6)
ax.add_patch(MplCircle((fc_x_mm, fc_y_mm), focal_r_mm, fill=False, edgecolor="lime", linewidth=2))
ax.set_title(f"Initial |P| — RMS: {rms_init_val:.4f} Pa")
plt.colorbar(im, ax=ax, label="|P| (Pa)", fraction=0.046)

ax = axes[1, 1]
im = ax.imshow(p_mean_opt[:, :, slice_z].T, cmap="hot", origin="lower", extent=extent_xy, vmin=0, vmax=vmax_slice)
ax.axvline(design_x_mm, color="gray", linewidth=1, linestyle="--", alpha=0.6)
ax.axvline(focus_x_mm, color="cyan", linewidth=1, linestyle="--", alpha=0.6)
ax.add_patch(MplCircle((fc_x_mm, fc_y_mm), focal_r_mm, fill=False, edgecolor="lime", linewidth=2))
ax.set_title(f"Optimized |P| — RMS: {rms_opt_val:.4f} Pa (+{improvement_full:.1f}%)")
plt.colorbar(im, ax=ax, label="|P| (Pa)", fraction=0.046)

ax = axes[1, 2]
ax.plot(optimization_history["iteration_times"], "g-o", linewidth=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Time (s)")
ax.set_title("Iteration Time")
ax.grid(True, alpha=0.3)

ax = axes[1, 3]
ax.axis("off")
summary = f"""
3D MLP + POSITION ENCODING SUMMARY

PosEnc frequencies: {posenc_num_frequencies}
Input dim: {siren_input_dim}
MLP type: {model_type}
Layers: {siren_layer_sizes}, ω₀={siren_omega0}

Results:
  Initial RMS: {rms_init_val:.6f} Pa
  Optimized RMS: {rms_opt_val:.6f} Pa
  Improvement: {improvement_full:.1f}%
"""
ax.text(
    0.05,
    0.95,
    summary,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
)

plt.suptitle("3D SIREN+PosEnc — Subwindow (GINN + J-Wave)", fontsize=14, fontweight="bold")
plt.tight_layout()
main_fig = os.path.join(results_dir, "siren_subwindow_3d_posenc_main.png")
plt.savefig(main_fig, dpi=150, bbox_inches="tight")
print(f"Saved: {main_fig}")
plt.close(fig)

# -------------------------------------------------------------------------
# Figure 2: All 3 planes — Pressure (initial vs optimized)
# -------------------------------------------------------------------------

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))


def plot_pressure_slice_row(axes_row, p_mean, is_opt=False):
    # Sphere intersection radii on each slicing plane
    z_slice = slice_z * dx[2] * 1e3
    r_xy = np.sqrt(max(0.0, focal_r_mm**2 - (z_slice - fc_z_mm) ** 2))
    y_slice = slice_y * dx[1] * 1e3
    r_xz = np.sqrt(max(0.0, focal_r_mm**2 - (y_slice - fc_y_mm) ** 2))
    x_slice = slice_x * dx[0] * 1e3
    r_yz = np.sqrt(max(0.0, focal_r_mm**2 - (x_slice - fc_x_mm) ** 2))

    # XY
    im = axes_row[0].imshow(
        p_mean[:, :, slice_z].T, cmap="hot", origin="lower", extent=extent_xy, vmin=0, vmax=vmax_slice
    )
    axes_row[0].axvline(design_x_mm, color="gray", linewidth=1.5, linestyle="--", alpha=0.7)
    axes_row[0].axvline(focus_x_mm, color="cyan", linewidth=1.5, linestyle="--", alpha=0.7)
    if r_xy > 0:
        axes_row[0].add_patch(MplCircle((fc_x_mm, fc_y_mm), r_xy, fill=False, edgecolor="lime", linewidth=2))
    axes_row[0].set_xlabel("x (mm)")
    axes_row[0].set_ylabel("y (mm)")
    axes_row[0].set_title(f'{"Optimized" if is_opt else "Initial"} — XY slice')
    plt.colorbar(im, ax=axes_row[0], label="|P| (Pa)", fraction=0.046)

    # XZ
    im = axes_row[1].imshow(
        p_mean[:, slice_y, :].T, cmap="hot", origin="lower", extent=extent_xz, vmin=0, vmax=vmax_slice
    )
    axes_row[1].axvline(design_x_mm, color="gray", linewidth=1.5, linestyle="--", alpha=0.7)
    axes_row[1].axvline(focus_x_mm, color="cyan", linewidth=1.5, linestyle="--", alpha=0.7)
    if r_xz > 0:
        axes_row[1].add_patch(MplCircle((fc_x_mm, fc_z_mm), r_xz, fill=False, edgecolor="lime", linewidth=2))
    axes_row[1].set_xlabel("x (mm)")
    axes_row[1].set_ylabel("z (mm)")
    axes_row[1].set_title(f'{"Optimized" if is_opt else "Initial"} — XZ slice')
    plt.colorbar(im, ax=axes_row[1], label="|P| (Pa)", fraction=0.046)

    # YZ
    im = axes_row[2].imshow(
        p_mean[slice_x, :, :].T, cmap="hot", origin="lower", extent=extent_yz, vmin=0, vmax=vmax_slice
    )
    if r_yz > 0:
        axes_row[2].add_patch(MplCircle((fc_y_mm, fc_z_mm), r_yz, fill=False, edgecolor="lime", linewidth=2))
    axes_row[2].set_xlabel("y (mm)")
    axes_row[2].set_ylabel("z (mm)")
    axes_row[2].set_title(f'{"Optimized" if is_opt else "Initial"} — YZ slice')
    plt.colorbar(im, ax=axes_row[2], label="|P| (Pa)", fraction=0.046)


plot_pressure_slice_row(axes2[0], p_mean_init, False)
plot_pressure_slice_row(axes2[1], p_mean_opt, True)

pressure_slices_fig = os.path.join(results_dir, "siren_subwindow_3d_pressure_slices.png")
plt.suptitle("Initial vs Optimized — All 3 Planes (XY, XZ, YZ) | Circle = focus sphere", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(pressure_slices_fig, dpi=150, bbox_inches="tight")
print(f"Saved: {pressure_slices_fig}")
plt.close(fig2)

# -------------------------------------------------------------------------
# Figure 3: All 3 planes — Material α (initial vs optimized)
# -------------------------------------------------------------------------

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))


def plot_alpha_slice_row(axes_row, alpha_data, label):
    z_slice = slice_z * dx[2] * 1e3
    r_xy = np.sqrt(max(0.0, focal_r_mm**2 - (z_slice - fc_z_mm) ** 2))
    y_slice = slice_y * dx[1] * 1e3
    r_xz = np.sqrt(max(0.0, focal_r_mm**2 - (y_slice - fc_y_mm) ** 2))
    x_slice = slice_x * dx[0] * 1e3
    r_yz = np.sqrt(max(0.0, focal_r_mm**2 - (x_slice - fc_x_mm) ** 2))

    # XY
    im = axes_row[0].imshow(alpha_data[:, :, slice_z].T, cmap="viridis", origin="lower", extent=extent_xy, vmin=0, vmax=1)
    axes_row[0].axvline(design_x_mm, color="red", linewidth=1, linestyle=":", alpha=0.5)
    axes_row[0].axvline(focus_x_mm, color="cyan", linewidth=1, linestyle=":", alpha=0.5)
    if r_xy > 0:
        axes_row[0].add_patch(MplCircle((fc_x_mm, fc_y_mm), r_xy, fill=False, edgecolor="lime", linewidth=2))
    axes_row[0].set_xlabel("x (mm)")
    axes_row[0].set_ylabel("y (mm)")
    axes_row[0].set_title(f"{label} — XY slice")
    plt.colorbar(im, ax=axes_row[0], label="α", fraction=0.046)

    # XZ
    im = axes_row[1].imshow(alpha_data[:, slice_y, :].T, cmap="viridis", origin="lower", extent=extent_xz, vmin=0, vmax=1)
    axes_row[1].axvline(design_x_mm, color="red", linewidth=1, linestyle=":", alpha=0.5)
    axes_row[1].axvline(focus_x_mm, color="cyan", linewidth=1, linestyle=":", alpha=0.5)
    if r_xz > 0:
        axes_row[1].add_patch(MplCircle((fc_x_mm, fc_z_mm), r_xz, fill=False, edgecolor="lime", linewidth=2))
    axes_row[1].set_xlabel("x (mm)")
    axes_row[1].set_ylabel("z (mm)")
    axes_row[1].set_title(f"{label} — XZ slice")
    plt.colorbar(im, ax=axes_row[1], label="α", fraction=0.046)

    # YZ
    im = axes_row[2].imshow(alpha_data[slice_x, :, :].T, cmap="viridis", origin="lower", extent=extent_yz, vmin=0, vmax=1)
    if r_yz > 0:
        axes_row[2].add_patch(MplCircle((fc_y_mm, fc_z_mm), r_yz, fill=False, edgecolor="lime", linewidth=2))
    axes_row[2].set_xlabel("y (mm)")
    axes_row[2].set_ylabel("z (mm)")
    axes_row[2].set_title(f"{label} — YZ slice")
    plt.colorbar(im, ax=axes_row[2], label="α", fraction=0.046)


plot_alpha_slice_row(axes3[0], alpha_init, "Initial")
plot_alpha_slice_row(axes3[1], alpha_opt, "Optimized")

material_slices_fig = os.path.join(results_dir, "siren_subwindow_3d_material_slices.png")
plt.suptitle("Material Field α — All 3 Planes", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(material_slices_fig, dpi=150, bbox_inches="tight")
print(f"Saved: {material_slices_fig}")
plt.close(fig3)


def plot_3d_structure_view(alpha_data, title, fpath):
    """Robust 3D structure plot with adaptive levels + fallback scatter."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    add_boundary_plane(ax, design_x_mm)
    add_boundary_plane(ax, focus_x_mm)

    # Subwindow box
    x0 = subwindow_x_start * dx[0] * 1e3
    x1 = subwindow_x_end * dx[0] * 1e3
    y0 = subwindow_y_start * dx[1] * 1e3
    y1 = subwindow_y_end * dx[1] * 1e3
    z0 = subwindow_z_start * dx[2] * 1e3
    z1 = subwindow_z_end * dx[2] * 1e3

    def draw_edge(p1, p2):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k-", linewidth=2)

    draw_edge([x0, y0, z0], [x1, y0, z0]); draw_edge([x1, y0, z0], [x1, y1, z0])
    draw_edge([x1, y1, z0], [x0, y1, z0]); draw_edge([x0, y1, z0], [x0, y0, z0])
    draw_edge([x0, y0, z1], [x1, y0, z1]); draw_edge([x1, y0, z1], [x1, y1, z1])
    draw_edge([x1, y1, z1], [x0, y1, z1]); draw_edge([x0, y1, z1], [x0, y0, z1])
    draw_edge([x0, y0, z0], [x0, y0, z1]); draw_edge([x1, y0, z0], [x1, y0, z1])
    draw_edge([x1, y1, z0], [x1, y1, z1]); draw_edge([x0, y1, z0], [x0, y1, z1])

    alpha_sub = alpha_data[
        subwindow_x_start:subwindow_x_end, subwindow_y_start:subwindow_y_end, subwindow_z_start:subwindow_z_end
    ]
    sm = None
    plotted = False
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib import cm

        nz = alpha_sub[alpha_sub > 1e-6]
        if nz.size > 0:
            q = np.quantile(nz, [0.35, 0.55, 0.75, 0.9])
            levels = np.unique(np.clip(q, 1e-4, 0.999))
        else:
            levels = np.array([0.1, 0.2, 0.3])

        for level in levels:
            try:
                verts, faces, _, _ = measure.marching_cubes(alpha_sub, level=float(level))
                verts_mm = (
                    verts + np.array([subwindow_x_start, subwindow_y_start, subwindow_z_start])
                ) * np.array([dx[0], dx[1], dx[2]]) * 1e3
                mesh = Poly3DCollection(
                    verts_mm[faces], alpha=0.6, facecolor=plt.cm.viridis(float(level)), edgecolor="none"
                )
                ax.add_collection3d(mesh)
                plotted = True
            except (ValueError, RuntimeError):
                continue

        sm = cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
    except ImportError:
        pass

    # Fallback if no surface extracted (common in initial state)
    if not plotted:
        idx = np.argwhere(alpha_sub > max(0.02, float(np.max(alpha_sub) * 0.2)))
        if idx.size > 0:
            # Subsample for readability
            idx = idx[:: max(1, len(idx) // 4000)]
            vals = alpha_sub[idx[:, 0], idx[:, 1], idx[:, 2]]
            xs = (idx[:, 0] + subwindow_x_start) * dx[0] * 1e3
            ys = (idx[:, 1] + subwindow_y_start) * dx[1] * 1e3
            zs = (idx[:, 2] + subwindow_z_start) * dx[2] * 1e3
            sc = ax.scatter(xs, ys, zs, c=vals, cmap="viridis", s=4, alpha=0.6)
            if sm is None:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])

    draw_focus_sphere(ax)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    ax.set_xlim(0, N[0] * dx[0] * 1e3)
    ax.set_ylim(0, N[1] * dx[1] * 1e3)
    ax.set_zlim(0, N[2] * dx[2] * 1e3)
    if sm is not None:
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label("α (material fraction)", fontsize=10)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


plot_3d_structure_view(
    alpha_init,
    "3D View — Initial Structure (PosEnc)",
    os.path.join(results_dir, "siren_subwindow_3d_posenc_material_initial.png"),
)
plot_3d_structure_view(
    alpha_opt,
    "3D View — Optimized Structure (PosEnc)",
    os.path.join(results_dir, "siren_subwindow_3d_posenc_material_optimized.png"),
)
print("Saved 3D structure views (initial/optimized).")
print("\nDone!")