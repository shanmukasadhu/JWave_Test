import numpy as np
from jax import numpy as jnp
from jax import grad, jit, lax, value_and_grad, vmap
import jax
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle, Rectangle
import os
import time as pytime

# Import jwave components
from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis

# Results directory
_run_id = os.environ.get('AUTODIFF_RUN_ID', '')
results_dir = os.path.join('Results/autodiff_mlp_256_SIREN_subwindow_multiobjective', _run_id) if _run_id else 'Results/autodiff_mlp_256_SIREN_subwindow_multiobjective'
os.makedirs('Results/autodiff_mlp_256_SIREN_subwindow_multiobjective', exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("="*70)
print("SIREN MATERIAL FIELD — SUBWINDOW + MULTI-OBJECTIVE (2 FOCAL CIRCLES)")
print("="*70)
print("\nMaximize 0.5×RMS(circle1) + 0.5×RMS(circle2). Subwindow 75% of design.")
print("="*70)

# Simulation parameters: 256×256 grid, full domain (no window)
N = (256, 256)
dx = (0.2e-3, 0.2e-3)
domain = Domain(N, dx)

# Full grid (no viewing window)
view_x_start = 0
view_x_end = N[0]
view_y_start = 0
view_y_end = N[1]

# Medium properties
c_water = 1500.0
c_cylinder = 2500.0
rho_water = 1000.0
rho_cylinder = 1200.0

# Wave parameters
frequency = 1.0e6
wavelength = c_water / frequency

# Design/Focus space boundaries (scaled for 256 grid)
design_boundary_x = 140
focus_boundary_x = 150

# Subwindow: 75% of design space (centered). Model can place material only here; rest is water.
subwindow_fraction = 0.50
subwindow_x_start = int(0.5 * (1.0 - subwindow_fraction) * design_boundary_x)  # centered in x
subwindow_x_end = int(0.5 * (1.0 + subwindow_fraction) * design_boundary_x)
subwindow_y_start = int(0.5 * (1.0 - subwindow_fraction) * N[1])  # centered in y
subwindow_y_end = int(0.5 * (1.0 + subwindow_fraction) * N[1])

# Target focal regions - multi-objective: maximize RMS at BOTH disks (weighted 50/50)
target_focal_center_1 = (160, 96)   # first circle
target_focal_center_2 = (160, 160)  # second circle (different y)
focal_disk_radius = 5  # grid points (same for both)

# Training time
training_time_end = 1.5e-04   # 150 μs
stabilization_time = 1.0e-04  # 100 μs

time_axis = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3,
    t_end=training_time_end
)
time_array = time_axis.to_array()
stabilization_index = int(jnp.argmin(jnp.abs(time_array - stabilization_time)))

# SIREN architecture: (x,y) -> [0,1] material. Sinusoidal activations (Sitzmann et al.)
siren_layer_sizes = [2, 64, 64, 32, 1]
siren_omega0 = 30.0
siren_seed = 42

print(f"\nSimulation Configuration:")
print(f"  Simulation time: {training_time_end*1e6:.1f} μs")
print(f"  Stabilization: {stabilization_time*1e6:.1f} μs")
print(f"  Target focal disks: center1 {target_focal_center_1}, center2 {target_focal_center_2}, radius {focal_disk_radius} pts")
print(f"  Grid: {N[0]} × {N[1]}")
print(f"  SIREN: {siren_layer_sizes}, ω₀={siren_omega0} (output α ∈ [0,1] continuous)")
print(f"  Subwindow: x=[{subwindow_x_start},{subwindow_x_end}], y=[{subwindow_y_start},{subwindow_y_end}] (75% of design)")
print(f"  Objective: maximize RMS pressure over focal disk")
print("="*70)

# ============================================================================
# SIREN: (x, y) -> material in [0, 1] via sin(ω₀·(Wx+b)) hidden layers
# ============================================================================

def siren_init(key, layer_sizes, omega_0=30.0):
    """Initialize SIREN params. First layer: U(-1/in,1/in). Hidden: U(-√(6/in)/ω₀, √(6/in)/ω₀). Last: small."""
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

def siren_apply(params, xy, omega_0=30.0):
    """Apply SIREN: hidden layers sin(ω₀·(Wx+b)), last layer linear then sigmoid."""
    h = xy
    for (W, b) in params[:-1]:
        h = jnp.sin(omega_0 * (h @ W + b))
    Wlast, blast = params[-1]
    h = h @ Wlast + blast
    return jax.nn.sigmoid(jnp.squeeze(h))

# Batched application over grid points (omega_0 fixed in closure for vmap)
def siren_apply_batch(params, grid_points, omega_0=siren_omega0):
    return vmap(lambda p: siren_apply(params, p, omega_0), in_axes=(0,))(grid_points)

def material_field_from_mlp(params):
    """
    Sample SIREN on the grid to get material α in [0,1] (continuous).
    Material is allowed ONLY inside the subwindow (75% of design space); rest is water.
    """
    xs = jnp.linspace(0.0, 1.0, N[0])
    ys = jnp.linspace(0.0, 1.0, N[1])
    X, Y = jnp.meshgrid(xs, ys, indexing='ij')
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (N[0]*N[1], 2)

    alpha = siren_apply_batch(params, grid_points)  # (N[0]*N[1],)
    alpha = alpha.reshape(N[0], N[1])

    # Design region only: zero material in focus/buffer (x >= design_boundary_x)
    x_norm_design = design_boundary_x / (N[0] - 1.0)
    X_norm = jnp.linspace(0.0, 1.0, N[0])
    design_mask = (X_norm < x_norm_design)[:, jnp.newaxis]  # (N[0], 1)

    # Subwindow: only inside [subwindow_x_start, subwindow_x_end] x [subwindow_y_start, subwindow_y_end]
    x_norm_lo = subwindow_x_start / (N[0] - 1.0)
    x_norm_hi = subwindow_x_end / (N[0] - 1.0)
    y_norm_lo = subwindow_y_start / (N[1] - 1.0)
    y_norm_hi = subwindow_y_end / (N[1] - 1.0)
    subwindow_mask = (
        (X >= x_norm_lo) & (X <= x_norm_hi) &
        (Y >= y_norm_lo) & (Y <= y_norm_hi)
    )

    alpha = alpha * design_mask * subwindow_mask

    sound_speed = c_water + (c_cylinder - c_water) * alpha
    density = rho_water + (rho_cylinder - rho_water) * alpha
    return sound_speed, density

def create_medium_from_material_field(domain, sound_speed, density):
    """Build jwave Medium from 2D sound_speed and density arrays."""
    sound_speed = jnp.expand_dims(sound_speed, -1)
    density = jnp.expand_dims(density, -1)
    return Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=60)

def run_simulation_from_medium(medium, sim_time_axis=None):
    """Run j-wave simulation from a Medium. Fully differentiable w.r.t. medium fields.
    If sim_time_axis is None, uses global time_axis (training length)."""
    if sim_time_axis is None:
        sim_time_axis = time_axis
    p0_array = jnp.zeros(N)
    p0_array = jnp.expand_dims(p0_array, -1)
    p0 = FourierSeries(p0_array, domain)

    class TimeVaryingSource:
        def __init__(self, ta, domain):
            self.domain = domain
            self.omega = 2 * jnp.pi * frequency
            self.time_array = ta.to_array()
            source_mask = jnp.zeros(N)
            source_mask = source_mask.at[:40, :].set(1.0)
            source_mask = jnp.expand_dims(source_mask, -1)
            source_fields_list = []
            for t in self.time_array:
                source_amplitude = jnp.sin(self.omega * t)
                source_fields_list.append(source_mask * source_amplitude)
            self.source_fields = jnp.stack(source_fields_list, axis=0)

        def on_grid(self, time_index):
            time_idx = lax.convert_element_type(time_index, jnp.int32)
            start_indices = (time_idx, 0, 0, 0)
            slice_sizes = (1, N[0], N[1], 1)
            sliced = lax.dynamic_slice(self.source_fields, start_indices, slice_sizes)
            return jnp.squeeze(sliced, axis=0)

    time_varying_source = TimeVaryingSource(sim_time_axis, domain)
    pressure_field = simulate_wave_propagation(
        medium, sim_time_axis, p0=p0, sources=time_varying_source
    )
    return pressure_field

def create_disk_mask(N, center, radius):
    """Create a circular disk mask in the x-y plane."""
    x = jnp.arange(N[0])
    y = jnp.arange(N[1])
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    dist = jnp.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist <= radius

def _rms_over_disk(pressure_3d, disk_mask):
    """Compute RMS pressure over a disk region. pressure_3d: (time, x, y)."""
    disk_expanded = jnp.expand_dims(disk_mask, axis=0)
    pressure_in_disk = pressure_3d * disk_expanded
    n_points = jnp.sum(disk_mask)
    n_time = pressure_3d.shape[0]
    n_total = n_time * n_points
    mean_sq = jnp.sum(pressure_in_disk ** 2) / jnp.maximum(n_total, 1.0)
    return jnp.sqrt(mean_sq)

def compute_objective_mlp(params):
    """
    Multi-objective: maximize 0.5×RMS(disk1) + 0.5×RMS(disk2).
    Loss = -(0.5×rms1 + 0.5×rms2). Fully differentiable.
    Returns (loss, pressure_on_grid) for aux output (no extra simulations for viz).
    """
    sound_speed, density = material_field_from_mlp(params)
    medium = create_medium_from_material_field(domain, sound_speed, density)
    pressure_field = run_simulation_from_medium(medium)

    pressure_on_grid = pressure_field.on_grid  # (time, x, y, 1)
    stabilized_pressure = pressure_on_grid[stabilization_index:, :, :, 0]

    disk_mask_1 = create_disk_mask(N, target_focal_center_1, focal_disk_radius)
    disk_mask_2 = create_disk_mask(N, target_focal_center_2, focal_disk_radius)

    rms_1 = _rms_over_disk(stabilized_pressure, disk_mask_1)
    rms_2 = _rms_over_disk(stabilized_pressure, disk_mask_2)

    combined_rms = 0.5 * rms_1 + 0.5 * rms_2
    loss = -combined_rms  # minimize -combined_RMS = maximize combined RMS
    return loss, pressure_on_grid

# ============================================================================
# SIREN init and Optimization Loop
# ============================================================================

key = jax.random.PRNGKey(siren_seed)
initial_params = siren_init(key, siren_layer_sizes, siren_omega0)

print("\n" + "="*70)
print("OPTIMIZATION: SIREN MATERIAL FIELD")
print("="*70)

learning_rate = float(os.environ.get('AUTODIFF_LR', '0.0001'))
n_iterations = int(os.environ.get('AUTODIFF_N_ITER', '100'))
checkpoint_path = os.path.join(results_dir, 'checkpoint_siren_subwindow_multiobjective.npz')

def params_to_arrays(params):
    """Flatten params to list of arrays for saving."""
    return [np.array(p) for layer in params for p in layer]

def arrays_to_params(arrs):
    """Restore params pytree from list of arrays (same order as params_to_arrays)."""
    params = []
    idx = 0
    for i in range(len(siren_layer_sizes) - 1):
        fan_in, fan_out = siren_layer_sizes[i], siren_layer_sizes[i + 1]
        W = jnp.array(arrs[idx]); idx += 1
        b = jnp.array(arrs[idx]); idx += 1
        params.append((W, b))
    return params

print(f"\nOptimization parameters:")
print(f"  Iterations: {n_iterations}")
print(f"  Learning rate: {learning_rate}")

params = initial_params
best_objective = float('inf')
best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), initial_params)
best_iteration = 0

optimization_history = {
    'objectives': [],
    'pressures': [],
    'gradient_norms': [],
    'iteration_times': []
}

pressure_on_grid_init = None   # from first iteration
pressure_on_grid_opt = None    # from last iteration

start_iteration = 0

print("\nStarting optimization (gradient descent on SIREN parameters)...")
print("-"*70)

for iteration in range(start_iteration, n_iterations):
    iter_start = pytime.time()

    print(f"\nIteration {iteration + 1}/{n_iterations}")

    (obj_val, pressure_on_grid), grad_params = value_and_grad(compute_objective_mlp, has_aux=True)(params)
    obj_float = float(obj_val)
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for (W, b) in grad_params for g in [W, b]))

    print(f"  Loss (-RMS): {obj_val:.6f}")
    print(f"  RMS pressure (focal disk): {-obj_val:.6f} Pa")
    print(f"  Gradient norm: {float(grad_norm):.6e}")

    if obj_float < best_objective:
        best_objective = obj_float
        best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
        best_iteration = iteration + 1
        print(f"  *** New best! (iter {best_iteration}) ***")

    if iteration == 0:
        pressure_on_grid_init = np.array(pressure_on_grid[..., 0])
    if iteration == n_iterations - 1:
        pressure_on_grid_opt = np.array(pressure_on_grid[..., 0])

    prev_objective = obj_float

    # Gradient descent: params = params - lr * grad_params
    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grad_params
    )

    optimization_history['objectives'].append(obj_float)
    optimization_history['pressures'].append(float(-obj_val))
    optimization_history['gradient_norms'].append(float(grad_norm))

    iter_end = pytime.time()
    iter_time = iter_end - iter_start
    optimization_history['iteration_times'].append(iter_time)
    print(f"  Iteration time: {iter_time:.2f} seconds")


final_params = best_params
print(f"\nUsing best state from iteration {best_iteration} (RMS {-best_objective:.6f} Pa)")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)
first_rms = optimization_history['pressures'][0] if optimization_history['pressures'] else 0.0
print(f"\nFirst iteration RMS: {first_rms:.6f} Pa")
print(f"Best RMS (iter {best_iteration}): {-best_objective:.6f} Pa")
improvement = (-best_objective / max(first_rms, 1e-10) - 1) * 100
print(f"Improvement: {improvement:.1f}%")

# ============================================================================
# Metrics from training runs (first & last iteration — no extra simulations)
# ============================================================================

print("\n" + "="*70)
print("METRICS (from first & last epoch pressure fields)")
print("="*70)

disk_mask_1_np = np.array(create_disk_mask(N, target_focal_center_1, focal_disk_radius))
disk_mask_2_np = np.array(create_disk_mask(N, target_focal_center_2, focal_disk_radius))
n_time_steps_val = len(time_array) - stabilization_index

def _rms_np(pressure_3d, mask, n_time):
    return np.sqrt(np.sum((pressure_3d * mask)**2) / (n_time * np.sum(mask)))

stabilized_init = pressure_on_grid_init[stabilization_index:, :, :]
stabilized_opt = pressure_on_grid_opt[stabilization_index:, :, :]
rms1_init = _rms_np(stabilized_init, disk_mask_1_np, n_time_steps_val)
rms2_init = _rms_np(stabilized_init, disk_mask_2_np, n_time_steps_val)
rms1_opt = _rms_np(stabilized_opt, disk_mask_1_np, n_time_steps_val)
rms2_opt = _rms_np(stabilized_opt, disk_mask_2_np, n_time_steps_val)
initial_max_pressure_full = 0.5 * rms1_init + 0.5 * rms2_init
optimized_max_pressure_full = 0.5 * rms1_opt + 0.5 * rms2_opt
improvement_full = (optimized_max_pressure_full / max(initial_max_pressure_full, 1e-10) - 1) * 100

print(f"Initial (iter 1): RMS1={rms1_init:.6f}, RMS2={rms2_init:.6f}, Combined={initial_max_pressure_full:.6f} Pa")
print(f"Optimized (iter {n_iterations}): RMS1={rms1_opt:.6f}, RMS2={rms2_opt:.6f}, Combined={optimized_max_pressure_full:.6f} Pa")
print(f"Improvement: {improvement_full:.1f}%")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
extent = [view_x_start * dx[0] * 1e3, view_x_end * dx[0] * 1e3,
          view_y_start * dx[1] * 1e3, view_y_end * dx[1] * 1e3]

# Plot 1: Optimization progress
ax = axes[0, 0]
ax.plot(optimization_history['pressures'], 'b-o', linewidth=2, markersize=6)
ax.axhline(initial_max_pressure_full, color='r', linestyle='--', alpha=0.5, label='Initial RMS (iter 1)')
ax.axhline(optimized_max_pressure_full, color='g', linestyle='--', alpha=0.5, label='Optimized RMS (last iter)')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Combined RMS (0.5×disk1 + 0.5×disk2) Pa', fontsize=11)
ax.set_title('SIREN Optimization Progress (Multi-Objective)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Gradient norm over iterations
ax = axes[0, 1]
ax.plot(optimization_history['gradient_norms'], 'b-o', linewidth=2, markersize=6)
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Gradient Norm', fontsize=11)
ax.set_title('SIREN Gradient Magnitude', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 3: Learned material field (initial, random SIREN)
ax = axes[0, 2]
alpha_init = np.array(material_field_from_mlp(initial_params)[0])
alpha_init = (alpha_init - c_water) / (c_cylinder - c_water)  # back to [0,1]
alpha_init = np.clip(alpha_init, 0, 1)
alpha_view_init = alpha_init[view_x_start:view_x_end, view_y_start:view_y_end]
im3 = ax.imshow(alpha_view_init.T, cmap='viridis', origin='lower', extent=extent, vmin=0, vmax=1)
ax.set_xlabel('x (mm)', fontsize=11)
ax.set_ylabel('y (mm)', fontsize=11)
ax.set_title('Initial Material Field (α)\n0=water, 1=solid', fontsize=12, fontweight='bold')
ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
# Subwindow: only region where material can be placed
sub_rect = Rectangle((subwindow_x_start*dx[0]*1e3, subwindow_y_start*dx[1]*1e3),
                     (subwindow_x_end - subwindow_x_start)*dx[0]*1e3,
                     (subwindow_y_end - subwindow_y_start)*dx[1]*1e3,
                     fill=False, edgecolor='white', linewidth=2, linestyle='--', label='Subwindow')
ax.add_patch(sub_rect)
plt.colorbar(im3, ax=ax, label='α', fraction=0.046)

# Plot 4: Learned material field (optimized SIREN)
ax = axes[0, 3]
alpha_opt = np.array(material_field_from_mlp(final_params)[0])
alpha_opt = (alpha_opt - c_water) / (c_cylinder - c_water)
alpha_opt = np.clip(alpha_opt, 0, 1)
alpha_view_opt = alpha_opt[view_x_start:view_x_end, view_y_start:view_y_end]
im4 = ax.imshow(alpha_view_opt.T, cmap='viridis', origin='lower', extent=extent, vmin=0, vmax=1)
ax.set_xlabel('x (mm)', fontsize=11)
ax.set_ylabel('y (mm)', fontsize=11)
ax.set_title('Optimized Material Field (α)', fontsize=12, fontweight='bold')
ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
sub_rect4 = Rectangle((subwindow_x_start*dx[0]*1e3, subwindow_y_start*dx[1]*1e3),
                       (subwindow_x_end - subwindow_x_start)*dx[0]*1e3,
                       (subwindow_y_end - subwindow_y_start)*dx[1]*1e3,
                       fill=False, edgecolor='white', linewidth=2, linestyle='--')
ax.add_patch(sub_rect4)
plt.colorbar(im4, ax=ax, label='α', fraction=0.046)

# Plot 5: Initial pressure field
ax = axes[1, 0]
final_pressure_init = np.mean(np.abs(pressure_on_grid_init[stabilization_index:, :, :]), axis=0)
pressure_view_init = final_pressure_init[view_x_start:view_x_end, view_y_start:view_y_end]
vmax_viz = max(np.max(pressure_view_init),
               np.max(np.mean(np.abs(pressure_on_grid_opt[stabilization_index:, view_x_start:view_x_end, view_y_start:view_y_end]), axis=0)))
im = ax.imshow(pressure_view_init.T, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=vmax_viz)
ax.set_xlabel('x (mm)', fontsize=11)
ax.set_ylabel('y (mm)', fontsize=11)
ax.set_title(f'Initial Pressure |P|\nRMS: {initial_max_pressure_full:.4f} Pa', fontsize=12, fontweight='bold')
ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
focal_disk_mm = focal_disk_radius * dx[0] * 1e3
ax.add_patch(MplCircle((target_focal_center_1[0]*dx[0]*1e3, target_focal_center_1[1]*dx[1]*1e3),
             focal_disk_mm, fill=False, edgecolor='lime', linewidth=2, zorder=10, label='Disk 1'))
ax.add_patch(MplCircle((target_focal_center_2[0]*dx[0]*1e3, target_focal_center_2[1]*dx[1]*1e3),
             focal_disk_mm, fill=False, edgecolor='cyan', linewidth=2, zorder=10, label='Disk 2'))
plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)

# Plot 6: Optimized pressure field
ax = axes[1, 1]
final_pressure_opt = np.mean(np.abs(pressure_on_grid_opt[stabilization_index:, :, :]), axis=0)
pressure_view_opt = final_pressure_opt[view_x_start:view_x_end, view_y_start:view_y_end]
im = ax.imshow(pressure_view_opt.T, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=vmax_viz)
ax.set_xlabel('x (mm)', fontsize=11)
ax.set_ylabel('y (mm)', fontsize=11)
ax.set_title(f'Optimized Pressure |P|\nCombined RMS: {optimized_max_pressure_full:.4f} Pa (+{improvement_full:.1f}%)', fontsize=12, fontweight='bold')
ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
ax.add_patch(MplCircle((target_focal_center_1[0]*dx[0]*1e3, target_focal_center_1[1]*dx[1]*1e3),
             focal_disk_mm, fill=False, edgecolor='lime', linewidth=2, zorder=10))
ax.add_patch(MplCircle((target_focal_center_2[0]*dx[0]*1e3, target_focal_center_2[1]*dx[1]*1e3),
             focal_disk_mm, fill=False, edgecolor='cyan', linewidth=2, zorder=10))
plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)

# Plot 7: Empty or iteration time
ax = axes[1, 2]
ax.plot(optimization_history['iteration_times'], 'g-o', linewidth=2, markersize=6)
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Time (s)', fontsize=11)
ax.set_title('Iteration Time', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 8: Summary
ax = axes[1, 3]
ax.axis('off')
summary_text = f"""
SIREN SUBWINDOW + MULTI-OBJECTIVE SUMMARY

Pipeline:
  • SIREN(x,y) → α ∈ [0,1] (continuous)
  • α non-zero only inside subwindow; rest = water
  • Loss = -(0.5×RMS(disk1) + 0.5×RMS(disk2))

SIREN: {siren_layer_sizes}, ω₀={siren_omega0}
Disk 1: {target_focal_center_1}, Disk 2: {target_focal_center_2}
Subwindow: x=[{subwindow_x_start},{subwindow_x_end}], y=[{subwindow_y_start},{subwindow_y_end}]

Optimization:
  • Iterations: {n_iterations}
  • Learning rate: {learning_rate}
  • Avg time/iter: {np.mean(optimization_history['iteration_times']):.1f} s

Results (from iter 1 & last, 0.5×RMS1 + 0.5×RMS2):
  • Initial: {initial_max_pressure_full:.6f} Pa (RMS1={rms1_init:.4f}, RMS2={rms2_init:.4f})
  • Optimized: {optimized_max_pressure_full:.6f} Pa (RMS1={rms1_opt:.4f}, RMS2={rms2_opt:.4f})
  • Improvement: {improvement_full:.1f}%
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('SIREN Subwindow + Multi-Objective (2 Focal Circles) — GINN + J-Wave',
            fontsize=14, fontweight='bold')
plt.tight_layout()

output_fig = os.path.join(results_dir, 'autodiff_siren_subwindow_multiobjective_results.png')
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"Saved results to: {output_fig}")

print("\n" + "="*70)
print("SIREN MATERIAL FIELD OPTIMIZATION COMPLETE!")
print("="*70)
print("\nLearned spatial material distribution via differentiable SIREN → jwave → RMS.")
print("="*70)
print("\nDone!")