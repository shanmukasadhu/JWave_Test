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

# Results directory: use subdir when AUTODIFF_RUN_ID is set (e.g. by overnight grid script)
_run_id = os.environ.get('AUTODIFF_RUN_ID', '')
results_dir = os.path.join('../Results/autodiff_optimization', _run_id) if _run_id else '../Results/autodiff_optimization'
os.makedirs('../Results/autodiff_optimization', exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("="*70)
print("MLP MATERIAL FIELD OPTIMIZATION (GINN + J-WAVE)")
print("="*70)
print("\nLearning a spatial material distribution via an MLP that maps")
print("(x,y) -> material in [0,1]. Gradients flow: loss -> RMS -> pressure")
print("-> wave solver -> material field -> MLP parameters.")
print("="*70)

# Simulation parameters
N = (512, 512)
dx = (0.2e-3, 0.2e-3)
domain = Domain(N, dx)

# Viewing window
view_x_start = 128
view_x_end = 384
view_y_start = 128
view_y_end = 384

# Medium properties
c_water = 1500.0
c_cylinder = 2500.0
rho_water = 1000.0
rho_cylinder = 1200.0

# Wave parameters
frequency = 1.0e6
wavelength = c_water / frequency

# Design/Focus space boundaries
design_boundary_x = 280
focus_boundary_x = 300

# Target focal region - RMS over a disk
target_focal_center = (320, 256)
focal_disk_radius = 10  # grid points

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

# MLP architecture: (x,y) -> [0,1] material. Layers: 2 -> 64 -> 64 -> 32 -> 1
mlp_layer_sizes = [2, 64, 64, 32, 1]
mlp_seed = 42

print(f"\nSimulation Configuration:")
print(f"  Simulation time: {training_time_end*1e6:.1f} μs")
print(f"  Stabilization: {stabilization_time*1e6:.1f} μs")
print(f"  Target focal disk: center {target_focal_center}, radius {focal_disk_radius} pts")
print(f"  Grid: {N[0]} × {N[1]}")
print(f"  MLP: {mlp_layer_sizes} (output = material in [0,1])")
print(f"  Objective: maximize RMS pressure over focal disk")
print("="*70)

# ============================================================================
# JAX MLP: (x, y) -> material in [0, 1]
# ============================================================================
#Random initalization of the MLP, only ran once at the beginning of the script. 
def mlp_init(key, layer_sizes, scale=0.1):
    """Initialize MLP params: list of (W, b) per layer. Output layer uses sigmoid."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        W = jax.random.normal(k1, (fan_in, fan_out)) * scale
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params

def mlp_apply(params, xy):
    """Apply MLP to a single (x, y) -> scalar in [0, 1]. Differentiable."""
    h = xy
    for (W, b) in params[:-1]:
        h = jax.nn.relu(h @ W + b)
    Wlast, blast = params[-1]
    h = h @ Wlast + blast
    return jax.nn.sigmoid(jnp.squeeze(h))

# Batched application over grid points
mlp_apply_batch = vmap(mlp_apply, in_axes=(None, 0))

def material_field_from_mlp(params):
    """
    Sample MLP on the grid to get material alpha in [0,1].
    Restrict material to design region (x < design_boundary_x).
    Returns (sound_speed, density) arrays of shape N.
    """
    # Normalized coords in [0, 1] so MLP is grid-size independent
    xs = jnp.linspace(0.0, 1.0, N[0])
    ys = jnp.linspace(0.0, 1.0, N[1])
    X, Y = jnp.meshgrid(xs, ys, indexing='ij')
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (N[0]*N[1], 2)

    alpha = mlp_apply_batch(params, grid_points)  # (N[0]*N[1],)
    alpha = alpha.reshape(N[0], N[1])

    # Design region only: zero material in focus/buffer (x >= design_boundary_x in grid index)
    x_norm_design = design_boundary_x / (N[0] - 1.0)
    X_norm = jnp.linspace(0.0, 1.0, N[0])
    design_mask = (X_norm < x_norm_design)[:, jnp.newaxis]  # (N[0], 1) -> broadcast to (N[0], N[1])
    alpha = alpha * design_mask

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

def compute_objective_mlp(params):
    """
    Objective: maximize RMS pressure over focal disk.
    Loss = -RMS so that minimizing loss maximizes RMS. Fully differentiable.
    """
    sound_speed, density = material_field_from_mlp(params)
    medium = create_medium_from_material_field(domain, sound_speed, density)
    pressure_field = run_simulation_from_medium(medium)

    pressure_on_grid = pressure_field.on_grid  # (time, x, y, 1)
    disk_mask = create_disk_mask(N, target_focal_center, focal_disk_radius)
    stabilized_pressure = pressure_on_grid[stabilization_index:, :, :, 0]
    disk_mask_expanded = jnp.expand_dims(disk_mask, axis=0)
    pressure_in_disk = stabilized_pressure * disk_mask_expanded

    n_points_in_disk = jnp.sum(disk_mask)
    total_sum_squares = jnp.sum(pressure_in_disk ** 2)
    n_time_steps = stabilized_pressure.shape[0]
    n_total_samples = n_time_steps * n_points_in_disk
    mean_square = total_sum_squares / jnp.maximum(n_total_samples, 1.0)
    rms_pressure = jnp.sqrt(mean_square)

    return -rms_pressure  # minimize -RMS = maximize RMS

# ============================================================================
# MLP init and Optimization Loop
# ============================================================================

key = jax.random.PRNGKey(mlp_seed)
initial_params = mlp_init(key, mlp_layer_sizes, scale=0.08)

print("\n" + "="*70)
print("OPTIMIZATION: MLP MATERIAL FIELD")
print("="*70)

learning_rate = float(os.environ.get('AUTODIFF_LR', '200.0'))
n_iterations = int(os.environ.get('AUTODIFF_N_ITER', '30'))
checkpoint_path = os.path.join(results_dir, 'checkpoint_mlp.npz')

def params_to_arrays(params):
    """Flatten params to list of arrays for saving."""
    return [np.array(p) for layer in params for p in layer]

def arrays_to_params(arrs):
    """Restore params pytree from list of arrays (same order as params_to_arrays)."""
    params = []
    idx = 0
    for i in range(len(mlp_layer_sizes) - 1):
        fan_in, fan_out = mlp_layer_sizes[i], mlp_layer_sizes[i + 1]
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

import sys
resume = '--resume' in sys.argv
start_iteration = 0
prev_objective = None
if resume and os.path.exists(checkpoint_path):
    try:
        ck = np.load(checkpoint_path, allow_pickle=True)
        arrs = list(ck['params_arrays'])
        params = arrays_to_params(arrs)
        best_params = arrays_to_params(list(ck['best_params_arrays']))
        best_objective = float(ck['best_objective'])
        best_iteration = int(ck['best_iteration'])
        start_iteration = int(ck['iteration']) + 1
        optimization_history = {
            'objectives': list(ck['history_objectives']),
            'pressures': list(ck['history_pressures']),
            'gradient_norms': list(ck['history_gradient_norms']),
            'iteration_times': list(ck['iteration_times']),
        }
        prev_objective = optimization_history['objectives'][-1]
        print(f"\nResuming from iteration {start_iteration} (best: iter {best_iteration})")
    except Exception as e:
        print(f"\nCould not load checkpoint: {e}. Starting from scratch.")

print("\nStarting optimization (gradient descent on MLP parameters)...")
print("-"*70)

for iteration in range(start_iteration, n_iterations):
    iter_start = pytime.time()

    print(f"\nIteration {iteration + 1}/{n_iterations}")

    obj_val, grad_params = value_and_grad(compute_objective_mlp)(params)
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
# Validation with Full Simulation
# ============================================================================

print("\n" + "="*70)
print("VALIDATION WITH FULL 300 μs SIMULATION")
print("="*70)

time_axis_full = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3,
    t_end=3.0e-04
)
time_array_full = time_axis_full.to_array()
stabilization_time_full = 2.0e-04
stabilization_index_full = int(jnp.argmin(jnp.abs(time_array_full - stabilization_time_full)))
disk_mask_np = np.array(create_disk_mask(N, target_focal_center, focal_disk_radius))
n_disk_points = np.sum(disk_mask_np)
n_time_steps_full = len(time_array_full) - stabilization_index_full

print(f"\nFull simulation: {time_axis_full.t_end*1e6:.0f} μs, stabilization at {stabilization_time_full*1e6:.0f} μs")

# Initial (random MLP) medium and run
print(f"\nRunning full simulation for initial (random) MLP...")
val_start = pytime.time()
sound_speed_init, density_init = material_field_from_mlp(initial_params)
medium_init = create_medium_from_material_field(domain, sound_speed_init, density_init)
pressure_field_init = run_simulation_from_medium(medium_init, time_axis_full)
pressure_on_grid_init = np.array(pressure_field_init.on_grid)[:, :, :, 0]
stabilized_init = pressure_on_grid_init[stabilization_index_full:, :, :]
rms_init = np.sqrt(np.sum((stabilized_init * disk_mask_np)**2) / (n_time_steps_full * n_disk_points))
initial_max_pressure_full = rms_init
print(f"  Time: {pytime.time() - val_start:.2f} s, RMS: {initial_max_pressure_full:.6f} Pa")

# Optimized MLP medium and run
print(f"\nRunning full simulation for optimized MLP...")
val_start = pytime.time()
sound_speed_opt, density_opt = material_field_from_mlp(final_params)
medium_opt = create_medium_from_material_field(domain, sound_speed_opt, density_opt)
pressure_field_opt = run_simulation_from_medium(medium_opt, time_axis_full)
pressure_on_grid_opt = np.array(pressure_field_opt.on_grid)[:, :, :, 0]
stabilized_opt = pressure_on_grid_opt[stabilization_index_full:, :, :]
rms_opt = np.sqrt(np.sum((stabilized_opt * disk_mask_np)**2) / (n_time_steps_full * n_disk_points))
optimized_max_pressure_full = rms_opt
print(f"  Time: {pytime.time() - val_start:.2f} s, RMS: {optimized_max_pressure_full:.6f} Pa")

improvement_full = (optimized_max_pressure_full / max(initial_max_pressure_full, 1e-10) - 1) * 100
print(f"\nFull simulation RMS over focal disk: Initial {initial_max_pressure_full:.6f} Pa, Optimized {optimized_max_pressure_full:.6f} Pa, Improvement {improvement_full:.1f}%")

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
ax.axhline(initial_max_pressure_full, color='r', linestyle='--', alpha=0.5, label='Initial RMS (full sim)')
ax.axhline(optimized_max_pressure_full, color='g', linestyle='--', alpha=0.5, label='Optimized RMS (full sim)')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('RMS Pressure over Focal Disk (Pa)', fontsize=11)
ax.set_title('MLP Optimization Progress (RMS over Focal Disk)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Gradient norm over iterations
ax = axes[0, 1]
ax.plot(optimization_history['gradient_norms'], 'b-o', linewidth=2, markersize=6)
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Gradient Norm', fontsize=11)
ax.set_title('MLP Gradient Magnitude', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 3: Learned material field (initial, random MLP)
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
plt.colorbar(im3, ax=ax, label='α', fraction=0.046)

# Plot 4: Learned material field (optimized MLP)
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
plt.colorbar(im4, ax=ax, label='α', fraction=0.046)

# Plot 5: Initial pressure field
ax = axes[1, 0]
final_pressure_init = np.mean(np.abs(pressure_on_grid_init[stabilization_index_full:, :, :]), axis=0)
pressure_view_init = final_pressure_init[view_x_start:view_x_end, view_y_start:view_y_end]
vmax_viz = max(np.max(pressure_view_init),
               np.max(np.mean(np.abs(pressure_on_grid_opt[stabilization_index_full:, view_x_start:view_x_end, view_y_start:view_y_end]), axis=0)))
im = ax.imshow(pressure_view_init.T, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=vmax_viz)
ax.set_xlabel('x (mm)', fontsize=11)
ax.set_ylabel('y (mm)', fontsize=11)
ax.set_title(f'Initial Pressure |P|\nRMS: {initial_max_pressure_full:.4f} Pa', fontsize=12, fontweight='bold')
ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
focal_disk_mm = focal_disk_radius * dx[0] * 1e3
focal_circle_patch = MplCircle((target_focal_center[0]*dx[0]*1e3, target_focal_center[1]*dx[1]*1e3),
                             focal_disk_mm, fill=False, edgecolor='lime', linewidth=2, zorder=10)
ax.add_patch(focal_circle_patch)
plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)

# Plot 6: Optimized pressure field
ax = axes[1, 1]
final_pressure_opt = np.mean(np.abs(pressure_on_grid_opt[stabilization_index_full:, :, :]), axis=0)
pressure_view_opt = final_pressure_opt[view_x_start:view_x_end, view_y_start:view_y_end]
im = ax.imshow(pressure_view_opt.T, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=vmax_viz)
ax.set_xlabel('x (mm)', fontsize=11)
ax.set_ylabel('y (mm)', fontsize=11)
ax.set_title(f'Optimized Pressure |P|\nRMS: {optimized_max_pressure_full:.4f} Pa (+{improvement_full:.1f}%)', fontsize=12, fontweight='bold')
ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
ax.add_patch(MplCircle((target_focal_center[0]*dx[0]*1e3, target_focal_center[1]*dx[1]*1e3),
                        focal_disk_mm, fill=False, edgecolor='lime', linewidth=2, zorder=10))
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
MLP MATERIAL FIELD (GINN) SUMMARY

Pipeline:
  • MLP(x,y) → α ∈ [0,1] (material)
  • α → sound_speed, density → jwave → pressure
  • Loss = -RMS(pressure in focal disk)
  • ∂Loss/∂(MLP params) via JAX autodiff

MLP: {mlp_layer_sizes}
Design: α non-zero only for x < design_boundary_x

Optimization:
  • Iterations: {n_iterations}
  • Learning rate: {learning_rate}
  • Avg time/iter: {np.mean(optimization_history['iteration_times']):.1f} s

Results (300 μs validation):
  • Initial RMS: {initial_max_pressure_full:.6f} Pa
  • Optimized RMS: {optimized_max_pressure_full:.6f} Pa
  • Improvement: {improvement_full:.1f}%
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('MLP Material Field Optimization (GINN + J-Wave)',
            fontsize=14, fontweight='bold')
plt.tight_layout()

output_fig = os.path.join(results_dir, 'autodiff_mlp_results.png')
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"Saved results to: {output_fig}")

print("\n" + "="*70)
print("MLP MATERIAL FIELD OPTIMIZATION COMPLETE!")
print("="*70)
print("\nLearned spatial material distribution via differentiable MLP → jwave → RMS.")
print("="*70)
print("\nDone!")