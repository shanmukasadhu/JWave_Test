"""
True Autodiff Optimization using J-Wave's Differentiability

This approach uses JAX's automatic differentiation directly through the j-wave
simulation to compute physics-informed gradients for optimization.

Key idea:
- The entire j-wave simulation is JAX-compatible and differentiable
- We compute ∂(max_pressure)/∂(cylinder_positions) using jax.grad()
- Gradients flow through the wave equation solver itself
- This gives us TRUE physics-informed gradients!

This is slower than surrogate models but gives exact physics gradients.
"""

import numpy as np
from jax import numpy as jnp
from jax import grad, jit, lax, value_and_grad
import jax
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
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
print("TRUE AUTODIFF OPTIMIZATION USING J-WAVE DIFFERENTIABILITY")
print("="*70)
print("\nThis approach computes gradients by differentiating through")
print("the entire wave equation solver using JAX's autodiff.")
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

# Target focal region - RMS over a disk instead of single point
target_focal_center = (320, 256)  # Center of disk in focus space
focal_disk_radius = 10  # Radius in grid points (10 points ≈ 2mm with 0.2mm spacing)

# Use shorter simulation for faster optimization (validate with full 300 μs later)
training_time_end = 1.5e-04  # 150 μs (faster optimization)
stabilization_time = 1.0e-04  # 100 μs

# Create time axis
time_axis = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3,
    t_end=training_time_end
)

time_array = time_axis.to_array()
stabilization_index = int(jnp.argmin(jnp.abs(time_array - stabilization_time)))

n_cylinders = 3
initial_cylinder_radius = 12.0   # Starting radius for all cylinders (grid points)
radius_min = 5.0                  # Minimum allowed radius (grid points)
radius_max = 25.0                 # Maximum allowed radius (grid points)

print(f"\nSimulation Configuration:")
print(f"  Simulation time: {training_time_end*1e6:.1f} μs")
print(f"  Stabilization time: {stabilization_time*1e6:.1f} μs")
print(f"  Stabilization index: {stabilization_index}/{len(time_array)}")
print(f"  Target focal region:")
print(f"    Center: {target_focal_center}")
print(f"    Disk radius: {focal_disk_radius} grid points ({focal_disk_radius*dx[0]*1e3:.2f} mm)")
print(f"    Physical center: ({target_focal_center[0]*dx[0]*1e3:.2f}, {target_focal_center[1]*dx[1]*1e3:.2f}) mm")
print(f"  Initial cylinder radius: {initial_cylinder_radius*dx[0]*1e3:.2f} mm")
print(f"  Radius bounds: [{radius_min*dx[0]*1e3:.2f}, {radius_max*dx[0]*1e3:.2f}] mm")
print(f"  Grid size: {N[0]} × {N[1]}")
print(f"  Objective: RMS pressure over focal disk (not single point)")
print("="*70)

# ============================================================================
# Differentiable J-Wave Simulation
# ============================================================================

def create_cylinder_mask(N, center, radius):
    """Create a circular mask for a cylinder - fully differentiable"""
    x = jnp.arange(N[0])
    y = jnp.arange(N[1])
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Use smooth approximation instead of hard threshold for better gradients
    dist_sq = (X - center[0])**2 + (Y - center[1])**2

    # Smooth step function: sigmoid with adjustable steepness
    # This makes the gradient continuous
    steepness = 2.0  # Higher = sharper boundary
    mask = jax.nn.sigmoid(steepness * (radius**2 - dist_sq))

    return mask

def create_medium_with_cylinders(domain, cylinder_positions, cylinder_radii):
    """
    Create medium with cylinders - fully differentiable

    The cylinder positions AND radii are the optimization variables, so this
    function must be differentiable w.r.t. both.
    """
    sound_speed = jnp.ones(N) * c_water
    density = jnp.ones(N) * rho_water

    # Add cylinders using smooth masks for differentiability
    for i in range(cylinder_positions.shape[0]):
        center = (cylinder_positions[i, 0], cylinder_positions[i, 1])
        radius = cylinder_radii[i]
        mask = create_cylinder_mask(N, center, radius)

        # Smooth interpolation between water and cylinder properties
        sound_speed = sound_speed * (1 - mask) + c_cylinder * mask
        density = density * (1 - mask) + rho_cylinder * mask
    # expand the dimensions to match the shape of the medium
    sound_speed = jnp.expand_dims(sound_speed, -1)
    density = jnp.expand_dims(density, -1)
    # return the medium with the sound speed and density
    return Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=60)

def run_simulation(cylinder_positions, cylinder_radii):
    """
    Run j-wave simulation - FULLY DIFFERENTIABLE

    This function takes cylinder positions and returns the pressure field.
    JAX can compute gradients of outputs w.r.t. cylinder_positions.
    """
    # Create medium (differentiable w.r.t. both positions and radii)
    medium = create_medium_with_cylinders(domain, cylinder_positions, cylinder_radii)

    # Create initial condition (zero pressure)
    p0_array = jnp.zeros(N)
    p0_array = jnp.expand_dims(p0_array, -1)
    p0 = FourierSeries(p0_array, domain)

    # Create time-varying source
    class TimeVaryingSource:
        def __init__(self, time_axis, domain):
            self.domain = domain
            self.omega = 2 * jnp.pi * frequency
            self.time_array = time_axis.to_array()

            # Create source mask (left edge region)
            source_mask = jnp.zeros(N)
            source_mask = source_mask.at[:40, :].set(1.0)
            source_mask = jnp.expand_dims(source_mask, -1)

            # Precompute all source fields
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

    time_varying_source = TimeVaryingSource(time_axis, domain)

    # Run simulation - this is the key differentiable step!
    pressure_field = simulate_wave_propagation(
        medium,
        time_axis,
        p0=p0,
        sources=time_varying_source
    )

    return pressure_field

def create_disk_mask(N, center, radius):
    """Create a circular disk mask in the x-y plane"""
    x = jnp.arange(N[0])
    y = jnp.arange(N[1])
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    dist = jnp.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist <= radius

def compute_cylinder_overlap_penalty(cylinder_positions, cylinder_radii):
    """
    Compute penalty for overlapping cylinders.

    Each cylinder pair uses their individual radii to determine the minimum
    safe separation distance (sum of both radii + buffer).
    """
    penalty = 0.0
    n_cyl = cylinder_positions.shape[0]

    for i in range(n_cyl):
        for j in range(i + 1, n_cyl):
            # Minimum separation: sum of both radii × 1.25 buffer
            min_separation = 1.25 * (cylinder_radii[i] + cylinder_radii[j])

            dx_ij = cylinder_positions[i, 0] - cylinder_positions[j, 0]
            dy_ij = cylinder_positions[i, 1] - cylinder_positions[j, 1]
            dist = jnp.sqrt(dx_ij**2 + dy_ij**2)

            overlap_amount = jnp.maximum(0.0, min_separation - dist)
            penalty += overlap_amount ** 2

    return penalty

def compute_objective(params):
    import time; time_start = time.time()
    """
    Objective function: RMS pressure over focal disk during stabilized period
    WITH non-overlap constraint.

    params is a tuple (cylinder_positions, cylinder_radii) so that JAX can
    differentiate w.r.t. both simultaneously.

    Loss = -RMS_pressure + penalty_weight * overlap_penalty
    We MINIMISE this (maximises RMS pressure while penalising overlaps).
    """
    cylinder_positions, cylinder_radii = params

    # Run simulation (differentiable w.r.t. both positions and radii)
    pressure_field = run_simulation(cylinder_positions, cylinder_radii)

    # Extract pressure on grid
    pressure_on_grid = pressure_field.on_grid  # Shape: (time, x, y, 1)

    # Create disk mask for focal region
    disk_mask = create_disk_mask(N, target_focal_center, focal_disk_radius)

    # Extract stabilized portion (time, x, y)
    stabilized_pressure = pressure_on_grid[stabilization_index:, :, :, 0]

    # Expand mask to match time dimension: (1, x, y) -> broadcast to (time, x, y)
    disk_mask_expanded = jnp.expand_dims(disk_mask, axis=0)

    # Extract pressure in disk region
    pressure_in_disk = stabilized_pressure * disk_mask_expanded

    # Compute RMS: sqrt(mean(p²)) over all points in disk and all stabilized time steps
    n_points_in_disk = jnp.sum(disk_mask)
    pressure_squared = pressure_in_disk ** 2
    total_sum_squares = jnp.sum(pressure_squared)
    n_time_steps = stabilized_pressure.shape[0]
    n_total_samples = n_time_steps * n_points_in_disk
    mean_square = total_sum_squares / n_total_samples
    rms_pressure = jnp.sqrt(mean_square)

    # Overlap penalty (uses per-cylinder radii)
    overlap_penalty = compute_cylinder_overlap_penalty(cylinder_positions, cylinder_radii)

    penalty_weight = 0.1
    loss = -rms_pressure + penalty_weight * overlap_penalty
    print(f"Compute Objective Time taken: {time.time() - time_start:.2f} seconds")
    return loss

# ============================================================================
# Test Differentiability
# ============================================================================

print("\n" + "="*70)
print("TESTING DIFFERENTIABILITY")
print("="*70)

# Test with initial configuration
initial_positions = jnp.array([
    [200.0, 200.0],
    [200.0, 256.0],
    [200.0, 312.0]
])

initial_radii = jnp.full((n_cylinders,), initial_cylinder_radius)
initial_params = (initial_positions, initial_radii)

print(f"\nTesting autodiff with initial configuration...")
print(f"Initial positions:\n{initial_positions}")
print(f"Initial radii: {initial_radii}")

# Test forward pass
print("\nRunning forward simulation...")
test_start = pytime.time()
try:
    test_objective = compute_objective(initial_params)
    test_end = pytime.time()
    print(f"✓ Forward pass successful!")
    print(f"  Objective value: {test_objective:.6f}")
    print(f"  RMS pressure over focal disk: {-test_objective:.6f} Pa")
    print(f"  Time: {test_end - test_start:.2f} seconds")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    raise

# Test gradient computation
print("\nComputing gradients via autodiff (positions AND radii)...")
grad_start = pytime.time()
try:
    # value_and_grad on a pytree (positions, radii)
    objective_value, (grad_positions, grad_radii) = value_and_grad(compute_objective)(initial_params)
    grad_end = pytime.time()

    print(f"✓ Gradient computation successful!")
    print(f"  Objective: {objective_value:.6f}")
    print(f"  Gradient shapes — positions: {grad_positions.shape}, radii: {grad_radii.shape}")
    print(f"  Position gradient norm: {jnp.linalg.norm(grad_positions):.6f}")
    print(f"  Radius gradient norm:   {jnp.linalg.norm(grad_radii):.6f}")
    print(f"  Time: {grad_end - grad_start:.2f} seconds")
    print(f"\nGradient values:")
    for i in range(grad_positions.shape[0]):
        print(f"  Cylinder {i+1}: ∂L/∂x = {grad_positions[i,0]:+.6f}, "
              f"∂L/∂y = {grad_positions[i,1]:+.6f}, ∂L/∂r = {grad_radii[i]:+.6f}")
except Exception as e:
    print(f"✗ Gradient computation failed: {e}")
    raise

print("\n" + "="*70)
print("DIFFERENTIABILITY TEST PASSED!")
print("="*70)

# ============================================================================
# Optimization Loop
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZATION USING TRUE AUTODIFF")
print("="*70)

# Optimization parameters (overridden by env AUTODIFF_LR, AUTODIFF_N_ITER when set)
learning_rate = float(os.environ['AUTODIFF_LR']) if os.environ.get('AUTODIFF_LR') else 175.0
n_iterations = int(os.environ['AUTODIFF_N_ITER']) if os.environ.get('AUTODIFF_N_ITER') else 40
momentum = 0.5  # Lower momentum to avoid overshooting (was 0.9)
lr_decay_on_worse = 1.0   # DISABLE LR decay (set to 1.0 = no decay)
lr_min = 0.05
checkpoint_every = 5
checkpoint_path = os.path.join(results_dir, 'checkpoint.npz')

print(f"\nOptimization parameters:")
print(f"  Iterations: {n_iterations}")
print(f"  Learning rate: {learning_rate} {'(FIXED - no decay)' if lr_decay_on_worse == 1.0 else f'(decay {lr_decay_on_worse}x when worse, min {lr_min})'}")
print(f"  Momentum: {momentum}")
print(f"  Checkpoint every {checkpoint_every} iters: {checkpoint_path}")
print(f"  Estimated time: ~{n_iterations * (grad_end - grad_start) / 60:.1f} minutes")
print(f"\nParameter tuning notes:")
print(f"  - High LR ({learning_rate}) to overcome small gradients")
print(f"  - Lower momentum ({momentum}) to avoid overshooting")
print(f"  - LR decay disabled for consistent updates")
print(f"  - Shorter sim time (150μs) for 2x+ speedup")
print(f"  - Non-overlap constraint: min separation = 2.5 × radius")

positions = initial_positions
radii = initial_radii
velocity_pos = jnp.zeros_like(positions)
velocity_rad = jnp.zeros_like(radii)
current_lr = learning_rate

# Best state (we minimize objective; lower is better)
best_objective = float(test_objective)
best_positions = positions
best_radii = radii
best_iteration = 0

optimization_history = {
    'positions': [positions],
    'radii': [radii],
    'objectives': [float(test_objective)],
    'pressures': [float(-test_objective)],
    'gradients_pos': [],
    'gradients_rad': [],
    'iteration_times': []
}

# Optional resume from checkpoint
import sys
resume = '--resume' in sys.argv
start_iteration = 0
prev_objective = None
if resume:
    if os.path.exists(checkpoint_path):
        try:
            ck = np.load(checkpoint_path, allow_pickle=True)
            positions = jnp.array(ck['positions'])
            radii = jnp.array(ck['radii'])
            velocity_pos = jnp.array(ck['velocity_pos'])
            velocity_rad = jnp.array(ck['velocity_rad'])
            current_lr = float(ck['current_lr'])
            best_objective = float(ck['best_objective'])
            best_positions = jnp.array(ck['best_positions'])
            best_radii = jnp.array(ck['best_radii'])
            best_iteration = int(ck['best_iteration'])
            start_iteration = int(ck['iteration']) + 1
            optimization_history = {
                'positions': list(ck['history_positions']),
                'radii': list(ck['history_radii']),
                'objectives': list(ck['history_objectives']),
                'pressures': list(ck['history_pressures']),
                'gradients_pos': list(ck['history_gradients_pos']),
                'gradients_rad': list(ck['history_gradients_rad']),
                'iteration_times': list(ck['iteration_times']),
            }
            prev_objective = float(optimization_history['objectives'][-1])
            print(f"\nResuming from iteration {start_iteration} (best so far: iter {best_iteration}, obj {best_objective:.6f})")
        except Exception as e:
            print(f"\nCould not load checkpoint: {e}. Starting from scratch.")
if prev_objective is None:
    prev_objective = float(test_objective)

print("\nStarting optimization...")
print("Note: Each iteration requires a full simulation + gradient computation")
print("Best state is tracked; final result uses best objective over all iterations.")
print("-"*70)

for iteration in range(start_iteration, n_iterations):
    iter_start = pytime.time()

    print(f"\nIteration {iteration + 1}/{n_iterations}")
    print(f"  Current positions and radii:")
    for i in range(positions.shape[0]):
        print(f"    Cylinder {i+1}: pos=({positions[i,0]:.2f}, {positions[i,1]:.2f}), r={radii[i]:.2f}")

    # Compute objective and gradients w.r.t. both positions and radii
    print(f"  Computing objective and gradients via autodiff...")
    params = (positions, radii)
    obj_val, (grads_pos, grads_rad) = value_and_grad(compute_objective)(params)
    obj_float = float(obj_val)

    # Compute overlap penalty for monitoring
    overlap_penalty = compute_cylinder_overlap_penalty(positions, radii)
    rms_only = -obj_val - 0.1 * overlap_penalty

    print(f"  Objective: {obj_val:.6f}")
    print(f"  RMS pressure (focal disk): {rms_only:.6f} Pa")
    print(f"  Overlap penalty: {overlap_penalty:.6f}")
    print(f"  Position gradient norm: {jnp.linalg.norm(grads_pos):.6f}")
    print(f"  Radius gradient norm:   {jnp.linalg.norm(grads_rad):.6f}")

    # Best state: lower objective is better
    if obj_float < best_objective:
        best_objective = obj_float
        best_positions = positions
        best_radii = radii
        best_iteration = iteration + 1
        print(f"  *** New best! (iter {best_iteration}) ***")

    # Learning rate decay when objective worsens
    if obj_float > prev_objective and iteration > start_iteration:
        old_lr = current_lr
        current_lr = max(lr_min, current_lr * lr_decay_on_worse)
        print(f"  Objective worsened → reducing LR: {old_lr:.4f} → {current_lr:.4f}")
    prev_objective = obj_float

    # Momentum update for positions
    velocity_pos = momentum * velocity_pos - current_lr * grads_pos
    new_positions = positions + velocity_pos

    # Momentum update for radii (use a scaled LR since radii are smaller numbers)
    velocity_rad = momentum * velocity_rad - current_lr * grads_rad
    new_radii = radii + velocity_rad

    # Constrain positions to design space
    new_positions = jnp.clip(
        new_positions,
        jnp.array([view_x_start + radius_max + 5, view_y_start + radius_max + 5]),
        jnp.array([design_boundary_x - radius_max - 5, view_y_end - radius_max - 5])
    )

    # Constrain radii to [radius_min, radius_max]
    new_radii = jnp.clip(new_radii, radius_min, radius_max)

    positions = new_positions
    radii = new_radii

    # Store history
    optimization_history['positions'].append(positions.copy())
    optimization_history['radii'].append(radii.copy())
    optimization_history['objectives'].append(obj_float)
    optimization_history['pressures'].append(float(-obj_val))
    optimization_history['gradients_pos'].append(grads_pos.copy())
    optimization_history['gradients_rad'].append(grads_rad.copy())

    iter_end = pytime.time()
    iter_time = iter_end - iter_start
    optimization_history['iteration_times'].append(iter_time)
    print(f"  Iteration time: {iter_time:.2f} seconds")
    print(f"  Velocity (pos) norm: {jnp.linalg.norm(velocity_pos):.6f}")
    print(f"  Velocity (rad) norm: {jnp.linalg.norm(velocity_rad):.6f}")

    # Checkpoint
    if (iteration + 1) % checkpoint_every == 0:
        np.savez(checkpoint_path,
                 iteration=iteration,
                 positions=np.array(positions),
                 radii=np.array(radii),
                 velocity_pos=np.array(velocity_pos),
                 velocity_rad=np.array(velocity_rad),
                 current_lr=current_lr,
                 best_objective=best_objective,
                 best_positions=np.array(best_positions),
                 best_radii=np.array(best_radii),
                 best_iteration=best_iteration,
                 history_positions=np.array([np.array(p) for p in optimization_history['positions']]),
                 history_radii=np.array([np.array(r) for r in optimization_history['radii']]),
                 history_objectives=np.array(optimization_history['objectives']),
                 history_pressures=np.array(optimization_history['pressures']),
                 history_gradients_pos=np.array([np.array(g) for g in optimization_history['gradients_pos']]),
                 history_gradients_rad=np.array([np.array(g) for g in optimization_history['gradients_rad']]),
                 iteration_times=np.array(optimization_history['iteration_times']))
        print(f"  Checkpoint saved ({checkpoint_path})")

# Use best positions/radii as final result
final_positions = best_positions
final_radii = best_radii
print(f"\nUsing best state from iteration {best_iteration} (objective {best_objective:.6f}, pressure {-best_objective:.6f} Pa)")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)
print(f"\nInitial positions:\n{initial_positions}")
print(f"Initial radii: {initial_radii}")
print(f"\nFinal positions:\n{final_positions}")
print(f"Final radii: {final_radii}")
print(f"\nInitial pressure: {optimization_history['pressures'][0]:.6f} Pa")
print(f"Final pressure: {optimization_history['pressures'][-1]:.6f} Pa")
improvement = (optimization_history['pressures'][-1] / optimization_history['pressures'][0] - 1) * 100
print(f"Improvement: {improvement:.1f}%")

# ============================================================================
# Validation with Full Simulation
# ============================================================================

print("\n" + "="*70)
print("VALIDATION WITH FULL 300 μs SIMULATION")
print("="*70)

# Create full time axis
time_axis_full = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3,
    t_end=3.0e-04
)

time_array_full = time_axis_full.to_array()
stabilization_time_full = 2.0e-04
stabilization_index_full = int(jnp.argmin(jnp.abs(time_array_full - stabilization_time_full)))

print(f"\nFull simulation parameters:")
print(f"  Simulation time: {time_axis_full.t_end*1e6:.1f} μs")
print(f"  Stabilization time: {stabilization_time_full*1e6:.1f} μs")

# Run full simulation for initial config
print(f"\nRunning full simulation for initial configuration...")
val_start = pytime.time()

# Create medium
medium_init = create_medium_with_cylinders(domain, initial_positions, initial_radii)
p0_array = jnp.zeros(N)
p0_array = jnp.expand_dims(p0_array, -1)
p0 = FourierSeries(p0_array, domain)

class TimeVaryingSource:
    def __init__(self, time_axis, domain):
        self.domain = domain
        self.omega = 2 * jnp.pi * frequency
        self.time_array = time_axis.to_array()

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

source_init = TimeVaryingSource(time_axis_full, domain)
pressure_field_init = simulate_wave_propagation(medium_init, time_axis_full, p0=p0, sources=source_init)

pressure_on_grid_init = np.array(pressure_field_init.on_grid)[:, :, :, 0]

# Compute RMS over disk for initial config
disk_mask_np = np.array(create_disk_mask(N, target_focal_center, focal_disk_radius))
stabilized_init = pressure_on_grid_init[stabilization_index_full:, :, :]
pressure_in_disk_init = stabilized_init * disk_mask_np
n_disk_points = np.sum(disk_mask_np)
n_time_steps = stabilized_init.shape[0]
rms_init = np.sqrt(np.sum(pressure_in_disk_init**2) / (n_time_steps * n_disk_points))
initial_max_pressure_full = rms_init

val_end = pytime.time()
print(f"  Time: {val_end - val_start:.2f} seconds")
print(f"  RMS pressure over focal disk: {initial_max_pressure_full:.6f} Pa")

# Run full simulation for optimized config
print(f"\nRunning full simulation for optimized configuration...")
val_start = pytime.time()

medium_opt = create_medium_with_cylinders(domain, final_positions, final_radii)
source_opt = TimeVaryingSource(time_axis_full, domain)
pressure_field_opt = simulate_wave_propagation(medium_opt, time_axis_full, p0=p0, sources=source_opt)

pressure_on_grid_opt = np.array(pressure_field_opt.on_grid)[:, :, :, 0]

# Compute RMS over disk for optimized config
stabilized_opt = pressure_on_grid_opt[stabilization_index_full:, :, :]
pressure_in_disk_opt = stabilized_opt * disk_mask_np
rms_opt = np.sqrt(np.sum(pressure_in_disk_opt**2) / (n_time_steps * n_disk_points))
optimized_max_pressure_full = rms_opt

val_end = pytime.time()
print(f"  Time: {val_end - val_start:.2f} seconds")
print(f"  RMS pressure over focal disk: {optimized_max_pressure_full:.6f} Pa")

improvement_full = (optimized_max_pressure_full / initial_max_pressure_full - 1) * 100
print(f"\nFull Simulation Results (RMS over focal disk):")
print(f"  Initial RMS: {initial_max_pressure_full:.6f} Pa")
print(f"  Optimized RMS: {optimized_max_pressure_full:.6f} Pa")
print(f"  Improvement: {improvement_full:.1f}%")
print(f"  Focal disk: radius = {focal_disk_radius} points ({focal_disk_radius*dx[0]*1e3:.2f} mm), {int(n_disk_points)} points")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Plot 1: Optimization progress
ax = axes[0, 0]
ax.plot(optimization_history['pressures'], 'b-o', linewidth=2, markersize=6)
ax.axhline(initial_max_pressure_full, color='r', linestyle='--', alpha=0.5, label='Initial RMS (full sim)')
ax.axhline(optimized_max_pressure_full, color='g', linestyle='--', alpha=0.5, label='Optimized RMS (full sim)')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('RMS Pressure over Focal Disk (Pa)', fontsize=11)
ax.set_title('Autodiff Optimization Progress (RMS Objective)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Gradient norms (positions)
ax = axes[0, 1]
grad_norms_pos = [float(jnp.linalg.norm(g)) for g in optimization_history['gradients_pos']]
ax.plot(grad_norms_pos, 'r-o', linewidth=2, markersize=6, label='Position grads')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Gradient Norm', fontsize=11)
ax.set_title('Position Gradient Magnitude', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.legend()

# Plot 3: Radius gradient norms + radius evolution
ax = axes[0, 2]
grad_norms_rad = [float(jnp.linalg.norm(g)) for g in optimization_history['gradients_rad']]
ax.plot(grad_norms_rad, 'b-o', linewidth=2, markersize=6, label='Radius grads')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Gradient Norm', fontsize=11)
ax.set_title('Radius Gradient Magnitude', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.legend()

# Plot 4: Radius evolution over iterations
ax = axes[0, 3]
colors = plt.cm.rainbow(np.linspace(0, 1, n_cylinders))
for cyl_idx in range(n_cylinders):
    rad_history = [float(optimization_history['radii'][i][cyl_idx])
                   for i in range(len(optimization_history['radii']))]
    ax.plot(rad_history, 'o-', color=colors[cyl_idx], linewidth=2,
            markersize=5, label=f'Cyl {cyl_idx+1}')
ax.axhline(radius_min, color='gray', linestyle='--', alpha=0.5, label=f'r_min={radius_min}')
ax.axhline(radius_max, color='gray', linestyle=':', alpha=0.5, label=f'r_max={radius_max}')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Radius (grid points)', fontsize=11)
ax.set_title('Cylinder Radius Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 5 (was 3): Cylinder trajectories
ax = axes[1, 0]
ax.set_xlim(view_x_start, view_x_end)
ax.set_ylim(view_y_start, view_y_end)
ax.set_aspect('equal')
ax.set_xlabel('X (grid points)', fontsize=11)
ax.set_ylabel('Y (grid points)', fontsize=11)
ax.set_title('Cylinder Optimization Trajectory\n(circle size = final radius)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Draw boundaries
ax.axvline(design_boundary_x, color='red', linewidth=2, linestyle=':', alpha=0.5, label='Design')
ax.axvline(focus_boundary_x, color='cyan', linewidth=2, linestyle=':', alpha=0.5, label='Focus')
focus_rect = Rectangle((focus_boundary_x, view_y_start), view_x_end - focus_boundary_x,
                       view_y_end - view_y_start, fill=True, facecolor='cyan', alpha=0.1)
ax.add_patch(focus_rect)

# Plot focal disk region
from matplotlib.patches import Circle as PlotCircle
focal_circle = PlotCircle((target_focal_center[0], target_focal_center[1]), focal_disk_radius,
                     fill=False, edgecolor='red', linewidth=2, linestyle='-',
                     label='Focal disk', zorder=10)
ax.add_patch(focal_circle)
ax.plot(target_focal_center[0], target_focal_center[1], 'r+', markersize=15,
       markeredgewidth=2, zorder=11, label='Disk center')

# Plot trajectories — draw final cylinder position as a circle with final radius
for cyl_idx in range(n_cylinders):
    traj_x = [optimization_history['positions'][i][cyl_idx, 0]
              for i in range(len(optimization_history['positions']))]
    traj_y = [optimization_history['positions'][i][cyl_idx, 1]
              for i in range(len(optimization_history['positions']))]
    ax.plot(traj_x, traj_y, 'o-', color=colors[cyl_idx], linewidth=2,
            markersize=4, alpha=0.7, label=f'Cyl {cyl_idx+1}')

    # Mark start (square) and end (circle scaled to final radius)
    ax.plot(traj_x[0], traj_y[0], 's', color=colors[cyl_idx], markersize=12,
           markeredgecolor='black', markeredgewidth=2)
    final_r = float(final_radii[cyl_idx])
    end_circle = PlotCircle((traj_x[-1], traj_y[-1]), final_r,
                            fill=True, facecolor=colors[cyl_idx], alpha=0.4,
                            edgecolor='black', linewidth=1.5, zorder=5)
    ax.add_patch(end_circle)

ax.legend(fontsize=9, loc='upper left')

# Plot 6: Initial pressure field
ax = axes[1, 1]
final_pressure_init = np.mean(np.abs(pressure_on_grid_init[stabilization_index_full:, :, :]), axis=0)
pressure_view_init = final_pressure_init[view_x_start:view_x_end, view_y_start:view_y_end]

extent = [view_x_start * dx[0] * 1e3, view_x_end * dx[0] * 1e3,
          view_y_start * dx[1] * 1e3, view_y_end * dx[1] * 1e3]

vmax_viz = max(np.max(pressure_view_init),
               np.max(np.mean(np.abs(pressure_on_grid_opt[stabilization_index_full:, view_x_start:view_x_end, view_y_start:view_y_end]), axis=0)))

im = ax.imshow(pressure_view_init.T, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=vmax_viz)
ax.set_xlabel('x (mm)', fontsize=11)
ax.set_ylabel('y (mm)', fontsize=11)
ax.set_title(f'Initial Configuration\nRMS Pressure: {initial_max_pressure_full:.4f} Pa', fontsize=12, fontweight='bold')

# Draw cylinders with initial radii
for i in range(n_cylinders):
    x, y = initial_positions[i, 0], initial_positions[i, 1]
    r = float(initial_radii[i])
    circle = Circle((x*dx[0]*1e3, y*dx[1]*1e3), r*dx[0]*1e3,
                   fill=False, edgecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(circle)

ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
focal_disk_mm = focal_disk_radius * dx[0] * 1e3
focal_circle_patch = Circle((target_focal_center[0]*dx[0]*1e3, target_focal_center[1]*dx[1]*1e3),
                             focal_disk_mm, fill=False, edgecolor='lime', linewidth=2, zorder=10)
ax.add_patch(focal_circle_patch)

plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)

# Plot 7: Optimized pressure field
ax = axes[1, 2]
final_pressure_opt = np.mean(np.abs(pressure_on_grid_opt[stabilization_index_full:, :, :]), axis=0)
pressure_view_opt = final_pressure_opt[view_x_start:view_x_end, view_y_start:view_y_end]

im = ax.imshow(pressure_view_opt.T, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=vmax_viz)
ax.set_xlabel('x (mm)', fontsize=11)
ax.set_ylabel('y (mm)', fontsize=11)
ax.set_title(f'Optimized Configuration\nPressure: {optimized_max_pressure_full:.4f} Pa (+{improvement_full:.1f}%)',
            fontsize=12, fontweight='bold')



# Draw optimized cylinders with their optimized radii
for i in range(n_cylinders):
    x, y = final_positions[i, 0], final_positions[i, 1]
    r = float(final_radii[i])
    circle = Circle((x*dx[0]*1e3, y*dx[1]*1e3), r*dx[0]*1e3,
                   fill=False, edgecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(circle)

ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
focal_circle_patch5 = Circle((target_focal_center[0]*dx[0]*1e3, target_focal_center[1]*dx[1]*1e3),
                              focal_disk_mm, fill=False, edgecolor='lime', linewidth=2, zorder=10)
ax.add_patch(focal_circle_patch5)

plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)

# Plot 8: Summary
ax = axes[1, 3]
ax.axis('off')

summary_text = f"""
TRUE AUTODIFF OPTIMIZATION SUMMARY

Method:
  • JAX autodiff through j-wave solver
  • Physics-informed gradients
  • Gradients computed via:
    ∂(RMS_pressure)/∂(positions, radii)

Simulation:
  • Training time: {training_time_end*1e6:.0f} μs
  • Stabilization: {stabilization_time*1e6:.0f} μs
  • Time steps: {len(time_array)}

Optimization:
  • Iterations: {n_iterations}
  • Learning rate: {learning_rate}
  • Momentum: {momentum}
  • Radius bounds: [{radius_min:.0f}, {radius_max:.0f}] grid pts
  • Avg time/iter: {float(np.mean(optimization_history['iteration_times'])) if optimization_history.get('iteration_times') else 0:.1f} s

Initial Radii: {[f'{float(r):.1f}' for r in initial_radii]}
Final Radii:   {[f'{float(r):.1f}' for r in final_radii]}

Results (Full 300μs Validation):
  • Initial: {initial_max_pressure_full:.6f} Pa
  • Optimized: {optimized_max_pressure_full:.6f} Pa
  • Improvement: {improvement_full:.1f}%

Key Advantages:
  ✓ TRUE physics gradients
  ✓ Joint pos+radius optimization
  ✓ No surrogate approximation
  ✓ Respects wave equation

Trade-offs:
  ⚠ Slower per iteration (~30s)
  ✓ But gradients are EXACT
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('True Autodiff Optimization using J-Wave Differentiability',
            fontsize=14, fontweight='bold')
plt.tight_layout()

output_fig = os.path.join(results_dir, 'autodiff_optimization_results.png')
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"Saved results to: {output_fig}")

# Save results (including pressure views for replot script)
results_file = os.path.join(results_dir, 'optimization_results.npz')
np.savez(results_file,
         initial_positions=np.array(initial_positions),
         initial_radii=np.array(initial_radii),
         final_positions=np.array(final_positions),
         final_radii=np.array(final_radii),
         optimization_history_positions=np.array([np.array(p) for p in optimization_history['positions']]),
         optimization_history_radii=np.array([np.array(r) for r in optimization_history['radii']]),
         optimization_history_pressures=np.array(optimization_history['pressures']),
         optimization_history_gradients_pos=np.array([np.array(g) for g in optimization_history['gradients_pos']]),
         optimization_history_gradients_rad=np.array([np.array(g) for g in optimization_history['gradients_rad']]),
         initial_pressure=initial_max_pressure_full,
         optimized_pressure=optimized_max_pressure_full,
         pressure_view_init=pressure_view_init,
         pressure_view_opt=pressure_view_opt,
         iteration_times=np.array(optimization_history.get('iteration_times', [])),
         n_iterations=n_iterations,
         learning_rate=learning_rate,
         momentum=momentum,
         training_time_end=training_time_end,
         stabilization_time=stabilization_time)
print(f"Saved optimization data to: {results_file}")

print("\n" + "="*70)
print("TRUE AUTODIFF OPTIMIZATION COMPLETE!")
print("="*70)
print("\nThis approach used JAX's autodiff to compute exact gradients")
print("by differentiating through the entire wave equation solver.")
print("="*70)
print("\nDone!")