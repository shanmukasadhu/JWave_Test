"""
3D Acoustic wave scattering simulation using jwave
- 3D cylinders in water with VARYING RADII
- CONTINUOUS spherical wave from a point-like source (left side of domain)
- DESIGN SPACE vs FOCUS SPACE split (3D version)
- Maximum pressure computed ONLY in focus space
- 3D visualization with volume rendering and slices
"""

import numpy as np
from jax import numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
import time

# Import jwave components
from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis

# Simulation parameters - 3D grid (smaller than 2D to keep it manageable)
N = (128, 128, 128)  # 3D grid size (reduced from 512 to keep computation reasonable)
dx = (0.3e-3, 0.3e-3, 0.3e-3)  # Grid spacing in meters (0.3 mm, slightly larger)
domain = Domain(N, dx)

# Define viewing window - must be within grid N=(128,128,128), so indices 0..127
view_x_start = 0
view_x_end = 128
view_y_start = 0
view_y_end = 128
view_z_start = 0
view_z_end = 128

# Medium properties
c_water = 1500.0  # Speed of sound in water (m/s)
c_cylinder = 2500.0  # Speed of sound in cylinders (m/s)
rho_water = 1000.0  # Density of water (kg/m^3)
rho_cylinder = 1200.0  # Density of cylinders (kg/m^3)

# Continuous wave parameters
frequency = 1.0e6  # 1 MHz continuous wave
wavelength = c_water / frequency  # Wavelength in water
k = 2 * np.pi / wavelength  # Wave number

# Time axis
time_axis = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3,
    t_end=5.0e-05  # 100 microseconds (shorter for 3D to save time)
)

# DESIGN SPACE vs FOCUS SPACE SPLIT (in x-direction, must be within 0..127)
design_boundary_x = 85   # Grid index - cylinders must be before this
focus_boundary_x = 100   # Grid index - focus measurement starts here

# Spherical wave source: center (grid indices) and radius (grid points)
source_center = (12, N[1] // 2, N[2] // 2)  # near left, center in y and z
source_radius = 5  # grid points (small sphere for point-like radiation)

print("="*70)
print("3D DESIGN SPACE vs FOCUS SPACE (SPHERICAL WAVE SOURCE)")
print("="*70)
print(f"Grid dimensions: {N[0]} × {N[1]} × {N[2]}")
print(f"Viewing window: x ∈ [{view_x_start}, {view_x_end}], y ∈ [{view_y_start}, {view_y_end}], z ∈ [{view_z_start}, {view_z_end}]")
print(f"Design space: x ∈ [{view_x_start}, {design_boundary_x}] (cylinders placed here)")
print(f"Buffer zone: x ∈ [{design_boundary_x}, {focus_boundary_x}] (no cylinders, no measurement)")
print(f"Focus space: x ∈ [{focus_boundary_x}, {view_x_end}] (pressure measurement only)")
print(f"\nPhysical dimensions:")
print(f"  Design space: x ∈ [{view_x_start*dx[0]*1e3:.1f}, {design_boundary_x*dx[0]*1e3:.1f}] mm")
print(f"  Buffer zone: x ∈ [{design_boundary_x*dx[0]*1e3:.1f}, {focus_boundary_x*dx[0]*1e3:.1f}] mm")
print(f"  Focus space: x ∈ [{focus_boundary_x*dx[0]*1e3:.1f}, {view_x_end*dx[0]*1e3:.1f}] mm")
print(f"\nGrid spacing: {dx[0]*1e6:.1f} μm")
print(f"Wavelength: {wavelength*1e3:.3f} mm")
print(f"Spherical source: center {source_center}, radius {source_radius} grid points")
print("="*70)

def create_cylinder_mask_3d(N, center, radius, height, axis='z'):
    """
    Create a 3D cylindrical mask

    Parameters:
    - N: Grid size (nx, ny, nz)
    - center: (cx, cy, cz) center of cylinder
    - radius: radius of cylinder
    - height: height of cylinder along specified axis
    - axis: 'x', 'y', or 'z' - direction of cylinder axis
    """
    x = jnp.arange(N[0])
    y = jnp.arange(N[1])
    z = jnp.arange(N[2])
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

    cx, cy, cz = center

    if axis == 'z':
        # Cylinder along z-axis
        dist = jnp.sqrt((X - cx)**2 + (Y - cy)**2)
        radial_mask = dist <= radius
        axial_mask = jnp.abs(Z - cz) <= height / 2
        mask = radial_mask & axial_mask
    elif axis == 'y':
        # Cylinder along y-axis
        dist = jnp.sqrt((X - cx)**2 + (Z - cz)**2)
        radial_mask = dist <= radius
        axial_mask = jnp.abs(Y - cy) <= height / 2
        mask = radial_mask & axial_mask
    elif axis == 'x':
        # Cylinder along x-axis
        dist = jnp.sqrt((Y - cy)**2 + (Z - cz)**2)
        radial_mask = dist <= radius
        axial_mask = jnp.abs(X - cx) <= height / 2
        mask = radial_mask & axial_mask

    return mask

def create_medium_with_cylinders_3d(domain, cylinder_configs):
    """
    Create a heterogeneous 3D medium with cylinders

    cylinder_configs: list of (x, y, z, radius, height, axis)
    """
    # Initialize with water properties
    sound_speed = jnp.ones(N) * c_water
    density = jnp.ones(N) * rho_water

    # Add cylinders with individual radii and heights
    for config in cylinder_configs:
        x, y, z, radius, height, axis = config
        mask = create_cylinder_mask_3d(N, (x, y, z), radius, height, axis)
        sound_speed = jnp.where(mask, c_cylinder, sound_speed)
        density = jnp.where(mask, rho_cylinder, density)

    # Add fourth dimension for jwave
    sound_speed = jnp.expand_dims(sound_speed, -1)
    density = jnp.expand_dims(density, -1)

    # Large PML to absorb boundary reflections
    return Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=20)

def find_max_pressure_in_focus_space_3d(pressure_field, focus_x_start, focus_x_end):
    """
    Find maximum pressure in the 3D focus space only

    Parameters:
    - pressure_field: 3D array of pressure values
    - focus_x_start, focus_x_end: Boundaries of focus space in grid coordinates

    Returns:
    - max_pressure: Maximum absolute pressure value in focus space
    - max_location: (x, y, z) coordinates of maximum in global grid
    """
    n_x, n_y, n_z = pressure_field.shape
    # Clamp to actual grid (avoid out-of-range and empty slices)
    focus_x_start = min(max(0, focus_x_start), n_x)
    focus_x_end = min(max(0, focus_x_end), n_x)
    if focus_x_start > focus_x_end:
        focus_x_start, focus_x_end = focus_x_end, focus_x_start
    if focus_x_start >= focus_x_end:
        # No focus region (e.g. boundaries outside grid)
        return 0.0, (focus_x_start, n_y // 2, n_z // 2)

    # Extract focus space region
    focus_region = pressure_field[focus_x_start:focus_x_end, :, :]

    # Find maximum absolute pressure
    pressure_magnitude = np.abs(focus_region)
    max_val = np.max(pressure_magnitude)

    # Find location in focus region
    max_idx_local = np.unravel_index(np.argmax(pressure_magnitude), pressure_magnitude.shape)

    # Convert to global coordinates
    max_location = (focus_x_start + max_idx_local[0], max_idx_local[1], max_idx_local[2])

    return max_val, max_location

# Define 3 different cylinder configurations: 5 cylinders each, randomly placed in design space
# Format: (x, y, z, radius, height, axis)
# All cylinders MUST be in design space (x + extent < design_boundary_x = 85), grid is 0..127
np.random.seed(42)

def cylinder_aabb(cyl):
    """Return (min_x, max_x, min_y, max_y, min_z, max_z) for cylinder (x, y, z, r, h, axis)."""
    x, y, z, r, h, axis = cyl
    half = h / 2.0
    if axis == 'z':
        return (x - r, x + r, y - r, y + r, z - half, z + half)
    if axis == 'y':
        return (x - r, x + r, y - half, y + half, z - r, z + r)
    # axis == 'x'
    return (x - half, x + half, y - r, y + r, z - r, z + r)

def aabbs_overlap(a, b, gap=2):
    """True if two AABBs (min_x, max_x, min_y, max_y, min_z, max_z) overlap (with optional gap)."""
    return (a[0] < b[1] + gap and b[0] < a[1] + gap and
            a[2] < b[3] + gap and b[2] < a[3] + gap and
            a[4] < b[5] + gap and b[4] < a[5] + gap)

def random_cylinder_config(n_cylinders=5, x_max=None, margin=12, min_gap=2, max_attempts=500):
    """Generate n_cylinders with random (x,y,z), radius, height, axis; all in design space, no overlap."""
    if x_max is None:
        x_max = design_boundary_x
    x_lo, x_hi = margin, x_max - 22
    y_lo, y_hi = margin, N[1] - margin
    z_lo, z_hi = margin, N[2] - margin
    axes = ['z', 'y', 'x']
    config = []
    for _ in range(n_cylinders):
        for _ in range(max_attempts):
            x = int(np.random.uniform(x_lo, x_hi))
            y = int(np.random.uniform(y_lo, y_hi))
            z = int(np.random.uniform(z_lo, z_hi))
            r = int(np.random.uniform(4, 7))
            h = int(np.random.uniform(24, 36))
            axis = np.random.choice(axes)
            cand = (x, y, z, r, h, axis)
            cand_box = cylinder_aabb(cand)
            if all(not aabbs_overlap(cand_box, cylinder_aabb(ex), min_gap) for ex in config):
                config.append(cand)
                break
        else:
            raise RuntimeError("random_cylinder_config: could not place non-overlapping cylinder after max_attempts")
    return config

configurations = [
    random_cylinder_config(5),
    random_cylinder_config(5),
    random_cylinder_config(5),
]

# Validate that all cylinders are in design space
print("\nValidating cylinder positions...")
for i, config in enumerate(configurations):
    for j, (x, y, z, r, h, axis) in enumerate(config):
        max_x = x + (h/2 if axis == 'x' else r)
        if max_x > design_boundary_x:
            print(f"WARNING: Config {i+1}, Cylinder {j+1} extends beyond design boundary!")
            print(f"  Position: x={x}, max_x={max_x}, boundary: {design_boundary_x}")
        else:
            print(f"Config {i+1}, Cylinder {j+1}: OK (x={x}, max_x={max_x:.1f} < {design_boundary_x})")

# Run simulations for all configurations
print("\n" + "="*70)
print("Running 3D jwave simulations...")
print("="*70)

results = []

for i, config in enumerate(configurations):
    print(f"\nConfiguration {i+1}/3: {len(config)} cylinders")
    start_time = time.time()

    # Create medium with cylinders
    medium = create_medium_with_cylinders_3d(domain, config)

    # Create initial condition
    p0_array = jnp.zeros(N)
    p0_array = jnp.expand_dims(p0_array, -1)
    p0 = FourierSeries(p0_array, domain)

    # Create time-varying source (spherical wave: small sphere radiating outward)
    class TimeVaryingSource:
        def __init__(self, time_axis, domain):
            self.domain = domain
            self.omega = 2 * jnp.pi * frequency
            self.time_array = time_axis.to_array()

            # Spherical source mask: 1.0 inside sphere of source_radius around source_center
            cx, cy, cz = source_center
            x = jnp.arange(N[0])
            y = jnp.arange(N[1])
            z = jnp.arange(N[2])
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
            source_mask = (dist_sq <= source_radius**2).astype(jnp.float32)
            source_mask = jnp.expand_dims(source_mask, -1)

            # Precompute all source fields
            source_fields_list = []
            for t in self.time_array:
                source_amplitude = 1.0 * jnp.sin(self.omega * t)
                source_fields_list.append(source_mask * source_amplitude)

            self.source_fields = jnp.stack(source_fields_list, axis=0)

        def on_grid(self, time_index):
            time_idx = lax.convert_element_type(time_index, jnp.int32)
            start_indices = (time_idx, 0, 0, 0, 0)
            slice_sizes = (1, N[0], N[1], N[2], 1)
            sliced = lax.dynamic_slice(self.source_fields, start_indices, slice_sizes)
            return jnp.squeeze(sliced, axis=0)

    time_varying_source = TimeVaryingSource(time_axis, domain)

    # Run simulation
    print("  Running 3D wave simulation...")
    pressure_field = simulate_wave_propagation(
        medium,
        time_axis,
        p0=p0,
        sources=time_varying_source
    )

    end_time = time.time()

    # Extract pressure data
    pressure_on_grid = np.array(pressure_field.on_grid)
    final_pressure = pressure_on_grid[-1, :, :, :, 0]

    # Extract sound speed
    sound_speed_array = np.array(medium.sound_speed)
    if sound_speed_array.ndim == 4:
        sound_speed = sound_speed_array[:, :, :, 0]
    else:
        sound_speed = np.ones(N) * c_water
        for x, y, z, radius, height, axis in config:
            mask = np.array(create_cylinder_mask_3d(N, (x, y, z), radius, height, axis))
            sound_speed = np.where(mask, c_cylinder, sound_speed)

    # Find maximum pressure in FOCUS SPACE ONLY
    max_pressure, max_loc = find_max_pressure_in_focus_space_3d(
        final_pressure,
        focus_boundary_x,
        view_x_end
    )

    print(f"  Simulation time: {end_time - start_time:.2f} seconds")
    print(f"  Maximum pressure in focus space: {max_pressure:.6f} Pa")
    print(f"  Location: ({max_loc[0]}, {max_loc[1]}, {max_loc[2]})")

    # Store results
    results.append({
        'pressure': final_pressure,
        'sound_speed': sound_speed,
        'config': config,
        'max_pressure': max_pressure,
        'max_location': max_loc,
        'pressure_timeseries': pressure_on_grid[:, :, :, :, 0]  # 4D: (time, x, y, z)
    })

print("\n" + "="*70)
print("MAXIMUM PRESSURE SUMMARY (Focus Space Only - 3D)")
print("="*70)
for i, result in enumerate(results):
    max_loc = result['max_location']
    print(f"Config {i+1}:")
    print(f"  Max pressure: {result['max_pressure']:.6f} Pa")
    print(f"  Location: ({max_loc[0]}, {max_loc[1]}, {max_loc[2]})")
    print(f"  Physical: ({max_loc[0]*dx[0]*1e3:.2f}, {max_loc[1]*dx[1]*1e3:.2f}, {max_loc[2]*dx[2]*1e3:.2f}) mm")
print("="*70)

# Create 3D visualizations
print("\nCreating 3D visualizations...")
import os
os.makedirs('../Results/continuous_wave_split_3d_spherical', exist_ok=True)

for i, result in enumerate(results):
    print(f"Visualization {i+1}/3...")

    # Create figure with 3D and 2D slice views
    fig = plt.figure(figsize=(20, 12))

    # Get data
    pressure_data = result['pressure']
    sound_speed_data = result['sound_speed']
    config = result['config']

    # Extract viewing window
    pressure_view = pressure_data[view_x_start:view_x_end,
                                  view_y_start:view_y_end,
                                  view_z_start:view_z_end]
    sound_speed_view = sound_speed_data[view_x_start:view_x_end,
                                       view_y_start:view_y_end,
                                       view_z_start:view_z_end]

    # Define extent for 2D slices
    extent_xy = [view_x_start * dx[0] * 1e3, view_x_end * dx[0] * 1e3,
                 view_y_start * dx[1] * 1e3, view_y_end * dx[1] * 1e3]
    extent_xz = [view_x_start * dx[0] * 1e3, view_x_end * dx[0] * 1e3,
                 view_z_start * dx[2] * 1e3, view_z_end * dx[2] * 1e3]
    extent_yz = [view_y_start * dx[1] * 1e3, view_y_end * dx[1] * 1e3,
                 view_z_start * dx[2] * 1e3, view_z_end * dx[2] * 1e3]

    # Get middle slices
    mid_z = (view_z_end - view_z_start) // 2
    mid_y = (view_y_end - view_y_start) // 2
    mid_x = (view_x_end - view_x_start) // 2

    # Row 1: XY slices (top view)
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(sound_speed_view[:, :, mid_z].T, cmap='viridis',
                     origin='lower', extent=extent_xy, aspect='auto')
    ax1.set_title(f'Config {i+1}: Sound Speed (XY slice at z={view_z_start + mid_z})',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('x (mm)', fontsize=9)
    ax1.set_ylabel('y (mm)', fontsize=9)
    ax1.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=2,
                linestyle=':', alpha=0.7, label='Design')
    ax1.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=2,
                linestyle=':', alpha=0.7, label='Focus')
    ax1.legend(fontsize=8)
    plt.colorbar(im1, ax=ax1, label='c (m/s)', fraction=0.046)

    ax2 = fig.add_subplot(2, 3, 2)
    vmax = np.max(np.abs(pressure_view))
    im2 = ax2.imshow(pressure_view[:, :, mid_z].T, cmap='RdBu_r',
                     origin='lower', extent=extent_xy, vmin=-vmax, vmax=vmax, aspect='auto')
    ax2.set_title(f'Pressure (XY slice)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('x (mm)', fontsize=9)
    ax2.set_ylabel('y (mm)', fontsize=9)
    ax2.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5,
                linestyle=':', alpha=0.5)
    ax2.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5,
                linestyle=':', alpha=0.5)
    plt.colorbar(im2, ax=ax2, label='P (Pa)', fraction=0.046)

    # XZ slice (side view)
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(pressure_view[:, mid_y, :].T, cmap='RdBu_r',
                     origin='lower', extent=extent_xz, vmin=-vmax, vmax=vmax, aspect='auto')
    ax3.set_title(f'Pressure (XZ slice)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('x (mm)', fontsize=9)
    ax3.set_ylabel('z (mm)', fontsize=9)
    ax3.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5,
                linestyle=':', alpha=0.5)
    ax3.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5,
                linestyle=':', alpha=0.5)
    plt.colorbar(im3, ax=ax3, label='P (Pa)', fraction=0.046)

    # Row 2: 3D isosurface plot
    ax_3d = fig.add_subplot(2, 3, 4, projection='3d')

    # Create coordinate grids for viewing window
    x_coords = np.arange(view_x_start, view_x_end) * dx[0] * 1e3
    y_coords = np.arange(view_y_start, view_y_end) * dx[1] * 1e3
    z_coords = np.arange(view_z_start, view_z_end) * dx[2] * 1e3

    # Plot cylinders as 3D objects (all axes so all cylinders are visible)
    theta = np.linspace(0, 2*np.pi, 20)
    for x, y, z, r, h, axis in config:
        x_mm = x * dx[0] * 1e3
        y_mm = y * dx[1] * 1e3
        z_mm = z * dx[2] * 1e3
        r_mm_x = r * dx[0] * 1e3
        r_mm_y = r * dx[1] * 1e3
        r_mm_z = r * dx[2] * 1e3
        half_h_mm_x = (h/2) * dx[0] * 1e3
        half_h_mm_y = (h/2) * dx[1] * 1e3
        half_h_mm_z = (h/2) * dx[2] * 1e3
        if axis == 'z':
            z_lin = np.linspace(z_mm - half_h_mm_z, z_mm + half_h_mm_z, 10)
            theta_grid, z_grid = np.meshgrid(theta, z_lin)
            x_cyl = x_mm + r_mm_x * np.cos(theta_grid)
            y_cyl = y_mm + r_mm_y * np.sin(theta_grid)
            ax_3d.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.6, color='orange')
        elif axis == 'y':
            y_lin = np.linspace(y_mm - half_h_mm_y, y_mm + half_h_mm_y, 10)
            theta_grid, y_grid = np.meshgrid(theta, y_lin)
            x_cyl = x_mm + r_mm_x * np.cos(theta_grid)
            z_cyl = z_mm + r_mm_z * np.sin(theta_grid)
            ax_3d.plot_surface(x_cyl, y_grid, z_cyl, alpha=0.6, color='orange')
        else:  # axis == 'x'
            x_lin = np.linspace(x_mm - half_h_mm_x, x_mm + half_h_mm_x, 10)
            theta_grid, x_grid = np.meshgrid(theta, x_lin)
            y_cyl = y_mm + r_mm_y * np.cos(theta_grid)
            z_cyl = z_mm + r_mm_z * np.sin(theta_grid)
            ax_3d.plot_surface(x_grid, y_cyl, z_cyl, alpha=0.6, color='orange')

    # Plot design/focus boundaries
    y_plane = np.linspace(view_y_start*dx[1]*1e3, view_y_end*dx[1]*1e3, 10)
    z_plane = np.linspace(view_z_start*dx[2]*1e3, view_z_end*dx[2]*1e3, 10)
    Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
    X_design = np.ones_like(Y_plane) * design_boundary_x * dx[0] * 1e3
    X_focus = np.ones_like(Y_plane) * focus_boundary_x * dx[0] * 1e3

    ax_3d.plot_surface(X_design, Y_plane, Z_plane, alpha=0.2, color='red')
    ax_3d.plot_surface(X_focus, Y_plane, Z_plane, alpha=0.2, color='cyan')

    ax_3d.set_xlabel('x (mm)', fontsize=9)
    ax_3d.set_ylabel('y (mm)', fontsize=9)
    ax_3d.set_zlabel('z (mm)', fontsize=9)
    ax_3d.set_title('3D Geometry View', fontsize=11, fontweight='bold')

    # YZ slice (front view)
    ax4 = fig.add_subplot(2, 3, 5)
    im4 = ax4.imshow(pressure_view[mid_x, :, :].T, cmap='RdBu_r',
                     origin='lower', extent=extent_yz, vmin=-vmax, vmax=vmax, aspect='auto')
    ax4.set_title(f'Pressure (YZ slice)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('y (mm)', fontsize=9)
    ax4.set_ylabel('z (mm)', fontsize=9)
    plt.colorbar(im4, ax=ax4, label='P (Pa)', fraction=0.046)

    # Pressure magnitude with max marked
    ax5 = fig.add_subplot(2, 3, 6)
    im5 = ax5.imshow(np.abs(pressure_view[:, :, mid_z]).T, cmap='hot',
                     origin='lower', extent=extent_xy, aspect='auto')
    ax5.set_title(f'|Pressure| (XY slice)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('x (mm)', fontsize=9)
    ax5.set_ylabel('y (mm)', fontsize=9)
    ax5.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5,
                linestyle=':', alpha=0.5)
    ax5.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5,
                linestyle=':', alpha=0.5)

    # Mark maximum if it's in this slice
    max_loc = result['max_location']
    if abs(max_loc[2] - (view_z_start + mid_z)) < 5:
        ax5.plot(max_loc[0]*dx[0]*1e3, max_loc[1]*dx[1]*1e3, 'co', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label=f'Max: {result["max_pressure"]:.4f} Pa')
        ax5.legend(fontsize=8)

    plt.colorbar(im5, ax=ax5, label='|P| (Pa)', fraction=0.046)

    plt.suptitle(f'3D Design/Focus Split - Configuration {i+1}\n' +
                 f'Max in focus: {result["max_pressure"]:.4f} Pa at ' +
                 f'({max_loc[0]}, {max_loc[1]}, {max_loc[2]})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = f'../Results/continuous_wave_split_3d_spherical/config_{i+1}_3d_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)

print("\nCreating 3D animations...")
from matplotlib import animation

# Create animation for first configuration as example
result = results[0]
pressure_time_series = result['pressure_timeseries']
sound_speed_map = result['sound_speed']

n_frames = pressure_time_series.shape[0]
mid_x_global = N[0] // 2
mid_y_global = N[1] // 2
mid_z_global = N[2] // 2

# Extents in mm for all three views
extent_xy = [view_x_start * dx[0] * 1e3, view_x_end * dx[0] * 1e3,
             view_y_start * dx[1] * 1e3, view_y_end * dx[1] * 1e3]
extent_xz = [view_x_start * dx[0] * 1e3, view_x_end * dx[0] * 1e3,
             view_z_start * dx[2] * 1e3, view_z_end * dx[2] * 1e3]
extent_yz = [view_y_start * dx[1] * 1e3, view_y_end * dx[1] * 1e3,
             view_z_start * dx[2] * 1e3, view_z_end * dx[2] * 1e3]

# Full 3D view: three orthogonal slices (XY, XZ, YZ) animated together
pressure_xy = pressure_time_series[:, view_x_start:view_x_end,
                                   view_y_start:view_y_end, mid_z_global]
pressure_xz = pressure_time_series[:, view_x_start:view_x_end, mid_y_global,
                                   view_z_start:view_z_end]
pressure_yz = pressure_time_series[:, mid_x_global,
                                   view_y_start:view_y_end, view_z_start:view_z_end]

vmax_anim = np.max(np.abs(pressure_time_series))

# Max pressure location (focus space) in grid indices and mm
max_loc = results[0]['max_location']
max_x_mm = max_loc[0] * dx[0] * 1e3
max_y_mm = max_loc[1] * dx[1] * 1e3
max_z_mm = max_loc[2] * dx[2] * 1e3

fig_3d, axes_3d = plt.subplots(1, 3, figsize=(18, 6))

# XY (top-down)
im_xy = axes_3d[0].imshow(pressure_xy[0].T, cmap='RdBu_r', origin='lower',
                          extent=extent_xy, vmin=-vmax_anim, vmax=vmax_anim, aspect='auto')
axes_3d[0].set_title('XY slice (top view)', fontsize=12, fontweight='bold')
axes_3d[0].set_xlabel('x (mm)')
axes_3d[0].set_ylabel('y (mm)')
axes_3d[0].axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.7)
axes_3d[0].axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.7)
pt_xy = axes_3d[0].scatter([max_x_mm], [max_y_mm], c='lime', s=200, marker='*', edgecolors='black',
                            linewidths=1, zorder=10, label='Max pressure')
plt.colorbar(im_xy, ax=axes_3d[0], label='Pressure (Pa)', fraction=0.046)

# XZ (side view, wave propagates left-to-right)
im_xz = axes_3d[1].imshow(pressure_xz[0].T, cmap='RdBu_r', origin='lower',
                          extent=extent_xz, vmin=-vmax_anim, vmax=vmax_anim, aspect='auto')
axes_3d[1].set_title('XZ slice (side view)', fontsize=12, fontweight='bold')
axes_3d[1].set_xlabel('x (mm)')
axes_3d[1].set_ylabel('z (mm)')
axes_3d[1].axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.7)
axes_3d[1].axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.7)
pt_xz = axes_3d[1].scatter([max_x_mm], [max_z_mm], c='lime', s=200, marker='*', edgecolors='black',
                            linewidths=1, zorder=10, label='Max pressure')
plt.colorbar(im_xz, ax=axes_3d[1], label='Pressure (Pa)', fraction=0.046)

# YZ (front view)
im_yz = axes_3d[2].imshow(pressure_yz[0].T, cmap='RdBu_r', origin='lower',
                          extent=extent_yz, vmin=-vmax_anim, vmax=vmax_anim, aspect='auto')
axes_3d[2].set_title('YZ slice (front view)', fontsize=12, fontweight='bold')
axes_3d[2].set_xlabel('y (mm)')
axes_3d[2].set_ylabel('z (mm)')
pt_yz = axes_3d[2].scatter([max_y_mm], [max_z_mm], c='lime', s=200, marker='*', edgecolors='black',
                            linewidths=1, zorder=10, label='Max pressure')
plt.colorbar(im_yz, ax=axes_3d[2], label='Pressure (Pa)', fraction=0.046)

time_text_3d = fig_3d.text(0.5, 0.02, '', ha='center', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def init_3d():
    im_xy.set_data(pressure_xy[0].T)
    im_xz.set_data(pressure_xz[0].T)
    im_yz.set_data(pressure_yz[0].T)
    time_text_3d.set_text('')
    return [im_xy, im_xz, im_yz, pt_xy, pt_xz, pt_yz, time_text_3d]

def animate_3d(frame):
    im_xy.set_data(pressure_xy[frame].T)
    im_xz.set_data(pressure_xz[frame].T)
    im_yz.set_data(pressure_yz[frame].T)
    time_val = time_axis.to_array()[frame] * 1e6
    time_text_3d.set_text(f'Time: {time_val:.2f} μs  |  Frame: {frame}/{n_frames}  |  Full 3D volume (XY, XZ, YZ)  |  Max: {results[0]["max_pressure"]:.4f} Pa')
    return [im_xy, im_xz, im_yz, pt_xy, pt_xz, pt_yz, time_text_3d]

target_video_frames = 300
frame_skip = max(1, n_frames // target_video_frames)
frames_to_use = list(range(0, n_frames, frame_skip))

anim_3d = animation.FuncAnimation(fig_3d, animate_3d, init_func=init_3d,
                                 frames=frames_to_use, interval=33,
                                 blit=True, repeat=True)

video_path_3d = f'../Results/continuous_wave_split_3d_spherical/config_1_3d_full_volume_animation.mp4'
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, bitrate=4000)
fig_3d.suptitle(f'3D Acoustic Wave Propagation - Full Volume (Configuration 1)\n' +
                f'Max pressure in focus: {results[0]["max_pressure"]:.4f} Pa',
                fontsize=14, fontweight='bold')
anim_3d.save(video_path_3d, writer=writer)
print(f"  Saved full 3D (3-panel) animation: {video_path_3d}")
plt.close(fig_3d)

# Optional: 3D isosurface animation with cylinders and design/focus boundaries
try:
    from skimage import measure
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # ---- Cylinder geometry in mm (for 3D viz) ----
    def cylinder_mesh_mm(center_ijk, radius_grid, height_grid, axis, n_theta=24, n_axial=8):
        """Create cylinder surface vertices and faces in mm. center_ijk and radius/height in grid units."""
        cx = center_ijk[0] * dx[0] * 1e3
        cy = center_ijk[1] * dx[1] * 1e3
        cz = center_ijk[2] * dx[2] * 1e3
        r_mm = radius_grid * (dx[0] if axis != 'x' else dx[1]) * 1e3
        h_mm = height_grid * (dx[2] if axis == 'z' else (dx[1] if axis == 'y' else dx[0])) * 1e3
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        t = np.linspace(-0.5, 0.5, n_axial) * h_mm
        verts = []
        if axis == 'z':
            for ti in t:
                for th in theta:
                    verts.append([cx + r_mm * np.cos(th), cy + r_mm * np.sin(th), cz + ti])
        elif axis == 'y':
            for ti in t:
                for th in theta:
                    verts.append([cx + r_mm * np.cos(th), cy + ti, cz + r_mm * np.sin(th)])
        else:  # x
            for ti in t:
                for th in theta:
                    verts.append([cx + ti, cy + r_mm * np.cos(th), cz + r_mm * np.sin(th)])
        verts = np.array(verts)
        # Build faces (quads as two tris)
        faces = []
        for j in range(n_axial - 1):
            for i in range(n_theta):
                i2 = (i + 1) % n_theta
                a = j * n_theta + i
                b = j * n_theta + i2
                c = (j + 1) * n_theta + i2
                d = (j + 1) * n_theta + i
                faces.append(np.array([verts[a], verts[b], verts[c]]))
                faces.append(np.array([verts[a], verts[c], verts[d]]))
        return verts, faces

    # Downsample for faster isosurface
    stride = 2
    # Full volume: signed pressure (to show crests/troughs and scattering)
    pressure_ds = pressure_time_series[:, view_x_start:view_x_end:stride,
                                       view_y_start:view_y_end:stride,
                                       view_z_start:view_z_end:stride]
    vol_ds = np.abs(pressure_ds)
    global_max_vol = np.max(vol_ds)
    if global_max_vol < 1e-12:
        raise ValueError("Pressure data is effectively zero; no isosurface to show.")

    # Two isosurface levels: main wave (strong) + weaker level to show scattered/diffracted field
    level_main = 0.22 * global_max_vol
    level_weak = 0.06 * global_max_vol   # scattered field is weaker -> shows at lower level
    level_signed = 0.12 * global_max_vol  # for signed crest/trough surfaces

    def verts_to_mm(verts_local):
        """Convert marching_cubes vertex indices (in downsampled view) to mm."""
        v = verts_local * stride * np.array([dx[0], dx[1], dx[2]]) * 1e3
        v[:, 0] += view_x_start * dx[0] * 1e3
        v[:, 1] += view_y_start * dx[1] * 1e3
        v[:, 2] += view_z_start * dx[2] * 1e3
        return v

    def make_isosurface_mesh(vol, level, facecolor='steelblue', alpha=0.5):
        """Single level (vol can be abs or signed)."""
        for frac in [1.0, 0.5, 0.25, 0.1]:
            l = level * frac
            if l <= 0:
                continue
            try:
                verts, faces, _, _ = measure.marching_cubes(vol, level=l)
                verts = verts_to_mm(verts)
                return Poly3DCollection(verts[faces], alpha=alpha, edgecolor='none', facecolor=facecolor)
            except (ValueError, RuntimeError):
                continue
        return None

    def make_signed_isosurfaces(vol_signed, level):
        """Two meshes: positive pressure (crest) and negative (trough) to show wavefront shape and scattering."""
        meshes = []
        for sign, color in [(1, 'crimson'), (-1, 'dodgerblue')]:
            for frac in [1.0, 0.5, 0.25]:
                l = sign * level * frac
                try:
                    verts, faces, _, _ = measure.marching_cubes(vol_signed, level=l)
                    verts = verts_to_mm(verts)
                    meshes.append(Poly3DCollection(verts[faces], alpha=0.45, edgecolor='none', facecolor=color))
                    break
                except (ValueError, RuntimeError):
                    continue
        return meshes

    n_frames_isosurf = min(60, len(frames_to_use))
    frames_isosurf = frames_to_use[::max(1, len(frames_to_use) // n_frames_isosurf)][:n_frames_isosurf]

    fig_iso = plt.figure(figsize=(12, 10))
    ax_iso = fig_iso.add_subplot(111, projection='3d')

    # Static geometry: cylinders (config 1)
    config_cyl = results[0]['config']
    static_collections = set()
    for (gx, gy, gz, r, h, axis) in config_cyl:
        _, faces = cylinder_mesh_mm((gx, gy, gz), r, h, axis)
        poly = Poly3DCollection(faces, alpha=0.85, facecolor='darkorange', edgecolor='black', linewidths=0.5)
        col = ax_iso.add_collection3d(poly)
        static_collections.add(col)

    # Design and focus boundary planes (vertical planes at x = const in mm)
    x_design_mm = design_boundary_x * dx[0] * 1e3
    x_focus_mm = focus_boundary_x * dx[0] * 1e3
    y0, y1 = extent_xy[2], extent_xy[3]
    z0, z1 = extent_xz[2], extent_xz[3]
    # Design boundary: red semi-transparent
    verts_design = np.array([[x_design_mm, y0, z0], [x_design_mm, y1, z0], [x_design_mm, y1, z1], [x_design_mm, y0, z1]])
    face_design = [verts_design]
    p_design = Poly3DCollection(face_design, alpha=0.25, facecolor='red', edgecolor='red', linewidths=1.5)
    col_d = ax_iso.add_collection3d(p_design)
    static_collections.add(col_d)
    # Focus boundary: cyan semi-transparent
    verts_focus = np.array([[x_focus_mm, y0, z0], [x_focus_mm, y1, z0], [x_focus_mm, y1, z1], [x_focus_mm, y0, z1]])
    face_focus = [verts_focus]
    p_focus = Poly3DCollection(face_focus, alpha=0.25, facecolor='cyan', edgecolor='cyan', linewidths=1.5)
    col_f = ax_iso.add_collection3d(p_focus)
    static_collections.add(col_f)

    # Initial frame: main + weak isosurfaces + signed (crest/trough)
    vol0 = vol_ds[0]
    p0_signed = pressure_ds[0]
    mesh_main0 = make_isosurface_mesh(vol0, level_main, facecolor='steelblue', alpha=0.45)
    if mesh_main0 is not None:
        ax_iso.add_collection3d(mesh_main0)
    mesh_weak0 = make_isosurface_mesh(vol0, level_weak, facecolor='lightcyan', alpha=0.35)
    if mesh_weak0 is not None:
        ax_iso.add_collection3d(mesh_weak0)
    for m in make_signed_isosurfaces(p0_signed, level_signed):
        ax_iso.add_collection3d(m)

    ax_iso.set_xlim(extent_xy[0], extent_xy[1])
    ax_iso.set_ylim(extent_xy[2], extent_xy[3])
    ax_iso.set_zlim(extent_xz[2], extent_xz[3])
    ax_iso.set_xlabel('x (mm)')
    ax_iso.set_ylabel('y (mm)')
    ax_iso.set_zlabel('z (mm)')
    ax_iso.set_title('3D: Red = crest, Blue = trough (wavefronts) | Light cyan = scattered field | Orange = cylinders')
    time_text_iso = fig_iso.text(0.5, 0.02, '', ha='center', fontsize=10,
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init_iso():
        return []

    def animate_iso(frame_idx):
        # Remove only dynamic collections, keep cylinders and boundary planes
        to_remove = [c for c in ax_iso.collections if c not in static_collections]
        for c in to_remove:
            c.remove()
        f = frames_isosurf[frame_idx]
        vol = vol_ds[f]
        p_signed = pressure_ds[f]
        # Main wave envelope (strong)
        mesh_main = make_isosurface_mesh(vol, level_main, facecolor='steelblue', alpha=0.45)
        if mesh_main is not None:
            ax_iso.add_collection3d(mesh_main)
        # Weaker level: shows scattered/diffracted field (bent wavefronts, sidelobes)
        mesh_weak = make_isosurface_mesh(vol, level_weak, facecolor='lightcyan', alpha=0.35)
        if mesh_weak is not None:
            ax_iso.add_collection3d(mesh_weak)
        # Signed: crest (red) and trough (blue) – wavefront shape; scattering = curved/broken surfaces
        for m in make_signed_isosurfaces(p_signed, level_signed):
            ax_iso.add_collection3d(m)
        ax_iso.scatter([max_x_mm], [max_y_mm], [max_z_mm], c='lime', s=300, marker='*',
                       edgecolors='black', linewidths=1.5, zorder=10)
        time_val = time_axis.to_array()[f] * 1e6
        time_text_iso.set_text(f'Time: {time_val:.2f} μs  |  Red/Blue = wavefronts (scattering bends them)  |  Light cyan = scattered field  |  Max: {results[0]["max_pressure"]:.4f} Pa')
        return []

    anim_iso = animation.FuncAnimation(fig_iso, animate_iso, init_func=init_iso,
                                      frames=len(frames_isosurf), interval=80, repeat=True)
    video_path_iso = f'../Results/continuous_wave_split_3d_spherical/config_1_3d_isosurface_animation.mp4'
    anim_iso.save(video_path_iso, writer=Writer(fps=15, bitrate=4000))
    print(f"  Saved 3D isosurface animation (with cylinders + design/focus): {video_path_iso}")
    plt.close(fig_iso)
except Exception as e:
    print(f"  Skipping 3D isosurface animation: {e}")

# Keep original single-slice animation as well (smaller file, quick view)
fig_anim, axes = plt.subplots(1, 2, figsize=(16, 7))
pressure_slice_series = pressure_time_series[:, view_x_start:view_x_end,
                                             view_y_start:view_y_end, mid_z_global]
sound_speed_slice = sound_speed_map[view_x_start:view_x_end,
                                    view_y_start:view_y_end, mid_z_global]
im1 = axes[0].imshow(sound_speed_slice.T, cmap='viridis', origin='lower',
                     extent=extent_xy, aspect='auto')
axes[0].set_title(f'Configuration 1: Sound Speed (XY slice)', fontsize=12)
axes[0].set_xlabel('x (mm)')
axes[0].set_ylabel('y (mm)')
axes[0].axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=2, linestyle=':', alpha=0.7)
axes[0].axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=2, linestyle=':', alpha=0.7)
plt.colorbar(im1, ax=axes[0], label='c (m/s)', fraction=0.046)
im2 = axes[1].imshow(pressure_slice_series[0].T, cmap='RdBu_r', origin='lower',
                     extent=extent_xy, vmin=-vmax_anim, vmax=vmax_anim, aspect='auto')
axes[1].set_xlabel('x (mm)')
axes[1].set_ylabel('y (mm)')
plt.colorbar(im2, ax=axes[1], label='Pressure (Pa)', fraction=0.046)
pt_slice = axes[1].scatter([max_x_mm], [max_y_mm], c='lime', s=200, marker='*', edgecolors='black',
                          linewidths=1, zorder=10, label='Max pressure')
time_text = axes[1].text(0.02, 0.98, '', transform=axes[1].transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def init():
    im2.set_data(pressure_slice_series[0].T)
    time_text.set_text('')
    return [im2, pt_slice, time_text]

def animate(frame):
    idx = min(frame, pressure_slice_series.shape[0] - 1)
    im2.set_data(pressure_slice_series[idx].T)
    time_val = time_axis.to_array()[min(frame, n_frames - 1)] * 1e6
    time_text.set_text(f'Time: {time_val:.2f} μs  |  Frame: {frame}/{n_frames}\nXY slice at z={mid_z_global}  |  Max: {results[0]["max_pressure"]:.4f} Pa at ({max_x_mm:.1f}, {max_y_mm:.1f}, {max_z_mm:.1f}) mm')
    return [im2, pt_slice, time_text]

anim = animation.FuncAnimation(fig_anim, animate, init_func=init,
                               frames=frames_to_use, interval=33, blit=True, repeat=True)
plt.suptitle(f'3D Simulation - XY Slice Only (Configuration 1)\n' +
             f'Max in focus: {results[0]["max_pressure"]:.4f} Pa', fontsize=14, fontweight='bold')
video_path = f'../Results/continuous_wave_split_3d_spherical/config_1_3d_animation_xy_slice.mp4'
anim.save(video_path, writer=Writer(fps=30, bitrate=3000))
print(f"  Saved XY-slice-only animation: {video_path}")
plt.close(fig_anim)

print("\n" + "="*70)
print("3D SIMULATION COMPLETE!")
print("="*70)
print("Results saved to: ../Results/continuous_wave_split_3d_spherical/")
print("="*70)
print("\nDone!")
