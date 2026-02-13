"""
Acoustic wave scattering simulation using jwave
- 3 cylinders in water
- Plane wave propagating from left to right
- One focus point
- 5 different cylinder configurations
- Large simulation domain with viewing window to avoid boundary artifacts
"""

import numpy as np
from jax import numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import jwave components
from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis

# Simulation parameters - LARGE domain to push boundaries far away
N = (512, 512)  # Large grid size
dx = (0.2e-3, 0.2e-3)  # Grid spacing in meters (0.2 mm)
domain = Domain(N, dx)

# Define viewing window - only show this region (boundaries are outside)
# This keeps the display clean while simulating on a larger domain
view_x_start = 128
view_x_end = 384
view_y_start = 128
view_y_end = 384

# Medium properties
c_water = 1500.0  # Speed of sound in water (m/s)
c_cylinder = 2500.0  # Speed of sound in cylinders (m/s) - e.g., plastic/bone
rho_water = 1000.0  # Density of water (kg/m^3)
rho_cylinder = 1200.0  # Density of cylinders (kg/m^3)

# Time axis
time_axis = TimeAxis.from_medium(
    Medium(domain=domain, sound_speed=c_water),
    cfl=0.3,
    t_end=5.0e-05  # 50 microseconds
)

def create_cylinder_mask(N, center, radius):
    """Create a circular mask for a cylinder"""
    x = jnp.arange(N[0])
    y = jnp.arange(N[1])
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    dist = jnp.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist <= radius

def create_medium_with_cylinders(domain, cylinder_configs):
    """
    Create a heterogeneous medium with cylinders of varying radii
    
    Parameters:
    - domain: jwave Domain object
    - cylinder_configs: list of (x, y, radius) tuples for cylinder centers and sizes
    """
    # Initialize with water properties
    sound_speed = jnp.ones(N) * c_water
    density = jnp.ones(N) * rho_water
    
    # Add cylinders with individual radii
    for config in cylinder_configs:
        x, y, radius = config
        mask = create_cylinder_mask(N, (x, y), radius)
        sound_speed = jnp.where(mask, c_cylinder, sound_speed)
        density = jnp.where(mask, rho_cylinder, density)
    
    # Add third dimension for jwave
    sound_speed = jnp.expand_dims(sound_speed, -1)
    density = jnp.expand_dims(density, -1)
    
    # Large PML to absorb boundary reflections
    return Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=60)

def create_focused_wave_source(N, focus_point=(256, 320), amplitude=1.0):
    """
    Create initial pressure distribution that focuses at a point
    Using a simple plane wave from the left
    """
    p0 = jnp.zeros(N)
    
    # Simple plane wave from left edge
    # Start wave deeper in from the boundary
    left_edge = 80
    for i in range(N[0]):
        if i < left_edge:
            p0 = p0.at[i, :].set(amplitude)
    
    p0 = jnp.expand_dims(p0, -1)
    return FourierSeries(p0, domain)

# Define 5 different cylinder configurations with varying radii
# Each configuration is a list of tuples: (x_position, y_position, radius)
# Viewing window is [128:384, 128:384], so center is around (256, 256)
configurations = [
    # Configuration 1: Horizontal line with increasing radii
    [(200, 200, 8), (200, 256, 12), (200, 312, 16)],
    
    # Configuration 2: Vertical line with decreasing radii
    [(200, 256, 16), (240, 256, 12), (280, 256, 8)],
    
    # Configuration 3: Diagonal with same radii
    [(200, 200, 10), (240, 256, 10), (280, 312, 10)],
    
    # Configuration 4: Triangle with varied radii
    [(210, 256, 14), (260, 220, 10), (260, 292, 10)],
    
    # Configuration 5: Scattered with random radii
    [(210, 220, 12), (260, 280, 8), (230, 310, 15)]
]

# Focus point (in full grid coordinates)
focus_point = (256, 340)

# Compile the simulation
@jit
def run_simulation(medium, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0)

# Run simulations for all configurations
print("Running jwave simulations for 5 cylinder configurations...")
print(f"Full simulation domain: {N[0]} x {N[1]} grid points")
print(f"Viewing window: [{view_x_start}:{view_x_end}, {view_y_start}:{view_y_end}]")
print(f"This hides boundary reflections while showing clean wave propagation\n")

results = []

for i, config in enumerate(configurations):
    print(f"Configuration {i+1}/5: Cylinder positions = {config}")
    import time;start_time = time.time()
    
    # Create medium with cylinders
    medium = create_medium_with_cylinders(domain, config)
    
    # Create source (plane wave from left)
    p0 = create_focused_wave_source(N, focus_point=focus_point, amplitude=1.0)
    
    # Run simulation
    pressure_field = run_simulation(medium, p0)
    print(f"Simulatio {i}: {time.time() - start_time}")
    
    # Extract pressure - pressure_field is the full field on grid
    pressure_on_grid = np.array(pressure_field.on_grid)
    
    # Store all time steps for video
    if pressure_on_grid.ndim == 4:
        # Shape is (time, x, y, 1)
        pressure_all_time = pressure_on_grid[:, :, :, 0]
    elif pressure_on_grid.ndim == 3:
        # Shape is (x, y, 1)
        pressure_all_time = pressure_on_grid[:, :, 0]
    else:
        pressure_all_time = pressure_on_grid
    
    # Get snapshot at 80% through simulation
    time_idx = int(len(time_axis.to_array()) * 0.8)
    if pressure_all_time.ndim == 3:
        pressure_at_time = pressure_all_time[time_idx]
    else:
        pressure_at_time = pressure_all_time
    
    # Extract sound speed
    sound_speed_field = np.array(medium.sound_speed.on_grid)
    if sound_speed_field.ndim == 3:
        sound_speed_field = sound_speed_field[:, :, 0]
    
    results.append({
        'config': config,
        'pressure': pressure_at_time,
        'pressure_all_time': pressure_all_time,
        'sound_speed': sound_speed_field,
        'medium': medium
    })

# Create videos for each configuration
print("\nCreating videos for each configuration...")
import matplotlib.animation as animation

for i, result in enumerate(results):
    print(f"Creating video {i+1}/5...")
    
    # Create figure for animation
    fig_anim, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get data - ONLY THE VIEWING WINDOW
    pressure_time_series = result['pressure_all_time']
    sound_speed_map = result['sound_speed']
    
    # Extract viewing window
    if pressure_time_series.ndim == 3:
        n_frames = pressure_time_series.shape[0]
        pressure_view = pressure_time_series[:, view_x_start:view_x_end, view_y_start:view_y_end]
    else:
        n_frames = 1
        pressure_view = pressure_time_series[view_x_start:view_x_end, view_y_start:view_y_end]
        pressure_view = pressure_view[np.newaxis, :, :]
    
    sound_speed_view = sound_speed_map[view_x_start:view_x_end, view_y_start:view_y_end]
    
    # Set up color limits
    vmax = np.max(np.abs(pressure_view))
    
    # Physical extent of viewing window
    extent = [
        view_x_start * dx[0] * 1e3,
        view_x_end * dx[0] * 1e3,
        view_y_start * dx[1] * 1e3,
        view_y_end * dx[1] * 1e3
    ]
    
    # Left subplot: Sound speed (cylinder positions) - static
    im1 = ax1.imshow(sound_speed_view.T, cmap='viridis', origin='lower', extent=extent)
    ax1.set_title(f'Configuration {i+1}: Cylinder Positions', fontsize=12)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.plot(focus_point[0]*dx[0]*1e3, focus_point[1]*dx[1]*1e3, 
             'r*', markersize=15, label='Focus')
    
    # Draw circles showing cylinder boundaries
    from matplotlib.patches import Circle
    for x, y, radius in result['config']:
        circle = Circle((x*dx[0]*1e3, y*dx[1]*1e3), radius*dx[0]*1e3, 
                       fill=False, edgecolor='white', linewidth=2, linestyle='--')
        ax1.add_patch(circle)
    
    ax1.legend(fontsize=10)
    plt.colorbar(im1, ax=ax1, label='Sound speed (m/s)', fraction=0.046)
    
    # Right subplot: Pressure field - animated
    im2 = ax2.imshow(pressure_view[0].T, cmap='RdBu_r', origin='lower',
                     extent=extent, vmin=-vmax, vmax=vmax)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    plt.colorbar(im2, ax=ax2, label='Pressure (Pa)', fraction=0.046)
    
    time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        im2.set_data(pressure_view[0].T)
        time_text.set_text('')
        return [im2, time_text]
    
    def animate(frame):
        im2.set_data(pressure_view[frame].T)
        time_val = time_axis.to_array()[frame] * 1e6  # Convert to microseconds
        time_text.set_text(f'Time: {time_val:.2f} μs\nFrame: {frame}/{n_frames}')
        ax2.set_title(f'Pressure Field (Wave Propagation)', fontsize=12)
        
        # Find and mark the maximum pressure point (focal point) in current frame
        pressure_magnitude = np.abs(pressure_view[frame])
        max_idx = np.unravel_index(np.argmax(pressure_magnitude), pressure_magnitude.shape)
        
        # Convert to physical coordinates
        max_x = (view_x_start + max_idx[0]) * dx[0] * 1e3
        max_y = (view_y_start + max_idx[1]) * dx[1] * 1e3
        max_pressure = pressure_magnitude[max_idx]
        
        # Clear previous markers and add new one
        # Remove old focal point markers (keep only original focus point and colorbar)
        while len(ax2.collections) > 1:  # Keep colorbar
            ax2.collections[-1].remove()
        while len(ax2.lines) > 0:
            ax2.lines[-1].remove()
        
        # Add new focal point marker
        ax2.plot(max_x, max_y, 'yo', markersize=12, markeredgecolor='black', 
                markeredgewidth=2, label=f'Max P: {max_pressure:.3f} Pa')
        ax2.legend(fontsize=9, loc='upper right')
        
        return [im2, time_text]
    
    # Create animation
    # Skip frames to make video reasonable length
    frame_skip = max(1, n_frames // 200)  # Limit to ~200 frames
    frames_to_use = range(0, n_frames, frame_skip)
    
    anim = animation.FuncAnimation(fig_anim, animate, init_func=init,
                                   frames=frames_to_use, interval=50, 
                                   blit=True, repeat=True)
    
    # Save video
    video_path = f'config_{i+1}_wave_propagation.mp4'
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, bitrate=1800)
    
    plt.suptitle(f'Acoustic Wave Scattering - Configuration {i+1}\n' +
                 f'Cylinder positions: {result["config"]}',
                 fontsize=14, fontweight='bold')
    
    anim.save(video_path, writer=writer)
    print(f"  Saved video to {video_path}")
    plt.close(fig_anim)

print("\nAll videos created successfully!")

# Visualization - static comparison
print("Creating static comparison visualization...")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)

for i, result in enumerate(results):
    # Extract viewing window for all plots
    pressure_data = result['pressure'][view_x_start:view_x_end, view_y_start:view_y_end]
    sound_speed_data = result['sound_speed'][view_x_start:view_x_end, view_y_start:view_y_end]
    
    extent = [
        view_x_start * dx[0] * 1e3,
        view_x_end * dx[0] * 1e3,
        view_y_start * dx[1] * 1e3,
        view_y_end * dx[1] * 1e3
    ]
    
    # Sound speed map (shows cylinder positions)
    ax1 = fig.add_subplot(gs[0, i])
    im1 = ax1.imshow(sound_speed_data.T, cmap='viridis', origin='lower', extent=extent)
    ax1.set_title(f'Config {i+1}: Cylinders', fontsize=10)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    plt.colorbar(im1, ax=ax1, label='Sound speed (m/s)', fraction=0.046)
    
    # Draw circles showing cylinder boundaries
    from matplotlib.patches import Circle
    for x, y, radius in result['config']:
        circle = Circle((x*dx[0]*1e3, y*dx[1]*1e3), radius*dx[0]*1e3, 
                       fill=False, edgecolor='white', linewidth=1.5, linestyle='--')
        ax1.add_patch(circle)
    
    # Mark focus point
    ax1.plot(focus_point[0]*dx[0]*1e3, focus_point[1]*dx[1]*1e3, 
             'r*', markersize=10, label='Focus')
    ax1.legend(fontsize=8)
    
    # Pressure distribution
    ax2 = fig.add_subplot(gs[1, i])
    vmax = np.max(np.abs(pressure_data))
    im2 = ax2.imshow(pressure_data.T, cmap='RdBu_r', origin='lower',
                     extent=extent, vmin=-vmax, vmax=vmax)
    ax2.set_title(f'Pressure Field', fontsize=10)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    plt.colorbar(im2, ax=ax2, label='Pressure (Pa)', fraction=0.046)
    
    # Find and mark maximum pressure point (focal point)
    pressure_magnitude = np.abs(pressure_data)
    max_idx = np.unravel_index(np.argmax(pressure_magnitude), pressure_magnitude.shape)
    max_x = (view_x_start + max_idx[0]) * dx[0] * 1e3
    max_y = (view_y_start + max_idx[1]) * dx[1] * 1e3
    ax2.plot(max_x, max_y, 'yo', markersize=10, markeredgecolor='black', 
            markeredgewidth=2, label='Focal Point')
    ax2.legend(fontsize=8)
    
    # Pressure magnitude
    ax3 = fig.add_subplot(gs[2, i])
    im3 = ax3.imshow(np.abs(pressure_data).T, cmap='hot', origin='lower', extent=extent)
    ax3.set_title(f'|Pressure|', fontsize=10)
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, label='|Pressure| (Pa)', fraction=0.046)
    
    # Mark focal point on magnitude plot too
    ax3.plot(max_x, max_y, 'co', markersize=10, markeredgecolor='black', 
            markeredgewidth=2, label='Focal Point')
    ax3.legend(fontsize=8)

plt.suptitle('Acoustic Wave Scattering: 3 Cylinders in Water with Different Configurations\n' + 
             'Plane wave propagating left to right (boundaries hidden)', 
             fontsize=14, fontweight='bold')

# Save figure
output_path = 'jwave_cylinder_scattering.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {output_path}")

# Print summary
print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)
print(f"Full grid size: {N[0]} x {N[1]} points")
print(f"Viewing window: [{view_x_start}:{view_x_end}, {view_y_start}:{view_y_end}]")
print(f"View size: {view_x_end-view_x_start} x {view_y_end-view_y_start} points")
print(f"Physical view size: {(view_x_end-view_x_start)*dx[0]*1e3:.1f} mm x {(view_y_end-view_y_start)*dx[1]*1e3:.1f} mm")
print(f"Grid spacing: {dx[0]*1e6:.1f} μm")
print(f"Water sound speed: {c_water} m/s")
print(f"Cylinder sound speed: {c_cylinder} m/s")
print(f"Simulation time: {time_axis.to_array()[-1]*1e6:.2f} μs")
print(f"Number of time steps: {len(time_axis.to_array())}")
print(f"Focus point: ({focus_point[0]}, {focus_point[1]})")
print(f"PML thickness: 60 grid points")
print("\nConfigurations (x, y, radius):")
for i, config in enumerate(configurations):
    print(f"  {i+1}. {config}")
    radii = [c[2] for c in config]
    print(f"      Radii in mm: [{radii[0]*dx[0]*1e3:.2f}, {radii[1]*dx[0]*1e3:.2f}, {radii[2]*dx[0]*1e3:.2f}]")

print("\nFocal Point Analysis (Maximum Pressure Locations):")
for i, result in enumerate(results):
    pressure_data = result['pressure'][view_x_start:view_x_end, view_y_start:view_y_end]
    pressure_magnitude = np.abs(pressure_data)
    max_idx = np.unravel_index(np.argmax(pressure_magnitude), pressure_magnitude.shape)
    max_x = (view_x_start + max_idx[0]) * dx[0] * 1e3
    max_y = (view_y_start + max_idx[1]) * dx[1] * 1e3
    max_pressure = pressure_magnitude[max_idx]
    print(f"  Config {i+1}: Position ({max_x:.2f}, {max_y:.2f}) mm, Pressure: {max_pressure:.4f} Pa")
print("="*60)

plt.show()