"""
Replot optimization results from a previous run of j_wave_true_autodiff_optimization.py.

Loads ../Results/autodiff_optimization/optimization_results.npz and recreates
the 2x3 summary figure (progress, gradient norms, trajectories, initial/optimized
pressure fields, summary).

Usage:
  python replot_autodiff_optimization_results.py [path_to_results.npz]

If no path is given, uses ../Results/autodiff_optimization/optimization_results.npz
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os
import sys

# Geometry and layout (must match the run that produced the npz)
N = (512, 512)
dx = (0.2e-3, 0.2e-3)
view_x_start = 128
view_x_end = 384
view_y_start = 128
view_y_end = 384
design_boundary_x = 280
focus_boundary_x = 300
target_focal_center = (320, 256)
focal_disk_radius = 10
cylinder_radius = 12.0

def main():
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = os.path.join(os.path.dirname(__file__),
                                    '..', 'Results', 'autodiff_optimization',
                                    'optimization_results.npz')

    if not os.path.isfile(results_path):
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading: {results_path}")
    data = np.load(results_path, allow_pickle=True)

    initial_positions = data['initial_positions']
    final_positions = data['final_positions']
    history_positions = data['optimization_history_positions']
    history_pressures = data['optimization_history_pressures']
    history_gradients = data['optimization_history_gradients']
    initial_pressure = float(data['initial_pressure'])
    optimized_pressure = float(data['optimized_pressure'])

    # Optional arrays (saved if optimization script was run with pressure views)
    pressure_view_init = data['pressure_view_init'] if 'pressure_view_init' in data else None
    pressure_view_opt = data['pressure_view_opt'] if 'pressure_view_opt' in data else None
    iteration_times = data['iteration_times'] if 'iteration_times' in data else np.array([])
    n_iterations = int(data['n_iterations']) if 'n_iterations' in data else len(history_pressures) - 1
    learning_rate = float(data['learning_rate']) if 'learning_rate' in data else 0.5
    momentum = float(data['momentum']) if 'momentum' in data else 0.9
    training_time_end = float(data['training_time_end']) if 'training_time_end' in data else 1.5e-4
    stabilization_time = float(data['stabilization_time']) if 'stabilization_time' in data else 1.0e-4

    improvement = (optimized_pressure / initial_pressure - 1) * 100
    extent = [view_x_start * dx[0] * 1e3, view_x_end * dx[0] * 1e3,
              view_y_start * dx[1] * 1e3, view_y_end * dx[1] * 1e3]
    focal_disk_mm = focal_disk_radius * dx[0] * 1e3

    # Build optimization_history for plotting (list of positions per iter)
    n_iters = history_positions.shape[0]
    optimization_history = {
        'positions': [history_positions[i] for i in range(n_iters)],
        'pressures': list(history_pressures),
        'gradients': [history_gradients[i] for i in range(min(len(history_gradients), n_iters - 1))],
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Optimization progress
    ax = axes[0, 0]
    ax.plot(optimization_history['pressures'], 'b-o', linewidth=2, markersize=6)
    ax.axhline(initial_pressure, color='r', linestyle='--', alpha=0.5, label='Initial RMS (full sim)')
    ax.axhline(optimized_pressure, color='g', linestyle='--', alpha=0.5, label='Optimized RMS (full sim)')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('RMS Pressure over Focal Disk (Pa)', fontsize=11)
    ax.set_title('Autodiff Optimization Progress (RMS Objective)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gradient norms
    ax = axes[0, 1]
    grad_norms = [float(np.linalg.norm(g)) for g in optimization_history['gradients']]
    ax.plot(grad_norms, 'r-o', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Gradient Norm', fontsize=11)
    ax.set_title('Gradient Magnitude (True Physics Gradients)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Cylinder trajectories
    ax = axes[0, 2]
    ax.set_xlim(view_x_start, view_x_end)
    ax.set_ylim(view_y_start, view_y_end)
    ax.set_aspect('equal')
    ax.set_xlabel('X (grid points)', fontsize=11)
    ax.set_ylabel('Y (grid points)', fontsize=11)
    ax.set_title('Cylinder Optimization Trajectory', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(design_boundary_x, color='red', linewidth=2, linestyle=':', alpha=0.5, label='Design')
    ax.axvline(focus_boundary_x, color='cyan', linewidth=2, linestyle=':', alpha=0.5, label='Focus')
    focus_rect = Rectangle((focus_boundary_x, view_y_start), view_x_end - focus_boundary_x,
                           view_y_end - view_y_start, fill=True, facecolor='cyan', alpha=0.1)
    ax.add_patch(focus_rect)
    from matplotlib.patches import Circle as PlotCircle
    focal_circle = PlotCircle((target_focal_center[0], target_focal_center[1]), focal_disk_radius,
                              fill=False, edgecolor='red', linewidth=2, linestyle='-',
                              label='Focal disk', zorder=10)
    ax.add_patch(focal_circle)
    ax.plot(target_focal_center[0], target_focal_center[1], 'r+', markersize=15,
            markeredgewidth=2, zorder=11, label='Disk center')
    colors = plt.cm.rainbow(np.linspace(0, 1, 3))
    for cyl_idx in range(3):
        traj_x = [optimization_history['positions'][i][cyl_idx, 0] for i in range(n_iters)]
        traj_y = [optimization_history['positions'][i][cyl_idx, 1] for i in range(n_iters)]
        ax.plot(traj_x, traj_y, 'o-', color=colors[cyl_idx], linewidth=2,
                markersize=4, alpha=0.7, label=f'Cyl {cyl_idx+1}')
        ax.plot(traj_x[0], traj_y[0], 's', color=colors[cyl_idx], markersize=12,
                markeredgecolor='black', markeredgewidth=2)
        ax.plot(traj_x[-1], traj_y[-1], 'o', color=colors[cyl_idx], markersize=12,
                markeredgecolor='black', markeredgewidth=2)
    ax.legend(fontsize=9, loc='upper left')

    # Plot 4: Initial pressure field (or placeholder)
    vmax_viz = None
    if pressure_view_init is not None or pressure_view_opt is not None:
        vmax_viz = 0
        if pressure_view_init is not None:
            vmax_viz = max(vmax_viz, np.max(np.abs(pressure_view_init)))
        if pressure_view_opt is not None:
            vmax_viz = max(vmax_viz, np.max(np.abs(pressure_view_opt)))

    ax = axes[1, 0]
    if pressure_view_init is not None:
        im = ax.imshow(pressure_view_init.T, cmap='hot', origin='lower', extent=extent,
                       vmin=0, vmax=vmax_viz or np.max(np.abs(pressure_view_init)))
        for i in range(3):
            x, y = initial_positions[i, 0], initial_positions[i, 1]
            circle = Circle((x*dx[0]*1e3, y*dx[1]*1e3), cylinder_radius*dx[0]*1e3,
                            fill=False, edgecolor='white', linewidth=2, linestyle='--')
            ax.add_patch(circle)
        ax.set_title(f'Initial Configuration\nRMS Pressure: {initial_pressure:.4f} Pa', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Pressure field not saved in npz.\nRe-run j_wave_true_autodiff_optimization.py\nto save pressure views.',
                transform=ax.transAxes, ha='center', va='center', fontsize=11)
        ax.set_title('Initial Configuration', fontsize=12, fontweight='bold')
    ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
    ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
    focal_circle_patch = Circle((target_focal_center[0]*dx[0]*1e3, target_focal_center[1]*dx[1]*1e3),
                                 focal_disk_mm, fill=False, edgecolor='lime', linewidth=2, zorder=10)
    ax.add_patch(focal_circle_patch)
    ax.set_xlabel('x (mm)', fontsize=11)
    ax.set_ylabel('y (mm)', fontsize=11)
    if pressure_view_init is not None:
        plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)

    # Plot 5: Optimized pressure field (or placeholder)
    ax = axes[1, 1]
    if pressure_view_opt is not None:
        im = ax.imshow(pressure_view_opt.T, cmap='hot', origin='lower', extent=extent,
                       vmin=0, vmax=vmax_viz or np.max(np.abs(pressure_view_opt)))
        for i in range(3):
            x, y = final_positions[i, 0], final_positions[i, 1]
            circle = Circle((x*dx[0]*1e3, y*dx[1]*1e3), cylinder_radius*dx[0]*1e3,
                            fill=False, edgecolor='white', linewidth=2, linestyle='--')
            ax.add_patch(circle)
        ax.set_title(f'Optimized Configuration\nRMS: {optimized_pressure:.4f} Pa (+{improvement:.1f}%)',
                    fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='|P| (Pa)', fraction=0.046)
    else:
        ax.text(0.5, 0.5, 'Pressure field not saved in npz.',
                transform=ax.transAxes, ha='center', va='center', fontsize=11)
        ax.set_title(f'Optimized Configuration\nRMS: {optimized_pressure:.4f} Pa (+{improvement:.1f}%)',
                    fontsize=12, fontweight='bold')
    ax.axvline(design_boundary_x*dx[0]*1e3, color='red', linewidth=1.5, linestyle=':', alpha=0.5)
    ax.axvline(focus_boundary_x*dx[0]*1e3, color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)
    focal_circle_patch5 = Circle((target_focal_center[0]*dx[0]*1e3, target_focal_center[1]*dx[1]*1e3),
                                  focal_disk_mm, fill=False, edgecolor='lime', linewidth=2, zorder=10)
    ax.add_patch(focal_circle_patch5)
    ax.set_xlabel('x (mm)', fontsize=11)
    ax.set_ylabel('y (mm)', fontsize=11)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    avg_time = float(np.mean(iteration_times)) if len(iteration_times) > 0 else 0
    summary_text = f"""
TRUE AUTODIFF OPTIMIZATION SUMMARY (replot)

Method:
  • JAX autodiff through j-wave solver
  • Physics-informed gradients
  • Objective: RMS over focal disk

Simulation:
  • Training time: {training_time_end*1e6:.0f} μs
  • Stabilization: {stabilization_time*1e6:.0f} μs

Optimization:
  • Iterations: {n_iterations}
  • Learning rate: {learning_rate}
  • Momentum: {momentum}
  • Avg time/iter: {avg_time:.1f} s

Results (Full 300μs Validation):
  • Initial RMS: {initial_pressure:.6f} Pa
  • Optimized RMS: {optimized_pressure:.6f} Pa
  • Improvement: {improvement:.1f}%
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('True Autodiff Optimization – Results (replot)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(results_path)
    output_fig = os.path.join(out_dir, 'autodiff_optimization_results_replot.png')
    plt.savefig(output_fig, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_fig}")
    plt.show()

if __name__ == '__main__':
    main()
