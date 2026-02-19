"""
Run j_wave_true_autodiff_optimization.py overnight with a grid of learning rates
and iteration counts. Each run writes to its own subfolder so results are not overwritten.

Usage (from the Code directory):
  python run_autodiff_overnight.py

Runs:
  learning_rates = [150, 175, 200, 225, 250, 275, 300]
  n_iterations   = [20, 30, 40, 50]

Total: 7 × 4 = 28 runs. Each run is independent (no --resume).
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Grid: learning rates and iteration counts
LEARNING_RATES = [150, 175, 200, 225, 250, 275, 300]
N_ITERATIONS_LIST = [20, 30, 40, 50]

# Path to the optimization script (same directory as this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZATION_SCRIPT = os.path.join(SCRIPT_DIR, 'j_wave_true_autodiff_optimization.py')
RESULTS_BASE = os.path.join(SCRIPT_DIR, '..', 'Results', 'autodiff_optimization')
LOG_FILE = os.path.join(SCRIPT_DIR, '..', 'Results', 'autodiff_optimization', 'overnight_runs.log')
SUMMARY_FILE = os.path.join(SCRIPT_DIR, '..', 'Results', 'autodiff_optimization', 'overnight_summary.txt')


def log(msg, also_print=True):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def main():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    log("=" * 60)
    log("OVERNIGHT GRID: autodiff optimization")
    log(f"  Learning rates: {LEARNING_RATES}")
    log(f"  Iterations:     {N_ITERATIONS_LIST}")
    log(f"  Total runs:     {len(LEARNING_RATES) * len(N_ITERATIONS_LIST)}")
    log("=" * 60)

    if not os.path.isfile(OPTIMIZATION_SCRIPT):
        log(f"ERROR: Optimization script not found: {OPTIMIZATION_SCRIPT}")
        sys.exit(1)

    results_summary = []
    total_start = time.time()

    for lr in LEARNING_RATES:
        for n_iter in N_ITERATIONS_LIST:
            run_id = f"lr{lr}_iter{n_iter}"
            run_dir = os.path.join(RESULTS_BASE, run_id)
            os.makedirs(run_dir, exist_ok=True)

            env = os.environ.copy()
            env['AUTODIFF_LR'] = str(lr)
            env['AUTODIFF_N_ITER'] = str(n_iter)
            env['AUTODIFF_RUN_ID'] = run_id

            log(f"Starting run: {run_id}")
            start = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, OPTIMIZATION_SCRIPT],
                    cwd=SCRIPT_DIR,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=None,
                )
                elapsed = time.time() - start
                if proc.returncode == 0:
                    log(f"  Completed {run_id} in {elapsed/60:.1f} min")
                    # Try to read final pressure from saved npz for summary
                    npz_path = os.path.join(run_dir, 'optimization_results.npz')
                    if os.path.isfile(npz_path):
                        try:
                            import numpy as np
                            d = np.load(npz_path, allow_pickle=True)
                            opt_p = float(d['optimized_pressure'])
                            init_p = float(d['initial_pressure'])
                            imp = (opt_p / init_p - 1) * 100
                            results_summary.append((run_id, opt_p, init_p, imp, elapsed))
                        except Exception:
                            results_summary.append((run_id, None, None, None, elapsed))
                    else:
                        results_summary.append((run_id, None, None, None, elapsed))
                else:
                    log(f"  FAILED {run_id} (exit {proc.returncode}) in {elapsed/60:.1f} min")
                    log(f"    stdout tail: {proc.stdout[-500:] if proc.stdout else 'none'}")
                    log(f"    stderr tail: {proc.stderr[-500:] if proc.stderr else 'none'}")
                    results_summary.append((run_id, None, None, None, elapsed))
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                log(f"  TIMEOUT {run_id} after {elapsed/60:.1f} min")
                results_summary.append((run_id, None, None, None, elapsed))
            except Exception as e:
                elapsed = time.time() - start
                log(f"  ERROR {run_id}: {e}")
                results_summary.append((run_id, None, None, None, elapsed))

    total_elapsed = time.time() - total_start
    log("=" * 60)
    log(f"All runs finished. Total time: {total_elapsed/3600:.2f} hours")
    log("=" * 60)

    # Write summary
    with open(SUMMARY_FILE, 'w') as f:
        f.write("Run ID\tOptimized RMS (Pa)\tInitial RMS (Pa)\tImprovement (%)\tTime (min)\n")
        for run_id, opt_p, init_p, imp, elapsed in results_summary:
            opt_s = f"{opt_p:.6f}" if opt_p is not None else "—"
            init_s = f"{init_p:.6f}" if init_p is not None else "—"
            imp_s = f"{imp:.1f}" if imp is not None else "—"
            f.write(f"{run_id}\t{opt_s}\t{init_s}\t{imp_s}\t{elapsed/60:.1f}\n")
    log(f"Summary written to: {SUMMARY_FILE}")

    # Best run by improvement
    valid_runs = [s for s in results_summary if s[3] is not None]
    if valid_runs:
        best = max(valid_runs, key=lambda s: s[3])
        log(f"Best improvement: {best[0]} ({best[3]:.1f}%)")
    else:
        log("No runs produced optimized_pressure in npz.")

    log("Done.")


if __name__ == '__main__':
    main()
