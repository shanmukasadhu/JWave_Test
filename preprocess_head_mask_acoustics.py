"""
Minimal body-only cleanup for acoustic maps.

Goal:
  Keep only the main body/head component from existing acoustic maps and
  force everything else to air values.

Inputs expected in --input-dir:
  - sound_speed_xyz.npy
  - density_xyz.npy
  - attenuation_db_cm_mhz_xyz.npy

Usage:
  python3 Code/preprocess_head_mask_acoustics.py \
    --input-dir "Results/jwave_brain_forward_64/ACRIN-FMISO-Brain-001"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


AIR_C = 343.0
AIR_RHO = 1.2
AIR_ATT = 1.0


def _load_maps(input_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_path = input_dir / "sound_speed_xyz.npy"
    rho_path = input_dir / "density_xyz.npy"
    att_path = input_dir / "attenuation_db_cm_mhz_xyz.npy"
    if not (c_path.exists() and rho_path.exists() and att_path.exists()):
        raise FileNotFoundError(
            "Missing required files in input-dir. Need: "
            "sound_speed_xyz.npy, density_xyz.npy, attenuation_db_cm_mhz_xyz.npy"
        )
    c = np.load(c_path).astype(np.float32)
    rho = np.load(rho_path).astype(np.float32)
    att = np.load(att_path).astype(np.float32)
    if c.ndim != 3 or rho.ndim != 3 or att.ndim != 3:
        raise ValueError("All input maps must be 3D.")
    if c.shape != rho.shape or c.shape != att.shape:
        raise ValueError(f"Shape mismatch: c={c.shape}, rho={rho.shape}, att={att.shape}")
    return c, rho, att


def largest_body_mask_from_speed(c_xyz: np.ndarray, body_speed_threshold: float = 1200.0) -> np.ndarray:
    """
    Simple rule:
      1) body candidate = c > threshold
      2) keep only largest connected component
    """
    try:
        from scipy.ndimage import label
    except ImportError as exc:
        raise ImportError("scipy is required. Install with: pip install scipy") from exc

    candidate = c_xyz > body_speed_threshold
    labeled, n = label(candidate)
    if n == 0:
        # Fallback: nothing detected, return all-False mask.
        return np.zeros_like(candidate, dtype=bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labeled == keep


def apply_air_outside_mask(
    c_xyz: np.ndarray, rho_xyz: np.ndarray, att_xyz: np.ndarray, body_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_out = c_xyz.copy()
    rho_out = rho_xyz.copy()
    att_out = att_xyz.copy()

    outside = ~body_mask
    c_out[outside] = AIR_C
    rho_out[outside] = AIR_RHO
    att_out[outside] = AIR_ATT
    return c_out, rho_out, att_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only main body/head component and force outside to air.")
    parser.add_argument("--input-dir", required=True, help="Directory with acoustic npy maps.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <input-dir>/body_only_air)",
    )
    parser.add_argument(
        "--body-speed-threshold",
        type=float,
        default=1200.0,
        help="Speed threshold for body candidate mask (default: 1200 m/s).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (input_dir / "body_only_air")
    out_dir.mkdir(parents=True, exist_ok=True)

    c_xyz, rho_xyz, att_xyz = _load_maps(input_dir)
    body_mask = largest_body_mask_from_speed(c_xyz, body_speed_threshold=float(args.body_speed_threshold))
    c_out, rho_out, att_out = apply_air_outside_mask(c_xyz, rho_xyz, att_xyz, body_mask)

    np.save(out_dir / "sound_speed_xyz.npy", c_out.astype(np.float32))
    np.save(out_dir / "density_xyz.npy", rho_out.astype(np.float32))
    np.save(out_dir / "attenuation_db_cm_mhz_xyz.npy", att_out.astype(np.float32))
    np.save(out_dir / "body_mask_xyz.npy", body_mask.astype(np.uint8))

    summary = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "body_speed_threshold": float(args.body_speed_threshold),
        "body_fraction": float(np.mean(body_mask)),
        "air_values": {"c": AIR_C, "rho": AIR_RHO, "att": AIR_ATT},
    }
    (out_dir / "body_only_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] Saved body-only maps to: {out_dir}")
    print(f"[info] Body fraction: {summary['body_fraction']:.4f}")


if __name__ == "__main__":
    main()

