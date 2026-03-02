"""
Batch download + plot top ranked TCIA brain/skull CT candidates.

This script:
1) Queries TCIA CT series candidates (head/brain-skull prioritized).
2) Picks top N candidates.
3) Runs brain_ct_to_jwave_acoustics.py for each candidate SeriesInstanceUID.
4) Runs plot_brain_ct_jwave_maps.py on each downloaded output.
5) Writes a summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _contains_any(text: str, keywords) -> bool:
    t = text.upper()
    return any(k in t for k in keywords)


def _series_text_blob(row) -> str:
    fields = [
        "BodyPartExamined",
        "SeriesDescription",
        "StudyDescription",
        "ProtocolName",
        "Collection",
        "Modality",
    ]
    vals = []
    for f in fields:
        if f in row.index:
            vals.append(str(row[f]))
    return " | ".join(vals).upper()


def _rank_series(series_df, bodypart: str, require_head_keywords: bool):
    head_kw = ["HEAD", "BRAIN", "SKULL", "CRANI", "NEURO", "CEREB"]
    chest_kw = ["CHEST", "LUNG", "THORAX", "ABDOMEN", "PELVIS", "CARDIAC"]

    filtered = series_df
    if require_head_keywords:
        keep_idx = []
        for idx, row in filtered.iterrows():
            blob = _series_text_blob(row)
            if _contains_any(blob, head_kw) and (not _contains_any(blob, chest_kw)):
                keep_idx.append(idx)
        if keep_idx:
            filtered = filtered.loc[keep_idx]
        else:
            return []

    ranked = []
    for idx, row in filtered.iterrows():
        blob = _series_text_blob(row)
        score = 0.0
        if _contains_any(blob, head_kw):
            score += 5.0
        if _contains_any(blob, chest_kw):
            score -= 10.0
        if bodypart and _contains_any(blob, [bodypart.upper()]):
            score += 3.0
        img_count = (
            float(row["ImageCount"])
            if "ImageCount" in row.index and str(row["ImageCount"]) != "nan"
            else 0.0
        )
        score += 0.001 * img_count
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    return ranked


def fetch_candidate_dataframe(input_spec: str, patient_id: str, bodypart: str):
    spec = input_spec.strip()
    if not (spec.startswith("tcia:") or spec.startswith("tcia://")):
        raise ValueError("Expected input like tcia:auto-head-ct or tcia:<collection>")
    collection = spec.split(":", 1)[1].replace("//", "").strip()
    auto_head_mode = collection.lower() in {"auto", "auto-head-ct", "head-ct", "ct-head"}

    try:
        from tcia_utils import nbia as tcia_nbia  # type: ignore
    except ImportError as exc:
        raise ImportError("Please install tcia_utils: pip install tcia_utils") from exc

    if auto_head_mode:
        df = tcia_nbia.getSeries(
            modality="CT",
            bodyPart=bodypart or "HEAD",
            patientId=patient_id or "",
            format="df",
        )
        if df is None or len(df) == 0:
            df = tcia_nbia.getSeries(modality="CT", patientId=patient_id or "", format="df")
    else:
        df = tcia_nbia.getSeries(
            collection=collection,
            modality="CT",
            patientId=patient_id or "",
            format="df",
        )
        if df is None or len(df) == 0:
            df = tcia_nbia.getSeries(
                collection=collection,
                patientId=patient_id or "",
                format="df",
            )
    if df is None or len(df) == 0:
        raise ValueError("No TCIA CT candidates found for the given query.")
    return df


def run_cmd(cmd: List[str], log_path: Path) -> Tuple[bool, int]:
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode == 0, proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Download and plot top-N TCIA brain/skull CT candidates.")
    parser.add_argument("--input", default="tcia:auto-head-ct", help="TCIA source (default: tcia:auto-head-ct)")
    parser.add_argument("--out-root", default="Results/brain_ct_candidates_topN", help="Root output directory.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of candidates to process.")
    parser.add_argument("--tcia-bodypart", default="HEAD", help="Body part hint (default: HEAD).")
    parser.add_argument("--tcia-patient-id", default="", help="Optional TCIA PatientID filter.")
    parser.add_argument("--tcia-require-head-keywords", type=int, default=1, choices=[0, 1])
    parser.add_argument("--target-shape", default=None, help="Optional Nx,Ny,Nz to pass through.")
    parser.add_argument("--resample-order", type=int, default=1, choices=[0, 1, 3])
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    code_dir = Path(__file__).resolve().parent
    convert_script = code_dir / "brain_ct_to_jwave_acoustics.py"
    plot_script = code_dir / "plot_brain_ct_jwave_maps.py"

    print("[info] Fetching TCIA candidates...")
    df = fetch_candidate_dataframe(args.input, args.tcia_patient_id, args.tcia_bodypart)
    ranked = _rank_series(df, args.tcia_bodypart, bool(args.tcia_require_head_keywords))
    if not ranked:
        raise ValueError("No candidates after ranking/filtering.")

    top = ranked[: min(args.top_n, len(ranked))]
    print(f"[info] Processing top {len(top)} candidates")

    summary_rows = []
    for i, (score, idx) in enumerate(top, start=1):
        row = df.loc[idx]
        uid = str(row.get("SeriesInstanceUID", ""))
        collection = str(row.get("Collection", ""))
        patient = str(row.get("PatientID", ""))
        img_count = str(row.get("ImageCount", ""))
        safe_uid = uid.replace(".", "_")
        cand_dir = out_root / f"{i:02d}_{safe_uid}"
        cand_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(top)}] UID={uid} score={score:.3f} imgs={img_count}")

        convert_cmd = [
            sys.executable,
            str(convert_script),
            "--input",
            args.input,
            "--tcia-series-uid",
            uid,
            "--out",
            str(cand_dir),
            "--tcia-bodypart",
            args.tcia_bodypart,
            "--tcia-require-head-keywords",
            str(args.tcia_require_head_keywords),
            "--resample-order",
            str(args.resample_order),
        ]
        if args.tcia_patient_id:
            convert_cmd.extend(["--tcia-patient-id", args.tcia_patient_id])
        if args.target_shape:
            convert_cmd.extend(["--target-shape", args.target_shape])

        ok_convert, rc_convert = run_cmd(convert_cmd, cand_dir / "convert.log")
        ok_plot = False
        rc_plot = -1
        if ok_convert:
            plot_cmd = [
                sys.executable,
                str(plot_script),
                "--input-dir",
                str(cand_dir),
            ]
            ok_plot, rc_plot = run_cmd(plot_cmd, cand_dir / "plot.log")

        summary_rows.append(
            {
                "rank": i,
                "score": score,
                "series_uid": uid,
                "collection": collection,
                "patient_id": patient,
                "image_count": img_count,
                "out_dir": str(cand_dir),
                "convert_ok": int(ok_convert),
                "plot_ok": int(ok_plot),
                "convert_rc": rc_convert,
                "plot_rc": rc_plot,
            }
        )

    summary_csv = out_root / "batch_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "score",
                "series_uid",
                "collection",
                "patient_id",
                "image_count",
                "out_dir",
                "convert_ok",
                "plot_ok",
                "convert_rc",
                "plot_rc",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    n_ok = sum(1 for r in summary_rows if r["convert_ok"] and r["plot_ok"])
    print("\n[done] Batch complete")
    print(f"[done] Success: {n_ok}/{len(summary_rows)}")
    print(f"[done] Summary: {summary_csv}")


if __name__ == "__main__":
    main()

