"""
demo_download.py
----------------
Uses wfdb.dl_files() to pull only seizure-containing EDF files
from PhysioNet — no full database download needed.

Strategy:
  1. Pull summary.txt per patient  (~5 KB each, instant)
  2. Parse it → find which EDFs contain seizures
  3. dl_files() those EDFs only   (~60-80 MB each)

Total: ~400-600 MB for 3 patients vs ~9 GB full download.

Run from project root:
    python data/demo_download.py --patients 1 2 3
"""

import argparse
import os
import re
import sys

import wfdb

DB_DIR = "chbmit/1.0.0"


# ── Summary parser ─────────────────────────────────────────────────────────────

def parse_summary(summary_path: str) -> dict[str, list]:
    """Returns {edf_filename: [(onset_sec, offset_sec), ...]}"""
    result: dict[str, list] = {}
    current, starts, ends = None, {}, {}

    with open(summary_path) as f:
        for line in f:
            line = line.strip()

            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.IGNORECASE)
            if m:
                if current:
                    result[current] = [
                        (starts[i], ends[i]) for i in sorted(starts) if i in ends
                    ]
                current = m.group(1)
                starts, ends = {}, {}
                result.setdefault(current, [])
                continue

            m = re.match(r"Seizure\s*(\d*)\s*Start Time:\s*(\d+)", line, re.IGNORECASE)
            if m:
                starts[int(m.group(1) or 1)] = float(m.group(2))
                continue

            m = re.match(r"Seizure\s*(\d*)\s*End Time:\s*(\d+)", line, re.IGNORECASE)
            if m:
                ends[int(m.group(1) or 1)] = float(m.group(2))
                continue

    if current:
        result[current] = [(starts[i], ends[i]) for i in sorted(starts) if i in ends]

    return result


# ── Per-patient download ───────────────────────────────────────────────────────

def download_patient(patient_id: int, output_dir: str, n_background: int = 1):
    pdir      = f"chb{patient_id:02d}"
    local_dir = os.path.join(output_dir, pdir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  {pdir}")
    print(f"{'─'*55}")

    # ── Step 1: pull summary.txt ───────────────────────────────────────────
    summary_file = f"{pdir}/{pdir}-summary.txt"
    summary_path = os.path.join(local_dir, f"{pdir}-summary.txt")

    if not os.path.exists(summary_path):
        print(f"  Fetching {pdir}-summary.txt ...")
        wfdb.dl_files(
            db_dir=DB_DIR,
            dl_dir=output_dir,
            files=[summary_file],
            keep_subdirs=True,
            overwrite=False,
        )
    else:
        print(f"  summary.txt already present")

    # ── Step 2: parse → identify seizure files ─────────────────────────────
    annotations  = parse_summary(summary_path)
    seizure_edfs = [f for f, s in annotations.items() if s]
    non_sz_edfs  = [f for f, s in annotations.items() if not s]

    print(f"  Total EDF files in summary : {len(annotations)}")
    print(f"  Files WITH seizures        : {len(seizure_edfs)}")
    print(f"  Files WITHOUT seizures     : {len(non_sz_edfs)}  (only {n_background} will be fetched)")

    print(f"\n  Seizure files:")
    for edf in seizure_edfs:
        intervals = annotations[edf]
        ivstr = ", ".join(f"{o:.0f}–{x:.0f}s" for o, x in intervals)
        print(f"    {edf:<30}  [{ivstr}]")

    # ── Step 3: dl_files() for seizure EDFs + background ──────────────────
    to_download = (
        [f"{pdir}/{f}" for f in seizure_edfs] +
        [f"{pdir}/{f}" for f in non_sz_edfs[:n_background]]
    )

    # filter out already-downloaded
    to_download = [
        f for f in to_download
        if not os.path.exists(os.path.join(output_dir, f.replace("/", os.sep)))
        or os.path.getsize(os.path.join(output_dir, f.replace("/", os.sep))) == 0
    ]

    if to_download:
        print(f"\n  Downloading {len(to_download)} EDF file(s) via wfdb ...")
        wfdb.dl_files(
            db_dir=DB_DIR,
            dl_dir=output_dir,
            files=to_download,
            keep_subdirs=True,
            overwrite=False,
        )
    else:
        print(f"\n  All EDF files already on disk — skipping download")

    # ── Summary ───────────────────────────────────────────────────────────
    on_disk = [
        f for f in os.listdir(local_dir)
        if f.endswith(".edf") or f.endswith(".txt")
    ]
    size_mb = sum(
        os.path.getsize(os.path.join(local_dir, f)) for f in on_disk
    ) / 1e6

    print(f"  {len([f for f in on_disk if f.endswith('.edf')])} EDF(s) on disk | {size_mb:.0f} MB")
    return size_mb


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients",   type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output",     type=str, default="data/chb-mit")
    parser.add_argument("--background", type=int, default=1,
                        help="Non-seizure EDF files to include per patient (default 1)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"\nUsing wfdb.dl_files()  →  {os.path.abspath(args.output)}")
    print(f"Patients: {[f'chb{p:02d}' for p in args.patients]}")

    total_mb = 0
    for pid in args.patients:
        total_mb += download_patient(pid, args.output, args.background)

    print(f"\n{'='*55}")
    print(f"  Done.  Total on disk: {total_mb:.0f} MB")
    print(f"  Next:  python scripts/preprocess.py --patients {' '.join(str(p) for p in args.patients)}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
