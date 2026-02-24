"""
inspect_data.py
---------------
Quick sanity check — loads patient chb01, prints data shape,
channel names, seizure annotations, and a few sample values.

Run from project root:
    python scripts/inspect_data.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import build_patient_manifest, load_recording, dataset_stats
from src.data.preprocessor import preprocess_recording, imbalance_report

RAW_DIR    = "data/chb-mit"
PATIENT_ID = 1
PATIENT_DIR = os.path.join(RAW_DIR, f"chb{PATIENT_ID:02d}")

# ── 1. Parse summary file ──────────────────────────────────────────────────
print("=" * 60)
print(f"PATIENT: chb{PATIENT_ID:02d}")
print("=" * 60)

recordings = build_patient_manifest(PATIENT_DIR, PATIENT_ID)
stats = dataset_stats(recordings)

print(f"\n[Dataset Stats]")
for k, v in stats.items():
    print(f"  {k:<35} {v}")

# ── 2. List recordings with seizure info ──────────────────────────────────
print(f"\n[Recordings]")
print(f"  {'File':<25} {'Seizures':>8}  {'Intervals (sec)'}")
print(f"  {'-'*25}  {'-'*8}  {'-'*30}")
for r in recordings:
    intervals = [(f"{s.onset:.0f}s", f"{s.offset:.0f}s") for s in r.seizures]
    print(f"  {r.filename:<25} {r.n_seizures:>8}  {intervals}")

# ── 3. Load the first recording with a seizure ────────────────────────────
seizure_recs = [r for r in recordings if r.has_seizure]
if not seizure_recs:
    print("\nNo seizure recordings found — check download.")
    sys.exit(1)

target_rec = seizure_recs[0]
print(f"\n[Loading] {target_rec.filename}")
raw = load_recording(target_rec, target_sfreq=256)

print(f"\n[Raw EEG Array]")
print(f"  Shape         : {raw.data.shape}   (channels x samples)")
print(f"  Sample rate   : {raw.sample_rate} Hz")
print(f"  Duration      : {raw.data.shape[1] / raw.sample_rate:.1f} seconds")
print(f"  n_channels    : {raw.data.shape[0]}")
print(f"  Value range   : [{raw.data.min():.2f}, {raw.data.max():.2f}] µV")

print(f"\n[Channel Names]")
for i, ch in enumerate(raw.channel_names):
    print(f"  [{i:02d}] {ch}")

print(f"\n[Seizure Label Array]")
print(f"  Shape         : {raw.label_array.shape}")
print(f"  Seizure samples: {raw.label_array.sum()} / {len(raw.label_array)}"
      f"  ({100*raw.label_array.mean():.2f}%)")

# ── 4. Print first 5 rows (samples) of raw data ──────────────────────────
print(f"\n[First 5 samples across all channels] (µV)")
header = "  Sample  | " + " | ".join(f"{ch[:8]:>8}" for ch in raw.channel_names[:6]) + " | ..."
print(header)
print("  " + "-" * (len(header) - 2))
for i in range(5):
    row = f"  {i:6d}  | " + " | ".join(f"{raw.data[ch, i]:8.2f}" for ch in range(min(6, raw.data.shape[0]))) + " | ..."
    print(row)

# ── 5. Run preprocessing and show windowed output ─────────────────────────
print(f"\n[Preprocessing → windows]")
windows, labels = preprocess_recording(raw, window_size_sec=4.0, overlap=0.5)

print(f"  windows shape : {windows.shape}   (n_windows x channels x samples)")
print(f"  labels shape  : {labels.shape}")

report = imbalance_report(labels)
print(f"\n[Class Balance]")
for k, v in report.items():
    print(f"  {k:<35} {v}")

# ── 6. Print first 3 windows, first 3 channels, first 5 time steps ────────
print(f"\n[First 3 windows — first 3 channels — first 5 time steps] (normalized µV)")
print(f"  {'Win':>4}  {'Ch':>4}  | " + "  ".join(f"t={t:4d}" for t in range(5)))
print("  " + "-" * 60)
for w in range(min(3, windows.shape[0])):
    for c in range(min(3, windows.shape[1])):
        vals = "  ".join(f"{windows[w, c, t]:7.3f}" for t in range(5))
        lbl = f"  [win={w} ch={c}  label={int(labels[w])}]  {vals}"
        print(lbl)
    print()

print("=" * 60)
print("Data pipeline OK.")
print("=" * 60)
