"""
preprocess.py
-------------
Runs the full preprocessing pipeline on downloaded CHB-MIT data
and saves .npz cache files to data/processed/.

Run from project root:
    python scripts/preprocess.py --patients 1 2 3
    python scripts/preprocess.py              # all 24 patients

Output:
    data/processed/train.npz
    data/processed/val.npz
    data/processed/test.npz
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import build_split_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default patient splits (can be overridden by --patients)
DEFAULT_TRAIN = list(range(1, 19))
DEFAULT_VAL   = [19, 20, 21]
DEFAULT_TEST  = [22, 23, 24]


def run(cfg: dict, patients: list[int] | None = None):
    raw_dir       = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    channels      = cfg["data"].get("channels", None)
    fs            = cfg["data"]["sample_rate"]
    window_sec    = cfg["data"]["window_size"]
    overlap       = cfg["data"]["overlap"]
    sz_thresh     = cfg["data"]["seizure_threshold"]
    bp_low        = cfg["preprocessing"]["bandpass_low"]
    bp_high       = cfg["preprocessing"]["bandpass_high"]
    notch         = cfg["preprocessing"]["notch_freq"]

    # If specific patients provided, figure out which splits they belong to
    train_pids = cfg["splits"]["train_patients"]
    val_pids   = cfg["splits"]["val_patients"]
    test_pids  = cfg["splits"]["test_patients"]

    if patients:
        train_pids = [p for p in train_pids if p in patients]
        val_pids   = [p for p in val_pids   if p in patients]
        test_pids  = [p for p in test_pids  if p in patients]

    common = dict(
        raw_dir=raw_dir,
        target_channels=channels,
        window_size_sec=window_sec,
        overlap=overlap,
        seizure_threshold=sz_thresh,
        bandpass_low=bp_low,
        bandpass_high=bp_high,
        notch_freq=notch,
        sample_rate=fs,
        processed_dir=processed_dir,
        use_cache=False,   # always reprocess when running this script
    )

    results = {}

    for split_name, pids in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
        if not pids:
            logger.info(f"Skipping {split_name} — no matching patients")
            continue

        logger.info(f"\n{'='*55}")
        logger.info(f"Processing {split_name.upper()} split: patients {pids}")
        logger.info(f"{'='*55}")

        t0 = time.time()
        ds = build_split_dataset(patient_ids=pids, split_name=split_name, **common)
        elapsed = time.time() - t0

        summary = ds.summary()
        results[split_name] = summary

        logger.info(f"\n  {split_name} done in {elapsed:.1f}s")
        logger.info(f"  total windows      : {summary['total_windows']:,}")
        logger.info(f"  seizure windows    : {summary['seizure_windows']:,}  "
                    f"({100*summary['seizure_fraction']:.2f}%)")
        logger.info(f"  non-seizure windows: {summary['non_seizure_windows']:,}")
        logger.info(f"  class weight (sz)  : {summary['class_weight_seizure']:.1f}x")

    # ── Final summary table ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'PREPROCESSING COMPLETE':^55}")
    print(f"{'='*55}")
    print(f"  {'Split':<8} {'Windows':>10} {'Seizure':>10} {'Fraction':>10}")
    print(f"  {'-'*42}")
    for split, s in results.items():
        print(f"  {split:<8} {s['total_windows']:>10,} "
              f"{s['seizure_windows']:>10,} "
              f"{s['seizure_fraction']:>10.2%}")
    print(f"\n  Saved to: {processed_dir}/")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CHB-MIT EEG data")
    parser.add_argument(
        "--patients", type=int, nargs="+", default=None,
        help="Patient IDs to process (default: all from config). Example: --patients 1 2 3"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"config.yaml not found at: {args.config}")
        logger.error("Run from the project root directory.")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg, patients=args.patients)


if __name__ == "__main__":
    main()
