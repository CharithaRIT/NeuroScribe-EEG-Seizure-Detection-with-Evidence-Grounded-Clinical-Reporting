"""
loader.py
---------
Handles all raw data I/O for the CHB-MIT Scalp EEG dataset.

Responsibilities:
  1. Parse *-summary.txt files → per-file seizure onset/offset annotations
  2. Load .edf files via MNE → numpy arrays + metadata
  3. Build a unified manifest of all recordings for a patient

CHB-MIT summary.txt format example:
    File Name: chb01_03.edf
    File Start Time: 14:20:24
    File End Time: 15:20:24
    Number of Seizures in File: 1
    Seizure 1 Start Time: 2996 seconds
    Seizure 1 End Time: 3036 seconds
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)
mne.set_log_level("WARNING")  # suppress MNE verbose output


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SeizureAnnotation:
    """A single seizure interval within an EDF file (in seconds)."""
    onset: float
    offset: float

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class RecordingInfo:
    """Metadata + seizure annotations for one EDF file."""
    edf_path: str
    patient_id: int
    filename: str
    start_time: Optional[str] = None   # "HH:MM:SS" from summary
    end_time: Optional[str] = None
    seizures: list[SeizureAnnotation] = field(default_factory=list)

    @property
    def has_seizure(self) -> bool:
        return len(self.seizures) > 0

    @property
    def n_seizures(self) -> int:
        return len(self.seizures)


@dataclass
class RawRecording:
    """Loaded EEG data for one EDF file."""
    info: RecordingInfo
    data: np.ndarray          # shape: (n_channels, n_samples)
    sample_rate: int          # Hz
    channel_names: list[str]
    label_array: np.ndarray   # shape: (n_samples,)  0=non-seizure, 1=seizure


# ---------------------------------------------------------------------------
# Summary file parser
# ---------------------------------------------------------------------------

def parse_summary_file(summary_path: str) -> dict[str, list[SeizureAnnotation]]:
    """
    Parses a CHB-MIT *-summary.txt file.

    Returns:
        dict mapping filename (e.g. 'chb01_03.edf') to list of SeizureAnnotation.
        Files with 0 seizures are included with an empty list.

    Handles edge cases:
        - 'Seizure Start Time' and 'Seizure N Start Time' variants
        - Files listed with no seizure count line
        - chb04 split into chb04a / chb04b subdirectories
    """
    annotations: dict[str, list[SeizureAnnotation]] = {}
    current_file: Optional[str] = None
    n_seizures_expected = 0
    seizure_starts: dict[int, float] = {}
    seizure_ends: dict[int, float] = {}

    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()

            # ---- New file block ----
            m = re.match(r"File Name:\s+(.+\.edf)", line, re.IGNORECASE)
            if m:
                # Flush previous file
                if current_file is not None:
                    annotations[current_file] = _build_annotations(
                        seizure_starts, seizure_ends, n_seizures_expected
                    )
                current_file = m.group(1).strip()
                n_seizures_expected = 0
                seizure_starts = {}
                seizure_ends = {}
                if current_file not in annotations:
                    annotations[current_file] = []
                continue

            # ---- Number of seizures ----
            m = re.match(r"Number of Seizures in File:\s+(\d+)", line, re.IGNORECASE)
            if m:
                n_seizures_expected = int(m.group(1))
                continue

            # ---- Seizure N Start Time ----
            # Matches: "Seizure Start Time: X seconds"
            #      or  "Seizure 1 Start Time: X seconds"
            m = re.match(r"Seizure\s*(\d*)\s*Start Time:\s*(\d+)\s*seconds?", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) if m.group(1) else 1
                seizure_starts[idx] = float(m.group(2))
                continue

            # ---- Seizure N End Time ----
            m = re.match(r"Seizure\s*(\d*)\s*End Time:\s*(\d+)\s*seconds?", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) if m.group(1) else 1
                seizure_ends[idx] = float(m.group(2))
                continue

    # Flush last file block
    if current_file is not None:
        annotations[current_file] = _build_annotations(
            seizure_starts, seizure_ends, n_seizures_expected
        )

    logger.debug(f"Parsed {summary_path}: {len(annotations)} files, "
                 f"{sum(len(v) for v in annotations.values())} total seizures")
    return annotations


def _build_annotations(
    starts: dict[int, float],
    ends: dict[int, float],
    n_expected: int,
) -> list[SeizureAnnotation]:
    """Zip start/end dicts into SeizureAnnotation list with validation."""
    result = []
    for idx in sorted(starts.keys()):
        if idx not in ends:
            logger.warning(f"Seizure {idx} has start but no end — skipping")
            continue
        onset = starts[idx]
        offset = ends[idx]
        if offset <= onset:
            logger.warning(f"Seizure {idx}: offset {offset} <= onset {onset} — skipping")
            continue
        result.append(SeizureAnnotation(onset=onset, offset=offset))

    if len(result) != n_expected and n_expected > 0:
        logger.warning(f"Expected {n_expected} seizures, parsed {len(result)}")

    return result


# ---------------------------------------------------------------------------
# EDF loader (MNE-based)
# ---------------------------------------------------------------------------

def load_edf(
    edf_path: str,
    target_channels: Optional[list[str]] = None,
    target_sfreq: int = 256,
) -> tuple[np.ndarray, int, list[str]]:
    """
    Loads one EDF file using MNE.

    Args:
        edf_path:         Path to the .edf file.
        target_channels:  List of channel names to select. If None, uses all channels.
        target_sfreq:     Target sample rate. Resamples if the file differs.

    Returns:
        data:          numpy array of shape (n_channels, n_samples), in µV
        sample_rate:   actual sample rate after resampling
        channel_names: list of channel names in output order

    Notes:
        - MNE stores EEG data in Volts internally; we convert to µV (×1e6)
        - CHB-MIT channels use bipolar notation: 'FP1-F7', 'F7-T7', etc.
        - Case-insensitive channel matching is applied automatically
    """
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # ---- Channel selection ----
    available = [ch.upper() for ch in raw.ch_names]

    if target_channels is not None:
        # Case-insensitive match
        selected = []
        missing = []
        for ch in target_channels:
            ch_upper = ch.upper()
            # MNE often prepends "EEG " prefix — strip it for matching
            available_stripped = [c.replace("EEG ", "").strip() for c in raw.ch_names]
            if ch_upper in [c.upper() for c in available_stripped]:
                idx = [c.upper() for c in available_stripped].index(ch_upper)
                selected.append(raw.ch_names[idx])
            else:
                missing.append(ch)

        if missing:
            logger.warning(f"{os.path.basename(edf_path)}: channels not found: {missing}")

        if not selected:
            logger.error(f"No target channels found in {edf_path}. Using all available.")
            selected = raw.ch_names
        else:
            raw.pick_channels(selected)

    # ---- Resample if needed ----
    actual_sfreq = int(raw.info["sfreq"])
    if actual_sfreq != target_sfreq:
        logger.info(f"Resampling {os.path.basename(edf_path)}: {actual_sfreq} → {target_sfreq} Hz")
        raw.resample(target_sfreq, npad="auto")

    # ---- Extract numpy data (convert V → µV) ----
    data, _ = raw.get_data(return_times=True)
    data = data * 1e6  # Volts → microvolts

    channel_names = raw.ch_names
    return data, target_sfreq, channel_names


# ---------------------------------------------------------------------------
# Patient-level manifest builder
# ---------------------------------------------------------------------------

def build_patient_manifest(
    patient_dir: str,
    patient_id: int,
    target_channels: Optional[list[str]] = None,
) -> list[RecordingInfo]:
    """
    Scans a patient directory and builds a list of RecordingInfo objects.

    Args:
        patient_dir:      Path to e.g. 'data/chb-mit/chb01'
        patient_id:       Integer patient ID (1–24)
        target_channels:  Channel list for downstream loading (not loaded here)

    Returns:
        List of RecordingInfo, one per EDF file found.
    """
    if not os.path.isdir(patient_dir):
        raise NotADirectoryError(f"Patient directory not found: {patient_dir}")

    # Find summary file
    summary_candidates = [
        f for f in os.listdir(patient_dir)
        if "summary" in f.lower() and f.endswith(".txt")
    ]
    if not summary_candidates:
        raise FileNotFoundError(f"No summary .txt found in {patient_dir}")

    summary_path = os.path.join(patient_dir, summary_candidates[0])
    annotations = parse_summary_file(summary_path)

    # Find all EDF files
    edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith(".edf")])

    recordings = []
    for edf_filename in edf_files:
        edf_path = os.path.join(patient_dir, edf_filename)
        seizures = annotations.get(edf_filename, [])

        info = RecordingInfo(
            edf_path=edf_path,
            patient_id=patient_id,
            filename=edf_filename,
            seizures=seizures,
        )
        recordings.append(info)

    n_with_seizures = sum(1 for r in recordings if r.has_seizure)
    total_seizures = sum(r.n_seizures for r in recordings)
    logger.info(
        f"Patient chb{patient_id:02d}: {len(recordings)} recordings, "
        f"{n_with_seizures} with seizures, {total_seizures} total seizures"
    )
    return recordings


def load_recording(
    info: RecordingInfo,
    target_channels: Optional[list[str]] = None,
    target_sfreq: int = 256,
) -> RawRecording:
    """
    Loads the EDF file for a RecordingInfo and builds the sample-level label array.

    Label array: 0 everywhere, 1 for samples within any seizure interval.

    Returns:
        RawRecording with data, sample_rate, channel_names, label_array
    """
    data, sample_rate, channel_names = load_edf(
        info.edf_path,
        target_channels=target_channels,
        target_sfreq=target_sfreq,
    )
    n_samples = data.shape[1]

    # Build sample-level label array
    label_array = np.zeros(n_samples, dtype=np.int8)
    for seizure in info.seizures:
        start_sample = int(seizure.onset * sample_rate)
        end_sample = int(seizure.offset * sample_rate)
        # Clamp to valid range
        start_sample = max(0, min(start_sample, n_samples - 1))
        end_sample = max(0, min(end_sample, n_samples))
        label_array[start_sample:end_sample] = 1

    return RawRecording(
        info=info,
        data=data,
        sample_rate=sample_rate,
        channel_names=channel_names,
        label_array=label_array,
    )


# ---------------------------------------------------------------------------
# Quick dataset statistics
# ---------------------------------------------------------------------------

def dataset_stats(recordings: list[RecordingInfo], sample_rate: int = 256) -> dict:
    """
    Computes summary statistics for a list of RecordingInfo objects.

    Useful for the dataset exploration notebook (Checkpoint 2).
    """
    total_files = len(recordings)
    files_with_seizures = sum(1 for r in recordings if r.has_seizure)
    total_seizures = sum(r.n_seizures for r in recordings)

    seizure_durations = [
        s.duration
        for r in recordings
        for s in r.seizures
    ]

    stats = {
        "total_recordings": total_files,
        "recordings_with_seizure": files_with_seizures,
        "recordings_without_seizure": total_files - files_with_seizures,
        "total_seizure_events": total_seizures,
        "seizure_duration_mean_s": float(np.mean(seizure_durations)) if seizure_durations else 0.0,
        "seizure_duration_std_s": float(np.std(seizure_durations)) if seizure_durations else 0.0,
        "seizure_duration_min_s": float(np.min(seizure_durations)) if seizure_durations else 0.0,
        "seizure_duration_max_s": float(np.max(seizure_durations)) if seizure_durations else 0.0,
    }
    return stats
