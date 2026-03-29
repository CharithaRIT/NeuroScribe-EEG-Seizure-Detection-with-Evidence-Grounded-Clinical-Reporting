"""
preprocessor.py
---------------
Signal preprocessing and windowing pipeline for CHB-MIT EEG data.

Pipeline order:
  1. Bandpass filter  (0.5 – 40 Hz)  — removes DC drift and high-freq noise
  2. Notch filter     (60 Hz)         — removes US powerline interference
  3. Window segmentation              — 4-second windows with 50% overlap
  4. Window labeling                  — seizure if ≥50% of samples are ictal
  5. Z-score normalization            — per channel, per window

Design notes:
  - All filtering uses zero-phase IIR (filtfilt) to avoid phase distortion
  - Windows are generated per recording then concatenated into arrays
  - Class imbalance info is computed and returned for loss weighting
"""

import logging
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from src.data.loader import RawRecording

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def bandpass_filter(
    data: np.ndarray,
    fs: int,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter applied per channel.

    Args:
        data:   (n_channels, n_samples) in µV
        fs:     sample rate in Hz
        low:    lower cutoff frequency in Hz
        high:   upper cutoff frequency in Hz
        order:  filter order (4 gives -80 dB/decade roll-off)

    Returns:
        Filtered array of same shape.
    """
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq

    # Clamp to valid range (must be strictly between 0 and 1)
    low_norm = max(1e-4, min(low_norm, 0.9999))
    high_norm = max(low_norm + 1e-4, min(high_norm, 0.9999))

    b, a = butter(order, [low_norm, high_norm], btype="band")
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered


def notch_filter(
    data: np.ndarray,
    fs: int,
    freq: float = 60.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Zero-phase IIR notch filter to remove powerline noise.

    Args:
        data:           (n_channels, n_samples)
        fs:             sample rate in Hz
        freq:           notch frequency in Hz (60 Hz in the US)
        quality_factor: Q = freq / bandwidth; higher Q = narrower notch

    Returns:
        Filtered array of same shape.
    """
    b, a = iirnotch(freq, quality_factor, fs)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered


def apply_filters(
    data: np.ndarray,
    fs: int,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    notch_freq: float = 60.0,
) -> np.ndarray:
    """Convenience function: bandpass then notch."""
    data = bandpass_filter(data, fs, bandpass_low, bandpass_high)
    data = notch_filter(data, fs, notch_freq)
    return data


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def create_windows(
    data: np.ndarray,
    label_array: np.ndarray,
    window_size: int,
    step_size: int,
    seizure_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments a continuous EEG recording into fixed-length windows.

    Args:
        data:               (n_channels, n_samples) filtered EEG in µV
        label_array:        (n_samples,) sample-level labels (0/1)
        window_size:        samples per window (e.g. 1024 for 4s @ 256Hz)
        step_size:          samples to advance per step (e.g. 512 for 50% overlap)
        seizure_threshold:  fraction of samples in window that must be ictal
                            to assign a seizure label to that window

    Returns:
        windows: (n_windows, n_channels, window_size)   float32
        labels:  (n_windows,)                           int8  (0 or 1)
    """
    n_channels, n_samples = data.shape
    n_windows = max(0, (n_samples - window_size) // step_size + 1)

    if n_windows == 0:
        logger.warning(f"Recording too short ({n_samples} samples) for window_size={window_size}")
        return np.empty((0, n_channels, window_size), dtype=np.float32), np.empty(0, dtype=np.int8)

    windows = np.empty((n_windows, n_channels, window_size), dtype=np.float32)
    labels = np.empty(n_windows, dtype=np.int8)

    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[:, start:end]
        seizure_fraction = label_array[start:end].mean()
        labels[i] = 1 if seizure_fraction >= seizure_threshold else 0

    return windows, labels


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def zscore_normalize(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalization per channel per window.

    Each window is normalized independently to make the model robust to
    inter-patient amplitude variations and recording-level baseline shifts.

    Args:
        windows: (n_windows, n_channels, window_size) float32
        eps:     small constant to prevent division by zero for flat channels

    Returns:
        Normalized array of same shape.
    """
    # mean/std over the time dimension (axis=-1), keepdims for broadcasting
    mean = windows.mean(axis=-1, keepdims=True)   # (n_windows, n_channels, 1)
    std = windows.std(axis=-1, keepdims=True)      # (n_windows, n_channels, 1)
    return (windows - mean) / (std + eps)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline for one recording
# ---------------------------------------------------------------------------

def preprocess_recording(
    recording: RawRecording,
    window_size_sec: float = 4.0,
    overlap: float = 0.5,
    seizure_threshold: float = 0.5,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    notch_freq: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: filter → window → label → normalize for one RawRecording.

    Args:
        recording:          RawRecording from loader.load_recording()
        window_size_sec:    window duration in seconds
        overlap:            fraction overlap (0.5 = 50%)
        seizure_threshold:  fraction of window that must be seizure
        bandpass_low/high:  bandpass filter cutoffs in Hz
        notch_freq:         notch filter frequency in Hz

    Returns:
        windows:  (n_windows, n_channels, window_size) float32 — normalized
        labels:   (n_windows,) int8 — 0 or 1
    """
    fs = recording.sample_rate
    window_size = int(window_size_sec * fs)
    step_size = int(window_size * (1.0 - overlap))

    # Step 1: Filter
    filtered = apply_filters(
        recording.data, fs,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        notch_freq=notch_freq,
    )

    # Step 2: Window + label
    windows, labels = create_windows(
        filtered, recording.label_array,
        window_size=window_size,
        step_size=step_size,
        seizure_threshold=seizure_threshold,
    )

    if windows.shape[0] == 0:
        return windows, labels

    # Step 3: Normalize
    windows = zscore_normalize(windows)

    n_seizure = labels.sum()
    n_total = len(labels)
    logger.debug(
        f"{recording.info.filename}: {n_total} windows, "
        f"{n_seizure} seizure ({100*n_seizure/max(n_total,1):.1f}%)"
    )
    return windows, labels


# ---------------------------------------------------------------------------
# Class imbalance utilities
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray) -> tuple[float, float]:
    """
    Computes inverse-frequency class weights for weighted loss.

    Returns (weight_for_class_0, weight_for_class_1).
    Used to initialize focal loss alpha or weighted cross-entropy.
    """
    n_total = len(labels)
    n_pos = labels.sum()
    n_neg = n_total - n_pos

    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0

    # Inverse frequency: weight_i = n_total / (2 * n_i)
    w0 = n_total / (2.0 * n_neg)
    w1 = n_total / (2.0 * n_pos)
    return float(w0), float(w1)


def imbalance_report(labels: np.ndarray) -> dict:
    """Returns a dict with class balance statistics for logging/notebooks."""
    n_total = len(labels)
    n_pos = int(labels.sum())
    n_neg = n_total - n_pos
    ratio = n_pos / max(n_total, 1)
    w0, w1 = compute_class_weights(labels)
    return {
        "total_windows": n_total,
        "seizure_windows": n_pos,
        "non_seizure_windows": n_neg,
        "seizure_fraction": round(ratio, 4),
        "class_weight_non_seizure": round(w0, 3),
        "class_weight_seizure": round(w1, 3),
    }
