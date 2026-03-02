"""
extractor.py
------------
Ground-truth EEG feature extraction from seizure segments.

Extracts quantitative features (temporal, amplitude, spatial, frequency) from
a seizure window in a CHB-MIT recording. These features serve as the ground
truth for the Evidence Verification Agent in the NeuroScribe pipeline.

Typical usage:
    from src.features.extractor import extract_features

    feat = extract_features(
        data, ch_names,
        onset_sec=2996.0, offset_sec=3036.0,
        patient='chb01', filename='chb01_03.edf',
        fs=256,
    )
    # feat['frequency']['dominant_hz'], feat['spatial']['top3_channels'], ...
"""

from typing import Optional
import numpy as np
from scipy.signal import welch


def extract_features(
    data: np.ndarray,
    ch_names: list[str],
    onset_sec: float,
    offset_sec: float,
    patient: str,
    filename: str,
    fs: int = 256,
) -> dict:
    """
    Extracts ground-truth quantitative features from one seizure segment.

    Args:
        data:        (n_channels, n_samples) EEG array in µV.
        ch_names:    List of channel names, length n_channels.
        onset_sec:   Seizure start time in seconds.
        offset_sec:  Seizure end time in seconds.
        patient:     Patient identifier string (e.g. 'chb01').
        filename:    Source EDF filename (e.g. 'chb01_03.edf').
        fs:          Sampling rate in Hz.

    Returns:
        dict with keys:
            patient   — patient ID
            file      — source EDF filename
            temporal  — onset_sec, offset_sec, duration_sec
            amplitude — mean_uV, max_uV, rms_uV
            spatial   — top3_channels (list), most_active (str)
            frequency — dominant_hz, delta/theta/alpha/beta/gamma band powers
    """
    start = int(onset_sec  * fs)
    end   = int(offset_sec * fs)
    seg   = data[:, start:end]   # (n_channels, seizure_samples)

    # ── Temporal ──────────────────────────────────────────────────────────
    duration_sec = offset_sec - onset_sec

    # ── Amplitude ─────────────────────────────────────────────────────────
    amp_mean = float(np.abs(seg).mean())
    amp_max  = float(np.abs(seg).max())
    amp_rms  = float(np.sqrt((seg ** 2).mean()))

    per_ch_rms = np.sqrt((seg ** 2).mean(axis=1))   # (n_channels,)

    # ── Spatial — top-3 most active channels ──────────────────────────────
    top3_idx   = per_ch_rms.argsort()[::-1][:3]
    top3_names = [ch_names[i] for i in top3_idx]

    # ── Frequency — dominant frequency and EEG band powers ────────────────
    # Average PSD across the top-3 active channels
    psds = []
    for i in top3_idx:
        f_ax, psd = welch(seg[i], fs=fs, nperseg=min(fs * 2, seg.shape[1]))
        psds.append(psd)
    mean_psd = np.mean(psds, axis=0)

    dominant_freq = float(f_ax[mean_psd.argmax()])

    def band_power(lo: float, hi: float) -> float:
        mask = (f_ax >= lo) & (f_ax <= hi)
        return float(mean_psd[mask].mean()) if mask.any() else 0.0

    return {
        "patient": patient,
        "file":    filename,
        "temporal": {
            "onset_sec":    onset_sec,
            "offset_sec":   offset_sec,
            "duration_sec": round(duration_sec, 1),
        },
        "amplitude": {
            "mean_uV": round(amp_mean, 2),
            "max_uV":  round(amp_max,  2),
            "rms_uV":  round(amp_rms,  2),
        },
        "spatial": {
            "top3_channels": top3_names,
            "most_active":   top3_names[0],
        },
        "frequency": {
            "dominant_hz":  round(dominant_freq, 1),
            "delta_power":  round(band_power(0.5,  4),  4),
            "theta_power":  round(band_power(4,    8),  4),
            "alpha_power":  round(band_power(8,   13),  4),
            "beta_power":   round(band_power(13,  30),  4),
            "gamma_power":  round(band_power(30,  45),  4),
        },
    }
