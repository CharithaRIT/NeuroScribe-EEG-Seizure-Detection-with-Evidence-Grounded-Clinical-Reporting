"""
dataset.py
----------
PyTorch Dataset and DataLoader builders for the CHB-MIT pipeline.

Key design decisions:
  - Patient-independent splits: train on patients 1–18, val on 19–21, test on 22–24
  - All preprocessing happens here; downstream models receive clean tensors
  - Supports optional pre-caching to .npz files for fast repeated loading
  - DataLoader uses WeightedRandomSampler to address class imbalance during training

Typical usage:
    train_loader, val_loader, test_loader = build_dataloaders(config)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.loader import (
    RecordingInfo,
    build_patient_manifest,
    dataset_stats,
    load_recording,
)
from src.data.preprocessor import imbalance_report, preprocess_recording

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patient split definitions (patient-independent evaluation)
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_PATIENTS = list(range(1, 19))   # chb01–chb18
DEFAULT_VAL_PATIENTS   = [19, 20, 21]          # chb19–chb21
DEFAULT_TEST_PATIENTS  = [22, 23, 24]          # chb22–chb24


# ---------------------------------------------------------------------------
# EEGDataset
# ---------------------------------------------------------------------------

class EEGDataset(Dataset):
    """
    PyTorch Dataset for windowed CHB-MIT EEG data.

    Each item is:
        x: torch.FloatTensor of shape (n_channels, window_size)
        y: torch.FloatTensor scalar  (0.0 or 1.0)

    Attributes:
        windows:  (N, C, T) float32 numpy array
        labels:   (N,) int8 numpy array
        patient_ids: (N,) int array — which patient each window came from
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
    ):
        assert len(windows) == len(labels), "windows and labels must have same length"
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.patient_ids = patient_ids if patient_ids is not None else np.zeros(len(labels), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx])     # (C, T)
        y = torch.tensor(self.labels[idx])           # scalar
        return x, y

    @property
    def n_seizure(self) -> int:
        return int(self.labels.sum())

    @property
    def n_non_seizure(self) -> int:
        return len(self.labels) - self.n_seizure

    @property
    def seizure_fraction(self) -> float:
        return self.n_seizure / max(len(self.labels), 1)

    def class_weight_tensor(self) -> torch.Tensor:
        """Returns [w_neg, w_pos] tensor for weighted loss initialization."""
        n = len(self.labels)
        n_pos = self.n_seizure
        n_neg = self.n_non_seizure
        w_neg = n / (2.0 * max(n_neg, 1))
        w_pos = n / (2.0 * max(n_pos, 1))
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)

    def summary(self) -> dict:
        return imbalance_report(self.labels.astype(np.int8))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(processed_dir: str, split: str) -> str:
    return os.path.join(processed_dir, f"{split}.npz")


def _save_cache(path: str, windows: np.ndarray, labels: np.ndarray, patient_ids: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, windows=windows, labels=labels, patient_ids=patient_ids)
    logger.info(f"Saved processed data → {path}")


def _load_cache(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["windows"], data["labels"], data["patient_ids"]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_split_dataset(
    raw_dir: str,
    patient_ids: list[int],
    split_name: str,
    target_channels: Optional[list[str]] = None,
    window_size_sec: float = 4.0,
    overlap: float = 0.5,
    seizure_threshold: float = 0.5,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    notch_freq: float = 60.0,
    sample_rate: int = 256,
    processed_dir: Optional[str] = None,
    use_cache: bool = True,
) -> EEGDataset:
    """
    Builds an EEGDataset for a list of patient IDs.

    Iterates over all patients → all recordings → preprocess → concatenate.
    Optionally caches to .npz for fast reloading.

    Args:
        raw_dir:          Root directory of CHB-MIT (contains chb01/, chb02/, etc.)
        patient_ids:      List of patient IDs to include
        split_name:       One of 'train', 'val', 'test' (used for cache filename)
        target_channels:  Channel names to select (None = all available)
        window_size_sec:  Window length in seconds
        overlap:          Overlap fraction (0.5 = 50%)
        seizure_threshold: Fraction of seizure samples to label window as seizure
        bandpass_low/high: Filter cutoffs
        notch_freq:       Notch frequency
        sample_rate:      Target sample rate
        processed_dir:    If provided, cache preprocessed arrays here
        use_cache:        Load from cache if available

    Returns:
        EEGDataset ready for DataLoader
    """
    # ---- Try cache ----
    if use_cache and processed_dir:
        cache_file = _cache_path(processed_dir, split_name)
        if os.path.exists(cache_file):
            logger.info(f"Loading {split_name} from cache: {cache_file}")
            windows, labels, patient_ids_arr = _load_cache(cache_file)
            dataset = EEGDataset(windows, labels, patient_ids_arr)
            logger.info(f"{split_name}: {dataset.summary()}")
            return dataset

    # ---- Build from raw data ----
    all_windows = []
    all_labels = []
    all_patient_ids = []

    for pid in patient_ids:
        patient_dir = os.path.join(raw_dir, f"chb{pid:02d}")
        if not os.path.isdir(patient_dir):
            logger.warning(f"Patient directory not found: {patient_dir} — skipping")
            continue

        try:
            recordings = build_patient_manifest(patient_dir, pid, target_channels)
        except Exception as e:
            logger.error(f"Failed to build manifest for chb{pid:02d}: {e}")
            continue

        patient_stats = dataset_stats(recordings, sample_rate)
        logger.info(
            f"chb{pid:02d}: {patient_stats['total_recordings']} files, "
            f"{patient_stats['total_seizure_events']} seizures"
        )

        for rec_info in recordings:
            try:
                recording = load_recording(rec_info, target_channels, sample_rate)
                windows, labels = preprocess_recording(
                    recording,
                    window_size_sec=window_size_sec,
                    overlap=overlap,
                    seizure_threshold=seizure_threshold,
                    bandpass_low=bandpass_low,
                    bandpass_high=bandpass_high,
                    notch_freq=notch_freq,
                )
                if windows.shape[0] == 0:
                    continue
                # Skip files whose channel count doesn't match target_channels.
                # This happens when a file's channel names don't match any of
                # the target names (load_edf falls back to all channels).
                if target_channels is not None and windows.shape[1] != len(target_channels):
                    logger.warning(
                        f"Skipping {rec_info.filename}: channel count mismatch "
                        f"(got {windows.shape[1]}, expected {len(target_channels)})"
                    )
                    continue
                all_windows.append(windows)
                all_labels.append(labels)
                all_patient_ids.append(np.full(len(labels), pid, dtype=np.int32))

            except Exception as e:
                logger.error(f"Failed to process {rec_info.filename}: {e}")
                continue

    if not all_windows:
        raise RuntimeError(f"No windows extracted for {split_name} split — check raw_dir: {raw_dir}")

    windows_arr = np.concatenate(all_windows, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    patient_ids_arr = np.concatenate(all_patient_ids, axis=0)

    # ---- Save cache ----
    if processed_dir:
        _save_cache(_cache_path(processed_dir, split_name), windows_arr, labels_arr, patient_ids_arr)

    dataset = EEGDataset(windows_arr, labels_arr, patient_ids_arr)
    logger.info(f"{split_name} dataset: {dataset.summary()}")
    return dataset


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_train_loader(
    dataset: EEGDataset,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Training DataLoader with WeightedRandomSampler to handle class imbalance.

    WeightedRandomSampler assigns higher probability to minority (seizure)
    samples so each batch has a more balanced seizure/non-seizure ratio,
    complementing focal loss during training.
    """
    # Per-sample weights: seizure windows get higher weight
    n = len(dataset)
    n_pos = dataset.n_seizure
    n_neg = dataset.n_non_seizure

    w_pos = n / (2.0 * max(n_pos, 1))
    w_neg = n / (2.0 * max(n_neg, 1))

    sample_weights = np.where(dataset.labels == 1, w_pos, w_neg)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=n,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_eval_loader(
    dataset: EEGDataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """Validation / test DataLoader — sequential, no sampling."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_dataloaders(
    config_path: str = "config.yaml",
    use_cache: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Top-level convenience function: reads config.yaml and returns
    (train_loader, val_loader, test_loader).

    Args:
        config_path: Path to config.yaml
        use_cache:   Use/write .npz cache files

    Returns:
        train_loader, val_loader, test_loader
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

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

    train_pids = cfg["splits"]["train_patients"]
    val_pids   = cfg["splits"]["val_patients"]
    test_pids  = cfg["splits"]["test_patients"]

    batch_size = cfg["training"]["batch_size"]
    seed       = cfg["training"]["seed"]

    common_kwargs = dict(
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
        use_cache=use_cache,
    )

    train_ds = build_split_dataset(patient_ids=train_pids, split_name="train", **common_kwargs)
    val_ds   = build_split_dataset(patient_ids=val_pids,   split_name="val",   **common_kwargs)
    test_ds  = build_split_dataset(patient_ids=test_pids,  split_name="test",  **common_kwargs)

    train_loader = build_train_loader(train_ds, batch_size=batch_size, seed=seed)
    val_loader   = build_eval_loader(val_ds,   batch_size=batch_size * 2)
    test_loader  = build_eval_loader(test_ds,  batch_size=batch_size * 2)

    logger.info("DataLoaders ready.")
    logger.info(f"  Train: {len(train_ds)} windows, {train_ds.seizure_fraction:.2%} seizure")
    logger.info(f"  Val:   {len(val_ds)}   windows, {val_ds.seizure_fraction:.2%} seizure")
    logger.info(f"  Test:  {len(test_ds)}  windows, {test_ds.seizure_fraction:.2%} seizure")

    return train_loader, val_loader, test_loader
