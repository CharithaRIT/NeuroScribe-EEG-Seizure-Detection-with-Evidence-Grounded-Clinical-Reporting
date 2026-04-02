"""
shared_loaders.py
-----------------
Builds EEG DataLoaders for all three splits.

If pre-saved .npz files exist (from a previous prepare_data.py run) they are
loaded instantly.  If not (e.g. disk quota exceeded), data is built directly
from the raw EDF files in memory — no files are written to disk.

Usage in any notebook:
    from src.data.shared_loaders import get_loaders

    train_loader, val_loader, test_loader, meta = get_loaders()
    # or with explicit subsampling:
    train_loader, val_loader, test_loader, meta = get_loaders(subsampled=True)
"""

import os
import yaml
import numpy as np
from src.data.dataset import (
    EEGDataset,
    build_split_dataset,
    build_train_loader,
    build_eval_loader,
)

SUBSAMPLE_RATIO = 5   # non-seizure : seizure kept per split


def _subsample(ds: EEGDataset, ratio: int = 5, seed: int = 42) -> EEGDataset:
    """Keep all seizure windows; downsample non-seizure to ratio × seizure."""
    np.random.seed(seed)
    s_idx  = np.where(ds.labels == 1)[0]
    ns_idx = np.where(ds.labels == 0)[0]
    ns_idx = np.random.choice(
        ns_idx, min(len(s_idx) * ratio, len(ns_idx)), replace=False
    )
    idx = np.concatenate([s_idx, ns_idx])
    np.random.shuffle(idx)
    return EEGDataset(ds.windows[idx], ds.labels[idx], ds.patient_ids[idx])


def get_loaders(
    config_path: str = 'config.yaml',
    batch_size: int | None = None,
    subsampled: bool = True,
    subsample_ratio: int = SUBSAMPLE_RATIO,
):
    """
    Return (train_loader, val_loader, test_loader, meta).

    Fast path  : loads pre-saved *_subsampled.npz from processed_dir (if they exist).
    Fallback   : builds from raw EDF in memory — nothing written to disk.

    Args:
        config_path:      path to config.yaml
        batch_size:       override config batch_size
        subsampled:       apply seizure/non-seizure subsampling (default True)
        subsample_ratio:  non-seizure : seizure ratio used when subsampled=True
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir       = cfg['data']['raw_dir']
    processed_dir = cfg['data']['processed_dir']
    channels      = cfg['data']['channels']
    window_sec    = cfg['data']['window_size']
    fs            = cfg['data']['sample_rate']
    overlap       = cfg['data']['overlap']
    sz_thresh     = cfg['data']['seizure_threshold']
    bp_low        = cfg['preprocessing']['bandpass_low']
    bp_high       = cfg['preprocessing']['bandpass_high']
    notch         = cfg['preprocessing']['notch_freq']
    seed          = cfg['training']['seed']
    bs            = batch_size or cfg['training']['batch_size']

    split_pids = {
        'train': cfg['splits']['train_patients'],
        'val':   cfg['splits']['val_patients'],
        'test':  cfg['splits']['test_patients'],
    }

    def _load_split(split: str) -> EEGDataset:
        # ── Fast path: pre-saved .npz ──────────────────────────────────
        npz_path = os.path.join(processed_dir, f'{split}_subsampled.npz')
        if subsampled and os.path.exists(npz_path):
            d = np.load(npz_path)
            print(f'  [{split}] Loaded from cache: {npz_path}')
            return EEGDataset(d['windows'], d['labels'], d['patient_ids'])

        # ── Fallback: build from raw EDF, no disk write ────────────────
        print(f'  [{split}] Building from raw EDF (no disk write)...')
        ds = build_split_dataset(
            raw_dir=raw_dir,
            patient_ids=split_pids[split],
            split_name=split,
            target_channels=channels,
            window_size_sec=window_sec,
            overlap=overlap,
            seizure_threshold=sz_thresh,
            bandpass_low=bp_low,
            bandpass_high=bp_high,
            notch_freq=notch,
            sample_rate=fs,
            processed_dir=None,   # ← no caching to disk
            use_cache=False,
        )
        print(f'    {len(ds):,} windows  seizure={ds.n_seizure:,} ({ds.seizure_fraction:.2%})')

        if subsampled:
            ds = _subsample(ds, ratio=subsample_ratio, seed=seed)
            print(f'    After subsample: {len(ds):,} windows  seizure={ds.n_seizure:,} ({ds.seizure_fraction:.2%})')

        return ds

    train_ds = _load_split('train')
    val_ds   = _load_split('val')
    test_ds  = _load_split('test')

    train_loader = build_train_loader(train_ds, batch_size=bs, seed=seed)
    val_loader   = build_eval_loader(val_ds,   batch_size=bs * 2)
    test_loader  = build_eval_loader(test_ds,  batch_size=bs * 2)

    meta = {
        'train': {'n': len(train_ds), 'n_seizure': train_ds.n_seizure,
                  'seizure_frac': train_ds.seizure_fraction},
        'val':   {'n': len(val_ds),   'n_seizure': val_ds.n_seizure,
                  'seizure_frac': val_ds.seizure_fraction},
        'test':  {'n': len(test_ds),  'n_seizure': test_ds.n_seizure,
                  'seizure_frac': test_ds.seizure_fraction},
        'batch_size': bs,
    }
    return train_loader, val_loader, test_loader, meta
