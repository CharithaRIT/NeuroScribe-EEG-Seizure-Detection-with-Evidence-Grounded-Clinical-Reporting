"""
prepare_data.py
---------------
Run ONCE from the project root to build and save all subsampled splits.

    python scripts/prepare_data.py

Saves to data/processed/:
    train_subsampled.npz
    val_subsampled.npz
    test_subsampled.npz

After this, all notebooks load in < 30 seconds via shared_loaders.get_loaders().
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
from src.data.dataset import build_split_dataset, EEGDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
with open(os.path.join(ROOT, 'config.yaml')) as f:
    cfg = yaml.safe_load(f)

RAW_DIR       = cfg['data']['raw_dir']
PROCESSED_DIR = cfg['data']['processed_dir']
CHANNELS      = cfg['data']['channels']
WINDOW_SEC    = cfg['data']['window_size']
FS            = cfg['data']['sample_rate']
SEED          = cfg['training']['seed']
SUBSAMPLE_RATIO = 5

os.makedirs(PROCESSED_DIR, exist_ok=True)
print(f'Output dir: {PROCESSED_DIR}\n')


def build(split_name, patient_ids):
    print(f'[{split_name}] Loading from raw EDF (no cache)...')
    ds = build_split_dataset(
        raw_dir=RAW_DIR,
        patient_ids=patient_ids,
        split_name=split_name,
        target_channels=CHANNELS,
        window_size_sec=WINDOW_SEC,
        overlap=0.0,
        sample_rate=FS,
        processed_dir=None,   # no internal caching
        use_cache=False,
    )
    print(f'  {len(ds):,} windows  seizure={ds.n_seizure:,} ({ds.seizure_fraction:.2%})')
    return ds


def subsample(ds, ratio=5, seed=42):
    np.random.seed(seed)
    s_idx  = np.where(ds.labels == 1)[0]
    ns_idx = np.where(ds.labels == 0)[0]
    ns_idx = np.random.choice(ns_idx, min(len(s_idx)*ratio, len(ns_idx)), replace=False)
    idx    = np.concatenate([s_idx, ns_idx])
    np.random.shuffle(idx)
    return EEGDataset(ds.windows[idx], ds.labels[idx], ds.patient_ids[idx])


def save(ds, name):
    path = os.path.join(PROCESSED_DIR, f'{name}_subsampled.npz')
    np.savez_compressed(path, windows=ds.windows, labels=ds.labels,
                        patient_ids=ds.patient_ids)
    mb = os.path.getsize(path) / 1e6
    print(f'  Saved → {path}  ({mb:.1f} MB)')


for split, pids in [('train', list(range(1,19))), ('val', [19,20,21]), ('test', [22,23,24])]:
    ds  = build(split, pids)
    sub = subsample(ds, ratio=SUBSAMPLE_RATIO, seed=SEED)
    save(sub, split)
    print(f'  After subsample: {len(sub):,} windows  seizure={sub.n_seizure:,} ({sub.seizure_fraction:.2%})\n')

print('Done. Run any notebook — data loads in < 30 seconds.')
