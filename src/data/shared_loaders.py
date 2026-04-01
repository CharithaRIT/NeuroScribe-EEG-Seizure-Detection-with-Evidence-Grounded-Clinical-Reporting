"""
shared_loaders.py
-----------------
Loads pre-saved subsampled splits from .npz files.
Requires scripts/prepare_data.py to have been run once.

Usage in any notebook:
    from src.data.shared_loaders import get_loaders

    train_loader, val_loader, test_loader, meta = get_loaders()
"""

import os
import yaml
import numpy as np
from src.data.dataset import EEGDataset, build_train_loader, build_eval_loader


def get_loaders(config_path: str = 'config.yaml', batch_size: int | None = None):
    """
    Load subsampled EEG DataLoaders from .npz files saved by prepare_data.py.

    Returns:
        (train_loader, val_loader, test_loader, meta)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg['data']['processed_dir']
    bs = batch_size or cfg['training']['batch_size']

    def load(split):
        path = os.path.join(processed_dir, f'{split}_subsampled.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'\nNot found: {path}\n'
                f'Run this once from the project root:\n'
                f'  python scripts/prepare_data.py\n'
            )
        d = np.load(path)
        return EEGDataset(d['windows'], d['labels'], d['patient_ids'])

    train_ds = load('train')
    val_ds   = load('val')
    test_ds  = load('test')

    train_loader = build_train_loader(train_ds, batch_size=bs)
    val_loader   = build_eval_loader(val_ds,   batch_size=bs * 2)
    test_loader  = build_eval_loader(test_ds,  batch_size=bs * 2)

    meta = {
        'train': {'n': len(train_ds), 'n_seizure': train_ds.n_seizure, 'seizure_frac': train_ds.seizure_fraction},
        'val':   {'n': len(val_ds),   'n_seizure': val_ds.n_seizure,   'seizure_frac': val_ds.seizure_fraction},
        'test':  {'n': len(test_ds),  'n_seizure': test_ds.n_seizure,  'seizure_frac': test_ds.seizure_fraction},
        'batch_size': bs,
    }
    return train_loader, val_loader, test_loader, meta
