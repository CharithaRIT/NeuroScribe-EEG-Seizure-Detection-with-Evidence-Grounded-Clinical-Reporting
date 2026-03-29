"""
trainer.py
----------
Training and evaluation loop for the CNN+GRU seizure detector.

Typical usage:
    from src.training.trainer import run_epoch

    # Training pass (pass optimizer to enable grad + weight updates)
    train_res = run_epoch(model, train_loader, criterion,
                          optimizer=optimizer, device=device)

    # Eval / inference pass (omit optimizer)
    val_res = run_epoch(model, val_loader, criterion, device=device)

    # Access results
    train_res['loss'], train_res['f1'], train_res['sensitivity']
    train_res['probs'], train_res['labels']   # numpy arrays for downstream metrics
"""

import numpy as np
import torch


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict:
    """
    Runs one full pass over a DataLoader.

    Args:
        model:     PyTorch model (CNNGRUClassifier or any binary classifier).
        loader:    DataLoader yielding (X_batch, y_batch) pairs.
        criterion: Loss function (e.g. FocalLoss).
        optimizer: If provided → training mode (gradients + weight update).
                   If None    → eval mode (no_grad, no weight update).
        device:    'cuda' or 'cpu'.
        threshold: Decision threshold for binary predictions.

    Returns:
        dict with keys:
            loss        — mean loss over the epoch
            f1          — F1 score at given threshold
            sensitivity — recall / true positive rate
            precision   — positive predictive value
            probs       — (N,) float32 numpy array of sigmoid probabilities
            labels      — (N,) float32 numpy array of ground-truth labels
    """
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    all_probs: list = []
    all_labels: list = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    probs  = np.array(all_probs,  dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    preds  = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    sensitivity = tp / max(tp + fn, 1)
    precision   = tp / max(tp + fp, 1)
    f1          = 2 * sensitivity * precision / max(sensitivity + precision, 1e-8)

    return {
        "loss":        total_loss / max(len(labels), 1),
        "f1":          f1,
        "sensitivity": sensitivity,
        "precision":   precision,
        "probs":       probs,
        "labels":      labels,
    }
