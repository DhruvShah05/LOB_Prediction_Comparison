"""
data/sequences.py — Windowed dataset for temporal models (Change 1.6.1).

Provides a torch.utils.data.Dataset that returns (X[t-T+1:t+1], y[t]) for
temporal models like DeepLOB, windowed Transformer, and LevelTransformer.

Key design decisions (per spec):
  - Indices are built per split so a window never spans a split boundary.
  - Uses memory-view / slicing (no copying the array T times).
  - Tree models continue to use single-snapshot; a flatten_window option can
    give trees the last T' snapshots concatenated as an ablation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedLOBDataset(Dataset):
    """
    Sliding-window dataset for temporal LOB models.

    Parameters
    ----------
    X : np.ndarray of shape (n, F)
        Feature matrix for a single split (train, val, or test).
    y : np.ndarray of shape (n,)
        Label vector for the same split.
    window_len : int
        Number of snapshots per window (T). Default 100.
    flatten : bool
        If True, flatten the window to (T*F,) for tree-based models.
        If False, return (T, F) for neural models.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window_len: int = 100,
                 flatten: bool = False):
        assert len(X) == len(y), f"X ({len(X)}) and y ({len(y)}) must have same length"
        assert window_len >= 1, f"window_len must be >= 1, got {window_len}"

        self.X = X
        self.y = y
        self.window_len = window_len
        self.flatten = flatten

        # Valid indices: t such that t >= T-1 (so window [t-T+1 : t+1] is within bounds)
        self.valid_start = window_len - 1
        self.n_samples = len(X) - self.valid_start

        if self.n_samples <= 0:
            raise ValueError(
                f"window_len={window_len} is larger than data length {len(X)}. "
                "No valid samples can be created."
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Map idx to actual position in the data
        t = idx + self.valid_start

        # Extract window: X[t-T+1 : t+1] = X[t - window_len + 1 : t + 1]
        window_start = t - self.window_len + 1
        window = self.X[window_start:t + 1]  # shape (T, F)

        label = self.y[t]

        if self.flatten:
            # Flatten for tree models: (T * F,)
            window = window.reshape(-1)

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


class SingleSnapshotDataset(Dataset):
    """
    Simple dataset for non-temporal models (tree, logistic).
    Returns (X[t], y[t]) as a single snapshot.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), f"X ({len(X)}) and y ({len(y)}) must have same length"
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


def create_datasets(X_train, y_train, X_val, y_val, X_test, y_test,
                    window_len: int = 1, flatten: bool = False):
    """
    Factory function to create appropriate datasets.

    Parameters
    ----------
    window_len : 1 for single-snapshot (trees), >1 for windowed (temporal models)
    flatten : if True and window_len > 1, flatten the window for tree ablation

    Returns
    -------
    train_ds, val_ds, test_ds : torch Dataset instances
    """
    if window_len <= 1:
        return (
            SingleSnapshotDataset(X_train, y_train),
            SingleSnapshotDataset(X_val, y_val),
            SingleSnapshotDataset(X_test, y_test),
        )
    else:
        return (
            WindowedLOBDataset(X_train, y_train, window_len, flatten),
            WindowedLOBDataset(X_val, y_val, window_len, flatten),
            WindowedLOBDataset(X_test, y_test, window_len, flatten),
        )
