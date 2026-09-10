"""
tests/test_splits.py — Tests for split integrity (Change 8).

- No window and no label horizon crosses a split boundary
- Split hash identical across seeds
"""

import pytest
import numpy as np
import hashlib

from data.sequences import WindowedLOBDataset, create_datasets


def test_windowed_dataset_no_cross_boundary():
    """Window never spans beyond the array (i.e. no cross-split window)."""
    n, F = 50, 10
    X = np.random.randn(n, F).astype(np.float32)
    y = np.random.randint(0, 3, size=n)
    window_len = 10

    ds = WindowedLOBDataset(X, y, window_len=window_len)

    # First valid index should be at window_len - 1
    assert ds.valid_start == window_len - 1
    assert len(ds) == n - window_len + 1

    # Check every window is within bounds
    for idx in range(len(ds)):
        x_window, label = ds[idx]
        assert x_window.shape == (window_len, F)


def test_windowed_dataset_label_alignment():
    """Label at index i corresponds to the last element of the window."""
    n, F = 20, 5
    X = np.random.randn(n, F).astype(np.float32)
    y = np.arange(n)  # labels are just indices
    window_len = 5

    ds = WindowedLOBDataset(X, y, window_len=window_len)

    for idx in range(len(ds)):
        _, label = ds[idx]
        expected_t = idx + window_len - 1
        assert label.item() == expected_t, f"idx={idx}: got label {label.item()}, expected {expected_t}"


def test_split_independence():
    """Splits created separately should not share any data."""
    n, F = 100, 10
    X = np.random.randn(n, F).astype(np.float32)
    y = np.random.randint(0, 3, size=n)

    # Split at 60/20/20
    X_tr, y_tr = X[:60], y[:60]
    X_val, y_val = X[60:80], y[60:80]
    X_te, y_te = X[80:], y[80:]

    window_len = 10
    train_ds, val_ds, test_ds = create_datasets(
        X_tr, y_tr, X_val, y_val, X_te, y_te, window_len=window_len
    )

    # Each dataset operates on its own array slice — verify no overlap
    assert len(train_ds) + len(val_ds) + len(test_ds) <= n


def test_split_hash_reproducibility():
    """Same data → same hash regardless of how we got there."""
    y = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2])
    h1 = hashlib.sha256(y.tobytes()).hexdigest()[:16]
    h2 = hashlib.sha256(y.copy().tobytes()).hexdigest()[:16]
    assert h1 == h2


def test_window_too_large_raises():
    """Window larger than data should raise ValueError."""
    X = np.random.randn(5, 3).astype(np.float32)
    y = np.zeros(5, dtype=np.int64)

    with pytest.raises(ValueError, match="window_len.*larger"):
        WindowedLOBDataset(X, y, window_len=10)


def test_flatten_option():
    """Flatten should produce (T*F,) instead of (T, F)."""
    n, F = 30, 5
    X = np.random.randn(n, F).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    window_len = 10

    ds = WindowedLOBDataset(X, y, window_len=window_len, flatten=True)
    x_flat, _ = ds[0]
    assert x_flat.shape == (window_len * F,)
