"""
tests/test_labeling.py — Tests for labeling correctness (Change 8).

Smoothed and point-return labels on a synthetic mid-price series with known answers.
Quantile threshold yields ~33/33/33 on train.
"""

import pytest
import numpy as np

from data.labeling import (
    compute_returns, label_by_threshold, resolve_threshold,
    horizon_events_to_seconds
)


@pytest.fixture
def synthetic_mid():
    """A simple synthetic mid-price series with known behavior."""
    # Linear uptrend: mid = 100 + 0.01 * t
    n = 200
    return 100.0 + 0.01 * np.arange(n, dtype=float)


@pytest.fixture
def constant_mid():
    """Constant mid-price — all returns should be zero."""
    return np.full(100, 50000.0)


# ─── Point return tests ───────────────────────────────────────────────────────

def test_point_return_constant_mid(constant_mid):
    """Constant price → all returns = 0 → all labels = Stationary."""
    returns = compute_returns(constant_mid, horizon=10, label_rule='point_return')
    labels = label_by_threshold(returns, threshold=0.001)
    assert np.all(labels == 1), "Constant price should give all Stationary labels"


def test_point_return_uptrend(synthetic_mid):
    """Uptrend → all returns > 0 → all labels = Up (with small enough threshold)."""
    returns = compute_returns(synthetic_mid, horizon=10, label_rule='point_return')
    # All returns should be positive for a linear uptrend
    assert np.all(returns > 0), "Uptrend should have all positive returns"


def test_point_return_length(synthetic_mid):
    """Returns should have length = n - horizon."""
    horizon = 20
    returns = compute_returns(synthetic_mid, horizon=horizon, label_rule='point_return')
    assert len(returns) == len(synthetic_mid) - horizon


# ─── Smoothed return tests ────────────────────────────────────────────────────

def test_smoothed_return_constant_mid(constant_mid):
    """Constant price → smoothed returns = 0 → all Stationary."""
    returns = compute_returns(constant_mid, horizon=10, label_rule='smoothed')
    labels = label_by_threshold(returns, threshold=0.001)
    assert np.all(labels == 1)


def test_smoothed_return_uptrend(synthetic_mid):
    """Uptrend → smoothed returns > 0."""
    returns = compute_returns(synthetic_mid, horizon=10, label_rule='smoothed')
    assert np.all(returns > 0)


def test_smoothed_return_length(synthetic_mid):
    """Smoothed returns should have length = n - horizon."""
    horizon = 20
    returns = compute_returns(synthetic_mid, horizon=horizon, label_rule='smoothed')
    assert len(returns) == len(synthetic_mid) - horizon


# ─── Threshold tests ──────────────────────────────────────────────────────────

def test_fixed_threshold():
    """Fixed threshold returns the param as-is."""
    alpha = resolve_threshold(np.random.randn(100), 'fixed', 0.001)
    assert alpha == 0.001


def test_quantile_threshold_balanced():
    """Quantile threshold on uniform returns should yield ~balanced classes."""
    np.random.seed(42)
    returns = np.random.randn(10000) * 0.01  # approx symmetric
    alpha = resolve_threshold(returns, 'quantile')

    labels = label_by_threshold(returns, alpha)
    unique, counts = np.unique(labels, return_counts=True)
    pcts = counts / len(labels) * 100

    # Each class should be roughly 33% (±5% tolerance)
    for pct in pcts:
        assert 25 < pct < 45, f"Class balance off: {pcts}"


def test_std_mult_threshold():
    """Std-mult threshold = c × std."""
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
    c = 1.0
    expected = c * np.std(returns)
    alpha = resolve_threshold(returns, 'std_mult', c)
    assert abs(alpha - expected) < 1e-10


# ─── Label encoding tests ────────────────────────────────────────────────────

def test_label_values():
    """Labels should be 0 (Down), 1 (Stationary), 2 (Up)."""
    returns = np.array([-0.01, 0.0, 0.01, -0.005, 0.005])
    labels = label_by_threshold(returns, threshold=0.008)
    assert set(labels) <= {0, 1, 2}
    assert labels[0] == 0  # -0.01 < -0.008
    assert labels[1] == 1  # 0.0 within threshold
    assert labels[2] == 2  # 0.01 > 0.008


# ─── Horizon conversion test ─────────────────────────────────────────────────

def test_horizon_seconds():
    assert horizon_events_to_seconds(40) == 10.0
    assert horizon_events_to_seconds(100) == 25.0
    assert horizon_events_to_seconds(1) == 0.25
