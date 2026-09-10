"""
tests/test_features.py — Tests for engineered feature computation (Change 8).

Tests that engineered features are computed correctly on synthetic data.
FI-2010 column validation (Change 1.5.2) requires the actual data and is
run separately via prepare_fi2010.py.
"""

import pytest
import numpy as np
import pandas as pd

from data.features import (
    compute_spread_mid, compute_price_differences, compute_means,
    compute_accumulated_differences, compute_derivatives,
    compute_ob_imbalance, compute_engineered_features,
    build_feature_matrix, to_relative_price, CANONICAL_RAW_COLS
)


@pytest.fixture
def synthetic_lob():
    """Create a synthetic LOB DataFrame with canonical column names."""
    np.random.seed(42)
    n = 50

    data = {}
    for i in range(1, 11):
        # Ask prices increase with level, bid decrease
        data[f'ask_p_{i}'] = 100.0 + i * 0.01 + np.random.randn(n) * 0.001
        data[f'ask_v_{i}'] = np.abs(np.random.randn(n) * 100) + 1
        data[f'bid_p_{i}'] = 100.0 - i * 0.01 + np.random.randn(n) * 0.001
        data[f'bid_v_{i}'] = np.abs(np.random.randn(n) * 100) + 1

    return pd.DataFrame(data)


def test_spread_mid_count(synthetic_lob):
    """Should produce 20 features (spread + mid × 10 levels)."""
    features, names = compute_spread_mid(synthetic_lob)
    assert len(names) == 20
    assert len(features) == 20


def test_spread_positive(synthetic_lob):
    """Spread should be positive (ask > bid)."""
    features, _ = compute_spread_mid(synthetic_lob)
    for i in range(1, 11):
        spread = features[f'spread_{i}']
        assert np.all(spread > 0), f"Spread at level {i} has non-positive values"


def test_mid_between_bid_ask(synthetic_lob):
    """Mid should be between bid and ask."""
    features, _ = compute_spread_mid(synthetic_lob)
    for i in range(1, 11):
        mid = features[f'mid_{i}']
        ask = synthetic_lob[f'ask_p_{i}'].values
        bid = synthetic_lob[f'bid_p_{i}'].values
        assert np.all(mid >= bid), f"Mid below bid at level {i}"
        assert np.all(mid <= ask), f"Mid above ask at level {i}"


def test_means_count(synthetic_lob):
    """Should produce exactly 4 features."""
    features, names = compute_means(synthetic_lob)
    assert len(names) == 4
    assert 'mean_ask_price' in names
    assert 'mean_bid_price' in names
    assert 'mean_ask_volume' in names
    assert 'mean_bid_volume' in names


def test_derivatives_shape(synthetic_lob):
    """Derivatives should have same shape as input."""
    raw = synthetic_lob[CANONICAL_RAW_COLS].values
    derivs, names = compute_derivatives(raw, CANONICAL_RAW_COLS)
    assert derivs.shape == raw.shape
    assert len(names) == 40


def test_derivatives_first_row_zero(synthetic_lob):
    """First row of derivatives should be 0 (no previous snapshot)."""
    raw = synthetic_lob[CANONICAL_RAW_COLS].values
    derivs, _ = compute_derivatives(raw, CANONICAL_RAW_COLS)
    assert np.all(derivs[0] == 0)


def test_ob_imbalance_range(synthetic_lob):
    """OB imbalance should be in [-1, 1]."""
    features, names = compute_ob_imbalance(synthetic_lob)
    assert len(names) == 10
    for name in names:
        imb = features[name]
        assert np.all(imb >= -1) and np.all(imb <= 1), \
            f"OB imbalance {name} out of range [-1, 1]"


def test_engineered_features_combined(synthetic_lob):
    """compute_engineered_features should produce a valid array."""
    eng_array, eng_names = compute_engineered_features(synthetic_lob)
    assert eng_array.shape[0] == len(synthetic_lob)
    assert eng_array.shape[1] == len(eng_names)
    assert not np.any(np.isnan(eng_array)), "NaN in engineered features"


def test_build_feature_matrix_raw40(synthetic_lob):
    """raw40 should return exactly 40 features."""
    X, names = build_feature_matrix(synthetic_lob, 'raw40')
    assert X.shape == (len(synthetic_lob), 40)
    assert len(names) == 40


def test_build_feature_matrix_raw40_eng(synthetic_lob):
    """raw40_eng should return 40 + engineered features."""
    X, names = build_feature_matrix(synthetic_lob, 'raw40_eng')
    assert X.shape[0] == len(synthetic_lob)
    assert X.shape[1] > 40
    assert len(names) == X.shape[1]


def test_relative_price_transform(synthetic_lob):
    """Relative price should center prices around 0."""
    result = to_relative_price(synthetic_lob)
    # Best bid/ask relative prices should be small numbers near 0
    ask_rel = result['ask_p_1'].values
    bid_rel = result['bid_p_1'].values
    assert np.all(np.abs(ask_rel) < 0.1), "Relative ask price too large"
    assert np.all(np.abs(bid_rel) < 0.1), "Relative bid price too large"
