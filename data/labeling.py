"""
data/labeling.py — Labeling functions for LOB price-direction classification.

Supports two label rules (Change 1.4.1):
  - point_return: r_t = (m_{t+H} - m_t) / m_t, threshold at ±α
  - smoothed: r_t = (mean(m_{t+1},...,m_{t+H}) - m_t) / m_t, threshold at ±α
    This is the rule the FI-2010 labels were built with.

Threshold modes (Change 1.4.2):
  - fixed: α given explicitly
  - quantile: α chosen on training split such that classes ≈ balanced (tertiles of |r_t|)
  - std_mult: α = c × std(r_t on train)

Labels are:
  0: Down  (return < -α)
  1: Stationary (-α <= return <= α)
  2: Up    (return > α)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def label_by_threshold(returns: np.ndarray, threshold: float) -> np.ndarray:
    """
    Given a 1D array of returns, labels them:
    0: Down (return < -threshold)
    1: Stationary (-threshold <= return <= threshold)
    2: Up (return > threshold)
    """
    labels = np.ones_like(returns, dtype=np.int64)  # Default to Stationary (1)
    labels[returns > threshold] = 2  # Up
    labels[returns < -threshold] = 0  # Down
    return labels


def compute_mid_price(df: pd.DataFrame) -> pd.Series:
    """Compute mid-price from canonical column names."""
    if 'ask_p_1' in df.columns and 'bid_p_1' in df.columns:
        return (df['ask_p_1'] + df['bid_p_1']) / 2.0
    elif '22' in df.columns and '2' in df.columns:
        # Legacy column names
        return (df['22'] + df['2']) / 2.0
    else:
        raise ValueError(
            "Cannot compute mid-price: expected columns 'ask_p_1'/'bid_p_1' or '22'/'2'"
        )


def compute_returns(mid_price: np.ndarray, horizon: int,
                    label_rule: str = 'point_return') -> np.ndarray:
    """
    Compute returns for labeling.

    Parameters
    ----------
    mid_price : 1D array of mid-prices
    horizon : number of events/snapshots ahead
    label_rule : 'point_return' or 'smoothed'

    Returns
    -------
    returns : 1D array of length len(mid_price) - horizon
    """
    n = len(mid_price)
    if horizon >= n:
        raise ValueError(f"Horizon {horizon} >= data length {n}")

    valid_end = n - horizon
    m_t = mid_price[:valid_end]

    if label_rule == 'point_return':
        # r_t = (m_{t+H} - m_t) / m_t
        m_future = mid_price[horizon:horizon + valid_end]
        returns = (m_future - m_t) / m_t

    elif label_rule == 'smoothed':
        # r_t = (mean(m_{t+1},...,m_{t+H}) - m_t) / m_t
        # Use cumulative sum for vectorized computation
        cumsum = np.concatenate([[0], np.cumsum(mid_price)])
        # m_plus(t) = mean(m_{t+1}, ..., m_{t+H})
        # = (cumsum[t+H+1] - cumsum[t+1]) / H
        # But we index from 0, so for t in [0, valid_end):
        #   m_plus(t) = (cumsum[t+1+H] - cumsum[t+1]) / H
        starts = np.arange(valid_end) + 1
        ends = starts + horizon
        m_plus = (cumsum[ends] - cumsum[starts]) / horizon
        returns = (m_plus - m_t) / m_t

    else:
        raise ValueError(f"Unknown label_rule: {label_rule!r}. Must be 'point_return' or 'smoothed'.")

    return returns


def resolve_threshold(returns_train: np.ndarray, threshold_mode: str,
                      threshold_param: float = 0.0001) -> float:
    """
    Change 1.4.2 — Resolve the labeling threshold α.

    Parameters
    ----------
    returns_train : returns computed on the training split only
    threshold_mode : 'fixed', 'quantile', or 'std_mult'
    threshold_param : α for fixed, c for std_mult, ignored for quantile

    Returns
    -------
    alpha : the resolved threshold
    """
    if threshold_mode == 'fixed':
        return threshold_param

    elif threshold_mode == 'quantile':
        # Choose α such that classes are approximately balanced (tertiles)
        abs_returns = np.abs(returns_train)
        alpha = np.percentile(abs_returns, 100.0 / 3.0)
        logger.info(f"Quantile threshold: α = {alpha:.8f} (33rd percentile of |r_t|)")
        return alpha

    elif threshold_mode == 'std_mult':
        # α = c × std(r_t)
        alpha = threshold_param * np.std(returns_train)
        logger.info(f"Std-mult threshold: c={threshold_param}, std={np.std(returns_train):.8f}, α = {alpha:.8f}")
        return alpha

    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode!r}")


def apply_horizon_labeling(df: pd.DataFrame, horizon: int, threshold: float,
                           label_rule: str = 'point_return',
                           feature_cols: list = None) -> tuple:
    """
    Computes future returns for a given horizon and applies thresholding.
    Returns the feature matrix (X) and label vector (y), truncating the last
    `horizon` rows.

    Parameters
    ----------
    df : DataFrame with canonical column names
    horizon : number of events ahead for the label
    threshold : ±fractional return threshold
    label_rule : 'point_return' or 'smoothed'
    feature_cols : list of column names to extract as features. If None, uses
                   the 40 canonical raw LOB columns.

    Returns
    -------
    X : np.ndarray of shape (n - horizon, n_features)
    y : np.ndarray of shape (n - horizon,)
    """
    mid_price = compute_mid_price(df)
    returns = compute_returns(mid_price.values, horizon, label_rule)
    labels = label_by_threshold(returns, threshold)

    # Extract feature columns
    if feature_cols is None:
        # Try canonical names first, then legacy
        if 'ask_p_1' in df.columns:
            feature_cols = []
            for i in range(1, 11):
                feature_cols.extend([
                    f'ask_p_{i}', f'ask_v_{i}', f'bid_p_{i}', f'bid_v_{i}'
                ])
        else:
            feature_cols = [str(i) for i in range(2, 42)]

    X = df.iloc[:len(labels)][feature_cols].values

    return X, labels


def apply_horizon_labeling_split_aware(
    mid_price: np.ndarray,
    horizon: int,
    label_rule: str,
    threshold_mode: str,
    threshold_param: float,
    train_end: int,
    val_end: int,
    gap: int
) -> tuple:
    """
    Compute labels with split-aware threshold resolution.

    Computes returns on the full series, resolves threshold on train split only,
    then labels everything with the same threshold.

    Returns
    -------
    labels : full label array (with NaN positions for gap/truncated rows)
    alpha : resolved threshold
    returns : full return array
    """
    returns = compute_returns(mid_price, horizon, label_rule)
    n_valid = len(returns)

    # Resolve threshold on train split returns only
    train_returns = returns[:max(0, train_end - gap - horizon)]
    if len(train_returns) == 0:
        raise ValueError("No training returns available for threshold resolution")

    alpha = resolve_threshold(train_returns, threshold_mode, threshold_param)

    # Label everything with the resolved alpha
    labels = label_by_threshold(returns, alpha)

    return labels, alpha, returns


def horizon_events_to_seconds(horizon_events: int, sampling_ms: int = 250) -> float:
    """
    Change 1.4.3 — Convert horizon in events to seconds.
    For crypto data sampled at 250 ms: horizon_events=40 → 10.0 seconds.
    """
    return horizon_events * sampling_ms / 1000.0


def run_threshold_sweep(df: pd.DataFrame, horizons: list, thresholds: list,
                        label_rule: str = 'point_return') -> pd.DataFrame:
    """
    Sweeps horizon × threshold combinations and computes class balances.
    """
    results = []

    mid_price = compute_mid_price(df)
    mid_vals = mid_price.values

    for h in horizons:
        returns = compute_returns(mid_vals, h, label_rule)

        for t in thresholds:
            labels = label_by_threshold(returns, t)
            unique, counts = np.unique(labels, return_counts=True)
            counts_dict = dict(zip(unique, counts))

            total = len(labels)
            p_down = counts_dict.get(0, 0) / total * 100
            p_stat = counts_dict.get(1, 0) / total * 100
            p_up = counts_dict.get(2, 0) / total * 100

            results.append({
                'label_rule': label_rule,
                'horizon_events': h,
                'threshold': t,
                'pct_down': p_down,
                'pct_stationary': p_stat,
                'pct_up': p_up,
                'total_samples': total
            })

    res_df = pd.DataFrame(results)
    return res_df
