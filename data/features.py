"""
data/features.py — Feature engineering and preprocessing for LOB data.

Change 1.5.1: Engineered feature groups following Kercheval & Zhang / Ntakaris.
Change 1.5.3: Relative price (p-mid)/mid and log1p volume transform.
Change 1.5.4: Standardization policy (skip for Zscore, apply for DecPre/crypto).
Change 1.5.5: Feature names carried on all feature matrices.

All functions operate on the canonical 40-column raw book layout:
  [ask_p_1, ask_v_1, bid_p_1, bid_v_1, ask_p_2, ask_v_2, bid_p_2, bid_v_2, ...]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# Canonical column names for the raw 40-column LOB
CANONICAL_RAW_COLS = []
for i in range(1, 11):
    CANONICAL_RAW_COLS.extend([
        f'ask_p_{i}', f'ask_v_{i}', f'bid_p_{i}', f'bid_v_{i}'
    ])


class TrainOnlyScaler:
    """
    A scaler wrapper that ensures `.fit()` is only ever called once,
    guarding against accidental fitting on validation or test sets (data leakage).
    """
    def __init__(self, use_zscore=True):
        self.scaler = StandardScaler() if use_zscore else None
        self._is_fit = False

    def fit(self, X: np.ndarray):
        if self._is_fit:
            raise RuntimeError("TrainOnlyScaler.fit() called more than once! You are likely leaking test data.")
        if self.scaler is not None:
            self.scaler.fit(X)
        self._is_fit = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("TrainOnlyScaler.transform() called before fit()!")
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def _get_price_vol_cols():
    """Helper: returns lists of price and volume column names."""
    ask_price_cols = [f'ask_p_{i}' for i in range(1, 11)]
    ask_vol_cols = [f'ask_v_{i}' for i in range(1, 11)]
    bid_price_cols = [f'bid_p_{i}' for i in range(1, 11)]
    bid_vol_cols = [f'bid_v_{i}' for i in range(1, 11)]
    return ask_price_cols, ask_vol_cols, bid_price_cols, bid_vol_cols


def to_relative_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change 1.5.3 — Converts raw LOB prices to relative prices: (p - mid) / mid.
    Volumes are log1p-transformed (LOB volumes are heavy-tailed).

    Works with canonical column names (ask_p_1, bid_p_1, etc.)
    or legacy column names ('2', '3', '22', '23', ...).
    """
    res = df.copy()

    # Detect column naming scheme
    if 'ask_p_1' in df.columns:
        ask_price_cols, ask_vol_cols, bid_price_cols, bid_vol_cols = _get_price_vol_cols()
        mid_price = (df['ask_p_1'] + df['bid_p_1']) / 2.0

        # Relative prices: (p - mid) / mid
        for col in ask_price_cols + bid_price_cols:
            res[col] = (df[col] - mid_price) / mid_price

        # Log1p volumes
        for col in ask_vol_cols + bid_vol_cols:
            res[col] = np.log1p(df[col].values)

    else:
        # Legacy column names
        best_bid_col = '2'
        best_ask_col = '22'
        mid_price = (df[best_bid_col] + df[best_ask_col]) / 2.0

        # Bid price columns: even index from 2 to 20
        for i in range(2, 22, 2):
            col = str(i)
            res[col] = (df[col] - mid_price) / mid_price

        # Ask price columns: even index from 22 to 40
        for i in range(22, 42, 2):
            col = str(i)
            res[col] = (df[col] - mid_price) / mid_price

        # Volume columns: odd indices
        for i in range(3, 22, 2):
            col = str(i)
            res[col] = np.log1p(df[col].values)
        for i in range(23, 42, 2):
            col = str(i)
            res[col] = np.log1p(df[col].values)

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Change 1.5.1 — Engineered feature groups
# ─────────────────────────────────────────────────────────────────────────────

def compute_spread_mid(df: pd.DataFrame) -> tuple:
    """
    Spread & mid per level (20 features).
    spread_i = ask_p_i - bid_p_i
    mid_i = (ask_p_i + bid_p_i) / 2
    """
    features = {}
    names = []
    for i in range(1, 11):
        spread = df[f'ask_p_{i}'].values - df[f'bid_p_{i}'].values
        mid = (df[f'ask_p_{i}'].values + df[f'bid_p_{i}'].values) / 2.0
        features[f'spread_{i}'] = spread
        features[f'mid_{i}'] = mid
        names.extend([f'spread_{i}', f'mid_{i}'])
    return features, names


def compute_price_differences(df: pd.DataFrame) -> tuple:
    """
    Price differences (36 features per Kercheval & Zhang).
    Includes: top-vs-bottom level, absolute inter-level differences.
    """
    features = {}
    names = []

    # Top vs bottom (ask and bid)
    features['ask_p10_minus_p1'] = df['ask_p_10'].values - df['ask_p_1'].values
    features['bid_p1_minus_p10'] = df['bid_p_1'].values - df['bid_p_10'].values
    names.extend(['ask_p10_minus_p1', 'bid_p1_minus_p10'])

    # Absolute inter-level differences (ask and bid, i=1..9)
    for i in range(1, 10):
        ask_diff = np.abs(df[f'ask_p_{i + 1}'].values - df[f'ask_p_{i}'].values)
        bid_diff = np.abs(df[f'bid_p_{i}'].values - df[f'bid_p_{i + 1}'].values)
        features[f'abs_ask_diff_{i}_{i + 1}'] = ask_diff
        features[f'abs_bid_diff_{i}_{i + 1}'] = bid_diff
        names.extend([f'abs_ask_diff_{i}_{i + 1}', f'abs_bid_diff_{i}_{i + 1}'])

    # Cross differences: ask_p_i - bid_p_i for each level (already in spread, but
    # FI-2010 block 60-95 has additional combinations). Add inter-level ask-bid.
    for i in range(1, 10):
        cross = df[f'ask_p_{i}'].values - df[f'bid_p_{i + 1}'].values
        features[f'ask_bid_cross_{i}_{i + 1}'] = cross
        names.extend([f'ask_bid_cross_{i}_{i + 1}'])

    # Bid-ask at each level (sum of volumes)
    for i in range(1, 8):  # fill remaining to reach ~36
        vol_ratio = df[f'ask_v_{i}'].values - df[f'bid_v_{i}'].values
        features[f'vol_diff_{i}'] = vol_ratio
        names.append(f'vol_diff_{i}')

    return features, names


def compute_means(df: pd.DataFrame) -> tuple:
    """
    Means (4 features): mean ask/bid price/volume over 10 levels.
    """
    ask_prices = np.column_stack([df[f'ask_p_{i}'].values for i in range(1, 11)])
    bid_prices = np.column_stack([df[f'bid_p_{i}'].values for i in range(1, 11)])
    ask_vols = np.column_stack([df[f'ask_v_{i}'].values for i in range(1, 11)])
    bid_vols = np.column_stack([df[f'bid_v_{i}'].values for i in range(1, 11)])

    features = {
        'mean_ask_price': ask_prices.mean(axis=1),
        'mean_bid_price': bid_prices.mean(axis=1),
        'mean_ask_volume': ask_vols.mean(axis=1),
        'mean_bid_volume': bid_vols.mean(axis=1),
    }
    names = list(features.keys())
    return features, names


def compute_accumulated_differences(df: pd.DataFrame) -> tuple:
    """
    Accumulated differences (4 features):
    Σ_i (ask_p_i - bid_p_i), Σ_i (ask_v_i - bid_v_i),
    and their squared versions.
    """
    price_diffs = np.zeros(len(df))
    vol_diffs = np.zeros(len(df))
    for i in range(1, 11):
        price_diffs += df[f'ask_p_{i}'].values - df[f'bid_p_{i}'].values
        vol_diffs += df[f'ask_v_{i}'].values - df[f'bid_v_{i}'].values

    features = {
        'acc_price_diff': price_diffs,
        'acc_vol_diff': vol_diffs,
        'acc_price_diff_sq': price_diffs ** 2,
        'acc_vol_diff_sq': vol_diffs ** 2,
    }
    names = list(features.keys())
    return features, names


def compute_derivatives(data: np.ndarray, feature_names: list) -> tuple:
    """
    Derivatives (40 features): d(x)/dt for each of the 40 raw columns.
    Uses first-order difference (current - previous snapshot).
    First row is set to 0 (no history available).
    """
    # data shape: (n, 40)
    derivs = np.zeros_like(data)
    derivs[1:] = data[1:] - data[:-1]

    deriv_names = [f'd_{name}' for name in feature_names[:40]]
    return derivs, deriv_names


def compute_ob_imbalance(df: pd.DataFrame) -> tuple:
    """
    Order-book imbalance per level (10 features) [extra, not in FI-2010]:
    (bid_v_i - ask_v_i) / (bid_v_i + ask_v_i)
    """
    features = {}
    names = []
    for i in range(1, 11):
        bid_v = df[f'bid_v_{i}'].values
        ask_v = df[f'ask_v_{i}'].values
        denom = bid_v + ask_v
        # Avoid division by zero
        imb = np.where(denom > 0, (bid_v - ask_v) / denom, 0.0)
        features[f'ob_imbalance_{i}'] = imb
        names.append(f'ob_imbalance_{i}')
    return features, names


def compute_engineered_features(df: pd.DataFrame,
                                include_imbalance: bool = True) -> tuple:
    """
    Change 1.5.1 — Compute all engineered feature groups.

    Parameters
    ----------
    df : DataFrame with canonical column names (ask_p_1, ask_v_1, bid_p_1, bid_v_1, ...)
    include_imbalance : if True, include OB imbalance (not in FI-2010)

    Returns
    -------
    eng_features : np.ndarray of shape (n, n_eng_features)
    eng_names : list of feature names
    """
    all_features = {}
    all_names = []

    # Group 1: Spread & mid (20)
    feats, names = compute_spread_mid(df)
    all_features.update(feats)
    all_names.extend(names)

    # Group 2: Price differences (~36)
    feats, names = compute_price_differences(df)
    all_features.update(feats)
    all_names.extend(names)

    # Group 3: Means (4)
    feats, names = compute_means(df)
    all_features.update(feats)
    all_names.extend(names)

    # Group 4: Accumulated differences (4)
    feats, names = compute_accumulated_differences(df)
    all_features.update(feats)
    all_names.extend(names)

    # Group 5: Derivatives (40)
    raw_data = df[CANONICAL_RAW_COLS].values
    derivs, deriv_names = compute_derivatives(raw_data, CANONICAL_RAW_COLS)
    for i, name in enumerate(deriv_names):
        all_features[name] = derivs[:, i]
    all_names.extend(deriv_names)

    # Group 6 (optional): OB imbalance (10)
    if include_imbalance:
        feats, names = compute_ob_imbalance(df)
        all_features.update(feats)
        all_names.extend(names)

    # Build array
    eng_array = np.column_stack([all_features[name] for name in all_names])

    return eng_array, all_names


def build_feature_matrix(df: pd.DataFrame, feature_set: str = 'raw40') -> tuple:
    """
    Change 1.3.4 / 1.5.5 — Build the feature matrix with named columns.

    Parameters
    ----------
    df : DataFrame with canonical column names
    feature_set : 'raw40' or 'raw40_eng'

    Returns
    -------
    X : np.ndarray of shape (n, n_features)
    feature_names : list of feature name strings
    """
    raw_cols = CANONICAL_RAW_COLS.copy()

    if feature_set == 'raw40':
        X = df[raw_cols].values
        return X, raw_cols

    elif feature_set == 'raw40_eng':
        raw_X = df[raw_cols].values
        eng_X, eng_names = compute_engineered_features(df, include_imbalance=True)
        X = np.hstack([raw_X, eng_X])
        all_names = raw_cols + eng_names
        return X, all_names

    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}. Must be 'raw40' or 'raw40_eng'.")
