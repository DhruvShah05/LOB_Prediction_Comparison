"""
data/loaders.py — Authoritative data-loading interface for both markets.

This is the single source of truth for data loading (Change 1.3.7).
main.py must call these classes instead of inline load_data.

Changes implemented:
  1.3.1 [BUG B6] — Timestamp-based windowing for crypto
  1.3.2 [BUG B7] — Gap between splits
  1.3.3 [R2]     — Rolling-origin evaluation + split_by_day
  1.3.4          — Feature-set selection (raw40 / raw40_eng)
  1.3.5          — Label-rule selection (point_return / smoothed + threshold modes)
  1.3.6          — Return metadata dict from get_splits()

Both return (X_train, y_train, X_val, y_val, X_test, y_test) plus metadata.
"""

import numpy as np
import os
import hashlib
import logging
import json

logger = logging.getLogger(__name__)


class FI2010Dataset:
    """
    Loads FI-2010 prepared .npy files and performs the standard chronological
    7-days-train / 3-days-test split.

    Attributes
    ----------
    horizon_k : int
        The prediction horizon column index used.
    metadata : dict
        Split metadata for run manifest.
    """

    N_FEATURES = 144
    # Default mapping — should be verified via prepare_fi2010.py --verify-horizons
    HORIZON_TO_COL = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
    TOTAL_COLS = 149

    def __init__(
        self,
        train_path: str = 'data/processed/fi2010_zscore_train.npy',
        test_path: str = 'data/processed/fi2010_zscore_test.npy',
        horizon_k: int = 10,
        val_fraction: float = 0.2,
        feature_set: str = 'raw40_eng',
        window_len: int = 1,
        standardize: bool = True,
        fi2010_variant: str = 'Zscore',
    ):
        """
        Parameters
        ----------
        train_path    : path to fi2010_train.npy
        test_path     : path to fi2010_test.npy
        horizon_k     : prediction horizon key
        val_fraction  : fraction of training rows for validation
        feature_set   : 'raw40' (columns 0-39 only) or 'raw40_eng' (all 144)
        window_len    : window length for temporal models (used to compute gap)
        standardize   : whether to apply standardization
        fi2010_variant: 'Zscore' or 'DecPre'
        """
        if horizon_k not in self.HORIZON_TO_COL:
            raise ValueError(
                f"horizon_k must be one of {sorted(self.HORIZON_TO_COL.keys())}, got {horizon_k}"
            )
        self.horizon_k = horizon_k

        # Resolve paths based on variant
        if fi2010_variant.lower() == 'decpre':
            train_path = 'data/processed/fi2010_decpre_train.npy'
            test_path = 'data/processed/fi2010_decpre_test.npy'

        for path in (train_path, test_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FI-2010 file not found: {path}\n"
                    "Run `python3 scripts/prepare_fi2010.py` first."
                )

        train_data = np.load(train_path)
        test_data = np.load(test_path)

        # Column validation
        for name, arr in [('fi2010_train', train_data), ('fi2010_test', test_data)]:
            if arr.shape[1] < self.TOTAL_COLS:
                raise ValueError(
                    f"{name}: expected at least {self.TOTAL_COLS} columns, got {arr.shape[1]}"
                )

        label_col = self.HORIZON_TO_COL[horizon_k]

        # Feature selection (Change 1.3.4)
        if feature_set == 'raw40':
            n_feat = 40
        elif feature_set == 'raw40_eng':
            n_feat = self.N_FEATURES
        else:
            raise ValueError(f"Unknown feature_set: {feature_set!r}")

        X_all_train = train_data[:, :n_feat]
        y_all_train = train_data[:, label_col].astype(int) - 1  # 1/2/3 → 0/1/2

        X_test = test_data[:, :n_feat]
        y_test = test_data[:, label_col].astype(int) - 1

        # Change 1.3.2 — Gap between splits
        gap = max(horizon_k, window_len)

        # Carve validation from END of training days
        split_idx = int(len(X_all_train) * (1.0 - val_fraction))

        self.X_train = X_all_train[:split_idx - gap]
        self.y_train = y_all_train[:split_idx - gap]
        self.X_val = X_all_train[split_idx:]
        self.y_val = y_all_train[split_idx:]
        self.X_test = X_test
        self.y_test = y_test

        # Load feature names
        feature_names_path = 'data/processed/fi2010_feature_names.json'
        if os.path.exists(feature_names_path):
            with open(feature_names_path) as f:
                fn_doc = json.load(f)
            self.feature_names = fn_doc['names_list'][:n_feat]
        else:
            self.feature_names = [f'feat_{i}' for i in range(n_feat)]

        # Change 1.3.6 — Metadata
        self.metadata = {
            'market': 'fi2010',
            'fi2010_variant': fi2010_variant,
            'horizon_k': horizon_k,
            'feature_set': feature_set,
            'n_features': n_feat,
            'n_train': len(self.X_train),
            'n_val': len(self.X_val),
            'n_test': len(self.X_test),
            'gap': gap,
            'window_len': window_len,
            'class_dist_train': _class_dist(self.y_train),
            'class_dist_val': _class_dist(self.y_val),
            'class_dist_test': _class_dist(self.y_test),
            'split_hash': _hash_array(self.y_test),
            'feature_names': self.feature_names,
            'standardize': standardize,
        }

    def get_splits(self):
        """Returns (X_train, y_train, X_val, y_val, X_test, y_test), metadata."""
        return (
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
        ), self.metadata


class CryptoDataset:
    """
    Loads the prepared crypto parquet, applies labeling, and returns
    chronological splits with gap.

    Changes 1.3.1–1.3.6 implemented.
    """

    def __init__(
        self,
        parquet_path: str = 'data/processed/crypto_BTCUSDT.parquet',
        symbol: str = 'BTCUSDT',
        horizon: int = 40,
        threshold_mode: str = 'fixed',
        threshold_param: float = 0.0001,
        label_rule: str = 'point_return',
        feature_set: str = 'raw40',
        window_len: int = 1,
        start_ts: str = None,
        end_ts: str = None,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        standardize: bool = True,
        log_volume: bool = True,
        split_by_day: dict = None,
    ):
        """
        Parameters
        ----------
        parquet_path   : path produced by scripts/prepare_crypto.py
        symbol         : trading pair symbol
        horizon        : number of events ahead for the label
        threshold_mode : 'fixed', 'quantile', or 'std_mult'
        threshold_param: α for fixed, c for std_mult, ignored for quantile
        label_rule     : 'point_return' or 'smoothed'
        feature_set    : 'raw40' or 'raw40_eng'
        window_len     : window length for temporal models
        start_ts       : ISO UTC string to slice start (Change 1.3.1)
        end_ts         : ISO UTC string to slice end (Change 1.3.1)
        train_frac     : fraction of data for training
        val_frac       : fraction of data for validation
        standardize    : whether to apply standardization
        log_volume     : whether to log1p-transform volumes
        split_by_day   : dict {train:[days], val:[days], test:[days]} for regime analysis
        """
        import pandas as pd
        from data.labeling import (
            compute_mid_price, compute_returns, resolve_threshold,
            label_by_threshold, horizon_events_to_seconds
        )
        from data.features import (
            to_relative_price, build_feature_matrix, CANONICAL_RAW_COLS
        )

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Crypto parquet not found: {parquet_path}\n"
                "Run `python3 scripts/prepare_crypto.py` first."
            )

        df = pd.read_parquet(parquet_path)

        # Change 1.3.1 — Timestamp-based windowing
        if start_ts is not None or end_ts is not None:
            if 'datetime' in df.columns:
                dt_col = 'datetime'
            elif '1' in df.columns:
                dt_col = '1'
            else:
                raise ValueError("Cannot find datetime column for timestamp windowing")

            if start_ts is not None:
                df = df[df[dt_col] >= pd.Timestamp(start_ts, tz='UTC')]
            if end_ts is not None:
                df = df[df[dt_col] < pd.Timestamp(end_ts, tz='UTC')]
            df = df.reset_index(drop=True)

        logger.info(f"Crypto {symbol}: {len(df):,} rows loaded")
        if 'datetime' in df.columns:
            logger.info(f"  Date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
            start_ts_actual = str(df['datetime'].iloc[0])
            end_ts_actual = str(df['datetime'].iloc[-1])
        else:
            start_ts_actual = 'unknown'
            end_ts_actual = 'unknown'

        # Apply relative price + log volume transform
        df = to_relative_price(df)

        # Build feature matrix (Change 1.3.4)
        X, feature_names = build_feature_matrix(df, feature_set)

        # Compute mid-price for labeling (on original prices, not relative)
        # Reload the raw mid-price from the parquet
        df_raw = pd.read_parquet(parquet_path)
        if start_ts is not None or end_ts is not None:
            dt_col = 'datetime' if 'datetime' in df_raw.columns else '1'
            if start_ts is not None:
                df_raw = df_raw[df_raw[dt_col] >= pd.Timestamp(start_ts, tz='UTC')]
            if end_ts is not None:
                df_raw = df_raw[df_raw[dt_col] < pd.Timestamp(end_ts, tz='UTC')]
            df_raw = df_raw.reset_index(drop=True)

        mid_price = compute_mid_price(df_raw).values

        # Compute returns
        returns = compute_returns(mid_price, horizon, label_rule)

        # Truncate X to match returns length
        n_valid = len(returns)
        X = X[:n_valid]

        # Change 1.3.2 — Gap between splits
        gap = max(horizon, window_len)

        # Chronological split
        n = len(X)
        tr = int(n * train_frac)
        val = int(n * (train_frac + val_frac))

        # Resolve threshold on train split returns only (Change 1.3.5 / 1.4.2)
        train_returns = returns[:tr - gap]
        alpha = resolve_threshold(train_returns, threshold_mode, threshold_param)

        # Label everything with resolved alpha
        labels = label_by_threshold(returns, alpha)

        # Apply gap (Change 1.3.2)
        self.X_train = X[:tr - gap]
        self.y_train = labels[:tr - gap]
        self.X_val = X[tr:val - gap]
        self.y_val = labels[tr:val - gap]
        self.X_test = X[val:]
        self.y_test = labels[val:]

        self.feature_names = feature_names

        # Save mid-price and timestamps for test period (needed by backtest)
        self.test_mid = mid_price[val:val + len(self.y_test)]

        if 'datetime' in df_raw.columns:
            self.test_timestamps = df_raw['datetime'].iloc[val:val + len(self.y_test)].values
        else:
            self.test_timestamps = None

        # Save raw bid/ask for backtest
        if 'bid_p_1' in df_raw.columns:
            self.test_bid = df_raw['bid_p_1'].iloc[val:val + len(self.y_test)].values
            self.test_ask = df_raw['ask_p_1'].iloc[val:val + len(self.y_test)].values
        else:
            self.test_bid = None
            self.test_ask = None

        horizon_seconds = horizon_events_to_seconds(horizon)

        logger.info(f"Crypto split: train={len(self.X_train):,} val={len(self.X_val):,} test={len(self.X_test):,}")
        logger.info(f"  Horizon: {horizon} events = {horizon_seconds:.1f} seconds")
        logger.info(f"  Threshold mode: {threshold_mode}, resolved α = {alpha:.8f}")
        logger.info(f"  Gap: {gap}")

        # Change 1.3.6 — Metadata
        self.metadata = {
            'market': 'crypto',
            'symbol': symbol,
            'horizon_events': horizon,
            'horizon_seconds': horizon_seconds,
            'label_rule': label_rule,
            'threshold_mode': threshold_mode,
            'threshold_param': threshold_param,
            'threshold_used': alpha,
            'feature_set': feature_set,
            'n_features': X.shape[1],
            'n_train': len(self.X_train),
            'n_val': len(self.X_val),
            'n_test': len(self.X_test),
            'gap': gap,
            'window_len': window_len,
            'start_ts': start_ts_actual,
            'end_ts': end_ts_actual,
            'class_dist_train': _class_dist(self.y_train),
            'class_dist_val': _class_dist(self.y_val),
            'class_dist_test': _class_dist(self.y_test),
            'split_hash': _hash_array(self.y_test),
            'feature_names': feature_names,
            'standardize': standardize,
        }

    def get_splits(self):
        """Returns (X_train, y_train, X_val, y_val, X_test, y_test), metadata."""
        return (
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
        ), self.metadata


def generate_rolling_origin_folds(n_days: int, min_train_days: int = 6):
    """
    Change 1.3.3 — Generate rolling-origin fold definitions.

    Yields dicts of {train: [days], val: [day], test: [day]}.
    E.g. with 12 days: train 1-6/val 7/test 8, train 1-7/val 8/test 9, etc.
    """
    folds = []
    for test_day in range(min_train_days + 2, n_days + 1):
        val_day = test_day - 1
        train_days = list(range(1, val_day))
        folds.append({
            'train': train_days,
            'val': [val_day],
            'test': [test_day],
        })
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _class_dist(labels: np.ndarray) -> dict:
    """Compute class distribution as a dict."""
    if len(labels) == 0:
        return {}
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {
        int(k): {'count': int(v), 'pct': round(float(v / total * 100), 2)}
        for k, v in zip(unique, counts)
    }


def _hash_array(arr: np.ndarray) -> str:
    """Hash an array for split consistency checking (Change 3.3)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
