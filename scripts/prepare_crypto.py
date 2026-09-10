"""
scripts/prepare_crypto.py — Prepares raw Crypto LOB CSV into a fast-loading parquet.

Data source (build.md §2.2 requirement — exact source documented here):
  Dataset : "Bitcoin Limit Order Book (LOB) Data"
  Platform: Kaggle
  URL     : https://www.kaggle.com/datasets/martinsn/high-frequency-lob-btcusdt-binance
  File    : bitcoin_lob_data.csv  (place in the project root before running this script)

  Exchange   : Binance
  Instrument : BTCUSDT perpetual futures
  Sampling   : 250 ms time-sampled snapshots
  Depth      : 10 levels each side (bid and ask)
  Coverage   : 12 consecutive days (~3.7 M rows, 42 columns)
               Exact start/end timestamps are printed at prepare time and
               recorded in data/processed/crypto_quality_report.json.

Column schema (verified against Kaggle documentation — build.md §2.2):
  Col '0'  : UNIX millisecond timestamp (chronological key)
  Col '1'  : Human-readable datetime string
  Cols '2'–'21'  : 10 bid levels — alternating price and volume
                   '2'=best_bid_price, '3'=best_bid_vol, '4'=next_bid_price, ...
  Cols '22'–'41' : 10 ask levels — alternating price and volume
                   '22'=best_ask_price, '23'=best_ask_vol, '24'=next_ask_price, ...

Output canonical column order (Change 1.1.4):
  After preparation, the parquet uses named columns in per-level interleaved
  FI-2010 layout:
    [ask_p_1, ask_v_1, bid_p_1, bid_v_1, ask_p_2, ask_v_2, bid_p_2, bid_v_2, ...]
  This makes level-grouping code identical across markets.
  Timestamp columns 'timestamp_ms' and 'datetime' are also retained.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COLS = 42  # cols 0–41

# Canonical column order: per-level interleaved [ask_p, ask_v, bid_p, bid_v] × 10 levels
CANONICAL_FEATURE_NAMES = []
for i in range(1, 11):
    CANONICAL_FEATURE_NAMES.extend([
        f'ask_p_{i}', f'ask_v_{i}', f'bid_p_{i}', f'bid_v_{i}'
    ])


def _build_column_mapping():
    """
    Maps raw CSV column indices to canonical named columns.

    Raw layout:
      Cols 2–21: bids  — (price, vol) × 10 levels  → '2','3','4','5',...,'20','21'
      Cols 22–41: asks — (price, vol) × 10 levels  → '22','23','24','25',...,'40','41'

    Canonical layout (per level i = 1..10):
      ask_p_i, ask_v_i, bid_p_i, bid_v_i
    """
    mapping = {}  # raw_col_name -> canonical_name
    for i in range(10):
        level = i + 1
        bid_price_col = str(2 + i * 2)
        bid_vol_col = str(3 + i * 2)
        ask_price_col = str(22 + i * 2)
        ask_vol_col = str(23 + i * 2)

        mapping[ask_price_col] = f'ask_p_{level}'
        mapping[ask_vol_col] = f'ask_v_{level}'
        mapping[bid_price_col] = f'bid_p_{level}'
        mapping[bid_vol_col] = f'bid_v_{level}'

    return mapping


def _compute_quality_report(df: pd.DataFrame) -> dict:
    """
    Change 1.1.2 — Emit a data-quality report.

    Computes inter-row time deltas, gap counts, crossed/locked books,
    zero volume at level 1, and per-day row counts.
    """
    report = {}

    # --- Timestamp analysis ---
    ts_col = 'timestamp_ms'
    ts = df[ts_col].values
    deltas_ms = np.diff(ts)

    report['min_timestamp_ms'] = int(ts[0])
    report['max_timestamp_ms'] = int(ts[-1])
    report['min_datetime'] = str(df['datetime'].iloc[0])
    report['max_datetime'] = str(df['datetime'].iloc[-1])
    report['total_rows'] = len(df)

    # Gaps > 1 second and > 60 seconds
    gaps_gt_1s = int(np.sum(deltas_ms > 1000))
    gaps_gt_60s = int(np.sum(deltas_ms > 60000))
    report['gaps_gt_1s_count'] = gaps_gt_1s
    report['gaps_gt_60s_count'] = gaps_gt_60s

    # Locations of large gaps (indices and timestamps)
    if gaps_gt_60s > 0:
        gap_indices = np.where(deltas_ms > 60000)[0]
        report['gaps_gt_60s_locations'] = [
            {
                'index': int(idx),
                'before_ts': str(df['datetime'].iloc[idx]),
                'after_ts': str(df['datetime'].iloc[idx + 1]),
                'gap_seconds': float(deltas_ms[idx] / 1000)
            }
            for idx in gap_indices[:20]  # cap at 20 for readability
        ]

    # Rows per calendar day (UTC)
    df_temp = df.copy()
    df_temp['_date'] = df_temp['datetime'].dt.date
    rows_per_day = df_temp.groupby('_date').size().to_dict()
    report['rows_per_calendar_day'] = {str(k): int(v) for k, v in rows_per_day.items()}

    # --- Data quality checks ---
    # Crossed/locked book: best_bid >= best_ask
    crossed = df['bid_p_1'] >= df['ask_p_1']
    n_crossed = int(crossed.sum())
    report['crossed_locked_book_rows'] = n_crossed

    # Zero volume at level 1
    zero_bid_vol = int((df['bid_v_1'] == 0).sum())
    zero_ask_vol = int((df['ask_v_1'] == 0).sum())
    report['zero_bid_vol_level1'] = zero_bid_vol
    report['zero_ask_vol_level1'] = zero_ask_vol

    # Summary statistics
    report['median_delta_ms'] = float(np.median(deltas_ms))
    report['mean_delta_ms'] = float(np.mean(deltas_ms))
    report['duration_hours'] = float((ts[-1] - ts[0]) / 1000 / 3600)

    return report, crossed


def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw crypto LOB CSV into fast-loading parquet."
    )
    parser.add_argument(
        '--symbol', type=str, default='BTCUSDT',
        help="Trading pair symbol (default: BTCUSDT)"
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()

    if symbol == 'BTCUSDT':
        input_file = 'bitcoin_lob_data.csv'
    else:
        input_file = f'{symbol.lower()}_lob_data.csv'

    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f'crypto_{symbol}.parquet')

    if not os.path.exists(input_file):
        logger.warning(f"Cannot find {input_file} locally. Attempting to download from Hugging Face...")
        try:
            from huggingface_hub import download_bucket_files
            download_bucket_files(
                bucket_id="DhruvShah05/Crypto_Stocks_Data",
                files=[(input_file, input_file)]
            )
            logger.info("Download completed.")
        except ImportError:
            logger.error("huggingface_hub is not installed. Please install it to enable automatic downloading: pip install huggingface_hub")
            return
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            logger.error(
                f"Cannot find {input_file} in the project root.\n"
                "Download from Kaggle:\n"
                "  https://www.kaggle.com/datasets/martinsn/high-frequency-lob-btcusdt-binance\n"
                "and place bitcoin_lob_data.csv in the project root."
            )
            return

    logger.info(f"Reading {input_file}... (this may take a minute for large files)")
    df = pd.read_csv(input_file)

    # Drop leading unnamed index column if present (artefact of CSV export with row index)
    if df.columns[0].startswith('Unnamed'):
        logger.info(f"Dropping unnamed index column: {df.columns[0]!r}")
        df = df.drop(columns=[df.columns[0]])

    logger.info(f"Loaded crypto data — shape: {df.shape}")

    # Column validation
    if df.shape[1] < EXPECTED_FEATURE_COLS:
        raise ValueError(
            f"Expected at least {EXPECTED_FEATURE_COLS} columns, got {df.shape[1]}. "
            "Verify you are using the correct Kaggle dataset (see source URL in this script's docstring)."
        )

    # Confirm chronological key columns are present
    for required_col in ('0', '1', '2', '22'):
        if required_col not in df.columns:
            raise ValueError(
                f"Required column '{required_col}' not found. "
                "Column names should be '0', '1', ..., '41' per build.md §2.2."
            )

    # Col '1' is a datetime string — convert for potential downstream use
    logger.info("Converting datetime column ('1') to pandas datetime...")
    df['1'] = pd.to_datetime(df['1'], errors='coerce')

    # Sort by UNIX timestamp (col '0') to guarantee chronological order
    logger.info("Sorting by timestamp column ('0')...")
    df = df.sort_values(by='0').reset_index(drop=True)

    # --- Change 1.1.4: Reorder to canonical column layout ---
    logger.info("Reordering columns to canonical per-level layout...")
    col_mapping = _build_column_mapping()

    # Rename timestamp columns
    rename_map = {'0': 'timestamp_ms', '1': 'datetime'}
    rename_map.update(col_mapping)
    df = df.rename(columns=rename_map)

    # Select columns in canonical order
    output_cols = ['timestamp_ms', 'datetime'] + CANONICAL_FEATURE_NAMES
    # Check all canonical feature columns exist after rename
    missing = set(output_cols) - set(df.columns)
    if missing:
        raise ValueError(f"After renaming, missing columns: {sorted(missing)}")
    df = df[output_cols]

    # --- Change 1.1.2: Quality report ---
    logger.info("Computing data quality report...")
    quality_report, crossed_mask = _compute_quality_report(df)
    quality_report['symbol'] = symbol

    # Drop crossed/locked book rows
    n_crossed = crossed_mask.sum()
    if n_crossed > 0:
        logger.warning(f"Dropping {n_crossed} crossed/locked book rows (best_bid >= best_ask)")
        df = df[~crossed_mask].reset_index(drop=True)
        quality_report['rows_after_cleaning'] = len(df)
    else:
        quality_report['rows_after_cleaning'] = len(df)
        logger.info("No crossed/locked book rows found.")

    # Save quality report
    report_path = os.path.join(out_dir, 'crypto_quality_report.json')
    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    logger.info(f"Quality report saved to {report_path}")

    # --- Summary ---
    logger.info(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    logger.info(f"Total rows after cleaning: {len(df):,}")
    logger.info(f"Columns: {list(df.columns[:6])} ... ({len(df.columns)} total)")

    # Save parquet
    logger.info(f"Saving to {out_file} (pyarrow parquet)...")
    df.to_parquet(out_file, engine='pyarrow', index=False)

    logger.info(f"Crypto preparation complete for {symbol}.")


if __name__ == '__main__':
    main()
