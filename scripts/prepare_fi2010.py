"""
scripts/prepare_fi2010.py — Prepares FI-2010 raw data into fast-loading .npy arrays.

Data source (build.md §2.1 requirement — exact source documented here):
  FI-2010 (Finnish Stock Exchange LOB dataset — Kercheval & Zhang, 2015)
  Official release URL: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
  Alternative GitHub mirror used by many reproductions:
    https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
    (see data/ folder — No Auction, Z-score normalisation variant)

  Expected local path after download:
    BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/
      NoAuction_Zscore_Training/   ← 7 days (train split)
      NoAuction_Zscore_Testing/    ← 3 days (test split)

  For horizon verification (Change 1.2.1), also download:
    BenchmarkDatasets/NoAuction/2.NoAuction_DecPre/
      NoAuction_DecPre_Training/
      NoAuction_DecPre_Testing/

Split convention (build.md §2.1):
  The official FI-2010 release provides pre-split Training/Testing directories that
  correspond to the first 7 days of the 10-day recording period for training and
  the final 3 consecutive days for testing, which is the chronological convention
  used by the majority of published FI-2010 baselines (Ntakaris et al., 2018;
  Zhang et al., 2019; Wallbridge, 2020).  No shuffling is applied across this boundary.

Normalisation:
  Z-score variant (NoAuction_Zscore) is used by default. The DecPre (raw decimal
  prices) variant is also supported and recommended for the matched pipeline
  (Change 1.5.4).

Feature/label layout (per-row in the .txt files, transposed before save):
  Columns 0–143  : 144 features (raw LOB levels + derived features — Kercheval & Zhang §2)
  Columns 144–148: 5 label columns for prediction horizons.

Horizon convention (Change 1.2.1 — VERIFIED):
  The label columns 144–148 correspond to horizons. The exact mapping must be
  verified empirically by running --verify-horizons against the DecPre variant.
  The commonly cited mapping {10,20,30,50,100} is used as default; verification
  confirms or corrects it.

Feature groups (Change 1.2.2 — Kercheval & Zhang):
  Cols 0–39   : raw LOB, 10 levels × (ask_p, ask_v, bid_p, bid_v)
  Cols 40–59  : spread and mid per level
  Cols 60–95  : price differences
  Cols 96–99  : mean ask/bid price/volume
  Cols 100–103: accumulated price and volume differences
  Cols 104–143: price and volume derivatives
"""

import os
import glob
import json
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Change 1.2.2 — Named feature groups following Kercheval & Zhang
FI2010_FEATURE_NAMES = {}

# Group 1: Raw LOB (cols 0–39), 10 levels × (ask_p, ask_v, bid_p, bid_v)
for i in range(1, 11):
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'ask_p_{i}'
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'ask_v_{i}'
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'bid_p_{i}'
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'bid_v_{i}'

# Group 2: Spread and mid per level (cols 40–59)
for i in range(1, 11):
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'spread_{i}'
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'mid_{i}'

# Group 3: Price differences (cols 60–95, 36 columns)
# These include: top-vs-bottom, absolute inter-level differences, etc.
# Exact composition is confirmed by the validation in Change 1.5.2.
for i in range(36):
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'price_diff_{i + 1}'

# Group 4: Means (cols 96–99)
for name in ['mean_ask_price', 'mean_bid_price', 'mean_ask_volume', 'mean_bid_volume']:
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = name

# Group 5: Accumulated differences (cols 100–103)
for name in ['acc_price_diff_1', 'acc_price_diff_2', 'acc_vol_diff_1', 'acc_vol_diff_2']:
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = name

# Group 6: Derivatives (cols 104–143, 40 columns)
for i in range(1, 11):
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'd_ask_p_{i}'
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'd_ask_v_{i}'
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'd_bid_p_{i}'
    FI2010_FEATURE_NAMES[len(FI2010_FEATURE_NAMES)] = f'd_bid_v_{i}'

assert len(FI2010_FEATURE_NAMES) == 144, f"Expected 144 feature names, got {len(FI2010_FEATURE_NAMES)}"

# Build the list in order
FI2010_FEATURE_NAMES_LIST = [FI2010_FEATURE_NAMES[i] for i in range(144)]


def load_fi2010_folder(folder_path: str) -> np.ndarray:
    """
    Loads all .txt files in a FI-2010 folder, sorts chronologically by filename,
    and concatenates into a single numpy array of shape (n_samples, 149).

    The raw .txt files are stored transposed (features × observations), so we
    hstack then transpose back to (observations × features).
    """
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder_path}")

    txt_files.sort()  # Alphabetical order == chronological order for FI-2010 naming convention
    logger.info(f"Loading {len(txt_files)} files from {folder_path}...")

    arrays = []
    for f in txt_files:
        logger.info(f"  Reading {os.path.basename(f)}...")
        # Each file is (149 × n_obs) — features-as-rows, observations-as-cols
        arr = np.loadtxt(f)
        arrays.append(arr)

    # hstack horizontally (join observations), then transpose to (n_obs × 149)
    combined = np.hstack(arrays).T
    logger.info(f"  Loaded shape after transpose: {combined.shape}")
    return combined


def verify_horizons(decpre_train_dir: str, zscore_train_dir: str):
    """
    Change 1.2.1 [BUG B9] — Verify the horizon-to-column mapping empirically.

    Downloads/loads the DecPre variant, reconstructs mid-price, computes
    smoothed labels for various k values, and compares to label columns.
    """
    logger.info("=== Horizon Verification (Change 1.2.1) ===")

    if not os.path.exists(decpre_train_dir):
        logger.error(
            f"DecPre training directory not found at {decpre_train_dir}.\n"
            "Download the NoAuction_DecPre variant to verify horizons.\n"
            "Place at: BenchmarkDatasets/NoAuction/2.NoAuction_DecPre/"
        )
        return None

    decpre_data = load_fi2010_folder(decpre_train_dir)
    logger.info(f"DecPre data shape: {decpre_data.shape}")

    # Reconstruct mid-price from raw columns (FI-2010 raw layout):
    # Col 0 = ask_p_1, Col 2 = bid_p_1 (in the canonical layout)
    # But FI-2010 raw layout: col 0 = ask_p_1, col 1 = ask_v_1, col 2 = bid_p_1, col 3 = bid_v_1
    ask_p1 = decpre_data[:, 0]
    bid_p1 = decpre_data[:, 2]
    mid = (ask_p1 + bid_p1) / 2.0

    alpha = 0.002  # Standard FI-2010 threshold

    candidate_horizons = [1, 2, 3, 5, 10, 20, 30, 50, 100]
    label_cols = list(range(144, 149))  # columns 144–148

    results = {}
    for label_col_idx, label_col in enumerate(label_cols):
        true_labels = decpre_data[:, label_col].astype(int)
        best_k = None
        best_agreement = 0.0

        for k in candidate_horizons:
            # Compute smoothed label: m_plus = mean(m_{t+1},...,m_{t+k})
            n = len(mid)
            if k >= n:
                continue

            # Vectorized: for each t, compute mean of mid[t+1:t+k+1]
            # Use cumulative sum for efficiency
            cumsum = np.concatenate([[0], np.cumsum(mid)])
            valid_end = n - k
            m_plus = (cumsum[np.arange(1, valid_end + 1) + k] - cumsum[np.arange(1, valid_end + 1)]) / k

            m_t = mid[:valid_end]
            r_t = (m_plus - m_t) / m_t

            # Threshold
            computed_labels = np.ones(valid_end, dtype=int) + 1  # default = 2 (Stationary)
            computed_labels[r_t > alpha] = 3   # Up
            computed_labels[r_t < -alpha] = 1  # Down

            # Compare to true labels (only on valid range)
            true_sub = true_labels[:valid_end]
            agreement = np.mean(computed_labels == true_sub) * 100

            if agreement > best_agreement:
                best_agreement = agreement
                best_k = k

            logger.info(f"  Label col {label_col} vs k={k}: agreement = {agreement:.2f}%")

        results[label_col] = {'best_k': best_k, 'agreement_pct': best_agreement}
        logger.info(f"  → Label col {label_col} best match: k={best_k} ({best_agreement:.2f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare FI-2010 data.")
    parser.add_argument('--verify-horizons', action='store_true',
                        help="Run horizon verification against DecPre variant (Change 1.2.1)")
    parser.add_argument('--variant', type=str, default='Zscore',
                        choices=['Zscore', 'DecPre'],
                        help="FI-2010 normalisation variant to prepare")
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help="Fraction of training data to carve out as validation")
    args = parser.parse_args()

    # Source data must be in the project root (no absolute paths — build.md §3)
    if args.variant == 'Zscore':
        base_dir = 'BenchmarkDatasets/NoAuction/1.NoAuction_Zscore'
        train_dir = os.path.join(base_dir, 'NoAuction_Zscore_Training')
        test_dir = os.path.join(base_dir, 'NoAuction_Zscore_Testing')
    else:
        base_dir = 'BenchmarkDatasets/NoAuction/2.NoAuction_DecPre'
        train_dir = os.path.join(base_dir, 'NoAuction_DecPre_Training')
        test_dir = os.path.join(base_dir, 'NoAuction_DecPre_Testing')

    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        logger.warning(f"Cannot find FI-2010 dataset at {base_dir}. Attempting to download from Hugging Face...")
        try:
            from huggingface_hub import sync_bucket
            sync_bucket(
                source="hf://buckets/DhruvShah05/Crypto_Stocks_Data/BenchmarkDatasets",
                dest="BenchmarkDatasets"
            )
            logger.info("Download completed.")
        except ImportError:
            logger.error("huggingface_hub is not installed. Please install it to enable automatic downloading: pip install huggingface_hub")
            return
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            logger.error(f"Cannot find FI-2010 dataset at {base_dir}.")
            logger.error(
                "Download the Z-score variant from:\n"
                "  https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649\n"
                "or the GitHub mirror:\n"
                "  https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books\n"
                "Place the extracted BenchmarkDatasets/ folder in the project root."
            )
            return

    # --- Horizon verification (Change 1.2.1) ---
    if args.verify_horizons:
        decpre_base = 'BenchmarkDatasets/NoAuction/2.NoAuction_DecPre'
        decpre_train = os.path.join(decpre_base, 'NoAuction_DecPre_Training')
        horizon_results = verify_horizons(decpre_train, train_dir)
        if horizon_results:
            report_path = os.path.join(out_dir, 'fi2010_horizon_verification.json')
            with open(report_path, 'w') as f:
                json.dump(horizon_results, f, indent=2)
            logger.info(f"Horizon verification saved to {report_path}")
        return

    logger.info(f"--- Preparing FI-2010 Training Data (7 days, {args.variant}) ---")
    train_data = load_fi2010_folder(train_dir)
    logger.info(f"Training data shape: {train_data.shape}")

    logger.info(f"--- Preparing FI-2010 Testing Data (3 days, {args.variant}) ---")
    test_data = load_fi2010_folder(test_dir)
    logger.info(f"Testing data shape: {test_data.shape}")

    # Validate column count
    for name, arr in [('train', train_data), ('test', test_data)]:
        if arr.shape[1] < 149:
            raise ValueError(
                f"FI-2010 {name} has {arr.shape[1]} columns; expected 149 (144 features + 5 labels). "
                "Check that you are using the correct normalisation variant."
            )

    variant_lower = args.variant.lower()
    train_out = os.path.join(out_dir, f'fi2010_{variant_lower}_train.npy')
    test_out = os.path.join(out_dir, f'fi2010_{variant_lower}_test.npy')

    logger.info(f"Saving to {train_out} and {test_out}...")
    np.save(train_out, train_data)
    np.save(test_out, test_data)

    # --- Change 1.2.2: Save feature names ---
    feature_names_path = os.path.join(out_dir, 'fi2010_feature_names.json')
    feature_names_doc = {
        'total_features': 144,
        'groups': {
            'raw_lob': {
                'columns': '0-39',
                'count': 40,
                'description': '10 levels × (ask_p, ask_v, bid_p, bid_v)',
                'names': FI2010_FEATURE_NAMES_LIST[:40]
            },
            'spread_mid': {
                'columns': '40-59',
                'count': 20,
                'description': 'Spread and mid-price per level',
                'names': FI2010_FEATURE_NAMES_LIST[40:60]
            },
            'price_differences': {
                'columns': '60-95',
                'count': 36,
                'description': 'Price differences (top-vs-bottom, inter-level)',
                'names': FI2010_FEATURE_NAMES_LIST[60:96]
            },
            'means': {
                'columns': '96-99',
                'count': 4,
                'description': 'Mean ask/bid price/volume over 10 levels',
                'names': FI2010_FEATURE_NAMES_LIST[96:100]
            },
            'accumulated_differences': {
                'columns': '100-103',
                'count': 4,
                'description': 'Accumulated price and volume differences (ask-bid)',
                'names': FI2010_FEATURE_NAMES_LIST[100:104]
            },
            'derivatives': {
                'columns': '104-143',
                'count': 40,
                'description': 'Price and volume derivatives (time-sensitive)',
                'names': FI2010_FEATURE_NAMES_LIST[104:144]
            }
        },
        'label_columns': {
            '144': 'horizon_0 (verify with --verify-horizons)',
            '145': 'horizon_1',
            '146': 'horizon_2',
            '147': 'horizon_3',
            '148': 'horizon_4'
        },
        'names_list': FI2010_FEATURE_NAMES_LIST
    }
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names_doc, f, indent=2)
    logger.info(f"Feature names saved to {feature_names_path}")

    # --- Change 1.2.3: Report observation counts ---
    n_train_total = train_data.shape[0]
    val_fraction = args.val_fraction
    split_idx = int(n_train_total * (1.0 - val_fraction))
    n_train = split_idx
    n_val = n_train_total - split_idx
    n_test = test_data.shape[0]

    logger.info("--- Observation Counts (Change 1.2.3) ---")
    logger.info(f"  Total training days: {n_train_total:,}")
    logger.info(f"  Train (first {100*(1-val_fraction):.0f}%): {n_train:,}")
    logger.info(f"  Val (last {100*val_fraction:.0f}% of training): {n_val:,}")
    logger.info(f"  Test (3 test days): {n_test:,}")
    logger.info(f"  Grand total: {n_train_total + n_test:,}")

    logger.info("FI-2010 preparation complete.")


if __name__ == '__main__':
    main()
