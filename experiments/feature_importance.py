"""
experiments/feature_importance.py — XGBoost gain-based feature importance (Change 5.8, NICE).

Per market and feature set: ranked importance by named feature.
Figure: top-20 features per market.
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from data.loaders import FI2010Dataset, CryptoDataset
from data.features import TrainOnlyScaler
from models.xgboost_model import build_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_importance(market, feature_set='raw40_eng', output_dir='results/importance/'):
    """Compute and save XGBoost feature importance."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if market == 'fi2010':
        ds = FI2010Dataset(feature_set=feature_set)
    else:
        ds = CryptoDataset(feature_set=feature_set, label_rule='smoothed')

    (X_tr, y_tr, X_v, y_v, X_te, y_te), meta = ds.get_splits()

    scaler = TrainOnlyScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_v = scaler.transform(X_v)

    feature_names = meta.get('feature_names', [f'feat_{i}' for i in range(X_tr.shape[1])])

    # Train XGBoost with reasonable params
    config = {
        'model': 'xgboost',
        'model_params': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'seed': 0,
        'imbalance': {'strategy': 'class_weight'},
    }

    model = build_model(config)
    sw = compute_sample_weight(class_weight='balanced', y=y_tr)
    model.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_v, y_v)], verbose=False)

    # Get importance
    importance = model.feature_importances_

    # Create DataFrame
    imp_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance,
    }).sort_values('importance', ascending=False)

    # Save
    csv_path = os.path.join(output_dir, f'{market}_{feature_set}_importance.csv')
    imp_df.to_csv(csv_path, index=False)
    logger.info(f"Feature importance saved to {csv_path}")

    # Top 20
    top20 = imp_df.head(20)
    logger.info(f"\nTop 20 features ({market}, {feature_set}):")
    for _, row in top20.iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")

    return imp_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature importance (Change 5.8)")
    parser.add_argument('--market', type=str, default='both', choices=['crypto', 'fi2010', 'both'])
    parser.add_argument('--feature-set', type=str, default='raw40_eng')
    parser.add_argument('--output-dir', type=str, default='results/importance/')
    args = parser.parse_args()

    markets = ['crypto', 'fi2010'] if args.market == 'both' else [args.market]
    for market in markets:
        compute_importance(market, args.feature_set, args.output_dir)
