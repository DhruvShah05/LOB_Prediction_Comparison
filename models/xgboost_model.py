"""
models/xgboost_model.py — XGBoost classifier with full hyperparameter set (Change 2.5).

Exposes: learning_rate, subsample, colsample_bytree, min_child_weight,
reg_lambda, reg_alpha, n_estimators (with early stopping on val logloss).
random_state from run seed only.
"""

import xgboost as xgb


def build_model(config: dict) -> xgb.XGBClassifier:
    """
    Builds an XGBoost Classifier from the full hyperparameter set in config.
    All parameters must be explicitly set (no nulls — Change 3.1).
    """
    model_params = config.get('model_params', {})
    seed = config.get('seed', 42)

    # Validate: no nulls allowed (Change 3.1)
    for key in ['n_estimators', 'max_depth']:
        if key in model_params and model_params[key] is None:
            raise ValueError(
                f"model_params.{key} is null. Run train/tune.py first to freeze "
                "hyperparameters, then set them in the config."
            )

    return xgb.XGBClassifier(
        n_estimators=model_params.get('n_estimators', 500),
        max_depth=model_params.get('max_depth', 6),
        learning_rate=model_params.get('learning_rate', 0.1),
        subsample=model_params.get('subsample', 0.8),
        colsample_bytree=model_params.get('colsample_bytree', 0.8),
        min_child_weight=model_params.get('min_child_weight', 1),
        reg_lambda=model_params.get('reg_lambda', 1.0),
        reg_alpha=model_params.get('reg_alpha', 0.0),
        objective='multi:softprob',
        num_class=3,
        random_state=seed,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=model_params.get('early_stopping_rounds', 50),
        eval_metric='mlogloss',
    )
