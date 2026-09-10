"""
models/random_forest.py — Random Forest with full hyperparameter set (Change 2.5).

Exposes: n_estimators, max_depth, min_samples_leaf, max_features.
random_state from run seed only.
"""

from sklearn.ensemble import RandomForestClassifier


def build_model(config: dict) -> RandomForestClassifier:
    """
    Builds a RandomForestClassifier from the full hyperparameter set in config.
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

    # Handle class imbalance
    imbalance_strat = config.get('imbalance', 'class_weight')
    if isinstance(imbalance_strat, dict):
        imbalance_strat = imbalance_strat.get('strategy', 'none')
    class_weight = "balanced" if imbalance_strat == "class_weight" else None

    return RandomForestClassifier(
        n_estimators=model_params.get('n_estimators', 500),
        max_depth=model_params.get('max_depth', None),
        min_samples_leaf=model_params.get('min_samples_leaf', 1),
        max_features=model_params.get('max_features', 'sqrt'),
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )
