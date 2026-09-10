"""
models/logistic.py — Multinomial logistic regression baseline (Change 2.4).

This is the floor every other model must beat. Uses sklearn with
class_weight='balanced'.
"""

from sklearn.linear_model import LogisticRegression


def build_model(config: dict) -> LogisticRegression:
    """Builds a multinomial logistic regression classifier."""
    model_params = config.get('model_params', {})

    return LogisticRegression(
        multi_class='multinomial',
        solver=model_params.get('solver', 'lbfgs'),
        max_iter=model_params.get('max_iter', 1000),
        class_weight='balanced',
        random_state=config.get('seed', 42),
        n_jobs=-1,
        C=model_params.get('C', 1.0),
    )
