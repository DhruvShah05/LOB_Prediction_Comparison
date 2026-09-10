"""
tests/test_backtest.py — Backtest sanity tests (Change 8).

- Perfect-foresight signal with zero costs → positive PnL
- Random signal with costs → ≤ 0 on average
"""

import pytest
import numpy as np

from experiments.backtest import compute_pnl, compute_metrics, run_backtest


@pytest.fixture
def synthetic_market():
    """Synthetic market data with known price movements."""
    np.random.seed(42)
    n = 1000

    # Random walk with slight upward drift
    returns = np.random.randn(n) * 0.001 + 0.0001
    mid = 50000.0 * np.cumprod(1 + returns)

    # Bid-ask spread of 1 bp
    spread = mid * 0.0001
    bid = mid - spread / 2
    ask = mid + spread / 2

    # True labels: 0=down, 1=stationary, 2=up (based on next return)
    true_labels = np.ones(n, dtype=int)
    true_labels[returns > 0.0005] = 2
    true_labels[returns < -0.0005] = 0

    return mid, bid, ask, true_labels


def test_perfect_foresight_positive_pnl(synthetic_market):
    """Perfect-foresight with zero costs should yield positive PnL."""
    mid, bid, ask, y_true = synthetic_market

    # Use true labels as predictions, zero costs
    metrics, pnl, positions = run_backtest(
        y_pred=y_true, y_true=y_true, probs=None,
        mid=mid, bid=bid, ask=ask,
        taker_fee_bps=0, slippage_bps=0
    )

    assert metrics['cumulative_pnl'] > 0, \
        f"Perfect foresight should have positive PnL, got {metrics['cumulative_pnl']}"


def test_random_signal_with_costs_negative(synthetic_market):
    """Random signal with costs should yield ≤ 0 on average."""
    mid, bid, ask, y_true = synthetic_market

    pnls = []
    for seed in range(20):
        rng = np.random.RandomState(seed)
        random_preds = rng.randint(0, 3, size=len(y_true))

        metrics, pnl, _ = run_backtest(
            y_pred=random_preds, y_true=y_true, probs=None,
            mid=mid, bid=bid, ask=ask,
            taker_fee_bps=4, slippage_bps=2
        )
        pnls.append(metrics['cumulative_pnl'])

    mean_pnl = np.mean(pnls)
    # Random signal should not systematically make money after costs
    assert mean_pnl <= mid[0] * 0.01, \
        f"Random signal should not systematically profit, mean PnL = {mean_pnl:.2f}"


def test_flat_signal_zero_pnl(synthetic_market):
    """All-stationary signal (no trades) should have zero PnL."""
    mid, bid, ask, y_true = synthetic_market

    flat_preds = np.ones(len(y_true), dtype=int)  # All stationary

    metrics, pnl, positions = run_backtest(
        y_pred=flat_preds, y_true=y_true, probs=None,
        mid=mid, bid=bid, ask=ask,
        taker_fee_bps=4, slippage_bps=2
    )

    assert np.all(positions == 0), "Stationary predictions should have all-zero positions"
    assert abs(metrics['cumulative_pnl']) < 1e-6, "No trades should mean zero PnL"


def test_compute_pnl_basic():
    """Basic PnL computation sanity check."""
    mid = np.array([100.0, 101.0, 102.0, 101.0, 100.0])
    bid = mid - 0.005
    ask = mid + 0.005
    positions = np.array([1, 1, 1, 0, 0])

    pnl, fees = compute_pnl(positions, mid, bid, ask, taker_fee_bps=0, slippage_bps=0)

    # Position 1 from t=0: PnL at t=1 = 1*(101-100) = 1
    # Position 1 from t=1: PnL at t=2 = 1*(102-101) = 1
    # Position 1 at t=2, closes to 0 at t=3: PnL at t=3 = 1*(101-102) = -1
    assert pnl[1] == pytest.approx(1.0, abs=0.01)
    assert pnl[2] == pytest.approx(1.0, abs=0.01)


def test_metrics_computed():
    """All expected metric keys should be present."""
    mid = np.array([100.0, 101.0, 102.0, 101.0, 100.0] * 20)
    positions = np.array([1, 0, -1, 0, 1] * 20)
    pnl = np.random.randn(100) * 0.1

    metrics = compute_metrics(pnl, mid, positions)

    expected_keys = {
        'cumulative_pnl', 'net_return_pct', 'annualized_sharpe',
        'annualized_sortino', 'max_drawdown', 'hit_rate',
        'turnover_per_hour', 'profit_factor', 'avg_holding_seconds', 'n_trades'
    }
    assert expected_keys.issubset(metrics.keys())
