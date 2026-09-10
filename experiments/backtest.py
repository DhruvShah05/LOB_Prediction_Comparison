"""
experiments/backtest.py — Trading simulation with costs (Change 5.6, R2).

Strategy: at time t, prediction from data up to t; order executes at t+1
(next 250ms snapshot) at the touch (buy at ask, sell at bid).
Position ∈ {-1, 0, +1} = argmax class (Down/Stationary/Up).

Costs: taker fee f (Binance USDT-M perp ≈ 0.04%), spread crossing (implicit),
plus slippage sweep s ∈ {0, 1, 2, 5, 10} bp.

Variants: (a) argmax, (b) confidence-gated, (c) long-only.

Baselines: buy-and-hold, random signal, perfect-foresight.

Metrics: cumulative net return, annualized Sharpe/Sortino, max drawdown,
hit rate, turnover, profit factor, average holding time.
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Binance USDT-M perpetual taker fee (as of Q3 2024)
DEFAULT_TAKER_FEE_BPS = 4.0  # 0.04%
SLIPPAGE_SWEEP_BPS = [0, 1, 2, 5, 10]
CONFIDENCE_THRESHOLDS = [0.34, 0.4, 0.5, 0.6, 0.7]
RANDOM_SEEDS = 100


def _bps_to_frac(bps):
    return bps / 10000.0


def compute_pnl(positions, mid_prices, bid_prices, ask_prices,
                taker_fee_bps=DEFAULT_TAKER_FEE_BPS, slippage_bps=0):
    """
    Compute PnL from position series.

    Position changes at t execute at t+1 prices:
      Buy (+1): execute at ask[t+1] + slippage
      Sell (-1): execute at bid[t+1] - slippage
    """
    n = len(positions)
    pnl = np.zeros(n)
    fees = np.zeros(n)

    fee_rate = _bps_to_frac(taker_fee_bps)
    slip_rate = _bps_to_frac(slippage_bps)

    for t in range(1, n):
        # Mark-to-market PnL from holding
        if positions[t - 1] != 0:
            price_change = mid_prices[t] - mid_prices[t - 1]
            pnl[t] += positions[t - 1] * price_change

        # Transaction costs when position changes
        if positions[t] != positions[t - 1]:
            # Closing old position + opening new position
            notional = mid_prices[t]
            n_trades = abs(positions[t] - positions[t - 1])
            cost = notional * (fee_rate + slip_rate) * n_trades
            fees[t] = cost
            pnl[t] -= cost

    return pnl, fees


def compute_metrics(pnl, mid_prices, positions, sampling_ms=250):
    """Compute all backtest metrics."""
    cumulative = np.cumsum(pnl)
    net_return = cumulative[-1] / mid_prices[0] if mid_prices[0] > 0 else 0

    # Per-minute PnL for Sharpe/Sortino
    snapshots_per_min = int(60000 / sampling_ms)
    n_minutes = len(pnl) // snapshots_per_min
    if n_minutes > 0:
        minute_pnl = np.array([
            pnl[i * snapshots_per_min:(i + 1) * snapshots_per_min].sum()
            for i in range(n_minutes)
        ])
    else:
        minute_pnl = pnl

    # Annualization: minutes per year ≈ 525600
    minutes_per_year = 525600
    if len(minute_pnl) > 1 and np.std(minute_pnl) > 0:
        sharpe = np.mean(minute_pnl) / np.std(minute_pnl) * np.sqrt(minutes_per_year)
        downside = minute_pnl[minute_pnl < 0]
        sortino = np.mean(minute_pnl) / np.std(downside) * np.sqrt(minutes_per_year) if len(downside) > 0 and np.std(downside) > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Hit rate (fraction of snapshots with positive PnL when in a position)
    active_mask = positions != 0
    if active_mask.sum() > 0:
        hit_rate = (pnl[active_mask] > 0).mean()
    else:
        hit_rate = 0

    # Turnover (trades per hour)
    position_changes = np.diff(positions) != 0
    n_trades = position_changes.sum()
    hours = len(pnl) * sampling_ms / 3600000
    turnover = n_trades / hours if hours > 0 else 0

    # Profit factor
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average holding time (in snapshots → seconds)
    holding_lengths = []
    in_position = False
    current_length = 0
    for p in positions:
        if p != 0:
            current_length += 1
            in_position = True
        else:
            if in_position:
                holding_lengths.append(current_length)
                current_length = 0
                in_position = False
    if in_position:
        holding_lengths.append(current_length)

    avg_holding = np.mean(holding_lengths) * sampling_ms / 1000 if holding_lengths else 0

    return {
        'cumulative_pnl': float(cumulative[-1]),
        'net_return_pct': float(net_return * 100),
        'annualized_sharpe': float(sharpe),
        'annualized_sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'hit_rate': float(hit_rate),
        'turnover_per_hour': float(turnover),
        'profit_factor': float(profit_factor),
        'avg_holding_seconds': float(avg_holding),
        'n_trades': int(n_trades),
    }


def run_backtest(y_pred, y_true, probs, mid, bid, ask,
                 taker_fee_bps=DEFAULT_TAKER_FEE_BPS, slippage_bps=0,
                 confidence_threshold=None, long_only=False):
    """
    Run a single backtest variant.

    Positions: 0=Down→short(-1), 1=Stationary→flat(0), 2=Up→long(+1)
    """
    n = min(len(y_pred), len(mid), len(bid), len(ask))
    y_pred = y_pred[:n]
    mid = mid[:n]
    bid = bid[:n]
    ask = ask[:n]

    # Map predictions to positions
    positions = np.zeros(n)
    positions[y_pred == 0] = -1  # Down → short
    positions[y_pred == 2] = 1   # Up → long

    # Confidence gating
    if confidence_threshold is not None and probs is not None:
        max_prob = probs[:n].max(axis=1)
        positions[max_prob < confidence_threshold] = 0

    # Long-only variant
    if long_only:
        positions[positions < 0] = 0

    pnl, fees = compute_pnl(positions, mid, bid, ask, taker_fee_bps, slippage_bps)
    metrics = compute_metrics(pnl, mid, positions)

    return metrics, pnl, positions


def run_random_baseline(y_true, mid, bid, ask, n_random=RANDOM_SEEDS,
                        taker_fee_bps=DEFAULT_TAKER_FEE_BPS, slippage_bps=0):
    """Random signal baseline with same trade frequency."""
    n = min(len(y_true), len(mid))
    sharpes = []

    for s in range(n_random):
        rng = np.random.RandomState(s)
        random_preds = rng.randint(0, 3, size=n)
        probs = None
        metrics, _, _ = run_backtest(random_preds, y_true[:n], probs, mid[:n],
                                      bid[:n], ask[:n], taker_fee_bps, slippage_bps)
        sharpes.append(metrics['annualized_sharpe'])

    return {
        'mean_sharpe': float(np.mean(sharpes)),
        'std_sharpe': float(np.std(sharpes)),
        'pct_positive': float(np.mean(np.array(sharpes) > 0) * 100),
    }


def run_perfect_foresight(y_true, mid, bid, ask,
                          taker_fee_bps=DEFAULT_TAKER_FEE_BPS, slippage_bps=0):
    """Perfect-foresight upper bound (uses true labels)."""
    probs = None
    metrics, _, _ = run_backtest(y_true, y_true, probs, mid, bid, ask,
                                  taker_fee_bps, slippage_bps)
    return metrics


def backtest_model(results_dir, model_dir, output_dir, seeds=None):
    """Run full backtest for one model."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    all_results = {}

    for seed in seeds:
        seed_dir = os.path.join(results_dir, model_dir, f'seed_{seed}')

        pred_path = os.path.join(seed_dir, 'test_predictions.npy')
        prob_path = os.path.join(seed_dir, 'test_probs.npy')
        label_path = os.path.join(seed_dir, 'test_labels.npy')
        mid_path = os.path.join(seed_dir, 'test_mid.npy')
        bid_path = os.path.join(seed_dir, 'test_bid.npy')
        ask_path = os.path.join(seed_dir, 'test_ask.npy')

        required = [pred_path, label_path, mid_path, bid_path, ask_path]
        if not all(os.path.exists(p) for p in required):
            logger.warning(f"Missing backtest data for {seed_dir}")
            continue

        y_pred = np.load(pred_path)
        probs = np.load(prob_path) if os.path.exists(prob_path) else None
        y_true = np.load(label_path)
        mid = np.load(mid_path)
        bid = np.load(bid_path)
        ask = np.load(ask_path)

        seed_results = {}

        # (a) Argmax with slippage sweep
        for slip in SLIPPAGE_SWEEP_BPS:
            metrics, _, _ = run_backtest(y_pred, y_true, probs, mid, bid, ask,
                                          slippage_bps=slip)
            seed_results[f'argmax_slip{slip}'] = metrics

        # (b) Confidence-gated
        if probs is not None:
            for tau in CONFIDENCE_THRESHOLDS:
                metrics, _, _ = run_backtest(y_pred, y_true, probs, mid, bid, ask,
                                              confidence_threshold=tau)
                seed_results[f'confidence_{tau}'] = metrics

        # (c) Long-only
        metrics, _, _ = run_backtest(y_pred, y_true, probs, mid, bid, ask,
                                      long_only=True)
        seed_results['long_only'] = metrics

        # Baselines
        seed_results['random'] = run_random_baseline(y_true, mid, bid, ask)
        seed_results['perfect_foresight'] = run_perfect_foresight(y_true, mid, bid, ask)

        all_results[f'seed_{seed}'] = seed_results

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{model_dir}_backtest.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    logger.info(f"Backtest saved: {out_path}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trading backtest (Change 5.6)")
    parser.add_argument('--results-dir', type=str, default='results/')
    parser.add_argument('--output-dir', type=str, default='results/backtest/')
    parser.add_argument('--model', type=str, default=None,
                        help="Specific model dir to backtest (default: all crypto)")
    args = parser.parse_args()

    import glob as glob_mod

    if args.model:
        backtest_model(args.results_dir, args.model, args.output_dir)
    else:
        for entry in sorted(os.listdir(args.results_dir)):
            if not entry.startswith('crypto'):
                continue
            entry_path = os.path.join(args.results_dir, entry)
            if os.path.isdir(entry_path) and glob_mod.glob(os.path.join(entry_path, 'seed_*')):
                backtest_model(args.results_dir, entry, args.output_dir)
