# LOB Prediction Comparison

Comparative study of limit order book (LOB) price-direction classification models across the FI-2010 benchmark dataset and BTCUSDT perpetual futures on Binance.

> **Scope:** All results apply to BTCUSDT perpetual futures on Binance (250 ms time-sampled snapshots, 10 levels) and the FI-2010 Finnish equity LOB benchmark. Cross-asset and cross-exchange generalization is outside this study's scope.

## Datasets

| Property | FI-2010 | Crypto (BTCUSDT) |
|----------|---------|------------------|
| Source | [LOBSTER / Ntakaris et al. (2018)](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) | [martinsn/high-frequency-lob-btcusdt-binance](https://www.kaggle.com/datasets/martinsn/high-frequency-lob-btcusdt-binance) |
| Exchange | Helsinki Stock Exchange (NASDAQ Nordic) | Binance |
| Instrument | 5 Finnish stocks | BTCUSDT perpetual futures |
| Sampling | Event-based | 250 ms time-sampled |
| Depth | 10 levels each side | 10 levels each side |
| Coverage | 10 consecutive trading days | 12 consecutive days |
| Features | 144 (40 raw + 104 engineered) | 40 raw (+ engineered optional) |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# For GPU support (CUDA):
pip install -r requirements-gpu.txt
```

### 2. Prepare data
```bash
python scripts/prepare_crypto.py          # Download & prepare crypto LOB
python scripts/prepare_fi2010.py          # Download & prepare FI-2010
python scripts/prepare_fi2010.py --verify-horizons  # Verify horizon mapping
```

### 3. Run the full pipeline
```bash
bash reproduce.sh
# Or run individual stages:
python run_all.py --stage prepare
python run_all.py --stage sweep
python run_all.py --stage tune
python run_all.py --stage train --market crypto
python run_all.py --stage ablate
python run_all.py --stage backtest
python run_all.py --stage regime
python run_all.py --stage significance
python run_all.py --stage aggregate
```

### 4. Run a single experiment
```bash
python main.py --config configs/crypto_raw40_smoothed_xgboost.yaml --seed 0
python main.py --config configs/fi2010_raw40_smoothed_deeplob_full.yaml --seed 0
```

### 5. Run tests
```bash
pytest tests/ -v
```

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| `prepare` | `scripts/prepare_*.py` | Download, clean, reorder data; quality reports |
| `sweep` | `experiments/threshold_sweep.py` | Label rule × horizon × threshold grid search |
| `tune` | `train/tune.py` | Optuna hyperparameter search per model |
| `train` | `main.py` | Train × 5 seeds for the full model grid |
| `ablate` | `experiments/ablation.py` | One-factor-at-a-time ablation studies |
| `backtest` | `experiments/backtest.py` | Trading simulation with costs |
| `regime` | `experiments/regime_split.py` | Per-regime and per-day analysis |
| `significance` | `experiments/run_significance.py` | Paired bootstrap + McNemar tests |
| `aggregate` | `eval/aggregate.py` | Combine results → LaTeX tables |

## Models

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| Logistic | Baseline | (F,) | Multinomial logistic regression floor |
| Random Forest | Tree | (F,) | Ensemble with tuned hyperparameters |
| XGBoost | Tree | (F,) | Gradient boosting with early stopping |
| DeepLOB | Neural | (T, 40) | Faithful implementation (Zhang et al. 2019) |
| DeepLOB-Attention | Neural | (T, 40) | Multi-horizon attention decoder (Zhang & Zohren 2021) |
| Windowed Transformer | Neural | (T, F) | Per-snapshot tokenization |
| LevelTransformer | Neural | (T, 40) | Two-stage: level attention → temporal attention |

Ablation-only:
- `FeatureConvBiLSTM` — old single-snapshot conv model (for comparison)
- `ScalarTokenTransformer` — old scalar-token model (for comparison)

## Config Keys

All config keys with defaults (see `configs/base.yaml`):

```yaml
data:
  market: crypto | fi2010
  symbol: BTCUSDT
  feature_set: raw40 | raw40_eng
  label_rule: point_return | smoothed
  horizon_events: 40
  threshold_mode: fixed | quantile | std_mult
  threshold_param: 0.0001
  window_len: 100  # 1 for trees, 100 for temporal
  standardize: true
  log_volume: true

model_params: {}  # Model-specific; set by tune.py

training:
  seeds: [0, 1, 2, 3, 4]
  epochs_max: 50
  patience: 5
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine | plateau
  grad_clip: 1.0

imbalance: class_weight | focal_loss | none
```

## Per-Run Outputs

Each run (`main.py --config ... --seed N`) saves to `<output_dir>/seed_N/`:
- `metrics.json` — accuracy, macro_f1, mcc, per-class F1, confusion matrix
- `run_manifest.json` — full metadata (split hash, params, timing, versions)
- `config_used.json` — resolved config
- `test_predictions.npy` — predicted class labels
- `test_probs.npy` — class probabilities (softmax / predict_proba)
- `test_labels.npy` — ground truth labels
- `test_timestamps.npy` — timestamps (crypto only, for backtest)
- `test_mid.npy` — mid-prices (crypto only, for backtest)
- `training_history.json` — per-epoch train/val metrics (neural only)

## Project Structure

```
├── configs/              # YAML config files per (market, feature_set, label_rule, model)
├── data/
│   ├── features.py       # Feature engineering (raw40 + engineered)
│   ├── labeling.py       # Labeling rules (point_return, smoothed + threshold modes)
│   ├── loaders.py        # Data loading (FI2010Dataset, CryptoDataset)
│   └── sequences.py      # Windowed torch Dataset for temporal models
├── eval/
│   ├── aggregate.py      # Aggregate across seeds → LaTeX tables
│   ├── metrics.py        # compute_all_metrics()
│   └── significance.py   # (legacy — use experiments/run_significance.py)
├── experiments/
│   ├── ablation.py       # One-factor-at-a-time ablation studies
│   ├── backtest.py       # Trading simulation with costs
│   ├── feature_importance.py  # XGBoost feature importance
│   ├── regime_split.py   # Per-regime metrics analysis
│   ├── run_significance.py  # Paired bootstrap + McNemar tests
│   ├── threshold_sweep.py  # Horizon × threshold grid search
│   └── transfer.py       # Cross-market transfer learning
├── models/
│   ├── deeplob.py        # DeepLOB (faithful) + FeatureConvBiLSTM (ablation)
│   ├── deeplob_attention.py  # DeepLOB-Attention (Zhang & Zohren 2021)
│   ├── logistic.py       # Multinomial logistic regression
│   ├── random_forest.py  # Random Forest
│   ├── structured_transformer.py  # LevelTransformer (two-stage)
│   ├── transformer.py    # Windowed Transformer + ScalarTokenTransformer
│   └── xgboost_model.py  # XGBoost
├── scripts/
│   ├── prepare_crypto.py # Crypto data preparation + quality report
│   └── prepare_fi2010.py # FI-2010 preparation + horizon verification
├── tests/                # pytest suite
├── train/
│   ├── train_neural.py   # Neural training (AdamW, scheduler, early stop)
│   ├── train_tree.py     # Tree training (frozen params, no Optuna)
│   └── tune.py           # Optuna tuning (shared by trees + neural)
├── utils/
│   ├── logging.py
│   └── seeding.py
├── main.py               # Single experiment entry point
├── run_all.py            # Stage-based pipeline
├── reproduce.sh          # One-command reproduction
└── requirements.txt
```
