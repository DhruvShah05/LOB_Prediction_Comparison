# LOB_Prediction_Comparison — Full Revision Specification

Target repo commit: `1d95d73539ffb33a2720d1cf1c11dd0ef057d771`
Purpose: every change required to (a) fix logic errors in the current code, (b) answer both reviewers, (c) close the additional gaps a new reviewer would find. Written so it can be executed file by file without re-deriving the reasoning.

Priority tags:
- **[BUG]** — current code is wrong or produces numbers that don't mean what the paper says. Must fix.
- **[R1]** / **[R2]** — directly answers Reviewer 1 / Reviewer 2.
- **[GAP]** — not raised by reviewers but likely to cause rejection.
- **[NICE]** — strengthens the paper; do if time allows.

---

## 0. Critical bug summary (read first)

| # | Where | What is wrong | Consequence in the paper |
|---|-------|---------------|--------------------------|
| B1 | `train/train_tree.py` | Optuna runs *inside every seed* with `RandomSampler(seed=run_seed)`, 5 trials. Each seed trains with different hyperparameters. | XGBoost/RF "± std across seeds" is mostly hyperparameter-search noise. Headline ±3–4.6 on XGBoost is not seed variance. |
| B2 | `models/deeplob.py` | 1D convolutions slide across the *feature index* of a single flat snapshot. No time axis, no Inception block, no level structure. | Not a DeepLOB variant. Reviewer 1's exact complaint; worse if they read the code. |
| B3 | `models/structured_transformer.py` + paper §IV-B | Paper says "deeper encoder". Code uses identical `d_model=64, nhead=4, num_layers=2` to the standard Transformer. Only token grouping and pooling differ. | Paper misdescribes the model. Section VII "extra capacity" discussion is based on a false premise. |
| B4 | `models/structured_transformer.py` | "Grouped by LOB level" = `view(batch, seq, 2)` → adjacent pairs. On FI-2010 columns 41–144 these pairs are arbitrary. | Model description doesn't match implementation. |
| B5 | Paper ref [6] vs `scripts/prepare_crypto.py` | Paper cites `siavashraz` Kaggle dataset; code uses `martinsn/high-frequency-lob-btcusdt-binance`. | Wrong data citation. Integrity red flag. |
| B6 | `main.py`, `data/loaders.py` | Crypto window sliced by hardcoded 345,600 rows/day (assumes zero gaps in a 250 ms feed). | "Day 1–3" may not be day 1–3. Date range never stated. |
| B7 | `main.py`, `data/loaders.py` | No gap between chronological splits. Last H rows of train are labelled with val prices; last H of val with test prices. | Mild label leakage contradicting the explicit "no leakage" claim. |
| B8 | `main.py` | README says `test_predictions.npy` is saved per run. It is not. | Significance tests and backtest cannot be run from saved outputs. |
| B9 | `main.py` + `scripts/prepare_fi2010.py` comments | Horizon mapping `{10,20,30,50,100}` asserted, never verified. Dataset paper (Ntakaris 2018) says label horizons are `{1,2,3,5,10}` events. | If wrong, "k=10" in the paper is actually k=1 or vice versa. |
| B10 | `main.py` | FI-2010 `NoAuction_Zscore` is already z-scored per day; `main.py` applies a second StandardScaler. | Harmless numerically but undocumented; a reviewer will ask. |
| B11 | `train/train_neural.py` | Fixed 15 epochs, constant LR, no early stopping, no weight decay, no tuning. Trees get Optuna; neural models get nothing. | Asymmetric tuning budget → comparison biased toward trees. |
| B12 | `experiments/ablation_structured_transformer.py` | Runs `seeds=[0]` only, 2×2 grid. | Ablation with n=1 is not evidence. |
| B13 | `eval/significance.py` | Implemented, never called anywhere. | No significance testing in the paper. |
| B14 | Configs | `crypto_random_forest.yaml` comment says values are justified by `results/threshold_sweep_f1.csv`; that file is not in the repo. | Labeling choice is asserted, not justified. |
| B15 | Paper §III-B vs code | Paper never states crypto data is time-sampled at 250 ms. H=40 is a 10-second horizon; FI-2010 k=10 is 10 *events*. | Hidden timescale confound in the cross-market comparison. |
| B16 | Paper §III vs code | FI-2010 uses 144 features (104 engineered); crypto uses 40 raw. | The "market-dependent" gap is confounded with feature-set richness. |

---

## 1. Data layer

### 1.1 `scripts/prepare_crypto.py`

**Change 1.1.1 [BUG B5] — Correct the source documentation.**
Docstring already names `martinsn`. Keep it. Fix the paper citation to match. Add to the docstring and README: exchange (Binance), instrument (BTCUSDT perpetual), sampling (250 ms time-sampled snapshots), depth (10 levels each side), coverage (12 days, exact start/end timestamps printed at prepare time).

**Change 1.1.2 [BUG B6] — Emit a data-quality report at prepare time.**
After sorting by column `0`:
- Compute inter-row time deltas. Report count and location of gaps > 1 s and > 60 s.
- Report min/max timestamp, total rows, rows per calendar day (UTC).
- Report rows where `best_bid >= best_ask` (crossed/locked book) and rows with zero volume at level 1. Drop crossed rows and log how many.
- Save this as `data/processed/crypto_quality_report.json`. Numbers go in the paper's protocol table.

**Change 1.1.3 [GAP] — Add a second asset.**
Add `prepare_crypto.py --symbol ETHUSDT` (or a separate script) that produces `crypto_ETHUSDT.parquet` with the same schema. Source: the same Kaggle author has other pairs, or record from Binance futures depth stream at 250 ms. If a second asset from the same exchange is impossible, record from a second exchange (Bybit) for BTC — either answers "one asset/one exchange".

**Change 1.1.4 — Reorder columns to FI-2010 layout at prepare time.**
Current: bids block `(p1,v1,...,p10,v10)` then asks block. FI-2010 raw layout is per-level interleaved: `(ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, ...)`. Write a canonical 40-column order `[ask_p_i, ask_v_i, bid_p_i, bid_v_i for i in 1..10]` with **named** columns, and use that order everywhere downstream for both markets. This makes level-grouping code identical across markets.

### 1.2 `scripts/prepare_fi2010.py`

**Change 1.2.1 [BUG B9] — Verify the horizon-to-column mapping empirically.**
Download the `NoAuction_DecPre` variant (raw decimal prices) alongside `Zscore`. From columns 1–40 of DecPre reconstruct mid-price `m_t = (ask_p1 + bid_p1)/2`. Compute the FI-2010 smoothed label for k ∈ {1,2,3,5,10,20,30,50,100} with α = 0.002: `l_t = sign((mean(m_{t+1..t+k}) − m_t)/m_t)` thresholded at ±α. Compare each to label columns 145–149 and report agreement %. Whichever k gives ≈100 % agreement for column 145 is the true horizon. Hard-code the verified mapping with a comment citing the check, and put one sentence in the paper stating the convention used and that it was verified.

**Change 1.2.2 — Name the 144 columns.**
Save a `fi2010_feature_names.json` listing the Kercheval & Zhang groups so the paper and code refer to them by name:
- cols 1–40: raw LOB, 10 levels × (ask_p, ask_v, bid_p, bid_v)
- cols 41–60: spread and mid per level
- cols 61–96: price differences (top-vs-bottom level, absolute inter-level differences)
- cols 97–100: mean ask price, mean bid price, mean ask volume, mean bid volume
- cols 101–104: accumulated price and volume differences (ask−bid)
- cols 105–144: price and volume derivatives (time-sensitive)
Exact composition of 61–96 should be confirmed by the validation in Change 1.5.2.

**Change 1.2.3 — Report the observation counts.**
Print train / val / test counts after the val carve-out so the paper's "354,825 observations" is replaced with an explicit train / val / test breakdown.

### 1.3 `data/loaders.py` (make this the single source of truth)

**Change 1.3.1 [BUG B6] — Timestamp-based windowing for crypto.**
Replace `window_days=[start_day, end_day]` with `start_ts` / `end_ts` (ISO strings, UTC). Slice by `df['1'] >= start_ts & df['1'] < end_ts`. Log actual first/last timestamp and row count. Default: use the full 12 days.

**Change 1.3.2 [BUG B7] — Gap between splits.**
After computing split indices `tr`, `val`, define `gap = max(horizon, window_len)`. Train = `[0, tr − gap)`, Val = `[tr, val − gap)`, Test = `[val, n)`. Same for FI-2010 (gap at the train/val boundary; the official train/test day boundary is already gap-free because labels are precomputed per day).

**Change 1.3.3 — Split ratios and regime coverage.**
Keep 70/15/15 chronological for crypto but with 12 days this gives ~8.4 / 1.8 / 1.8 days. Additionally support `split_by_day: {train:[...], val:[...], test:[...]}` so test can be set to specific days for the regime analysis (Change 5.4).

**Change 1.3.4 — Feature-set selection.**
Add `feature_set` parameter ∈ `{raw40, raw40_eng}`:
- FI-2010 `raw40` → columns 1–40 only. `raw40_eng` → all 144.
- Crypto `raw40` → the 40 canonical columns. `raw40_eng` → 40 + engineered set from Change 1.5.1 (computed here, train-stat-free so no leakage).

**Change 1.3.5 — Label-rule selection.**
Add `label_rule` ∈ `{point_return, smoothed}` and `threshold_mode` ∈ `{fixed, quantile, std_mult}` — see 1.4.

**Change 1.3.6 — Return metadata.**
`get_splits()` should also return a dict: `{n_train, n_val, n_test, class_dist_train, class_dist_val, class_dist_test, start_ts, end_ts, gap, horizon, threshold_used, feature_names}`. `main.py` writes this into `run_manifest.json`.

**Change 1.3.7 — Remove duplicated loading logic from `main.py`.** `main.py.load_data` must call `FI2010Dataset` / `CryptoDataset`.

### 1.4 `data/labeling.py`

**Change 1.4.1 [GAP B15/B16] — Add the FI-2010-style smoothed label.**
```
m_t          = (ask_p1 + bid_p1)/2
m_plus(t,k)  = mean(m_{t+1}, ..., m_{t+k})
r_t          = (m_plus − m_t) / m_t
label        = Up if r_t > α, Down if r_t < −α, else Stationary
```
This is the rule the FI-2010 labels were built with. Use it on crypto for the matched comparison. Keep `point_return` as the secondary rule.

**Change 1.4.2 [GAP] — Data-driven threshold.**
- `fixed`: current behaviour (α given).
- `quantile`: α chosen on the **training split only** such that classes are approximately balanced (α = 33rd/67th percentile of |r_t|, i.e. tertiles). Store the resolved α and apply the same α to val/test.
- `std_mult`: α = c × std(r_t on train), c ∈ {0.25, 0.5, 1.0}.
The resolved α, and the resulting class distribution per split, must be logged and reported.

**Change 1.4.3 — Horizon in time units for crypto.**
Because data is 250 ms sampled, `horizon_events=40` = 10 s. Add a `horizon_seconds` alias in config and print both. Paper must state this.

**Change 1.4.4 — Sweep utility must also return downstream metrics.**
`run_threshold_sweep` currently returns only class balance. Extend `experiments/threshold_sweep.py` (already partially does this) to produce, for each (label_rule, horizon, threshold_mode/α) cell: class distribution per split, and val Macro-F1 / MCC for logistic regression and XGBoost (fixed reasonable params, 1 seed). Output two CSVs and a heatmap figure. This table goes in the paper.

**Change 1.4.5 — Truncation.**
`apply_horizon_labeling` drops the last `horizon` rows. Also drop rows whose label window would cross a split boundary (handled by the gap in 1.3.2; assert this).

### 1.5 `data/features.py`

**Change 1.5.1 [GAP B16] — Engineered feature groups for both markets.**
Implement functions on the canonical 40-column raw book (per snapshot, plus a short history for derivatives). Groups, following Kercheval & Zhang / Ntakaris:

- **Spread & mid per level (20):** `ask_p_i − bid_p_i`, `(ask_p_i + bid_p_i)/2` for i = 1..10.
- **Price differences:** `ask_p_10 − ask_p_1`, `bid_p_1 − bid_p_10`, `|ask_p_{i+1} − ask_p_i|`, `|bid_p_{i+1} − bid_p_i|` for i = 1..9 (and whatever additional pairs are needed to reproduce the 36-column FI-2010 block — determine by validation in 1.5.2).
- **Means (4):** mean ask price, mean bid price, mean ask volume, mean bid volume over 10 levels.
- **Accumulated differences (2–4):** `Σ_i (ask_p_i − bid_p_i)`, `Σ_i (ask_v_i − bid_v_i)` (FI-2010 has 4 here; determine exact set by validation).
- **Derivatives (40):** `d(ask_p_i)/dt`, `d(bid_p_i)/dt`, `d(ask_v_i)/dt`, `d(bid_v_i)/dt` over a short window (FI-2010 uses the previous event; for crypto use the previous snapshot, i.e. 250 ms).
- **Order-book imbalance per level (10) [extra, not in FI-2010]:** `(bid_v_i − ask_v_i)/(bid_v_i + ask_v_i)`. Add as a separate optional group so the FI-2010-matched set stays pure.

Derivatives need history, so compute them before splitting and make sure the first row of each split doesn't use the previous split's last row (drop the first few rows of val/test after the gap, or set derivative = 0 there).

**Change 1.5.2 — Self-validation against FI-2010.**
Run the engineered-feature code on FI-2010 `DecPre` columns 1–40, then compare each output column to `DecPre` columns 41–144 by Pearson correlation. Every FI-2010 engineered column should have a |ρ| ≈ 1 match. Save the mapping. If some don't match, either fix the formula or explicitly exclude that FI-2010 column from the matched set on both markets. Write a unit test in `tests/test_features.py` that asserts the match on a small fixture.

**Change 1.5.3 — Relative price for crypto.**
Change `to_relative_price` from `p − mid` to `(p − mid)/mid` so BTC and ETH share a scale. Apply the same transform to FI-2010 raw prices when using `raw40` on `DecPre` (or use `Zscore` and skip — but then say so). Volumes: log1p-transform before standardization (LOB volumes are heavy-tailed; z-scoring raw volumes lets a few huge orders dominate).

**Change 1.5.4 [BUG B10] — Standardization policy.**
- Crypto and FI-2010 `DecPre`: `TrainOnlyScaler` on train split. Keep.
- FI-2010 `Zscore`: skip the second scaler (or keep it and state it's a no-op). Recommended: switch to `DecPre` for everything so both markets go through the identical pipeline, and cite that z-score-on-train is the standard alternative.

**Change 1.5.5 — Feature names everywhere.**
All feature matrices carry a `feature_names` list. Tree models expose feature importances by name (useful figure for the paper: which features matter in each market).

### 1.6 New: `data/sequences.py`

**Change 1.6.1 [R1] — Windowed dataset for temporal models.**
A `torch.utils.data.Dataset` over an `(n, F)` array and `(n,)` labels that returns `(X[t−T+1 : t+1], y[t])` for `t ≥ T−1`, with `T = window_len` (default 100). Indices must be built per split so a window never spans a split boundary. Use a memory-view (no copying the array T times). Tree models continue to use the single-snapshot representation; a `flatten_window` option can optionally give trees the last `T'` snapshots concatenated (T' = 10) as an ablation of "history helps trees too".

---

## 2. Models

### 2.1 `models/deeplob.py` — replace with a faithful DeepLOB [BUG B2] [R1]

Reference: Zhang, Zohren, Roberts (2019), and the authors' public implementation. Input shape `(batch, 1, T=100, 40)`; the 40 columns must be in the canonical per-level order from 1.1.4.

Architecture (all convs followed by LeakyReLU(0.01) and BatchNorm2d):
1. **Block 1 — price/volume pairs:** `Conv2d(1, 32, kernel=(1,2), stride=(1,2))` → `Conv2d(32, 32, (4,1), padding='same' along time)` × 2.
2. **Block 2 — ask/bid sides:** `Conv2d(32, 32, (1,2), stride=(1,2))` → `Conv2d(32, 32, (4,1))` × 2.
3. **Block 3 — across levels:** `Conv2d(32, 32, (1,10))` → `Conv2d(32, 32, (4,1))` × 2. Output `(batch, 32, T, 1)`.
4. **Inception module** (three parallel branches, 64 channels each): `1×1 → 3×1`, `1×1 → 5×1`, `MaxPool 3×1 → 1×1`. Concatenate → `(batch, 192, T, 1)`.
5. Reshape to `(batch, T, 192)` → `LSTM(hidden=64)` → last hidden state → `Linear(64, 3)`.

Training defaults from the paper: Adam, lr = 0.01, ε = 1, batch 32; in practice lr = 1e-3 with early stopping works fine — treat as a tuned hyperparameter. Parameter count ≈ 140 k; log it.

For the `raw40_eng` feature set, the 104 engineered columns don't fit the price/volume/side/level convolution structure. Options: (a) DeepLOB sees only the raw 40 and the engineered features are concatenated to the LSTM output before the classifier; (b) run DeepLOB on raw40 only and report that. Use (a), and state it.

Keep the current model, renamed `FeatureConvBiLSTM`, only as an ablation row ("no history, conv over feature index") — it is useful evidence that the old paper's model was the problem.

### 2.2 `models/transformer.py` — windowed standard Transformer [R1]

Input `(batch, T, F)`. One token per snapshot: `Linear(F, d_model)` → learned or sinusoidal positional encoding over time → `TransformerEncoder(num_layers, nhead, d_model, dim_ff=4·d_model, dropout)` → pooling (mean over time by default) → `Linear(d_model, 3)`. Defaults `d_model=64, nhead=4, layers=2, dropout=0.1` and tune. Log parameter count.

Keep the current scalar-token model renamed `ScalarTokenTransformer` as an ablation row ("no history, one token per feature").

### 2.3 `models/structured_transformer.py` — make it match its description [BUG B3, B4] [R1]

Rename to `LevelTransformer` (or rewrite the paper to say what it is). Tokenization on the canonical raw-40 layout: one token per **level** = the 4 values `(ask_p_i, ask_v_i, bid_p_i, bid_v_i)` plus that level's spread/mid/imbalance if the engineered set is on. With a time window, tokens are (time × level) → either flatten to `T·10` tokens with 2-D positional encoding (time, level) or use a two-stage encoder (level attention within a snapshot, then temporal attention over pooled snapshot vectors). Use the two-stage version; it's cleaner to ablate.

Pooling options `{mean, cls, attention}` as now. Expose `num_layers`, `d_model`, `nhead`, `dropout` in config so depth is an explicit, ablatable choice. If the paper is going to claim "deeper", the headline config must actually be deeper (e.g. 4 layers vs 2) and a parameter-matched control (2 layers, wider `d_model`) must be in the ablation.

### 2.4 New baselines

- `models/logistic.py` — multinomial logistic regression on the same features (sklearn, `class_weight='balanced'`). This is the floor every other model must beat.
- One published temporal baseline besides DeepLOB: **DeepLOB-Attention** (Zhang & Zohren 2021) or **TransLOB** (Wallbridge 2020). Pick one, cite it, implement from the paper.
- Optional: an MLP on the flattened last-10-snapshot window as a "history without inductive bias" control.

### 2.5 `models/xgboost_model.py`, `models/random_forest.py`

Expose the full hyperparameter set in config: XGBoost `learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda, reg_alpha, n_estimators (with early stopping on val logloss, `early_stopping_rounds=50`)`; RF `n_estimators, max_depth, min_samples_leaf, max_features`. Both take `random_state` from the run seed only.

---

## 3. Training

### 3.1 `train/train_tree.py` — two-phase tuning [BUG B1]

Split into:
- **`tune.py --model --market --feature_set --label_rule`** (new script, shared with neural): one Optuna study, TPE sampler with a fixed sampler seed (e.g. 12345), `n_trials ≥ 30` (trees) / `≥ 20` (neural), objective = val Macro-F1 (or MCC), all trials use seed 0 for the model. Writes `configs/tuned/<market>_<feature_set>_<label_rule>_<model>.yaml` and the full trial log.
- **`train_tree_model`**: reads the frozen tuned config; runs with `random_state = run_seed`; does not touch Optuna. Raise if `model_params` contains nulls.

Search spaces:
- XGBoost: `max_depth [3,12]`, `learning_rate log[1e-3, 0.3]`, `n_estimators [200, 2000]` with early stopping, `subsample [0.5,1]`, `colsample_bytree [0.3,1]`, `min_child_weight [1,20]`, `reg_lambda log[1e-3,10]`.
- RF: `n_estimators {200,500,1000}`, `max_depth [4,30] or None`, `min_samples_leaf [1,50]`, `max_features {sqrt, 0.3, 0.5}`.

Report tuned values in an appendix table.

### 3.2 `train/train_neural.py` [BUG B11]

- Optimizer: AdamW with `weight_decay` from config.
- Scheduler: linear warmup (1 epoch) then cosine decay, or `ReduceLROnPlateau` on val loss. Config key `scheduler`.
- Early stopping: `patience` (default 5) on val loss; `epochs_max` (default 50). Keep best-val-Macro-F1 checkpoint (already done).
- Gradient clipping at 1.0.
- Log per-epoch train/val loss, val Macro-F1, LR; save `training_history.json` (already done). Produce one training-curve figure per model/market for the appendix (proves not underfit).
- Log `n_params` and wall-clock time to the manifest.
- Class-weighting: keep inverse-frequency; also support `none` so the effect of weighting can be ablated. Make sure weights are computed on the train split only (already true).
- Tuning: same `tune.py` over `lr log[1e-4, 1e-2]`, `d_model {32,64,128}`, `num_layers {1,2,4}`, `dropout [0,0.3]`, `weight_decay log[1e-6,1e-2]`, `batch_size {64,128,256}`.

### 3.3 Seeds

Seeds `0–4` vary only: model init, DataLoader shuffle order, tree bootstrap/column sampling. **The split, α, scaler, and hyperparameters are identical across seeds.** Add an assertion in `main.py` that the split hashes are identical across seeds (store a hash of `y_test` in the manifest and compare in `aggregate.py`).

---

## 4. `main.py`

1. Delete inline `load_data`; call `loaders.py` (1.3.7).
2. Read `feature_set`, `label_rule`, `threshold_mode`, `horizon`, `window_len` from config; pass through.
3. Route neural models through `data/sequences.py` when `window_len > 1`.
4. **[BUG B8]** After test inference save `test_predictions.npy`, `test_probs.npy` (softmax / `predict_proba`), and `test_labels.npy`, plus `test_timestamps.npy` / `test_mid.npy` for crypto (needed by the backtest).
5. Save `config_used.json` for every run (currently only neural).
6. Manifest additions: `n_params`, `train_time_s`, `feature_set`, `label_rule`, `threshold_used`, `horizon_events`, `horizon_seconds`, `window_len`, `start_ts`, `end_ts`, `gap`, `split_hash`, `class_dist_{train,val,test}`, git commit hash, full library versions.
7. Smoke-test path: keep, but it must go through the same code path (sequence dataset etc.).

---

## 5. Evaluation and experiments

### 5.1 `eval/aggregate.py`

- Keep the confusion matrix: average across seeds and save per (market, feature_set, label_rule, model). Produce a confusion-matrix grid figure.
- Add per-class F1 columns to the output table (already in `metrics.json`).
- Add 95 % CI (t-distribution, n=5) alongside ± std.
- Check `split_hash` identical across seeds; fail loudly if not.
- Produce the paper tables directly as LaTeX (`booktabs`) so numbers can't be mistyped.

### 5.2 `eval/significance.py` → new `experiments/run_significance.py` [BUG B13]

Load `test_predictions.npy` for every run. For each (market, feature_set, label_rule) and each model pair: paired bootstrap on Macro-F1 and MCC using seed-matched predictions (pair seed i of A with seed i of B, 2000 resamples, fixed RNG seed), plus McNemar on accuracy. Holm-correct across pairs. Output a significance matrix per market and mark significant differences (†) in the LaTeX tables.

### 5.3 `experiments/threshold_sweep.py` [BUG B14]

Grid: `label_rule ∈ {point_return, smoothed}` × `horizon ∈ {10, 20, 40, 100, 200, 400}` (2.5 s – 100 s) × `threshold_mode ∈ {fixed 0.5/1/2 bp, quantile, std_mult 0.5}`. For each cell: class distribution per split, and val Macro-F1/MCC for logistic regression + XGBoost (fixed params, seed 0). Save CSV + heatmap. The headline (horizon, threshold) is chosen from this table by a stated rule (e.g. "smallest horizon at which XGBoost val MCC exceeds 0.05 with quantile thresholds") and the sweep figure goes in the paper. Also run a horizon sweep on FI-2010 over its available label columns.

### 5.4 New `experiments/regime_split.py` [R2]

On the crypto test period compute rolling realized volatility (e.g. 5-minute std of 250 ms mid returns). Bucket test rows into low/mid/high volatility tertiles. Recompute all metrics per bucket from saved predictions. Also report per-calendar-day metrics. Figure: Macro-F1 and MCC per regime per model.

### 5.5 New `experiments/ablation.py` (replaces `ablation_structured_transformer.py`) [BUG B12] [R1]

Base = windowed standard Transformer, tuned config, **5 seeds**. One factor at a time:
- `num_layers ∈ {1, 2, 4, 6}`
- `pooling ∈ {mean, cls, attention}`
- `tokenization ∈ {snapshot, level, scalar}`
- parameter-matched control: 2 layers with `d_model` raised to match the 4-layer model's parameter count
- `dropout ∈ {0, 0.1, 0.3}`
- scheduler `∈ {constant, cosine}` and early stopping on/off
- `window_len ∈ {1, 10, 50, 100}` (this directly shows the value of history — the core of Reviewer 1's point)
- class weighting on/off
Output one table per market with mean ± std and parameter counts.

DeepLOB ablation: `FeatureConvBiLSTM` (old model) vs faithful DeepLOB at `window_len ∈ {1, 10, 50, 100}` — shows exactly how much the old paper's shortcut cost.

### 5.6 New `experiments/backtest.py` [R2]

Inputs: `test_probs.npy`, `test_mid.npy`, `test_timestamps.npy`, best bid/ask series for the test period.

Strategy: at time t, prediction made from data up to t; **order executes at t+1 (next 250 ms snapshot)** at the touch (buy at ask, sell at bid). Position ∈ {−1, 0, +1} = argmax class (Down/Stationary/Up). Hold until the signal changes or `H` snapshots elapse. Position size = 1 unit notional.

Costs: taker fee `f` (Binance USDT-M perp ≈ 0.04–0.05 %; state the number and date), spread crossing (implicit from executing at the touch), plus a slippage sweep `s ∈ {0, 1, 2, 5, 10} bp`.

Variants: (a) argmax; (b) confidence-gated — trade only when `max prob ≥ τ`, sweep `τ ∈ {0.34, 0.4, 0.5, 0.6, 0.7}`; (c) long-only.

Metrics: cumulative net return, annualized Sharpe and Sortino (state annualization: √(snapshots per year) with 250 ms sampling, or compute on per-minute aggregated PnL — state which), max drawdown, hit rate, turnover (trades per hour), profit factor, average holding time.

Baselines: buy-and-hold; random signal with the same trade frequency (100 random seeds, report the distribution); perfect-foresight upper bound (uses true labels) to show the ceiling.

Run for every model and both markets (FI-2010 with a nominal 5 bp cost and the same latency assumption). Figures: equity curves per model, Sharpe vs cost sweep, Sharpe vs confidence threshold. Expected finding: after costs nearly everything is ≤ 0 — report it as the finding.

### 5.7 New `experiments/transfer.py` [NICE]

Train on FI-2010 `raw40` (relative prices, log volumes, per-market scaler), test on crypto `raw40` and vice versa, with the smoothed label rule on both. Report metrics vs in-market training. Also a "fine-tune" variant: pretrain on FI-2010, fine-tune on the first crypto day.

### 5.8 New `experiments/feature_importance.py` [NICE]

XGBoost gain-based importance by named feature, per market and feature set. Figure: top-20 features per market. Supports the "why is crypto harder" narrative.

---

## 6. Configs

Every config gains these keys (with defaults in `base.yaml`):
```
data:
  market: fi2010 | crypto
  symbol: BTCUSDT | ETHUSDT           # crypto only
  start_ts / end_ts: ISO UTC          # crypto only, replaces crypto_window_days
  fi2010_variant: DecPre | Zscore     # default DecPre
  feature_set: raw40 | raw40_eng
  label_rule: point_return | smoothed
  horizon_events: int
  threshold_mode: fixed | quantile | std_mult
  threshold_param: float              # α for fixed, c for std_mult, ignored for quantile
  window_len: 1 | 100                 # 1 for trees, 100 for temporal models
  split: {train: 0.70, val: 0.15, test: 0.15}
  gap: auto                           # max(horizon, window_len)
  standardize: true
  log_volume: true
model_params: <path to configs/tuned/... or explicit values; nulls are an error>
training:
  seeds: [0,1,2,3,4]
  epochs_max, patience, batch_size, lr, weight_decay, dropout, scheduler, grad_clip
imbalance: class_weight | none
```
Delete `crypto_window_days`, `threshold`, `n_estimators: null`. Fix the FI-2010 horizon comment after Change 1.2.1. Add configs for `deeplob_full`, `transformer_windowed`, `level_transformer`, `logistic`, `translob` (or `deeplob_attention`), and keep `feature_conv_bilstm` / `scalar_token_transformer` as ablation-only configs.

Config file naming: `<market>_<symbol>_<feature_set>_<label_rule>_<model>.yaml` to make the grid explicit.

---

## 7. `run_all.py` — stage-based pipeline

Replace the flat loop with stages, each idempotent and resumable:
```
prepare      → scripts/prepare_*.py, quality reports, feature validation test
sweep        → experiments/threshold_sweep.py (both markets)   → choose headline labels
tune         → tune.py for every (market, feature_set, label_rule, model)
train        → main.py × 5 seeds for the full grid
ablate       → experiments/ablation.py
backtest     → experiments/backtest.py
regime       → experiments/regime_split.py
significance → experiments/run_significance.py
aggregate    → eval/aggregate.py → LaTeX tables
figures      → all paper figures from saved outputs
```
`run_all.py --stage train --market crypto` etc. Every stage writes to `results/<stage>/` and is skipped if outputs exist unless `--force`.

---

## 8. Tests (`tests/`)

- `test_features.py`: engineered features reproduce FI-2010 columns 41–144 on a fixture (Change 1.5.2).
- `test_labeling.py`: smoothed and point-return labels on a synthetic mid-price series with known answers; quantile threshold yields ~33/33/33 on train.
- `test_splits.py`: no window and no label horizon crosses a split boundary; split hash identical across seeds.
- `test_deeplob.py`: forward pass shape `(B,100,40) → (B,3)`; parameter count within ±10 % of the reference implementation.
- `test_backtest.py`: perfect-foresight signal with zero costs yields positive PnL; random signal with costs yields ≤ 0 on average.

---

## 9. Documentation and reproducibility

- README: correct the dataset URL, state exact commit hash used for the paper, document the stage pipeline and every config key, correct the `test_predictions.npy` claim once implemented.
- `requirements.txt`: pin `huggingface_hub` to an exact version; add a `requirements-gpu.txt`.
- One command `bash reproduce.sh` (or `make reproduce`) that runs all stages and regenerates every table and figure in the paper.
- Put the git commit hash and the exact dataset file checksums in the paper's reproducibility statement.

---

## 10. Paper changes tied to the code changes

| Paper section | Change |
|---|---|
| Title / Abstract | Reframe: "isolating why LOB models degrade on crypto — feature set, labeling rule, or market" or "above-chance classification ≠ profitability". State the headline numbers from the matched (raw40_eng, smoothed) setting. |
| §II Related work | 20–30 refs: DeepLOB variants, TransLOB, TABL/BiN-CTABL, HLOB, crypto LOB papers, FI-2010 generalization critiques, backtest-overfitting literature. Cite the FI-2010 horizon-convention discrepancy. |
| §III Datasets | Protocol table: source (correct citation), exchange, instrument, sampling (250 ms), date range, gaps dropped, split sizes, gap size, horizon in events *and* seconds, threshold mode and resolved α, class distribution per split, feature-set definition. Same table for FI-2010 with the verified horizon convention. |
| §IV Methodology | Describe every model as implemented. Faithful DeepLOB with window 100. Windowed Transformer. LevelTransformer with stated depth/params. Logistic floor. Two-phase tuning protocol with search spaces. What seeds vary. |
| §V Metrics | Add per-class F1, confusion matrices, 95 % CI, significance test description. Remove Weighted-F1 or report it. |
| §VI Results | Main table: market × feature_set × label_rule × model. Per-class F1 table. Horizon sweep figure. Threshold sweep figure. |
| §VII Discussion | Feature-set effect vs market effect decomposition. Window-length ablation (value of history). Depth/param-matched ablation (answers R1). Regime breakdown. Transfer results. |
| New §VIII Trading evaluation | Backtest protocol, cost sweep, confidence gate, comparison to random and perfect-foresight. Honest conclusion (answers R2). |
| §IX Limitations | Only things *not* fixed: e.g. two assets/one exchange (if ETH added), no market-impact model, no order-flow features. |
| Appendix | Tuned hyperparameters, parameter counts, training curves, full ablation tables, significance matrices, feature-importance plots. |
| Code availability | Correct URL (no space), commit hash, `reproduce.sh`. |

---

## 11. Execution order

1. **Data** (1.1–1.5): prepare both markets, quality report, canonical column order, feature engineering + FI-2010 self-validation test, horizon-convention verification, gap-aware splits. *Nothing downstream is valid until `tests/test_features.py` and `tests/test_splits.py` pass.*
2. **Sweep** (5.3): choose headline horizon/threshold from evidence.
3. **Sequence dataset + models** (1.6, 2.1–2.5): forward-pass tests.
4. **Tune** (3.1–3.2): one study per cell; freeze configs.
5. **Train** the grid × 5 seeds. Minimum grid: 2 markets × 2 feature sets × 2 label rules × 7 models (logistic, RF, XGB, DeepLOB, windowed Transformer, LevelTransformer, TransLOB/DeepLOB-Attn) × 5 seeds = 280 runs (+ ETH). Budget accordingly; trees are fast, neural runs dominate.
6. **Ablations** (5.5), **backtest** (5.6), **regime** (5.4), **significance** (5.2), **transfer** (5.7), **importance** (5.8).
7. **Aggregate → LaTeX → figures**, then write. Do not draft the paper before step 6 — the framing depends on what the feature-set decomposition and backtest show.

## 12. Definition of done (pre-submission checklist)

- [ ] Every number in the abstract appears in a table generated by `aggregate.py`.
- [ ] Every model description in §IV matches `models/*.py` line for line.
- [ ] Dataset citation, date range, sampling interval, and gaps match `crypto_quality_report.json`.
- [ ] FI-2010 horizon convention verified and cited.
- [ ] Split hash identical across seeds for every run.
- [ ] Tuned hyperparameters frozen before seeded runs; appendix table present.
- [ ] Per-class F1 + confusion matrices for every model/market.
- [ ] Significance markers on every headline comparison.
- [ ] Backtest with costs, latency, confidence gate, random and perfect-foresight baselines, both markets.
- [ ] Ablation tables at 5 seeds with parameter counts.
- [ ] At least two crypto assets (or two exchanges).
- [ ] Limitations section lists only what is still open.
- [ ] `reproduce.sh` regenerates every table and figure from raw data on a clean machine.
