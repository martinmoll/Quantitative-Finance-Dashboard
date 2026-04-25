# Alpha Challenge Strategy Design

## Overview

Build a multi-model walk-forward trading strategy to predict cross-sectional stock returns and beat the S&P 500 benchmark (OOS Sharpe Ratio = 0.66). Three models of increasing complexity, each with tailored feature engineering, compared on out-of-sample performance. Long-only portfolio, K=30 stocks, equal-weight, monthly rebalancing from 2005-01 onward.

**Design principle:** Each model is defined by a `(feature_builder, estimator)` pair. Adding, removing, or swapping models is a one-line change. This allows simplification to a single-model approach (Approach B) at any time.

---

## Architecture

The notebook is organized into these sequential blocks:

1. **Setup** (existing) — imports, data loading, constants, helper functions
2. **Feature Engineering** — two feature-builder functions (Tier 1 and Tier 2)
3. **Model Definitions** — dictionary mapping model names to `(feature_builder, estimator)` pairs
4. **Walk-Forward Engine** — a single `run_model()` function that takes a feature builder and estimator, returns monthly return series
5. **Evaluation & Comparison** — summary table, cumulative wealth plots, long-short diagnostics
6. **Optional Blend** — average predictions from top models, re-evaluate
7. **Writeup Template** — structured markdown cell for Task 2

---

## Feature Engineering

All features built from existing `_xs` (cross-sectional standardized) columns. No raw feature standardization. All composites are simple arithmetic on per-month `_xs` values — no time-series operations, no look-ahead risk.

**Missing values:** `_xs` features may contain NaNs (e.g., stocks missing analyst coverage). Strategy: fill NaN with 0.0 (the cross-sectional median, since `_xs` features are median-centered). This is neutral — it says "this stock is average on this feature." HistGradientBoosting handles NaN natively, but Lasso and RF do not, so filling is required for consistency.

### Tier 1 — Linear Feature Set (~25-35 features, for Lasso)

**Core factor features:**
- Momentum: `ret_1_xs`, `ret_2_12_xs`
- Value: `bm_xs`, `ep_xs`
- Quality: `gpa_xs`, `roe_xs`
- Volatility: `vol_12m_xs`, `ivol_xs`
- Analyst: `sue_xs`, `revision_xs`

**Pre-built interactions from dataset:**
- `mom_x_size_xs`, `val_x_prof_xs`, `mom_x_vol_xs`, and other existing interaction `_xs` columns

**Hand-crafted composites:**
- Quality composite: mean of `gpa_xs`, `roe_xs`, negated `ag_xs`
- Earnings momentum: mean of `sue_xs`, `revision_xs`
- Reversal + momentum combo: `ret_2_12_xs` minus `ret_1_xs`

### Tier 2 — Ensemble Feature Set (~50-70 features, for RF and HGB)

**Everything from Tier 1, plus:**
- All remaining `_xs` features (technical, options, peer/industry, quarterly fundamentals)
- Sector-relative momentum: `ret_2_12_xs` minus `ind_mom_xs`
- Peer-adjusted earnings: `sue_xs` minus `peer_sue_xs`
- Value-momentum combo: `bm_xs` + `ret_2_12_xs`

---

## Model Definitions

### 1. Lasso (Linear)
- **Estimator:** `sklearn.linear_model.Lasso`
- **Features:** Tier 1
- **Hyperparameters:** `alpha=0.001`
- **Rationale:** Automatic feature selection via L1 regularization. Provides interpretable baseline and insight into which features matter.

### 2. Random Forest
- **Estimator:** `sklearn.ensemble.RandomForestRegressor`
- **Features:** Tier 2
- **Hyperparameters:**
  - `n_estimators=300`
  - `max_depth=5`
  - `max_features=0.33`
  - `min_samples_leaf=50`
- **Rationale:** Captures non-linear relationships and feature interactions. Shallow trees and large leaf sizes prevent overfitting.

### 3. HistGradientBoosting
- **Estimator:** `sklearn.ensemble.HistGradientBoostingRegressor`
- **Features:** Tier 2
- **Hyperparameters:**
  - `max_iter=300`
  - `max_depth=4`
  - `learning_rate=0.05`
  - `min_samples_leaf=100`
  - `early_stopping=False`
- **Rationale:** Strongest model for tabular data. Shallower than RF since sequential boosting builds complexity incrementally. No early stopping because we control training data via walk-forward.

**Philosophy:** All models configured conservatively (shallow, regularized). Overfitting is the dominant failure mode in finance.

---

## Walk-Forward Engine

### Core Loop

```
run_model(df, feature_builder, estimator, name, K=30):
    For each retraining window (every 12 months from OOS_START='2005-01'):
        1. Training set = all months before current OOS window
        2. X_train = feature_builder(training data)
        3. y_train = training data['y_xs']
        4. Fit estimator on (X_train, y_train)
        5. For each month in next 12 OOS months:
            a. X_test = feature_builder(this month's data)
            b. predictions = estimator.predict(X_test)
            c. Long-only: top K stocks by prediction, equal-weight, return using y_raw
            d. Long-short: top K minus bottom K (diagnostic only)
    Return: dict with long-only and long-short monthly return Series
```

### Key Details
- **Train on `y_xs`** (standardized target, better for learning cross-sectional patterns)
- **Evaluate on `y_raw`** (actual returns, for realistic portfolio performance)
- **Retrain every 12 months** via expanding window
- **No future leakage:** feature builders only use `_xs` columns (already per-month standardized), composites are pure arithmetic

---

## Evaluation & Comparison

### Per-Model Metrics (via existing `perf()` helper)
- Annualized Sharpe Ratio (primary)
- Annualized return and volatility
- Maximum Drawdown (constraint: < 40%)
- Total cumulative return

### Outputs
1. **Summary table** — all models + SPY benchmark side by side
2. **Cumulative wealth plot** — all models + SPY on one chart (via `plot_strats()`)
3. **Long-short diagnostics** — same metrics for long-short versions

### Optional Blend
- If 2+ models individually beat SPY: average their monthly predictions before ranking
- Re-run top-K selection on blended predictions
- Keep blend only if it improves SR over best individual model

---

## Writeup Template (Task 2)

Structured markdown cell covering:
- **Features:** Two feature tiers, what composites/interactions were created and why
- **Models:** Three models tested, key hyperparameter choices and rationale
- **What worked:** Best model, best features, blend results, surprising findings from Lasso feature selection
- **What didn't work:** Underperforming models/features, overfitting issues encountered

---

## Simplification Path (Approach B Fallback)

To simplify to single-model approach at any time:
1. Keep only HistGradientBoosting in the model dictionary
2. Create 2-3 variations of the Tier 2 feature set instead
3. Compare feature sets rather than models
4. All other infrastructure (walk-forward, evaluation, plotting) remains unchanged
