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

### Grading Rubric — Performance Component (20 points)

**Disqualifying conditions (automatic 0/20):**
- Look-ahead bias in signal construction or backtesting
- Overfitting or excessive parameterization (insufficient OOS validation, parameter count disproportionate to sample size, in-sample tuning without proper holdout)
- Maximum drawdown exceeding 40%

**Performance tiers (OOS Sharpe Ratio):**

| Sharpe Ratio | Minimum Score |
|---|---|
| SR > 1.7 | 18 / 20 |
| 1.5 < SR ≤ 1.7 | 15 / 20 |
| 1.3 < SR ≤ 1.5 | 12 / 20 |
| 1.2 < SR ≤ 1.3 | 10 / 20 |
| SR ≤ 1.2 | Instructor's discretion |

Scores above tier minimum for: transaction cost realism, robustness analysis, clarity of methodology.

**Current status:** Best strategy HGB_vt0.05 has SR=1.20, MDD=-38.7% → 10/20 tier. **Target: SR > 1.3 minimum (12/20), stretch goal SR > 1.5 (15/20).**

### Per-Model Metrics (via existing `perf()` helper)
- Annualized Sharpe Ratio (primary — determines grade tier)
- Annualized return and volatility
- Maximum Drawdown (hard constraint: < 40%, disqualifying if exceeded)
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

---

## SR Improvement Strategies (Post-Rubric Update)

Current best: HGB_vt0.05 → SR=1.20, MDD=-38.7%. Lands in 10/20 tier. Need SR > 1.3 for 12/20, ideally > 1.5 for 15/20.

**Key constraint:** Must avoid overfitting — the rubric explicitly flags "excessive parameterization" and "in-sample tuning without proper holdout" as disqualifying. Every strategy below must be justifiable as economically motivated, not data-mined.

### Strategy A: Retrain Frequency Tuning
- Current: retrain every 12 months (expanding window)
- Test: retrain every 6 months, every 3 months
- Rationale: more frequent retraining captures regime shifts faster. Financial relationships change — a model trained in 2005 may be stale by 2006. More frequent retraining is standard practice, not overfitting, as long as the model itself isn't re-tuned.
- Risk: overfitting concern is low — we're not adding parameters, just updating the same model more often on fresh data.

### Strategy B: Portfolio Concentration (K Tuning)
- Current: K=30 (top 30 stocks, equal weight)
- Test: K=20, K=15, K=10
- Rationale: if the model's top picks are genuinely better, concentrating on fewer stocks should increase returns. K=30 may dilute alpha by including weaker picks at positions 20-30.
- Risk: higher concentration = higher vol. But if return rises proportionally more than vol, SR improves. Need to verify MDD stays under 40%.

### Strategy C: Prediction-Weighted Portfolio
- Current: equal weight across top K stocks
- Test: weight stocks by predicted alpha (prediction-proportional weighting within top K)
- Rationale: the model's #1 pick should get more weight than pick #30. Equal weighting throws away the model's conviction signal.
- Risk: if the model's ranking is noisy, this concentrates on noise. Start with a mild version: weight = softmax(predictions) or clipped prediction weights.

### Strategy D: Feature Selection / Reduction for HGB
- Current: ~97 Tier 2 features
- Test: reduce to 30-50 most important features (based on HGB feature importances from a preliminary run)
- Rationale: fewer noisy features = cleaner signal for the model. Too many features dilute the important ones and can lead to spurious splits. This also directly addresses the rubric's overfitting concern — a model with fewer features is harder to call "excessively parameterized."
- Method: train HGB once on full feature set, extract feature importances, keep top N features, retrain.

### Strategy E: Target Variable Transformation
- Current: predicting `y_xs` (cross-sectional standardized returns)
- Test: predict `rank(y_raw)` or winsorized `y_xs` (clip at ±3 sigma)
- Rationale: extreme returns (10x, -90%) distort the loss function. Winsorizing or rank-transforming the target reduces the influence of outliers on model fitting. The model only needs to rank stocks correctly, not predict magnitude.

### Strategy F: Ensemble of HGB Variants
- Current: single HGB model
- Test: average predictions from 2-3 HGB models with different hyperparameters (e.g., different max_depth, learning_rate) or different random seeds
- Rationale: reduces model variance. If each HGB model makes slightly different errors, averaging smooths them out. This is standard ensemble practice, not overfitting.
- Must use fixed hyperparameters (no tuning), just diversity through different configs.

### Strategy G: Momentum Regime Filter
- Current: always invest in top K stocks regardless of market conditions
- Test: reduce position size or go to cash when trailing 12-month market return is negative (bear market filter)
- Rationale: long-only strategy inherently loses in bear markets. A simple "don't invest in bear markets" rule reduces drawdowns AND removes negative-return months, improving SR. Uses only backward-looking data.
- Risk: may miss recovery rallies. But for SR purposes, avoiding the worst months helps more than catching the first bounce.

### Strategy H: Minimum Holding Period / Turnover Reduction
- Current: fully rebalance monthly (new top-K every month)
- Test: only replace stocks that drop below rank 2K (i.e., keep a stock unless it falls out of the top 60)
- Rationale: reduces turnover, which improves transaction cost realism (rubric bonus). Also reduces whipsawing — a stock that was #29 last month and #31 this month shouldn't trigger a trade.

### Testing Priority
1. **Strategy B (K tuning)** — simplest change, no model modification, could have big SR impact
2. **Strategy A (retrain frequency)** — also simple, just a constant change
3. **Strategy E (target winsorizing)** — easy to implement, addresses a real statistical issue
4. **Strategy D (feature selection)** — reduces overfitting risk, aligns with rubric
5. **Strategy G (momentum regime filter)** — could dramatically help MDD and SR
6. **Strategy F (HGB ensemble)** — moderate effort, reliable SR improvement
7. **Strategy C (prediction-weighted)** — moderate effort, uncertain payoff
8. **Strategy H (turnover reduction)** — lower priority for SR but good for rubric bonus points

---

## Simplification Path (Approach B Fallback)

To simplify to single-model approach at any time:
1. Keep only HistGradientBoosting in the model dictionary
2. Create 2-3 variations of the Tier 2 feature set instead
3. Compare feature sets rather than models
4. All other infrastructure (walk-forward, evaluation, plotting) remains unchanged
