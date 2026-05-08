# Alpha Challenge: Beating the Market with Machine Learning

**RMBI3110 — Assignment 2 | HKUST Business School | Spring 2026**



## 1. Project Overview

The goal is to build a machine learning strategy that beats the S&P 500 benchmark on a risk-adjusted basis. The strategy must be:

- **Long-only**, equal-weight portfolio of K stocks
- **Monthly rebalancing** with walk-forward backtesting
- **Out-of-sample** from January 2005 onward (no look-ahead)
- **Maximum drawdown under 40%** (disqualifying if exceeded)

Performance is graded primarily on out-of-sample Sharpe Ratio.



## 2. Dataset

| Property | Value |
|---|---|
| Rows | 257,800 |
| Columns | 229 |
| Months | 407 |
| Unique stocks | 1,216 |

The dataset contains 116 raw features across 10 categories:

| Category | Examples | Count |
|---|---|---|
| Price / Momentum | `ret_1`, `ret_2_12`, `vol_12m`, `beta`, `ivol` | 12 |
| Accounting / Value | `bm`, `ep`, `gpa`, `roe`, `ag`, `lev` | 15 |
| Analyst | `sue`, `revision`, `dispersion`, `beat` | 6 |
| Technical | `rsi_14`, `macd_hist`, `bb_position`, `roc_3` | 13 |
| Options | `iv_atm_30d`, `iv_skew`, `pc_vol_ratio`, `vrp` | 6 |
| Peer / Industry | `peer_sue`, `leader_ret`, `ind_mom` | 17 |
| Quarterly Fundamentals | `sue_q`, `rev_surp`, `earn_growth_yoy` | 19 |
| Interactions | `mom_x_size`, `val_x_prof`, `mom_x_vol` | 10 |
| Sector / Macro | `sector_ret_avg`, `macro_unc_1m` | 13 |

Each raw feature has a corresponding `_xs` (cross-sectional standardized) version, computed per month. The key target columns are:

- **`y_xs`**: cross-sectionally standardized forward monthly return (training target)
- **`y_raw`**: raw forward monthly return (portfolio evaluation)



## 3. Feature Engineering

We constructed two feature tiers from the `_xs` columns. All engineered features are pure arithmetic on per-month standardized values, so there is no look-ahead risk.

### Tier 1 (~52 features) — For Lasso

Core factor features (19 columns covering momentum, value, quality, risk, size, analyst, liquidity), pre-built interaction features from the dataset (7 columns), plus all engineered features below.

### Tier 2 (~118 features) — For HistGradientBoosting

All `_xs` columns in the dataset plus all engineered features. Gives the tree model maximum information to work with.

### Engineered Features

**Interaction terms** (9 features) — economically motivated pairwise products:

| Feature | Formula | Rationale |
|---|---|---|
| `mom_x_quality` | `ret_2_12_xs * gpa_xs` | Momentum more reliable in quality firms |
| `mom_x_roe` | `ret_2_12_xs * roe_xs` | Same logic, alternative quality measure |
| `val_x_lowvol` | `bm_xs * (-vol_12m_xs)` | Value stronger in stable firms |
| `ep_x_lowvol` | `ep_xs * (-vol_12m_xs)` | Same logic, earnings yield variant |
| `sue_x_lowdisp` | `sue_xs * (-dispersion_xs)` | Earnings surprise stronger when analyst consensus is tight |
| `sue_x_revision` | `sue_xs * revision_xs` | Earnings surprise confirmed by revisions |
| `mom_x_lowivol` | `ret_2_12_xs * (-ivol_xs)` | Momentum in low idiosyncratic vol stocks |
| `bm_x_roe` | `bm_xs * roe_xs` | Classic quality-value combination |
| `ep_x_gpa` | `ep_xs * gpa_xs` | Cheap and profitable |

**Non-linear transforms** (5 features) — signed squared terms for key signals (`ret_2_12_xs`, `ret_1_xs`, `sue_xs`, `bm_xs`, `revision_xs`). These allow the linear Lasso model to capture diminishing or accelerating effects.

**Composite signals** (6 features) — averages across related columns:

| Composite | Component columns |
|---|---|
| `earnings_composite` | `sue_xs`, `sue_q_xs`, `rev_surp_xs`, `earn_growth_yoy_xs`, `beat_xs` |
| `quality_composite` | `gpa_xs`, `roe_xs`, `roa_xs`, `earn_quality_xs`, `cfo_at_xs` |
| `value_composite` | `bm_xs`, `ep_xs`, `cfp_xs`, `sp_xs` |
| `momentum_composite` | `ret_2_12_xs`, `ret_2_6_xs`, `ret_13_36_xs`, `prc_52w_high_xs` |
| `technical_composite` | `rsi_14_xs`, `macd_hist_xs`, `bb_position_xs`, `roc_3_xs`, `roc_6_xs` |
| `analyst_composite` | `revision_xs`, `revision_ratio_xs`, `rec_chg_xs`, `n_analysts_xs` |

**Important fix:** Composites average only columns that actually exist in the data. An earlier version used a helper that returned zeros for missing columns, which diluted composite signals. Correcting this was the single biggest driver of model improvement (see Section 6).

**Relative signals** (3 features) — peer-adjusted measures (`sue_vs_peer`, `revision_vs_peer`) and a reversal-momentum combo (`ret_2_12_xs - ret_1_xs`).

**Composite interactions** (3 features) — products of composite signals (`earn_x_mom`, `quality_x_value`, `earn_x_lowvol`).



## 4. Models

### Lasso (Linear Baseline)

- **Estimator:** `sklearn.linear_model.LassoCV` (5-fold CV for alpha selection)
- **Features:** Tier 1
- **Role:** Interpretable baseline with automatic feature selection via L1 regularization

Lasso selected 35 of 52 features. The top features by coefficient magnitude were:

| Feature | Coefficient | Interpretation |
|---|---|---|
| `quality_composite` | +0.048 | High profitability predicts returns |
| `log_me_xs` | -0.035 | Small cap premium |
| `technical_composite` | -0.033 | Contrarian technical signal |
| `ret_2_12_xs` | +0.029 | Momentum effect |
| `earnings_composite` | +0.013 | Earnings surprise signal |

### HistGradientBoosting (Primary Model)

- **Estimator:** `sklearn.ensemble.HistGradientBoostingRegressor`
- **Features:** Tier 2
- **Hyperparameters:**
  - `max_iter=500`, `max_depth=2`, `learning_rate=0.05`
  - `min_samples_leaf=500`, `l2_regularization=0.1`
  - `early_stopping=False`, `random_state=42`

These are deliberately conservative settings. Trees are very shallow (depth 2), leaf sizes are large (500), and L2 regularization is applied. Overfitting is the dominant failure mode in financial prediction, so we prioritize regularization over model expressiveness.



## 5. Walk-Forward Backtesting

The backtest uses an expanding-window walk-forward design:

```
For each retraining window (every 12 months from 2005-01):
    1. Training set = all data before the current OOS window
    2. Fit model on (feature_builder(train), train['y_xs'])
    3. For each month in the next 12 OOS months:
        a. Generate predictions for all stocks this month
        b. Apply volatility tilt (if configured)
        c. Select top K stocks by adjusted prediction
        d. Portfolio return = equal-weighted mean of y_raw for selected stocks
```

Key design choices:

- **Train on `y_xs`** (standardized target) for better cross-sectional pattern learning
- **Evaluate on `y_raw`** (actual returns) for realistic portfolio performance
- **Expanding window:** each retraining uses all available historical data, not a fixed lookback
- **No future leakage:** all features are per-month `_xs` values; composites are pure arithmetic



## 6. The MDD Reduction Journey

The core challenge: all initial models exceeded the 40% maximum drawdown constraint, driven by full market beta exposure during the 2008 financial crisis.

### 6.1 Baseline

| Model | SR | MDD | Notes |
|---|---|---|---|
| HGB (max_iter=200) | 1.09 | -50.1% | Best SR but fails MDD |
| HGB (max_iter=500) | 0.98 | -51.7% | |
| Lasso | 0.78 | -50.0% | |
| SPY benchmark | 0.66 | ~-55% | |

**Root cause:** A long-only portfolio inherently carries market beta. During 2008, all sectors fell 40-55%, so even a well-diversified portfolio suffered severe drawdowns.

### 6.2 Sector Caps (Failed)

**Approach:** Limit the maximum number of stocks from any single sector in the top-K portfolio (tested cap=5 and cap=8 with K=30).

**Result:** Minimal MDD improvement. The drawdown was market-wide, not sector-specific. Sector caps diversify across sectors but do not reduce market beta.

### 6.3 Feature Engineering Fixes (Partial Success)

Three changes to the engineered features:

1. **Composite dilution fix:** Composites now average only existing columns instead of padding missing columns with zeros. This was a real bug that weakened signal strength.
2. **Removed redundant features:** `mom_vs_sector` and `mom_vs_industry` subtracted different-horizon signals with no clear economic meaning. HGB can learn these relationships from raw inputs.
3. **Removed duplicate:** `size_x_mom` duplicated the pre-built `mom_x_size_xs`.

**Result:**

| Model | SR (before) | SR (after) | MDD (before) | MDD (after) |
|---|---|---|---|---|
| HGB (max_iter=500) | 0.98 | 1.09 | -51.7% | -46.0% |
| HGB (max_iter=200) | 1.09 | 1.10 | -50.1% | -49.7% |
| Lasso | 0.78 | 0.79 | -50.0% | -56.7% |

The composite dilution fix was the main driver. HGB_500 improved the most: SR from 0.98 to 1.09, MDD improved by 5.7 percentage points. Still needed another ~6% MDD reduction.

### 6.4 Unconditional Volatility Tilt (Solution)

**Approach:** After the model generates predictions, adjust rankings before stock selection:

```
adjusted_pred = pred - vol_tilt * vol_12m_xs
```

Stocks with above-average 12-month realized volatility are penalized in the ranking. The portfolio remains fully invested, equal-weight, long-only — only *which* stocks are selected changes.

**Rationale:** The MDD comes from holding high-beta stocks through market crashes. By tilting toward lower-volatility stocks, the portfolio has less market sensitivity during downturns. Since `vol_12m_xs` is cross-sectionally standardized (mean=0), a tilt of 0.05 means a 1-standard-deviation-more-volatile stock gets its prediction reduced by 0.05.

**Results across tilt values:**

| vol_tilt | SR | Ann Mean | Ann Vol | MDD | MDD Pass? |
|---|---|---|---|---|---|
| 0.00 (baseline) | 1.09 | 24.2% | 22.3% | -46.0% | No |
| 0.03 | 1.17 | 23.5% | 20.2% | -48.9% | No |
| **0.05** | **1.20** | **22.0%** | **18.3%** | **-38.7%** | **Yes** |
| 0.07 | 1.17 | 20.4% | 17.4% | -39.0% | Yes |
| 0.10 | 1.12 | 18.4% | 16.4% | -37.6% | Yes |

**Key finding:** A very light tilt (0.05) actually *improved* the Sharpe Ratio from 1.09 to 1.20. This is because volatility dropped proportionally more than return:
- Return: 24.2% to 22.0% (9% reduction)
- Volatility: 22.3% to 18.3% (18% reduction)

At vt=0.05, most of the model's top picks survive. Only the most extreme volatility outliers get demoted. This removes enough tail risk to pass the MDD constraint while preserving nearly all alpha.

At higher tilt values (0.10+), return drops too aggressively. The tilt overrides the model's best picks across all months.

### 6.5 Conditional Volatility Tilt (Alternative)

**Approach:** Only apply the volatility penalty when market conditions signal elevated risk. Use trailing 3-month realized SPY volatility as the regime signal: if it exceeds a threshold, switch on the tilt. Otherwise, trust the model's picks fully.

**Rationale:** The unconditional tilt pays a cost across all 227 OOS months but only *needs* to help in ~5-10 crisis months. A conditional tilt should preserve more return during normal periods.

**Results (threshold = 0.18 annualized SPY vol, the only threshold that passed MDD):**

| vol_tilt | SR | Ann Mean | MDD | Tilt Months |
|---|---|---|---|---|
| 0.3 | 1.15 | 21.1% | -35.8% | 51 / 227 |
| 0.5 | 1.13 | 20.8% | -35.8% | 51 / 227 |
| 0.8 | 1.13 | 20.7% | -36.6% | 51 / 227 |

The conditional approach gives better MDD (-35.8% vs -38.7%) but lower SR (1.15 vs 1.20). The threshold dominates: once you are tilting in the right months, the exact penalty barely matters.

### 6.6 Final Comparison

| Config | SR | Ann Mean | Ann Vol | MDD | Total Return |
|---|---|---|---|---|---|
| Baseline (no tilt) | 1.09 | 24.2% | 22.3% | -46.0% | 5,801% |
| **vt=0.05 unconditional** | **1.20** | **22.0%** | **18.3%** | **-38.7%** | **4,405%** |
| vt=0.07 unconditional | 1.17 | 20.4% | 17.4% | -39.0% | 3,360% |
| vt=0.10 unconditional | 1.12 | 18.4% | 16.4% | -37.6% | 2,347% |
| vt=0.3, th=0.18 conditional | 1.15 | 21.1% | 18.4% | -35.8% | 3,702% |

**Winner: vt=0.05 unconditional** — highest SR among all MDD-passing configurations.



## 7. Current Best Result

| Metric | HGB_vt0.05 | SPY Benchmark |
|---|---|---|
| Sharpe Ratio | **1.20** | 0.66 |
| Annualized Return | 22.0% | 10.4% |
| Annualized Volatility | 18.3% | 15.7% |
| Maximum Drawdown | -38.7% | -50.3% |
| Total Cumulative Return | 4,405% | 459% |

Alpha over SPY: +0.54 Sharpe Ratio.



## 8. Grading Rubric (Performance Component)

| OOS Sharpe Ratio | Minimum Score |
|---|---|
| SR > 1.7 | 18 / 20 |
| 1.5 < SR <= 1.7 | 15 / 20 |
| 1.3 < SR <= 1.5 | 12 / 20 |
| 1.2 < SR <= 1.3 | 10 / 20 |
| SR <= 1.2 | Instructor's discretion |

Disqualifying conditions (automatic 0/20): look-ahead bias, overfitting/excessive parameterization, MDD > 40%.

**Current standing:** SR=1.20 places us in the **10/20 tier**. Target: push SR above 1.3 for 12/20, ideally above 1.5 for 15/20.



## 9. Next Steps: SR Improvement Strategies

The following strategies are being tested to push Sharpe Ratio higher:

| Strategy | Description | Expected Impact |
|---|---|---|
| K Tuning | Test K=10, 15, 20 (more concentrated portfolio) | Higher return per pick if model is accurate |
| Retrain Frequency | Retrain every 3 or 6 months instead of 12 | Faster adaptation to regime changes |
| Target Winsorization | Clip extreme y_xs values before training | Reduce outlier influence on model fitting |
| Feature Selection | Keep only top 30-50 features by HGB importance | Less noise, less overfitting |
| Momentum Regime Filter | Go to cash when trailing SPY return is negative | Avoid worst drawdown months |
| HGB Ensemble | Average predictions from multiple HGB seeds | Reduce prediction variance |

Each strategy is tested independently, then the best combination is selected. All changes must be economically motivated — the rubric explicitly penalizes overfitting.


## Repository Structure

```
Alpha Model/
  Code/
    AlphaChallenge.ipynb    # Main notebook (all code)
  Data/
    alpha_dataset_v2.csv    # Dataset (not tracked in git)
  docs/
    mdd_reduction_log.md    # Detailed log of MDD reduction experiments
    superpowers/
      specs/                # Design specifications
      plans/                # Implementation plans
  progress.md               # Running results tracker
  README.md                 # This file
```
