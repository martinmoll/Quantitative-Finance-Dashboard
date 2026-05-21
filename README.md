# Quantitative Finance Dashboard

A Streamlit-based interactive backtesting dashboard for ML-driven equity strategies. Supports long-only and long-short portfolios with equal-weight allocation, walk-forward backtesting, volatility tilt, momentum regime filter, and strategy comparison.



## Quick Start

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
cd dashboard
streamlit run app.py
```



## Dataset

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



## Feature Engineering

Two feature tiers are constructed from the `_xs` columns. All engineered features are pure arithmetic on per-month standardized values — no look-ahead.

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

**Non-linear transforms** (5 features) — signed squared terms for key signals (`ret_2_12_xs`, `ret_1_xs`, `sue_xs`, `bm_xs`, `revision_xs`). Allow the linear Lasso model to capture diminishing or accelerating effects.

**Composite signals** (6 features) — averages across related columns:

| Composite | Component columns |
|---|---|
| `earnings_composite` | `sue_xs`, `sue_q_xs`, `rev_surp_xs`, `earn_growth_yoy_xs`, `beat_xs` |
| `quality_composite` | `gpa_xs`, `roe_xs`, `roa_xs`, `earn_quality_xs`, `cfo_at_xs` |
| `value_composite` | `bm_xs`, `ep_xs`, `cfp_xs`, `sp_xs` |
| `momentum_composite` | `ret_2_12_xs`, `ret_2_6_xs`, `ret_13_36_xs`, `prc_52w_high_xs` |
| `technical_composite` | `rsi_14_xs`, `macd_hist_xs`, `bb_position_xs`, `roc_3_xs`, `roc_6_xs` |
| `analyst_composite` | `revision_xs`, `revision_ratio_xs`, `rec_chg_xs`, `n_analysts_xs` |

Composites average only columns that actually exist in the data. An earlier version padded missing columns with zeros, which diluted composite signals. Correcting this was the single largest driver of model improvement.

**Relative signals** (3 features) — peer-adjusted measures (`sue_vs_peer`, `revision_vs_peer`) and a reversal-momentum combo (`ret_2_12_xs - ret_1_xs`).

**Composite interactions** (3 features) — products of composite signals (`earn_x_mom`, `quality_x_value`, `earn_x_lowvol`).



## Models

| Model | Estimator | Features | Key Settings |
|---|---|---|---|
| Lasso | `LassoCV` (5-fold CV) | Tier 1 (~52) | L1 regularization, automatic alpha selection |
| HGB | `HistGradientBoostingRegressor` | Tier 2 (~118) | `max_iter=500`, `max_depth=2`, `lr=0.05`, `min_samples_leaf=500`, `l2_reg=0.1` |

HGB settings are deliberately conservative: very shallow trees (depth 2), large leaf sizes (500), and L2 regularization. Overfitting is the dominant failure mode in financial prediction.



## Walk-Forward Backtesting

Expanding-window design with 12-month retraining intervals:

```
For each retraining window (every 12 months from 2005-01):
    1. Training set = all data before the current OOS window
    2. Fit model on (feature_builder(train), train['y_xs'])
    3. For each month in the next 12 OOS months:
        a. Generate predictions for all stocks this month
        b. Apply volatility tilt (if configured)
        c. Apply momentum regime filter (if configured)
        d. Select top K stocks by adjusted prediction
        e. Portfolio return = equal-weighted mean of y_raw for selected stocks
```

Key design choices:

- **Train on `y_xs`** (standardized target) for better cross-sectional pattern learning
- **Evaluate on `y_raw`** (actual returns) for realistic portfolio performance
- **Expanding window:** each retraining uses all available historical data, not a fixed lookback
- **No future leakage:** all features are per-month `_xs` values; composites are pure arithmetic



## Strategy Modes

**Long-only:** Equal-weight portfolio of the top K stocks by model prediction (optionally adjusted for volatility tilt). Monthly rebalancing.

**Long-short (dollar-neutral):** Return = mean(top K stocks) - mean(bottom K_short stocks). The long and short legs are sized equally, giving zero net market exposure. Positive return requires the long leg to outperform the short leg.



## Dashboard Controls

| Control | Range / Options | Default |
|---|---|---|
| Model | HGB, Lasso | HGB |
| Strategy | Long-only, Long-short | Long-only |
| K (long holdings) | 5 – 100 | 30 |
| K_short (short holdings) | 5 – 100 | 30 |
| Vol tilt | 0.00 – 0.30 | 0.05 |
| Regime lookback (months) | 1 – 12 | 3 |

Additional expandable sections provide:

- **Hyperparameter controls:** HGB `max_iter`, `max_depth`, `learning_rate`, `min_samples_leaf`; Lasso `cv` folds
- **Display controls:** Benchmark overlay, chart date range, return/drawdown toggle
- **Pin & compare:** Save the current run's equity curve and metrics alongside a new configuration for direct comparison



## Experiment Results

| Config | SR | Ann Return | Ann Vol | MDD |
|---|---|---|---|---|
| Baseline HGB (no tilt) | 1.09 | 24.2% | 22.3% | -46.0% |
| HGB vt=0.03 | 1.17 | 23.5% | 20.2% | -48.9% |
| **HGB vt=0.05** | **1.20** | **22.0%** | **18.3%** | **-38.7%** |
| HGB vt=0.07 | 1.17 | 20.4% | 17.4% | -39.0% |
| HGB vt=0.10 | 1.12 | 18.4% | 16.4% | -37.6% |
| SPY benchmark | 0.66 | 10.4% | 15.7% | -50.3% |

At vt=0.05, return drops modestly (24.2% to 22.0%) while volatility drops proportionally more (22.3% to 18.3%), improving the Sharpe Ratio from 1.09 to 1.20.



## Repository Structure

```
Alpha Model/
  dashboard/
    app.py                    # Streamlit dashboard entry point
    controls.py               # Sidebar control definitions
    charts.py                 # Chart rendering functions
    backtest_runner.py        # Walk-forward backtest engine
  Code/
    AlphaChallenge.ipynb      # Research notebook (feature engineering, model experiments)
  Data/
    alpha_dataset_v2.csv      # Dataset (not tracked in git)
  docs/
    mdd_reduction_log.md      # Detailed log of volatility tilt experiments
    superpowers/
      specs/                  # Design specifications
      plans/                  # Implementation plans
  README.md                   # This file
```
