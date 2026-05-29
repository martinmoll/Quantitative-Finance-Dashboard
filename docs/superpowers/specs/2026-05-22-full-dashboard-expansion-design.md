# Full Dashboard Expansion — Design Spec

**Date**: 2026-05-22
**Status**: Draft
**Source**: RMBI3110 course material (docs/RMBI3110_Dashboard_Spec.md), adapted to existing codebase

## Goal

Transform the current single-page Alpha Model dashboard into a full 7-page Bloomberg-style application implementing the complete quantitative alpha pipeline from RMBI3110. Add advanced strategies, portfolio construction methods, monitoring tools, and integrated educational content explaining the theories and methodologies.

---

## 1. Project Structure & Migration

### Current State

```
dashboard/
├── app.py              # Single-page UI (~534 lines): controls + results display
├── engine.py           # Walk-forward predictions + portfolio construction
├── features.py         # Two-tier feature engineering (Tier 1: ~52, Tier 2: ~118)
├── cache_manager.py    # Two-tier disk caching (predictions + portfolios)
└── __init__.py
```

### Target State

```
dashboard/
├── app.py                  # Slim router: page config, data loading, session state init
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Factor_Analysis.py
│   ├── 3_Alpha_Model_Lab.py
│   ├── 4_Backtest_Results.py
│   ├── 5_Portfolio_Construction.py
│   ├── 6_Monitoring.py
│   └── 7_Theory_Methods.py
├── core/
│   ├── models.py           # Model registry (6 models, extensible)
│   ├── backtest.py         # Walk-forward engine
│   ├── portfolio.py        # 5 construction methods
│   ├── factor_models.py    # CAPM, FF3, FF5, rolling beta, VIF, Wald
│   ├── diagnostics.py      # IC, ICIR, Fundamental Law, KS, alpha decay
│   ├── risk.py             # Risk contribution, turnover, TC, factor exposure
│   └── data_loader.py      # Dataset + FF5 loading and validation
├── components/
│   ├── charts.py           # Reusable Plotly chart builders (Bloomberg style)
│   ├── metrics.py          # Metric cards, comparison tables, regression output
│   ├── theory.py           # Expandable theory section component
│   └── theory_content.py   # All educational text (markdown + LaTeX), keyed by topic
├── features.py             # Unchanged
├── cache_manager.py        # Extended (new portfolio method key)
└── __init__.py
```

### Migration Mapping

| Current location | Destination | Notes |
|---|---|---|
| `engine.py: run_predictions()` | `core/backtest.py` | Now accepts AlphaModel instance |
| `engine.py: build_portfolio()` | `core/portfolio.py: build_portfolio_series()` | Extended with 5 methods |
| `engine.py: compute_perf()` | `core/diagnostics.py` | Plus new metrics |
| `engine.py: compute_market_monthly()` | `core/data_loader.py` | Plus FF5 loading |
| `engine.py: model instantiation` | `core/models.py` | Registry pattern |
| `app.py: control bar` | `pages/3_Alpha_Model_Lab.py` | Dynamic controls from registry |
| `app.py: results display` | `pages/4_Backtest_Results.py` | Plus new panels |
| `app.py: sector/holdings` | `pages/5_Portfolio_Construction.py` | Plus risk/TC analysis |
| `app.py: chart building (inline)` | `components/charts.py` | Reusable builders |
| `app.py: metric rendering` | `components/metrics.py` | Reusable components |
| `app.py: data loading` | `app.py` (stays, slimmed) | Only init + session state |
| `features.py` | `features.py` (unchanged) | Already well-structured |
| `cache_manager.py` | `cache_manager.py` (extended) | Add construction method to portfolio key |

---

## 2. Core Modules

### `core/models.py` — Model Registry

Protocol-based interface:

```python
class AlphaModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def get_feature_importance(self) -> pd.Series | None: ...
    def get_params(self) -> dict: ...
```

Six registered models:

| Name | Wrapper | Feature Tier | Key Hyperparameters |
|---|---|---|---|
| HGB | `HistGradientBoostingRegressor` | Tier 2 (~118) | max_depth, learning_rate, min_samples_leaf, l2_reg, max_iter |
| RF | `RandomForestRegressor` | Tier 2 (~118) | n_estimators, max_depth, max_features, min_samples_leaf |
| Lasso | `LassoCV` | Tier 1 (~52) | cv folds, max_iter |
| Ridge | `RidgeCV` | Tier 1 (~52) | alphas (auto-selected via CV) |
| ElasticNet | `ElasticNetCV` | Tier 1 (~52) | l1_ratio, cv folds, max_iter |
| FamaMacBeth | Custom | Tier 1 (~52) | None (monthly cross-sectional OLS, averaged slopes) |

Registry API:
- `get_model(name: str, params: dict) -> AlphaModel`
- `get_default_params(name: str) -> dict`
- `get_param_ranges(name: str) -> dict` (for dynamic UI slider generation)
- `get_feature_tier(name: str) -> int` (1 or 2)
- `list_models() -> list[str]`

### `core/backtest.py` — Walk-Forward Engine

```python
@dataclass
class BacktestResult:
    predictions: dict[str, pd.DataFrame]  # month -> DataFrame[permno, pred, y_raw, vol_12m_xs, sector]
    monthly_returns: pd.Series
    ic: pd.Series
    holdings: dict[str, pd.DataFrame]
    turnover: pd.Series
    feature_importance: pd.DataFrame | None
    train_dates: list[str]
    model_params: dict

def run_walk_forward(
    data: pd.DataFrame,
    model: AlphaModel,
    feature_cols: list[str],
    oos_start: str,
    retrain_freq: int,
    window_type: str,           # "expanding" or "rolling"
    rolling_window: int | None, # months for rolling window
    progress_callback=None,
) -> BacktestResult:
```

Key changes from current `engine.py`:
- Accepts `AlphaModel` instance instead of hardcoded model creation
- Supports rolling window (currently only expanding)
- Returns structured `BacktestResult` dataclass
- Feature importance extracted via `model.get_feature_importance()`

### `core/portfolio.py` — Portfolio Construction

Five construction methods:

| Method | Algorithm | Inputs |
|---|---|---|
| Equal-Weight | `w_i = 1/K` for selected stocks | predictions only |
| Score-Weighted | `w_i = pred_i / sum(pred)` | predictions only |
| Inverse-Vol | `w_i proportional to 1/sigma_i` (trailing 12m vol) | predictions + returns history |
| Full ERC | `scipy.optimize.minimize` for equal risk contribution. Trailing 12-month covariance. | predictions + covariance matrix |
| MVO | Maximize Sharpe, Ledoit-Wolf covariance shrinkage. Max 5% per position, long-only constraint. Trailing 12-month covariance. | predictions + covariance matrix |

Common interface:

```python
def construct_portfolio(
    predictions: pd.DataFrame,
    method: str,
    K: int,
    strategy_type: str,
    K_short: int,
    vol_tilt: float,
    returns_history: pd.DataFrame | None,
    **method_params,
) -> pd.DataFrame:  # permno, weight, side
```

`build_portfolio_series()` wraps this in the monthly loop with regime filter, producing the full backtest portfolio time series.

### `core/factor_models.py` — Regressions

All time-series regressions use HAC standard errors: `cov_type='HAC', cov_kwds={'maxlags': 5}`.

Functions:
- `run_regression(returns, factors, model_type) -> RegressionResult` — CAPM, FF3, or FF5. Returns coefficients, HAC SEs, t-stats, p-values, R-squared, adj-R-squared, Durbin-Watson, Jarque-Bera.
- `rolling_beta(returns, market, window) -> pd.DataFrame` — rolling OLS beta with confidence bands
- `bloomberg_shrink_beta(raw_beta) -> pd.Series` — `0.67 * beta + 0.33`
- `compute_vif(X) -> pd.DataFrame` — VIF per regressor
- `wald_test(model, R, q) -> WaldResult` — joint hypothesis test
- `hedge_ratio(portfolio_beta, market_beta) -> float` — SPY hedge weight

### `core/diagnostics.py` — Performance & Signal Diagnostics

- `compute_performance_metrics(returns) -> dict` — SR, ann return, ann vol, MDD, Calmar, total return
- `compute_ic_stats(ic_series) -> dict` — mean IC, IC t-stat, ICIR, hit rate
- `fundamental_law(ic_mean, K, rebal_freq, sr_target, tc_bps) -> dict` — IR bounds, implied BR, IC required
- `feature_importance_analysis(model, X, y) -> pd.DataFrame` — tree importance + permutation importance
- `feature_ic(X, y_realized) -> pd.Series` — univariate Spearman IC per feature
- `ks_test(X_train, X_current, threshold) -> pd.DataFrame` — per-feature KS statistic
- `alpha_decay(predictions, horizons) -> pd.Series` — IC at multiple forward horizons
- `signal_staleness(turnover_series, threshold, consecutive) -> pd.DataFrame` — flag stale periods

### `core/risk.py` — Risk Analysis

- `risk_contribution(weights, cov_matrix) -> pd.Series` — RC_i per position
- `factor_exposure(portfolio_returns, ff5_factors) -> pd.DataFrame` — FF5 betas on portfolio
- `rolling_factor_exposure(portfolio_returns, ff5_factors, window) -> pd.DataFrame` — rolling FF5 betas
- `turnover(holdings_t, holdings_prev) -> float` — monthly turnover (0.5 * sum(|delta_w|))
- `transaction_cost_drag(turnover_series, cost_bps, portfolio_vol) -> dict` — TC_ann, Cost_SR, net Sharpe

### `core/data_loader.py` — Data Loading

- `load_dataset(path) -> pd.DataFrame` — load and validate the alpha dataset
- `compute_market_monthly(df) -> pd.DataFrame` — extract SPY returns (from current engine.py)
- `load_ff5_factors(path) -> pd.DataFrame` — load Fama-French 5-factor CSV, align to dataset dates

---

## 3. Pages

### Page 1: Data Explorer

**Purpose**: Understand the dataset before modeling. Independent of backtest.

Components:
- Summary stats table: N stocks, N months, date range, feature count
- Feature distribution viewer: select feature -> cross-sectional histogram + box plot for chosen month
- Correlation heatmap: top-N features pairwise Spearman matrix (user picks N, default 20)
- Missing data summary: % NaN per feature, per month
- Universe size time series: stocks per month

Theory section: cross-sectional standardization, Spearman vs Pearson, economic meaning of features.

### Page 2: Factor Analysis

**Purpose**: CAPM and Fama-French regressions on individual stocks or portfolios. Independent of backtest.

Components:
- Stock selector: dropdown by permno with sector filter
- Regression panel: CAPM / FF3 / FF5 output table (coef, HAC SE, t-stat, p-value, R-squared, adj-R-squared, DW, JB). Alpha significance highlighted green (|t| > 1.96) / red.
- Rolling beta plot: configurable window (21-504, default 252). OLS vs Bloomberg-shrunk overlay.
- VIF table: for FF5, color-coded green (<5) / amber (5-10) / red (>10)
- Wald test panel: buttons for joint hypotheses (SMB=HML=0, RMW=CMA=0, all four=0)
- Hedge calculator: input portfolio beta -> SPY hedge weight and hedged return decomposition

Theory section: CAPM derivation, beta/alpha meaning, HAC standard errors rationale, FF5 factors, VIF, omitted variable bias.

Data: requires bundled `data/ff5_factors.csv`.

### Page 3: Alpha Model Lab

**Purpose**: Configure, train, and launch backtests. Writes results to session state.

Components:
- Model selector: dropdown for 6 models. Hyperparameter controls rendered dynamically from `models.get_param_ranges()`.
- Feature selection panel: checkboxes grouped by category (momentum, value, quality, size, vol, technical, analyst). Presets: "Tier default" (Tier 1 or 2 based on model), "Momentum only", "Value only", "Quality only", "Top-20 by univariate IC", "Kitchen sink" (all 118 Tier 2 features, even for linear models).
- Walk-forward config: training start, OOS start, retrain frequency (6/12/24 months), window type (expanding/rolling), top-K, portfolio type (long-only/long-short with K_short).
- Portfolio construction method: dropdown (EW, Score-Weight, Inverse-Vol, ERC, MVO).
- Vol tilt slider (0.0-0.50) and regime filter lookback (0-12 months).
- Run Backtest button: triggers walk-forward, shows progress bar, stores BacktestResult in session state.
- Pin Config button: saves result for comparison (max 4).

Theory section: walk-forward and look-ahead bias prevention, L1/L2 regularization, decision trees and gradient boosting intuition, Fama-MacBeth methodology, expanding vs rolling window.

### Page 4: Backtest Results & Diagnostics

**Purpose**: Comprehensive evaluation. Reads backtest_result from session state.

Components:
- Performance panel: cumulative return chart (log scale, SPY + pinned overlays), metric cards (SR, Ann Return, Ann Vol, MDD, Calmar), monthly returns heatmap, drawdown chart.
- IC dashboard: monthly IC bars, cumulative IC, rolling 12m IC with +/-1 sigma bands, summary stats (mean IC, t-stat, ICIR, hit rate) color-coded against thresholds.
- Fundamental Law panel: IR_realized, IC_mean, BR_nominal (K * 12), BR_implied = (SR/IC)^2, IR upper bound = IC * sqrt(BR) vs realized, IC_required calculator with user inputs for SR target and TC.
- Feature importance: top-20 bar chart (tree-based), permutation importance, univariate feature IC, importance drift across retraining windows.
- Strategy comparison table: side-by-side metrics for current + pinned configs.

Theory section: IC interpretation, Fundamental Law (IR = IC * sqrt(BR)), feature importance for tree vs linear models, alpha decay.

### Page 5: Portfolio Construction & Risk

**Purpose**: Compare construction methods and analyze risk. Reads backtest_result, writes portfolio_weights.

Components:
- Construction method selector: pick 1-2 methods for side-by-side comparison.
- Sector allocation chart: stacked area of sector weights over time.
- Holdings table: current month positions (side, permno, sector, predicted, actual).
- Risk decomposition: RC pie chart (top-10 + "other"), side-by-side RC for two methods.
- Factor exposure table: FF5 betas on portfolio with alert thresholds (|beta| > 0.10).
- Factor exposure time series: rolling FF5 exposures over the backtest.
- Turnover & cost panel: monthly turnover chart, average/annualized turnover, TC drag calculator (user inputs cost bps), gross vs net Sharpe.
- Method comparison table: SR, Ann Return, Vol, MDD, Turnover, TC Drag, Net SR for all methods.

Theory section: ERC motivation, naive diversification failure, Ledoit-Wolf shrinkage intuition, turnover as hidden cost, Cost_SR.

### Page 6: Monitoring & OOD Detection

**Purpose**: Production-style monitoring. Reads backtest_result + portfolio_weights + ff5_factors.

Components:
- KS test panel: per-feature D statistic table (feature, D, p-value, flag if D > 0.10), summary count, D heatmap over time (months x features).
- Alpha decay curve: IC(h) for h = 1..12, half-life annotation, rebalance frequency recommendation.
- Signal staleness: turnover with stale period highlighting (TO < 10%/month for 3+ months).
- Automated alert dashboard: traffic-light indicators for IC decay (rolling 6m IC < 0.02), IC collapse (monthly IC < -0.03), OOD shift (>20% features D > 0.10), factor drift (|beta_k| > 0.10), stale signal.
- Retraining recommendation: "No action" / "Consider retraining" / "Retrain immediately" based on alerts. Shows scheduled vs triggered retraining comparison.

Theory section: distribution shift and model failure, KS test mechanics, alpha decay and signal half-life, monitoring cadence, hybrid retraining policy.

### Page 7: Theory & Methods

**Purpose**: Standalone educational overview. No computation, purely content.

Sections:
- The Alpha Pipeline: visual flow diagram (data -> features -> model -> predictions -> portfolio -> monitoring)
- Factor Investing Foundations: CAPM, Fama-French, why factors earn premiums
- Machine Learning for Alpha: cross-sectional prediction, regularization, trees vs linear, walk-forward
- Portfolio Construction: predictions to weights, risk budgeting, optimization
- Performance Evaluation: Sharpe, IC, Fundamental Law, benchmarks
- Model Monitoring: distribution shift, decay, retraining
- Glossary: key terms with definitions

Uses formatted markdown, LaTeX via `st.latex()`, and small illustrative Plotly charts.

---

## 4. Shared Components

### `components/charts.py`

Reusable Plotly chart builders. All apply the Bloomberg dark template.

Style constants:

```python
STYLE = {
    "bg": "#0A1628",
    "positive": "#00D26A",
    "negative": "#FF4444",
    "warning": "#FFB800",
    "text": "#FFFFFF",
    "muted": "#8899AA",
    "template": "plotly_dark",
}
```

Chart builders:
- `cumulative_wealth_chart(returns_dict, start_val, cash_flow)`
- `drawdown_chart(returns_dict)`
- `monthly_heatmap(returns_series)`
- `rolling_metric_chart(series, window, name)` — rolling line with +/-1 sigma bands
- `bar_chart(series, name, color_pos, color_neg)` — signed bar chart
- `correlation_heatmap(corr_matrix)`
- `sector_allocation_chart(holdings_dict, strategy_type)`
- `traffic_light_dashboard(alerts_dict)`
- `risk_pie_chart(risk_contributions)`

### `components/metrics.py`

- `metric_row(metrics: list[dict])` — row of st.metric() cards
- `comparison_table(configs: list[dict])` — pinned config comparison with highlighting
- `regression_table(result)` — statsmodels-style regression output
- `vif_table(vif_values)` — color-coded VIF display

### `components/theory.py`

```python
def theory_section(title: str, content_key: str):
```

Renders `st.expander` with educational content. Content stored in `theory_content.py` as a dict mapping content_key to markdown strings with LaTeX.

### `components/theory_content.py`

Single data file containing all educational text, organized by topic key. Keeps theory out of page logic and centralizes content editing.

---

## 5. Data Flow & Session State

### Session State Schema

```python
st.session_state = {
    # Loaded once by app.py
    "df": pd.DataFrame,              # full dataset
    "market_monthly": pd.DataFrame,  # SPY returns by month
    "ff5_factors": pd.DataFrame,     # Fama-French 5-factor data

    # Written by Page 3 (Alpha Model Lab)
    "backtest_result": BacktestResult | None,
    "backtest_params": dict | None,
    "pinned_configs": list[dict],     # max 4

    # Written by Page 5 (Portfolio Construction)
    "portfolio_weights": dict | None,
}
```

### Page Dependency Graph

```
Page 1 (Data Explorer)      <- df only, independent
Page 2 (Factor Analysis)    <- df + ff5_factors, independent
Page 3 (Alpha Model Lab)    <- df, WRITES backtest_result + backtest_params
Page 4 (Backtest Results)   <- READS backtest_result + pinned_configs
Page 5 (Portfolio Constr.)  <- READS backtest_result, WRITES portfolio_weights
Page 6 (Monitoring)         <- READS backtest_result + portfolio_weights + ff5_factors
Page 7 (Theory & Methods)   <- nothing, static content
```

Pages 4, 5, and 6 display a "Run a backtest first on the Alpha Model Lab page" message when backtest_result is None.

### Caching Strategy

- **Tier 1 (disk, expensive)**: Walk-forward predictions. Keyed by model config + features + window type.
- **Tier 2 (disk, cheap)**: Portfolio results. Keyed by prediction key + construction method + K + vol_tilt + regime + strategy.
- **Tier 3 (session, fast)**: Factor regressions and monitoring diagnostics. Stored in session_state, not persisted to disk.
- **Page 2**: Uses `@st.cache_data` for factor regressions (independent of backtest cache).

---

## 6. External Data

### Fama-French 5-Factor CSV

File: `data/ff5_factors.csv`

Source: Ken French Data Library. Monthly Mkt-RF, SMB, HML, RMW, CMA, RF.

Used by: Pages 2, 5, 6.

The current dataset already contains `Mkt_RF` and `rf_ff` columns for market returns, but not the other four factors. The bundled CSV provides the complete set.

---

## 7. Dependencies

Current `requirements.txt` plus:

```
statsmodels>=0.14      # OLS with HAC, Wald tests, Durbin-Watson, Jarque-Bera
```

Already available via existing dependencies:
- `scipy` — optimize.minimize (ERC, MVO), stats.ks_2samp, stats.spearmanr
- `sklearn.covariance.LedoitWolf` — covariance shrinkage for MVO

---

## 8. Design Principles

- **No look-ahead bias**: the walk-forward engine trains only on data strictly before the prediction month. This is non-negotiable.
- **HAC standard errors**: all time-series regressions use `cov_type='HAC', cov_kwds={'maxlags': 5}`. Never classical SEs for financial data.
- **Spearman IC**: all information coefficient computations use Spearman rank correlation, not Pearson.
- **Modularity**: core/ modules have zero Streamlit imports. Pages import from core/ and components/. Components are reusable across pages.
- **Registry extensibility**: adding a new model requires only writing a class and registering it. No page changes needed.
- **Separation of expensive and cheap**: predictions (expensive) and portfolio construction (cheap) are cached independently so portfolio parameter changes are instant.
