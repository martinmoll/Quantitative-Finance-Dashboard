# Alpha Strategy Dashboard — Design Spec

## Overview

A Streamlit-based interactive dashboard for backtesting alpha strategies with tunable parameters. Wraps the existing walk-forward ML pipeline (from `MartinMoellenhus_Assignment2.ipynb`) into an interactive tool for rapid experimentation.

**Primary use case:** Personal experimentation — utility over polish.

---

## Architecture

### File Structure

```
dashboard/
  app.py              # Streamlit entry point — layout, controls, chart rendering
  engine.py           # Walk-forward backtest engine (extracted from notebook)
  features.py         # Feature engineering functions (extracted from notebook)
  cache_manager.py    # Disk-based caching for backtest results
```

### Data Flow

1. On startup, `app.py` loads `alpha_dataset_v2.csv` once via `@st.cache_data`
2. User adjusts params in the top control bar and clicks "Run Backtest"
3. `cache_manager` checks if this exact param combo has been computed before (hash of all params → cache key)
4. If cached → return instantly. If not → run walk-forward engine, store results to disk (`dashboard/.cache/`), return
5. Results include: monthly returns, per-month stock holdings (permno + sector), per-month predictions, IC, turnover
6. Charts render from the cached result object

### Two-Tier Caching

- **Prediction cache** (keyed on model type + hyperparams + feature set): Stores per-stock monthly predictions for every OOS month. This is the expensive part (~30-60s). Persisted to disk as pickle files in `dashboard/.cache/predictions/`.
- **Portfolio cache** (keyed on prediction cache key + K + vol_tilt + regime_lookback): Rebuilds portfolio selection from cached predictions. Instant (<1s). Persisted to `dashboard/.cache/portfolios/`.

Changing K / vol_tilt / regime_lookback is instant once the model has been run. Only model hyperparam changes trigger a full walk-forward rerun.

Cache key = deterministic hash of all relevant parameters (model type, hyperparams, feature tier for predictions; prediction hash + portfolio params for portfolios).

### Data Source

- `Data/alpha_dataset_v2.csv` — 257,800 rows, 229 columns, 407 months, 1,216 stocks
- Loaded once on startup, held in memory via `@st.cache_data`
- SPY benchmark reconstructed from Fama-French `Mkt_RF + rf_ff` (same as notebook)

---

## Controls

### Top Control Bar (always visible, horizontal)

| Control | Type | Range | Default |
|---|---|---|---|
| Model | Dropdown | HGB, Lasso | HGB |
| K (num stocks) | Slider | 5–50, step 5 | 10 |
| Vol tilt | Slider | 0.00–0.50, step 0.01 | 0.05 |
| Regime lookback | Slider | 0–12 months (0 = disabled) | 6 |
| **Run Backtest** | Button | — | — |
| **Pin Config** | Button | — | — |

### Expandable "Model Hyperparams" Section (collapsed by default)

**HGB params:**

| Control | Type | Range | Default |
|---|---|---|---|
| max_depth | Slider | 1–6 | 2 |
| learning_rate | Slider | 0.01–0.20, step 0.01 | 0.05 |
| min_samples_leaf | Slider | 100–2000, step 100 | 500 |
| l2_regularization | Slider | 0.0–1.0, step 0.05 | 0.1 |
| max_iter | Slider | 100–1000, step 50 | 500 |

**Lasso params:**

| Control | Type | Range | Default |
|---|---|---|---|
| cv folds | Slider | 3–10 | 5 |
| max_iter | Slider | 1000–10000, step 1000 | 5000 |

**Shared:**

| Control | Type | Range | Default |
|---|---|---|---|
| Retrain frequency | Slider | 3–24 months, step 3 | 12 |

### Pin/Compare System

- "Pin" saves current config + results to `st.session_state` with auto-label (e.g., "HGB K=10 vt=0.05")
- Pinned configs appear as dismissable chips below the control bar
- Each pinned config gets a unique color for chart overlays
- Max 4 pinned configs (keeps charts readable)

---

## Charts & Layout

Single scrollable page, 2-column grid. All charts are interactive Plotly.

### Row 1: KPI Cards (full width, 4 columns)

| Card | Color Logic |
|---|---|
| Sharpe Ratio | Green if > 1.0, yellow 0.5–1.0, red < 0.5 |
| Annualized Return | Green if positive |
| Annualized Volatility | Neutral |
| Max Drawdown | Red if > 40% (disqualifying), yellow if > 30% |

### Row 2: Core Performance (2 columns)

- **Left: Cumulative Wealth** — Equity curve vs SPY benchmark. Pinned configs overlaid with distinct colors. Log scale optional.
- **Right: Drawdown Chart** — Underwater plot showing drawdown % over time.

### Row 3: Portfolio Composition (2 columns)

- **Left: Sector Allocation Over Time** — Stacked area chart of sector weights per month.
- **Right: Current Holdings Table** — Sortable table of stocks in most recent OOS month: permno, sector, predicted return, actual return.

### Row 4: Risk & Signal (2 columns)

- **Left: Rolling 12-Month Sharpe Ratio** — Line chart showing SR stability. Pinned configs overlaid.
- **Right: Information Coefficient** — Monthly IC bars with 12-month rolling mean overlay.

### Row 5: Deeper Diagnostics (2 columns)

- **Left: Monthly Returns Heatmap** — Year x month grid, colored red (negative) to green (positive).
- **Right: Turnover** — Monthly turnover bars with mean line.

### Row 6: Comparison Table (full width, conditional)

Only visible when 1+ configs are pinned. Side-by-side stats: SR, Ann Return, Ann Vol, MDD, Total Return, Mean IC, Mean Turnover for each config.

---

## Engine Logic (extracted from notebook)

### Feature Engineering (`features.py`)

Two functions extracted directly from the notebook:
- `build_features_linear(df_slice)` → Tier 1 (~52 features) for Lasso
- `build_features_ensemble(df_slice)` → Tier 2 (~118 features) for HGB

Plus shared helpers: `_get_existing()`, `_build_engineered_features()`.

### Walk-Forward Engine (`engine.py`)

`run_model()` extracted from notebook with one change: instead of returning only monthly return series, it also returns per-month prediction DataFrames and stock holdings. This enables:
- Portfolio-level param changes without rerunning the model (reapply K/vol_tilt/regime to cached predictions)
- Holdings/sector charts (need to know which stocks were selected)

Key function signatures:
- `run_predictions(df, feature_builder, estimator, retrain_every)` → dict of `{month: DataFrame with permno, pred, sector, y_raw}`
- `build_portfolio(predictions, K, vol_tilt, regime_lookback, market_monthly)` → dict with monthly returns, holdings, IC, turnover

### SPY Benchmark

Reconstructed from `Mkt_RF + rf_ff` per the notebook. Computed once on data load.

---

## Dependencies

- `streamlit` — dashboard framework
- `plotly` — interactive charts
- `pandas`, `numpy`, `scipy` — data manipulation (already in notebook)
- `scikit-learn` — models (already in notebook)
- `hashlib` — cache key generation (stdlib)
- `pickle` — cache serialization (stdlib)

---

## Out of Scope (v1)

- Custom CSS / Bloomberg dark theme (can add later)
- Feature set toggling (always uses the model's default tier)
- Long-short mode (long-only only for v1)
- Authentication / multi-user
- Deployment (local only)
