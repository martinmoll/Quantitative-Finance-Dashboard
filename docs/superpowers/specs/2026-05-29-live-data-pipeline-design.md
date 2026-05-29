# Live Data Pipeline Design

**Date:** 2026-05-29
**Status:** Approved

## Overview

Add a live data pipeline to the Alpha Strategy Dashboard that fetches recent market data from free sources (yfinance, FRED, Ken French library), computes ~100 features matching the existing dataset schema, and appends new monthly observations to the historical dataset. The pipeline is triggered manually from a new dashboard page.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Stock universe | Current S&P 500 (~500 stocks) | Fast downloads, matches best configs |
| RED features (14) | Fill with 0.0 | Keeps column structure identical to historical data; model sees them as neutral |
| Refresh trigger | Manual button in dashboard | User controls when to fetch; avoids slow auto-startup |
| Storage format | Parquet primary + CSV export | Parquet for speed; CSV for compatibility |
| Pipeline scope | Fetch + feature computation only | Model re-run handled via existing Alpha Model Lab page |
| Architecture | Staged pipeline with intermediate caching | Resilient to partial failures; separates network I/O from computation |

## Data Sources

| Source | Library | Data | API Key |
|--------|---------|------|---------|
| Yahoo Finance | `yfinance>=0.2.28` | Daily OHLCV, quarterly financials, sector/industry, market cap | No |
| Ken French Library | `pandas-datareader` | FF5 factors + Momentum + risk-free rate | No |
| FRED | `fredapi>=0.5` | VIX (VIXCLS), yield curve (T10Y2Y), credit spread (BAA10Y), EPU (USEPUINDXM), financial stress (STLFSI4) | Free key required |

## File Structure

```
pipeline/
  __init__.py
  config.py                # Universe size, API keys, paths, rate-limit constants
  universe.py              # Fetch current S&P 500 ticker list
  fetchers/
    __init__.py
    prices.py              # yfinance OHLCV batched download
    fundamentals.py        # yfinance quarterly financials per-ticker
    factors.py             # FF5 + Momentum via pandas-datareader
    macro.py               # FRED series download
  features/
    __init__.py
    price_features.py      # Momentum, volatility, beta, technicals, liquidity, size
    fundamental_features.py # Value, quality, growth ratios
    peer_features.py       # Sector/size/beta group relative signals
    macro_features.py      # Macro uncertainty proxies
  assembler.py             # Merge features, standardize, fill RED cols, export
  cache/                   # Gitignored local cache
    prices/                # Parquet: one file per fetch batch
    fundamentals/          # Parquet: one file per ticker
    factors/               # Single parquet: ff5 + momentum
    macro/                 # Single parquet: all FRED series

dashboard/
  pages/
    8_Data_Pipeline.py     # New page with Refresh Data button + progress UI
```

## Data Flow

### Stage 1: Fetch (network I/O, cached to disk)

1. **Universe** (`universe.py`): Fetch current S&P 500 constituent ticker list. Source: Wikipedia S&P 500 page via pandas `read_html`, or a hardcoded/cached list with periodic refresh.

2. **Prices** (`fetchers/prices.py`):
   - `yf.download(tickers, period="3y", interval="1d")` in batches of 50 tickers
   - 2-second pause between batches
   - Save to `cache/prices/` as parquet
   - Columns: Open, High, Low, Close, Volume per ticker (auto-adjusted)

3. **Fundamentals** (`fetchers/fundamentals.py`):
   - Loop over each ticker: `yf.Ticker(t).get_income_stmt(freq='quarterly')`, `.get_balance_sheet(freq='quarterly')`, `.get_cashflow(freq='quarterly')`
   - 0.5s delay between tickers, retry up to 3x on failure
   - Save per-ticker to `cache/fundamentals/{ticker}.parquet`
   - Extract: TotalRevenue, GrossProfit, OperatingIncome, NetIncome, TotalAssets, TotalEquityGrossMinorityInterest, TotalDebt, CashAndCashEquivalents, OperatingCashFlow, DepreciationAndAmortization, DilutedEPS

4. **Factors** (`fetchers/factors.py`):
   - FF5 + Momentum from Ken French library via `pandas_datareader`
   - Save to `cache/factors/ff5_mom.parquet`

5. **Macro** (`fetchers/macro.py`):
   - FRED series: VIXCLS, T10Y2Y, BAA10Y, USEPUINDXM, STLFSI4
   - Single fredapi call per series
   - Save to `cache/macro/fred_data.parquet`

### Stage 2: Feature Computation (pure local computation)

**price_features.py** — reads `cache/prices/`, computes per-stock monthly features:

| Feature | Computation |
|---------|-------------|
| `ret_1` | (close_t / close_{t-1}) - 1 (month-end to month-end) |
| `ret_2_12` | (close_{t-1} / close_{t-12}) - 1 (skip most recent month) |
| `ret_2_6` | (close_{t-1} / close_{t-6}) - 1 |
| `ret_13_36` | (close_{t-12} / close_{t-36}) - 1 |
| `vol_12m` | std(daily_returns) * sqrt(252) over trailing 252 days |
| `max_ret_12m` | max(daily_return) over trailing 252 days |
| `beta` | cov(stock_daily, market_daily) / var(market_daily), trailing 252 days |
| `ivol` | std(residuals from CAPM regression), trailing 252 days |
| `turnover` | mean(daily_volume) / shares_outstanding, trailing month |
| `log_me` | log(price * shares_outstanding) at month-end |
| `prc_52w_high` | close / max(close over 252 days) |
| `age` | 0.0 (filled — not reliably available from yfinance) |
| `rsi_14` | Standard RSI formula on daily close, sampled at month-end |
| `macd_hist` | EMA(12) - EMA(26) - signal(9), at month-end |
| `bb_position` | (close - SMA20) / (2 * rolling_std20), at month-end |
| `prc_ma6` | close / SMA(126 days) |
| `prc_ma12` | close / SMA(252 days) |
| `prc_ma24` | close / SMA(504 days) |
| `roc_3` | (close / close_63_days_ago) - 1 |
| `roc_6` | (close / close_126_days_ago) - 1 |
| `vol_ratio` | current_month_avg_volume / trailing_3month_avg_volume |
| `skew_12m` | skew(daily_returns) over 252 days |
| `kurt_12m` | kurtosis(daily_returns) over 252 days |
| `illiq_12m` | mean(|daily_ret| / dollar_volume) over 252 days |

**fundamental_features.py** — reads `cache/fundamentals/`, computes per-stock:

| Feature | Computation |
|---------|-------------|
| `bm` | TotalEquity / market_cap |
| `ep` | NetIncome(TTM) / market_cap |
| `cfp` | OperatingCashFlow(TTM) / market_cap |
| `sp` | TotalRevenue(TTM) / market_cap |
| `ag` | (TotalAssets_q / TotalAssets_q4) - 1 |
| `gpa` | GrossProfit(TTM) / TotalAssets |
| `roe` | NetIncome(TTM) / TotalEquity |
| `roa` | NetIncome(TTM) / TotalAssets |
| `acc` | (NetIncome - OperatingCashFlow) / TotalAssets |
| `nsi` | 0.0 (filled — share count history unreliable from yfinance) |
| `lev` | TotalDebt / TotalEquity |
| `cash_at` | CashAndCashEquivalents / TotalAssets |
| `sgr` | (Revenue_q / Revenue_q4) - 1 |
| `ato` | TotalRevenue(TTM) / TotalAssets |
| `dp_ratio` | annual_dividends / DilutedEPS |
| `sue_q` | Seasonal random walk: (EPS_q - EPS_q4) / std(EPS_q - EPS_q4) |
| `rev_growth_qq` | (Revenue_q / Revenue_q1) - 1 |
| `earn_growth_yoy` | (EPS_q / EPS_q4) - 1 |
| `gm_q` | GrossProfit / Revenue (latest quarter) |
| `gm_chg` | gm_q - gm_q1 |
| `op_margin_q` | OperatingIncome / Revenue (latest quarter) |
| `op_margin_chg` | op_margin_q - op_margin_q1 |

Additional quality/growth features from quarterly statements follow the same pattern: ratios of statement items, YoY or QoQ changes.

**peer_features.py** — reads price features + sector classification:

| Feature | Computation |
|---------|-------------|
| `sector_ret_avg` | mean(ret_1) by sector for the month |
| `sector_ret_dispersion` | std(ret_1) by sector |
| `sector_mom_lag1` | sector_ret_avg from previous month |
| `peer_ret_1` | Same as sector_ret_avg (sector-level peer group) |
| `size_peer_ret` | mean(ret_1) by size quintile |
| `size_grp_mom` | size_peer_ret from previous month |
| `size_grp_disp` | std(ret_1) by size quintile |
| `beta_grp_ret` | mean(ret_1) by beta quintile |
| `val_grp_mom` | mean(ret_1) by bm quintile, lagged |
| `leader_ret_lag1` | ret_1 of largest stock in sector, lagged |
| `ind_mom` | mean(ret_1) by industry (from yfinance info) |
| `ind_dispersion` | std(ret_1) by industry |
| `ret_vs_sector` | ret_1 - sector_ret_avg |
| `ret_vs_ind` | ret_1 - ind_mom |
| `bm_vs_sector` | bm - sector mean(bm) |
| `bm_vs_size` | bm - size quintile mean(bm) |

**macro_features.py** — reads `cache/macro/`:

| Feature | Computation |
|---------|-------------|
| `macro_unc_1m` | VIX monthly average, z-scored against trailing 60-month window |
| `macro_unc_12m` | VIX 12-month rolling average, z-scored |
| `fin_unc_1m` | STLFSI4 monthly value (already a z-score by construction) |
| `fin_unc_12m` | STLFSI4 12-month rolling average |

### Stage 3: Assembly (`assembler.py`)

1. Merge price_features, fundamental_features, peer_features, macro_features on `(permno, ym)`
2. Add metadata columns: `permno`, `ym`, `date`, `sector`, `in_sp500=1`, `exchcd`, `siccd`, `me`, `prc_abs`
3. Fill all RED feature columns with 0.0:
   - Options: `iv_atm_30d`, `iv_atm_91d`, `iv_skew`, `pc_vol_ratio`, `pc_oi_ratio`, `vrp`
   - Analyst: `sue` (monthly aggregate), `beat`, `n_analysts`, `revision`, `dispersion`, `revision_ratio`, `rev_surp`
   - Peer-analyst: `peer_revision`, `peer_sue`, `ind_sue`, `ind_crowding`
4. Cross-sectional standardization: for each raw feature, compute `feature_xs = (feature - mean_by_ym) / std_by_ym`
5. Compute interaction/pre-built features: `mom_x_size_xs`, `val_x_prof_xs`, `mom_x_vol_xs`, `mom_x_unc_xs`, `val_x_finunc_xs`, `beta_x_disp_xs`, `mom_accel_xs`, `delta_vol_xs`, `delta_bm_xs`, `ret_streak_xs`, `mom_of_mom_xs`
6. Compute target columns:
   - `y_raw`: forward 1-month return (NaN for latest month)
   - `y_xs`: cross-sectional z-score of y_raw (NaN for latest month)
   - `ret_excess`: ret_1 - rf_ff
   - `ret_adj`: same as ret_excess (or adjusted for specific factors)
7. Merge FF5 factors + `rf_ff` + `spy_ret` by month
8. Ensure all 229 columns match existing dataset schema exactly
9. Load existing parquet, append new month(s), deduplicate on `(permno, ym)`, save
10. Export CSV copy

## Dashboard Page: 8_Data_Pipeline

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Data Pipeline                                          │
│                                                         │
│  Current Dataset                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ 1,216    │  │ 407      │  │ 2003-01 → 2026-04    │  │
│  │ Stocks   │  │ Months   │  │ Date Range           │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
│                                                         │
│  Last updated: 2026-05-28 14:32                         │
│  Next available month: 2026-05                          │
│                                                         │
│  ┌────────────────────┐                                 │
│  │  Refresh Data      │                                 │
│  └────────────────────┘                                 │
│                                                         │
│  Pipeline Progress                                      │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░  75%                            │
│  ✓ Fetched S&P 500 universe (503 tickers)               │
│  ✓ Downloaded prices (503/503)                          │
│  ✓ Downloaded fundamentals (498/503, 5 failed)          │
│  ● Computing features...                                │
│  ○ Assembling dataset                                   │
│                                                         │
│  ─── Fetch Log ───                                      │
│  Failed tickers: BRK.B (no data), BF.B (delisted), ... │
└─────────────────────────────────────────────────────────┘
```

### Behavior

- Button disabled while pipeline is running
- Each stage updates the progress bar and log in real-time via `st.status()` or `st.progress()`
- On completion: success banner with summary stats, session state `df` is reloaded
- On partial failure: warning banner listing failed tickers, pipeline continues with available data
- FRED API key: if not set, show a text input field with link to registration

## Rate Limiting

| Source | Strategy |
|--------|----------|
| yfinance prices | Batches of 50 tickers, 2s pause between batches |
| yfinance fundamentals | Per-ticker, 0.5s delay, 3 retries on failure |
| FRED | 120 req/min limit, we make ~5 calls total |
| Ken French | Single download, no limit |

## Error Handling

- Network failures: retry with exponential backoff (3 attempts max)
- Ticker-level failures: log and skip, fill that ticker's row with NaN
- Partial dataset: if <50% of tickers succeed, abort and show error
- Schema mismatch: validate output columns match existing dataset before appending
- Duplicate months: if fetched month already exists in dataset, overwrite (re-fetch)

## New Dependencies

```
yfinance>=0.2.28
fredapi>=0.5
```

Already present: `pandas-datareader`, `pyarrow`, `pandas`, `numpy`, `scipy`.

## Migration Path

1. One-time: convert existing `alpha_dataset_v2.csv` to `alpha_dataset_v2.parquet`
2. `data_loader.py` updated to read parquet first, fall back to CSV
3. Pipeline appends to parquet; CSV export happens alongside
4. Existing dashboard pages work unchanged (same DataFrame schema)

## Out of Scope

- Auto-prediction on new data (handled by Alpha Model Lab)
- Real-time / intraday data
- Paid data sources
- Historical backfill (the existing CSV covers 2003-2023)
- Options and analyst features from live sources
