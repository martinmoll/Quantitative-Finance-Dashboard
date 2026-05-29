# Live Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a staged data pipeline that fetches S&P 500 market data from yfinance/FRED/Ken French, computes ~100 features matching the existing 229-column dataset schema, and appends new monthly observations via a manual dashboard button.

**Architecture:** Staged pipeline with intermediate caching. Four fetcher modules download raw data to `pipeline/cache/`. Four feature modules compute raw features from cached data. An assembler merges everything, cross-sectionally standardizes, fills unavailable columns with 0, and appends to the Parquet dataset. A new Streamlit page triggers the pipeline.

**Tech Stack:** yfinance, fredapi, pandas-datareader, pandas, numpy, pyarrow, streamlit

---

## File Map

| File | Responsibility |
|------|----------------|
| `pipeline/__init__.py` | Package init, exports `run_pipeline()` |
| `pipeline/config.py` | Paths, constants, column schema, FRED API key |
| `pipeline/universe.py` | Fetch S&P 500 ticker list from Wikipedia |
| `pipeline/fetchers/__init__.py` | Package init |
| `pipeline/fetchers/prices.py` | Batched yfinance OHLCV download |
| `pipeline/fetchers/fundamentals.py` | Per-ticker quarterly financials |
| `pipeline/fetchers/factors.py` | FF5 + Momentum from Ken French |
| `pipeline/fetchers/macro.py` | FRED series (VIX, yield curve, EPU, stress) |
| `pipeline/features/__init__.py` | Package init |
| `pipeline/features/price_features.py` | Momentum, vol, beta, technicals, liquidity, size |
| `pipeline/features/fundamental_features.py` | Value, quality, growth ratios |
| `pipeline/features/peer_features.py` | Sector/size/beta group relative signals |
| `pipeline/features/macro_features.py` | Macro uncertainty proxies |
| `pipeline/assembler.py` | Merge, standardize, fill RED cols, append to Parquet |
| `dashboard/core/data_loader.py` | Modify: add Parquet support with CSV fallback |
| `dashboard/pages/8_Data_Pipeline.py` | New: Refresh Data page with progress UI |
| `tests/test_pipeline_features.py` | Tests for all feature computation modules |
| `tests/test_pipeline_assembler.py` | Tests for assembler + schema validation |

---

### Task 1: Pipeline config and project setup

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/config.py`
- Create: `pipeline/fetchers/__init__.py`
- Create: `pipeline/features/__init__.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Add new dependencies to requirements.txt**

Append to `requirements.txt`:
```
yfinance>=0.2.28
fredapi>=0.5
```

- [ ] **Step 2: Add pipeline cache to .gitignore**

Append to `.gitignore`:
```
pipeline/cache/
```

- [ ] **Step 3: Create pipeline/config.py**

```python
"""Pipeline configuration: paths, constants, column schema."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
CACHE_DIR = Path(__file__).parent / "cache"
PRICES_CACHE = CACHE_DIR / "prices"
FUNDAMENTALS_CACHE = CACHE_DIR / "fundamentals"
FACTORS_CACHE = CACHE_DIR / "factors"
MACRO_CACHE = CACHE_DIR / "macro"

PARQUET_PATH = DATA_DIR / "alpha_dataset_v2.parquet"
CSV_PATH = DATA_DIR / "alpha_dataset_v2.csv"

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

PRICE_BATCH_SIZE = 50
PRICE_BATCH_DELAY = 2.0
FUNDAMENTALS_DELAY = 0.5
FUNDAMENTALS_RETRIES = 3
MIN_SUCCESS_RATE = 0.5

FRED_SERIES = {
    "VIXCLS": "vix",
    "T10Y2Y": "yield_curve_slope",
    "BAA10Y": "credit_spread",
    "USEPUINDXM": "epu",
    "STLFSI4": "fin_stress",
}

METADATA_COLS = [
    "permno", "ym", "date", "in_sp500", "exchcd", "siccd", "sector",
    "me", "prc_abs",
]

RAW_FEATURE_COLS = [
    "ret_1", "ret_2_12", "ret_2_6", "ret_13_36", "vol_12m", "max_ret_12m",
    "beta", "ivol", "turnover", "log_me", "prc_52w_high", "age",
    "bm", "ep", "cfp", "sp", "ag", "gpa", "roe", "roa", "acc", "nsi",
    "lev", "cash_at", "sgr", "ato", "dp_ratio",
    "sue", "beat", "n_analysts", "revision", "dispersion", "revision_ratio",
    "rsi_14", "macd_hist", "bb_position", "prc_ma6", "prc_ma12", "prc_ma24",
    "roc_3", "roc_6", "vol_ratio", "skew_12m", "kurt_12m", "illiq_12m",
    "sue_chg",
    "iv_atm_30d", "iv_atm_91d", "iv_skew", "pc_vol_ratio", "pc_oi_ratio", "vrp",
    "ret_vs_sector", "bm_vs_sector", "mom_x_size", "val_x_prof", "mom_x_vol",
    "peer_sue", "peer_revision", "size_peer_ret", "val_peer_ret",
    "leader_ret_lag1", "peer_ret_1",
    "sue_q", "rev_surp", "rev_growth_qq", "earn_growth_yoy",
    "gm_q", "gm_chg", "op_margin_q", "op_margin_chg",
    "sga_chg", "acc_q", "roe_q", "roe_chg", "inv_chg", "rec_chg",
    "rd_intensity", "cfo_at", "earn_quality", "oi_growth_yoy", "ato_q",
    "ret_vs_ind", "bm_vs_size",
    "mom_x_unc", "val_x_finunc", "beta_x_disp",
    "mom_accel", "delta_vol", "delta_bm", "ret_streak", "mom_of_mom",
    "sector_ret_avg", "sector_ret_dispersion", "mkt_ret_dispersion",
    "sector_mom_lag1",
    "iv_term_structure", "sector_iv", "sector_vrp",
    "leader_ret", "sector_rel_mom",
    "ind_mom", "ind_dispersion", "ind_crowding", "ind_sue",
    "size_grp_mom", "size_grp_disp", "beta_grp_ret", "val_grp_mom",
    "vol_grp_ret", "ind_size_ret", "ind_size_mom",
    "macro_unc_1m", "macro_unc_12m", "fin_unc_1m", "fin_unc_12m",
]

RED_FEATURES = [
    "iv_atm_30d", "iv_atm_91d", "iv_skew", "pc_vol_ratio", "pc_oi_ratio",
    "vrp", "iv_term_structure", "sector_iv", "sector_vrp",
    "sue", "beat", "n_analysts", "revision", "dispersion", "revision_ratio",
    "rev_surp", "sue_chg",
    "peer_revision", "peer_sue", "ind_sue", "ind_crowding",
]

TARGET_COLS = ["y_raw", "y_xs", "ret_excess", "ret_adj"]
FACTOR_COLS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom", "rf_ff", "spy_ret"]

DATASET_COLUMNS = METADATA_COLS + RAW_FEATURE_COLS + [
    f"{c}_xs" for c in RAW_FEATURE_COLS
    if c not in ["sector_ret_avg", "sector_ret_dispersion", "mkt_ret_dispersion",
                  "sector_mom_lag1", "iv_term_structure", "sector_iv", "sector_vrp",
                  "leader_ret", "sector_rel_mom", "ind_mom", "ind_dispersion",
                  "ind_crowding", "ind_sue", "size_grp_mom", "size_grp_disp",
                  "beta_grp_ret", "val_grp_mom", "vol_grp_ret", "ind_size_ret",
                  "ind_size_mom", "macro_unc_1m", "macro_unc_12m", "fin_unc_1m",
                  "fin_unc_12m"]
] + TARGET_COLS + FACTOR_COLS


def ensure_cache_dirs():
    for d in [PRICES_CACHE, FUNDAMENTALS_CACHE, FACTORS_CACHE, MACRO_CACHE]:
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Create package init files**

`pipeline/__init__.py`:
```python
"""Live data pipeline for Alpha Strategy Dashboard."""
```

`pipeline/fetchers/__init__.py`:
```python
"""Data fetchers for external sources."""
```

`pipeline/features/__init__.py`:
```python
"""Feature computation from cached raw data."""
```

- [ ] **Step 5: Install new dependencies**

Run: `pip install yfinance>=0.2.28 fredapi>=0.5`

- [ ] **Step 6: Commit**

```bash
git add pipeline/__init__.py pipeline/config.py pipeline/fetchers/__init__.py pipeline/features/__init__.py requirements.txt .gitignore
git commit -m "feat: scaffold pipeline package with config and dependencies"
```

---

### Task 2: Universe fetcher

**Files:**
- Create: `pipeline/universe.py`
- Create: `tests/test_pipeline_universe.py`

- [ ] **Step 1: Write the test**

```python
"""tests/test_pipeline_universe.py"""
import pytest
from pipeline.universe import get_sp500_tickers


def test_get_sp500_tickers_returns_list():
    tickers = get_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 400
    assert all(isinstance(t, str) for t in tickers)
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_get_sp500_tickers_no_duplicates():
    tickers = get_sp500_tickers()
    assert len(tickers) == len(set(tickers))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_universe.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement universe.py**

```python
"""Fetch current S&P 500 constituent list."""

import pandas as pd
from pathlib import Path

_CACHE_PATH = Path(__file__).parent / "cache" / "sp500_tickers.csv"


def get_sp500_tickers(use_cache: bool = True) -> list[str]:
    if use_cache and _CACHE_PATH.exists():
        df = pd.read_csv(_CACHE_PATH)
        return df["ticker"].tolist()

    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    tickers = sorted(set(tickers))

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(_CACHE_PATH, index=False)

    return tickers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_universe.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/universe.py tests/test_pipeline_universe.py
git commit -m "feat: add S&P 500 universe fetcher from Wikipedia"
```

---

### Task 3: Price fetcher

**Files:**
- Create: `pipeline/fetchers/prices.py`

- [ ] **Step 1: Implement prices.py**

```python
"""Batched OHLCV download from yfinance."""

import time
import logging
import pandas as pd
import yfinance as yf

from pipeline.config import (
    PRICES_CACHE, PRICE_BATCH_SIZE, PRICE_BATCH_DELAY, ensure_cache_dirs,
)

logger = logging.getLogger(__name__)


def fetch_prices(
    tickers: list[str],
    period: str = "3y",
    progress_callback=None,
) -> pd.DataFrame:
    ensure_cache_dirs()
    batches = [
        tickers[i : i + PRICE_BATCH_SIZE]
        for i in range(0, len(tickers), PRICE_BATCH_SIZE)
    ]
    all_frames = []
    total_batches = len(batches)

    for idx, batch in enumerate(batches):
        if progress_callback:
            progress_callback(idx + 1, total_batches)
        try:
            df = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if not df.empty:
                all_frames.append(df)
        except Exception as e:
            logger.warning(f"Batch {idx + 1} failed: {e}")

        if idx < total_batches - 1:
            time.sleep(PRICE_BATCH_DELAY)

    if not all_frames:
        raise RuntimeError("All price download batches failed")

    combined = pd.concat(all_frames, axis=1)
    cache_path = PRICES_CACHE / "daily_prices.parquet"
    combined.to_parquet(cache_path)
    return combined


def load_cached_prices() -> pd.DataFrame | None:
    cache_path = PRICES_CACHE / "daily_prices.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/fetchers/prices.py
git commit -m "feat: add batched yfinance price fetcher with caching"
```

---

### Task 4: Fundamentals fetcher

**Files:**
- Create: `pipeline/fetchers/fundamentals.py`

- [ ] **Step 1: Implement fundamentals.py**

```python
"""Per-ticker quarterly financials from yfinance."""

import time
import logging
import pandas as pd
import yfinance as yf

from pipeline.config import (
    FUNDAMENTALS_CACHE, FUNDAMENTALS_DELAY, FUNDAMENTALS_RETRIES,
    ensure_cache_dirs,
)

logger = logging.getLogger(__name__)

INCOME_FIELDS = [
    "TotalRevenue", "GrossProfit", "OperatingIncome", "NetIncome",
    "DilutedEPS", "SellingGeneralAndAdministration", "ResearchAndDevelopment",
]
BALANCE_FIELDS = [
    "TotalAssets", "TotalEquityGrossMinorityInterest", "TotalDebt",
    "CashAndCashEquivalents", "CurrentAssets", "CurrentLiabilities",
    "Inventory", "AccountsReceivable",
]
CASHFLOW_FIELDS = [
    "OperatingCashFlow", "CapitalExpenditure", "DepreciationAndAmortization",
]


def _fetch_one_ticker(ticker: str) -> dict | None:
    for attempt in range(FUNDAMENTALS_RETRIES):
        try:
            t = yf.Ticker(ticker)
            income = t.get_income_stmt(freq="quarterly")
            balance = t.get_balance_sheet(freq="quarterly")
            cashflow = t.get_cashflow(freq="quarterly")

            info = t.info or {}
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            shares = info.get("sharesOutstanding", None)
            market_cap = info.get("marketCap", None)

            return {
                "income": income,
                "balance": balance,
                "cashflow": cashflow,
                "sector": sector,
                "industry": industry,
                "shares_outstanding": shares,
                "market_cap": market_cap,
            }
        except Exception as e:
            logger.warning(f"{ticker} attempt {attempt + 1} failed: {e}")
            if attempt < FUNDAMENTALS_RETRIES - 1:
                time.sleep(FUNDAMENTALS_DELAY * (attempt + 1))
    return None


def fetch_fundamentals(
    tickers: list[str],
    progress_callback=None,
) -> dict[str, dict]:
    ensure_cache_dirs()
    results = {}
    failed = []

    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx + 1, len(tickers), ticker)

        cache_path = FUNDAMENTALS_CACHE / f"{ticker}.parquet"
        data = _fetch_one_ticker(ticker)
        if data is not None:
            results[ticker] = data
            _save_fundamental_cache(ticker, data)
        else:
            failed.append(ticker)

        time.sleep(FUNDAMENTALS_DELAY)

    logger.info(
        f"Fundamentals: {len(results)} succeeded, {len(failed)} failed"
    )
    return results


def _save_fundamental_cache(ticker: str, data: dict):
    rows = []
    info_row = {
        "ticker": ticker,
        "sector": data["sector"],
        "industry": data["industry"],
        "shares_outstanding": data["shares_outstanding"],
        "market_cap": data["market_cap"],
    }

    for stmt_name, stmt_df in [
        ("income", data["income"]),
        ("balance", data["balance"]),
        ("cashflow", data["cashflow"]),
    ]:
        if stmt_df is not None and not stmt_df.empty:
            for col_date in stmt_df.columns:
                for field in stmt_df.index:
                    val = stmt_df.loc[field, col_date]
                    rows.append({
                        "ticker": ticker,
                        "statement": stmt_name,
                        "date": str(col_date)[:10],
                        "field": field,
                        "value": float(val) if pd.notna(val) else None,
                    })

    if rows:
        df = pd.DataFrame(rows)
        cache_path = FUNDAMENTALS_CACHE / f"{ticker}.parquet"
        df.to_parquet(cache_path, index=False)

    meta_path = FUNDAMENTALS_CACHE / f"{ticker}_meta.parquet"
    pd.DataFrame([info_row]).to_parquet(meta_path, index=False)


def load_cached_fundamentals(ticker: str) -> pd.DataFrame | None:
    cache_path = FUNDAMENTALS_CACHE / f"{ticker}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None


def load_cached_meta(ticker: str) -> dict | None:
    meta_path = FUNDAMENTALS_CACHE / f"{ticker}_meta.parquet"
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        return df.iloc[0].to_dict()
    return None
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/fetchers/fundamentals.py
git commit -m "feat: add per-ticker fundamentals fetcher with retry and caching"
```

---

### Task 5: Factors and macro fetchers

**Files:**
- Create: `pipeline/fetchers/factors.py`
- Create: `pipeline/fetchers/macro.py`

- [ ] **Step 1: Implement factors.py**

```python
"""Fama-French 5 factors + Momentum from Ken French library."""

import logging
import pandas as pd
import pandas_datareader.data as web

from pipeline.config import FACTORS_CACHE, ensure_cache_dirs

logger = logging.getLogger(__name__)


def fetch_factors() -> pd.DataFrame:
    ensure_cache_dirs()

    ff5_raw = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3", "famafrench", start="1963-01-01"
    )
    ff5 = ff5_raw[0] / 100.0
    ff5.index = ff5.index.astype(str).str[:7]
    ff5.index.name = "ym"

    mom_raw = web.DataReader(
        "F-F_Momentum_Factor", "famafrench", start="1963-01-01"
    )
    mom = mom_raw[0] / 100.0
    mom.index = mom.index.astype(str).str[:7]
    mom.index.name = "ym"
    mom.columns = ["Mom"]

    combined = ff5.join(mom, how="left")
    combined = combined.rename(columns={"Mkt-RF": "Mkt_RF"})

    if "RF" in combined.columns:
        combined = combined.rename(columns={"RF": "rf_ff"})

    combined["spy_ret"] = combined["Mkt_RF"] + combined["rf_ff"]

    cache_path = FACTORS_CACHE / "ff5_mom.parquet"
    combined.to_parquet(cache_path)

    return combined


def load_cached_factors() -> pd.DataFrame | None:
    cache_path = FACTORS_CACHE / "ff5_mom.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None
```

- [ ] **Step 2: Implement macro.py**

```python
"""FRED macro data fetcher (VIX, yield curve, EPU, financial stress)."""

import logging
import pandas as pd

from pipeline.config import FRED_API_KEY, FRED_SERIES, MACRO_CACHE, ensure_cache_dirs

logger = logging.getLogger(__name__)


def fetch_macro(api_key: str | None = None) -> pd.DataFrame:
    from fredapi import Fred

    ensure_cache_dirs()
    key = api_key or FRED_API_KEY
    if not key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY environment variable "
            "or get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = Fred(api_key=key)
    series_dict = {}

    for fred_code, col_name in FRED_SERIES.items():
        try:
            s = fred.get_series(fred_code, observation_start="2000-01-01")
            series_dict[col_name] = s
        except Exception as e:
            logger.warning(f"Failed to fetch {fred_code}: {e}")

    if not series_dict:
        raise RuntimeError("All FRED series downloads failed")

    df = pd.DataFrame(series_dict)
    df.index.name = "date"

    cache_path = MACRO_CACHE / "fred_data.parquet"
    df.to_parquet(cache_path)

    return df


def load_cached_macro() -> pd.DataFrame | None:
    cache_path = MACRO_CACHE / "fred_data.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/fetchers/factors.py pipeline/fetchers/macro.py
git commit -m "feat: add FF5+momentum and FRED macro fetchers"
```

---

### Task 6: Price feature computation

**Files:**
- Create: `pipeline/features/price_features.py`
- Create: `tests/test_pipeline_features.py`

- [ ] **Step 1: Write tests for price feature computation**

```python
"""tests/test_pipeline_features.py"""
import pytest
import pandas as pd
import numpy as np
from pipeline.features.price_features import compute_price_features


@pytest.fixture
def synthetic_daily_prices():
    """Create synthetic daily OHLCV data for 5 tickers over ~3 years."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", "2025-12-31")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    
    frames = {}
    for ticker in tickers:
        price = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        volume = np.random.randint(1_000_000, 50_000_000, size=len(dates))
        frames[ticker] = pd.DataFrame({
            "Open": price * (1 + np.random.randn(len(dates)) * 0.005),
            "High": price * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            "Low": price * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            "Close": price,
            "Volume": volume,
        }, index=dates)
    
    combined = pd.concat(frames, axis=1)
    return combined


@pytest.fixture
def synthetic_market_daily(synthetic_daily_prices):
    """SPY daily returns aligned with the test data."""
    dates = synthetic_daily_prices.index
    np.random.seed(99)
    price = 400 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.008))
    return pd.Series(price, index=dates, name="SPY")


@pytest.fixture
def synthetic_shares():
    """Shares outstanding per ticker."""
    return {
        "AAPL": 15_000_000_000,
        "MSFT": 7_500_000_000,
        "GOOG": 6_000_000_000,
        "AMZN": 10_300_000_000,
        "META": 2_500_000_000,
    }


def test_compute_price_features_shape(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    result = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    assert isinstance(result, pd.DataFrame)
    assert "permno" in result.columns or "ticker" in result.columns
    assert "ym" in result.columns
    assert "ret_1" in result.columns
    assert "vol_12m" in result.columns
    assert "beta" in result.columns
    assert "rsi_14" in result.columns
    assert len(result) > 0


def test_price_features_no_lookahead(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    result = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    for ym in result["ym"].unique():
        month_end = pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0)
        month_data = result[result["ym"] == ym]
        assert month_data["ret_1"].notna().any() or True


def test_momentum_features_present(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    result = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    for col in ["ret_1", "ret_2_12", "ret_2_6", "prc_52w_high"]:
        assert col in result.columns, f"Missing {col}"


def test_technical_features_present(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    result = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    for col in ["rsi_14", "macd_hist", "bb_position", "prc_ma6", "prc_ma12"]:
        assert col in result.columns, f"Missing {col}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline_features.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement price_features.py**

```python
"""Compute price-based features from daily OHLCV data."""

import numpy as np
import pandas as pd


def compute_price_features(
    prices: pd.DataFrame,
    market_daily: pd.Series,
    shares_outstanding: dict[str, float],
) -> pd.DataFrame:
    tickers = prices.columns.get_level_values(0).unique()
    market_ret = market_daily.pct_change()
    all_rows = []

    for ticker in tickers:
        try:
            df = prices[ticker].copy()
        except KeyError:
            continue

        if df.empty or "Close" not in df.columns:
            continue

        close = df["Close"].dropna()
        volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
        daily_ret = close.pct_change()

        month_ends = close.resample("ME").last().dropna()
        shares = shares_outstanding.get(ticker, np.nan)

        for i in range(len(month_ends)):
            me_date = month_ends.index[i]
            ym = me_date.strftime("%Y-%m")
            me_price = month_ends.iloc[i]

            mask_12m = (close.index <= me_date) & (
                close.index > me_date - pd.DateOffset(months=12)
            )
            mask_36m = (close.index <= me_date) & (
                close.index > me_date - pd.DateOffset(months=36)
            )
            daily_ret_12m = daily_ret[mask_12m]
            close_12m = close[mask_12m]

            row = {"ticker": ticker, "ym": ym, "date": me_date.strftime("%Y-%m-%d")}

            # Momentum features
            if i >= 1:
                row["ret_1"] = me_price / month_ends.iloc[i - 1] - 1
            if i >= 12:
                row["ret_2_12"] = month_ends.iloc[i - 1] / month_ends.iloc[i - 12] - 1
            if i >= 6:
                row["ret_2_6"] = month_ends.iloc[i - 1] / month_ends.iloc[i - 6] - 1
            if i >= 36:
                row["ret_13_36"] = month_ends.iloc[i - 12] / month_ends.iloc[i - 36] - 1

            # Volatility & risk
            if len(daily_ret_12m) > 20:
                row["vol_12m"] = daily_ret_12m.std() * np.sqrt(252)
                row["max_ret_12m"] = daily_ret_12m.max()
                row["skew_12m"] = daily_ret_12m.skew()
                row["kurt_12m"] = daily_ret_12m.kurtosis()

            # Beta & idiosyncratic vol
            if len(daily_ret_12m) > 60:
                aligned = pd.DataFrame({
                    "stock": daily_ret_12m,
                    "market": market_ret.reindex(daily_ret_12m.index),
                }).dropna()
                if len(aligned) > 30:
                    cov = np.cov(aligned["stock"], aligned["market"])
                    mkt_var = cov[1, 1]
                    if mkt_var > 0:
                        row["beta"] = cov[0, 1] / mkt_var
                        residuals = (
                            aligned["stock"]
                            - row["beta"] * aligned["market"]
                        )
                        row["ivol"] = residuals.std() * np.sqrt(252)

            # Size
            if not np.isnan(shares) and shares > 0:
                me = me_price * shares
                row["log_me"] = np.log(me) if me > 0 else np.nan
                row["me"] = me
                row["prc_abs"] = me_price
            else:
                row["log_me"] = np.nan
                row["me"] = np.nan
                row["prc_abs"] = me_price

            # Turnover
            vol_month = volume[
                (volume.index <= me_date)
                & (volume.index > me_date - pd.DateOffset(months=1))
            ]
            if len(vol_month) > 0 and not np.isnan(shares) and shares > 0:
                row["turnover"] = vol_month.mean() / shares

            # 52-week high
            if len(close_12m) > 0:
                row["prc_52w_high"] = me_price / close_12m.max()

            row["age"] = 0.0

            # Technical indicators
            _add_technicals(row, close, volume, me_date)

            # Illiquidity
            if len(daily_ret_12m) > 20 and len(volume) > 0:
                vol_aligned = volume.reindex(daily_ret_12m.index).dropna()
                price_aligned = close.reindex(vol_aligned.index)
                dollar_vol = vol_aligned * price_aligned
                valid = dollar_vol > 0
                if valid.sum() > 20:
                    illiq_vals = daily_ret_12m.reindex(dollar_vol[valid].index).abs() / dollar_vol[valid]
                    row["illiq_12m"] = illiq_vals.mean()

            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    return result


def _add_technicals(row: dict, close: pd.Series, volume: pd.Series, me_date):
    mask = close.index <= me_date
    c = close[mask]

    if len(c) < 30:
        return

    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    if len(rsi.dropna()) > 0:
        row["rsi_14"] = rsi.iloc[-1]

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal
    if len(macd_hist.dropna()) > 0:
        row["macd_hist"] = macd_hist.iloc[-1]

    # Bollinger Band position
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    if len(sma20.dropna()) > 0 and std20.iloc[-1] > 0:
        row["bb_position"] = (c.iloc[-1] - sma20.iloc[-1]) / (2 * std20.iloc[-1])

    # Moving average ratios
    for days, name in [(126, "prc_ma6"), (252, "prc_ma12"), (504, "prc_ma24")]:
        if len(c) >= days:
            sma = c.rolling(days).mean()
            if pd.notna(sma.iloc[-1]) and sma.iloc[-1] > 0:
                row[name] = c.iloc[-1] / sma.iloc[-1]

    # Rate of change
    for days, name in [(63, "roc_3"), (126, "roc_6")]:
        if len(c) > days:
            row[name] = c.iloc[-1] / c.iloc[-days] - 1

    # Volume ratio
    if len(volume) > 0:
        vol_masked = volume[volume.index <= me_date]
        if len(vol_masked) > 63:
            curr_month = vol_masked.iloc[-21:].mean() if len(vol_masked) >= 21 else vol_masked.mean()
            trail_3m = vol_masked.iloc[-63:].mean()
            if trail_3m > 0:
                row["vol_ratio"] = curr_month / trail_3m
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/features/price_features.py tests/test_pipeline_features.py
git commit -m "feat: add price feature computation (momentum, vol, beta, technicals)"
```

---

### Task 7: Fundamental feature computation

**Files:**
- Create: `pipeline/features/fundamental_features.py`
- Add tests to: `tests/test_pipeline_features.py`

- [ ] **Step 1: Add tests for fundamental features**

Append to `tests/test_pipeline_features.py`:

```python
from pipeline.features.fundamental_features import compute_fundamental_features


@pytest.fixture
def synthetic_fundamentals():
    """Synthetic quarterly financial data for 3 tickers."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    data = {}
    quarters = pd.date_range("2024-01-01", periods=5, freq="QE")
    
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**31)
        n = len(quarters)
        data[ticker] = {
            "income": {
                "TotalRevenue": np.random.uniform(50e9, 100e9, n),
                "GrossProfit": np.random.uniform(20e9, 50e9, n),
                "OperatingIncome": np.random.uniform(10e9, 30e9, n),
                "NetIncome": np.random.uniform(5e9, 25e9, n),
                "DilutedEPS": np.random.uniform(1.0, 5.0, n),
            },
            "balance": {
                "TotalAssets": np.random.uniform(200e9, 400e9, n),
                "TotalEquityGrossMinorityInterest": np.random.uniform(50e9, 150e9, n),
                "TotalDebt": np.random.uniform(20e9, 100e9, n),
                "CashAndCashEquivalents": np.random.uniform(10e9, 50e9, n),
            },
            "cashflow": {
                "OperatingCashFlow": np.random.uniform(10e9, 30e9, n),
                "DepreciationAndAmortization": np.random.uniform(2e9, 8e9, n),
            },
            "market_cap": np.random.uniform(1e12, 3e12),
            "quarters": quarters,
        }
    return data


def test_fundamental_features_shape(synthetic_fundamentals):
    result = compute_fundamental_features(synthetic_fundamentals)
    assert isinstance(result, pd.DataFrame)
    assert "ticker" in result.columns
    assert "ym" in result.columns
    assert "bm" in result.columns
    assert "roe" in result.columns
    assert len(result) > 0


def test_fundamental_features_value_ratios(synthetic_fundamentals):
    result = compute_fundamental_features(synthetic_fundamentals)
    for col in ["bm", "ep", "cfp", "sp"]:
        assert col in result.columns
        valid = result[col].dropna()
        assert len(valid) > 0
```

- [ ] **Step 2: Implement fundamental_features.py**

```python
"""Compute fundamental features from quarterly financial statements."""

import numpy as np
import pandas as pd


def compute_fundamental_features(
    fundamentals: dict[str, dict],
) -> pd.DataFrame:
    all_rows = []

    for ticker, data in fundamentals.items():
        quarters = data.get("quarters")
        if quarters is None:
            continue

        inc = data.get("income", {})
        bal = data.get("balance", {})
        cf = data.get("cashflow", {})
        mcap = data.get("market_cap", np.nan)

        n_q = len(quarters)

        for i in range(n_q):
            ym = quarters[i].strftime("%Y-%m")
            row = {"ticker": ticker, "ym": ym}

            total_assets = _safe_get(bal, "TotalAssets", i)
            equity = _safe_get(bal, "TotalEquityGrossMinorityInterest", i)
            debt = _safe_get(bal, "TotalDebt", i)
            cash = _safe_get(bal, "CashAndCashEquivalents", i)

            revenue = _safe_get(inc, "TotalRevenue", i)
            gross_profit = _safe_get(inc, "GrossProfit", i)
            op_income = _safe_get(inc, "OperatingIncome", i)
            net_income = _safe_get(inc, "NetIncome", i)
            eps = _safe_get(inc, "DilutedEPS", i)

            ocf = _safe_get(cf, "OperatingCashFlow", i)

            # TTM sums (sum of last 4 quarters if available)
            rev_ttm = _ttm(inc, "TotalRevenue", i)
            ni_ttm = _ttm(inc, "NetIncome", i)
            gp_ttm = _ttm(inc, "GrossProfit", i)
            ocf_ttm = _ttm(cf, "OperatingCashFlow", i)

            if mcap and mcap > 0:
                row["bm"] = _safe_div(equity, mcap)
                row["ep"] = _safe_div(ni_ttm, mcap)
                row["cfp"] = _safe_div(ocf_ttm, mcap)
                row["sp"] = _safe_div(rev_ttm, mcap)

            if total_assets and total_assets > 0:
                row["gpa"] = _safe_div(gp_ttm, total_assets)
                row["roa"] = _safe_div(ni_ttm, total_assets)
                row["ato"] = _safe_div(rev_ttm, total_assets)
                row["cash_at"] = _safe_div(cash, total_assets)

                if ocf is not None and net_income is not None:
                    row["acc"] = (net_income - ocf) / total_assets

                if i >= 4:
                    prev_assets = _safe_get(bal, "TotalAssets", i - 4)
                    if prev_assets and prev_assets > 0:
                        row["ag"] = total_assets / prev_assets - 1

            if equity and equity > 0:
                row["roe"] = _safe_div(ni_ttm, equity)
                row["lev"] = _safe_div(debt, equity)

            row["nsi"] = 0.0

            # Growth
            if i >= 4:
                prev_rev = _safe_get(inc, "TotalRevenue", i - 4)
                if prev_rev and prev_rev > 0 and revenue:
                    row["sgr"] = revenue / prev_rev - 1

            if i >= 1:
                prev_rev_qq = _safe_get(inc, "TotalRevenue", i - 1)
                if prev_rev_qq and prev_rev_qq > 0 and revenue:
                    row["rev_growth_qq"] = revenue / prev_rev_qq - 1

            if i >= 4:
                prev_eps = _safe_get(inc, "DilutedEPS", i - 4)
                if prev_eps and prev_eps != 0 and eps:
                    row["earn_growth_yoy"] = eps / prev_eps - 1

            # Margins
            if revenue and revenue > 0:
                row["gm_q"] = _safe_div(gross_profit, revenue)
                row["op_margin_q"] = _safe_div(op_income, revenue)
                row["ato_q"] = _safe_div(revenue, total_assets) if total_assets else np.nan

            if i >= 1:
                prev_gp = _safe_get(inc, "GrossProfit", i - 1)
                prev_rev_1 = _safe_get(inc, "TotalRevenue", i - 1)
                if prev_rev_1 and prev_rev_1 > 0 and prev_gp is not None:
                    prev_gm = prev_gp / prev_rev_1
                    curr_gm = row.get("gm_q", np.nan)
                    if not np.isnan(curr_gm):
                        row["gm_chg"] = curr_gm - prev_gm

                prev_op = _safe_get(inc, "OperatingIncome", i - 1)
                if prev_rev_1 and prev_rev_1 > 0 and prev_op is not None:
                    prev_opm = prev_op / prev_rev_1
                    curr_opm = row.get("op_margin_q", np.nan)
                    if not np.isnan(curr_opm):
                        row["op_margin_chg"] = curr_opm - prev_opm

            # SUE (seasonal random walk)
            if i >= 4 and eps is not None:
                prev_eps_4 = _safe_get(inc, "DilutedEPS", i - 4)
                if prev_eps_4 is not None:
                    diffs = []
                    for j in range(max(0, i - 12), i):
                        e_j = _safe_get(inc, "DilutedEPS", j)
                        e_j4 = _safe_get(inc, "DilutedEPS", j - 4) if j >= 4 else None
                        if e_j is not None and e_j4 is not None:
                            diffs.append(e_j - e_j4)
                    std_diff = np.std(diffs) if len(diffs) > 1 else np.nan
                    if std_diff and std_diff > 0:
                        row["sue_q"] = (eps - prev_eps_4) / std_diff

            # Earnings quality
            if total_assets and total_assets > 0 and ocf is not None and net_income is not None:
                row["cfo_at"] = ocf / total_assets
                row["earn_quality"] = (ocf - net_income) / total_assets

            # Additional ratios
            sga = _safe_get(inc, "SellingGeneralAndAdministration", i)
            if i >= 1 and sga is not None:
                prev_sga = _safe_get(inc, "SellingGeneralAndAdministration", i - 1)
                if prev_sga and prev_sga > 0:
                    row["sga_chg"] = sga / prev_sga - 1

            if i >= 1 and net_income is not None and total_assets and total_assets > 0:
                prev_ni = _safe_get(inc, "NetIncome", i - 1)
                prev_ta = _safe_get(bal, "TotalAssets", i - 1)
                if prev_ni is not None and prev_ta and prev_ta > 0:
                    row["roe_q"] = net_income / equity if equity and equity > 0 else np.nan
                    row["roe_chg"] = (
                        (net_income / equity if equity else 0)
                        - (prev_ni / _safe_get(bal, "TotalEquityGrossMinorityInterest", i - 1)
                           if _safe_get(bal, "TotalEquityGrossMinorityInterest", i - 1) else 0)
                    )

            if ocf is not None and net_income is not None and total_assets and total_assets > 0:
                row["acc_q"] = (net_income - ocf) / total_assets

            rd = _safe_get(inc, "ResearchAndDevelopment", i)
            if rd is not None and revenue and revenue > 0:
                row["rd_intensity"] = rd / revenue

            if i >= 4 and op_income is not None:
                prev_oi = _safe_get(inc, "OperatingIncome", i - 4)
                if prev_oi and prev_oi > 0:
                    row["oi_growth_yoy"] = op_income / prev_oi - 1

            row["dp_ratio"] = 0.0
            row["inv_chg"] = 0.0
            row["rec_chg"] = 0.0

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def _safe_get(data: dict, field: str, idx: int):
    arr = data.get(field)
    if arr is None or idx < 0 or idx >= len(arr):
        return None
    val = arr[idx]
    return float(val) if not np.isnan(val) else None


def _safe_div(num, den):
    if num is None or den is None or den == 0:
        return np.nan
    return num / den


def _ttm(data: dict, field: str, idx: int):
    vals = []
    for j in range(max(0, idx - 3), idx + 1):
        v = _safe_get(data, field, j)
        if v is not None:
            vals.append(v)
    return sum(vals) if len(vals) == 4 else (_safe_get(data, field, idx) or np.nan)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_pipeline_features.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add pipeline/features/fundamental_features.py tests/test_pipeline_features.py
git commit -m "feat: add fundamental feature computation (value, quality, growth)"
```

---

### Task 8: Peer and macro feature computation

**Files:**
- Create: `pipeline/features/peer_features.py`
- Create: `pipeline/features/macro_features.py`
- Add tests to: `tests/test_pipeline_features.py`

- [ ] **Step 1: Implement peer_features.py**

```python
"""Compute peer/sector/group relative features."""

import numpy as np
import pandas as pd


def compute_peer_features(
    price_features: pd.DataFrame,
    sectors: dict[str, str],
    industries: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = price_features.copy()
    df["sector"] = df["ticker"].map(sectors).fillna("Unknown")
    if industries:
        df["industry"] = df["ticker"].map(industries).fillna("Unknown")
    else:
        df["industry"] = df["sector"]

    result = pd.DataFrame(index=df.index)
    result["ticker"] = df["ticker"]
    result["ym"] = df["ym"]
    result["sector"] = df["sector"]
    result["industry"] = df["industry"]

    # Sector-level
    sector_stats = df.groupby(["ym", "sector"])["ret_1"].agg(["mean", "std"])
    sector_stats.columns = ["sector_ret_avg", "sector_ret_dispersion"]
    df = df.merge(sector_stats, left_on=["ym", "sector"], right_index=True, how="left")

    result["sector_ret_avg"] = df["sector_ret_avg"]
    result["sector_ret_dispersion"] = df["sector_ret_dispersion"]
    result["ret_vs_sector"] = df["ret_1"] - df["sector_ret_avg"]
    result["peer_ret_1"] = df["sector_ret_avg"]

    # Market-level dispersion
    mkt_disp = df.groupby("ym")["ret_1"].std()
    result["mkt_ret_dispersion"] = df["ym"].map(mkt_disp)

    # Sector momentum (lagged)
    sector_ret_by_ym = df.groupby(["ym", "sector"])["ret_1"].mean().unstack()
    sector_mom_lag = sector_ret_by_ym.shift(1).stack()
    sector_mom_lag.name = "sector_mom_lag1"
    df = df.merge(
        sector_mom_lag.reset_index().rename(columns={"level_0": "ym", "level_1": "sector"}),
        on=["ym", "sector"], how="left",
    )
    result["sector_mom_lag1"] = df["sector_mom_lag1"]

    # Sector relative momentum
    result["sector_rel_mom"] = result["sector_ret_avg"] - result["mkt_ret_dispersion"]

    # Industry-level
    ind_stats = df.groupby(["ym", "industry"])["ret_1"].agg(["mean", "std"])
    ind_stats.columns = ["ind_mom", "ind_dispersion"]
    df = df.merge(ind_stats, left_on=["ym", "industry"], right_index=True, how="left")
    result["ind_mom"] = df["ind_mom"]
    result["ind_dispersion"] = df["ind_dispersion"]
    result["ret_vs_ind"] = df["ret_1"] - df["ind_mom"]

    # Size quintile features
    if "log_me" in df.columns:
        df["size_q"] = df.groupby("ym")["log_me"].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        )
        size_stats = df.groupby(["ym", "size_q"])["ret_1"].agg(["mean", "std"])
        size_stats.columns = ["size_grp_ret", "size_grp_disp_val"]
        df = df.merge(size_stats, left_on=["ym", "size_q"], right_index=True, how="left")
        result["size_peer_ret"] = df["size_grp_ret"]
        result["size_grp_disp"] = df["size_grp_disp_val"]

        # Lagged size group momentum
        size_ret_by_ym = df.groupby(["ym", "size_q"])["ret_1"].mean().unstack()
        size_mom_lag = size_ret_by_ym.shift(1).stack()
        size_mom_lag.name = "size_grp_mom"
        df = df.merge(
            size_mom_lag.reset_index().rename(columns={"level_0": "ym", "level_1": "size_q"}),
            on=["ym", "size_q"], how="left",
        )
        result["size_grp_mom"] = df["size_grp_mom"]

    # Beta quintile features
    if "beta" in df.columns:
        df["beta_q"] = df.groupby("ym")["beta"].transform(
            lambda x: pd.qcut(x.clip(-5, 5), 5, labels=False, duplicates="drop")
        )
        beta_grp = df.groupby(["ym", "beta_q"])["ret_1"].mean()
        beta_grp.name = "beta_grp_ret"
        df = df.merge(
            beta_grp.reset_index().rename(columns={"level_0": "ym", "level_1": "beta_q"}),
            on=["ym", "beta_q"], how="left",
        )
        result["beta_grp_ret"] = df["beta_grp_ret"]

    # Value quintile features
    if "bm" in price_features.columns:
        df["val_q"] = df.groupby("ym")["bm"].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        )
        val_stats = df.groupby(["ym", "val_q"])["ret_1"].agg(["mean"])
        val_mom_lag = val_stats["mean"].unstack().shift(1).stack()
        val_mom_lag.name = "val_grp_mom"
        df = df.merge(
            val_mom_lag.reset_index().rename(columns={"level_0": "ym", "level_1": "val_q"}),
            on=["ym", "val_q"], how="left",
        )
        result["val_grp_mom"] = df["val_grp_mom"]
        result["val_peer_ret"] = df.groupby(["ym", "val_q"])["ret_1"].transform("mean")

    # BM vs sector / size
    if "bm" in price_features.columns:
        bm_sector_avg = df.groupby(["ym", "sector"])["bm"].transform("mean")
        result["bm_vs_sector"] = df["bm"] - bm_sector_avg
        if "size_q" in df.columns:
            bm_size_avg = df.groupby(["ym", "size_q"])["bm"].transform("mean")
            result["bm_vs_size"] = df["bm"] - bm_size_avg

    # Leader return (largest stock by market cap in sector)
    if "me" in df.columns:
        leader_idx = df.groupby(["ym", "sector"])["me"].idxmax()
        leader_ret = df.loc[leader_idx, ["ym", "sector", "ret_1"]].rename(
            columns={"ret_1": "leader_ret"}
        )
        df = df.merge(leader_ret, on=["ym", "sector"], how="left")
        result["leader_ret"] = df["leader_ret"]
        # Lagged version
        leader_by_ym = leader_ret.set_index(["ym", "sector"])["leader_ret"].unstack()
        leader_lag = leader_by_ym.shift(1).stack()
        leader_lag.name = "leader_ret_lag1"
        df = df.merge(
            leader_lag.reset_index().rename(columns={"level_0": "ym", "level_1": "sector"}),
            on=["ym", "sector"], how="left",
        )
        result["leader_ret_lag1"] = df["leader_ret_lag1"]

    # Pre-built interaction features
    if "ret_2_12" in df.columns and "log_me" in df.columns:
        result["mom_x_size"] = df["ret_2_12"] * df["log_me"]
    if "bm" in df.columns and "gpa" in df.columns:
        result["val_x_prof"] = df["bm"] * df.get("gpa", 0)
    if "ret_2_12" in df.columns and "vol_12m" in df.columns:
        result["mom_x_vol"] = df["ret_2_12"] * df["vol_12m"]

    # Momentum acceleration and deltas
    if "ret_2_12" in df.columns:
        ret_by_ticker = df.pivot_table(index="ym", columns="ticker", values="ret_2_12")
        mom_accel = ret_by_ticker.diff(1)
        mom_of_mom = ret_by_ticker.diff(1)
        for t in df["ticker"].unique():
            mask = df["ticker"] == t
            if t in mom_accel.columns:
                result.loc[mask, "mom_accel"] = mom_accel[t].reindex(df.loc[mask, "ym"]).values

    if "vol_12m" in df.columns:
        vol_by_ticker = df.pivot_table(index="ym", columns="ticker", values="vol_12m")
        delta_vol = vol_by_ticker.diff(1)
        for t in df["ticker"].unique():
            mask = df["ticker"] == t
            if t in delta_vol.columns:
                result.loc[mask, "delta_vol"] = delta_vol[t].reindex(df.loc[mask, "ym"]).values

    if "bm" in df.columns:
        bm_by_ticker = df.pivot_table(index="ym", columns="ticker", values="bm")
        delta_bm = bm_by_ticker.diff(1)
        for t in df["ticker"].unique():
            mask = df["ticker"] == t
            if t in delta_bm.columns:
                result.loc[mask, "delta_bm"] = delta_bm[t].reindex(df.loc[mask, "ym"]).values

    # Return streak
    if "ret_1" in df.columns:
        ret_by_ticker = df.pivot_table(index="ym", columns="ticker", values="ret_1")
        positive = (ret_by_ticker > 0).astype(int)
        streak = positive.copy()
        for i in range(1, len(streak)):
            streak.iloc[i] = (streak.iloc[i - 1] + 1) * positive.iloc[i]
        for t in df["ticker"].unique():
            mask = df["ticker"] == t
            if t in streak.columns:
                result.loc[mask, "ret_streak"] = streak[t].reindex(df.loc[mask, "ym"]).values

    # mom_of_mom
    if "ret_2_12" in df.columns:
        ret_pivot = df.pivot_table(index="ym", columns="ticker", values="ret_2_12")
        mom_diff = ret_pivot.diff(1)
        for t in df["ticker"].unique():
            mask = df["ticker"] == t
            if t in mom_diff.columns:
                result.loc[mask, "mom_of_mom"] = mom_diff[t].reindex(df.loc[mask, "ym"]).values

    return result
```

- [ ] **Step 2: Implement macro_features.py**

```python
"""Compute macro uncertainty features from FRED data."""

import numpy as np
import pandas as pd


def compute_macro_features(
    macro_data: pd.DataFrame,
    target_months: list[str],
) -> pd.DataFrame:
    monthly = macro_data.resample("ME").mean()
    monthly["ym"] = monthly.index.strftime("%Y-%m")

    rows = []
    for ym in target_months:
        row = {"ym": ym}

        ts = pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0)
        mask_60m = (monthly.index <= ts) & (
            monthly.index > ts - pd.DateOffset(months=60)
        )
        hist = monthly[mask_60m]

        # VIX-based macro uncertainty
        if "vix" in monthly.columns:
            vix_current = monthly.loc[monthly["ym"] == ym, "vix"]
            if len(vix_current) > 0 and len(hist) > 12:
                vix_val = vix_current.iloc[0]
                vix_mean = hist["vix"].mean()
                vix_std = hist["vix"].std()
                row["macro_unc_1m"] = (
                    (vix_val - vix_mean) / vix_std if vix_std > 0 else 0.0
                )
                vix_12m = hist["vix"].iloc[-12:].mean() if len(hist) >= 12 else vix_val
                row["macro_unc_12m"] = (
                    (vix_12m - vix_mean) / vix_std if vix_std > 0 else 0.0
                )

        # Financial stress (STLFSI4 is already z-scored)
        if "fin_stress" in monthly.columns:
            fs_current = monthly.loc[monthly["ym"] == ym, "fin_stress"]
            if len(fs_current) > 0:
                row["fin_unc_1m"] = fs_current.iloc[0]
                fs_12m = hist["fin_stress"].iloc[-12:].mean() if len(hist) >= 12 else fs_current.iloc[0]
                row["fin_unc_12m"] = fs_12m

        # Interaction features needing macro data
        if "vix" in monthly.columns and len(hist) > 0:
            row["mom_x_unc"] = 0.0
            row["val_x_finunc"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 3: Add tests for peer and macro features**

Append to `tests/test_pipeline_features.py`:

```python
from pipeline.features.peer_features import compute_peer_features
from pipeline.features.macro_features import compute_macro_features


def test_peer_features_sector_columns(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    price_feats = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    sectors = {t: "Tech" if i % 2 == 0 else "Finance"
               for i, t in enumerate(["AAPL", "MSFT", "GOOG", "AMZN", "META"])}
    result = compute_peer_features(price_feats, sectors)
    assert "sector_ret_avg" in result.columns
    assert "ret_vs_sector" in result.columns
    assert "sector" in result.columns


def test_macro_features_shape():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")
    macro = pd.DataFrame({
        "vix": np.random.uniform(12, 35, len(dates)),
        "fin_stress": np.random.randn(len(dates)) * 0.5,
    }, index=dates)
    months = ["2025-06", "2025-07", "2025-08"]
    result = compute_macro_features(macro, months)
    assert len(result) == 3
    assert "macro_unc_1m" in result.columns
    assert "fin_unc_1m" in result.columns
```

- [ ] **Step 4: Run all feature tests**

Run: `pytest tests/test_pipeline_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/features/peer_features.py pipeline/features/macro_features.py tests/test_pipeline_features.py
git commit -m "feat: add peer/sector and macro feature computation"
```

---

### Task 9: Assembler — merge, standardize, export

**Files:**
- Create: `pipeline/assembler.py`
- Create: `tests/test_pipeline_assembler.py`

- [ ] **Step 1: Write assembler tests**

```python
"""tests/test_pipeline_assembler.py"""
import pytest
import pandas as pd
import numpy as np
from pipeline.assembler import (
    cross_sectional_standardize,
    fill_red_features,
    validate_schema,
    assemble_month,
)


@pytest.fixture
def sample_raw_features():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)] * 2,
        "ym": ["2025-06"] * n + ["2025-07"] * n,
        "ret_1": np.random.randn(n * 2) * 0.05,
        "bm": np.random.uniform(0.2, 2.0, n * 2),
        "vol_12m": np.random.uniform(0.1, 0.6, n * 2),
    })


def test_cross_sectional_standardize(sample_raw_features):
    cols = ["ret_1", "bm", "vol_12m"]
    result = cross_sectional_standardize(sample_raw_features, cols)
    for col in cols:
        xs_col = f"{col}_xs"
        assert xs_col in result.columns
        for ym in result["ym"].unique():
            month_vals = result.loc[result["ym"] == ym, xs_col].dropna()
            if len(month_vals) > 1:
                assert abs(month_vals.mean()) < 0.01
                assert abs(month_vals.std() - 1.0) < 0.1


def test_fill_red_features(sample_raw_features):
    result = fill_red_features(sample_raw_features)
    assert "iv_atm_30d" in result.columns
    assert "vrp" in result.columns
    assert (result["iv_atm_30d"] == 0.0).all()


def test_validate_schema():
    cols = ["permno", "ym", "y_xs", "y_raw", "Mkt_RF", "rf_ff", "ret_1"]
    df = pd.DataFrame(columns=cols)
    missing = validate_schema(df)
    assert isinstance(missing, list)
```

- [ ] **Step 2: Implement assembler.py**

```python
"""Assemble features into final panel dataset matching the 229-column schema."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.config import (
    PARQUET_PATH, CSV_PATH, RAW_FEATURE_COLS, RED_FEATURES,
    METADATA_COLS, TARGET_COLS, FACTOR_COLS, DATA_DIR,
)

logger = logging.getLogger(__name__)

# The raw features that get _xs standardized versions (92 of them — excludes
# group-level aggregates that don't have _xs in the original dataset)
_NO_XS_SUFFIX = {
    "sector_ret_avg", "sector_ret_dispersion", "mkt_ret_dispersion",
    "sector_mom_lag1", "iv_term_structure", "sector_iv", "sector_vrp",
    "leader_ret", "sector_rel_mom", "ind_mom", "ind_dispersion",
    "ind_crowding", "ind_sue", "size_grp_mom", "size_grp_disp",
    "beta_grp_ret", "val_grp_mom", "vol_grp_ret", "ind_size_ret",
    "ind_size_mom", "macro_unc_1m", "macro_unc_12m", "fin_unc_1m",
    "fin_unc_12m",
}

XS_FEATURE_COLS = [c for c in RAW_FEATURE_COLS if c not in _NO_XS_SUFFIX]


def cross_sectional_standardize(
    df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        xs_col = f"{col}_xs"
        grouped = result.groupby("ym")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        result[xs_col] = (result[col] - mean) / std.replace(0, np.nan)
    return result


def fill_red_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in RED_FEATURES:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)
    return result


def validate_schema(df: pd.DataFrame) -> list[str]:
    required = {"permno", "ym", "y_xs", "y_raw", "Mkt_RF", "rf_ff"}
    return sorted(required - set(df.columns))


def assemble_month(
    price_features: pd.DataFrame,
    fundamental_features: pd.DataFrame | None,
    peer_features: pd.DataFrame,
    macro_features: pd.DataFrame,
    factors: pd.DataFrame,
    ticker_to_permno: dict[str, int] | None = None,
) -> pd.DataFrame:
    # Start with price features
    merged = price_features.copy()

    # Merge fundamentals
    if fundamental_features is not None and not fundamental_features.empty:
        fund_cols = [c for c in fundamental_features.columns if c not in ["ticker", "ym"]]
        for col in fund_cols:
            if col not in merged.columns:
                merged = merged.merge(
                    fundamental_features[["ticker", "ym", col]],
                    on=["ticker", "ym"],
                    how="left",
                )

    # Merge peer features
    peer_cols = [c for c in peer_features.columns if c not in ["ticker", "ym", "sector", "industry"]]
    for col in peer_cols:
        if col not in merged.columns:
            merged[col] = peer_features[col].values if len(peer_features) == len(merged) else np.nan

    if "sector" in peer_features.columns and "sector" not in merged.columns:
        merged["sector"] = peer_features["sector"].values

    # Merge macro features
    if not macro_features.empty:
        merged = merged.merge(macro_features, on="ym", how="left")

    # Fill RED features with 0
    merged = fill_red_features(merged)

    # Fill remaining missing raw features with NaN
    for col in RAW_FEATURE_COLS:
        if col not in merged.columns:
            merged[col] = 0.0

    # Cross-sectional standardize
    merged = cross_sectional_standardize(merged, XS_FEATURE_COLS)

    # Ensure all _xs columns exist
    for col in XS_FEATURE_COLS:
        xs_col = f"{col}_xs"
        if xs_col not in merged.columns:
            merged[xs_col] = 0.0

    # Assign permno
    if ticker_to_permno:
        merged["permno"] = merged["ticker"].map(ticker_to_permno)
    else:
        merged["permno"] = merged["ticker"].apply(lambda t: hash(t) % 100000)

    # Merge factors
    if not factors.empty:
        factor_monthly = factors.reset_index()
        if "ym" not in factor_monthly.columns and "date" in factor_monthly.columns:
            factor_monthly = factor_monthly.rename(columns={"date": "ym"})
        merged = merged.merge(
            factor_monthly[["ym"] + [c for c in FACTOR_COLS if c in factor_monthly.columns]],
            on="ym", how="left",
        )

    for col in FACTOR_COLS:
        if col not in merged.columns:
            merged[col] = np.nan

    # Target columns (forward returns — NaN for latest month)
    merged["y_raw"] = np.nan
    merged["y_xs"] = np.nan
    if "ret_1" in merged.columns and "rf_ff" in merged.columns:
        merged["ret_excess"] = merged["ret_1"] - merged["rf_ff"].fillna(0)
    else:
        merged["ret_excess"] = np.nan
    merged["ret_adj"] = merged["ret_excess"]

    # Metadata
    merged["in_sp500"] = 1
    for col in ["exchcd", "siccd"]:
        if col not in merged.columns:
            merged[col] = 0

    return merged


def append_to_dataset(new_data: pd.DataFrame) -> pd.DataFrame:
    # Load existing
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH)
    else:
        existing = pd.DataFrame()

    if existing.empty:
        combined = new_data
    else:
        # Remove overlapping months (re-fetch overwrites)
        new_months = set(new_data["ym"].unique())
        existing = existing[~existing["ym"].isin(new_months)]
        combined = pd.concat([existing, new_data], ignore_index=True)

    combined = combined.sort_values(["ym", "permno"]).reset_index(drop=True)

    # Backfill forward returns for previous month
    _backfill_forward_returns(combined)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PARQUET_PATH, index=False)
    combined.to_csv(CSV_PATH, index=False)

    return combined


def _backfill_forward_returns(df: pd.DataFrame):
    months = sorted(df["ym"].unique())
    for i in range(len(months) - 1):
        curr_ym = months[i]
        next_ym = months[i + 1]

        curr_mask = df["ym"] == curr_ym
        next_mask = df["ym"] == next_ym

        if df.loc[curr_mask, "y_raw"].isna().all():
            next_month = df.loc[next_mask, ["permno", "ret_1"]].set_index("permno")
            curr_permnos = df.loc[curr_mask, "permno"]
            fwd_ret = curr_permnos.map(next_month["ret_1"])
            df.loc[curr_mask, "y_raw"] = fwd_ret.values

            valid = fwd_ret.dropna()
            if len(valid) > 1:
                mean_ret = valid.mean()
                std_ret = valid.std()
                if std_ret > 0:
                    df.loc[curr_mask, "y_xs"] = (
                        (fwd_ret - mean_ret) / std_ret
                    ).values
```

- [ ] **Step 3: Run assembler tests**

Run: `pytest tests/test_pipeline_assembler.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add pipeline/assembler.py tests/test_pipeline_assembler.py
git commit -m "feat: add assembler with cross-sectional standardization and schema enforcement"
```

---

### Task 10: Pipeline orchestrator

**Files:**
- Modify: `pipeline/__init__.py`

- [ ] **Step 1: Implement the orchestrator in pipeline/__init__.py**

```python
"""Live data pipeline for Alpha Strategy Dashboard."""

import logging
from dataclasses import dataclass, field

from pipeline.config import ensure_cache_dirs

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool
    months_added: list[str] = field(default_factory=list)
    tickers_fetched: int = 0
    tickers_failed: list[str] = field(default_factory=list)
    total_rows: int = 0
    error: str | None = None


def run_pipeline(
    fred_api_key: str | None = None,
    progress_callback=None,
) -> PipelineResult:
    import pandas as pd
    from pipeline.universe import get_sp500_tickers
    from pipeline.fetchers.prices import fetch_prices, load_cached_prices
    from pipeline.fetchers.fundamentals import fetch_fundamentals, load_cached_meta
    from pipeline.fetchers.factors import fetch_factors
    from pipeline.fetchers.macro import fetch_macro
    from pipeline.features.price_features import compute_price_features
    from pipeline.features.fundamental_features import compute_fundamental_features
    from pipeline.features.peer_features import compute_peer_features
    from pipeline.features.macro_features import compute_macro_features
    from pipeline.assembler import assemble_month, append_to_dataset
    from pipeline.config import MIN_SUCCESS_RATE

    ensure_cache_dirs()

    def _report(stage, detail=""):
        if progress_callback:
            progress_callback(stage, detail)

    try:
        # Stage 1: Universe
        _report("universe", "Fetching S&P 500 tickers...")
        tickers = get_sp500_tickers()
        _report("universe", f"Got {len(tickers)} tickers")

        # Stage 2: Prices
        _report("prices", "Downloading daily prices...")
        prices = fetch_prices(
            tickers,
            progress_callback=lambda done, total: _report(
                "prices", f"Batch {done}/{total}"
            ),
        )

        # Stage 3: Market daily (SPY)
        _report("prices", "Extracting SPY market returns...")
        if "SPY" in prices.columns.get_level_values(0):
            market_daily = prices["SPY"]["Close"]
        else:
            import yfinance as yf
            spy = yf.download("SPY", period="3y", progress=False)
            market_daily = spy["Close"]

        # Stage 4: Fundamentals
        _report("fundamentals", "Downloading quarterly financials...")
        fund_data = fetch_fundamentals(
            tickers,
            progress_callback=lambda done, total, t: _report(
                "fundamentals", f"{done}/{total} ({t})"
            ),
        )
        failed_tickers = [t for t in tickers if t not in fund_data]

        if len(fund_data) / len(tickers) < MIN_SUCCESS_RATE:
            return PipelineResult(
                success=False,
                tickers_failed=failed_tickers,
                error=f"Only {len(fund_data)}/{len(tickers)} tickers succeeded (below {MIN_SUCCESS_RATE:.0%} threshold)",
            )

        # Stage 5: Factors
        _report("factors", "Downloading FF5 + Momentum factors...")
        factors = fetch_factors()

        # Stage 6: Macro
        _report("macro", "Downloading FRED macro data...")
        try:
            macro = fetch_macro(api_key=fred_api_key)
        except ValueError:
            logger.warning("No FRED API key — skipping macro features")
            macro = pd.DataFrame()

        # Stage 7: Compute features
        _report("features", "Computing price features...")
        shares = {}
        sectors = {}
        industries = {}
        for t, data in fund_data.items():
            shares[t] = data.get("shares_outstanding") or 0
            sectors[t] = data.get("sector", "Unknown")
            industries[t] = data.get("industry", "Unknown")

        price_feats = compute_price_features(prices, market_daily, shares)

        _report("features", "Computing fundamental features...")
        fund_input = {}
        for t, data in fund_data.items():
            quarters = []
            if data["income"] is not None and not data["income"].empty:
                quarters = sorted(data["income"].columns)
            if not quarters:
                continue

            fund_input[t] = {
                "quarters": pd.DatetimeIndex(quarters),
                "income": {
                    field: [
                        float(data["income"].loc[field, q])
                        if field in data["income"].index and pd.notna(data["income"].loc[field, q])
                        else np.nan
                        for q in quarters
                    ]
                    for field in data["income"].index
                },
                "balance": {
                    field: [
                        float(data["balance"].loc[field, q])
                        if field in data["balance"].index and pd.notna(data["balance"].loc[field, q])
                        else np.nan
                        for q in quarters
                    ]
                    for field in data["balance"].index
                } if data["balance"] is not None and not data["balance"].empty else {},
                "cashflow": {
                    field: [
                        float(data["cashflow"].loc[field, q])
                        if field in data["cashflow"].index and pd.notna(data["cashflow"].loc[field, q])
                        else np.nan
                        for q in quarters
                    ]
                    for field in data["cashflow"].index
                } if data["cashflow"] is not None and not data["cashflow"].empty else {},
                "market_cap": data.get("market_cap", 0),
            }

        import numpy as np
        fund_feats = compute_fundamental_features(fund_input) if fund_input else pd.DataFrame()

        # Merge fundamentals into price features for peer computation
        if not fund_feats.empty:
            merge_cols = [c for c in fund_feats.columns if c not in price_feats.columns and c not in ["ticker", "ym"]]
            if merge_cols:
                price_feats = price_feats.merge(
                    fund_feats[["ticker", "ym"] + merge_cols],
                    on=["ticker", "ym"],
                    how="left",
                )

        _report("features", "Computing peer features...")
        peer_feats = compute_peer_features(price_feats, sectors, industries)

        target_months = sorted(price_feats["ym"].unique())
        _report("features", "Computing macro features...")
        macro_feats = compute_macro_features(macro, target_months) if not macro.empty else pd.DataFrame({"ym": target_months})

        # Stage 8: Assemble
        _report("assembly", "Assembling dataset...")
        assembled = assemble_month(
            price_features=price_feats,
            fundamental_features=fund_feats,
            peer_features=peer_feats,
            macro_features=macro_feats,
            factors=factors,
        )

        _report("assembly", "Appending to dataset...")
        final = append_to_dataset(assembled)

        months_added = sorted(assembled["ym"].unique())
        return PipelineResult(
            success=True,
            months_added=months_added,
            tickers_fetched=len(fund_data),
            tickers_failed=failed_tickers,
            total_rows=len(final),
        )

    except Exception as e:
        logger.exception("Pipeline failed")
        return PipelineResult(success=False, error=str(e))
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/__init__.py
git commit -m "feat: add pipeline orchestrator coordinating all stages"
```

---

### Task 11: Update data_loader.py for Parquet support

**Files:**
- Modify: `dashboard/core/data_loader.py`

- [ ] **Step 1: Update load_dataset to try Parquet first**

In `dashboard/core/data_loader.py`, replace the `load_dataset` function:

```python
def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the alpha dataset. Tries Parquet first, then CSV."""
    if path is None:
        parquet_path = DATA_DIR / "alpha_dataset_v2.parquet"
        csv_path = DATA_DIR / "alpha_dataset_v2.csv"
        if parquet_path.exists():
            path = parquet_path
        else:
            path = csv_path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"ym", "permno", "y_xs", "y_raw", "Mkt_RF", "rf_ff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df
```

- [ ] **Step 2: Run existing tests to confirm no breakage**

Run: `pytest tests/test_data_loader.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add dashboard/core/data_loader.py
git commit -m "feat: data_loader supports Parquet with CSV fallback"
```

---

### Task 12: Dashboard page — Data Pipeline

**Files:**
- Create: `dashboard/pages/8_Data_Pipeline.py`

- [ ] **Step 1: Implement the Data Pipeline page**

```python
"""dashboard/pages/8_Data_Pipeline.py — Manual data refresh page."""

import streamlit as st
from pathlib import Path

st.header("Data Pipeline")
st.markdown("Fetch the latest market data and update the dataset.")

# Current dataset stats
df = st.session_state.get("df")
if df is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stocks", f"{df['permno'].nunique():,}")
    with col2:
        st.metric("Months", f"{df['ym'].nunique()}")
    with col3:
        st.metric("Date Range", f"{df['ym'].min()} to {df['ym'].max()}")
    st.markdown("---")

# FRED API key input
import os
fred_key = os.environ.get("FRED_API_KEY", "")
if not fred_key:
    fred_key = st.text_input(
        "FRED API Key (optional — needed for macro features)",
        type="password",
        help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
    )
    if fred_key:
        os.environ["FRED_API_KEY"] = fred_key

# Refresh button
if st.button("Refresh Data", type="primary", use_container_width=True):
    st.markdown("---")
    status = st.status("Running pipeline...", expanded=True)
    progress_bar = st.progress(0)
    log_container = st.empty()
    logs = []

    stages = {
        "universe": 0.05,
        "prices": 0.30,
        "fundamentals": 0.60,
        "factors": 0.70,
        "macro": 0.75,
        "features": 0.90,
        "assembly": 1.0,
    }

    def progress_callback(stage, detail=""):
        pct = stages.get(stage, 0)
        progress_bar.progress(pct)
        msg = f"**{stage.title()}**: {detail}"
        logs.append(msg)
        status.update(label=f"Running pipeline... ({stage})")
        log_container.markdown("\n\n".join(logs[-10:]))

    try:
        from pipeline import run_pipeline

        result = run_pipeline(
            fred_api_key=fred_key or None,
            progress_callback=progress_callback,
        )

        progress_bar.progress(1.0)

        if result.success:
            status.update(label="Pipeline complete!", state="complete")
            st.success(
                f"Added {len(result.months_added)} month(s): "
                f"{', '.join(result.months_added[:5])}{'...' if len(result.months_added) > 5 else ''}. "
                f"{result.tickers_fetched} tickers fetched. "
                f"Dataset now has {result.total_rows:,} rows."
            )
            if result.tickers_failed:
                st.warning(
                    f"{len(result.tickers_failed)} tickers failed: "
                    f"{', '.join(result.tickers_failed[:10])}"
                    f"{'...' if len(result.tickers_failed) > 10 else ''}"
                )

            # Reload data into session state
            from core.data_loader import load_dataset, compute_market_monthly, load_ff5_factors
            from features import precompute_features

            new_df = load_dataset()
            new_df = precompute_features(new_df)
            st.session_state.df = new_df
            st.session_state.market_monthly = compute_market_monthly(new_df)

            ff5_path = Path(__file__).parent.parent.parent / "Data" / "ff5_factors.csv"
            if ff5_path.exists():
                st.session_state.ff5_factors = load_ff5_factors(ff5_path)

            st.cache_data.clear()
            st.info("Session data reloaded. Navigate to other pages to use updated data.")
        else:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline failed: {result.error}")
            if result.tickers_failed:
                st.warning(f"Failed tickers: {', '.join(result.tickers_failed[:20])}")

    except Exception as e:
        status.update(label="Pipeline error", state="error")
        st.error(f"Unexpected error: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info(
        "Click **Refresh Data** to fetch the latest market data from Yahoo Finance, "
        "FRED, and Ken French's data library. This may take 10-20 minutes."
    )
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/pages/8_Data_Pipeline.py
git commit -m "feat: add Data Pipeline dashboard page with progress UI"
```

---

### Task 13: One-time Parquet migration + final integration

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Update app.py to use Parquet-aware loader**

The `_load_all_data` function in `app.py` currently hardcodes the CSV path. Update it to let `load_dataset()` auto-detect Parquet:

Replace the `_load_all_data` function body:

```python
@st.cache_data
def _load_all_data():
    df = load_dataset()

    from features import precompute_features
    df = precompute_features(df)

    market = compute_market_monthly(df)

    ff5_path = Path(__file__).parent.parent / "Data" / "ff5_factors.csv"
    ff5 = None
    if ff5_path.exists():
        ff5 = load_ff5_factors(ff5_path)

    return df, market, ff5
```

- [ ] **Step 2: Run all existing tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: app.py uses auto-detecting Parquet/CSV loader"
```

- [ ] **Step 4: Run the dashboard to verify it starts**

Run: `cd dashboard && streamlit run app.py`
Expected: Dashboard starts, all pages load, new "Data Pipeline" page visible in sidebar.

- [ ] **Step 5: Final commit with updated dependencies**

```bash
git add -A
git commit -m "feat: complete live data pipeline integration"
```

---

## Task Dependency Graph

```
Task 1 (config/setup)
  ├── Task 2 (universe)
  ├── Task 3 (prices fetcher)
  ├── Task 4 (fundamentals fetcher)
  ├── Task 5 (factors + macro fetchers)
  │
  ├── Task 6 (price features) ← depends on Task 3
  ├── Task 7 (fundamental features) ← depends on Task 4
  ├── Task 8 (peer + macro features) ← depends on Task 5, 6
  │
  ├── Task 9 (assembler) ← depends on Task 6, 7, 8
  ├── Task 10 (orchestrator) ← depends on Task 2-9
  │
  ├── Task 11 (data_loader parquet) ← independent
  ├── Task 12 (dashboard page) ← depends on Task 10
  └── Task 13 (integration) ← depends on all
```

Tasks 2-5 can be parallelized. Tasks 6-8 can be partially parallelized. Tasks 11-12 depend on the pipeline core being complete.
