"""Market series + CAPM factor frame for regions without a Fama-French factor set.

US uses Ken French's FF5 (see factors.py). Other regions (e.g. Norway) have no
published regional factor set, so we build a single-factor CAPM frame against a
local market proxy: the same factor-frame shape the assembler already merges, with
the four style factors left NaN.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from pipeline.config import FACTORS_CACHE, ensure_cache_dirs

logger = logging.getLogger(__name__)


def fetch_market_daily(ticker: str, period: str = "3y") -> pd.Series:
    """Daily adjusted close for a market proxy (e.g. SPY, ^OSEAX), tz-naive."""
    data = yf.download(
        ticker, period=period, interval="1d", auto_adjust=True, progress=False
    )
    if data is None or data.empty:
        raise RuntimeError(f"Market series download failed for {ticker}")
    close = data["Close"]
    series = close.squeeze() if hasattr(close, "squeeze") else close
    if hasattr(series, "index") and series.index.tz is not None:
        series = series.tz_localize(None)
    return series


def _fetch_rf_monthly(fred_api_key: str | None, series_id: str | None) -> pd.Series | None:
    """Monthly risk-free rate (decimal) from FRED, or None to fall back to 0."""
    if not fred_api_key or not series_id:
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key)
        s = fred.get_series(series_id, observation_start="2010-01-01")
        s = s / 100.0 / 12.0  # percent-per-annum -> monthly decimal
        s.index = pd.DatetimeIndex(s.index).strftime("%Y-%m")
        return s[~s.index.duplicated(keep="last")]
    except Exception as e:  # noqa: BLE001 - rf is optional; degrade to 0
        logger.warning("Risk-free fetch failed (%s); using rf=0", e)
        return None


def build_capm_factors(
    market_daily: pd.Series, rf_monthly: pd.Series | None = None
) -> pd.DataFrame:
    """CAPM factor frame indexed by 'ym' (YYYY-MM).

    Columns match the US factor frame: Mkt_RF, SMB, HML, RMW, CMA, Mom, rf_ff,
    spy_ret. Only the market and risk-free are populated; the style factors are NaN.
    """
    monthly_close = market_daily.resample("ME").last()
    mkt_ret = monthly_close.pct_change()
    ym = pd.DatetimeIndex(mkt_ret.index).strftime("%Y-%m")

    df = pd.DataFrame({"spy_ret": np.asarray(mkt_ret.values, dtype=float)}, index=ym)
    df.index.name = "ym"

    if rf_monthly is not None:
        df["rf_ff"] = df.index.map(rf_monthly.to_dict())
    else:
        df["rf_ff"] = 0.0
    df["rf_ff"] = df["rf_ff"].fillna(0.0)

    df["Mkt_RF"] = df["spy_ret"] - df["rf_ff"]
    for col in ["SMB", "HML", "RMW", "CMA", "Mom"]:
        df[col] = np.nan

    return df.dropna(subset=["spy_ret"])


def fetch_capm_factors(
    market_daily: pd.Series,
    fred_api_key: str | None = None,
    rf_series_id: str | None = None,
) -> pd.DataFrame:
    ensure_cache_dirs()
    rf = _fetch_rf_monthly(fred_api_key, rf_series_id)
    factors = build_capm_factors(market_daily, rf)
    factors.to_parquet(FACTORS_CACHE / "capm_factors.parquet")
    return factors
