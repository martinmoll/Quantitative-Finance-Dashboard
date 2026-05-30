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

    meta = {
        "ticker": ticker,
        "sector": data["sector"],
        "industry": data["industry"],
        "shares_outstanding": data["shares_outstanding"],
        "market_cap": data["market_cap"],
    }
    meta_path = FUNDAMENTALS_CACHE / f"{ticker}_meta.parquet"
    pd.DataFrame([meta]).to_parquet(meta_path, index=False)


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
