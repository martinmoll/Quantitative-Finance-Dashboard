"""Fetch index constituent lists (S&P 500, Nasdaq-100)."""

import io
import urllib.request
import pandas as pd
from pathlib import Path

_CACHE_DIR = Path(__file__).parent / "cache"

UNIVERSES = {
    "S&P 500": "sp500",
    "Nasdaq-100": "nasdaq100",
    "S&P 500 + Nasdaq-100": "sp500_nasdaq100",
}


def get_tickers(universe: str = "S&P 500", use_cache: bool = True) -> list[str]:
    key = UNIVERSES.get(universe, universe)

    if key == "sp500":
        return _get_sp500(use_cache)
    elif key == "nasdaq100":
        return _get_nasdaq100(use_cache)
    elif key == "sp500_nasdaq100":
        sp = _get_sp500(use_cache)
        nq = _get_nasdaq100(use_cache)
        return sorted(set(sp + nq))
    else:
        return _get_sp500(use_cache)


def get_sp500_tickers(use_cache: bool = True) -> list[str]:
    return _get_sp500(use_cache)


def _read_wiki(url: str) -> list[pd.DataFrame]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req).read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def _get_sp500(use_cache: bool) -> list[str]:
    cache_path = _CACHE_DIR / "sp500_tickers.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path)
        return df["ticker"].tolist()

    tables = _read_wiki(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    tickers = sorted(set(tickers))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers


def _get_nasdaq100(use_cache: bool) -> list[str]:
    cache_path = _CACHE_DIR / "nasdaq100_tickers.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path)
        return df["ticker"].tolist()

    tables = _read_wiki(
        "https://en.wikipedia.org/wiki/Nasdaq-100"
    )
    for table in tables:
        if "Ticker" in table.columns:
            tickers = table["Ticker"].str.replace(".", "-", regex=False).tolist()
            break
        elif "Symbol" in table.columns:
            tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
            break
    else:
        raise RuntimeError("Could not find ticker column in Nasdaq-100 Wikipedia table")

    tickers = sorted(set(tickers))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers
