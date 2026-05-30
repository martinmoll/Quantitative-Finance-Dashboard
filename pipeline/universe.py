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
