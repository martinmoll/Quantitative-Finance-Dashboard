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
