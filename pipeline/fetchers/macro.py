"""FRED macro data fetcher (VIX, yield curve, EPU, financial stress)."""

import logging
import pandas as pd

from pipeline.config import FRED_API_KEY, FRED_SERIES, MACRO_CACHE, ensure_cache_dirs

logger = logging.getLogger(__name__)

# Features that go empty when a FRED series is absent. Named here so a partial
# download says which model inputs it costs, instead of leaving the caller to
# find four all-NaN columns in the assembled panel.
_SERIES_FEATURES = {
    "vix": "macro_unc_1m, macro_unc_12m, mom_x_unc",
    "yield_curve_slope": "yield_curve_slope",
    "fin_stress": "fin_unc_1m, fin_unc_12m, val_x_finunc",
}


def _warn_if_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Name the requested series the frame does not carry.

    fetch_macro only raised when *every* series failed, so one failed series
    was a warning nobody read: the partial frame was cached and every later
    build reused it. The live cache lost vix and yield_curve_slope this way,
    which is why macro_unc_1m, macro_unc_12m and mom_x_unc have always been
    empty.
    """
    missing = [c for c in FRED_SERIES.values() if c not in df.columns]
    if missing:
        costs = [_SERIES_FEATURES[m] for m in missing if m in _SERIES_FEATURES]
        logger.warning(
            "FRED macro data is missing %s. Empty features: %s. Set "
            "FRED_API_KEY and rerun the pipeline to fill them.",
            ", ".join(missing), "; ".join(costs) or "none",
        )
    return df


def fetch_macro(api_key: str | None = None) -> pd.DataFrame:
    ensure_cache_dirs()
    key = api_key or FRED_API_KEY

    # No key: reuse a previous download if we have one, else ask for a key.
    if not key:
        cached = load_cached_macro()
        if cached is not None:
            logger.info("No FRED API key — using cached macro data.")
            return _warn_if_incomplete(cached)
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY environment variable "
            "or get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    try:
        from fredapi import Fred
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
        df.to_parquet(MACRO_CACHE / "fred_data.parquet")
        return _warn_if_incomplete(df)
    except Exception as e:
        # Network down, bad key, rate-limited: fall back to the last good
        # download (mirrors the FF5 factor fetcher).
        cached = load_cached_macro()
        if cached is not None:
            logger.warning("FRED fetch failed (%s) — using cached macro data.", e)
            return _warn_if_incomplete(cached)
        raise RuntimeError(
            f"FRED macro fetch failed and no local cache exists: {e}"
        ) from e


def load_cached_macro() -> pd.DataFrame | None:
    cache_path = MACRO_CACHE / "fred_data.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None
