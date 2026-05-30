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
