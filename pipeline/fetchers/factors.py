"""Fama-French 5 factors + Momentum from Ken French library."""

import io
import logging
import zipfile
import urllib.error
import urllib.request

import pandas as pd

from pipeline.config import FACTORS_CACHE, ensure_cache_dirs

logger = logging.getLogger(__name__)

_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Momentum_Factor_CSV.zip"
)


def _download_french_csv(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8")

    lines = raw.strip().split("\n")

    # Find the header row: first line where the next line starts with a digit (date)
    header_idx = None
    for i in range(len(lines) - 1):
        next_stripped = lines[i + 1].strip()
        if next_stripped and next_stripped[0].isdigit() and "," in next_stripped:
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError(f"Could not find header row in {url}")

    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = df["date"].astype(str).str.strip()
    df = df[df["date"].str.len() == 6]
    df["date"] = df["date"].str[:4] + "-" + df["date"].str[4:6]
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return df


def fetch_factors() -> pd.DataFrame:
    ensure_cache_dirs()

    try:
        ff5 = _download_french_csv(_FF5_URL)
        ff5.index.name = "ym"

        mom = _download_french_csv(_MOM_URL)
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
    except (OSError, urllib.error.URLError) as e:
        logger.warning("Factor download failed (%s), trying local cache...", e)
        cached = load_cached_factors()
        if cached is not None:
            logger.info("Using cached factors from %s", FACTORS_CACHE / "ff5_mom.parquet")
            return cached
        raise RuntimeError(
            f"Cannot download FF5 factors and no local cache exists. "
            f"Check your internet connection and retry. Original error: {e}"
        ) from e


def load_cached_factors() -> pd.DataFrame | None:
    cache_path = FACTORS_CACHE / "ff5_mom.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None
