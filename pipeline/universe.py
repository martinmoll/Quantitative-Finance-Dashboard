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
    "Oslo Børs": "ose",
}

# Curated list of liquid Oslo Stock Exchange names (~OSEBX scale), with the
# Yahoo Finance ".OL" suffix. Hardcoded because the Oslo Wikipedia table is far
# less reliable than the S&P/Nasdaq ones; every ticker here was verified to
# return data from Yahoo Finance. The pipeline drops any that later fail to
# fetch, so a few stale names are harmless.
_OSE_TICKERS = [
    # OBX core + large caps
    "EQNR.OL", "DNB.OL", "AKRBP.OL", "TEL.OL", "MOWI.OL", "NHY.OL", "YAR.OL",
    "ORK.OL", "KOG.OL", "SALM.OL", "STB.OL", "GJF.OL", "SUBC.OL", "TGS.OL",
    "FRO.OL", "AKSO.OL", "BAKKA.OL", "SCATC.OL", "TOM.OL", "NOD.OL", "ELK.OL",
    "AKER.OL", "LSG.OL", "BWLPG.OL", "VAR.OL", "AUSS.OL", "OTL.OL", "KID.OL",
    "MPCC.OL", "RECSI.OL", "ENTRA.OL", "BWO.OL", "OET.OL", "HAFNI.OL",
    "WAWI.OL", "PROT.OL", "EPR.OL", "NORBT.OL",
    # Banks & finance
    "MING.OL", "NONG.OL", "B2I.OL", "ELMRA.OL", "ABG.OL",
    # Industrials & services
    "BRG.OL", "ATEA.OL", "VEI.OL", "MULTI.OL", "KIT.OL", "AFG.OL", "SNI.OL",
    "AUTO.OL", "ELO.OL", "NSKOG.OL", "KOA.OL",
    # Energy, shipping & oil services
    "OKEA.OL", "HAUTO.OL", "CADLR.OL", "DOFG.OL", "BORR.OL", "PEN.OL", "ODF.OL",
    # Renewables & cleantech
    "HEX.OL", "NEL.OL", "HPUR.OL", "ZAP.OL", "VOW.OL", "AGLX.OL",
    # Seafood, consumer & healthcare
    "GSF.OL", "NAS.OL", "MEDI.OL", "PHO.OL", "NYKD.OL",
]


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
    elif key == "ose":
        return _get_ose(use_cache)
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


def _get_ose(use_cache: bool) -> list[str]:
    cache_path = _CACHE_DIR / "ose_tickers.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path)
        return df["ticker"].tolist()

    tickers = sorted(set(_OSE_TICKERS))
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
