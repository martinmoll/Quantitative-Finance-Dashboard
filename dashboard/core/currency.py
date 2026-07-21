"""Restate a USD strategy return series in NOK (unhedged).

A Norwegian investor funding USD purchases with kroner earns both the stock's
local return and the USDNOK move. This is the compounded, unhedged conversion:

    r_nok = (1 + r_usd) * (1 + r_fx) - 1

where r_fx is the monthly USDNOK return (see pipeline/fetchers/fx.py). Currency
hedging and full FX risk attribution are tracked in FUTURE_IMPROVEMENTS.md.
"""

import pandas as pd


def to_nok_unhedged(returns: pd.Series, fx_ret: pd.Series) -> pd.Series:
    """Compound each monthly USD return with the same month's USDNOK return.

    Months without an FX observation are dropped rather than assumed flat, so the
    converted series never silently understates currency risk.
    """
    fx = fx_ret.reindex(returns.index)
    nok = (1 + returns) * (1 + fx) - 1
    return nok.dropna()
