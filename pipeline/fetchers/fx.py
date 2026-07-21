"""USD/NOK exchange-rate series for currency-risk conversion.

A Norwegian investor buying USD-denominated stocks earns the stock's local (USD)
return plus the move in USDNOK. This fetches the monthly USDNOK return so a USD
backtest can be restated in NOK — see dashboard/core/currency.py for the maths.
"""

import pandas as pd
import yfinance as yf


def fetch_usdnok_monthly(period: str = "max") -> pd.Series:
    """Monthly USDNOK return (decimal), indexed by 'YYYY-MM'.

    Yahoo's 'NOK=X' quotes USD/NOK (kroner per dollar), so a positive return means
    the dollar strengthened against the krone — a tailwind for an unhedged NOK
    investor holding USD assets.
    """
    data = yf.download(
        "NOK=X", period=period, interval="1d", auto_adjust=True, progress=False
    )
    if data is None or data.empty:
        raise RuntimeError("USDNOK download failed for NOK=X")
    close = data["Close"]
    series = close.squeeze() if hasattr(close, "squeeze") else close
    monthly = series.resample("ME").last()
    fx_ret = monthly.pct_change()
    fx_ret.index = pd.DatetimeIndex(fx_ret.index).strftime("%Y-%m")
    return fx_ret.dropna()
