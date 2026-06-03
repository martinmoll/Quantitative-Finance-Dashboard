"""Compute price-based features from daily OHLCV data."""

import numpy as np
import pandas as pd


def compute_price_features(
    prices: pd.DataFrame,
    market_daily: pd.Series,
    shares_outstanding: dict[str, float],
    dividends: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    tickers = prices.columns.get_level_values(0).unique()
    if market_daily.index.tz is not None:
        market_daily = market_daily.tz_localize(None)
    market_ret = market_daily.pct_change()
    all_rows = []

    for ticker in tickers:
        try:
            df = prices[ticker].copy()
        except KeyError:
            continue

        if df.empty or "Close" not in df.columns:
            continue

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        close = df["Close"].dropna()
        volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
        daily_ret = close.pct_change()

        month_ends = close.resample("ME").last().dropna()
        shares = shares_outstanding.get(ticker, np.nan)

        for i in range(len(month_ends)):
            me_date = month_ends.index[i]
            ym = me_date.strftime("%Y-%m")
            me_price = month_ends.iloc[i]

            mask_12m = (close.index <= me_date) & (
                close.index > me_date - pd.DateOffset(months=12)
            )
            daily_ret_12m = daily_ret[mask_12m]
            close_12m = close[mask_12m]

            row = {"ticker": ticker, "ym": ym, "date": me_date.strftime("%Y-%m-%d")}

            if i >= 1:
                row["ret_1"] = me_price / month_ends.iloc[i - 1] - 1
            if i >= 12:
                row["ret_2_12"] = month_ends.iloc[i - 1] / month_ends.iloc[i - 12] - 1
            if i >= 6:
                row["ret_2_6"] = month_ends.iloc[i - 1] / month_ends.iloc[i - 6] - 1
            if i >= 36:
                row["ret_13_36"] = month_ends.iloc[i - 12] / month_ends.iloc[i - 36] - 1

            if len(daily_ret_12m) > 20:
                row["vol_12m"] = daily_ret_12m.std() * np.sqrt(252)
                row["max_ret_12m"] = daily_ret_12m.max()
                row["skew_12m"] = daily_ret_12m.skew()
                row["kurt_12m"] = daily_ret_12m.kurtosis()

            if len(daily_ret_12m) > 60:
                aligned = pd.DataFrame({
                    "stock": daily_ret_12m,
                    "market": market_ret.reindex(daily_ret_12m.index),
                }).dropna()
                if len(aligned) > 30:
                    cov = np.cov(aligned["stock"], aligned["market"])
                    mkt_var = cov[1, 1]
                    if mkt_var > 0:
                        row["beta"] = cov[0, 1] / mkt_var
                        residuals = aligned["stock"] - row["beta"] * aligned["market"]
                        row["ivol"] = residuals.std() * np.sqrt(252)

            if not np.isnan(shares) and shares > 0:
                me_val = me_price * shares
                row["log_me"] = np.log(me_val) if me_val > 0 else np.nan
                row["me"] = me_val
                row["prc_abs"] = abs(me_price)
            else:
                row["log_me"] = np.nan
                row["me"] = np.nan
                row["prc_abs"] = abs(me_price)

            vol_month = volume[
                (volume.index <= me_date)
                & (volume.index > me_date - pd.DateOffset(months=1))
            ]
            if len(vol_month) > 0 and not np.isnan(shares) and shares > 0:
                row["turnover"] = vol_month.mean() / shares

            if len(close_12m) > 0:
                row["prc_52w_high"] = me_price / close_12m.max()

            first_date = close.index[0]
            row["age"] = (me_date.year - first_date.year) * 12 + (me_date.month - first_date.month)

            if dividends and ticker in dividends:
                div_series = dividends[ticker]
                if div_series.index.tz is not None:
                    div_series = div_series.tz_localize(None)
                div_12m = div_series[
                    (div_series.index <= me_date)
                    & (div_series.index > me_date - pd.DateOffset(months=12))
                ]
                div_sum = div_12m.sum()
                if me_price > 0:
                    row["dp_ratio"] = div_sum / me_price

            _add_technicals(row, close, volume, me_date)

            if len(daily_ret_12m) > 20 and len(volume) > 0:
                vol_aligned = volume.reindex(daily_ret_12m.index).dropna()
                price_aligned = close.reindex(vol_aligned.index)
                dollar_vol = vol_aligned * price_aligned
                valid = dollar_vol > 0
                if valid.sum() > 20:
                    illiq_vals = daily_ret_12m.reindex(dollar_vol[valid].index).abs() / dollar_vol[valid]
                    row["illiq_12m"] = illiq_vals.mean()

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def _add_technicals(row: dict, close: pd.Series, volume: pd.Series, me_date):
    mask = close.index <= me_date
    c = close[mask]

    if len(c) < 30:
        return

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    if len(rsi.dropna()) > 0:
        row["rsi_14"] = rsi.iloc[-1]

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal
    if len(macd_hist.dropna()) > 0:
        row["macd_hist"] = macd_hist.iloc[-1]

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    if len(sma20.dropna()) > 0 and std20.iloc[-1] > 0:
        row["bb_position"] = (c.iloc[-1] - sma20.iloc[-1]) / (2 * std20.iloc[-1])

    for days, name in [(126, "prc_ma6"), (252, "prc_ma12"), (504, "prc_ma24")]:
        if len(c) >= days:
            sma = c.rolling(days).mean()
            if pd.notna(sma.iloc[-1]) and sma.iloc[-1] > 0:
                row[name] = c.iloc[-1] / sma.iloc[-1]

    for days, name in [(63, "roc_3"), (126, "roc_6")]:
        if len(c) > days:
            row[name] = c.iloc[-1] / c.iloc[-days] - 1

    if len(volume) > 0:
        vol_masked = volume[volume.index <= me_date]
        if len(vol_masked) > 63:
            curr_month = vol_masked.iloc[-21:].mean() if len(vol_masked) >= 21 else vol_masked.mean()
            trail_3m = vol_masked.iloc[-63:].mean()
            if trail_3m > 0:
                row["vol_ratio"] = curr_month / trail_3m
