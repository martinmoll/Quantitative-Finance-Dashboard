"""Compute macro uncertainty features from FRED data."""

import numpy as np
import pandas as pd


def compute_macro_features(
    macro_data: pd.DataFrame,
    target_months: list[str],
) -> pd.DataFrame:
    if macro_data.empty:
        return pd.DataFrame({"ym": target_months})

    monthly = macro_data.resample("ME").mean()
    monthly["ym"] = monthly.index.strftime("%Y-%m")

    rows = []
    for ym in target_months:
        row = {"ym": ym}

        ts = pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0)
        mask_60m = (monthly.index <= ts) & (
            monthly.index > ts - pd.DateOffset(months=60)
        )
        hist = monthly[mask_60m]

        if "vix" in monthly.columns:
            vix_current = monthly.loc[monthly["ym"] == ym, "vix"]
            if len(vix_current) > 0 and len(hist) > 12:
                vix_val = vix_current.iloc[0]
                vix_mean = hist["vix"].mean()
                vix_std = hist["vix"].std()
                row["macro_unc_1m"] = (
                    (vix_val - vix_mean) / vix_std if vix_std > 0 else 0.0
                )
                vix_12m = hist["vix"].iloc[-12:].mean() if len(hist) >= 12 else vix_val
                row["macro_unc_12m"] = (
                    (vix_12m - vix_mean) / vix_std if vix_std > 0 else 0.0
                )

        if "fin_stress" in monthly.columns:
            fs_current = monthly.loc[monthly["ym"] == ym, "fin_stress"]
            if len(fs_current) > 0:
                row["fin_unc_1m"] = fs_current.iloc[0]
                fs_12m = hist["fin_stress"].iloc[-12:].mean() if len(hist) >= 12 else fs_current.iloc[0]
                row["fin_unc_12m"] = fs_12m

        rows.append(row)

    return pd.DataFrame(rows)
