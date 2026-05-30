"""Assemble features into final panel dataset matching the 229-column schema."""

import logging
import pandas as pd
import numpy as np

from pipeline.config import (
    PARQUET_PATH, CSV_PATH, RAW_FEATURE_COLS, RED_FEATURES,
    TARGET_COLS, FACTOR_COLS, DATA_DIR, XS_FEATURE_COLS,
)

logger = logging.getLogger(__name__)


def cross_sectional_standardize(
    df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        xs_col = f"{col}_xs"
        grouped = result.groupby("ym")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        result[xs_col] = (result[col] - mean) / std.replace(0, np.nan)
    return result


def fill_red_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in RED_FEATURES:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)
    return result


def validate_schema(df: pd.DataFrame) -> list[str]:
    required = {"permno", "ym", "y_xs", "y_raw", "Mkt_RF", "rf_ff"}
    return sorted(required - set(df.columns))


def assemble_month(
    price_features: pd.DataFrame,
    fundamental_features: pd.DataFrame | None,
    peer_features: pd.DataFrame,
    macro_features: pd.DataFrame,
    factors: pd.DataFrame,
    ticker_to_permno: dict[str, int] | None = None,
) -> pd.DataFrame:
    merged = price_features.copy()

    if fundamental_features is not None and not fundamental_features.empty:
        fund_cols = [c for c in fundamental_features.columns
                     if c not in ["ticker", "ym"] and c not in merged.columns]
        if fund_cols:
            merged = merged.merge(
                fundamental_features[["ticker", "ym"] + fund_cols],
                on=["ticker", "ym"],
                how="left",
            )

    peer_cols = [c for c in peer_features.columns
                 if c not in ["ticker", "ym", "sector", "industry"]
                 and c not in merged.columns]
    if peer_cols and len(peer_features) == len(merged):
        for col in peer_cols:
            merged[col] = peer_features[col].values

    if "sector" in peer_features.columns and "sector" not in merged.columns:
        merged["sector"] = peer_features["sector"].values
    if "industry" in peer_features.columns and "industry" not in merged.columns:
        merged["industry"] = peer_features["industry"].values

    if not macro_features.empty and "ym" in macro_features.columns:
        macro_cols = [c for c in macro_features.columns if c != "ym" and c not in merged.columns]
        if macro_cols:
            merged = merged.merge(macro_features[["ym"] + macro_cols], on="ym", how="left")

    merged = fill_red_features(merged)

    for col in RAW_FEATURE_COLS:
        if col not in merged.columns:
            merged[col] = 0.0

    merged = cross_sectional_standardize(merged, XS_FEATURE_COLS)

    for col in XS_FEATURE_COLS:
        xs_col = f"{col}_xs"
        if xs_col not in merged.columns:
            merged[xs_col] = 0.0

    if ticker_to_permno:
        merged["permno"] = merged["ticker"].map(ticker_to_permno)
    else:
        merged["permno"] = merged["ticker"].apply(lambda t: abs(hash(t)) % 100000)

    if not factors.empty:
        factor_monthly = factors.reset_index()
        if "ym" not in factor_monthly.columns and "date" in factor_monthly.columns:
            factor_monthly = factor_monthly.rename(columns={"date": "ym"})
        avail_factor_cols = [c for c in FACTOR_COLS if c in factor_monthly.columns]
        if avail_factor_cols:
            merged = merged.merge(
                factor_monthly[["ym"] + avail_factor_cols],
                on="ym", how="left",
            )

    for col in FACTOR_COLS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["y_raw"] = np.nan
    merged["y_xs"] = np.nan
    if "ret_1" in merged.columns and "rf_ff" in merged.columns:
        merged["ret_excess"] = merged["ret_1"] - merged["rf_ff"].fillna(0)
    else:
        merged["ret_excess"] = np.nan
    merged["ret_adj"] = merged["ret_excess"]

    merged["in_sp500"] = 1
    for col in ["exchcd", "siccd"]:
        if col not in merged.columns:
            merged[col] = 0

    return merged


def append_to_dataset(new_data: pd.DataFrame) -> pd.DataFrame:
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH)
    else:
        existing = pd.DataFrame()

    if existing.empty:
        combined = new_data
    else:
        new_months = set(new_data["ym"].unique())
        existing = existing[~existing["ym"].isin(new_months)]
        combined = pd.concat([existing, new_data], ignore_index=True)

    combined = combined.sort_values(["ym", "permno"]).reset_index(drop=True)

    _backfill_forward_returns(combined)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PARQUET_PATH, index=False)
    combined.to_csv(CSV_PATH, index=False)

    return combined


def _backfill_forward_returns(df: pd.DataFrame):
    months = sorted(df["ym"].unique())
    for i in range(len(months) - 1):
        curr_ym = months[i]
        next_ym = months[i + 1]

        curr_mask = df["ym"] == curr_ym
        next_mask = df["ym"] == next_ym

        if df.loc[curr_mask, "y_raw"].isna().all():
            next_month = df.loc[next_mask, ["permno", "ret_1"]].set_index("permno")
            if "ret_1" in next_month.columns:
                curr_permnos = df.loc[curr_mask, "permno"]
                fwd_ret = curr_permnos.map(next_month["ret_1"])
                df.loc[curr_mask, "y_raw"] = fwd_ret.values

                valid = fwd_ret.dropna()
                if len(valid) > 1:
                    mean_ret = valid.mean()
                    std_ret = valid.std()
                    if std_ret > 0:
                        df.loc[curr_mask, "y_xs"] = (
                            (fwd_ret - mean_ret) / std_ret
                        ).values
