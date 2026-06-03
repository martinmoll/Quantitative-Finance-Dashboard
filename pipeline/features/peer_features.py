"""Compute peer/sector/group relative features."""

import numpy as np
import pandas as pd


def compute_peer_features(
    price_features: pd.DataFrame,
    sectors: dict[str, str],
    industries: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = price_features.copy()
    df["sector"] = df["ticker"].map(sectors).fillna("Unknown")
    if industries:
        df["industry"] = df["ticker"].map(industries).fillna("Unknown")
    else:
        df["industry"] = df["sector"]

    result = pd.DataFrame(index=df.index)
    result["ticker"] = df["ticker"].values
    result["ym"] = df["ym"].values
    result["sector"] = df["sector"].values
    result["industry"] = df["industry"].values

    if "ret_1" not in df.columns:
        return result

    # Sector-level
    result["sector_ret_avg"] = df.groupby(["ym", "sector"])["ret_1"].transform("mean")
    result["sector_ret_dispersion"] = df.groupby(["ym", "sector"])["ret_1"].transform("std")
    result["ret_vs_sector"] = df["ret_1"] - result["sector_ret_avg"]
    result["peer_ret_1"] = result["sector_ret_avg"]

    result["mkt_ret_dispersion"] = df.groupby("ym")["ret_1"].transform("std")

    # Sector momentum (lagged)
    sector_ret_pivot = df.groupby(["ym", "sector"])["ret_1"].mean().unstack()
    sector_mom_lag = sector_ret_pivot.shift(1).stack()
    sector_mom_lag.name = "sector_mom_lag1"
    sector_mom_df = sector_mom_lag.reset_index()
    sector_mom_df.columns = ["ym", "sector", "sector_mom_lag1"]
    df = df.merge(sector_mom_df, on=["ym", "sector"], how="left")
    result["sector_mom_lag1"] = df["sector_mom_lag1"].values

    result["sector_rel_mom"] = result["sector_ret_avg"] - result["mkt_ret_dispersion"]

    # Industry-level
    result["ind_mom"] = df.groupby(["ym", "industry"])["ret_1"].transform("mean")
    result["ind_dispersion"] = df.groupby(["ym", "industry"])["ret_1"].transform("std")
    result["ret_vs_ind"] = df["ret_1"] - result["ind_mom"]

    # Size quintile features
    if "log_me" in df.columns:
        df["size_q"] = df.groupby("ym")["log_me"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
            if len(x) >= 5 else pd.Series(0, index=x.index)
        )
        result["size_peer_ret"] = df.groupby(["ym", "size_q"])["ret_1"].transform("mean")
        result["size_grp_disp"] = df.groupby(["ym", "size_q"])["ret_1"].transform("std")

        size_pivot = df.groupby(["ym", "size_q"])["ret_1"].mean().unstack()
        size_lag = size_pivot.shift(1).stack()
        size_lag.name = "size_grp_mom"
        size_df = size_lag.reset_index()
        size_df.columns = ["ym", "size_q", "size_grp_mom"]
        df = df.merge(size_df, on=["ym", "size_q"], how="left")
        result["size_grp_mom"] = df["size_grp_mom"].values

    # Beta quintile features
    if "beta" in df.columns:
        df["beta_q"] = df.groupby("ym")["beta"].transform(
            lambda x: pd.qcut(x.clip(-5, 5).rank(method="first"), 5, labels=False, duplicates="drop")
            if len(x) >= 5 else pd.Series(0, index=x.index)
        )
        result["beta_grp_ret"] = df.groupby(["ym", "beta_q"])["ret_1"].transform("mean")

    # Value quintile features
    if "bm" in df.columns:
        df["val_q"] = df.groupby("ym")["bm"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
            if len(x) >= 5 else pd.Series(0, index=x.index)
        )
        result["val_peer_ret"] = df.groupby(["ym", "val_q"])["ret_1"].transform("mean")

        val_pivot = df.groupby(["ym", "val_q"])["ret_1"].mean().unstack()
        val_lag = val_pivot.shift(1).stack()
        val_lag.name = "val_grp_mom"
        val_df = val_lag.reset_index()
        val_df.columns = ["ym", "val_q", "val_grp_mom"]
        df = df.merge(val_df, on=["ym", "val_q"], how="left")
        result["val_grp_mom"] = df["val_grp_mom"].values

    # BM relative signals
    if "bm" in df.columns:
        result["bm_vs_sector"] = df["bm"] - df.groupby(["ym", "sector"])["bm"].transform("mean")
        if "size_q" in df.columns:
            result["bm_vs_size"] = df["bm"] - df.groupby(["ym", "size_q"])["bm"].transform("mean")

    # Leader return
    if "me" in df.columns:
        leader_df = df.loc[df.groupby(["ym", "sector"])["me"].idxmax()][["ym", "sector", "ret_1"]].copy()
        leader_df = leader_df.rename(columns={"ret_1": "leader_ret"})
        df = df.merge(leader_df, on=["ym", "sector"], how="left")
        result["leader_ret"] = df["leader_ret"].values

        leader_pivot = leader_df.pivot_table(index="ym", columns="sector", values="leader_ret")
        leader_lag = leader_pivot.shift(1).stack()
        leader_lag.name = "leader_ret_lag1"
        leader_lag_df = leader_lag.reset_index()
        leader_lag_df.columns = ["ym", "sector", "leader_ret_lag1"]
        df = df.merge(leader_lag_df, on=["ym", "sector"], how="left")
        result["leader_ret_lag1"] = df["leader_ret_lag1"].values

    # Peer earnings surprise
    if "sue" in df.columns:
        result["peer_sue"] = df.groupby(["ym", "sector"])["sue"].transform("mean")
        result["ind_sue"] = df.groupby(["ym", "industry"])["sue"].transform("mean")

    # Pre-built interaction features
    if "ret_2_12" in df.columns and "log_me" in df.columns:
        result["mom_x_size"] = df["ret_2_12"] * df["log_me"]
    if "bm" in df.columns and "gpa" in df.columns:
        result["val_x_prof"] = df["bm"] * df["gpa"]
    if "ret_2_12" in df.columns and "vol_12m" in df.columns:
        result["mom_x_vol"] = df["ret_2_12"] * df["vol_12m"]

    # Momentum-derived features using pivot tables
    _add_lagged_diffs(df, result)

    return result


def _add_lagged_diffs(df: pd.DataFrame, result: pd.DataFrame):
    months = sorted(df["ym"].unique())
    if len(months) < 2:
        return

    ym_ticker_idx = df.set_index(["ym", "ticker"])

    for col, out_col in [
        ("ret_2_12", "mom_accel"),
        ("ret_2_12", "mom_of_mom"),
        ("vol_12m", "delta_vol"),
        ("bm", "delta_bm"),
    ]:
        if col not in df.columns:
            continue
        pivot = df.pivot_table(index="ym", columns="ticker", values=col)
        diff = pivot.diff(1)
        melted = diff.stack().reset_index()
        melted.columns = ["ym", "ticker", out_col]
        merged = df[["ticker", "ym"]].merge(melted, on=["ym", "ticker"], how="left")
        result[out_col] = merged[out_col].values

    # Return streak
    if "ret_1" in df.columns:
        pivot = df.pivot_table(index="ym", columns="ticker", values="ret_1")
        positive = (pivot > 0).astype(int)
        streak = positive.copy()
        for i in range(1, len(streak)):
            streak.iloc[i] = (streak.iloc[i - 1] + 1) * positive.iloc[i]
        melted = streak.stack().reset_index()
        melted.columns = ["ym", "ticker", "ret_streak"]
        merged = df[["ticker", "ym"]].merge(melted, on=["ym", "ticker"], how="left")
        result["ret_streak"] = merged["ret_streak"].values

    # Vol group return
    if "vol_12m" in df.columns:
        df_temp = df.copy()
        df_temp["vol_q"] = df_temp.groupby("ym")["vol_12m"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
            if len(x) >= 5 else pd.Series(0, index=x.index)
        )
        result["vol_grp_ret"] = df_temp.groupby(["ym", "vol_q"])["ret_1"].transform("mean")

    # Ind-size features (interaction group)
    result["ind_size_ret"] = np.nan
    result["ind_size_mom"] = np.nan

    # Interaction features needing macro data (filled by assembler)
    result["mom_x_unc"] = np.nan
    result["val_x_finunc"] = np.nan
    result["beta_x_disp"] = np.nan
