"""Performance metrics, IC analysis, Fundamental Law, KS test, alpha decay."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp


def compute_performance_metrics(returns: pd.Series) -> dict:
    s = returns.dropna()
    if len(s) == 0:
        return {k: np.nan for k in
                ["SR", "Ann Return", "Ann Vol", "MDD", "Calmar", "Total Return"]}

    ann_ret = s.mean() * 12
    ann_vol = s.std() * np.sqrt(12)
    sr = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + s).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
    total_ret = cum.iloc[-1] - 1

    return {
        "SR": round(float(sr), 4),
        "Ann Return": float(ann_ret),
        "Ann Vol": float(ann_vol),
        "MDD": float(mdd),
        "Calmar": round(float(calmar), 4) if not np.isnan(calmar) else np.nan,
        "Total Return": float(total_ret),
    }


def compute_ic_stats(ic: pd.Series) -> dict:
    ic_clean = ic.dropna()
    if len(ic_clean) == 0:
        return {"mean_ic": np.nan, "ic_tstat": np.nan, "icir": np.nan, "hit_rate": np.nan}

    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    n = len(ic_clean)
    ic_tstat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    hit_rate = (ic_clean > 0).mean()

    return {
        "mean_ic": float(mean_ic),
        "ic_tstat": float(ic_tstat),
        "icir": float(icir),
        "hit_rate": float(hit_rate),
    }


def fundamental_law(
    ic_mean: float,
    K: int,
    rebal_freq: int = 12,
    sr_target: float = 1.0,
    tc_bps: float = 10.0,
) -> dict:
    BR_nominal = K * rebal_freq
    IR_upper = ic_mean * np.sqrt(BR_nominal) if BR_nominal > 0 else np.nan
    BR_implied = (IR_upper / ic_mean) ** 2 if ic_mean != 0 else np.nan

    tc_annual = tc_bps / 10000 * 2 * rebal_freq
    port_vol = 0.15
    Cost_SR = tc_annual / port_vol

    IC_required = (
        (sr_target + Cost_SR) / np.sqrt(BR_nominal)
        if BR_nominal > 0 else np.nan
    )

    return {
        "BR_nominal": int(BR_nominal),
        "IR_upper_bound": float(IR_upper) if not np.isnan(IR_upper) else np.nan,
        "BR_implied": float(BR_implied) if not np.isnan(BR_implied) else np.nan,
        "IC_required": float(IC_required) if not np.isnan(IC_required) else np.nan,
        "Cost_SR": float(Cost_SR),
    }


def feature_ic(X: pd.DataFrame, y_realized: pd.Series) -> pd.Series:
    results = {}
    for col in X.columns:
        valid = pd.DataFrame({"x": X[col], "y": y_realized}).dropna()
        if len(valid) > 10:
            results[col] = spearmanr(valid["x"], valid["y"])[0]
        else:
            results[col] = np.nan
    return pd.Series(results).sort_values(ascending=False)


def ks_test(
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    threshold: float = 0.10,
) -> pd.DataFrame:
    results = []
    for col in X_train.columns:
        if col not in X_current.columns:
            continue
        train_vals = X_train[col].dropna()
        curr_vals = X_current[col].dropna()
        if len(train_vals) == 0 or len(curr_vals) == 0:
            continue
        D, pval = ks_2samp(train_vals, curr_vals)
        results.append({
            "feature": col,
            "D": float(D),
            "pval": float(pval),
            "flag": D > threshold,
        })
    return pd.DataFrame(results).sort_values("D", ascending=False).reset_index(drop=True)


def alpha_decay(
    predictions: dict[str, pd.DataFrame],
    horizons: list[int] | None = None,
) -> pd.Series:
    if horizons is None:
        horizons = list(range(1, 13))

    months = sorted(predictions.keys())
    results = {}

    for h in horizons:
        ic_vals = []
        for i, m in enumerate(months):
            target_idx = i + h - 1
            if target_idx >= len(months):
                break
            target_month = months[target_idx]
            pred_df = predictions[m]
            target_df = predictions[target_month]

            merged = pred_df[["permno", "pred"]].merge(
                target_df[["permno", "y_raw"]], on="permno", how="inner"
            )
            if len(merged) > 10:
                ic_vals.append(spearmanr(merged["pred"], merged["y_raw"])[0])

        results[h] = np.nanmean(ic_vals) if ic_vals else np.nan

    return pd.Series(results)


def compute_r2_oos(predictions: dict[str, pd.DataFrame]) -> float:
    """Out-of-sample R² (Campbell & Thompson 2008).

    R²_OOS = 1 - Σ(r_i - r̂_i)² / Σ(r_i - r̄)²

    where r̂_i are model forecasts and r̄ is the expanding-window
    cross-sectional mean of realized returns up to (but not including)
    each month.
    """
    months = sorted(predictions.keys())
    ss_res = 0.0
    ss_tot = 0.0
    cumulative_returns: list[float] = []

    for m in months:
        df = predictions[m]
        if "pred" not in df.columns or "y_raw" not in df.columns:
            continue
        valid = df[["pred", "y_raw"]].dropna()
        if len(valid) < 5:
            continue

        if len(cumulative_returns) < 10:
            cumulative_returns.extend(valid["y_raw"].tolist())
            continue

        mean_prev = np.mean(cumulative_returns)

        residuals = (valid["y_raw"] - valid["pred"]).values
        ss_res += float((residuals ** 2).sum())
        ss_tot += float(((valid["y_raw"].values - mean_prev) ** 2).sum())

        cumulative_returns.extend(valid["y_raw"].tolist())

    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def signal_staleness(
    turnover: pd.Series,
    threshold: float = 0.10,
    consecutive: int = 3,
) -> pd.DataFrame:
    below = turnover < threshold
    streak = below.astype(int)
    streak_count = streak.groupby((~below).cumsum()).cumsum()
    stale = streak_count >= consecutive

    return pd.DataFrame({
        "turnover": turnover,
        "below_threshold": below,
        "stale": stale,
    })
