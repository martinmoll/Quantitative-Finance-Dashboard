"""Performance metrics, IC analysis, Fundamental Law, KS test, alpha decay."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp, norm


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
    if not results:
        return pd.DataFrame(columns=["feature", "D", "pval", "flag"])
    return pd.DataFrame(results).sort_values("D", ascending=False).reset_index(drop=True)


def feature_drift(
    train_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.10,
) -> pd.DataFrame:
    """KS drift per feature between a training window and the current month.

    NaNs are handled per feature inside ks_test, so a feature that is entirely
    NaN in one window is simply skipped. This avoids the trap of a row-wise
    dropna across all features, which collapses the whole panel to zero rows
    whenever any single feature is all-NaN (common in early months of a short
    panel, where long-lookback features have no history yet).
    """
    cols = [c for c in features
            if c in train_df.columns and c in current_df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "D", "pval", "flag"])
    return ks_test(train_df[cols], current_df[cols], threshold)


def recent_training_window(
    all_months: list[str],
    oos_start: str,
    retrain_freq: int,
    window_type: str,
    rolling_window: int | None,
    last_pred_month: str,
) -> list[str]:
    """Months the model was most recently trained on.

    This is the right baseline for drift detection: "drift" should mean the
    current data differs from what the model *actually learned*, not from the
    earliest history. Mirrors run_walk_forward's training-set selection for the
    last retrain — expanding uses everything before it, rolling uses the last
    ``rolling_window`` months. Returns an empty list if there is no pre-OOS
    history to baseline against (caller can fall back).
    """
    months = sorted(all_months)
    oos_months = [m for m in months if m >= oos_start]
    if not oos_months:
        return []
    retrain_schedule = oos_months[:: max(int(retrain_freq), 1)]
    candidates = [m for m in retrain_schedule if m <= last_pred_month]
    last_retrain = candidates[-1] if candidates else oos_months[0]
    cutoff = months.index(last_retrain)
    if window_type == "rolling" and rolling_window:
        return months[max(0, cutoff - int(rolling_window)):cutoff]
    return months[:cutoff]


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
    """Out-of-sample R² (Campbell & Thompson 2008), on the model's scale.

    The models forecast the cross-sectionally *standardized* return (y_xs,
    std ≈ 1), not raw returns (std ≈ 0.1). Comparing a standardized forecast
    directly to raw returns makes R² dominated by that ~9x scale gap rather than
    forecast skill — it reads hugely negative on every run. So realized returns
    are standardized within each month before the comparison, which is
    equivalent to evaluating the forecast against y_xs (what it trained on).

    R²_OOS = 1 - Σ(z_i - r̂_i)² / Σ(z_i - z̄)², where z is the within-month
    z-score of the realized return and z̄ the expanding-window mean (≈ 0).
    """
    months = sorted(predictions.keys())
    ss_res = 0.0
    ss_tot = 0.0
    cumulative: list[float] = []

    for m in months:
        df = predictions[m]
        if "pred" not in df.columns or "y_raw" not in df.columns:
            continue
        valid = df[["pred", "y_raw"]].dropna()
        if len(valid) < 5:
            continue

        r = valid["y_raw"].values
        sd = r.std()
        if sd == 0:
            continue
        z = (r - r.mean()) / sd  # within-month standardized realized return

        if len(cumulative) < 10:
            cumulative.extend(z.tolist())
            continue

        mean_prev = np.mean(cumulative)
        pred = valid["pred"].values
        ss_res += float(((z - pred) ** 2).sum())
        ss_tot += float(((z - mean_prev) ** 2).sum())

        cumulative.extend(z.tolist())

    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for the annualized Sharpe ratio."""
    s = returns.dropna().values
    if len(s) < 6:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "ci": ci}

    rng = np.random.default_rng(seed)
    n = len(s)
    boot_srs = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(s, size=n, replace=True)
        mu = sample.mean() * 12
        sigma = sample.std(ddof=1) * np.sqrt(12)
        boot_srs[i] = mu / sigma if sigma > 0 else np.nan

    alpha = (1 - ci) / 2
    lo, hi = float(np.nanpercentile(boot_srs, alpha * 100)), float(np.nanpercentile(boot_srs, (1 - alpha) * 100))
    point = float(np.nanmean(s) * 12 / (np.std(s, ddof=1) * np.sqrt(12))) if np.std(s, ddof=1) > 0 else np.nan

    return {"point": point, "lo": lo, "hi": hi, "ci": ci}


def bootstrap_alpha_ci(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for annualized FF5 alpha."""
    from core.risk import _ff5_available
    available = _ff5_available(ff5_factors)
    common = portfolio_returns.dropna().index.intersection(ff5_factors.dropna().index)
    if len(common) < 12:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "ci": ci}

    y = portfolio_returns.loc[common].values
    rf = ff5_factors.loc[common, "RF"].values if "RF" in ff5_factors.columns else np.zeros(len(common))
    y_excess = y - rf
    X = ff5_factors.loc[common, available].values
    X_const = np.column_stack([np.ones(len(common)), X])

    rng = np.random.default_rng(seed)
    n = len(common)
    boot_alphas = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_excess[idx]
        X_b = X_const[idx]
        try:
            beta = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
            boot_alphas[i] = beta[0] * 12
        except np.linalg.LinAlgError:
            boot_alphas[i] = np.nan

    alpha_frac = (1 - ci) / 2
    lo = float(np.nanpercentile(boot_alphas, alpha_frac * 100))
    hi = float(np.nanpercentile(boot_alphas, (1 - alpha_frac) * 100))

    try:
        beta_full = np.linalg.lstsq(X_const, y_excess, rcond=None)[0]
        point = float(beta_full[0] * 12)
    except np.linalg.LinAlgError:
        point = np.nan

    return {"point": point, "lo": lo, "hi": hi, "ci": ci}


def multiple_testing_hurdle(n_trials: int, alpha: float = 0.05) -> float:
    """Bonferroni-adjusted two-sided critical t-value for ``n_trials`` tests.

    Trying many strategy configurations and keeping the best inflates the
    false-positive rate: the naive |t| > 1.96 bar no longer controls the
    family-wise error at ``alpha``. Bonferroni tightens the per-test level to
    ``alpha / n_trials``; the returned value is the corresponding critical t
    (normal approximation, valid for the sample sizes used here).
    """
    n = max(int(n_trials), 1)
    per_test = alpha / n
    return float(norm.ppf(1 - per_test / 2))


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
