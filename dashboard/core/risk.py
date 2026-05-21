"""Risk analysis: risk contribution, factor exposure, turnover, transaction costs."""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> pd.Series:
    port_var = weights @ cov_matrix @ weights
    if port_var <= 0:
        return pd.Series(np.ones(len(weights)) / len(weights))

    port_vol = np.sqrt(port_var)
    marginal = cov_matrix @ weights
    rc = weights * marginal / port_vol
    rc_frac = rc / rc.sum()
    return pd.Series(rc_frac)


def factor_exposure(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
) -> pd.Series:
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    available = [c for c in factor_cols if c in ff5_factors.columns]
    common = portfolio_returns.dropna().index.intersection(ff5_factors.dropna().index)

    if len(common) < 10:
        return pd.Series(dtype=float)

    y = portfolio_returns.loc[common]
    X = sm.add_constant(ff5_factors.loc[common, available])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    return model.params.drop("const", errors="ignore")


def rolling_factor_exposure(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
    window: int = 36,
) -> pd.DataFrame:
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    available = [c for c in factor_cols if c in ff5_factors.columns]
    common = portfolio_returns.dropna().index.intersection(ff5_factors.dropna().index)

    y_all = portfolio_returns.loc[common]
    X_all = ff5_factors.loc[common, available]

    results = pd.DataFrame(np.nan, index=common, columns=available)

    for i in range(window, len(common) + 1):
        idx = common[i - window : i]
        y = y_all.loc[idx]
        X = sm.add_constant(X_all.loc[idx])
        try:
            model = sm.OLS(y, X).fit()
            betas = model.params.drop("const", errors="ignore")
            results.iloc[i - 1] = betas
        except Exception:
            continue

    return results


def compute_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series,
) -> float:
    aligned_c, aligned_p = current_weights.align(previous_weights, fill_value=0.0)
    return 0.5 * (aligned_c - aligned_p).abs().sum()


def transaction_cost_drag(
    turnover_series: pd.Series,
    cost_bps: float = 10.0,
    portfolio_vol: float = 0.15,
) -> dict:
    mean_monthly_to = turnover_series.dropna().mean()
    tc_monthly = mean_monthly_to * cost_bps / 10000 * 2
    tc_annual = tc_monthly * 12
    cost_sr = tc_annual / portfolio_vol if portfolio_vol > 0 else np.nan

    return {
        "TC_annual": float(tc_annual),
        "Cost_SR": float(cost_sr),
        "net_SR_reduction": float(cost_sr),
        "mean_monthly_turnover": float(mean_monthly_to),
    }
