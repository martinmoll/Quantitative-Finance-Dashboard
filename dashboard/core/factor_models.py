"""CAPM, Fama-French regressions, rolling beta, VIF, Wald test, hedge ratio.

All regressions use HAC standard errors.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


_FACTOR_SETS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


def _align_and_regress(
    returns: pd.Series,
    factors: pd.DataFrame,
    factor_cols: list[str],
    hac: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Align indices, add constant, run OLS. Shared by all regression functions."""
    common_idx = returns.dropna().index.intersection(factors.dropna().index)
    y = returns.loc[common_idx]
    X = sm.add_constant(factors.loc[common_idx, factor_cols])
    if hac:
        return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return sm.OLS(y, X).fit()


def run_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    model_type: str = "CAPM",
) -> dict:
    factor_cols = _FACTOR_SETS[model_type]
    model = _align_and_regress(returns, factors, factor_cols)

    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    dw = durbin_watson(model.resid)
    jb_stat, jb_pval, _, _ = jarque_bera(model.resid)

    return {
        "coefficients": model.params.to_dict(),
        "hac_se": model.bse.to_dict(),
        "t_stats": model.tvalues.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "durbin_watson": float(dw),
        "jarque_bera": {"statistic": float(jb_stat), "pvalue": float(jb_pval)},
        "nobs": int(model.nobs),
        "model": model,
    }


def rolling_beta(
    returns: pd.Series,
    market: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    common = returns.dropna().index.intersection(market.dropna().index)
    y_all = returns.loc[common]
    x_all = market.loc[common]
    n = len(common)

    betas = pd.Series(np.nan, index=common)
    lower = pd.Series(np.nan, index=common)
    upper = pd.Series(np.nan, index=common)

    for i in range(window, n + 1):
        idx = common[i - window : i]
        y = y_all.loc[idx]
        X = sm.add_constant(x_all.loc[idx])
        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            b = model.params.iloc[1]
            se = model.bse.iloc[1]
            betas.iloc[i - 1] = b
            lower.iloc[i - 1] = b - 1.96 * se
            upper.iloc[i - 1] = b + 1.96 * se
        except Exception:
            continue

    return pd.DataFrame({"beta": betas, "lower": lower, "upper": upper})


def bloomberg_shrink_beta(raw_beta: pd.Series) -> pd.Series:
    return 0.67 * raw_beta + 0.33


def compute_vif(factors: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(factors)
    vif_data = []
    for i, col in enumerate(factors.columns):
        vif_data.append({
            "feature": col,
            "VIF": variance_inflation_factor(X.values, i + 1),
        })
    return pd.DataFrame(vif_data)


def wald_test(
    returns: pd.Series,
    factors: pd.DataFrame,
    indices: list[str],
) -> dict:
    reg = run_regression(returns, factors, model_type="FF5")
    model = reg["model"]

    restriction = ", ".join([f"{name} = 0" for name in indices])

    try:
        result = model.wald_test(restriction)
        stat = float(result.statistic[0][0])
        pval = float(result.pvalue)
    except Exception:
        R = np.zeros((len(indices), len(model.params)))
        for i, name in enumerate(indices):
            col_idx = list(model.params.index).index(name)
            R[i, col_idx] = 1
        q = np.zeros(len(indices))
        result = model.wald_test(R)
        stat = float(result.statistic[0][0])
        pval = float(result.pvalue)

    return {
        "statistic": stat,
        "pvalue": pval,
        "reject": pval < 0.05,
    }


def hedge_ratio(portfolio_beta: float, market_beta: float = 1.0) -> float:
    return -portfolio_beta / market_beta
