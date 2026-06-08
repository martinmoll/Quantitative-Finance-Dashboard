"""Risk analysis: VaR, CVaR, risk contribution, factor exposure, turnover, costs."""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats


def parametric_var(
    returns: pd.Series, confidence: float = 0.95,
) -> dict:
    """Parametric (Gaussian) VaR and CVaR."""
    mu = returns.mean()
    sigma = returns.std()
    z = sp_stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    cvar = -(mu - sigma * sp_stats.norm.pdf(z) / (1 - confidence))
    return {"VaR": float(var), "CVaR": float(cvar), "method": "Parametric"}


def historical_var(
    returns: pd.Series, confidence: float = 0.95,
) -> dict:
    """Historical (empirical quantile) VaR and CVaR."""
    sorted_rets = returns.sort_values()
    var = -sorted_rets.quantile(1 - confidence)
    tail = sorted_rets[sorted_rets <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return {"VaR": float(var), "CVaR": float(cvar), "method": "Historical"}


def monte_carlo_var(
    returns: pd.Series, confidence: float = 0.95, n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    """Monte Carlo VaR and CVaR using fitted normal distribution."""
    rng = np.random.default_rng(seed)
    mu = returns.mean()
    sigma = returns.std()
    simulated = rng.normal(mu, sigma, n_sims)
    var = -np.percentile(simulated, (1 - confidence) * 100)
    tail = simulated[simulated <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return {"VaR": float(var), "CVaR": float(cvar), "method": "Monte Carlo"}


def cornish_fisher_var(
    returns: pd.Series, confidence: float = 0.95,
) -> dict:
    """Cornish-Fisher VaR adjusting for skewness and kurtosis."""
    mu = returns.mean()
    sigma = returns.std()
    skew = float(sp_stats.skew(returns.dropna()))
    kurt = float(sp_stats.kurtosis(returns.dropna()))
    z = sp_stats.norm.ppf(1 - confidence)
    z_cf = (z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36)
    var = -(mu + z_cf * sigma)
    return {
        "VaR": float(var), "CVaR": float(np.nan),
        "method": "Cornish-Fisher", "skewness": skew, "excess_kurtosis": kurt,
    }


def compute_all_var(
    returns: pd.Series, confidence: float = 0.95, n_sims: int = 10_000,
) -> pd.DataFrame:
    """Compute VaR and CVaR across all methods. Returns a summary DataFrame."""
    results = [
        parametric_var(returns, confidence),
        historical_var(returns, confidence),
        monte_carlo_var(returns, confidence, n_sims),
        cornish_fisher_var(returns, confidence),
    ]
    return pd.DataFrame(results).set_index("method")


def component_var(
    weights: np.ndarray, cov_matrix: np.ndarray, confidence: float = 0.95,
    mu: np.ndarray | None = None,
) -> pd.Series:
    """Decompose portfolio VaR into per-position contributions.

    Component VaR_i = w_i * (Sigma @ w)_i * z / port_vol
    Sum of component VaRs = total parametric VaR.
    """
    z = sp_stats.norm.ppf(confidence)
    port_var = weights @ cov_matrix @ weights
    if port_var <= 0:
        return pd.Series(np.zeros(len(weights)))
    port_vol = np.sqrt(port_var)
    marginal_var = z * (cov_matrix @ weights) / port_vol
    comp = weights * marginal_var
    return pd.Series(comp)


def rolling_var(
    returns: pd.Series, window: int = 24, confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute rolling parametric and historical VaR over a sliding window."""
    param_vars = []
    hist_vars = []
    idx = []
    for i in range(window, len(returns) + 1):
        chunk = returns.iloc[i - window:i]
        pv = parametric_var(chunk, confidence)
        hv = historical_var(chunk, confidence)
        param_vars.append(pv["VaR"])
        hist_vars.append(hv["VaR"])
        idx.append(returns.index[i - 1])
    return pd.DataFrame(
        {"Parametric": param_vars, "Historical": hist_vars}, index=idx,
    )


def var_backtest(
    returns: pd.Series, window: int = 24, confidence: float = 0.95,
) -> dict:
    """Backtest VaR: count breaches and run Kupiec POF test."""
    breaches = 0
    total = 0
    breach_dates = []
    for i in range(window, len(returns)):
        train = returns.iloc[i - window:i]
        var_est = parametric_var(train, confidence)["VaR"]
        actual = returns.iloc[i]
        total += 1
        if actual < -var_est:
            breaches += 1
            breach_dates.append(returns.index[i])

    if total == 0:
        return {"breaches": 0, "total": 0, "breach_rate": 0.0,
                "expected_rate": 1 - confidence, "kupiec_p": np.nan,
                "breach_dates": []}

    breach_rate = breaches / total
    expected_rate = 1 - confidence

    # Kupiec proportion-of-failures (POF) likelihood ratio test
    if breaches == 0 or breaches == total:
        kupiec_p = 0.0
    else:
        lr = -2 * (
            np.log((1 - expected_rate) ** (total - breaches) * expected_rate ** breaches)
            - np.log((1 - breach_rate) ** (total - breaches) * breach_rate ** breaches)
        )
        kupiec_p = float(1 - sp_stats.chi2.cdf(lr, df=1))

    return {
        "breaches": breaches,
        "total": total,
        "breach_rate": breach_rate,
        "expected_rate": expected_rate,
        "kupiec_p": kupiec_p,
        "breach_dates": breach_dates,
    }


def risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> pd.Series:
    port_var = weights @ cov_matrix @ weights
    if port_var <= 0:
        return pd.Series(np.ones(len(weights)) / len(weights))

    port_vol = np.sqrt(port_var)
    marginal = cov_matrix @ weights
    rc = weights * marginal / port_vol
    rc_frac = rc / rc.sum()
    return pd.Series(rc_frac)


def _ff5_available(ff5_factors: pd.DataFrame) -> list[str]:
    return [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if c in ff5_factors.columns]


def factor_exposure(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
) -> pd.Series:
    available = _ff5_available(ff5_factors)
    from core.factor_models import _align_and_regress
    common = portfolio_returns.dropna().index.intersection(ff5_factors.dropna().index)
    if len(common) < 10:
        return pd.Series(dtype=float)

    model = _align_and_regress(portfolio_returns, ff5_factors, available)
    return model.params.drop("const", errors="ignore")


def factor_alpha(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
) -> dict | None:
    """Run FF5 regression and return alpha (intercept) with statistical tests."""
    available = _ff5_available(ff5_factors)
    common = portfolio_returns.dropna().index.intersection(ff5_factors.dropna().index)
    if len(common) < 10:
        return None

    from core.factor_models import _align_and_regress
    model = _align_and_regress(portfolio_returns, ff5_factors, available)

    monthly_alpha = model.params["const"]
    return {
        "monthly_alpha": float(monthly_alpha),
        "annual_alpha": float(monthly_alpha * 12),
        "t_stat": float(model.tvalues["const"]),
        "p_value": float(model.pvalues["const"]),
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "n_months": len(common),
    }


def rolling_factor_exposure(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
    window: int = 36,
) -> pd.DataFrame:
    available = _ff5_available(ff5_factors)
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
