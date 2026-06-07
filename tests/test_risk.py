import numpy as np
import pandas as pd
from core.risk import (
    risk_contribution,
    factor_exposure,
    rolling_factor_exposure,
    compute_turnover,
    transaction_cost_drag,
    parametric_var,
    historical_var,
    monte_carlo_var,
    cornish_fisher_var,
    compute_all_var,
    component_var,
    rolling_var,
    var_backtest,
    factor_alpha,
)


# ── Existing tests ──────────────────────────────────────────────────────


def test_risk_contribution():
    np.random.seed(42)
    weights = np.array([0.3, 0.3, 0.4])
    cov = np.array([[0.04, 0.01, 0.005],
                     [0.01, 0.03, 0.008],
                     [0.005, 0.008, 0.05]])
    rc = risk_contribution(weights, cov)
    assert len(rc) == 3
    np.testing.assert_almost_equal(rc.sum(), 1.0, decimal=5)
    assert (rc > 0).all()


def test_factor_exposure(sample_ff5):
    np.random.seed(42)
    port_returns = pd.Series(
        np.random.randn(len(sample_ff5)) * 0.03,
        index=sample_ff5.index,
    )
    result = factor_exposure(port_returns, sample_ff5)
    assert "Mkt-RF" in result.index
    assert "SMB" in result.index


def test_rolling_factor_exposure(sample_ff5):
    np.random.seed(42)
    port_returns = pd.Series(
        np.random.randn(len(sample_ff5)) * 0.03,
        index=sample_ff5.index,
    )
    result = rolling_factor_exposure(port_returns, sample_ff5, window=24)
    assert "Mkt-RF" in result.columns
    assert len(result) == len(sample_ff5)


def test_compute_turnover():
    curr = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    prev = pd.Series({"A": 0.5, "B": 0.5, "D": 0.0})
    to = compute_turnover(curr, prev)
    assert to > 0


def test_transaction_cost_drag():
    turnover = pd.Series([0.3, 0.25, 0.35, 0.28])
    result = transaction_cost_drag(turnover, cost_bps=10, portfolio_vol=0.15)
    assert "TC_annual" in result
    assert "Cost_SR" in result
    assert "net_SR_reduction" in result
    assert result["TC_annual"] > 0


# ── VaR method tests ────────────────────────────────────────────────────


def _make_returns(n=500, seed=42):
    np.random.seed(seed)
    return pd.Series(np.random.randn(n) * 0.04 + 0.005)


def test_parametric_var_keys_and_signs():
    rets = _make_returns()
    result = parametric_var(rets, confidence=0.95)
    assert result["method"] == "Parametric"
    assert result["VaR"] > 0
    assert result["CVaR"] > result["VaR"]


def test_historical_var_keys_and_signs():
    rets = _make_returns()
    result = historical_var(rets, confidence=0.95)
    assert result["method"] == "Historical"
    assert result["VaR"] > 0
    assert result["CVaR"] >= result["VaR"]


def test_monte_carlo_var_deterministic():
    rets = _make_returns()
    r1 = monte_carlo_var(rets, confidence=0.95, seed=0)
    r2 = monte_carlo_var(rets, confidence=0.95, seed=0)
    assert r1["VaR"] == r2["VaR"]
    assert r1["CVaR"] == r2["CVaR"]
    assert r1["VaR"] > 0


def test_cornish_fisher_var_returns_skew_kurt():
    rets = _make_returns()
    result = cornish_fisher_var(rets, confidence=0.95)
    assert result["method"] == "Cornish-Fisher"
    assert "skewness" in result
    assert "excess_kurtosis" in result
    assert result["VaR"] > 0


def test_higher_confidence_means_higher_var():
    rets = _make_returns()
    var_90 = parametric_var(rets, confidence=0.90)["VaR"]
    var_99 = parametric_var(rets, confidence=0.99)["VaR"]
    assert var_99 > var_90


def test_compute_all_var_returns_dataframe():
    rets = _make_returns()
    df = compute_all_var(rets, confidence=0.95)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "method"
    assert set(df.index) == {"Parametric", "Historical", "Monte Carlo", "Cornish-Fisher"}
    assert "VaR" in df.columns
    assert "CVaR" in df.columns
    assert (df["VaR"] > 0).all()


# ── Component VaR ───────────────────────────────────────────────────────


def test_component_var_sums_to_total():
    weights = np.array([0.4, 0.35, 0.25])
    cov = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.03, 0.008],
        [0.005, 0.008, 0.05],
    ])
    cv = component_var(weights, cov, confidence=0.95)
    assert len(cv) == 3
    from scipy import stats as sp_stats
    z = sp_stats.norm.ppf(0.95)
    port_vol = np.sqrt(weights @ cov @ weights)
    expected_total = z * port_vol
    np.testing.assert_almost_equal(cv.sum(), expected_total, decimal=6)


def test_component_var_zero_variance():
    weights = np.array([0.5, 0.5])
    cov = np.zeros((2, 2))
    cv = component_var(weights, cov, confidence=0.95)
    assert (cv == 0).all()


# ── Rolling VaR ─────────────────────────────────────────────────────────


def test_rolling_var_shape():
    rets = _make_returns(n=60)
    window = 24
    result = rolling_var(rets, window=window, confidence=0.95)
    assert isinstance(result, pd.DataFrame)
    assert "Parametric" in result.columns
    assert "Historical" in result.columns
    assert len(result) == len(rets) - window + 1


# ── VaR Backtest (Kupiec) ──────────────────────────────────────────────


def test_var_backtest_keys():
    rets = _make_returns(n=100)
    result = var_backtest(rets, window=24, confidence=0.95)
    assert "breaches" in result
    assert "total" in result
    assert "breach_rate" in result
    assert "expected_rate" in result
    assert "kupiec_p" in result
    assert "breach_dates" in result
    assert result["total"] == 100 - 24
    np.testing.assert_almost_equal(result["expected_rate"], 0.05)


def test_var_backtest_breach_rate_bounded():
    rets = _make_returns(n=200)
    result = var_backtest(rets, window=24, confidence=0.95)
    assert 0 <= result["breach_rate"] <= 1.0


# ── Factor Alpha ────────────────────────────────────────────────────────


def test_factor_alpha_returns_dict(sample_ff5):
    np.random.seed(42)
    port_returns = pd.Series(
        np.random.randn(len(sample_ff5)) * 0.03 + 0.005,
        index=sample_ff5.index,
    )
    result = factor_alpha(port_returns, sample_ff5)
    assert result is not None
    assert "monthly_alpha" in result
    assert "annual_alpha" in result
    np.testing.assert_almost_equal(
        result["annual_alpha"], result["monthly_alpha"] * 12, decimal=10,
    )
    assert "t_stat" in result
    assert "p_value" in result
    assert "r_squared" in result
    assert result["n_months"] == len(sample_ff5)


def test_factor_alpha_insufficient_data(sample_ff5):
    port_returns = pd.Series(
        [0.01] * 5, index=sample_ff5.index[:5],
    )
    result = factor_alpha(port_returns, sample_ff5)
    assert result is None
