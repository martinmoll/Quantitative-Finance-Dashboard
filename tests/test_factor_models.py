import numpy as np
import pandas as pd
from core.factor_models import (
    run_regression,
    rolling_beta,
    bloomberg_shrink_beta,
    compute_vif,
    wald_test,
    hedge_ratio,
)


def test_run_regression_capm(sample_ff5):
    np.random.seed(42)
    n = len(sample_ff5)
    stock_returns = sample_ff5["Mkt-RF"] * 1.2 + np.random.randn(n) * 0.02
    result = run_regression(stock_returns, sample_ff5, model_type="CAPM")
    assert "coefficients" in result
    assert "const" in result["coefficients"]
    assert "Mkt-RF" in result["coefficients"]
    assert "r_squared" in result
    assert "hac_se" in result


def test_run_regression_ff5(sample_ff5):
    np.random.seed(42)
    n = len(sample_ff5)
    stock_returns = (sample_ff5["Mkt-RF"] * 1.1 + sample_ff5["SMB"] * 0.5 +
                     np.random.randn(n) * 0.02)
    result = run_regression(stock_returns, sample_ff5, model_type="FF5")
    assert "SMB" in result["coefficients"]
    assert "HML" in result["coefficients"]
    assert "RMW" in result["coefficients"]
    assert "CMA" in result["coefficients"]


def test_rolling_beta(sample_ff5):
    np.random.seed(42)
    n = len(sample_ff5)
    stock_returns = sample_ff5["Mkt-RF"] * 1.2 + np.random.randn(n) * 0.02
    result = rolling_beta(stock_returns, sample_ff5["Mkt-RF"], window=24)
    assert "beta" in result.columns
    assert "lower" in result.columns
    assert "upper" in result.columns
    assert len(result) == n


def test_bloomberg_shrink():
    raw = pd.Series([0.5, 1.0, 1.5, 2.0])
    shrunk = bloomberg_shrink_beta(raw)
    expected = 0.67 * raw + 0.33
    pd.testing.assert_series_equal(shrunk, expected)


def test_compute_vif(sample_ff5):
    factors = sample_ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    result = compute_vif(factors)
    assert "feature" in result.columns
    assert "VIF" in result.columns
    assert len(result) == 5


def test_wald_test(sample_ff5):
    np.random.seed(42)
    n = len(sample_ff5)
    stock_returns = sample_ff5["Mkt-RF"] * 1.2 + np.random.randn(n) * 0.02
    result = wald_test(stock_returns, sample_ff5, indices=["SMB", "HML"])
    assert "statistic" in result
    assert "pvalue" in result
    assert "reject" in result


def test_hedge_ratio():
    assert hedge_ratio(1.2, 1.0) == -1.2
    assert hedge_ratio(0.8, 1.0) == -0.8
