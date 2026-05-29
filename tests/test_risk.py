import numpy as np
import pandas as pd
from core.risk import (
    risk_contribution,
    factor_exposure,
    rolling_factor_exposure,
    compute_turnover,
    transaction_cost_drag,
)


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
