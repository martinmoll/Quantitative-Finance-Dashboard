import numpy as np
import pandas as pd
from core.portfolio import construct_portfolio, build_portfolio_series


def _make_month_df(n=50):
    np.random.seed(42)
    return pd.DataFrame({
        "permno": range(10001, 10001 + n),
        "pred": np.random.randn(n) * 0.1,
        "y_raw": np.random.randn(n) * 0.08,
        "vol_12m_xs": np.abs(np.random.randn(n)) * 0.5 + 0.5,
        "sector": ["Tech", "Finance", "Health", "Energy", "Consumer"] * (n // 5),
    })


def test_equal_weight():
    df = _make_month_df()
    result = construct_portfolio(df, method="equal_weight", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    assert "weight" in result.columns
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0)


def test_score_weight():
    df = _make_month_df()
    result = construct_portfolio(df, method="score_weight", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=5)


def test_inverse_vol():
    df = _make_month_df()
    result = construct_portfolio(df, method="inverse_vol", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=5)


def test_erc():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    result = construct_portfolio(
        df, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)
    assert (result["weight"] > 0).all()


def test_mvo():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    result = construct_portfolio(
        df, method="mvo", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)
    assert (result["weight"] >= -0.001).all()
    assert (result["weight"] <= 0.151).all()


def test_mvo_tc_aware():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    prev_w = np.ones(10) / 10
    result = construct_portfolio(
        df, method="mvo", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
        prev_weights=prev_w, tc_bps=50.0,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)


def test_long_short():
    df = _make_month_df()
    result = construct_portfolio(df, method="equal_weight", K=10,
                                strategy_type="long_short", K_short=5, vol_tilt=0.0)
    assert len(result) == 15
    assert "side" in result.columns
    assert (result[result["side"] == "long"]["weight"] > 0).all()
    assert (result[result["side"] == "short"]["weight"] < 0).all()


def test_build_portfolio_series(sample_predictions):
    result = build_portfolio_series(
        predictions=sample_predictions,
        method="equal_weight",
        K=10,
        strategy_type="long_only",
        K_short=10,
        vol_tilt=0.05,
        regime_lookback=0,
    )
    assert "monthly_returns" in result
    assert "holdings" in result
    assert "ic" in result
    assert "turnover" in result
    assert len(result["monthly_returns"]) > 0
