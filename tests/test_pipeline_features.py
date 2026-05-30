"""Tests for pipeline feature computation modules."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.features.price_features import compute_price_features
from pipeline.features.fundamental_features import compute_fundamental_features
from pipeline.features.peer_features import compute_peer_features
from pipeline.features.macro_features import compute_macro_features


@pytest.fixture
def synthetic_daily_prices():
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", "2025-12-31")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    frames = {}
    for ticker in tickers:
        price = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        volume = np.random.randint(1_000_000, 50_000_000, size=len(dates)).astype(float)
        frames[ticker] = pd.DataFrame({
            "Open": price * (1 + np.random.randn(len(dates)) * 0.005),
            "High": price * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            "Low": price * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            "Close": price,
            "Volume": volume,
        }, index=dates)

    combined = pd.concat(frames, axis=1)
    return combined


@pytest.fixture
def synthetic_market_daily(synthetic_daily_prices):
    dates = synthetic_daily_prices.index
    np.random.seed(99)
    price = 400 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.008))
    return pd.Series(price, index=dates, name="SPY")


@pytest.fixture
def synthetic_shares():
    return {
        "AAPL": 15_000_000_000,
        "MSFT": 7_500_000_000,
        "GOOG": 6_000_000_000,
        "AMZN": 10_300_000_000,
        "META": 2_500_000_000,
    }


def test_compute_price_features_shape(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    result = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    assert isinstance(result, pd.DataFrame)
    assert "ticker" in result.columns
    assert "ym" in result.columns
    assert "ret_1" in result.columns
    assert "vol_12m" in result.columns
    assert "beta" in result.columns
    assert "rsi_14" in result.columns
    assert len(result) > 0


def test_momentum_features_present(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    result = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    for col in ["ret_1", "ret_2_12", "ret_2_6", "prc_52w_high"]:
        assert col in result.columns, f"Missing {col}"


def test_technical_features_present(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    result = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    for col in ["rsi_14", "macd_hist", "bb_position", "prc_ma6", "prc_ma12"]:
        assert col in result.columns, f"Missing {col}"


def test_fundamental_features_shape():
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOG"]
    data = {}
    quarters = pd.date_range("2024-01-01", periods=5, freq="QE")

    for ticker in tickers:
        np.random.seed(abs(hash(ticker)) % 2**31)
        n = len(quarters)
        data[ticker] = {
            "income": {
                "TotalRevenue": np.random.uniform(50e9, 100e9, n),
                "GrossProfit": np.random.uniform(20e9, 50e9, n),
                "OperatingIncome": np.random.uniform(10e9, 30e9, n),
                "NetIncome": np.random.uniform(5e9, 25e9, n),
                "DilutedEPS": np.random.uniform(1.0, 5.0, n),
            },
            "balance": {
                "TotalAssets": np.random.uniform(200e9, 400e9, n),
                "TotalEquityGrossMinorityInterest": np.random.uniform(50e9, 150e9, n),
                "TotalDebt": np.random.uniform(20e9, 100e9, n),
                "CashAndCashEquivalents": np.random.uniform(10e9, 50e9, n),
            },
            "cashflow": {
                "OperatingCashFlow": np.random.uniform(10e9, 30e9, n),
            },
            "market_cap": np.random.uniform(1e12, 3e12),
            "quarters": quarters,
        }

    result = compute_fundamental_features(data)
    assert isinstance(result, pd.DataFrame)
    assert "ticker" in result.columns
    assert "ym" in result.columns
    assert "bm" in result.columns
    assert "roe" in result.columns
    assert len(result) > 0


def test_fundamental_features_value_ratios():
    np.random.seed(42)
    quarters = pd.date_range("2024-01-01", periods=5, freq="QE")
    data = {
        "TEST": {
            "income": {
                "TotalRevenue": [100e9] * 5,
                "GrossProfit": [40e9] * 5,
                "OperatingIncome": [20e9] * 5,
                "NetIncome": [15e9] * 5,
                "DilutedEPS": [3.0] * 5,
            },
            "balance": {
                "TotalAssets": [500e9] * 5,
                "TotalEquityGrossMinorityInterest": [200e9] * 5,
                "TotalDebt": [100e9] * 5,
                "CashAndCashEquivalents": [50e9] * 5,
            },
            "cashflow": {
                "OperatingCashFlow": [25e9] * 5,
            },
            "market_cap": 2e12,
            "quarters": quarters,
        }
    }

    result = compute_fundamental_features(data)
    for col in ["bm", "ep", "cfp", "sp"]:
        assert col in result.columns
        valid = result[col].dropna()
        assert len(valid) > 0


def test_peer_features_sector_columns(
    synthetic_daily_prices, synthetic_market_daily, synthetic_shares
):
    price_feats = compute_price_features(
        synthetic_daily_prices, synthetic_market_daily, synthetic_shares
    )
    sectors = {t: "Tech" if i % 2 == 0 else "Finance"
               for i, t in enumerate(["AAPL", "MSFT", "GOOG", "AMZN", "META"])}
    result = compute_peer_features(price_feats, sectors)
    assert "sector_ret_avg" in result.columns
    assert "ret_vs_sector" in result.columns
    assert "sector" in result.columns


def test_macro_features_shape():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")
    macro = pd.DataFrame({
        "vix": np.random.uniform(12, 35, len(dates)),
        "fin_stress": np.random.randn(len(dates)) * 0.5,
    }, index=dates)
    months = ["2025-06", "2025-07", "2025-08"]
    result = compute_macro_features(macro, months)
    assert len(result) == 3
    assert "macro_unc_1m" in result.columns
    assert "fin_unc_1m" in result.columns
