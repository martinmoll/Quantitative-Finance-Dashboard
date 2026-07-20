import pytest
import pandas as pd
import numpy as np
from core.data_loader import (
    load_dataset, compute_market_monthly, load_ff5_factors, permno_ticker_map,
)


def test_compute_market_monthly(sample_panel):
    market = compute_market_monthly(sample_panel)
    assert isinstance(market, pd.DataFrame)
    assert "spy_ret" in market.columns
    assert "Mkt_RF" in market.columns
    assert "rf_ff" in market.columns
    assert market.index.name == "ym"
    np.testing.assert_allclose(
        market["spy_ret"].values,
        (market["Mkt_RF"] + market["rf_ff"]).values,
    )


def test_compute_market_monthly_sorted(sample_panel):
    market = compute_market_monthly(sample_panel)
    assert list(market.index) == sorted(market.index)


def test_load_ff5_factors_columns(tmp_path):
    months = ["2015-01", "2015-02", "2015-03"]
    df = pd.DataFrame(
        {
            "Mkt-RF": [0.01, -0.02, 0.015],
            "SMB": [0.005, -0.003, 0.002],
            "HML": [-0.01, 0.008, 0.004],
            "RMW": [0.003, -0.001, 0.006],
            "CMA": [0.002, 0.004, -0.002],
            "RF": [0.001, 0.001, 0.001],
        },
        index=months,
    )
    csv_path = tmp_path / "ff5_factors.csv"
    df.to_csv(csv_path, index_label="date")

    loaded = load_ff5_factors(csv_path)
    assert set(loaded.columns) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}
    assert len(loaded) == 3


def test_load_ff5_factors_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ff5_factors(tmp_path / "nonexistent.csv")


def test_permno_ticker_map():
    df = pd.DataFrame({
        "permno": [1, 1, 2, 2],
        "ym": ["2020-01", "2020-02", "2020-01", "2020-02"],
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
    })
    assert permno_ticker_map(df) == {1: "AAPL", 2: "MSFT"}


def test_permno_ticker_map_no_ticker_column():
    df = pd.DataFrame({"permno": [1, 2], "ym": ["2020-01", "2020-01"]})
    assert permno_ticker_map(df) == {}


def test_load_dataset_from_csv(tmp_path):
    df = pd.DataFrame({
        "ym": ["2020-01", "2020-02"],
        "permno": [1, 2],
        "y_xs": [0.1, 0.2],
        "y_raw": [0.1, 0.2],
        "Mkt_RF": [0.01, -0.01],
        "rf_ff": [0.002, 0.002],
    })
    csv_path = tmp_path / "alpha_dataset_v2.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_dataset(path=csv_path)
    assert len(loaded) == 2
    assert set(df.columns).issubset(set(loaded.columns))


def test_load_dataset_from_parquet(tmp_path):
    df = pd.DataFrame({
        "ym": ["2020-01", "2020-02"],
        "permno": [1, 2],
        "y_xs": [0.1, 0.2],
        "y_raw": [0.1, 0.2],
        "Mkt_RF": [0.01, -0.01],
        "rf_ff": [0.002, 0.002],
    })
    pq_path = tmp_path / "test.parquet"
    df.to_parquet(pq_path, index=False)

    loaded = load_dataset(path=pq_path)
    assert len(loaded) == 2


def test_load_dataset_missing_columns(tmp_path):
    df = pd.DataFrame({"ym": ["2020-01"], "permno": [1]})
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path=csv_path)


def test_load_dataset_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(path=tmp_path / "nonexistent.csv")
