import pandas as pd
import numpy as np
from core.data_loader import load_dataset, compute_market_monthly, load_ff5_factors


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
    # Create a minimal CSV matching expected format
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
