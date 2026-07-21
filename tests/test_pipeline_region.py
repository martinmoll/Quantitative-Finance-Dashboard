import numpy as np
import pandas as pd

from pipeline.universe import get_tickers
from pipeline.config import dataset_paths, region_config
from pipeline.fetchers.market import build_capm_factors


def test_oslo_universe_all_ol():
    tickers = get_tickers("Oslo Børs")
    assert len(tickers) > 20
    assert all(t.endswith(".OL") for t in tickers)


def test_region_dataset_paths_differ():
    us_pq, _ = dataset_paths("US")
    no_pq, _ = dataset_paths("NO")
    assert us_pq != no_pq
    assert no_pq.name == "alpha_dataset_no.parquet"


def test_region_config_market_ticker():
    assert region_config("NO")["market_ticker"] == "^OSEAX"
    assert region_config("US")["market_ticker"] == "SPY"


def test_build_capm_factors_shape():
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    rng = np.random.RandomState(0)
    close = pd.Series(100 * np.cumprod(1 + rng.randn(400) * 0.01), index=idx)

    f = build_capm_factors(close, rf_monthly=None)

    assert f[["SMB", "HML", "RMW", "CMA", "Mom"]].isna().all().all()
    assert f["Mkt_RF"].notna().all()
    assert np.allclose(f["Mkt_RF"], f["spy_ret"] - f["rf_ff"])
    assert all(len(i) == 7 and i[4] == "-" for i in f.index)  # YYYY-MM


def test_build_capm_factors_applies_rf():
    idx = pd.date_range("2022-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(100, 130, 120), index=idx)
    rf = pd.Series(0.001, index=[f"2022-{m:02d}" for m in range(1, 6)])

    f = build_capm_factors(close, rf_monthly=rf)

    common = [i for i in f.index if i in rf.index]
    assert common
    for i in common:
        assert abs(f.loc[i, "rf_ff"] - 0.001) < 1e-12
