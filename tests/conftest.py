import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_panel():
    """Small synthetic panel dataset mimicking alpha_dataset_v2.csv.

    50 stocks x 72 months (2010-01 to 2015-12).
    OOS starts at 2015-01 (last 12 months).
    """
    np.random.seed(42)
    dates = [f"20{y:02d}-{m:02d}" for y in range(10, 16) for m in range(1, 13)]
    permnos = list(range(10001, 10051))
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]

    xs_features = [
        "ret_1_xs", "ret_2_12_xs", "ret_2_6_xs", "ret_13_36_xs",
        "bm_xs", "ep_xs", "cfp_xs", "sp_xs",
        "gpa_xs", "roe_xs", "roa_xs", "earn_quality_xs", "cfo_at_xs",
        "vol_12m_xs", "ivol_xs", "beta_xs", "log_me_xs",
        "sue_xs", "revision_xs", "beat_xs", "dispersion_xs",
        "turnover_xs", "illiq_12m_xs",
        "prc_52w_high_xs", "rsi_14_xs", "macd_hist_xs",
        "bb_position_xs", "roc_3_xs", "roc_6_xs",
        "sue_q_xs", "rev_surp_xs", "earn_growth_yoy_xs",
        "revision_ratio_xs", "rec_chg_xs", "n_analysts_xs",
        "peer_sue_xs", "peer_revision_xs",
        "mom_x_size_xs", "val_x_prof_xs", "mom_x_vol_xs",
        "ret_vs_sector_xs", "bm_vs_sector_xs", "ret_vs_ind_xs",
        "bm_vs_size_xs",
    ]

    rows = []
    for d in dates:
        mkt_rf = np.random.randn() * 0.04
        rf = 0.002
        for p in permnos:
            row = {
                "ym": d,
                "permno": p,
                "sector": sectors[p % 5],
                "y_xs": np.random.randn() * 0.05,
                "y_raw": np.random.randn() * 0.08,
                "Mkt_RF": mkt_rf,
                "rf_ff": rf,
            }
            for feat in xs_features:
                row[feat] = np.random.randn()
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def sample_returns():
    """Monthly return series for testing performance metrics."""
    np.random.seed(42)
    months = [f"20{y:02d}-{m:02d}" for y in range(15, 20) for m in range(1, 13)]
    returns = np.random.randn(len(months)) * 0.05 + 0.008
    return pd.Series(returns, index=months, name="returns")


@pytest.fixture
def sample_ff5():
    """Synthetic Fama-French 5-factor monthly data."""
    np.random.seed(42)
    months = [f"20{y:02d}-{m:02d}" for y in range(10, 16) for m in range(1, 13)]
    return pd.DataFrame(
        {
            "Mkt-RF": np.random.randn(len(months)) * 0.04,
            "SMB": np.random.randn(len(months)) * 0.02,
            "HML": np.random.randn(len(months)) * 0.02,
            "RMW": np.random.randn(len(months)) * 0.015,
            "CMA": np.random.randn(len(months)) * 0.015,
            "RF": np.full(len(months), 0.002),
        },
        index=months,
    )


@pytest.fixture
def sample_predictions():
    """Synthetic month->predictions dict as returned by run_walk_forward."""
    np.random.seed(42)
    months = [f"2015-{m:02d}" for m in range(1, 13)]
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]
    preds = {}
    for m in months:
        n = 50
        preds[m] = pd.DataFrame(
            {
                "permno": list(range(10001, 10001 + n)),
                "sector": [sectors[i % 5] for i in range(n)],
                "pred": np.random.randn(n) * 0.1,
                "y_raw": np.random.randn(n) * 0.08,
                "vol_12m_xs": np.abs(np.random.randn(n)) * 0.5 + 0.5,
            }
        )
    return preds
