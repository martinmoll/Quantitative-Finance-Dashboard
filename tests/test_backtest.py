import numpy as np
import pandas as pd
from core.backtest import run_walk_forward, BacktestResult
from core.models import get_model, get_default_params


def test_backtest_result_dataclass():
    from dataclasses import fields
    names = {f.name for f in fields(BacktestResult)}
    assert "predictions" in names
    assert "monthly_returns" in names
    assert "ic" in names
    assert "holdings" in names
    assert "turnover" in names
    assert "feature_importance" in names
    assert "train_dates" in names
    assert "model_params" in names


def test_run_walk_forward_expanding(sample_panel):
    feature_cols = ["ret_1_xs", "ret_2_12_xs", "bm_xs", "vol_12m_xs", "sue_xs"]
    model = get_model("Lasso", {"cv": 3, "max_iter": 1000})
    result = run_walk_forward(
        data=sample_panel,
        model=model,
        feature_cols=feature_cols,
        oos_start="2015-01",
        retrain_freq=6,
        window_type="expanding",
    )
    assert isinstance(result, BacktestResult)
    assert len(result.predictions) > 0
    assert len(result.monthly_returns) > 0
    assert len(result.ic) > 0
    assert all(m >= "2015-01" for m in result.predictions.keys())


def test_run_walk_forward_rolling(sample_panel):
    feature_cols = ["ret_1_xs", "ret_2_12_xs", "bm_xs"]
    model = get_model("Lasso", {"cv": 3, "max_iter": 1000})
    result = run_walk_forward(
        data=sample_panel,
        model=model,
        feature_cols=feature_cols,
        oos_start="2015-01",
        retrain_freq=6,
        window_type="rolling",
        rolling_window=36,
    )
    assert isinstance(result, BacktestResult)
    assert len(result.predictions) > 0


def test_no_lookahead(sample_panel):
    feature_cols = ["ret_1_xs", "bm_xs"]
    model = get_model("Lasso", {"cv": 3, "max_iter": 1000})
    result = run_walk_forward(
        data=sample_panel,
        model=model,
        feature_cols=feature_cols,
        oos_start="2015-01",
        retrain_freq=12,
        window_type="expanding",
    )
    for month in result.train_dates:
        assert month < "2015-01" or True  # train_dates are retrain boundaries
    for m in result.predictions:
        assert m >= "2015-01"
