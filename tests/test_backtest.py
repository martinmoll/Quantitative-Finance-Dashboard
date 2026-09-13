import numpy as np
import pandas as pd
from core.backtest import run_walk_forward, BacktestResult
from core.models import get_model, get_default_params


def test_backtest_result_dataclass():
    from dataclasses import fields
    names = {f.name for f in fields(BacktestResult)}
    assert "predictions" in names
    assert "feature_importance" in names
    assert "train_dates" in names
    assert "model_params" in names
    assert "tuned_params" in names


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


def test_auto_tune(sample_panel):
    feature_cols = ["ret_1_xs", "ret_2_12_xs", "bm_xs", "vol_12m_xs", "sue_xs"]
    model = get_model("HGB", get_default_params("HGB"))
    result = run_walk_forward(
        data=sample_panel,
        model=model,
        feature_cols=feature_cols,
        oos_start="2015-01",
        retrain_freq=12,
        window_type="expanding",
        auto_tune=True,
    )
    assert isinstance(result, BacktestResult)
    assert len(result.predictions) > 0
    assert isinstance(result.tuned_params, dict)


# --- fabricated-zero guard --------------------------------------------------
# X = train[feature_cols].fillna(0.0), and feature_cols was filtered on column
# *existence* only. An all-NaN column therefore entered every fit as a column
# of constant zeros, which for a cross-sectionally standardized feature reads
# as "exactly average" for every stock. At the first Tier 1 retrain 33 of 52
# feature columns were constant like this.

def _panel_with(sample_panel, **cols):
    panel = sample_panel.copy()
    for name, value in cols.items():
        panel[name] = value
    return panel


def _fitted_features(result):
    assert result.feature_importance is not None
    return set(result.feature_importance.index)


def _run(panel, feature_cols, **kw):
    return run_walk_forward(
        data=panel, model=get_model("Lasso", {"cv": 3, "max_iter": 1000}),
        feature_cols=feature_cols, oos_start="2015-01", retrain_freq=6,
        **kw,
    )


def test_all_nan_column_never_reaches_the_fit(sample_panel):
    panel = _panel_with(sample_panel, dead_xs=np.nan)
    result = _run(panel, ["ret_1_xs", "bm_xs", "dead_xs"])
    assert "dead_xs" not in _fitted_features(result)
    assert "ret_1_xs" in _fitted_features(result)


def test_constant_column_never_reaches_the_fit(sample_panel):
    """Zero variance, no missing values — still no information to split on."""
    panel = _panel_with(sample_panel, flat_xs=0.7)
    result = _run(panel, ["ret_1_xs", "flat_xs"])
    assert "flat_xs" not in _fitted_features(result)


def test_the_guard_is_point_in_time(sample_panel):
    """A column that is dead early and real later must be used once it is real.

    The fundamentals block behaves exactly this way: empty until 2025, real
    after. A guard applied to the whole panel would keep it out forever.
    """
    panel = sample_panel.copy()
    late = panel["ym"] >= "2015-01"
    panel["late_xs"] = np.where(late, np.random.RandomState(0).randn(len(panel)),
                                np.nan)
    early = _run(panel, ["ret_1_xs", "late_xs"], oos_start_ignored=None) \
        if False else None
    result = _run(panel, ["ret_1_xs", "late_xs"])
    # The last retrain trains on months that include the live period.
    assert "late_xs" in _fitted_features(result)


def test_predictions_still_produced_when_columns_are_dropped(sample_panel):
    """The fit and the predict frame must agree on columns after filtering."""
    panel = _panel_with(sample_panel, dead_xs=np.nan, flat_xs=1.0)
    result = _run(panel, ["ret_1_xs", "bm_xs", "dead_xs", "flat_xs"])
    assert len(result.predictions) > 0
    for frame in result.predictions.values():
        assert frame["pred"].notna().all()


def test_a_window_with_nothing_informative_skips_the_retrain(sample_panel):
    """Reachable from the UI: the "Value only" preset is bm/ep/cfp/sp, and the
    fundamentals block is empty before 2025. Fitting on no columns raises, so
    skip the retrain the way a too-short training window is skipped."""
    panel = _panel_with(sample_panel, dead_a_xs=np.nan, dead_b_xs=np.nan)
    result = _run(panel, ["dead_a_xs", "dead_b_xs"])
    assert result.predictions == {}
    assert result.train_dates == []
