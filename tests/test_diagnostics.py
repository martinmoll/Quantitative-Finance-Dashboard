import numpy as np
import pandas as pd
import pytest
from core.diagnostics import (
    compute_performance_metrics,
    compute_ic_stats,
    fundamental_law,
    feature_ic,
    ks_test,
    feature_drift,
    recent_training_window,
    alpha_decay,
    signal_staleness,
    bootstrap_sharpe_ci,
    bootstrap_alpha_ci,
    multiple_testing_hurdle,
    compute_r2_oos,
)


def _r2_preds(pred_fn, n_months=14, n=50, seed=0):
    rng = np.random.RandomState(seed)
    months = [f"{2020 + i // 12:04d}-{i % 12 + 1:02d}" for i in range(n_months)]
    out = {}
    for m in months:
        r = rng.randn(n) * 0.10 + 0.01
        out[m] = pd.DataFrame({"permno": range(n), "pred": pred_fn(r), "y_raw": r})
    return out


def test_performance_metrics(sample_returns):
    metrics = compute_performance_metrics(sample_returns)
    assert "SR" in metrics
    assert "Ann Return" in metrics
    assert "Ann Vol" in metrics
    assert "MDD" in metrics
    assert "Calmar" in metrics
    assert "Total Return" in metrics
    assert metrics["Ann Vol"] > 0
    assert metrics["MDD"] <= 0


def test_ic_stats():
    np.random.seed(42)
    ic = pd.Series(np.random.randn(60) * 0.05 + 0.03)
    stats = compute_ic_stats(ic)
    assert "mean_ic" in stats
    assert "ic_tstat" in stats
    assert "icir" in stats
    assert "hit_rate" in stats
    assert 0 <= stats["hit_rate"] <= 1


def test_fundamental_law():
    result = fundamental_law(ic_mean=0.05, K=30, rebal_freq=12)
    assert "BR_nominal" in result
    assert result["BR_nominal"] == 360
    assert "IR_upper_bound" in result
    assert result["IR_upper_bound"] > 0


def test_feature_ic():
    np.random.seed(42)
    X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
    y = pd.Series(np.random.randn(100))
    result = feature_ic(X, y)
    assert len(result) == 2
    assert "f1" in result.index


def test_ks_test():
    np.random.seed(42)
    X_train = pd.DataFrame({"f1": np.random.randn(500), "f2": np.random.randn(500)})
    X_current = pd.DataFrame({"f1": np.random.randn(50) + 2, "f2": np.random.randn(50)})
    result = ks_test(X_train, X_current)
    assert "feature" in result.columns
    assert "D" in result.columns
    assert "pval" in result.columns
    assert "flag" in result.columns
    f1_row = result[result["feature"] == "f1"]
    assert f1_row["D"].values[0] > 0.3


def test_ks_test_empty_when_no_valid_data():
    train = pd.DataFrame({"a": [np.nan, np.nan]})
    current = pd.DataFrame({"a": [np.nan, np.nan]})
    result = ks_test(train, current)
    assert list(result.columns) == ["feature", "D", "pval", "flag"]
    assert len(result) == 0


def test_feature_drift_survives_all_nan_columns():
    """A feature that is entirely NaN in the training window must not wipe out
    drift detection for the features that do have data (the Monitoring bug)."""
    np.random.seed(0)
    train = pd.DataFrame({
        "good": np.random.randn(300),
        "all_nan": np.full(300, np.nan),
    })
    current = pd.DataFrame({
        "good": np.random.randn(60) + 2,  # shifted → should flag
        "all_nan": np.full(60, np.nan),
    })
    result = feature_drift(train, current, ["good", "all_nan"])
    assert "good" in result["feature"].values
    assert "all_nan" not in result["feature"].values
    assert bool(result[result["feature"] == "good"]["flag"].iloc[0])


def test_r2_oos_perfect_standardized_forecast():
    # pred == within-month standardized realized return -> R2 ~ 1 on the model's scale
    preds = _r2_preds(lambda r: (r - r.mean()) / r.std())
    assert compute_r2_oos(preds) > 0.99


def test_r2_oos_mean_forecast_is_about_zero():
    # forecasting 0 (the standardized mean) -> R2 ~ 0, not hugely negative
    preds = _r2_preds(lambda r: np.zeros_like(r))
    assert abs(compute_r2_oos(preds)) < 0.05


def test_r2_oos_not_dominated_by_scale():
    # A standardized-scale forecast with real skill scores positive, where the old
    # raw-scale comparison would have been dominated by the y_xs/y_raw scale gap.
    preds = _r2_preds(lambda r: 0.6 * (r - r.mean()) / r.std())
    assert compute_r2_oos(preds) > 0.3


def _monthly(start_year, n_years):
    return [f"{y:04d}-{m:02d}" for y in range(start_year, start_year + n_years)
            for m in range(1, 13)]


def test_recent_training_window_expanding():
    months = _monthly(2016, 5)  # 2016-01 .. 2020-12
    win = recent_training_window(months, "2018-01", 12, "expanding", None, "2020-12")
    # last retrain is 2020-01; expanding trains on everything before it
    assert win[0] == "2016-01" and win[-1] == "2019-12"
    assert "2020-01" not in win


def test_recent_training_window_rolling_is_recent_not_earliest():
    months = _monthly(2016, 5)
    win = recent_training_window(months, "2018-01", 12, "rolling", 12, "2020-12")
    # last 12 months before the 2020-01 retrain — recent, not the 2016 baseline
    assert win == _monthly(2019, 1)
    assert "2016-01" not in win


def test_recent_training_window_empty_when_no_pre_oos():
    months = _monthly(2016, 5)
    assert recent_training_window(months, "2099-01", 12, "expanding", None, "2020-12") == []


def test_feature_drift_empty_when_no_train_data():
    train = pd.DataFrame({"good": []})
    current = pd.DataFrame({"good": [0.1, 0.2, 0.3]})
    result = feature_drift(train, current, ["good"])
    assert result.empty


def test_alpha_decay(sample_predictions):
    result = alpha_decay(sample_predictions, horizons=[1, 2, 3])
    assert len(result) == 3
    assert result.index.tolist() == [1, 2, 3]


def test_signal_staleness():
    turnover = pd.Series(
        [0.5, 0.4, 0.08, 0.07, 0.06, 0.09, 0.5, 0.4],
        index=[f"2015-{m:02d}" for m in range(1, 9)],
    )
    result = signal_staleness(turnover, threshold=0.10, consecutive=3)
    assert "stale" in result.columns
    assert result["stale"].any()


def test_bootstrap_sharpe_ci(sample_returns):
    result = bootstrap_sharpe_ci(sample_returns, n_boot=1000)
    assert "point" in result and "lo" in result and "hi" in result
    assert not np.isnan(result["point"])
    assert result["lo"] < result["point"] < result["hi"]
    assert result["ci"] == 0.95


def test_bootstrap_sharpe_ci_short_series():
    short = pd.Series([0.01, 0.02, -0.01])
    result = bootstrap_sharpe_ci(short)
    assert np.isnan(result["point"])


def test_bootstrap_alpha_ci(sample_returns, sample_ff5):
    common = sample_returns.index.intersection(sample_ff5.index)
    rets = sample_returns.loc[common]
    result = bootstrap_alpha_ci(rets, sample_ff5, n_boot=1000)
    assert "point" in result and "lo" in result and "hi" in result
    assert not np.isnan(result["point"])
    assert result["lo"] < result["hi"]


def test_bootstrap_alpha_ci_insufficient_data(sample_ff5):
    short = pd.Series([0.01] * 5, index=sample_ff5.index[:5])
    result = bootstrap_alpha_ci(short, sample_ff5, n_boot=100)
    assert np.isnan(result["point"])


def test_multiple_testing_hurdle_single_trial():
    # One test reduces to the ordinary two-sided 5% critical value.
    assert multiple_testing_hurdle(1) == pytest.approx(1.959964, abs=1e-4)


def test_multiple_testing_hurdle_rises_with_trials():
    # More trials → stricter (higher) t-hurdle to hold family-wise error at 5%.
    h1 = multiple_testing_hurdle(1)
    h5 = multiple_testing_hurdle(5)
    h20 = multiple_testing_hurdle(20)
    assert h1 < h5 < h20
    assert h20 == pytest.approx(3.023, abs=1e-3)


def test_multiple_testing_hurdle_floors_at_one_trial():
    # Zero/negative trial counts are treated as a single test, not a crash.
    assert multiple_testing_hurdle(0) == multiple_testing_hurdle(1)
