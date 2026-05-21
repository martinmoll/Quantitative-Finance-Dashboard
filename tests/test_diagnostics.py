import numpy as np
import pandas as pd
from core.diagnostics import (
    compute_performance_metrics,
    compute_ic_stats,
    fundamental_law,
    feature_ic,
    ks_test,
    alpha_decay,
    signal_staleness,
)


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
