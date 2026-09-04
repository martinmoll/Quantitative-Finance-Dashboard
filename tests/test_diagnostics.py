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
    latest_month_staleness,
    alpha_decay,
    signal_staleness,
    bootstrap_sharpe_ci,
    bootstrap_alpha_ci,
    multiple_testing_hurdle,
    compute_r2_oos,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
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


def test_psr_high_for_strong_track_record():
    rng = np.random.RandomState(0)
    r = pd.Series(rng.randn(120) * 0.03 + 0.02)   # positive monthly Sharpe, long
    assert probabilistic_sharpe_ratio(r) > 0.95


def test_psr_about_half_for_zero_mean():
    rng = np.random.RandomState(1)
    r = pd.Series(rng.randn(120) * 0.03)          # zero mean -> ~50/50
    assert 0.3 < probabilistic_sharpe_ratio(r) < 0.7


def test_psr_decreases_with_higher_benchmark():
    rng = np.random.RandomState(2)
    r = pd.Series(rng.randn(120) * 0.03 + 0.015)
    assert probabilistic_sharpe_ratio(r, 0.0) > probabilistic_sharpe_ratio(r, 0.3)


def test_dsr_below_psr_and_stricter_with_more_trials():
    rng = np.random.RandomState(3)
    r = pd.Series(rng.randn(120) * 0.03 + 0.02)
    psr0 = probabilistic_sharpe_ratio(r, 0.0)
    few = deflated_sharpe_ratio(r, [0.2, 0.1, 0.15])
    many = deflated_sharpe_ratio(r, [0.2, 0.1, 0.15, 0.3, -0.1, 0.05, 0.25, 0.0])
    assert few["dsr"] <= psr0 + 1e-9          # deflation only lowers it
    assert many["sr0"] > few["sr0"]           # more trials -> higher hurdle
    assert many["dsr"] <= few["dsr"] + 1e-9


def test_dsr_needs_two_trials():
    r = pd.Series(np.random.RandomState(4).randn(60) * 0.03 + 0.01)
    assert np.isnan(deflated_sharpe_ratio(r, [0.2])["dsr"])


def test_pbo_range_and_guards():
    rng = np.random.RandomState(5)
    M = pd.DataFrame(rng.randn(80, 4) * 0.03)
    pbo = probability_of_backtest_overfitting(M, n_splits=8)
    assert 0.0 <= pbo <= 1.0
    assert np.isnan(probability_of_backtest_overfitting(M.iloc[:, :1]))  # <2 configs
    assert np.isnan(probability_of_backtest_overfitting(M, n_splits=7))  # odd splits


def test_pbo_low_when_one_config_dominates():
    rng = np.random.RandomState(6)
    base = rng.randn(96, 4) * 0.03
    base[:, 0] += 0.05                          # config 0 has a persistent edge
    pbo = probability_of_backtest_overfitting(pd.DataFrame(base), n_splits=8)
    assert pbo < 0.3                            # backtest winner rarely disappoints


def test_latest_month_staleness_flags_newly_sparse():
    rng = np.random.RandomState(0)
    rows = []
    base_months = ["2020-01", "2020-02", "2020-03"]
    for m in base_months:                      # baseline: fresh & stale populated
        for i in range(50):
            rows.append({"ym": m, "fresh": rng.randn(), "stale": rng.randn(),
                         "always_sparse": 0.0 if i < 30 else rng.randn()})
    for i in range(50):                        # latest month: 'stale' collapses to 0
        rows.append({"ym": "2020-04", "fresh": rng.randn(),
                     "stale": 0.0 if i < 40 else rng.randn(),
                     "always_sparse": 0.0 if i < 30 else rng.randn()})
    df = pd.DataFrame(rows)
    res = latest_month_staleness(df, base_months, "2020-04",
                                 ["fresh", "stale", "always_sparse"])
    assert "stale" in res["newly_stale"]          # collapsed this month
    assert "fresh" not in res["newly_stale"]       # still populated
    assert "always_sparse" not in res["newly_stale"]  # sparse in both -> not new
    assert res["n_features"] == 3


def test_survivorship_premium_measures_the_gap():
    """Panel beats the benchmark by a known amount; the helper must recover it."""
    from core.diagnostics import survivorship_premium
    months = [f"2015-{m:02d}" for m in range(1, 13)] + [f"2016-{m:02d}" for m in range(1, 13)]
    rows = []
    for m in months:
        for p in range(5):
            rows.append({"ym": m, "permno": p, "y_raw": 0.02, "spy_ret": 0.01})
    out = survivorship_premium(pd.DataFrame(rows))
    assert out["n_months"] == 24
    np.testing.assert_almost_equal(out["panel_ann"], 0.24, decimal=6)
    np.testing.assert_almost_equal(out["bench_ann"], 0.12, decimal=6)
    np.testing.assert_almost_equal(out["gap_ann"], 0.12, decimal=6)
    assert out["names_first"] == 5


def test_survivorship_premium_needs_enough_months():
    from core.diagnostics import survivorship_premium
    short = pd.DataFrame({"ym": ["2015-01"] * 5, "permno": range(5),
                          "y_raw": [0.02] * 5, "spy_ret": [0.01] * 5})
    assert survivorship_premium(short) == {}


def test_return_by_vol_decile_orders_and_locates_the_book():
    """High-vol decile is built to outperform; the book is built to sit in it."""
    from core.diagnostics import return_by_vol_decile
    rng = np.random.default_rng(11)
    months = [f"2016-{m:02d}" for m in range(1, 13)]
    rows = []
    for m in months:
        for p in range(100):
            vol = (p - 50) / 20.0                 # -2.5 .. +2.45
            rows.append({"ym": m, "permno": p, "vol_12m_xs": vol,
                         "y_raw": 0.01 + 0.02 * max(vol, 0) + rng.normal(0, 1e-6)})
    panel = pd.DataFrame(rows)
    # A book that only ever holds the very highest-vol names.
    holdings = {m: pd.DataFrame({"permno": range(90, 100)}) for m in months}

    out = return_by_vol_decile(panel, holdings=holdings)
    assert list(out.index) == list(range(1, 11))
    assert out.loc[10, "ann_return"] > out.loc[1, "ann_return"]
    np.testing.assert_almost_equal(out["share_of_book"].sum(), 1.0, decimal=6)
    assert out.loc[10, "share_of_book"] == 1.0
    assert out["n_obs"].sum() == len(panel)


def test_return_by_vol_decile_needs_data():
    from core.diagnostics import return_by_vol_decile
    assert return_by_vol_decile(pd.DataFrame({"ym": [], "permno": [],
                                              "vol_12m_xs": [], "y_raw": []})).empty


# --- effective number of bets ---------------------------------------------
# Position count is not diversification. These fixtures have analytic answers,
# so the tests need no seeds and no tolerance handwaving.

def _orthogonal_history(n_assets=4, scale=0.05):
    """A returns panel whose *sample* covariance is exactly diagonal.

    Hadamard columns 1.. are mutually orthogonal and have exactly zero mean, so
    the sample covariance is (T/(T-1)) * scale^2 * I to floating-point
    exactness. Random data would only be diagonal in expectation.
    """
    from scipy.linalg import hadamard
    months = ([f"2015-{m:02d}" for m in range(1, 13)]
              + [f"2016-{m:02d}" for m in range(1, 5)])          # T = 16
    permnos = list(range(10001, 10001 + n_assets))
    hist = pd.DataFrame(hadamard(len(months))[:, 1:1 + n_assets] * scale,
                        index=months, columns=permnos)
    return hist, months, permnos


def _equal_weight_book(permnos, months):
    w = np.ones(len(permnos)) / len(permnos)
    return {m: pd.DataFrame({"permno": permnos, "weight": w}) for m in months}


def test_effective_bets_counts_uncorrelated_names():
    """n identical, uncorrelated, equally weighted names is exactly n bets."""
    from core.diagnostics import effective_bets
    hist, months, permnos = _orthogonal_history(4)
    enb = effective_bets(_equal_weight_book(permnos, months[-2:]), hist)
    assert len(enb) == 2
    np.testing.assert_allclose(enb.values, 4.0, atol=1e-10)


def test_effective_bets_collapses_when_everything_moves_together():
    """n perfectly correlated names is one bet however many you hold.

    The covariance is singular here and the metric must still return a number,
    because it never inverts it.
    """
    from core.diagnostics import effective_bets
    rng = np.random.default_rng(5)
    months = ([f"2015-{m:02d}" for m in range(1, 13)]
              + [f"2016-{m:02d}" for m in range(1, 5)])
    permnos = list(range(10001, 10005))
    col = rng.normal(0, 0.04, len(months))
    hist = pd.DataFrame(np.column_stack([col] * 4), index=months, columns=permnos)
    enb = effective_bets(_equal_weight_book(permnos, months[-2:]), hist)
    np.testing.assert_allclose(enb.values, 1.0, atol=1e-8)


def test_effective_bets_sees_through_the_position_count():
    """Four names in two identical pairs is two bets. len(holdings) says four."""
    from core.diagnostics import effective_bets
    hist, months, _ = _orthogonal_history(2)
    a, b = hist.columns
    paired = pd.DataFrame({10001: hist[a], 10002: hist[a],
                           10003: hist[b], 10004: hist[b]}, index=months)
    book = _equal_weight_book([10001, 10002, 10003, 10004], months[-1:])
    assert len(book[months[-1]]) == 4
    np.testing.assert_allclose(effective_bets(book, paired).values, 2.0, atol=1e-8)


def test_effective_bets_is_point_in_time():
    """Returns realized after month m must not move the reading at m."""
    from core.diagnostics import effective_bets
    hist, months, permnos = _orthogonal_history(4)
    rng = np.random.default_rng(2)
    future = pd.DataFrame(rng.normal(0, 0.5, (6, 4)),
                          index=[f"2016-{m:02d}" for m in range(5, 11)],
                          columns=permnos)
    book = _equal_weight_book(permnos, months[-1:])
    pd.testing.assert_series_equal(
        effective_bets(book, hist),
        effective_bets(book, pd.concat([hist, future])),
    )


def test_effective_bets_is_scale_free_for_a_long_short_book():
    """Long-short weights sum to about zero, so gross normalization is required.

    Without it the metric is meaningless off a long-only book: only the
    denominator moves with the weight scale.
    """
    from core.diagnostics import effective_bets
    hist, months, permnos = _orthogonal_history(4)
    m = months[-1]
    ls = {m: pd.DataFrame({"permno": permnos, "weight": [0.5, 0.5, -0.5, -0.5]})}
    doubled = {m: pd.DataFrame({"permno": permnos, "weight": [1.0, 1.0, -1.0, -1.0]})}
    np.testing.assert_allclose(effective_bets(ls, hist).values,
                               effective_bets(doubled, hist).values, atol=1e-10)
    np.testing.assert_allclose(effective_bets(ls, hist).values, 4.0, atol=1e-10)


def test_effective_bets_degenerates_quietly():
    """Never raise, and never invent an answer out of missing data."""
    from core.diagnostics import effective_bets
    hist, months, permnos = _orthogonal_history(4)
    book = _equal_weight_book(permnos, months[-1:])

    # No history at all. An identity-covariance fallback would read N_eff = n
    # here — the most reassuring answer possible, from nothing.
    assert effective_bets(book, None).empty
    assert effective_bets(book, pd.DataFrame()).empty
    assert effective_bets({}, hist).empty

    too_few = _equal_weight_book(permnos[:2], months[-1:])
    assert len(effective_bets(too_few, hist)) == 1
    assert effective_bets(too_few, hist).isna().all()

    assert effective_bets(book, hist.iloc[:6]).isna().all()          # < min_obs

    no_weight = {months[-1]: pd.DataFrame({"permno": permnos})}
    assert effective_bets(no_weight, hist).isna().all()


def test_effective_bets_weighting_variants_agree_under_equal_weights():
    """The unweighted mean and the diversification ratio coincide at 1/N and
    diverge otherwise, so the flag has to be an explicit choice."""
    from core.diagnostics import effective_bets
    hist, months, permnos = _orthogonal_history(4)
    hist = hist * np.array([1.0, 1.0, 4.0, 4.0])       # unequal vols
    eq = _equal_weight_book(permnos, months[-1:])
    np.testing.assert_allclose(effective_bets(eq, hist).values,
                               effective_bets(eq, hist, weighted=True).values, atol=1e-10)
    tilted = {months[-1]: pd.DataFrame(
        {"permno": permnos, "weight": [0.7, 0.1, 0.1, 0.1]})}
    assert not np.isclose(effective_bets(tilted, hist).iloc[0],
                          effective_bets(tilted, hist, weighted=True).iloc[0])
