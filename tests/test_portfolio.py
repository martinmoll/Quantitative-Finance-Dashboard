import numpy as np
import pandas as pd
from core.portfolio import construct_portfolio, build_portfolio_series


def _make_month_df(n=50):
    np.random.seed(42)
    return pd.DataFrame({
        "permno": range(10001, 10001 + n),
        "pred": np.random.randn(n) * 0.1,
        "y_raw": np.random.randn(n) * 0.08,
        "vol_12m_xs": np.abs(np.random.randn(n)) * 0.5 + 0.5,
        "sector": ["Tech", "Finance", "Health", "Energy", "Consumer"] * (n // 5),
    })


def test_equal_weight():
    df = _make_month_df()
    result = construct_portfolio(df, method="equal_weight", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    assert "weight" in result.columns
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0)


def test_score_weight():
    df = _make_month_df()
    result = construct_portfolio(df, method="score_weight", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=5)


def test_inverse_vol():
    df = _make_month_df()
    result = construct_portfolio(df, method="inverse_vol", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=5)


def test_erc():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    result = construct_portfolio(
        df, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)
    assert (result["weight"] > 0).all()


def test_mvo():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    result = construct_portfolio(
        df, method="mvo", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)
    assert (result["weight"] >= -0.001).all()
    assert (result["weight"] <= 0.151).all()


def test_mvo_tc_aware():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    prev_w = np.ones(10) / 10
    result = construct_portfolio(
        df, method="mvo", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
        prev_weights=prev_w, tc_bps=50.0,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)


def test_long_short():
    df = _make_month_df()
    result = construct_portfolio(df, method="equal_weight", K=10,
                                strategy_type="long_short", K_short=5, vol_tilt=0.0)
    assert len(result) == 15
    assert "side" in result.columns
    assert (result[result["side"] == "long"]["weight"] > 0).all()
    assert (result[result["side"] == "short"]["weight"] < 0).all()


def test_build_portfolio_series(sample_predictions):
    result = build_portfolio_series(
        predictions=sample_predictions,
        method="equal_weight",
        K=10,
        strategy_type="long_only",
        K_short=10,
        vol_tilt=0.05,
        regime_lookback=0,
    )
    assert "monthly_returns" in result
    assert "holdings" in result
    assert "ic" in result
    assert "turnover" in result
    assert len(result["monthly_returns"]) > 0


# --- returns_history plumbing and point-in-time guard -----------------------
# The covariance matrix was always the identity, because no caller ever passed
# returns_history. These tests pin both the fix and the point-in-time rule that
# the fix must not break.

def _series_inputs(n_months=6, n_stocks=30):
    np.random.seed(7)
    months = [f"2015-{m:02d}" for m in range(1, n_months + 1)]
    permnos = list(range(10001, 10001 + n_stocks))
    preds = {}
    for m in months:
        preds[m] = pd.DataFrame({
            "permno": permnos,
            "pred": np.random.randn(n_stocks) * 0.1,
            "y_raw": np.random.randn(n_stocks) * 0.08,
            "vol_12m_xs": np.abs(np.random.randn(n_stocks)) * 0.5 + 0.5,
        })
    return months, permnos, preds


def test_build_returns_history_shape():
    from core.portfolio import build_returns_history
    panel = pd.DataFrame({
        "ym": ["2015-01", "2015-01", "2015-02", "2015-02"],
        "permno": [1, 2, 1, 2],
        "ret_1": [0.01, 0.02, 0.03, 0.04],
    })
    hist = build_returns_history(panel)
    assert list(hist.index) == ["2015-01", "2015-02"]
    assert set(hist.columns) == {1, 2}
    assert hist.loc["2015-02", 1] == 0.03


def test_erc_weights_use_the_covariance():
    """With a real covariance, ERC must not collapse to equal weight."""
    months, permnos, preds = _series_inputs()
    rng = np.random.default_rng(0)
    # Two blocks with very different volatility -> ERC must tilt away from 1/N.
    base = rng.normal(0, 0.01, (36, len(permnos)))
    base[:, :10] *= 8.0
    hist = pd.DataFrame(base, index=[f"2012-{i:02d}" for i in range(1, 13)]
                        + [f"2013-{i:02d}" for i in range(1, 13)]
                        + [f"2014-{i:02d}" for i in range(1, 13)],
                        columns=permnos)
    out = build_portfolio_series(
        predictions=preds, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, returns_history=hist,
    )
    w = out["holdings"][months[0]]["weight"].values
    assert not np.allclose(w, np.ones(len(w)) / len(w), atol=1e-3), \
        "ERC collapsed to equal weight -- covariance was ignored"


def test_covariance_is_point_in_time():
    """Weights at month m must not change when data AFTER m is added."""
    months, permnos, preds = _series_inputs()
    rng = np.random.default_rng(1)
    past_idx = [f"2014-{i:02d}" for i in range(1, 13)]
    past = pd.DataFrame(rng.normal(0, 0.02, (12, len(permnos))),
                        index=past_idx, columns=permnos)

    # A future block with wildly different covariance structure. It must start
    # strictly AFTER the month under test: ret_1[2015-01] is the return over
    # January and is legitimately known at the January decision point, so only
    # 2015-02 onward counts as future information.
    future_idx = [f"2015-{i:02d}" for i in range(2, 8)]
    future = pd.DataFrame(rng.normal(0, 0.50, (6, len(permnos))),
                          index=future_idx, columns=permnos)
    with_future = pd.concat([past, future])

    out_past = build_portfolio_series(
        predictions=preds, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, returns_history=past,
    )
    out_full = build_portfolio_series(
        predictions=preds, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, returns_history=with_future,
    )
    w_past = out_past["holdings"][months[0]]["weight"].values
    w_full = out_full["holdings"][months[0]]["weight"].values
    np.testing.assert_allclose(
        w_past, w_full, atol=1e-6,
        err_msg="Future returns changed the weights -- covariance is not point-in-time",
    )


def test_costs_reduce_returns_and_gross_is_kept():
    months, permnos, preds = _series_inputs()
    free = build_portfolio_series(
        predictions=preds, method="equal_weight", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, cost_bps=0.0,
    )
    costed = build_portfolio_series(
        predictions=preds, method="equal_weight", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, cost_bps=10.0,
    )
    # Gross is identical; net is strictly worse in every month.
    pd.testing.assert_series_equal(
        free["monthly_returns_gross"], costed["monthly_returns_gross"])
    assert (costed["monthly_returns"] < costed["monthly_returns_gross"]).all()
    # A frictionless run leaves net == gross.
    pd.testing.assert_series_equal(
        free["monthly_returns"], free["monthly_returns_gross"], check_names=False)


def test_first_month_is_charged_a_full_build():
    months, permnos, preds = _series_inputs(n_months=1)
    out = build_portfolio_series(
        predictions=preds, method="equal_weight", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, cost_bps=10.0,
    )
    m0 = months[0]
    drag = out["monthly_returns_gross"][m0] - out["monthly_returns"][m0]
    np.testing.assert_almost_equal(drag, 1.0 * 10.0 / 10_000 * 2, decimal=10)


# --- idiosyncratic volatility cap ------------------------------------------
# The model concentrates the book at ivol_xs ~ +2.6, which is exactly where a
# survivor-only universe is most distorted. The cap is a stopgap that keeps the
# strategy out of that tail until the universe is point-in-time.

def _ivol_month(n=60):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "permno": range(10001, 10001 + n),
        # Highest predictions land on the highest-ivol names, mimicking the
        # real model, so an uncapped run must pick them.
        "pred": np.linspace(-1, 1, n),
        "y_raw": rng.normal(0, 0.08, n),
        "vol_12m_xs": np.linspace(-2, 3, n),
        "ivol_xs": np.linspace(-2, 3, n),
    })


def test_ivol_cap_excludes_the_tail():
    from core.portfolio import construct_portfolio
    df = _ivol_month()
    uncapped = construct_portfolio(df, method="equal_weight", K=10,
                                   strategy_type="long_only", K_short=10, vol_tilt=0.0)
    capped = construct_portfolio(df, method="equal_weight", K=10,
                                 strategy_type="long_only", K_short=10, vol_tilt=0.0,
                                 max_ivol_xs=1.0)
    assert uncapped["ivol_xs"].max() > 1.0, "fixture should tempt the model into the tail"
    assert capped["ivol_xs"].max() <= 1.0
    assert capped["ivol_xs"].mean() < uncapped["ivol_xs"].mean()
    np.testing.assert_almost_equal(capped["weight"].sum(), 1.0, decimal=6)


def test_ivol_cap_keeps_names_with_no_ivol():
    """A missing ivol must not silently shrink the universe."""
    from core.portfolio import construct_portfolio
    df = _ivol_month()
    df.loc[df.index[-5:], "ivol_xs"] = np.nan
    capped = construct_portfolio(df, method="equal_weight", K=10,
                                 strategy_type="long_only", K_short=10, vol_tilt=0.0,
                                 max_ivol_xs=0.0)
    assert capped["ivol_xs"].isna().any()


def test_ivol_cap_flows_through_the_series_builder():
    months, permnos, preds = _series_inputs()
    rng = np.random.default_rng(4)
    for m in preds:
        preds[m]["ivol_xs"] = rng.uniform(-2, 3, len(preds[m]))
    out = build_portfolio_series(
        predictions=preds, method="equal_weight", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, max_ivol_xs=1.0, cost_bps=0.0,
    )
    for m, held in out["holdings"].items():
        assert held["ivol_xs"].max() <= 1.0, f"{m} breached the cap"


# --- MVO turnover reference must match the book it prices ------------------
# _mvo_weights applies its turnover penalty positionally: `np.abs(w - ref)`
# lines ref[i] up with selected.iloc[i]. build_portfolio_series built that ref
# by ranking on the *raw* pred while the book itself is selected on the
# vol-tilted pred, so at any vol_tilt > 0 the penalty priced names the book
# does not hold. The UI default is vol_tilt=0.05, so this fired on every run.

def _tilt_month(n=30, seed=11):
    """Predictions where the vol tilt genuinely reorders the top of the book."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "permno": range(10001, 10001 + n),
        "pred": rng.normal(0, 0.1, n),
        "y_raw": rng.normal(0, 0.08, n),
        # Large spread, so a 0.05 tilt moves names across the K boundary.
        "vol_12m_xs": rng.normal(0, 2.0, n),
    })


def test_mvo_turnover_reference_mirrors_the_book():
    """The reference must be the book's own names, in the book's own order."""
    rng = np.random.default_rng(3)
    months = ["2015-01", "2015-02"]
    preds = {m: _tilt_month(seed=11 + i) for i, m in enumerate(months)}
    hist = pd.DataFrame(rng.normal(0, 0.05, (24, 30)),
                        index=[f"2013-{m:02d}" for m in range(1, 13)]
                              + [f"2014-{m:02d}" for m in range(1, 13)],
                        columns=range(10001, 10031))
    vol_tilt, tc = 0.05, 50.0

    out = build_portfolio_series(
        predictions=preds, method="mvo", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=vol_tilt, regime_lookback=0,
        returns_history=hist, cost_bps=0.0, tc_bps=tc,
    )

    # Rebuild month 2 by hand with a reference aligned to the book it prices.
    m1, m2 = months
    prev = out["holdings"][m1].set_index("permno")["weight"]
    tilted = preds[m2].copy()
    tilted["pred"] = tilted["pred"] - vol_tilt * tilted["vol_12m_xs"].fillna(0.0)
    book2 = tilted.nlargest(10, "pred")["permno"]

    naive = preds[m2].nlargest(10, "pred")["permno"]
    assert list(naive) != list(book2), "fixture must expose the mismatch"

    ref = np.array([prev.get(p, 0.0) for p in book2])
    ref = ref / ref.sum() if ref.sum() > 0 else None
    expected = construct_portfolio(
        preds[m2], method="mvo", K=10, strategy_type="long_only", K_short=10,
        vol_tilt=vol_tilt, returns_history=hist.loc[hist.index <= m2],
        prev_weights=ref, tc_bps=tc,
    )

    assert list(out["holdings"][m2]["permno"]) == list(expected["permno"])
    np.testing.assert_allclose(
        out["holdings"][m2]["weight"].to_numpy(),
        expected["weight"].to_numpy(), atol=1e-8,
    )


# --- sector concentration cap ---------------------------------------------
# Position count is not diversification. In July 2026 the book held nine of ten
# names in the semiconductor supply chain and lost 30.8% gross while the market
# rose 1.4%. The cap forces breadth by skipping a name whose sector budget is
# spent and backfilling from deeper in the ranking, so the book still holds K.
# Capping on `industry` was measured and rejected: the theme spanned six
# industry labels, so capping semicap just backfilled with semiconductors.

def _sector_month(n=40, n_sectors=4):
    """Predictions ranked so the top of the book is a single sector.

    ``pred`` descends with row order and each sector occupies one contiguous
    block, so an uncapped run holds nothing but Tech and a capped run has to
    reach past it into names it would otherwise never see.
    """
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"][:n_sectors]
    per = n // n_sectors
    return pd.DataFrame({
        "permno": range(10001, 10001 + n),
        "pred": np.linspace(1.0, -1.0, n),
        "y_raw": np.zeros(n),
        "vol_12m_xs": np.ones(n),
        "sector": [sectors[i // per] for i in range(n)],
    })


def test_sector_cap_binds_and_backfills_in_pred_order():
    df = _sector_month()
    uncapped = construct_portfolio(df, method="equal_weight", K=10,
                                   strategy_type="long_only", K_short=10, vol_tilt=0.0)
    capped = construct_portfolio(df, method="equal_weight", K=10,
                                 strategy_type="long_only", K_short=10, vol_tilt=0.0,
                                 max_per_sector=3)
    assert uncapped["sector"].value_counts().max() == 10, \
        "fixture should tempt the model into a single sector"
    assert capped["sector"].value_counts().max() == 3
    # Best three of each sector in turn, not an arbitrary reshuffle.
    assert list(capped["permno"]) == [10001, 10002, 10003, 10011, 10012, 10013,
                                      10021, 10022, 10023, 10031]
    np.testing.assert_almost_equal(capped["weight"].sum(), 1.0, decimal=6)


def test_sector_cap_keeps_the_book_at_K():
    """Skipping a full sector must not shrink the book."""
    df = _sector_month(n=50, n_sectors=5)
    for cap in (2, 3, 5, 10):
        held = construct_portfolio(df, method="equal_weight", K=10,
                                   strategy_type="long_only", K_short=10,
                                   vol_tilt=0.0, max_per_sector=cap)
        assert len(held) == 10, f"cap={cap} shrank the book"
        assert held["sector"].value_counts().max() <= cap


def test_sector_cap_none_reproduces_nlargest_exactly():
    """The default path must stay bit-identical to today's nlargest.

    nlargest's tie-breaking is not reproduced by sort_values(ascending=False):
    on ties they return different *sets*, not just a different order (measured:
    189 of 200 random draws). Books are pickled under a config key, so a
    drifting tie-break would make a cached book disagree with a freshly
    computed one under that same key.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 40
        df = pd.DataFrame({
            "permno": range(10001, 10001 + n),
            "pred": rng.choice([0.9, 0.5, 0.5, 0.5, 0.2, -0.1], size=n),
            "y_raw": np.zeros(n),
            "sector": ["Tech", "Finance", "Health", "Energy"] * (n // 4),
        })
        held = construct_portfolio(df, method="equal_weight", K=10,
                                   strategy_type="long_only", K_short=10,
                                   vol_tilt=0.0, max_per_sector=None)
        assert list(held["permno"]) == list(df.nlargest(10, "pred")["permno"])


def test_sector_cap_is_inert_without_a_sector_column():
    """Prediction frames cached before sector was carried have no column."""
    df = _sector_month().drop(columns=["sector"])
    held = construct_portfolio(df, method="equal_weight", K=10,
                               strategy_type="long_only", K_short=10,
                               vol_tilt=0.0, max_per_sector=1)
    assert list(held["permno"]) == list(df.nlargest(10, "pred")["permno"])


def test_sector_cap_never_blocks_an_unknown_sector():
    """Unknown sectors get a private bucket each, not one shared one.

    Sharing a bucket would drop names because their label is missing — the
    selection-on-missing-data effect the ivol cap refuses to introduce. The
    panel writes the literal string "Unknown" (123 rows, all in the OOS
    window) rather than NaN, so both spellings have to be handled.
    """
    df = _sector_month()
    df.loc[df.index[:6], "sector"] = np.nan
    df.loc[df.index[6:10], "sector"] = "Unknown"
    held = construct_portfolio(df, method="equal_weight", K=10,
                               strategy_type="long_only", K_short=10,
                               vol_tilt=0.0, max_per_sector=1)
    assert list(held["permno"]) == list(range(10001, 10011))


def test_sector_cap_wins_when_the_universe_lacks_breadth():
    """2 sectors x a cap of 2 cannot fill 10 names.

    The book comes up short rather than silently breaching the cap. Weights
    still normalize, so a thin book is more concentrated per name — which is
    what "no breadth available" should look like.
    """
    df = _sector_month(n=40, n_sectors=2)
    held = construct_portfolio(df, method="equal_weight", K=10,
                               strategy_type="long_only", K_short=10,
                               vol_tilt=0.0, max_per_sector=2)
    assert len(held) == 4
    np.testing.assert_almost_equal(held["weight"].sum(), 1.0, decimal=6)


def test_sector_cap_budgets_the_two_legs_independently():
    """Long Tech and short Tech is a hedge, not a concentration."""
    df = _sector_month()
    held = construct_portfolio(df, method="equal_weight", K=10,
                               strategy_type="long_short", K_short=10,
                               vol_tilt=0.0, max_per_sector=3)
    longs = held[held["side"] == "long"]
    shorts = held[held["side"] == "short"]
    assert len(longs) == 10 and len(shorts) == 10
    assert longs["sector"].value_counts().max() == 3
    assert shorts["sector"].value_counts().max() == 3
    assert set(longs["sector"]) & set(shorts["sector"]), "legs share no budget"


def test_sector_cap_flows_through_the_series_builder():
    months, permnos, preds = _series_inputs(n_stocks=40)
    sectors = ["Tech", "Finance", "Health", "Energy"]
    for m in preds:
        preds[m]["sector"] = [sectors[i % 4] for i in range(len(preds[m]))]
    out = build_portfolio_series(
        predictions=preds, method="equal_weight", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, max_per_sector=3, cost_bps=0.0,
    )
    assert out["holdings"]
    for m, held in out["holdings"].items():
        assert held["sector"].value_counts().max() <= 3, f"{m} breached the cap"
        assert len(held) == 10


# --- covariance estimation window -----------------------------------------
# _get_cov_matrix has no window of its own, and build_portfolio_series fed it
# an expanding slice, so by 2026 ERC and MVO were estimating on 10+ years of
# monthly data. This is an estimator fix, not drawdown protection — see the
# build_portfolio_series docstring for the measurement that rejected the
# early-warning framing.

def _cov_window_inputs(n_hist=120, n_stocks=30):
    rng = np.random.default_rng(19)
    hist_months = [f"{y}-{m:02d}" for y in range(2005, 2015) for m in range(1, 13)]
    hist_months = hist_months[-n_hist:]
    hist = pd.DataFrame(rng.normal(0, 0.05, (n_hist, n_stocks)),
                        index=hist_months, columns=range(10001, 10001 + n_stocks))
    preds = {"2015-01": pd.DataFrame({
        "permno": range(10001, 10001 + n_stocks),
        "pred": rng.normal(0, 0.1, n_stocks),
        "y_raw": rng.normal(0, 0.08, n_stocks),
        "vol_12m_xs": np.ones(n_stocks),
    })}
    return hist, preds


def test_cov_window_trims_the_estimation_history():
    """A 60-month window must equal passing only the last 60 months."""
    hist, preds = _cov_window_inputs()
    kw = dict(predictions=preds, method="erc", K=10, strategy_type="long_only",
              K_short=10, vol_tilt=0.0, regime_lookback=0, cost_bps=0.0)
    windowed = build_portfolio_series(returns_history=hist, cov_window=60, **kw)
    pretrimmed = build_portfolio_series(
        returns_history=hist.tail(60), cov_window=None, **kw)
    np.testing.assert_allclose(
        windowed["holdings"]["2015-01"]["weight"].to_numpy(),
        pretrimmed["holdings"]["2015-01"]["weight"].to_numpy(), atol=1e-10,
    )


def test_cov_window_none_keeps_the_expanding_estimate():
    """None restores the old behaviour, so a pinned config can reproduce it."""
    hist, preds = _cov_window_inputs()
    kw = dict(predictions=preds, method="erc", K=10, strategy_type="long_only",
              K_short=10, vol_tilt=0.0, regime_lookback=0, cost_bps=0.0)
    expanding = build_portfolio_series(returns_history=hist, cov_window=None, **kw)
    windowed = build_portfolio_series(returns_history=hist, cov_window=60, **kw)
    assert not np.allclose(expanding["holdings"]["2015-01"]["weight"].to_numpy(),
                           windowed["holdings"]["2015-01"]["weight"].to_numpy())
