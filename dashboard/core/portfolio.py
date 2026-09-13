"""Portfolio construction: 5 methods + monthly series builder with regime filter."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

EVAL_TARGET = "y_raw"


def _apply_ivol_cap(df: pd.DataFrame, max_ivol_xs: float | None) -> pd.DataFrame:
    """Drop names whose idiosyncratic volatility exceeds the cap.

    The model concentrates the book around ``ivol_xs`` +2.6 — the extreme tail
    of the cross-section, and precisely where a survivor-only universe is most
    distorted, because every high-volatility name that went to zero is missing.
    This cap keeps the strategy out of that tail. It is a stopgap: it makes the
    backtest less wrong, not right. The fix is a point-in-time universe.

    Names with no ``ivol_xs`` are kept. Excluding them would shrink the universe
    on the basis of missing data, which is a different selection effect.
    """
    if max_ivol_xs is None or "ivol_xs" not in df.columns:
        return df
    return df[~(df["ivol_xs"] > max_ivol_xs)]


# Sector labels that mean "we do not know". They must not share one budget: the
# panel writes the literal string "Unknown" rather than NaN, so a plain NaN
# check alone would be dead code here.
_UNKNOWN_SECTORS = frozenset({"", "unknown", "n/a", "na", "none", "nan"})


def _apply_vol_tilt(df: pd.DataFrame, vol_tilt: float) -> pd.DataFrame:
    """Penalize ``pred`` by cross-sectional volatility before ranking.

    Selection ranks on the *tilted* score, so anything that needs to know which
    names the book will hold must go through here first. Returns a copy.
    """
    out = df.copy()
    if vol_tilt > 0.0 and "vol_12m_xs" in out.columns:
        out["pred"] = out["pred"] - vol_tilt * out["vol_12m_xs"].fillna(0.0)
    return out


def _select(
    df: pd.DataFrame,
    K: int,
    max_per_sector: int | None = None,
    largest: bool = True,
) -> pd.DataFrame:
    """The names one leg of the book holds, ranked by ``pred``.

    The single source of truth for selection. All five weighting methods run on
    what this returns, so a change here changes every method at once.

    With no cap this is exactly ``nlargest``/``nsmallest``. The early return is
    deliberate: ``sort_values(ascending=False)`` does not reproduce nlargest's
    tie-breaking — on ties it returns a different *set*, not merely a different
    order. Books are pickled under a config key, so a drifting tie-break would
    make a cached book disagree with a freshly computed one under that key.

    With a cap, candidates are walked best-first and a name is skipped once its
    sector budget is spent, so the book still reaches K by backfilling from
    deeper in the ranking. Names with an unknown sector are never blocked: each
    gets a private bucket. Collapsing them into one shared bucket would drop
    names because their label is missing, which is the selection-on-missing-data
    effect ``_apply_ivol_cap`` refuses to introduce.

    If the universe has too few sectors to fill K under the cap, the cap wins
    and the book is shorter than K. Every weighting method normalizes by the
    number selected, so a short book is still weighted correctly — just more
    concentrated per name, which is what "not enough breadth" should look like
    rather than a silently breached cap.
    """
    if max_per_sector is None or "sector" not in df.columns:
        return df.nlargest(K, "pred") if largest else df.nsmallest(K, "pred")

    # Reproduces nlargest/nsmallest ordering exactly: a stable sort leaves ties
    # in original row order and puts NaN predictions last, as pandas does.
    pred = df["pred"].to_numpy(dtype=float)
    order = np.argsort(-pred if largest else pred, kind="mergesort")
    sectors = df["sector"].to_numpy(object)

    counts: dict[object, int] = {}
    picked: list[int] = []
    for pos in order:
        if len(picked) >= K:
            break
        sec = sectors[pos]
        if pd.isna(sec) or str(sec).strip().lower() in _UNKNOWN_SECTORS:
            picked.append(pos)          # unknown sector: own bucket, never full
            continue
        if counts.get(sec, 0) >= max_per_sector:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        picked.append(pos)

    return df.iloc[picked]


def construct_portfolio(
    predictions: pd.DataFrame,
    method: str,
    K: int,
    strategy_type: str,
    K_short: int,
    vol_tilt: float,
    returns_history: pd.DataFrame | None = None,
    max_ivol_xs: float | None = None,
    max_per_sector: int | None = None,
    **method_params,
) -> pd.DataFrame:
    df = _apply_vol_tilt(_apply_ivol_cap(predictions, max_ivol_xs), vol_tilt)

    top = _select(df, K, max_per_sector, largest=True).copy()

    if strategy_type == "long_short":
        # The legs get independent sector budgets. A shared budget would let
        # long Tech crowd out short Tech, but long and short the same sector is
        # a hedge, not a concentration — and it would make the answer depend on
        # which leg happens to be counted first.
        bottom = _select(df, K_short, max_per_sector, largest=False).copy()
        long_weights = _compute_weights(top, method, returns_history, **method_params)
        short_weights = _compute_weights(bottom, method, returns_history, **method_params)
        top["weight"] = long_weights
        top["side"] = "long"
        bottom["weight"] = -short_weights
        bottom["side"] = "short"
        return pd.concat([top, bottom], ignore_index=True)
    else:
        weights = _compute_weights(top, method, returns_history, **method_params)
        top["weight"] = weights
        top["side"] = "long"
        return top.reset_index(drop=True)


def _compute_weights(
    selected: pd.DataFrame,
    method: str,
    returns_history: pd.DataFrame | None = None,
    **params,
) -> np.ndarray:
    n = len(selected)
    if n == 0:
        return np.array([])

    if method == "equal_weight":
        return np.ones(n) / n

    elif method == "score_weight":
        scores = selected["pred"].values
        scores_shifted = scores - scores.min() + 1e-8
        return scores_shifted / scores_shifted.sum()

    elif method == "inverse_vol":
        if "vol_12m_xs" in selected.columns:
            vols = selected["vol_12m_xs"].fillna(1.0).values
            vols = np.maximum(vols, 0.01)
        else:
            vols = np.ones(n)
        inv_vol = 1.0 / vols
        return inv_vol / inv_vol.sum()

    elif method == "erc":
        return _erc_weights(selected, returns_history)

    elif method == "mvo":
        max_weight = params.get("max_weight", 0.15)
        prev_w = params.get("prev_weights")
        tc_bps = params.get("tc_bps", 0.0)
        return _mvo_weights(selected, returns_history, max_weight,
                            prev_weights=prev_w, tc_bps=tc_bps)

    else:
        return np.ones(n) / n


def _erc_weights(
    selected: pd.DataFrame,
    returns_history: pd.DataFrame | None,
) -> np.ndarray:
    n = len(selected)
    permnos = selected["permno"].values

    cov = _get_cov_matrix(permnos, returns_history, n)

    def objective(w):
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e10
        # Work with *relative* risk contributions, which sum to 1. The absolute
        # contributions are ~1e-3 for monthly returns, so their squared spread
        # lands below SLSQP's default ftol (1e-6) and the optimizer would stop
        # on the first iteration and return the equal-weight starting point.
        rel_rc = (w * (cov @ w)) / port_var
        return np.sum((rel_rc - 1.0 / n) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0.001, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 500})

    weights = result.x if result.success else w0
    return weights / weights.sum()


def _mvo_weights(
    selected: pd.DataFrame,
    returns_history: pd.DataFrame | None,
    max_weight: float = 0.15,
    prev_weights: np.ndarray | None = None,
    tc_bps: float = 0.0,
) -> np.ndarray:
    n = len(selected)
    permnos = selected["permno"].values
    expected_returns = selected["pred"].values

    cov = _get_cov_matrix(permnos, returns_history, n, shrink=True)

    tc_frac = tc_bps / 10_000
    ref = prev_weights if prev_weights is not None else np.ones(n) / n

    def neg_sharpe(w):
        port_ret = w @ expected_returns
        if tc_frac > 0:
            port_ret -= tc_frac * np.sum(np.abs(w - ref))
        port_var = w @ cov @ w
        if port_var <= 0:
            return 0
        return -port_ret / np.sqrt(port_var)

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 500})

    weights = result.x if result.success else w0
    return weights / weights.sum()


def _get_cov_matrix(
    permnos: np.ndarray,
    returns_history: pd.DataFrame | None,
    n: int,
    shrink: bool = False,
) -> np.ndarray:
    if returns_history is not None:
        available = [p for p in permnos if p in returns_history.columns]
        if len(available) >= 2:
            hist = returns_history[available].dropna()
            if len(hist) >= 12:
                if shrink:
                    from sklearn.covariance import LedoitWolf
                    lw = LedoitWolf().fit(hist.values)
                    cov_full = lw.covariance_
                else:
                    cov_full = hist.cov().values

                idx_map = {p: i for i, p in enumerate(available)}
                cov = np.eye(n) * 0.01
                for i, p in enumerate(permnos):
                    if p in idx_map:
                        for j, q in enumerate(permnos):
                            if q in idx_map:
                                cov[i, j] = cov_full[idx_map[p], idx_map[q]]
                return cov

    return np.eye(n) * 0.01


def build_returns_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly realized returns as a ym x permno frame, for covariance estimation.

    Uses ``ret_1``, which is the return realized *over* month t and is therefore
    known at the month-t decision point. Do not build this from ``y_raw`` — that
    is the forward return and would be look-ahead.
    """
    return panel.pivot_table(index="ym", columns="permno", values="ret_1").sort_index()


def build_portfolio_series(
    predictions: dict[str, pd.DataFrame],
    method: str = "equal_weight",
    K: int = 10,
    strategy_type: str = "long_only",
    K_short: int = 10,
    vol_tilt: float = 0.0,
    regime_lookback: int = 6,
    market_monthly: pd.DataFrame | None = None,
    returns_history: pd.DataFrame | None = None,
    cost_bps: float = 10.0,
    max_ivol_xs: float | None = None,
    max_per_sector: int | None = None,
    cov_window: int | None = 60,
    **method_params,
) -> dict:
    """Build the monthly return series for a strategy.

    ``monthly_returns`` is **net of transaction costs** at ``cost_bps`` one-way.
    The gross series is returned alongside it as ``monthly_returns_gross`` so
    the two can be compared. Pass ``cost_bps=0.0`` for a frictionless run.

    ``cov_window`` bounds the covariance estimation window in months for the
    ERC and MVO methods; ``None`` restores the old expanding behaviour, which
    by 2026 was estimating on more than ten years of monthly data. The other
    three methods never look at the covariance.

    This bound is an estimator correctness fix, **not** drawdown protection. A
    shorter window was tested as an early warning and rejected: it moves the
    2026-06 book, which went on to lose 30.8%, only from the 10th to the
    26th-33rd percentile of ex-ante risk — still below median — and every
    estimator tried correlates *negatively* with next-month absolute return.
    Portfolio covariance on a ten-name book does not forecast this. Use
    ``diagnostics.effective_bets`` for that.

    ``regime_lookback`` gates on trailing SPY, so it cannot help either when a
    single theme unwinds inside a flat index: it was on throughout June and
    July 2026 because the market was up. An index-level filter cannot protect a
    book whose risk is not index risk.
    """
    regime_on = None
    if market_monthly is not None and regime_lookback > 0:
        trailing = market_monthly["spy_ret"].rolling(regime_lookback).sum().shift(1)
        regime_on = trailing >= 0

    months = sorted(predictions.keys())
    monthly_returns: dict[str, float] = {}
    gross_returns: dict[str, float] = {}
    holdings: dict[str, pd.DataFrame] = {}
    ic_vals: dict[str, float] = {}
    turnover_vals: dict[str, float] = {}
    prev_weights: pd.Series | None = None

    for m in months:
        # Cap first, so the breadth check below sees the tradable universe.
        df_m = _apply_ivol_cap(predictions[m], max_ivol_xs)
        # Row-count floor only. The sector cap does not shrink the universe, so
        # it does not belong here; its feasibility question (n_sectors * cap >= K)
        # is answered in _select by letting the book come up short.
        min_required = K + K_short if strategy_type == "long_short" else 2 * K
        if len(df_m) < min_required:
            prev_weights = None
            continue

        valid = df_m[["pred", EVAL_TARGET]].dropna()
        if len(valid) > 10:
            ic_vals[m] = spearmanr(valid["pred"], valid[EVAL_TARGET])[0]
        else:
            ic_vals[m] = np.nan

        mvo_prev = None
        if method == "mvo" and prev_weights is not None:
            # _mvo_weights lines this vector up *positionally* against the
            # selected frame, so it has to be the same names in the same order.
            # Going through _apply_vol_tilt/_select is what guarantees that.
            # Ranking on the untilted pred here — as this did — already picks a
            # different book for any vol_tilt > 0, so the turnover penalty was
            # priced against names the book does not hold.
            top_permnos = _select(
                _apply_vol_tilt(df_m, vol_tilt), K, max_per_sector, largest=True,
            )["permno"].values
            mvo_prev = np.array([prev_weights.get(p, 0.0) for p in top_permnos])
            pw_sum = mvo_prev.sum()
            if pw_sum > 0:
                mvo_prev = mvo_prev / pw_sum
            else:
                mvo_prev = None

        # Point-in-time slice: at month m only returns realized up to and
        # including m are known. ret_1[m] is the return over month m, so it is
        # available at the month-end decision point; anything after m is not.
        hist_m = None
        if returns_history is not None:
            hist_m = returns_history.loc[returns_history.index <= m]
            if cov_window:
                hist_m = hist_m.tail(int(cov_window))

        held = construct_portfolio(
            df_m, method=method, K=K, strategy_type=strategy_type,
            K_short=K_short, vol_tilt=vol_tilt,
            returns_history=hist_m,
            # max_ivol_xs is deliberately NOT forwarded: it is a universe filter
            # and df_m is already capped above. max_per_sector is a selection
            # rule rather than a filter, so it has to be forwarded here.
            max_per_sector=max_per_sector,
            prev_weights=mvo_prev, **method_params,
        )

        port_ret = (held["weight"] * held[EVAL_TARGET]).sum()

        if regime_on is not None and m in regime_on.index and not regime_on[m]:
            port_ret = 0.0

        holdings[m] = held

        curr_weights = pd.Series(0.0, index=df_m["permno"].values)
        held_weights = held.set_index("permno")["weight"]
        curr_weights.loc[held_weights.index] = held_weights.values

        if prev_weights is not None:
            aligned_c, aligned_p = curr_weights.align(prev_weights, fill_value=0.0)
            traded = 0.5 * (aligned_c - aligned_p).abs().sum()
            turnover_vals[m] = traded
        else:
            # First month builds the whole book from cash, so it is a full
            # one-way turnover. Reported as NaN (there is no prior month to
            # compare against) but it still costs money to trade.
            traded = 1.0
            turnover_vals[m] = np.nan

        # Turnover is 0.5*sum|dw|, so doubling recovers the notional traded.
        gross_returns[m] = port_ret
        monthly_returns[m] = port_ret - traded * cost_bps / 10_000 * 2

        prev_weights = curr_weights

    return {
        "monthly_returns": pd.Series(monthly_returns).sort_index(),
        "monthly_returns_gross": pd.Series(gross_returns).sort_index(),
        "holdings": holdings,
        "ic": pd.Series(ic_vals).sort_index(),
        "turnover": pd.Series(turnover_vals).sort_index(),
        "cost_bps": cost_bps,
    }
