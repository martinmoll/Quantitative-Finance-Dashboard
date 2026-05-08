"""
Walk-forward prediction engine for the Alpha Model dashboard.

Design split:
  run_predictions()  — trains model, generates raw per-stock predictions (EXPENSIVE, ~30-60s)
  build_portfolio()  — applies portfolio params to cached predictions (CHEAP, <1s)

This enables instant portfolio-level param changes without rerunning the model.
"""

import pandas as pd
import numpy as np
from sklearn.base import clone
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_TARGET = 'y_xs'   # Standardized cross-sectional return — used for training
EVAL_TARGET = 'y_raw'   # Raw forward return — used for portfolio P&L evaluation
OOS_START = '2015-01'


# ---------------------------------------------------------------------------
# Market data helper
# ---------------------------------------------------------------------------

def compute_market_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a monthly market-return series from the panel data.

    Groups by 'ym', takes the first Mkt_RF and rf_ff observation per month,
    then reconstructs the SPY total return as Mkt_RF + rf_ff.

    Parameters
    ----------
    df : pd.DataFrame
        Raw panel data containing columns 'ym', 'Mkt_RF', 'rf_ff'.

    Returns
    -------
    pd.DataFrame
        Index = 'ym', columns = ['Mkt_RF', 'rf_ff', 'spy_ret'], sorted ascending.
    """
    market_monthly = (
        df.groupby('ym')[['Mkt_RF', 'rf_ff']]
        .first()
        .sort_index()
    )
    market_monthly['spy_ret'] = market_monthly['Mkt_RF'] + market_monthly['rf_ff']
    return market_monthly


# ---------------------------------------------------------------------------
# Walk-forward prediction generator
# ---------------------------------------------------------------------------

def run_predictions(
    df: pd.DataFrame,
    feature_builder,
    estimator,
    retrain_every: int = 12,
    progress_callback=None,
) -> dict:
    """Train the model walk-forward and collect per-stock raw predictions.

    Only the raw model output is stored here — vol_tilt and regime filtering
    are intentionally left to build_portfolio() so that portfolio-level params
    can be changed without retraining.

    Parameters
    ----------
    df : pd.DataFrame
        Full panel data.  Must contain 'ym', TRAIN_TARGET, EVAL_TARGET,
        'permno', 'sector', and optionally 'vol_12m_xs'.
    feature_builder : callable
        Function that takes a DataFrame slice and returns a feature matrix.
    estimator : sklearn estimator
        Unfitted (or fitted — clone() is always called before fitting).
    retrain_every : int
        Retrain the model every N OOS months.
    progress_callback : callable or None
        Called with (current_step: int, total_steps: int, month_str: str)
        after each OOS month is processed.  Use for progress bars.

    Returns
    -------
    dict
        Keys are month strings (e.g. '2015-01').
        Values are DataFrames with columns:
          permno, sector, pred, y_raw, vol_12m_xs
    """
    all_months = sorted(df['ym'].unique())
    oos_months = [m for m in all_months if m >= OOS_START]
    retrain_months = set(oos_months[::retrain_every])

    model = None
    predictions: dict = {}

    total = len(oos_months)

    for step, m in enumerate(oos_months):
        # ---- Retrain if scheduled ----
        if m in retrain_months:
            train = df[df['ym'] < m].dropna(subset=[TRAIN_TARGET])
            model = clone(estimator)
            model.fit(feature_builder(train), train[TRAIN_TARGET])

        test = df[df['ym'] == m].copy()

        # Need at least some stocks to be meaningful
        if len(test) < 2:
            if progress_callback is not None:
                progress_callback(step + 1, total, m)
            continue

        # Raw model predictions — no vol_tilt applied here
        test['pred'] = model.predict(feature_builder(test))

        # Carry through the columns needed downstream
        keep_cols = ['permno', 'pred', EVAL_TARGET]
        if 'sector' in test.columns:
            keep_cols.insert(1, 'sector')
        if 'vol_12m_xs' in test.columns:
            keep_cols.append('vol_12m_xs')

        predictions[m] = test[keep_cols].copy().reset_index(drop=True)

        if progress_callback is not None:
            progress_callback(step + 1, total, m)

    return predictions


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def build_portfolio(
    predictions: dict,
    K: int = 10,
    vol_tilt: float = 0.0,
    regime_lookback: int = 6,
    market_monthly: pd.DataFrame | None = None,
) -> dict:
    """Apply portfolio parameters to cached predictions and compute returns.

    Parameters
    ----------
    predictions : dict
        Output of run_predictions().  Keys = month strings, values = DataFrames
        with at least [permno, pred, y_raw] and optionally [vol_12m_xs].
    K : int
        Number of stocks in the long portfolio.
    vol_tilt : float
        Penalise high-volatility stocks before ranking:
        adjusted_pred = pred - vol_tilt * vol_12m_xs
        Set to 0.0 to disable.
    regime_lookback : int
        Number of months of trailing SPY return used to determine the regime.
        When trailing sum < 0 the portfolio goes to cash (return = 0.0).
        Ignored (regime always ON) when market_monthly is None.
    market_monthly : pd.DataFrame or None
        Output of compute_market_monthly().  Required for regime filtering.
        If None, regime filter is disabled.

    Returns
    -------
    dict with keys:
        monthly_returns : pd.Series   — monthly portfolio returns (OOS)
        holdings        : dict        — {month: list of permno in long portfolio}
        ic              : pd.Series   — monthly Spearman IC (after vol_tilt)
        turnover        : pd.Series   — monthly one-way turnover
    """
    # ---- Regime filter mask (shift(1) avoids look-ahead bias) ----
    regime_on: pd.Series | None = None
    if market_monthly is not None and regime_lookback > 0:
        trailing_spy = (
            market_monthly['spy_ret']
            .rolling(regime_lookback)
            .sum()
            .shift(1)
        )
        regime_on = trailing_spy >= 0  # True = invested, False = cash

    months = sorted(predictions.keys())

    monthly_returns: dict = {}
    holdings: dict = {}
    ic_vals: dict = {}
    turnover_vals: dict = {}
    prev_weights: pd.Series | None = None

    for m in months:
        df_m = predictions[m].copy()

        # ---- Apply vol tilt to scores before ranking ----
        if vol_tilt > 0.0 and 'vol_12m_xs' in df_m.columns:
            df_m['pred'] = df_m['pred'] - vol_tilt * df_m['vol_12m_xs'].fillna(0.0)

        # ---- Need enough stocks ----
        if len(df_m) < 2 * K:
            prev_weights = None  # portfolio doesn't exist this month
            continue

        # ---- IC (Spearman rank correlation of adjusted pred vs y_raw) ----
        valid = df_m[['pred', EVAL_TARGET]].dropna()
        if len(valid) > 10:
            ic_vals[m] = spearmanr(valid['pred'], valid[EVAL_TARGET])[0]
        else:
            ic_vals[m] = np.nan

        # ---- Select top-K ----
        top = df_m.nlargest(K, 'pred')
        port_ret = top[EVAL_TARGET].mean()

        # ---- Regime filter — go to cash when trailing SPY is negative ----
        if regime_on is not None and m in regime_on.index and not regime_on[m]:
            port_ret = 0.0

        monthly_returns[m] = port_ret
        holdings[m] = top['permno'].tolist()

        # ---- Turnover (one-way) ----
        curr_weights = pd.Series(0.0, index=df_m['permno'].values)
        curr_weights[top['permno'].values] = 1.0 / K

        if prev_weights is not None:
            aligned_curr, aligned_prev = curr_weights.align(prev_weights, fill_value=0.0)
            turnover_vals[m] = 0.5 * (aligned_curr - aligned_prev).abs().sum()
        else:
            turnover_vals[m] = np.nan

        prev_weights = curr_weights

    return {
        'monthly_returns': pd.Series(monthly_returns).sort_index(),
        'holdings': holdings,
        'ic': pd.Series(ic_vals).sort_index(),
        'turnover': pd.Series(turnover_vals).sort_index(),
    }


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------

def compute_perf(
    monthly_returns: pd.Series,
    name: str = '',
    ic: pd.Series | None = None,
    turnover: pd.Series | None = None,
) -> dict:
    """Compute annualised performance statistics.

    Parameters
    ----------
    monthly_returns : pd.Series
        Monthly portfolio returns.
    name : str
        Label for the 'Strategy' field in the output dict.
    ic : pd.Series or None
        Monthly IC values (optional).
    turnover : pd.Series or None
        Monthly turnover values (optional).

    Returns
    -------
    dict with keys:
        Strategy, SR, Ann Return, Ann Vol, MDD, Total Return,
        Mean IC, Mean Turnover
    """
    s = monthly_returns.dropna()
    if len(s) == 0:
        return {
            'Strategy': name, 'SR': np.nan,
            'Ann Return': np.nan, 'Ann Vol': np.nan,
            'MDD': np.nan, 'Total Return': np.nan,
            'Mean IC': np.nan, 'Mean Turnover': np.nan,
        }

    ann_ret = s.mean() * 12
    ann_vol = s.std() * np.sqrt(12)
    sr = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + s).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    total_ret = cum.iloc[-1] - 1

    mean_ic = ic.mean() if ic is not None and len(ic) > 0 else np.nan
    mean_turnover = (
        turnover.dropna().mean()
        if turnover is not None and len(turnover.dropna()) > 0
        else np.nan
    )

    return {
        'Strategy': name,
        'SR': round(float(sr), 4),
        'Ann Return': float(ann_ret),
        'Ann Vol': float(ann_vol),
        'MDD': float(mdd),
        'Total Return': float(total_ret),
        'Mean IC': float(mean_ic) if not np.isnan(mean_ic) else np.nan,
        'Mean Turnover': float(mean_turnover) if not np.isnan(mean_turnover) else np.nan,
    }
