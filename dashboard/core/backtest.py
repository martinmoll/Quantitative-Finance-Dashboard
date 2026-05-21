"""Walk-forward prediction engine.

Extracted from engine.py. Now accepts AlphaModel instances from the registry
and supports both expanding and rolling windows.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

TRAIN_TARGET = "y_xs"
EVAL_TARGET = "y_raw"
OOS_START = "2015-01"


@dataclass
class BacktestResult:
    predictions: dict[str, pd.DataFrame]
    monthly_returns: pd.Series
    ic: pd.Series
    holdings: dict[str, pd.DataFrame]
    turnover: pd.Series
    feature_importance: pd.DataFrame | None
    train_dates: list[str]
    model_params: dict


def run_walk_forward(
    data: pd.DataFrame,
    model,
    feature_cols: list[str],
    oos_start: str = OOS_START,
    retrain_freq: int = 12,
    window_type: str = "expanding",
    rolling_window: int | None = None,
    progress_callback=None,
) -> BacktestResult:
    all_months = sorted(data["ym"].unique())
    oos_months = [m for m in all_months if m >= oos_start]
    retrain_schedule = set(oos_months[::retrain_freq])

    fitted_model = None
    predictions: dict[str, pd.DataFrame] = {}
    train_dates: list[str] = []
    total = len(oos_months)

    available_features = [c for c in feature_cols if c in data.columns]

    for step, m in enumerate(oos_months):
        if m in retrain_schedule:
            if window_type == "rolling" and rolling_window is not None:
                cutoff_idx = all_months.index(m)
                start_idx = max(0, cutoff_idx - rolling_window)
                train_months = all_months[start_idx:cutoff_idx]
                train = data[data["ym"].isin(train_months)].dropna(subset=[TRAIN_TARGET])
            else:
                train = data[data["ym"] < m].dropna(subset=[TRAIN_TARGET])

            if len(train) < 10:
                if progress_callback:
                    progress_callback(step + 1, total, m)
                continue

            fitted_model = _clone_and_fit(model, train, available_features)
            train_dates.append(m)

        if fitted_model is None:
            if progress_callback:
                progress_callback(step + 1, total, m)
            continue

        test = data[data["ym"] == m].copy()
        if len(test) < 2:
            if progress_callback:
                progress_callback(step + 1, total, m)
            continue

        X_test = test[available_features].fillna(0.0)
        test["pred"] = fitted_model.predict(X_test)

        keep = ["permno", "pred", EVAL_TARGET]
        if "sector" in test.columns:
            keep.insert(1, "sector")
        if "vol_12m_xs" in test.columns:
            keep.append("vol_12m_xs")

        predictions[m] = test[keep].reset_index(drop=True)

        if progress_callback:
            progress_callback(step + 1, total, m)

    monthly_returns, holdings, ic, turnover = _compute_basic_portfolio(predictions)

    importance = None
    if fitted_model is not None:
        importance = fitted_model.get_feature_importance()
        if importance is not None:
            importance = importance.to_frame("importance")

    return BacktestResult(
        predictions=predictions,
        monthly_returns=monthly_returns,
        ic=ic,
        holdings=holdings,
        turnover=turnover,
        feature_importance=importance,
        train_dates=train_dates,
        model_params=fitted_model.get_params() if fitted_model else {},
    )


def _clone_and_fit(model, train: pd.DataFrame, feature_cols: list[str]):
    """Clone model and fit on training data."""
    from core.models import get_model
    params = model.get_params()
    model_type = params.pop("model_type", "HGB")
    new_model = get_model(model_type, params)
    X = train[feature_cols].fillna(0.0)
    y = train[TRAIN_TARGET]
    new_model.fit(X, y)
    return new_model


def _compute_basic_portfolio(predictions: dict[str, pd.DataFrame], K: int = 10):
    """Compute equal-weight top-K portfolio from raw predictions."""
    months = sorted(predictions.keys())
    returns_dict: dict[str, float] = {}
    holdings_dict: dict[str, pd.DataFrame] = {}
    ic_dict: dict[str, float] = {}
    turnover_dict: dict[str, float] = {}
    prev_permnos: set | None = None

    for m in months:
        df_m = predictions[m]
        if len(df_m) < K:
            continue

        valid = df_m[["pred", EVAL_TARGET]].dropna()
        if len(valid) > 10:
            ic_dict[m] = spearmanr(valid["pred"], valid[EVAL_TARGET])[0]
        else:
            ic_dict[m] = np.nan

        top = df_m.nlargest(K, "pred")
        returns_dict[m] = top[EVAL_TARGET].mean()
        holdings_dict[m] = top

        curr_permnos = set(top["permno"])
        if prev_permnos is not None:
            overlap = len(curr_permnos & prev_permnos)
            turnover_dict[m] = 1.0 - overlap / K
        else:
            turnover_dict[m] = np.nan
        prev_permnos = curr_permnos

    return (
        pd.Series(returns_dict).sort_index(),
        holdings_dict,
        pd.Series(ic_dict).sort_index(),
        pd.Series(turnover_dict).sort_index(),
    )
