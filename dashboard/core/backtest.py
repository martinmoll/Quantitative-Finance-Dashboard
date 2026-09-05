"""Walk-forward prediction engine.

Accepts AlphaModel instances from the registry
and supports both expanding and rolling windows.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

TRAIN_TARGET = "y_xs"
EVAL_TARGET = "y_raw"
OOS_START = "2015-01"


@dataclass
class BacktestResult:
    predictions: dict[str, pd.DataFrame]
    feature_importance: pd.DataFrame | None
    train_dates: list[str]
    model_params: dict
    tuned_params: dict[str, dict] = field(default_factory=dict)


def run_walk_forward(
    data: pd.DataFrame,
    model,
    feature_cols: list[str],
    oos_start: str = OOS_START,
    retrain_freq: int = 12,
    window_type: str = "expanding",
    rolling_window: int | None = None,
    auto_tune: bool = False,
    progress_callback=None,
) -> BacktestResult:
    all_months = sorted(data["ym"].unique())
    oos_months = [m for m in all_months if m >= oos_start]
    retrain_schedule = set(oos_months[::retrain_freq])

    fitted_model = None
    predictions: dict[str, pd.DataFrame] = {}
    train_dates: list[str] = []
    tuned_params: dict[str, dict] = {}
    total = len(oos_months)

    available_features = [c for c in feature_cols if c in data.columns]
    fit_features = available_features

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

            fit_features = _informative_features(train, available_features)
            if not fit_features:
                # Nothing to learn from yet. Reachable from the UI: the "Value
                # only" preset is the fundamentals block, which is empty before
                # 2025. Skip the retrain, as a too-short window is skipped.
                if progress_callback:
                    progress_callback(step + 1, total, m)
                continue

            if auto_tune:
                best_params = _tune_hyperparams(model, train, fit_features)
                if best_params is not None:
                    tuned_params[m] = best_params
                    from core.models import get_model
                    base_params = model.get_params()
                    model_type = base_params.get("model_type", "HGB")
                    tuned_model = get_model(model_type, best_params)
                    fitted_model = _fit_model(tuned_model, train, fit_features)
                else:
                    fitted_model = _clone_and_fit(model, train, fit_features)
            else:
                fitted_model = _clone_and_fit(model, train, fit_features)
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

        X_test = test[fit_features].fillna(0.0)
        test["pred"] = fitted_model.predict(X_test)

        keep = ["permno", "pred", EVAL_TARGET]
        if "sector" in test.columns:
            keep.insert(1, "sector")
        for col in ("vol_12m_xs", "ivol_xs"):
            # ivol_xs is carried so portfolio construction can cap idiosyncratic
            # volatility. The model concentrates the book at ivol_xs ~ +2.6,
            # which is where a survivor-only universe is most distorted.
            if col in test.columns:
                keep.append(col)

        predictions[m] = test[keep].reset_index(drop=True)

        if progress_callback:
            progress_callback(step + 1, total, m)

    importance = None
    if fitted_model is not None:
        importance = fitted_model.get_feature_importance()
        if importance is not None:
            importance = importance.to_frame("importance")

    return BacktestResult(
        predictions=predictions,
        feature_importance=importance,
        train_dates=train_dates,
        model_params=fitted_model.get_params() if fitted_model else {},
        tuned_params=tuned_params,
    )


def _informative_features(train: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Drop the columns that carry no information in this training window.

    The model is fed ``train[feature_cols].fillna(0.0)``, and the caller only
    checked that each column exists. A column that is all-NaN therefore arrived
    as constant zeros — and because these features are cross-sectionally
    standardized, zero reads as "exactly average", so a column with no data was
    indistinguishable from a stock with average everything. At the first Tier 1
    retrain, 33 of 52 columns were constant like this; by 2026 it is 9 of 52
    (the sue, revision and beat families, which have no data source in the
    repo).

    Measured: dropping them leaves the out-of-sample IC, annual return, Sharpe
    and worst month unchanged (HGB tier 2, OOS 2016-07: IC +0.0321, t +2.88,
    64.3% a year, SR 1.35, worst month -27.5% both with and without the
    guard), and halves the run time of an HGB walk-forward, 284s to 141s. The
    gain is run time and an honest feature-importance report, not return.

    Judged on the training window at every retrain, so a column that is empty
    early and real later — the fundamentals block, which starts in 2025 — is
    used from the retrain after it becomes real, with no look-ahead.
    """
    filled = train[feature_cols].fillna(0.0)
    return [c for c in feature_cols if filled[c].std() > 0]


def _clone_and_fit(model, train: pd.DataFrame, feature_cols: list[str]):
    """Clone model and fit on training data."""
    from core.models import get_model
    params = model.get_params()
    model_type = params.pop("model_type", "HGB")
    new_model = get_model(model_type, params)
    return _fit_model(new_model, train, feature_cols)


def _fit_model(model, train: pd.DataFrame, feature_cols: list[str]):
    """Fit a model on training data."""
    X = train[feature_cols].fillna(0.0)
    y = train[TRAIN_TARGET]
    model.fit(X, y)
    return model


def _tune_hyperparams(
    model,
    train: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = 3,
) -> dict | None:
    """Inner time-series CV to select hyperparameters via IC scoring."""
    from core.models import get_model, get_hp_grid

    base_params = model.get_params()
    model_type = base_params.get("model_type", "HGB")
    grid = get_hp_grid(model_type)

    if not grid:
        return None

    param_names = list(grid.keys())
    param_values = list(grid.values())
    combos = [dict(zip(param_names, vals)) for vals in product(*param_values)]

    train_months = sorted(train["ym"].unique())
    n_months = len(train_months)
    if n_months < 36:
        return None

    fold_size = 12
    folds = []
    for i in range(n_folds):
        val_end = n_months - i * fold_size
        val_start = val_end - fold_size
        if val_start < 12:
            break
        val_months = set(train_months[val_start:val_end])
        train_months_fold = set(train_months[:val_start])
        folds.append((train_months_fold, val_months))

    if not folds:
        return None

    best_ic = -np.inf
    best_params = None

    for combo in combos:
        fold_ics = []
        for train_set, val_set in folds:
            fold_train = train[train["ym"].isin(train_set)]
            fold_val = train[train["ym"].isin(val_set)]

            if len(fold_train) < 10 or len(fold_val) < 10:
                continue

            try:
                m = get_model(model_type, combo)
                X_tr = fold_train[feature_cols].fillna(0.0)
                y_tr = fold_train[TRAIN_TARGET]
                m.fit(X_tr, y_tr)

                X_val = fold_val[feature_cols].fillna(0.0)
                preds = m.predict(X_val)
                actual = fold_val[EVAL_TARGET].values

                valid_mask = ~np.isnan(actual)
                if valid_mask.sum() > 10:
                    ic = spearmanr(preds[valid_mask], actual[valid_mask])[0]
                    fold_ics.append(ic)
            except Exception:
                continue

        if fold_ics:
            mean_ic = np.mean(fold_ics)
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_params = combo

    return best_params
