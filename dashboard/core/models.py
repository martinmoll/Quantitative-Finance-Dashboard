"""Model registry with 6 alpha models and extensible registration.

All models conform to the AlphaModel protocol: fit, predict,
get_feature_importance, get_params.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Protocol, runtime_checkable
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.inspection import permutation_importance
import statsmodels.api as sm


@runtime_checkable
class AlphaModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def get_feature_importance(self) -> pd.Series | None: ...
    def get_params(self) -> dict: ...


class HGBModel:
    def __init__(self, **params):
        self._params = params
        self._model = HistGradientBoostingRegressor(
            max_depth=params.get("max_depth", 2),
            learning_rate=params.get("learning_rate", 0.05),
            min_samples_leaf=params.get("min_samples_leaf", 500),
            l2_regularization=params.get("l2_regularization", 0.1),
            max_iter=params.get("max_iter", 500),
            early_stopping=False,
            random_state=42,
        )
        self._feature_names: list[str] = []
        self._perm_importance: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self._model.fit(X, y)
        result = permutation_importance(
            self._model, X, y, n_repeats=5, random_state=42, n_jobs=-1
        )
        self._perm_importance = result.importances_mean

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.Series | None:
        if self._perm_importance is None:
            return None
        return pd.Series(
            self._perm_importance, index=self._feature_names
        ).sort_values(ascending=False)

    def get_params(self) -> dict:
        return {**self._params, "model_type": "HGB"}


class RFModel:
    def __init__(self, **params):
        self._params = params
        n_features = params.get("max_features", "sqrt")
        self._model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 4),
            max_features=n_features,
            min_samples_leaf=params.get("min_samples_leaf", 50),
            random_state=42,
            n_jobs=-1,
        )
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.Series | None:
        return pd.Series(
            self._model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)

    def get_params(self) -> dict:
        return {**self._params, "model_type": "RF"}


class LassoModel:
    def __init__(self, **params):
        self._params = params
        self._model = LassoCV(
            cv=params.get("cv", 5),
            max_iter=params.get("max_iter", 5000),
        )
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.Series | None:
        return pd.Series(
            np.abs(self._model.coef_), index=self._feature_names
        ).sort_values(ascending=False)

    def get_params(self) -> dict:
        return {**self._params, "model_type": "Lasso"}


class RidgeModel:
    def __init__(self, **params):
        self._params = params
        self._model = RidgeCV(
            alphas=np.logspace(-5, 2, 50),
        )
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.Series | None:
        return pd.Series(
            np.abs(self._model.coef_), index=self._feature_names
        ).sort_values(ascending=False)

    def get_params(self) -> dict:
        return {**self._params, "model_type": "Ridge"}


class ElasticNetModel:
    def __init__(self, **params):
        self._params = params
        self._model = ElasticNetCV(
            l1_ratio=params.get("l1_ratio", 0.5),
            cv=params.get("cv", 5),
            max_iter=params.get("max_iter", 5000),
        )
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.Series | None:
        return pd.Series(
            np.abs(self._model.coef_), index=self._feature_names
        ).sort_values(ascending=False)

    def get_params(self) -> dict:
        return {**self._params, "model_type": "ElasticNet"}


class FamaMacBethModel:
    """Fama-MacBeth: monthly cross-sectional OLS, average slopes."""

    def __init__(self, **params):
        self._params = params
        self._avg_coefs: pd.Series | None = None
        self._feature_names: list[str] = []
        self._date_col = params.get("date_col", "ym")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        X_with_const = sm.add_constant(X)
        self._avg_coefs = X_with_const.iloc[0:0].mean()
        all_coefs = []
        if self._date_col in X.columns:
            dates = X[self._date_col].unique()
            X_fit = X.drop(columns=[self._date_col])
            self._feature_names = list(X_fit.columns)
            for d in dates:
                mask = X[self._date_col] == d
                X_d = sm.add_constant(X_fit.loc[mask])
                y_d = y.loc[mask]
                if len(y_d) < X_d.shape[1] + 1:
                    continue
                try:
                    res = sm.OLS(y_d, X_d).fit()
                    all_coefs.append(res.params)
                except Exception:
                    continue
        else:
            self._feature_names = list(X.columns)
            X_with_const = sm.add_constant(X)
            self._avg_coefs = pd.Series(
                np.zeros(X_with_const.shape[1]), index=X_with_const.columns
            )
            all_coefs = [self._avg_coefs]

        if all_coefs:
            self._avg_coefs = pd.DataFrame(all_coefs).mean()
        else:
            self._avg_coefs = pd.Series(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_pred = X.drop(columns=[self._date_col], errors="ignore")
        X_const = sm.add_constant(X_pred)
        common = self._avg_coefs.index.intersection(X_const.columns)
        return (X_const[common] * self._avg_coefs[common]).sum(axis=1).values

    def get_feature_importance(self) -> pd.Series | None:
        if self._avg_coefs is None:
            return None
        coefs = self._avg_coefs.drop("const", errors="ignore")
        return coefs.abs().sort_values(ascending=False)

    def get_params(self) -> dict:
        return {**self._params, "model_type": "FamaMacBeth"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "HGB": HGBModel,
    "RF": RFModel,
    "Lasso": LassoModel,
    "Ridge": RidgeModel,
    "ElasticNet": ElasticNetModel,
    "FamaMacBeth": FamaMacBethModel,
}

_FEATURE_TIERS: dict[str, int] = {
    "HGB": 2, "RF": 2,
    "Lasso": 1, "Ridge": 1, "ElasticNet": 1, "FamaMacBeth": 1,
}

_DEFAULT_PARAMS: dict[str, dict] = {
    "HGB": {"max_depth": 2, "learning_rate": 0.05, "min_samples_leaf": 500, "l2_regularization": 0.1, "max_iter": 500},
    "RF": {"n_estimators": 200, "max_depth": 4, "max_features": "sqrt", "min_samples_leaf": 50},
    "Lasso": {"cv": 5, "max_iter": 5000},
    "Ridge": {},
    "ElasticNet": {"l1_ratio": 0.5, "cv": 5, "max_iter": 5000},
    "FamaMacBeth": {},
}

_PARAM_RANGES: dict[str, dict] = {
    "HGB": {
        "max_depth": {"min": 1, "max": 6, "default": 2, "step": 1},
        "learning_rate": {"min": 0.01, "max": 0.20, "default": 0.05, "step": 0.01},
        "min_samples_leaf": {"min": 100, "max": 2000, "default": 500, "step": 100},
        "l2_regularization": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.05},
        "max_iter": {"min": 100, "max": 1000, "default": 500, "step": 50},
    },
    "RF": {
        "n_estimators": {"min": 50, "max": 500, "default": 200, "step": 50},
        "max_depth": {"min": 1, "max": 8, "default": 4, "step": 1},
        "min_samples_leaf": {"min": 20, "max": 200, "default": 50, "step": 10},
    },
    "Lasso": {
        "cv": {"min": 3, "max": 10, "default": 5, "step": 1},
        "max_iter": {"min": 1000, "max": 10000, "default": 5000, "step": 1000},
    },
    "Ridge": {},
    "ElasticNet": {
        "l1_ratio": {"min": 0.1, "max": 0.9, "default": 0.5, "step": 0.1},
        "cv": {"min": 3, "max": 10, "default": 5, "step": 1},
        "max_iter": {"min": 1000, "max": 10000, "default": 5000, "step": 1000},
    },
    "FamaMacBeth": {},
}


def list_models() -> list[str]:
    return list(_REGISTRY.keys())


def get_model(name: str, params: dict) -> AlphaModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list_models()}")
    return _REGISTRY[name](**params)


def get_default_params(name: str) -> dict:
    return dict(_DEFAULT_PARAMS[name])


def get_param_ranges(name: str) -> dict:
    return dict(_PARAM_RANGES[name])


def get_feature_tier(name: str) -> int:
    return _FEATURE_TIERS[name]
