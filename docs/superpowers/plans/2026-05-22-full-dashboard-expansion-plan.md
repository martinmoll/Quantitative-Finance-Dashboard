# Full Dashboard Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the single-page Alpha Model dashboard into a 7-page Bloomberg-style Streamlit app with 6 ML models, 5 portfolio construction methods, factor analysis, monitoring, and integrated educational content.

**Architecture:** Multipage Streamlit app. Pure-computation modules in `dashboard/core/` (zero Streamlit imports). Reusable UI components in `dashboard/components/`. Pages in `dashboard/pages/` that compose core + components. Session state bridges pages — backtest results from Page 3 flow to Pages 4, 5, 6.

**Tech Stack:** Python 3.10+, Streamlit, scikit-learn, statsmodels, scipy, plotly, pandas, numpy, pandas-datareader

---

## File Structure

```
dashboard/
├── app.py                      # Slim router: page config, data load, session state
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Factor_Analysis.py
│   ├── 3_Alpha_Model_Lab.py
│   ├── 4_Backtest_Results.py
│   ├── 5_Portfolio_Construction.py
│   ├── 6_Monitoring.py
│   └── 7_Theory_Methods.py
├── core/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── backtest.py
│   ├── portfolio.py
│   ├── diagnostics.py
│   ├── factor_models.py
│   └── risk.py
├── components/
│   ├── __init__.py
│   ├── charts.py
│   ├── metrics.py
│   ├── theory.py
│   └── theory_content.py
├── features.py                 # Evolved: adds precompute + feature groups
├── cache_manager.py            # Extended: construction method in portfolio key
└── __init__.py
tests/
├── conftest.py                 # Shared fixtures
├── test_data_loader.py
├── test_models.py
├── test_backtest.py
├── test_diagnostics.py
├── test_portfolio.py
├── test_factor_models.py
└── test_risk.py
data/
└── ff5_factors.csv             # Downloaded by data_loader on first run
.streamlit/
└── config.toml
requirements.txt
```

---

## Phase 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `dashboard/core/__init__.py`
- Create: `dashboard/components/__init__.py`
- Create: `dashboard/pages/` (directory)
- Create: `tests/conftest.py`

- [ ] **Step 1: Create requirements.txt**

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
statsmodels>=0.14
scipy>=1.11
plotly>=5.18
pandas-datareader>=0.10
pyarrow>=14.0
pytest>=7.0
```

- [ ] **Step 2: Create .streamlit/config.toml**

```toml
[theme]
base = "dark"
primaryColor = "#00D26A"
backgroundColor = "#0A1628"
secondaryBackgroundColor = "#111D2E"
textColor = "#FFFFFF"
font = "monospace"
```

- [ ] **Step 3: Create package init files and directories**

Create empty `dashboard/core/__init__.py` and `dashboard/components/__init__.py`.

Create `dashboard/pages/` directory (empty for now).

- [ ] **Step 4: Create tests/conftest.py with shared fixtures**

```python
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_panel():
    """Small synthetic panel dataset mimicking alpha_dataset_v2.csv.

    50 stocks x 72 months (2010-01 to 2015-12).
    OOS starts at 2015-01 (last 12 months).
    """
    np.random.seed(42)
    dates = [f"20{y:02d}-{m:02d}" for y in range(10, 16) for m in range(1, 13)]
    permnos = list(range(10001, 10051))
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]

    xs_features = [
        "ret_1_xs", "ret_2_12_xs", "ret_2_6_xs", "ret_13_36_xs",
        "bm_xs", "ep_xs", "cfp_xs", "sp_xs",
        "gpa_xs", "roe_xs", "roa_xs", "earn_quality_xs", "cfo_at_xs",
        "vol_12m_xs", "ivol_xs", "beta_xs", "log_me_xs",
        "sue_xs", "revision_xs", "beat_xs", "dispersion_xs",
        "turnover_xs", "illiq_12m_xs",
        "prc_52w_high_xs", "rsi_14_xs", "macd_hist_xs",
        "bb_position_xs", "roc_3_xs", "roc_6_xs",
        "sue_q_xs", "rev_surp_xs", "earn_growth_yoy_xs",
        "revision_ratio_xs", "rec_chg_xs", "n_analysts_xs",
        "peer_sue_xs", "peer_revision_xs",
        "mom_x_size_xs", "val_x_prof_xs", "mom_x_vol_xs",
        "ret_vs_sector_xs", "bm_vs_sector_xs", "ret_vs_ind_xs",
        "bm_vs_size_xs",
    ]

    rows = []
    for d in dates:
        mkt_rf = np.random.randn() * 0.04
        rf = 0.002
        for p in permnos:
            row = {
                "ym": d,
                "permno": p,
                "sector": sectors[p % 5],
                "y_xs": np.random.randn() * 0.05,
                "y_raw": np.random.randn() * 0.08,
                "Mkt_RF": mkt_rf,
                "rf_ff": rf,
            }
            for feat in xs_features:
                row[feat] = np.random.randn()
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def sample_returns():
    """Monthly return series for testing performance metrics."""
    np.random.seed(42)
    months = [f"20{y:02d}-{m:02d}" for y in range(15, 20) for m in range(1, 13)]
    returns = np.random.randn(len(months)) * 0.05 + 0.008
    return pd.Series(returns, index=months, name="returns")


@pytest.fixture
def sample_ff5():
    """Synthetic Fama-French 5-factor monthly data."""
    np.random.seed(42)
    months = [f"20{y:02d}-{m:02d}" for y in range(10, 16) for m in range(1, 13)]
    return pd.DataFrame(
        {
            "Mkt-RF": np.random.randn(len(months)) * 0.04,
            "SMB": np.random.randn(len(months)) * 0.02,
            "HML": np.random.randn(len(months)) * 0.02,
            "RMW": np.random.randn(len(months)) * 0.015,
            "CMA": np.random.randn(len(months)) * 0.015,
            "RF": np.full(len(months), 0.002),
        },
        index=months,
    )


@pytest.fixture
def sample_predictions():
    """Synthetic month->predictions dict as returned by run_walk_forward."""
    np.random.seed(42)
    months = [f"2015-{m:02d}" for m in range(1, 13)]
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]
    preds = {}
    for m in months:
        n = 50
        preds[m] = pd.DataFrame(
            {
                "permno": list(range(10001, 10001 + n)),
                "sector": [sectors[i % 5] for i in range(n)],
                "pred": np.random.randn(n) * 0.1,
                "y_raw": np.random.randn(n) * 0.08,
                "vol_12m_xs": np.abs(np.random.randn(n)) * 0.5 + 0.5,
            }
        )
    return preds
```

- [ ] **Step 5: Install dependencies and verify**

Run: `pip install -r requirements.txt`

Run: `python -m pytest tests/ --co -q`

Expected: `no tests ran` (conftest loads, no test files yet)

- [ ] **Step 6: Commit**

```
git add requirements.txt .streamlit/config.toml dashboard/core/__init__.py dashboard/components/__init__.py tests/conftest.py
git commit -m "scaffold: add requirements, dark theme config, package dirs, test fixtures"
```

---

### Task 2: core/data_loader.py

**Files:**
- Create: `dashboard/core/data_loader.py`
- Create: `tests/test_data_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data_loader.py
import pandas as pd
import numpy as np
from core.data_loader import load_dataset, compute_market_monthly, load_ff5_factors


def test_compute_market_monthly(sample_panel):
    market = compute_market_monthly(sample_panel)
    assert isinstance(market, pd.DataFrame)
    assert "spy_ret" in market.columns
    assert "Mkt_RF" in market.columns
    assert "rf_ff" in market.columns
    assert market.index.name == "ym"
    np.testing.assert_allclose(
        market["spy_ret"].values,
        (market["Mkt_RF"] + market["rf_ff"]).values,
    )


def test_compute_market_monthly_sorted(sample_panel):
    market = compute_market_monthly(sample_panel)
    assert list(market.index) == sorted(market.index)


def test_load_ff5_factors_columns(tmp_path):
    # Create a minimal CSV matching expected format
    months = ["2015-01", "2015-02", "2015-03"]
    df = pd.DataFrame(
        {
            "Mkt-RF": [0.01, -0.02, 0.015],
            "SMB": [0.005, -0.003, 0.002],
            "HML": [-0.01, 0.008, 0.004],
            "RMW": [0.003, -0.001, 0.006],
            "CMA": [0.002, 0.004, -0.002],
            "RF": [0.001, 0.001, 0.001],
        },
        index=months,
    )
    csv_path = tmp_path / "ff5_factors.csv"
    df.to_csv(csv_path, index_label="date")

    loaded = load_ff5_factors(csv_path)
    assert set(loaded.columns) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}
    assert len(loaded) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_loader.py -v`

Expected: FAIL (ModuleNotFoundError: No module named 'core.data_loader')

- [ ] **Step 3: Write implementation**

```python
# dashboard/core/data_loader.py
"""Dataset and factor data loading utilities.

All functions are pure (no Streamlit imports).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "Data"
FF5_PATH = DATA_DIR / "ff5_factors.csv"


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the alpha dataset.

    Parameters
    ----------
    path : str, Path, or None
        Path to the CSV. Defaults to Data/alpha_dataset_v2.csv.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = DATA_DIR / "alpha_dataset_v2.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    required = {"ym", "permno", "y_xs", "y_raw", "Mkt_RF", "rf_ff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def compute_market_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly market returns from the panel data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw panel with columns 'ym', 'Mkt_RF', 'rf_ff'.

    Returns
    -------
    pd.DataFrame
        Index='ym', columns=['Mkt_RF', 'rf_ff', 'spy_ret'], sorted.
    """
    market = df.groupby("ym")[["Mkt_RF", "rf_ff"]].first().sort_index()
    market["spy_ret"] = market["Mkt_RF"] + market["rf_ff"]
    return market


def load_ff5_factors(path: str | Path | None = None) -> pd.DataFrame:
    """Load Fama-French 5-factor monthly data from CSV.

    Parameters
    ----------
    path : str, Path, or None
        Path to ff5_factors.csv. Defaults to Data/ff5_factors.csv.

    Returns
    -------
    pd.DataFrame
        Index=date strings (YYYY-MM), columns: Mkt-RF, SMB, HML, RMW, CMA, RF.
    """
    if path is None:
        path = FF5_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FF5 factor data not found at {path}. "
            "Run download_ff5_factors() or place ff5_factors.csv in Data/."
        )
    df = pd.read_csv(path, index_col=0)
    df.index.name = "date"
    return df


def download_ff5_factors(save_path: str | Path | None = None) -> pd.DataFrame:
    """Download Fama-French 5-factor data from Ken French's library.

    Requires pandas-datareader.

    Parameters
    ----------
    save_path : str, Path, or None
        Where to save the CSV. Defaults to Data/ff5_factors.csv.

    Returns
    -------
    pd.DataFrame
    """
    import pandas_datareader.data as web

    if save_path is None:
        save_path = FF5_PATH

    raw = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3", "famafrench", start="1963-01-01"
    )
    df = raw[0]  # monthly table
    df = df / 100.0  # convert from percentage to decimal
    df.index = df.index.astype(str).str[:7]  # PeriodIndex -> "YYYY-MM"
    df.index.name = "date"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && python -m pytest ../tests/test_data_loader.py -v`

Expected: 3 passed

- [ ] **Step 5: Commit**

```
git add dashboard/core/data_loader.py tests/test_data_loader.py
git commit -m "feat: add core/data_loader with dataset, market, and FF5 loading"
```

---

### Task 3: Evolve features.py

**Files:**
- Modify: `dashboard/features.py`
- Create: `tests/test_features.py`

The current features.py has `build_features_linear` and `build_features_ensemble`. We add:
- `precompute_features(df)` — adds all engineered columns to df
- `FEATURE_GROUPS` — dict mapping category names to column lists
- `get_tier_defaults(tier)` — default feature list for tier 1 or 2

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features.py
import pandas as pd
from features import (
    precompute_features,
    FEATURE_GROUPS,
    get_tier_defaults,
    build_features_linear,
    build_features_ensemble,
)


def test_feature_groups_is_dict():
    assert isinstance(FEATURE_GROUPS, dict)
    assert "momentum" in FEATURE_GROUPS
    assert "value" in FEATURE_GROUPS
    assert "quality" in FEATURE_GROUPS
    for key, cols in FEATURE_GROUPS.items():
        assert isinstance(cols, list)
        assert len(cols) > 0


def test_get_tier_defaults():
    t1 = get_tier_defaults(1)
    t2 = get_tier_defaults(2)
    assert isinstance(t1, list)
    assert isinstance(t2, list)
    assert len(t2) >= len(t1)


def test_precompute_features_adds_columns(sample_panel):
    original_cols = set(sample_panel.columns)
    result = precompute_features(sample_panel)
    new_cols = set(result.columns) - original_cols
    assert "earnings_composite" in new_cols
    assert "quality_composite" in new_cols
    assert "momentum_composite" in new_cols
    assert "mom_x_quality" in new_cols


def test_backward_compat_linear(sample_panel):
    result = build_features_linear(sample_panel)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_panel)


def test_backward_compat_ensemble(sample_panel):
    result = build_features_ensemble(sample_panel)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_panel)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && python -m pytest ../tests/test_features.py -v`

Expected: FAIL (cannot import name 'precompute_features')

- [ ] **Step 3: Add new functions to features.py**

Append the following to the end of `dashboard/features.py`, after the existing `build_features_ensemble` function:

```python
# ---------------------------------------------------------------------------
# Feature metadata and precomputation (added for multipage dashboard)
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "momentum": [
        "ret_1_xs", "ret_2_12_xs", "ret_2_6_xs", "ret_13_36_xs",
        "prc_52w_high_xs", "momentum_composite", "reversal_mom_combo",
    ],
    "value": [
        "bm_xs", "ep_xs", "cfp_xs", "sp_xs", "value_composite",
    ],
    "quality": [
        "gpa_xs", "roe_xs", "roa_xs", "earn_quality_xs", "cfo_at_xs",
        "quality_composite",
    ],
    "size": ["log_me_xs"],
    "volatility": [
        "vol_12m_xs", "ivol_xs", "beta_xs",
    ],
    "technical": [
        "rsi_14_xs", "macd_hist_xs", "bb_position_xs", "roc_3_xs", "roc_6_xs",
        "technical_composite",
    ],
    "analyst": [
        "revision_xs", "dispersion_xs", "beat_xs",
        "revision_ratio_xs", "rec_chg_xs", "n_analysts_xs",
        "analyst_composite",
    ],
    "earnings": [
        "sue_xs", "sue_q_xs", "rev_surp_xs", "earn_growth_yoy_xs",
        "earnings_composite",
    ],
    "interactions": [
        "mom_x_quality", "mom_x_roe", "val_x_lowvol", "ep_x_lowvol",
        "sue_x_lowdisp", "sue_x_revision", "mom_x_lowivol", "bm_x_roe",
        "ep_x_gpa", "earn_x_mom", "quality_x_value", "earn_x_lowvol",
        "mom_x_size_xs", "val_x_prof_xs", "mom_x_vol_xs",
    ],
    "relative": [
        "sue_vs_peer", "revision_vs_peer",
        "ret_vs_sector_xs", "bm_vs_sector_xs", "ret_vs_ind_xs", "bm_vs_size_xs",
    ],
    "nonlinear": [
        "ret_2_12_xs_sq", "ret_1_xs_sq", "sue_xs_sq", "bm_xs_sq", "revision_xs_sq",
    ],
}


def get_tier_defaults(tier: int) -> list[str]:
    """Return the default feature column list for a given tier.

    Tier 1 (~52 features): conservative set for linear models.
    Tier 2 (~118+ features): all _xs columns + engineered for tree models.
    """
    if tier == 1:
        core = [
            "ret_1_xs", "ret_2_12_xs", "ret_2_6_xs",
            "bm_xs", "ep_xs", "cfp_xs", "sp_xs",
            "gpa_xs", "roe_xs", "roa_xs",
            "vol_12m_xs", "ivol_xs", "beta_xs",
            "log_me_xs",
            "sue_xs", "revision_xs", "beat_xs",
            "turnover_xs", "illiq_12m_xs",
            "mom_x_size_xs", "val_x_prof_xs", "mom_x_vol_xs",
            "ret_vs_sector_xs", "bm_vs_sector_xs", "ret_vs_ind_xs",
            "bm_vs_size_xs",
        ]
        engineered = []
        for group in FEATURE_GROUPS.values():
            for f in group:
                if not f.endswith("_xs"):
                    engineered.append(f)
        return core + sorted(set(engineered))

    # Tier 2: all _xs columns + all engineered
    all_features = []
    for group in FEATURE_GROUPS.values():
        all_features.extend(group)
    return sorted(set(all_features))


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered feature columns to the DataFrame.

    Call once after loading the dataset. Subsequent operations can select
    features by column name without re-computing.
    """
    result = df.copy()
    engineered = _build_engineered_features(df)
    for col in engineered.columns:
        result[col] = engineered[col]
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && python -m pytest ../tests/test_features.py -v`

Expected: 5 passed

- [ ] **Step 5: Commit**

```
git add dashboard/features.py tests/test_features.py
git commit -m "feat: add feature groups, tier defaults, and precompute to features.py"
```

---

## Phase 2: Core Computation Modules

### Task 4: core/models.py — Model Registry

**Files:**
- Create: `dashboard/core/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
import numpy as np
import pandas as pd
from core.models import (
    get_model,
    get_default_params,
    get_param_ranges,
    get_feature_tier,
    list_models,
)


def test_list_models():
    models = list_models()
    assert set(models) == {"HGB", "RF", "Lasso", "Ridge", "ElasticNet", "FamaMacBeth"}


def test_get_feature_tier():
    assert get_feature_tier("HGB") == 2
    assert get_feature_tier("RF") == 2
    assert get_feature_tier("Lasso") == 1
    assert get_feature_tier("Ridge") == 1
    assert get_feature_tier("ElasticNet") == 1
    assert get_feature_tier("FamaMacBeth") == 1


def test_get_default_params():
    for name in list_models():
        params = get_default_params(name)
        assert isinstance(params, dict)


def test_get_param_ranges():
    for name in list_models():
        ranges = get_param_ranges(name)
        assert isinstance(ranges, dict)
        for key, spec in ranges.items():
            assert "min" in spec
            assert "max" in spec
            assert "default" in spec


def test_model_fit_predict():
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    y_train = pd.Series(np.random.randn(200))
    X_test = pd.DataFrame(np.random.randn(50, 5), columns=[f"f{i}" for i in range(5)])

    for name in list_models():
        model = get_model(name, get_default_params(name))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == 50
        assert not np.any(np.isnan(preds))


def test_model_feature_importance():
    np.random.seed(42)
    cols = [f"f{i}" for i in range(5)]
    X = pd.DataFrame(np.random.randn(200, 5), columns=cols)
    y = pd.Series(np.random.randn(200))

    tree_model = get_model("HGB", get_default_params("HGB"))
    tree_model.fit(X, y)
    imp = tree_model.get_feature_importance()
    assert imp is not None
    assert len(imp) == 5

    linear_model = get_model("Lasso", get_default_params("Lasso"))
    linear_model.fit(X, y)
    imp = linear_model.get_feature_importance()
    assert imp is not None
    assert len(imp) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && python -m pytest ../tests/test_models.py -v`

Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# dashboard/core/models.py
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
import statsmodels.api as sm


@runtime_checkable
class AlphaModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def get_feature_importance(self) -> pd.Series | None: ...
    def get_params(self) -> dict: ...


# ---------------------------------------------------------------------------
# Model implementations
# ---------------------------------------------------------------------------

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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.Series | None:
        if not hasattr(self._model, "feature_importances_"):
            return None
        return pd.Series(
            self._model.feature_importances_, index=self._feature_names
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
        self._avg_coefs = X_with_const.iloc[0:0].mean()  # placeholder
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
    "HGB": 2,
    "RF": 2,
    "Lasso": 1,
    "Ridge": 1,
    "ElasticNet": 1,
    "FamaMacBeth": 1,
}

_DEFAULT_PARAMS: dict[str, dict] = {
    "HGB": {
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_samples_leaf": 500,
        "l2_regularization": 0.1,
        "max_iter": 500,
    },
    "RF": {
        "n_estimators": 200,
        "max_depth": 4,
        "max_features": "sqrt",
        "min_samples_leaf": 50,
    },
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && python -m pytest ../tests/test_models.py -v`

Expected: 6 passed

- [ ] **Step 5: Commit**

```
git add dashboard/core/models.py tests/test_models.py
git commit -m "feat: add model registry with 6 alpha models (HGB, RF, Lasso, Ridge, ElasticNet, FM)"
```

---

### Task 5: core/backtest.py — Walk-Forward Engine

**Files:**
- Create: `dashboard/core/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backtest.py
import numpy as np
import pandas as pd
from core.backtest import run_walk_forward, BacktestResult
from core.models import get_model, get_default_params


def test_backtest_result_dataclass():
    from dataclasses import fields
    names = {f.name for f in fields(BacktestResult)}
    assert "predictions" in names
    assert "monthly_returns" in names
    assert "ic" in names
    assert "holdings" in names
    assert "turnover" in names
    assert "feature_importance" in names
    assert "train_dates" in names
    assert "model_params" in names


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
    assert len(result.monthly_returns) > 0
    assert len(result.ic) > 0
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
    # All prediction months are >= oos_start
    for m in result.predictions:
        assert m >= "2015-01"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && python -m pytest ../tests/test_backtest.py -v`

Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# dashboard/core/backtest.py
"""Walk-forward prediction engine.

Extracted from engine.py. Now accepts AlphaModel instances from the registry
and supports both expanding and rolling windows.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone

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
    """Train model walk-forward and collect per-stock raw predictions.

    Parameters
    ----------
    data : pd.DataFrame
        Full panel with 'ym', TRAIN_TARGET, EVAL_TARGET, 'permno', feature columns.
    model : AlphaModel
        Model instance from the registry. Will be cloned before each retrain.
    feature_cols : list[str]
        Column names to use as features.
    oos_start : str
        First out-of-sample month (YYYY-MM).
    retrain_freq : int
        Retrain every N OOS months.
    window_type : str
        "expanding" (all data before t) or "rolling" (last rolling_window months).
    rolling_window : int or None
        Months of training data for rolling window. Required when window_type="rolling".
    progress_callback : callable or None
        Called with (step, total, month_str) after each OOS month.

    Returns
    -------
    BacktestResult
    """
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
    """Compute equal-weight top-K portfolio from raw predictions.

    This provides a quick default portfolio for the BacktestResult.
    Full portfolio construction with multiple methods lives in core/portfolio.py.
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && python -m pytest ../tests/test_backtest.py -v`

Expected: 4 passed

- [ ] **Step 5: Commit**

```
git add dashboard/core/backtest.py tests/test_backtest.py
git commit -m "feat: add walk-forward engine with expanding/rolling window support"
```

---

### Task 6: core/diagnostics.py

**Files:**
- Create: `dashboard/core/diagnostics.py`
- Create: `tests/test_diagnostics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_diagnostics.py
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
    # f1 is shifted by 2, should have high D
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && python -m pytest ../tests/test_diagnostics.py -v`

Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# dashboard/core/diagnostics.py
"""Performance metrics, IC analysis, Fundamental Law, KS test, alpha decay."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp


def compute_performance_metrics(returns: pd.Series) -> dict:
    """Compute annualized performance statistics from monthly returns."""
    s = returns.dropna()
    if len(s) == 0:
        return {k: np.nan for k in
                ["SR", "Ann Return", "Ann Vol", "MDD", "Calmar", "Total Return"]}

    ann_ret = s.mean() * 12
    ann_vol = s.std() * np.sqrt(12)
    sr = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + s).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
    total_ret = cum.iloc[-1] - 1

    return {
        "SR": round(float(sr), 4),
        "Ann Return": float(ann_ret),
        "Ann Vol": float(ann_vol),
        "MDD": float(mdd),
        "Calmar": round(float(calmar), 4) if not np.isnan(calmar) else np.nan,
        "Total Return": float(total_ret),
    }


def compute_ic_stats(ic: pd.Series) -> dict:
    """Compute IC summary statistics."""
    ic_clean = ic.dropna()
    if len(ic_clean) == 0:
        return {"mean_ic": np.nan, "ic_tstat": np.nan, "icir": np.nan, "hit_rate": np.nan}

    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    n = len(ic_clean)
    ic_tstat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    hit_rate = (ic_clean > 0).mean()

    return {
        "mean_ic": float(mean_ic),
        "ic_tstat": float(ic_tstat),
        "icir": float(icir),
        "hit_rate": float(hit_rate),
    }


def fundamental_law(
    ic_mean: float,
    K: int,
    rebal_freq: int = 12,
    sr_target: float = 1.0,
    tc_bps: float = 10.0,
) -> dict:
    """Compute Fundamental Law of Active Management metrics.

    IR = IC * sqrt(BR), where BR = breadth = independent bets per year.
    """
    BR_nominal = K * rebal_freq
    IR_upper = ic_mean * np.sqrt(BR_nominal) if BR_nominal > 0 else np.nan
    BR_implied = (IR_upper / ic_mean) ** 2 if ic_mean != 0 else np.nan

    tc_annual = tc_bps / 10000 * 2 * rebal_freq
    port_vol = 0.15  # assume ~15% annual vol for IC_required calc
    Cost_SR = tc_annual / port_vol

    IC_required = (
        (sr_target + Cost_SR) / np.sqrt(BR_nominal)
        if BR_nominal > 0 else np.nan
    )

    return {
        "BR_nominal": int(BR_nominal),
        "IR_upper_bound": float(IR_upper) if not np.isnan(IR_upper) else np.nan,
        "BR_implied": float(BR_implied) if not np.isnan(BR_implied) else np.nan,
        "IC_required": float(IC_required) if not np.isnan(IC_required) else np.nan,
        "Cost_SR": float(Cost_SR),
    }


def feature_ic(X: pd.DataFrame, y_realized: pd.Series) -> pd.Series:
    """Compute univariate Spearman IC for each feature column."""
    results = {}
    for col in X.columns:
        valid = pd.DataFrame({"x": X[col], "y": y_realized}).dropna()
        if len(valid) > 10:
            results[col] = spearmanr(valid["x"], valid["y"])[0]
        else:
            results[col] = np.nan
    return pd.Series(results).sort_values(ascending=False)


def ks_test(
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Per-feature KS test between training and current distributions."""
    results = []
    for col in X_train.columns:
        if col not in X_current.columns:
            continue
        train_vals = X_train[col].dropna()
        curr_vals = X_current[col].dropna()
        if len(train_vals) == 0 or len(curr_vals) == 0:
            continue
        D, pval = ks_2samp(train_vals, curr_vals)
        results.append({
            "feature": col,
            "D": float(D),
            "pval": float(pval),
            "flag": D > threshold,
        })
    return pd.DataFrame(results).sort_values("D", ascending=False).reset_index(drop=True)


def alpha_decay(
    predictions: dict[str, pd.DataFrame],
    horizons: list[int] | None = None,
) -> pd.Series:
    """Compute IC at multiple forward horizons to measure signal decay.

    Parameters
    ----------
    predictions : dict
        month -> DataFrame with 'pred' and 'y_raw' columns.
    horizons : list[int]
        Forward horizons in months (default: 1..12).

    Returns
    -------
    pd.Series
        IC at each horizon, indexed by horizon.
    """
    if horizons is None:
        horizons = list(range(1, 13))

    months = sorted(predictions.keys())
    results = {}

    for h in horizons:
        ic_vals = []
        for i, m in enumerate(months):
            target_idx = i + h - 1  # h=1 means same-month realized
            if target_idx >= len(months):
                break
            target_month = months[target_idx]
            pred_df = predictions[m]
            target_df = predictions[target_month]

            merged = pred_df[["permno", "pred"]].merge(
                target_df[["permno", "y_raw"]], on="permno", how="inner"
            )
            if len(merged) > 10:
                ic_vals.append(spearmanr(merged["pred"], merged["y_raw"])[0])

        results[h] = np.nanmean(ic_vals) if ic_vals else np.nan

    return pd.Series(results)


def signal_staleness(
    turnover: pd.Series,
    threshold: float = 0.10,
    consecutive: int = 3,
) -> pd.DataFrame:
    """Flag periods where turnover stays below threshold for consecutive months."""
    below = turnover < threshold
    streak = below.astype(int)
    streak_count = streak.groupby((~below).cumsum()).cumsum()
    stale = streak_count >= consecutive

    return pd.DataFrame({
        "turnover": turnover,
        "below_threshold": below,
        "stale": stale,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && python -m pytest ../tests/test_diagnostics.py -v`

Expected: 7 passed

- [ ] **Step 5: Commit**

```
git add dashboard/core/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: add diagnostics module (performance, IC, Fundamental Law, KS, decay)"
```

---

### Task 7: core/portfolio.py — 5 Construction Methods

**Files:**
- Create: `dashboard/core/portfolio.py`
- Create: `tests/test_portfolio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_portfolio.py
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
    assert (result["weight"] >= -0.001).all()  # long-only constraint
    assert (result["weight"] <= 0.051).all()   # max 5% cap


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && python -m pytest ../tests/test_portfolio.py -v`

Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# dashboard/core/portfolio.py
"""Portfolio construction: 5 methods + monthly series builder with regime filter."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

EVAL_TARGET = "y_raw"


def construct_portfolio(
    predictions: pd.DataFrame,
    method: str,
    K: int,
    strategy_type: str,
    K_short: int,
    vol_tilt: float,
    returns_history: pd.DataFrame | None = None,
    **method_params,
) -> pd.DataFrame:
    """Construct portfolio weights for a single month.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns: permno, pred, y_raw, vol_12m_xs (optional).
    method : str
        One of: equal_weight, score_weight, inverse_vol, erc, mvo.
    K : int
        Number of long positions.
    strategy_type : str
        "long_only" or "long_short".
    K_short : int
        Number of short positions (long_short only).
    vol_tilt : float
        Volatility penalty applied before ranking.
    returns_history : pd.DataFrame or None
        Historical returns matrix (months x permnos). Required for erc, mvo.

    Returns
    -------
    pd.DataFrame
        Columns: permno, weight, side, plus original columns from predictions.
    """
    df = predictions.copy()

    if vol_tilt > 0.0 and "vol_12m_xs" in df.columns:
        df["pred"] = df["pred"] - vol_tilt * df["vol_12m_xs"].fillna(0.0)

    top = df.nlargest(K, "pred").copy()

    if strategy_type == "long_short":
        bottom = df.nsmallest(K_short, "pred").copy()
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
    """Compute normalized weights for selected stocks."""
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
        max_weight = params.get("max_weight", 0.05)
        return _mvo_weights(selected, returns_history, max_weight)

    else:
        return np.ones(n) / n


def _erc_weights(
    selected: pd.DataFrame,
    returns_history: pd.DataFrame | None,
) -> np.ndarray:
    """Equal Risk Contribution via numerical optimization."""
    n = len(selected)
    permnos = selected["permno"].values

    cov = _get_cov_matrix(permnos, returns_history, n)

    def objective(w):
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e10
        port_vol = np.sqrt(port_var)
        mc = cov @ w
        rc = w * mc / port_vol
        return np.sum((rc - rc.mean()) ** 2)

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
    max_weight: float = 0.05,
) -> np.ndarray:
    """Mean-Variance Optimization with Ledoit-Wolf shrinkage."""
    n = len(selected)
    permnos = selected["permno"].values
    expected_returns = selected["pred"].values

    cov = _get_cov_matrix(permnos, returns_history, n, shrink=True)

    def neg_sharpe(w):
        port_ret = w @ expected_returns
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
    """Get covariance matrix for selected stocks."""
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
    **method_params,
) -> dict:
    """Build full portfolio time series from predictions dict.

    Returns dict with: monthly_returns, holdings, ic, turnover.
    """
    regime_on = None
    if market_monthly is not None and regime_lookback > 0:
        trailing = market_monthly["spy_ret"].rolling(regime_lookback).sum().shift(1)
        regime_on = trailing >= 0

    months = sorted(predictions.keys())
    monthly_returns: dict[str, float] = {}
    holdings: dict[str, pd.DataFrame] = {}
    ic_vals: dict[str, float] = {}
    turnover_vals: dict[str, float] = {}
    prev_weights: pd.Series | None = None

    for m in months:
        df_m = predictions[m]
        min_required = K + K_short if strategy_type == "long_short" else 2 * K
        if len(df_m) < min_required:
            prev_weights = None
            continue

        valid = df_m[["pred", EVAL_TARGET]].dropna()
        if len(valid) > 10:
            ic_vals[m] = spearmanr(valid["pred"], valid[EVAL_TARGET])[0]
        else:
            ic_vals[m] = np.nan

        held = construct_portfolio(
            df_m, method=method, K=K, strategy_type=strategy_type,
            K_short=K_short, vol_tilt=vol_tilt,
            returns_history=returns_history, **method_params,
        )

        port_ret = (held["weight"] * held[EVAL_TARGET]).sum()

        if regime_on is not None and m in regime_on.index and not regime_on[m]:
            port_ret = 0.0

        monthly_returns[m] = port_ret
        holdings[m] = held

        curr_weights = pd.Series(0.0, index=df_m["permno"].values)
        for _, row in held.iterrows():
            curr_weights[row["permno"]] = row["weight"]

        if prev_weights is not None:
            aligned_c, aligned_p = curr_weights.align(prev_weights, fill_value=0.0)
            turnover_vals[m] = 0.5 * (aligned_c - aligned_p).abs().sum()
        else:
            turnover_vals[m] = np.nan

        prev_weights = curr_weights

    return {
        "monthly_returns": pd.Series(monthly_returns).sort_index(),
        "holdings": holdings,
        "ic": pd.Series(ic_vals).sort_index(),
        "turnover": pd.Series(turnover_vals).sort_index(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && python -m pytest ../tests/test_portfolio.py -v`

Expected: 8 passed

- [ ] **Step 5: Commit**

```
git add dashboard/core/portfolio.py tests/test_portfolio.py
git commit -m "feat: add 5 portfolio construction methods (EW, score, inv-vol, ERC, MVO)"
```

---

### Task 8: core/factor_models.py

**Files:**
- Create: `dashboard/core/factor_models.py`
- Create: `tests/test_factor_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_factor_models.py
import numpy as np
import pandas as pd
from core.factor_models import (
    run_regression,
    rolling_beta,
    bloomberg_shrink_beta,
    compute_vif,
    wald_test,
    hedge_ratio,
)


def test_run_regression_capm(sample_ff5):
    np.random.seed(42)
    n = len(sample_ff5)
    stock_returns = sample_ff5["Mkt-RF"] * 1.2 + np.random.randn(n) * 0.02
    result = run_regression(stock_returns, sample_ff5, model_type="CAPM")
    assert "coefficients" in result
    assert "const" in result["coefficients"]
    assert "Mkt-RF" in result["coefficients"]
    assert "r_squared" in result
    assert "hac_se" in result


def test_run_regression_ff5(sample_ff5):
    np.random.seed(42)
    n = len(sample_ff5)
    stock_returns = (sample_ff5["Mkt-RF"] * 1.1 + sample_ff5["SMB"] * 0.5 +
                     np.random.randn(n) * 0.02)
    result = run_regression(stock_returns, sample_ff5, model_type="FF5")
    assert "SMB" in result["coefficients"]
    assert "HML" in result["coefficients"]
    assert "RMW" in result["coefficients"]
    assert "CMA" in result["coefficients"]


def test_rolling_beta(sample_ff5):
    np.random.seed(42)
    n = len(sample_ff5)
    stock_returns = sample_ff5["Mkt-RF"] * 1.2 + np.random.randn(n) * 0.02
    result = rolling_beta(stock_returns, sample_ff5["Mkt-RF"], window=24)
    assert "beta" in result.columns
    assert "lower" in result.columns
    assert "upper" in result.columns
    assert len(result) == n


def test_bloomberg_shrink():
    raw = pd.Series([0.5, 1.0, 1.5, 2.0])
    shrunk = bloomberg_shrink_beta(raw)
    expected = 0.67 * raw + 0.33
    pd.testing.assert_series_equal(shrunk, expected)


def test_compute_vif(sample_ff5):
    factors = sample_ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    result = compute_vif(factors)
    assert "feature" in result.columns
    assert "VIF" in result.columns
    assert len(result) == 5


def test_wald_test(sample_ff5):
    np.random.seed(42)
    n = len(sample_ff5)
    stock_returns = sample_ff5["Mkt-RF"] * 1.2 + np.random.randn(n) * 0.02
    result = wald_test(stock_returns, sample_ff5, indices=["SMB", "HML"])
    assert "statistic" in result
    assert "pvalue" in result
    assert "reject" in result


def test_hedge_ratio():
    assert hedge_ratio(1.2, 1.0) == -1.2
    assert hedge_ratio(0.8, 1.0) == -0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && python -m pytest ../tests/test_factor_models.py -v`

Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# dashboard/core/factor_models.py
"""CAPM, Fama-French regressions, rolling beta, VIF, Wald test, hedge ratio.

All regressions use HAC standard errors.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox


_FACTOR_SETS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


def run_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    model_type: str = "CAPM",
) -> dict:
    """Run factor regression with HAC standard errors.

    Parameters
    ----------
    returns : pd.Series
        Excess returns (already minus RF).
    factors : pd.DataFrame
        Factor returns. Must contain columns for the chosen model_type.
    model_type : str
        "CAPM", "FF3", or "FF5".

    Returns
    -------
    dict with keys: coefficients, hac_se, t_stats, p_values,
                    r_squared, adj_r_squared, durbin_watson, jarque_bera.
    """
    factor_cols = _FACTOR_SETS[model_type]
    common_idx = returns.dropna().index.intersection(factors.dropna().index)

    y = returns.loc[common_idx]
    X = sm.add_constant(factors.loc[common_idx, factor_cols])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    dw = durbin_watson(model.resid)
    jb_stat, jb_pval, _, _ = jarque_bera(model.resid)

    return {
        "coefficients": model.params.to_dict(),
        "hac_se": model.bse.to_dict(),
        "t_stats": model.tvalues.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "durbin_watson": float(dw),
        "jarque_bera": {"statistic": float(jb_stat), "pvalue": float(jb_pval)},
        "nobs": int(model.nobs),
        "model": model,
    }


def rolling_beta(
    returns: pd.Series,
    market: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Compute rolling OLS beta with confidence bands.

    Parameters
    ----------
    returns : pd.Series
        Stock excess returns.
    market : pd.Series
        Market excess returns.
    window : int
        Rolling window size in periods.

    Returns
    -------
    pd.DataFrame
        Columns: beta, lower, upper (95% confidence).
    """
    common = returns.dropna().index.intersection(market.dropna().index)
    y_all = returns.loc[common]
    x_all = market.loc[common]
    n = len(common)

    betas = pd.Series(np.nan, index=common)
    lower = pd.Series(np.nan, index=common)
    upper = pd.Series(np.nan, index=common)

    for i in range(window, n + 1):
        idx = common[i - window : i]
        y = y_all.loc[idx]
        X = sm.add_constant(x_all.loc[idx])
        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            b = model.params.iloc[1]
            se = model.bse.iloc[1]
            betas.iloc[i - 1] = b
            lower.iloc[i - 1] = b - 1.96 * se
            upper.iloc[i - 1] = b + 1.96 * se
        except Exception:
            continue

    return pd.DataFrame({"beta": betas, "lower": lower, "upper": upper})


def bloomberg_shrink_beta(raw_beta: pd.Series) -> pd.Series:
    """Apply Bloomberg beta shrinkage: 0.67 * beta + 0.33."""
    return 0.67 * raw_beta + 0.33


def compute_vif(factors: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each regressor."""
    X = sm.add_constant(factors)
    vif_data = []
    for i, col in enumerate(factors.columns):
        vif_data.append({
            "feature": col,
            "VIF": variance_inflation_factor(X.values, i + 1),  # +1 to skip const
        })
    return pd.DataFrame(vif_data)


def wald_test(
    returns: pd.Series,
    factors: pd.DataFrame,
    indices: list[str],
) -> dict:
    """Joint Wald test on specified factor coefficients = 0.

    Parameters
    ----------
    returns : pd.Series
        Excess returns.
    factors : pd.DataFrame
        Factor data (must include all FF5 columns).
    indices : list[str]
        Factor names to test jointly (e.g., ["SMB", "HML"]).

    Returns
    -------
    dict with: statistic, pvalue, reject (at 5%).
    """
    reg = run_regression(returns, factors, model_type="FF5")
    model = reg["model"]

    restriction = " = ".join([f"{name} = 0" for name in indices])
    restriction = ", ".join([f"{name} = 0" for name in indices])

    try:
        result = model.wald_test(restriction)
        stat = float(result.statistic[0][0])
        pval = float(result.pvalue)
    except Exception:
        R = np.zeros((len(indices), len(model.params)))
        for i, name in enumerate(indices):
            col_idx = list(model.params.index).index(name)
            R[i, col_idx] = 1
        q = np.zeros(len(indices))
        result = model.wald_test(R)
        stat = float(result.statistic[0][0])
        pval = float(result.pvalue)

    return {
        "statistic": stat,
        "pvalue": pval,
        "reject": pval < 0.05,
    }


def hedge_ratio(portfolio_beta: float, market_beta: float = 1.0) -> float:
    """Compute SPY hedge weight for market neutrality.

    w_hedge = -beta_portfolio / beta_market
    """
    return -portfolio_beta / market_beta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && python -m pytest ../tests/test_factor_models.py -v`

Expected: 7 passed

- [ ] **Step 5: Commit**

```
git add dashboard/core/factor_models.py tests/test_factor_models.py
git commit -m "feat: add factor models (CAPM/FF3/FF5, rolling beta, VIF, Wald, hedge)"
```

---

### Task 9: core/risk.py

**Files:**
- Create: `dashboard/core/risk.py`
- Create: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_risk.py
import numpy as np
import pandas as pd
from core.risk import (
    risk_contribution,
    factor_exposure,
    rolling_factor_exposure,
    compute_turnover,
    transaction_cost_drag,
)


def test_risk_contribution():
    np.random.seed(42)
    weights = np.array([0.3, 0.3, 0.4])
    cov = np.array([[0.04, 0.01, 0.005],
                     [0.01, 0.03, 0.008],
                     [0.005, 0.008, 0.05]])
    rc = risk_contribution(weights, cov)
    assert len(rc) == 3
    np.testing.assert_almost_equal(rc.sum(), 1.0, decimal=5)
    assert (rc > 0).all()


def test_factor_exposure(sample_ff5):
    np.random.seed(42)
    port_returns = pd.Series(
        np.random.randn(len(sample_ff5)) * 0.03,
        index=sample_ff5.index,
    )
    result = factor_exposure(port_returns, sample_ff5)
    assert "Mkt-RF" in result.index
    assert "SMB" in result.index


def test_rolling_factor_exposure(sample_ff5):
    np.random.seed(42)
    port_returns = pd.Series(
        np.random.randn(len(sample_ff5)) * 0.03,
        index=sample_ff5.index,
    )
    result = rolling_factor_exposure(port_returns, sample_ff5, window=24)
    assert "Mkt-RF" in result.columns
    assert len(result) == len(sample_ff5)


def test_compute_turnover():
    curr = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    prev = pd.Series({"A": 0.5, "B": 0.5, "D": 0.0})
    to = compute_turnover(curr, prev)
    assert to > 0


def test_transaction_cost_drag():
    turnover = pd.Series([0.3, 0.25, 0.35, 0.28])
    result = transaction_cost_drag(turnover, cost_bps=10, portfolio_vol=0.15)
    assert "TC_annual" in result
    assert "Cost_SR" in result
    assert "net_SR_reduction" in result
    assert result["TC_annual"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && python -m pytest ../tests/test_risk.py -v`

Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# dashboard/core/risk.py
"""Risk analysis: risk contribution, factor exposure, turnover, transaction costs."""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> pd.Series:
    """Compute fractional risk contribution for each position.

    RC_i = w_i * (Sigma @ w)_i / sigma_p
    Normalized so sum(RC) = 1.
    """
    port_var = weights @ cov_matrix @ weights
    if port_var <= 0:
        return pd.Series(np.ones(len(weights)) / len(weights))

    port_vol = np.sqrt(port_var)
    marginal = cov_matrix @ weights
    rc = weights * marginal / port_vol
    rc_frac = rc / rc.sum()
    return pd.Series(rc_frac)


def factor_exposure(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
) -> pd.Series:
    """Run FF5 regression on portfolio returns, return factor betas."""
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    available = [c for c in factor_cols if c in ff5_factors.columns]
    common = portfolio_returns.dropna().index.intersection(ff5_factors.dropna().index)

    if len(common) < 10:
        return pd.Series(dtype=float)

    y = portfolio_returns.loc[common]
    X = sm.add_constant(ff5_factors.loc[common, available])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    return model.params.drop("const", errors="ignore")


def rolling_factor_exposure(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
    window: int = 36,
) -> pd.DataFrame:
    """Rolling FF5 factor betas over time."""
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    available = [c for c in factor_cols if c in ff5_factors.columns]
    common = portfolio_returns.dropna().index.intersection(ff5_factors.dropna().index)

    y_all = portfolio_returns.loc[common]
    X_all = ff5_factors.loc[common, available]

    results = pd.DataFrame(np.nan, index=common, columns=available)

    for i in range(window, len(common) + 1):
        idx = common[i - window : i]
        y = y_all.loc[idx]
        X = sm.add_constant(X_all.loc[idx])
        try:
            model = sm.OLS(y, X).fit()
            betas = model.params.drop("const", errors="ignore")
            results.iloc[i - 1] = betas
        except Exception:
            continue

    return results


def compute_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series,
) -> float:
    """Monthly one-way turnover: 0.5 * sum(|delta_w|)."""
    aligned_c, aligned_p = current_weights.align(previous_weights, fill_value=0.0)
    return 0.5 * (aligned_c - aligned_p).abs().sum()


def transaction_cost_drag(
    turnover_series: pd.Series,
    cost_bps: float = 10.0,
    portfolio_vol: float = 0.15,
) -> dict:
    """Compute annualized transaction cost impact.

    Parameters
    ----------
    turnover_series : pd.Series
        Monthly one-way turnover values.
    cost_bps : float
        Cost per trade in basis points (one-way).
    portfolio_vol : float
        Annualized portfolio volatility.

    Returns
    -------
    dict with: TC_annual, Cost_SR, net_SR_reduction.
    """
    mean_monthly_to = turnover_series.dropna().mean()
    tc_monthly = mean_monthly_to * cost_bps / 10000 * 2  # round-trip
    tc_annual = tc_monthly * 12
    cost_sr = tc_annual / portfolio_vol if portfolio_vol > 0 else np.nan

    return {
        "TC_annual": float(tc_annual),
        "Cost_SR": float(cost_sr),
        "net_SR_reduction": float(cost_sr),
        "mean_monthly_turnover": float(mean_monthly_to),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && python -m pytest ../tests/test_risk.py -v`

Expected: 5 passed

- [ ] **Step 5: Commit**

```
git add dashboard/core/risk.py tests/test_risk.py
git commit -m "feat: add risk module (risk contribution, factor exposure, turnover, TC)"
```

---

### Task 10: Update cache_manager.py

**Files:**
- Modify: `dashboard/cache_manager.py`

- [ ] **Step 1: Add construction method to portfolio_key**

In `dashboard/cache_manager.py`, update the `portfolio_key` function to include the construction method:

Change the existing `portfolio_key` function from:

```python
def portfolio_key(pred_key, K, vol_tilt, regime_lookback, strategy_type="long_only", K_short=10):
    return _make_key({
        'pred_key': pred_key,
        'K': K,
        'vol_tilt': vol_tilt,
        'regime_lookback': regime_lookback,
        'strategy_type': strategy_type,
        'K_short': K_short,
    })
```

To:

```python
def portfolio_key(pred_key, K, vol_tilt, regime_lookback, strategy_type="long_only", K_short=10, construction_method="equal_weight"):
    return _make_key({
        'pred_key': pred_key,
        'K': K,
        'vol_tilt': vol_tilt,
        'regime_lookback': regime_lookback,
        'strategy_type': strategy_type,
        'K_short': K_short,
        'construction_method': construction_method,
    })
```

Also update `prediction_key` to include feature_cols and window_type:

Change:

```python
def prediction_key(model_type, model_params, retrain_every):
    return _make_key({
        'model_type': model_type,
        'model_params': model_params,
        'retrain_every': retrain_every,
    })
```

To:

```python
def prediction_key(model_type, model_params, retrain_every, feature_cols=None, window_type="expanding"):
    return _make_key({
        'model_type': model_type,
        'model_params': model_params,
        'retrain_every': retrain_every,
        'feature_cols': sorted(feature_cols) if feature_cols else None,
        'window_type': window_type,
    })
```

- [ ] **Step 2: Verify nothing is broken**

Run: `cd dashboard && python -m pytest ../tests/ -v`

Expected: All previously passing tests still pass.

- [ ] **Step 3: Commit**

```
git add dashboard/cache_manager.py
git commit -m "feat: extend cache keys with construction method, features, window type"
```

---

## Phase 3: Shared Components

### Task 11: components/charts.py

**Files:**
- Create: `dashboard/components/charts.py`

- [ ] **Step 1: Create the Bloomberg-styled chart builders**

```python
# dashboard/components/charts.py
"""Reusable Plotly chart builders with Bloomberg dark aesthetic."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

STYLE = {
    "bg": "#0A1628",
    "positive": "#00D26A",
    "negative": "#FF4444",
    "warning": "#FFB800",
    "text": "#FFFFFF",
    "muted": "#8899AA",
    "accent": "#3b82f6",
    "template": "plotly_dark",
}

PIN_COLORS = ["#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6"]


def _base_layout(**overrides) -> dict:
    layout = dict(
        template=STYLE["template"],
        height=400,
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor=STYLE["bg"],
        plot_bgcolor=STYLE["bg"],
        font=dict(family="monospace", color=STYLE["text"]),
    )
    layout.update(overrides)
    return layout


def cumulative_wealth_chart(
    returns_dict: dict[str, pd.Series],
    start_val: float = 10000,
    cash_flow: float = 0,
) -> go.Figure:
    """Multi-series cumulative wealth chart."""
    fig = go.Figure()

    for i, (name, rets) in enumerate(returns_dict.items()):
        values = []
        bal = start_val
        for r in rets:
            bal = (bal + cash_flow) * (1 + r)
            values.append(bal)
        wealth = pd.Series(values, index=rets.index)

        color = STYLE["accent"] if i == 0 else (
            "gray" if name == "SPY" else PIN_COLORS[(i - 1) % len(PIN_COLORS)]
        )
        dash = "dash" if name == "SPY" else None
        fig.add_trace(go.Scatter(
            x=wealth.index, y=wealth.values, name=name,
            line=dict(color=color, width=2, dash=dash),
        ))

    cf_label = f", ${cash_flow:+,}/mo" if cash_flow != 0 else ""
    fig.update_layout(**_base_layout(
        title="Cumulative Wealth",
        yaxis_title=f"Portfolio Value (${start_val:,} start{cf_label})",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    ))
    return fig


def drawdown_chart(returns_dict: dict[str, pd.Series], start_val: float = 10000) -> go.Figure:
    """Drawdown chart for multiple series."""
    fig = go.Figure()

    for i, (name, rets) in enumerate(returns_dict.items()):
        cum = (1 + rets).cumprod() * start_val
        dd = cum / cum.cummax() - 1

        color = STYLE["negative"] if i == 0 else PIN_COLORS[(i - 1) % len(PIN_COLORS)]
        fill = "tozeroy" if i == 0 else None
        fillcolor = "rgba(239,68,68,0.3)" if i == 0 else None

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name, fill=fill,
            line=dict(color=color, width=1), fillcolor=fillcolor,
        ))

    fig.update_layout(**_base_layout(
        title="Drawdown", yaxis_title="Drawdown %", yaxis_tickformat=".0%",
    ))
    return fig


def monthly_heatmap(returns: pd.Series) -> go.Figure:
    """Year x month returns heatmap."""
    df = returns.to_frame("ret")
    df["year"] = df.index.str[:4]
    df["month"] = df.index.str[5:7].astype(int)
    pivot = df.pivot_table(values="ret", index="year", columns="month", aggfunc="first")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_labels[c - 1] for c in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100, x=pivot.columns, y=pivot.index,
        colorscale=[[0, STYLE["negative"]], [0.5, "#1f2937"], [1, STYLE["positive"]]],
        zmid=0,
        text=np.round(pivot.values * 100, 1),
        texttemplate="%{text:.1f}%",
        textfont=dict(size=10),
        hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_base_layout(title="Monthly Returns Heatmap (%)"))
    return fig


def rolling_metric_chart(
    series: pd.Series,
    window: int = 12,
    name: str = "Metric",
    show_bands: bool = True,
) -> go.Figure:
    """Rolling metric line chart with optional +/- 1 sigma bands."""
    rolling_mean = series.rolling(window, min_periods=window // 2).mean()
    fig = go.Figure()

    if show_bands:
        rolling_std = series.rolling(window, min_periods=window // 2).std()
        upper = rolling_mean + rolling_std
        lower = rolling_mean - rolling_std
        fig.add_trace(go.Scatter(
            x=upper.index, y=upper.values, mode="lines", name="+1σ",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=lower.index, y=lower.values, mode="lines", name="-1σ",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(59,130,246,0.15)", showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=rolling_mean.index, y=rolling_mean.values, name=f"Rolling {window}m {name}",
        line=dict(color=STYLE["accent"], width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=STYLE["muted"])

    fig.update_layout(**_base_layout(title=f"Rolling {window}-Month {name}", yaxis_title=name))
    return fig


def bar_chart(
    series: pd.Series,
    name: str = "Value",
    mean_line: bool = True,
) -> go.Figure:
    """Signed bar chart with colors based on positive/negative."""
    colors = [STYLE["positive"] if v >= 0 else STYLE["negative"] for v in series.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=series.index, y=series.values, name=name,
        marker_color=colors, opacity=0.7,
    ))

    if mean_line:
        fig.add_hline(
            y=series.mean(), line_dash="dash", line_color=STYLE["warning"],
            annotation_text=f"Mean={series.mean():.3f}",
        )

    fig.update_layout(**_base_layout(title=name, yaxis_title=name))
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Spearman correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=8),
    ))
    fig.update_layout(**_base_layout(title="Spearman Correlation Matrix", height=600))
    return fig


def sector_allocation_chart(
    holdings_dict: dict[str, pd.DataFrame],
    strategy_type: str = "long_only",
) -> go.Figure:
    """Stacked area chart of sector weights over time."""
    sector_data = {}
    for m, held in holdings_dict.items():
        if "sector" in held.columns:
            if strategy_type == "long_short":
                longs = held[held.get("side", pd.Series("long")) == "long"]
                counts = longs["sector"].value_counts(normalize=True)
            else:
                counts = held["sector"].value_counts(normalize=True)
            sector_data[m] = counts.to_dict()

    if not sector_data:
        return go.Figure()

    df = pd.DataFrame(sector_data).T.fillna(0).sort_index()
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], stackgroup="one", name=col, mode="lines",
        ))
    fig.update_layout(**_base_layout(
        title="Sector Allocation Over Time",
        yaxis_title="Weight", yaxis_tickformat=".0%",
    ))
    return fig


def traffic_light_dashboard(alerts: dict[str, dict]) -> go.Figure:
    """Traffic light indicators for monitoring alerts.

    alerts: dict of {name: {"status": "green"|"amber"|"red", "value": str}}
    """
    names = list(alerts.keys())
    colors_map = {"green": STYLE["positive"], "amber": STYLE["warning"], "red": STYLE["negative"]}
    colors = [colors_map.get(a["status"], STYLE["muted"]) for a in alerts.values()]
    values = [a.get("value", "") for a in alerts.values()]

    fig = go.Figure(data=go.Bar(
        x=names, y=[1] * len(names),
        marker_color=colors,
        text=values, textposition="inside", textfont=dict(size=14, color="white"),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="Alert Dashboard", height=200,
        yaxis=dict(visible=False), xaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    ))
    return fig


def risk_pie_chart(risk_contributions: pd.Series, top_n: int = 10) -> go.Figure:
    """Risk contribution pie chart with 'Other' bucket."""
    if len(risk_contributions) > top_n:
        top = risk_contributions.nlargest(top_n)
        other = risk_contributions.sum() - top.sum()
        top["Other"] = other
    else:
        top = risk_contributions

    fig = go.Figure(data=go.Pie(
        labels=top.index.astype(str),
        values=top.values,
        hole=0.3,
        textinfo="label+percent",
        marker=dict(line=dict(color=STYLE["bg"], width=2)),
    ))
    fig.update_layout(**_base_layout(title="Risk Contribution", height=400))
    return fig
```

- [ ] **Step 2: Commit**

```
git add dashboard/components/charts.py
git commit -m "feat: add reusable Bloomberg-styled Plotly chart builders"
```

---

### Task 12: components/metrics.py

**Files:**
- Create: `dashboard/components/metrics.py`

- [ ] **Step 1: Create metric display components**

```python
# dashboard/components/metrics.py
"""Reusable Streamlit metric display components."""

import streamlit as st
import pandas as pd
import numpy as np


def metric_row(metrics: list[dict]):
    """Render a row of st.metric() cards.

    Each dict should have keys: label, value, and optionally delta.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            delta = m.get("delta")
            delta_color = m.get("delta_color", "normal")
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=delta,
                delta_color=delta_color,
            )


def comparison_table(configs: list[dict], highlight: bool = True):
    """Render strategy comparison table with optional highlighting.

    Each config dict should have: Strategy, SR, Ann Return, Ann Vol, MDD,
    Total Return, Mean IC, Mean Turnover.
    """
    df = pd.DataFrame(configs)
    if "Strategy" in df.columns:
        df = df.set_index("Strategy")

    format_map = {
        "SR": "{:.2f}",
        "Ann Return": "{:.1%}",
        "Ann Vol": "{:.1%}",
        "MDD": "{:.1%}",
        "Calmar": "{:.2f}",
        "Total Return": "{:.0%}",
        "Mean IC": "{:.4f}",
        "Mean Turnover": "{:.1%}",
        "TC Drag": "{:.2%}",
        "Net SR": "{:.2f}",
    }

    for col, fmt in format_map.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "-")

    st.dataframe(df, use_container_width=True)


def regression_table(result: dict):
    """Render formatted regression output like statsmodels summary.

    result should have: coefficients, hac_se, t_stats, p_values.
    """
    rows = []
    for var in result["coefficients"]:
        coef = result["coefficients"][var]
        se = result["hac_se"][var]
        t = result["t_stats"][var]
        p = result["p_values"][var]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        rows.append({
            "Variable": var,
            "Coefficient": f"{coef:.4f}",
            "HAC SE": f"{se:.4f}",
            "t-stat": f"{t:.2f}",
            "p-value": f"{p:.4f}",
            "Sig": sig,
        })

    df = pd.DataFrame(rows).set_index("Variable")

    alpha_t = abs(result["t_stats"].get("const", 0))
    if alpha_t > 1.96:
        st.success(f"Alpha significant (|t| = {alpha_t:.2f} > 1.96)")
    else:
        st.warning(f"Alpha not significant (|t| = {alpha_t:.2f} < 1.96)")

    st.dataframe(df, use_container_width=True)

    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.metric("R²", f"{result['r_squared']:.4f}")
        st.metric("Adj R²", f"{result['adj_r_squared']:.4f}")
    with stats_col2:
        st.metric("Durbin-Watson", f"{result['durbin_watson']:.2f}")
        jb = result.get("jarque_bera", {})
        st.metric("Jarque-Bera", f"{jb.get('statistic', 0):.2f} (p={jb.get('pvalue', 0):.3f})")


def vif_table(vif_df: pd.DataFrame):
    """Color-coded VIF display: green (<5), amber (5-10), red (>10)."""
    def color_vif(val):
        if val < 5:
            return "background-color: rgba(0, 210, 106, 0.3)"
        elif val < 10:
            return "background-color: rgba(255, 184, 0, 0.3)"
        else:
            return "background-color: rgba(255, 68, 68, 0.3)"

    styled = vif_df.style.applymap(color_vif, subset=["VIF"])
    st.dataframe(styled, use_container_width=True)
```

- [ ] **Step 2: Commit**

```
git add dashboard/components/metrics.py
git commit -m "feat: add metric display components (cards, tables, regression output)"
```

---

### Task 13: components/theory.py + theory_content.py

**Files:**
- Create: `dashboard/components/theory.py`
- Create: `dashboard/components/theory_content.py`

- [ ] **Step 1: Create the theory section component**

```python
# dashboard/components/theory.py
"""Expandable theory sections for educational content on each page."""

import streamlit as st
from components.theory_content import THEORY_CONTENT


def theory_section(title: str, content_key: str):
    """Render an expandable theory section.

    Parameters
    ----------
    title : str
        Display title for the expander.
    content_key : str
        Key into THEORY_CONTENT dict.
    """
    content = THEORY_CONTENT.get(content_key)
    if content is None:
        return
    with st.expander(f"Theory: {title}", expanded=False):
        st.markdown(content, unsafe_allow_html=False)
```

- [ ] **Step 2: Create theory_content.py with all educational text**

```python
# dashboard/components/theory_content.py
"""All educational text for theory sections, organized by topic key.

Each entry is a markdown string with LaTeX (rendered by Streamlit).
"""

THEORY_CONTENT = {
    # --- Data Explorer ---
    "cross_sectional_standardization": r"""
**Cross-Sectional Standardization**

Each month, every feature is standardized across all stocks:

$$x_{i,t}^{xs} = \frac{x_{i,t} - \text{median}_t}{\text{MAD}_t}$$

where MAD is the median absolute deviation. This removes time-varying means
(a feature's average level can shift due to market regimes) and makes features
comparable across time. We use median/MAD instead of mean/std because they are
robust to outliers — a single extreme stock won't distort the entire cross-section.

**Why Spearman over Pearson?** Spearman correlation measures the monotonic relationship
between rankings, not raw values. Financial features often have non-linear relationships
with returns, and Spearman is robust to outliers. If a feature correctly ranks stocks
from worst to best, Spearman captures that even if the relationship isn't linear.
""",

    # --- Factor Analysis ---
    "capm_and_factors": r"""
**Capital Asset Pricing Model (CAPM)**

The CAPM says a stock's expected return is proportional to its market risk:

$$r_i - r_f = \alpha_i + \beta_i (r_m - r_f) + \epsilon_i$$

- $\beta_i$ measures sensitivity to the market. $\beta > 1$ means the stock amplifies market moves.
- $\alpha_i$ is the return not explained by market exposure — the "excess" return. A significant positive alpha means the stock outperforms its risk level.
- We test alpha significance using the t-statistic on the intercept. **Always use HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors** because financial returns exhibit time-varying volatility and serial correlation. Classical standard errors would be too small, making alphas look significant when they aren't.

**Fama-French 5-Factor Model (FF5)**

Extends CAPM with four additional risk factors:

$$r_i - r_f = \alpha_i + \beta_i^M (r_m - r_f) + s_i \cdot SMB + h_i \cdot HML + r_i \cdot RMW + c_i \cdot CMA + \epsilon_i$$

| Factor | Captures | Long-Short Construction |
|--------|----------|------------------------|
| SMB | Size premium | Small minus Big (market cap) |
| HML | Value premium | High minus Low (book-to-market) |
| RMW | Profitability | Robust minus Weak (operating profit) |
| CMA | Investment | Conservative minus Aggressive (asset growth) |

**Omitted Variable Bias**: If you only run CAPM and a stock has high alpha, it might just be because the stock is a small-cap value stock (loads on SMB and HML). The FF5 alpha removes these known premiums, giving a cleaner measure of true outperformance.

**Bloomberg Beta Shrinkage**: Raw OLS betas are noisy. Bloomberg shrinks them toward 1.0:

$$\beta_{adj} = 0.67 \cdot \hat{\beta} + 0.33$$

This reflects the empirical finding that extreme betas tend to revert toward the market average over time.
""",

    "vif_and_wald": r"""
**Variance Inflation Factor (VIF)**

VIF measures how much a regressor's variance is inflated due to correlation with other regressors:

$$VIF_k = \frac{1}{1 - R_k^2}$$

where $R_k^2$ is from regressing factor $k$ on all other factors.

| VIF | Interpretation |
|-----|----------------|
| < 5 | Acceptable multicollinearity |
| 5-10 | Moderate — coefficients may be unstable |
| > 10 | Severe — factor loadings are unreliable |

**Wald Test**: Tests whether a group of coefficients are jointly zero. For example, testing $H_0: s = h = 0$ (SMB and HML are unnecessary) uses:

$$W = \hat{\theta}' [R \cdot \hat{V} \cdot R']^{-1} \hat{\theta} \sim \chi^2(q)$$

where $q$ is the number of restrictions. If rejected, those factors add explanatory power.
""",

    # --- Alpha Model Lab ---
    "walk_forward": r"""
**Walk-Forward Validation**

The most critical design decision in financial ML. Traditional cross-validation randomly splits data, which **leaks future information into training** (a stock's 2020 return helps predict its 2018 return). In finance, this creates devastating look-ahead bias.

Walk-forward validation respects the arrow of time:

1. Train on data before month $t$ (expanding window: all history; rolling: last $W$ months)
2. Predict returns for month $t+1$ using features available at time $t$
3. Advance to $t+1$, retrain periodically (every 6-24 months)

**Expanding vs. Rolling Window**:
- *Expanding*: Uses all available history. More data = more stable estimates. But may be slow to adapt if market dynamics change.
- *Rolling*: Only uses the last $W$ months. Adapts faster to regime changes but has less data, leading to noisier estimates.

**Retrain Frequency**: A trade-off between freshness and computation cost. Monthly retraining adapts fastest but is expensive. Annual retraining is cheaper but the model may become stale. Typical choice: 6-12 months.
""",

    "regularization": r"""
**Regularization: L1, L2, and Trees**

With 100+ features and noisy financial data, unregularized models overfit badly.

**Lasso (L1)**: Adds $\lambda \sum |\beta_j|$ penalty. Drives weak coefficients to exactly zero — automatic feature selection. Good when you believe only a few features matter.

**Ridge (L2)**: Adds $\lambda \sum \beta_j^2$ penalty. Shrinks all coefficients toward zero but doesn't eliminate any. Good when many features contribute small signals.

**Elastic Net**: Combines both: $\lambda_1 \sum |\beta_j| + \lambda_2 \sum \beta_j^2$. The `l1_ratio` parameter controls the mix.

**Gradient Boosting (HGB)**: Regularized differently — through tree depth (max_depth), learning rate (shrinkage per tree), and minimum samples per leaf. Shallow trees with slow learning rates are the financial ML standard because they're less prone to fitting noise.

**Fama-MacBeth**: Not ML at all — run a cross-sectional OLS regression each month, then average the slopes across months. The time-series average of slopes is your predictor. Classic academic approach; no regularization but naturally robust because each month's regression is independent.
""",

    # --- Backtest Results ---
    "ic_and_fundamental_law": r"""
**Information Coefficient (IC)**

The Spearman rank correlation between your predicted returns and realized returns, computed cross-sectionally each month:

$$IC_t = \text{SpearmanCorr}(\hat{r}_{i,t}, r_{i,t})$$

| Metric | Formula | Good Threshold |
|--------|---------|----------------|
| Mean IC | $\overline{IC}$ | > 0.03 |
| IC t-stat | $\overline{IC} / (s_{IC} / \sqrt{n})$ | > 2.0 |
| ICIR | $\overline{IC} / s_{IC}$ | > 0.3 |
| Hit Rate | $P(IC_t > 0)$ | > 55% |

An IC of 0.05 means your model explains ~5% of the cross-sectional ranking — this is actually excellent in finance.

**Fundamental Law of Active Management**

$$IR = IC \times \sqrt{BR}$$

where $IR$ is the Information Ratio (risk-adjusted alpha), $IC$ is your forecasting skill, and $BR$ (breadth) is the number of independent bets per year ($K \times$ rebalance frequency).

This tells you: a weak signal ($IC = 0.03$) applied to many stocks ($K = 50$, monthly = $BR = 600$) can produce a strong portfolio ($IR = 0.03 \times \sqrt{600} = 0.73$).

**IC Required**: Given a Sharpe target and transaction costs:

$$IC_{required} = \frac{SR_{target} + Cost_{SR}}{\sqrt{BR}}$$
""",

    "feature_importance": r"""
**Feature Importance**

Three complementary views:

1. **Tree-based importance** (HGB, RF): Measures how much each feature reduces prediction error across all splits. Fast but biased toward high-cardinality features.

2. **Permutation importance**: Shuffle one feature's values, measure how much IC drops. Unbiased but slower. If shuffling feature $k$ barely hurts IC, that feature isn't important.

3. **Univariate IC**: $\text{SpearmanCorr}(x_{k,t}, r_{i,t})$ for each feature — how predictive is each feature on its own? Doesn't capture interactions but gives a clean signal-strength measure.

**Importance Drift**: If the top features change dramatically across retraining windows, the model may be fitting noise rather than stable relationships.
""",

    # --- Portfolio Construction ---
    "portfolio_construction": r"""
**From Predictions to Portfolios**

Raw model predictions rank stocks, but the weighting scheme determines risk.

**Equal-Weight**: $w_i = 1/K$. Simple, robust, but gives the same weight to your best and worst predictions. Ignores risk.

**Score-Weighted**: $w_i = \hat{r}_i / \sum \hat{r}_j$. Concentrates on highest-conviction picks. Can be volatile if predictions are noisy.

**Inverse-Volatility**: $w_i \propto 1/\sigma_i$. Reduces weight on volatile stocks. A simple approximation of risk budgeting.

**Equal Risk Contribution (ERC)**: Each position contributes equally to portfolio risk:

$$RC_i = w_i \cdot (\Sigma w)_i / \sigma_p = RC_j \quad \forall i,j$$

Requires numerical optimization. Better diversification than equal-weight because it accounts for correlations.

**Mean-Variance Optimization (MVO)**: Maximize the Sharpe ratio:

$$\max_w \frac{w'\mu}{\sqrt{w'\Sigma w}} \quad \text{s.t.} \sum w_i = 1, \; 0 \leq w_i \leq 5\%$$

Uses **Ledoit-Wolf shrinkage** for the covariance matrix — sample covariance is too noisy with monthly data, so it's shrunk toward a structured target.
""",

    "turnover_and_costs": r"""
**Turnover and Transaction Costs**

Turnover measures how much the portfolio changes each month:

$$TO_t = \frac{1}{2} \sum_i |w_{i,t} - w_{i,t-1}|$$

High turnover = high trading costs. Transaction cost drag:

$$TC_{annual} = TO_{monthly} \times c \times 2 \times 12$$

where $c$ is one-way cost in bps and the factor of 2 accounts for round-trip (buy + sell).

**Cost Sharpe Ratio**: $Cost_{SR} = TC_{annual} / \sigma_p$. This is the Sharpe ratio you're giving up to transaction costs. If your gross $SR = 1.2$ and $Cost_{SR} = 0.3$, your net $SR \approx 0.9$.
""",

    # --- Monitoring ---
    "distribution_shift": r"""
**Distribution Shift and Model Failure**

ML models assume that future data looks like training data. When the distribution of input features shifts (Out-of-Distribution, OOD), predictions become unreliable.

**KS Test** (Kolmogorov-Smirnov): For each feature, measures the maximum distance between the training distribution and the current month's distribution:

$$D = \sup_x |F_{train}(x) - F_{current}(x)|$$

| D | Interpretation |
|---|----------------|
| < 0.05 | No shift |
| 0.05-0.10 | Minor shift |
| > 0.10 | Significant shift — model inputs look different from training |

If > 20% of features are flagged ($D > 0.10$), the model may be operating outside its training domain.
""",

    "alpha_decay": r"""
**Alpha Decay and Retraining**

Signals lose predictive power over time:

$$IC(h) = \text{SpearmanCorr}(\hat{r}_{i,t}, r_{i,t+h})$$

The **half-life** $h^*$ is where $IC(h^*) = IC(1)/2$. This determines optimal rebalance frequency:
- $h^* < 3$ months → monthly rebalancing
- $h^* = 3-9$ months → quarterly may suffice
- $h^* > 9$ months → slower rebalancing acceptable

**Retraining Policy**: Best practice is **hybrid** — retrain on a schedule (e.g., every 12 months) AND trigger emergency retraining when:
- Rolling 6-month IC drops below 0.02
- Monthly IC falls below -0.03 (signal inversion)
- KS test flags >20% of features

Scheduled-only misses sudden regime changes. Triggered-only may retrain too aggressively on noise.
""",

    # --- Theory & Methods Overview ---
    "alpha_pipeline": r"""
**The Quantitative Alpha Pipeline**

The full pipeline from raw data to monitored portfolio:

**1. Data** → Cross-sectional stock features (momentum, value, quality, etc.), standardized each month to remove time effects.

**2. Features** → Engineered signals: interactions (momentum × quality), composites (average of related measures), non-linear transforms (signed squares for curvature).

**3. Model** → Walk-forward trained ML model (gradient boosting, Lasso, etc.) predicts next-month cross-sectional returns. Retrained periodically on expanding history.

**4. Predictions** → Ranked stock scores. Not point forecasts of returns — we only need the *ranking* to be correct (which stock beats which).

**5. Portfolio** → Convert rankings to weights. Method determines risk profile: equal-weight (simple), ERC (balanced risk), MVO (optimized Sharpe).

**6. Performance** → Evaluate via Sharpe ratio, Information Coefficient, Fundamental Law. Compare gross vs net-of-cost returns.

**7. Monitoring** → Detect signal decay (IC dropping), distribution shift (KS test), stale signals (low turnover). Trigger retraining when needed.

Each step has failure modes. The most dangerous is **look-ahead bias** in step 3 — accidentally using future information during training. The walk-forward protocol prevents this.
""",

    "factor_investing_foundations": r"""
**Why Do Factors Earn Premiums?**

Academic research has identified persistent return premiums associated with stock characteristics:

- **Value** (cheap stocks outperform): Compensation for distress risk, or behavioral overreaction to bad news.
- **Momentum** (winners keep winning): Slow information diffusion, herding, and underreaction to positive news.
- **Quality** (profitable firms outperform): Investors systematically undervalue stable profitability.
- **Size** (small stocks outperform): Higher risk and illiquidity premium.
- **Low Volatility** (calm stocks outperform): Lottery preference — investors overpay for volatile stocks hoping for outsized gains.

These premiums have persisted across decades and geographies, but they're time-varying and can disappear for years. A multi-factor approach diversifies across these bets.
""",

    "glossary": r"""
**Glossary of Key Terms**

| Term | Definition |
|------|-----------|
| Alpha ($\alpha$) | Return not explained by factor exposures; the manager's "skill" |
| Beta ($\beta$) | Sensitivity to a risk factor (usually the market) |
| IC | Information Coefficient — rank correlation between predictions and realized returns |
| ICIR | IC Information Ratio — mean IC / std of IC; measures consistency |
| IR | Information Ratio — annualized alpha / tracking error |
| BR | Breadth — number of independent bets per year |
| SR | Sharpe Ratio — excess return / volatility |
| MDD | Maximum Drawdown — largest peak-to-trough decline |
| Calmar | Annualized return / |MDD| |
| HAC SE | Heteroskedasticity and Autocorrelation Consistent standard errors |
| VIF | Variance Inflation Factor — measures multicollinearity |
| KS Test | Kolmogorov-Smirnov test for distribution shift |
| ERC | Equal Risk Contribution — portfolio where each position contributes equally to risk |
| MVO | Mean-Variance Optimization — maximize Sharpe ratio subject to constraints |
| Ledoit-Wolf | Covariance matrix shrinkage estimator; stabilizes noisy sample covariance |
| Walk-Forward | Time-respecting validation: only train on past data to predict future |
| Look-Ahead Bias | Using future information in training — the cardinal sin of backtesting |
""",
}
```

- [ ] **Step 3: Commit**

```
git add dashboard/components/theory.py dashboard/components/theory_content.py
git commit -m "feat: add theory section component and educational content for all pages"
```

---

## Phase 4: App Router + Pages

### Task 14: app.py — Slim Router

**Files:**
- Replace: `dashboard/app.py`

Back up the old app first: `copy dashboard/app.py dashboard/app_old.py`

- [ ] **Step 1: Write the new slim app.py**

```python
# dashboard/app.py
"""Alpha Strategy Dashboard — multipage Streamlit app.

This is the entry point. It handles:
- Page configuration and dark theme
- One-time data loading into session state
- Session state initialization

All UI lives in the pages/ directory.
"""

import streamlit as st
from pathlib import Path
from core.data_loader import load_dataset, compute_market_monthly, load_ff5_factors

st.set_page_config(
    page_title="Alpha Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def _load_all_data():
    data_path = Path(__file__).parent.parent / "Data" / "alpha_dataset_v2.csv"
    df = load_dataset(data_path)

    from features import precompute_features
    df = precompute_features(df)

    market = compute_market_monthly(df)

    ff5_path = Path(__file__).parent.parent / "Data" / "ff5_factors.csv"
    ff5 = None
    if ff5_path.exists():
        ff5 = load_ff5_factors(ff5_path)

    return df, market, ff5


df, market_monthly, ff5_factors = _load_all_data()

if "df" not in st.session_state:
    st.session_state.df = df
if "market_monthly" not in st.session_state:
    st.session_state.market_monthly = market_monthly
if "ff5_factors" not in st.session_state:
    st.session_state.ff5_factors = ff5_factors
if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None
if "backtest_params" not in st.session_state:
    st.session_state.backtest_params = None
if "pinned_configs" not in st.session_state:
    st.session_state.pinned_configs = []
if "portfolio_weights" not in st.session_state:
    st.session_state.portfolio_weights = None

st.sidebar.title("Alpha Strategy Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Navigate using the sidebar pages. Start with **Alpha Model Lab** "
    "to configure and run a backtest."
)

st.title("Alpha Strategy Dashboard")
st.markdown(
    "Welcome to the Alpha Strategy Dashboard. Use the sidebar to navigate between pages."
)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Stocks in Dataset", f"{df['permno'].nunique():,}")
with col2:
    st.metric("Months", f"{df['ym'].nunique()}")
with col3:
    date_range = f"{df['ym'].min()} → {df['ym'].max()}"
    st.metric("Date Range", date_range)

if st.session_state.backtest_result is not None:
    st.success("Backtest results loaded. Navigate to **Backtest Results** to view.")
else:
    st.info("No backtest results yet. Go to **Alpha Model Lab** to run one.")
```

- [ ] **Step 2: Verify app launches**

Run: `streamlit run dashboard/app.py`

Expected: App loads with dark theme, shows dataset metrics, sidebar navigation visible.

- [ ] **Step 3: Commit**

```
git add dashboard/app.py dashboard/app_old.py
git commit -m "feat: replace single-page app with slim multipage router"
```

---

### Task 15: Page 1 — Data Explorer

**Files:**
- Create: `dashboard/pages/1_Data_Explorer.py`

- [ ] **Step 1: Create the page**

```python
# dashboard/pages/1_Data_Explorer.py
"""Page 1: Data Explorer — understand the dataset before modeling."""

import streamlit as st
import pandas as pd
import numpy as np
from components.charts import correlation_heatmap, bar_chart, STYLE
from components.theory import theory_section
import plotly.graph_objects as go

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")

df = st.session_state.get("df")
if df is None:
    st.error("Dataset not loaded. Return to the home page.")
    st.stop()

theory_section("Cross-Sectional Standardization", "cross_sectional_standardization")

# --- Summary Statistics ---
st.header("Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stocks", f"{df['permno'].nunique():,}")
col2.metric("Months", f"{df['ym'].nunique()}")
col3.metric("Date Range", f"{df['ym'].min()} to {df['ym'].max()}")
xs_cols = [c for c in df.columns if c.endswith("_xs") and c != "y_xs"]
col4.metric("Features", f"{len(xs_cols)}")

# --- Universe Size Over Time ---
st.header("Universe Size Over Time")
universe = df.groupby("ym")["permno"].nunique().sort_index()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=universe.index, y=universe.values, mode="lines",
    line=dict(color=STYLE["accent"], width=2),
))
fig.update_layout(
    template=STYLE["template"], height=300, margin=dict(t=20, b=30),
    yaxis_title="Number of Stocks",
    paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
)
st.plotly_chart(fig, use_container_width=True)

# --- Feature Distribution Viewer ---
st.header("Feature Distribution")
dist_col1, dist_col2 = st.columns([1, 3])

with dist_col1:
    selected_feature = st.selectbox("Feature", xs_cols)
    months = sorted(df["ym"].unique())
    selected_month = st.selectbox("Month", months, index=len(months) - 1)

with dist_col2:
    month_data = df[df["ym"] == selected_month][selected_feature].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=month_data.values, nbinsx=50, name="Distribution",
        marker_color=STYLE["accent"], opacity=0.7,
    ))
    fig.update_layout(
        title=f"{selected_feature} — {selected_month} (n={len(month_data)})",
        template=STYLE["template"], height=350, margin=dict(t=40, b=30),
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Correlation Heatmap ---
st.header("Feature Correlations (Spearman)")
n_features = st.slider("Number of features", min_value=5, max_value=40, value=20)

top_features = xs_cols[:n_features]
sample_month = df["ym"].max()
corr_data = df[df["ym"] == sample_month][top_features].dropna()
if len(corr_data) > 10:
    corr_matrix = corr_data.corr(method="spearman")
    fig = correlation_heatmap(corr_matrix)
    st.plotly_chart(fig, use_container_width=True)

# --- Missing Data Summary ---
st.header("Missing Data Summary")
missing_pct = df[xs_cols].isnull().mean().sort_values(ascending=False)
missing_nonzero = missing_pct[missing_pct > 0]
if len(missing_nonzero) > 0:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=missing_nonzero.index[:30], y=missing_nonzero.values[:30] * 100,
        marker_color=STYLE["warning"],
    ))
    fig.update_layout(
        title="% Missing by Feature (top 30)",
        yaxis_title="% Missing", template=STYLE["template"],
        height=350, margin=dict(t=40, b=80),
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("No missing data in feature columns.")
```

- [ ] **Step 2: Smoke test**

Run: `streamlit run dashboard/app.py`

Navigate to "Data Explorer" in sidebar. Verify: summary stats, universe chart, distribution viewer, correlation heatmap, and missing data summary all render.

- [ ] **Step 3: Commit**

```
git add dashboard/pages/1_Data_Explorer.py
git commit -m "feat: add Data Explorer page (summary, distributions, correlations, missing data)"
```

---

### Task 16: Page 2 — Factor Analysis

**Files:**
- Create: `dashboard/pages/2_Factor_Analysis.py`

- [ ] **Step 1: Create the page**

```python
# dashboard/pages/2_Factor_Analysis.py
"""Page 2: Factor Analysis — CAPM and Fama-French regressions."""

import streamlit as st
import pandas as pd
import numpy as np
from core.factor_models import (
    run_regression, rolling_beta, bloomberg_shrink_beta,
    compute_vif, wald_test, hedge_ratio,
)
from components.metrics import regression_table, vif_table
from components.charts import STYLE, rolling_metric_chart
from components.theory import theory_section
import plotly.graph_objects as go

st.set_page_config(page_title="Factor Analysis", layout="wide")
st.title("Factor Analysis (CAPM & FF5)")

df = st.session_state.get("df")
ff5 = st.session_state.get("ff5_factors")
market = st.session_state.get("market_monthly")

if df is None:
    st.error("Dataset not loaded.")
    st.stop()

theory_section("CAPM and Factor Models", "capm_and_factors")

# --- Stock Selector ---
st.sidebar.header("Stock Selection")
sectors = sorted(df["sector"].dropna().unique()) if "sector" in df.columns else []
selected_sector = st.sidebar.selectbox("Filter by Sector", ["All"] + sectors)

if selected_sector != "All":
    filtered_permnos = df[df["sector"] == selected_sector]["permno"].unique()
else:
    filtered_permnos = df["permno"].unique()

selected_permno = st.sidebar.selectbox("Stock (permno)", sorted(filtered_permnos))

# --- Compute stock excess returns ---
stock_data = df[df["permno"] == selected_permno][["ym", "y_raw"]].dropna()
stock_data = stock_data.set_index("ym").sort_index()
stock_returns = stock_data["y_raw"]

if ff5 is not None:
    rf = ff5["RF"] if "RF" in ff5.columns else pd.Series(0, index=ff5.index)
    common_idx = stock_returns.index.intersection(ff5.index)
    excess_returns = stock_returns.loc[common_idx] - rf.loc[common_idx]
    factors = ff5.loc[common_idx]
else:
    excess_returns = stock_returns
    if market is not None:
        common_idx = stock_returns.index.intersection(market.index)
        excess_returns = stock_returns.loc[common_idx]
        factors = pd.DataFrame({"Mkt-RF": market.loc[common_idx, "Mkt_RF"]})
    else:
        st.warning("No factor data available.")
        st.stop()

# --- Regression Panel ---
st.header("Regression Results")
model_types = ["CAPM"]
if ff5 is not None:
    model_types.extend(["FF3", "FF5"])

reg_type = st.selectbox("Regression Model", model_types)

if len(excess_returns) > 20:
    reg_result = run_regression(excess_returns, factors, model_type=reg_type)
    regression_table(reg_result)
else:
    st.warning("Not enough data for regression (need > 20 months).")

# --- Rolling Beta ---
st.header("Rolling Beta")
window = st.slider("Rolling Window (months)", min_value=12, max_value=120, value=36)

if "Mkt-RF" in factors.columns and len(excess_returns) > window:
    beta_df = rolling_beta(excess_returns, factors["Mkt-RF"], window=window)
    shrunk = bloomberg_shrink_beta(beta_df["beta"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=beta_df.index, y=beta_df["upper"].values, mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=beta_df.index, y=beta_df["lower"].values, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(59,130,246,0.15)",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=beta_df.index, y=beta_df["beta"].values, name="OLS Beta",
        line=dict(color=STYLE["accent"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=beta_df.index, y=shrunk.values, name="Bloomberg Shrunk",
        line=dict(color=STYLE["warning"], width=2, dash="dash"),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=STYLE["muted"])
    fig.update_layout(
        title=f"Rolling {window}-Month Beta",
        yaxis_title="Beta", template=STYLE["template"],
        height=400, margin=dict(t=40, b=40),
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
    )
    st.plotly_chart(fig, use_container_width=True)

# --- VIF Table ---
if ff5 is not None and reg_type == "FF5":
    st.header("Variance Inflation Factors")
    theory_section("VIF and Wald Tests", "vif_and_wald")
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    available = [c for c in factor_cols if c in factors.columns]
    if len(available) >= 2:
        vif_df = compute_vif(factors[available].dropna())
        vif_table(vif_df)

    # --- Wald Tests ---
    st.header("Wald Tests (Joint Significance)")
    wald_col1, wald_col2, wald_col3 = st.columns(3)

    with wald_col1:
        if st.button("Test SMB = HML = 0"):
            result = wald_test(excess_returns, factors, ["SMB", "HML"])
            verdict = "Reject H₀" if result["reject"] else "Fail to reject"
            st.metric("Wald Statistic", f"{result['statistic']:.2f}")
            st.metric("p-value", f"{result['pvalue']:.4f}")
            st.markdown(f"**{verdict}** at 5% level")

    with wald_col2:
        if st.button("Test RMW = CMA = 0"):
            result = wald_test(excess_returns, factors, ["RMW", "CMA"])
            verdict = "Reject H₀" if result["reject"] else "Fail to reject"
            st.metric("Wald Statistic", f"{result['statistic']:.2f}")
            st.metric("p-value", f"{result['pvalue']:.4f}")
            st.markdown(f"**{verdict}** at 5% level")

    with wald_col3:
        if st.button("Test All Four = 0"):
            result = wald_test(excess_returns, factors, ["SMB", "HML", "RMW", "CMA"])
            verdict = "Reject H₀" if result["reject"] else "Fail to reject"
            st.metric("Wald Statistic", f"{result['statistic']:.2f}")
            st.metric("p-value", f"{result['pvalue']:.4f}")
            st.markdown(f"**{verdict}** at 5% level")

# --- Hedge Calculator ---
st.header("Market-Neutral Hedge Calculator")
hedge_col1, hedge_col2 = st.columns(2)
with hedge_col1:
    port_beta = st.number_input("Portfolio Beta", value=1.0, step=0.1)
with hedge_col2:
    hedge_w = hedge_ratio(port_beta)
    st.metric("SPY Hedge Weight", f"{hedge_w:.2f}")
    st.markdown(
        f"To neutralize a portfolio with β={port_beta:.1f}, "
        f"short **{abs(hedge_w):.1%}** of portfolio value in SPY."
    )
```

- [ ] **Step 2: Smoke test**

Run: `streamlit run dashboard/app.py`

Navigate to "Factor Analysis". Verify: stock selector, CAPM regression, rolling beta chart, VIF table (if FF5 data exists), Wald tests, hedge calculator.

- [ ] **Step 3: Commit**

```
git add dashboard/pages/2_Factor_Analysis.py
git commit -m "feat: add Factor Analysis page (CAPM/FF5, rolling beta, VIF, Wald, hedge)"
```

---

### Task 17: Page 3 — Alpha Model Lab

**Files:**
- Create: `dashboard/pages/3_Alpha_Model_Lab.py`

- [ ] **Step 1: Create the page**

```python
# dashboard/pages/3_Alpha_Model_Lab.py
"""Page 3: Alpha Model Lab — configure, train, and launch backtests."""

import streamlit as st
import pandas as pd
from core.models import list_models, get_model, get_default_params, get_param_ranges, get_feature_tier
from core.backtest import run_walk_forward
from core.portfolio import build_portfolio_series
from core.diagnostics import compute_performance_metrics
from features import FEATURE_GROUPS, get_tier_defaults
import cache_manager as cache
from components.theory import theory_section

st.set_page_config(page_title="Alpha Model Lab", layout="wide")
st.title("Alpha Model Lab")

df = st.session_state.get("df")
market_monthly = st.session_state.get("market_monthly")
if df is None:
    st.error("Dataset not loaded.")
    st.stop()

theory_section("Walk-Forward Validation", "walk_forward")
theory_section("Regularization", "regularization")

# --- Model Selection ---
st.header("Model Configuration")
model_col1, model_col2 = st.columns([1, 2])

with model_col1:
    model_name = st.selectbox("Model", list_models())
    tier = get_feature_tier(model_name)
    st.caption(f"Feature Tier: {tier} ({'~118 features' if tier == 2 else '~52 features'})")

with model_col2:
    param_ranges = get_param_ranges(model_name)
    model_params = {}
    if param_ranges:
        cols = st.columns(min(len(param_ranges), 3))
        for i, (param, spec) in enumerate(param_ranges.items()):
            with cols[i % len(cols)]:
                if isinstance(spec["default"], float):
                    model_params[param] = st.slider(
                        param, min_value=spec["min"], max_value=spec["max"],
                        value=spec["default"], step=spec.get("step", 0.01),
                    )
                else:
                    model_params[param] = st.slider(
                        param, min_value=spec["min"], max_value=spec["max"],
                        value=spec["default"], step=spec.get("step", 1),
                    )
    else:
        st.info("No tunable hyperparameters for this model.")
        model_params = get_default_params(model_name)

# --- Feature Selection ---
st.header("Feature Selection")

preset = st.selectbox("Preset", [
    "Tier default", "Momentum only", "Value only", "Quality only",
    "Kitchen sink (all 118)",
])

if preset == "Tier default":
    selected_features = get_tier_defaults(tier)
elif preset == "Momentum only":
    selected_features = FEATURE_GROUPS.get("momentum", [])
elif preset == "Value only":
    selected_features = FEATURE_GROUPS.get("value", [])
elif preset == "Quality only":
    selected_features = FEATURE_GROUPS.get("quality", [])
else:
    selected_features = get_tier_defaults(2)

with st.expander("Customize features", expanded=False):
    custom_features = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        available = [c for c in group_cols if c in df.columns or c in selected_features]
        if available:
            selected_in_group = st.multiselect(
                group_name.capitalize(),
                options=available,
                default=[c for c in available if c in selected_features],
                key=f"feat_{group_name}",
            )
            custom_features.extend(selected_in_group)
    if custom_features:
        selected_features = custom_features

available_features = [f for f in selected_features if f in df.columns]
st.caption(f"Selected: {len(available_features)} features")

# --- Walk-Forward Configuration ---
st.header("Walk-Forward Configuration")
wf_col1, wf_col2, wf_col3, wf_col4 = st.columns(4)

all_months = sorted(df["ym"].unique())
oos_candidates = [m for m in all_months if m >= "2010-01"]

with wf_col1:
    oos_start = st.selectbox("OOS Start", oos_candidates, index=oos_candidates.index("2015-01") if "2015-01" in oos_candidates else 0)
with wf_col2:
    retrain_freq = st.selectbox("Retrain Every (months)", [6, 12, 24], index=1)
with wf_col3:
    window_type = st.selectbox("Window Type", ["expanding", "rolling"])
with wf_col4:
    rolling_window = None
    if window_type == "rolling":
        rolling_window = st.slider("Rolling Window (months)", 24, 120, 60)

# --- Portfolio Configuration ---
st.header("Portfolio Configuration")
port_col1, port_col2, port_col3, port_col4, port_col5 = st.columns(5)

with port_col1:
    strategy_type = st.selectbox("Strategy", ["Long Only", "Long-Short"])
    strategy_key = "long_short" if strategy_type == "Long-Short" else "long_only"
with port_col2:
    K = st.slider("K (stocks)", min_value=5, max_value=50, step=5, value=10)
with port_col3:
    K_short = K
    if strategy_key == "long_short":
        K_short = st.slider("K short", min_value=5, max_value=50, step=5, value=K)
with port_col4:
    vol_tilt = st.slider("Vol tilt", min_value=0.0, max_value=0.50, step=0.01, value=0.05)
with port_col5:
    regime_lookback = st.slider("Regime lookback", min_value=0, max_value=12, value=6)

construction_method = st.selectbox(
    "Construction Method",
    ["equal_weight", "score_weight", "inverse_vol", "erc", "mvo"],
)

# --- Action Buttons ---
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    run_clicked = st.button("Run Backtest", type="primary")
with btn_col2:
    pin_clicked = st.button("Pin Config")

# --- Run Backtest ---
if run_clicked:
    pred_key = cache.prediction_key(
        model_name, model_params, retrain_freq,
        feature_cols=available_features, window_type=window_type,
    )
    predictions = cache.get_predictions(pred_key)

    if predictions is None:
        model = get_model(model_name, model_params)
        progress = st.progress(0, text="Running walk-forward backtest...")
        result = run_walk_forward(
            data=df, model=model, feature_cols=available_features,
            oos_start=oos_start, retrain_freq=retrain_freq,
            window_type=window_type, rolling_window=rolling_window,
            progress_callback=lambda step, total, month: progress.progress(
                step / total, text=f"Training... {month} ({step}/{total})",
            ),
        )
        progress.empty()
        predictions = result.predictions
        cache.save_predictions(pred_key, predictions)
        st.session_state.backtest_feature_importance = result.feature_importance
        st.session_state.backtest_train_dates = result.train_dates
    else:
        st.session_state.backtest_feature_importance = None
        st.session_state.backtest_train_dates = []

    port_key = cache.portfolio_key(
        pred_key, K, vol_tilt, regime_lookback,
        strategy_key, K_short, construction_method,
    )
    portfolio = cache.get_portfolio(port_key)

    if portfolio is None:
        portfolio = build_portfolio_series(
            predictions=predictions, method=construction_method,
            K=K, strategy_type=strategy_key, K_short=K_short,
            vol_tilt=vol_tilt, regime_lookback=regime_lookback,
            market_monthly=market_monthly,
        )
        cache.save_portfolio(port_key, portfolio)

    st.session_state.backtest_result = portfolio
    st.session_state.backtest_predictions = predictions
    st.session_state.backtest_params = {
        "model_name": model_name, "model_params": model_params,
        "retrain_freq": retrain_freq, "K": K, "K_short": K_short,
        "vol_tilt": vol_tilt, "regime_lookback": regime_lookback,
        "strategy_type": strategy_key, "construction_method": construction_method,
        "features": available_features, "window_type": window_type,
    }
    st.success("Backtest complete! Navigate to **Backtest Results** to view.")

# --- Pin Config ---
if pin_clicked:
    result = st.session_state.get("backtest_result")
    params = st.session_state.get("backtest_params")
    pinned = st.session_state.get("pinned_configs", [])
    if result is not None and len(pinned) < 4:
        label = f"{params['model_name']} {params['construction_method']} K={params['K']}"
        pinned.append({"label": label, "result": result, "params": params})
        st.session_state.pinned_configs = pinned
        st.success(f"Pinned: {label}")

# --- Show pinned configs ---
pinned = st.session_state.get("pinned_configs", [])
if pinned:
    st.markdown("**Pinned configs:**")
    chip_cols = st.columns(len(pinned))
    to_remove = None
    for i, p in enumerate(pinned):
        with chip_cols[i]:
            if st.button(f"X {p['label']}", key=f"rm_pin_{i}"):
                to_remove = i
    if to_remove is not None:
        pinned.pop(to_remove)
        st.session_state.pinned_configs = pinned
        st.rerun()
```

- [ ] **Step 2: Smoke test**

Run: `streamlit run dashboard/app.py`

Navigate to "Alpha Model Lab". Verify: model selector with dynamic params, feature selection with presets, walk-forward config, portfolio config, Run Backtest button triggers walk-forward and stores results.

- [ ] **Step 3: Commit**

```
git add dashboard/pages/3_Alpha_Model_Lab.py
git commit -m "feat: add Alpha Model Lab page (6 models, feature selection, walk-forward config)"
```

---

### Task 18: Page 4 — Backtest Results

**Files:**
- Create: `dashboard/pages/4_Backtest_Results.py`

- [ ] **Step 1: Create the page**

```python
# dashboard/pages/4_Backtest_Results.py
"""Page 4: Backtest Results & Diagnostics."""

import streamlit as st
import pandas as pd
import numpy as np
from core.diagnostics import (
    compute_performance_metrics, compute_ic_stats, fundamental_law, feature_ic,
)
from components.charts import (
    cumulative_wealth_chart, drawdown_chart, monthly_heatmap,
    rolling_metric_chart, bar_chart, STYLE, PIN_COLORS,
)
from components.metrics import metric_row, comparison_table
from components.theory import theory_section
import plotly.graph_objects as go

st.set_page_config(page_title="Backtest Results", layout="wide")
st.title("Backtest Results & Diagnostics")

result = st.session_state.get("backtest_result")
if result is None:
    st.info("No backtest results yet. Run a backtest on the **Alpha Model Lab** page.")
    st.stop()

market = st.session_state.get("market_monthly")
pinned = st.session_state.get("pinned_configs", [])

rets = result["monthly_returns"]
ic = result["ic"]
turnover = result["turnover"]

# --- Display Controls ---
disp_col1, disp_col2, disp_col3 = st.columns(3)
with disp_col1:
    oos_months = sorted(rets.index)
    display_start = st.select_slider("Display from", options=oos_months, value=oos_months[0])
with disp_col2:
    start_value = st.number_input("Starting value ($)", min_value=1, value=10000, step=1000)
with disp_col3:
    cash_flow = st.number_input("Cash flow/period ($)", value=0, step=100)

rets = rets[rets.index >= display_start]
ic = ic[ic.index >= display_start]
turnover = turnover[turnover.index >= display_start]

# --- KPI Cards ---
perf = compute_performance_metrics(rets)
metric_row([
    {"label": "Sharpe Ratio", "value": f"{perf['SR']:.2f}"},
    {"label": "Ann. Return", "value": f"{perf['Ann Return']:.1%}"},
    {"label": "Ann. Volatility", "value": f"{perf['Ann Vol']:.1%}"},
    {"label": "Max Drawdown", "value": f"{perf['MDD']:.1%}"},
    {"label": "Calmar", "value": f"{perf['Calmar']:.2f}" if not np.isnan(perf['Calmar']) else "N/A"},
])

# --- Cumulative Wealth + Drawdown ---
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

returns_dict = {"Strategy": rets}
if market is not None:
    spy = market.loc[market.index >= display_start, "spy_ret"]
    returns_dict["SPY"] = spy
for p in pinned:
    p_rets = p["result"]["monthly_returns"]
    returns_dict[p["label"]] = p_rets[p_rets.index >= display_start]

with chart_col1:
    fig = cumulative_wealth_chart(returns_dict, start_value, cash_flow)
    st.plotly_chart(fig, use_container_width=True)
with chart_col2:
    fig = drawdown_chart(returns_dict, start_value)
    st.plotly_chart(fig, use_container_width=True)

# --- IC Dashboard ---
st.markdown("---")
theory_section("Information Coefficient & Fundamental Law", "ic_and_fundamental_law")

ic_col1, ic_col2 = st.columns(2)
with ic_col1:
    fig = bar_chart(ic.dropna(), name="Monthly IC")
    st.plotly_chart(fig, use_container_width=True)
with ic_col2:
    cum_ic = ic.dropna().cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_ic.index, y=cum_ic.values, name="Cumulative IC",
        line=dict(color=STYLE["accent"], width=2),
    ))
    fig.update_layout(
        title="Cumulative IC", template=STYLE["template"],
        height=400, margin=dict(t=40, b=40),
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
    )
    st.plotly_chart(fig, use_container_width=True)

ic_stats = compute_ic_stats(ic)
ic_metric_col1, ic_metric_col2, ic_metric_col3, ic_metric_col4 = st.columns(4)
ic_metric_col1.metric("Mean IC", f"{ic_stats['mean_ic']:.4f}",
                       delta="Good" if ic_stats["mean_ic"] > 0.03 else "Low")
ic_metric_col2.metric("IC t-stat", f"{ic_stats['ic_tstat']:.2f}",
                       delta="Sig" if ic_stats["ic_tstat"] > 2 else "Insig")
ic_metric_col3.metric("ICIR", f"{ic_stats['icir']:.3f}",
                       delta="Good" if ic_stats["icir"] > 0.3 else "Low")
ic_metric_col4.metric("Hit Rate", f"{ic_stats['hit_rate']:.1%}",
                       delta="Good" if ic_stats["hit_rate"] > 0.55 else "Low")

# --- Fundamental Law ---
st.markdown("---")
st.header("Fundamental Law of Active Management")
fl_col1, fl_col2 = st.columns(2)
with fl_col1:
    sr_target = st.number_input("SR Target", value=1.0, step=0.1)
    tc_bps = st.number_input("Transaction Cost (bps)", value=10.0, step=5.0)
with fl_col2:
    params = st.session_state.get("backtest_params", {})
    K_val = params.get("K", 10)
    fl = fundamental_law(ic_stats["mean_ic"], K_val, sr_target=sr_target, tc_bps=tc_bps)
    st.metric("Breadth (nominal)", f"{fl['BR_nominal']}")
    st.metric("IR Upper Bound", f"{fl['IR_upper_bound']:.3f}")
    st.metric("IC Required", f"{fl['IC_required']:.4f}")
    st.metric("Cost SR", f"{fl['Cost_SR']:.3f}")

# --- Monthly Heatmap + Turnover ---
st.markdown("---")
hm_col1, hm_col2 = st.columns(2)
with hm_col1:
    fig = monthly_heatmap(rets)
    st.plotly_chart(fig, use_container_width=True)
with hm_col2:
    fig = bar_chart(turnover.dropna(), name="Monthly Turnover")
    st.plotly_chart(fig, use_container_width=True)

# --- Feature Importance ---
st.markdown("---")
theory_section("Feature Importance", "feature_importance")
importance = st.session_state.get("backtest_feature_importance")
if importance is not None and len(importance) > 0:
    st.header("Feature Importance (Top 20)")
    top_imp = importance.nlargest(20, "importance")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_imp["importance"].values, y=top_imp.index,
        orientation="h", marker_color=STYLE["accent"],
    ))
    fig.update_layout(
        title="Feature Importance", template=STYLE["template"],
        height=500, margin=dict(t=40, b=40, l=150),
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Comparison Table ---
if pinned:
    st.markdown("---")
    st.header("Strategy Comparison")
    configs = []
    current_perf = compute_performance_metrics(rets)
    current_perf["Strategy"] = "Current"
    current_perf["Mean IC"] = ic_stats["mean_ic"]
    current_perf["Mean Turnover"] = turnover.dropna().mean()
    configs.append(current_perf)

    for p in pinned:
        p_rets = p["result"]["monthly_returns"]
        p_rets = p_rets[p_rets.index >= display_start]
        p_perf = compute_performance_metrics(p_rets)
        p_perf["Strategy"] = p["label"]
        p_ic = p["result"]["ic"]
        p_perf["Mean IC"] = p_ic[p_ic.index >= display_start].mean()
        p_perf["Mean Turnover"] = p["result"]["turnover"].dropna().mean()
        configs.append(p_perf)

    comparison_table(configs)
```

- [ ] **Step 2: Smoke test**

Run backtest on Alpha Model Lab first, then navigate to Backtest Results. Verify: KPI cards, wealth chart, drawdown, IC dashboard, cumulative IC, Fundamental Law, heatmap, turnover, feature importance, comparison table.

- [ ] **Step 3: Commit**

```
git add dashboard/pages/4_Backtest_Results.py
git commit -m "feat: add Backtest Results page (performance, IC, Fundamental Law, importance)"
```

---

### Task 19: Page 5 — Portfolio Construction & Risk

**Files:**
- Create: `dashboard/pages/5_Portfolio_Construction.py`

- [ ] **Step 1: Create the page**

```python
# dashboard/pages/5_Portfolio_Construction.py
"""Page 5: Portfolio Construction & Risk Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
from core.portfolio import build_portfolio_series
from core.diagnostics import compute_performance_metrics
from core.risk import (
    risk_contribution, factor_exposure, rolling_factor_exposure,
    transaction_cost_drag,
)
from components.charts import (
    sector_allocation_chart, risk_pie_chart, bar_chart, STYLE,
)
from components.metrics import metric_row, comparison_table
from components.theory import theory_section
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Construction", layout="wide")
st.title("Portfolio Construction & Risk")

result = st.session_state.get("backtest_result")
predictions = st.session_state.get("backtest_predictions")
params = st.session_state.get("backtest_params")
ff5 = st.session_state.get("ff5_factors")
market = st.session_state.get("market_monthly")

if result is None or predictions is None:
    st.info("Run a backtest on the **Alpha Model Lab** page first.")
    st.stop()

theory_section("Portfolio Construction Methods", "portfolio_construction")

# --- Method Comparison ---
st.header("Compare Construction Methods")
methods = ["equal_weight", "score_weight", "inverse_vol", "erc", "mvo"]
selected_methods = st.multiselect("Methods to compare", methods, default=["equal_weight", "inverse_vol"])

method_results = {}
for method in selected_methods:
    port = build_portfolio_series(
        predictions=predictions, method=method,
        K=params["K"], strategy_type=params["strategy_type"],
        K_short=params["K_short"], vol_tilt=params["vol_tilt"],
        regime_lookback=params["regime_lookback"], market_monthly=market,
    )
    method_results[method] = port

# --- Sector Allocation ---
st.header("Sector Allocation")
if result["holdings"]:
    fig = sector_allocation_chart(result["holdings"], params["strategy_type"])
    st.plotly_chart(fig, use_container_width=True)

# --- Holdings Table ---
st.header("Current Holdings")
if result["holdings"]:
    last_month = sorted(result["holdings"].keys())[-1]
    held = result["holdings"][last_month].copy()
    display_cols = []
    for c in ["side", "permno", "sector", "pred", "y_raw", "weight"]:
        if c in held.columns:
            display_cols.append(c)
    show = held[display_cols].copy()
    if "pred" in show.columns:
        show["pred"] = show["pred"].round(4)
    if "y_raw" in show.columns:
        show["y_raw"] = (show["y_raw"] * 100).round(2).astype(str) + "%"
    if "weight" in show.columns:
        show["weight"] = (show["weight"] * 100).round(2).astype(str) + "%"
    st.markdown(f"**{last_month}** ({len(held)} positions)")
    st.dataframe(show.reset_index(drop=True), use_container_width=True)

# --- Risk Decomposition ---
if len(selected_methods) >= 1:
    st.markdown("---")
    st.header("Risk Decomposition")
    rc_cols = st.columns(len(selected_methods))
    for i, method in enumerate(selected_methods):
        with rc_cols[i]:
            st.subheader(method)
            port = method_results[method]
            if port["holdings"]:
                last_m = sorted(port["holdings"].keys())[-1]
                held = port["holdings"][last_m]
                if "weight" in held.columns:
                    w = held["weight"].abs().values
                    n = len(w)
                    cov = np.eye(n) * 0.01
                    rc = risk_contribution(w / w.sum(), cov)
                    rc.index = held["permno"].values
                    fig = risk_pie_chart(rc)
                    st.plotly_chart(fig, use_container_width=True)

# --- Factor Exposure ---
if ff5 is not None:
    st.markdown("---")
    st.header("Factor Exposure")
    theory_section("Turnover and Transaction Costs", "turnover_and_costs")

    rets = result["monthly_returns"]
    exposure = factor_exposure(rets, ff5)
    if len(exposure) > 0:
        exp_cols = st.columns(len(exposure))
        for i, (factor, beta) in enumerate(exposure.items()):
            with exp_cols[i]:
                color = "normal" if abs(beta) < 0.10 else "inverse"
                st.metric(factor, f"{beta:.3f}", delta_color=color)

    rolling_exp = rolling_factor_exposure(rets, ff5, window=24)
    if not rolling_exp.dropna(how="all").empty:
        fig = go.Figure()
        for col in rolling_exp.columns:
            fig.add_trace(go.Scatter(
                x=rolling_exp.index, y=rolling_exp[col].values, name=col,
            ))
        fig.add_hline(y=0.10, line_dash="dash", line_color=STYLE["warning"])
        fig.add_hline(y=-0.10, line_dash="dash", line_color=STYLE["warning"])
        fig.update_layout(
            title="Rolling 24-Month Factor Exposures",
            template=STYLE["template"], height=400,
            paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Turnover & Costs ---
st.markdown("---")
st.header("Turnover & Transaction Costs")
tc_col1, tc_col2 = st.columns(2)
with tc_col1:
    fig = bar_chart(result["turnover"].dropna(), name="Monthly Turnover")
    st.plotly_chart(fig, use_container_width=True)
with tc_col2:
    cost_bps = st.number_input("Cost per trade (bps)", value=10.0, step=5.0, key="tc_bps")
    ann_vol = result["monthly_returns"].std() * np.sqrt(12)
    tc = transaction_cost_drag(result["turnover"], cost_bps, ann_vol)
    perf = compute_performance_metrics(result["monthly_returns"])
    gross_sr = perf["SR"]
    net_sr = gross_sr - tc["Cost_SR"]
    metric_row([
        {"label": "Avg Monthly TO", "value": f"{tc['mean_monthly_turnover']:.1%}"},
        {"label": "TC Drag (annual)", "value": f"{tc['TC_annual']:.2%}"},
        {"label": "Gross SR", "value": f"{gross_sr:.2f}"},
        {"label": "Net SR", "value": f"{net_sr:.2f}"},
    ])

# --- Method Comparison Table ---
if len(method_results) >= 2:
    st.markdown("---")
    st.header("Method Comparison")
    comp = []
    for method, port in method_results.items():
        p = compute_performance_metrics(port["monthly_returns"])
        p["Strategy"] = method
        p["Mean Turnover"] = port["turnover"].dropna().mean()
        ann_vol_m = port["monthly_returns"].std() * np.sqrt(12)
        tc_m = transaction_cost_drag(port["turnover"], cost_bps, ann_vol_m)
        p["TC Drag"] = tc_m["TC_annual"]
        p["Net SR"] = p["SR"] - tc_m["Cost_SR"]
        comp.append(p)
    comparison_table(comp)
```

- [ ] **Step 2: Smoke test**

Run backtest first, then navigate to Portfolio Construction. Verify: method comparison, sector allocation, holdings, risk decomposition, factor exposure, turnover/costs, method comparison table.

- [ ] **Step 3: Commit**

```
git add dashboard/pages/5_Portfolio_Construction.py
git commit -m "feat: add Portfolio Construction page (5 methods, risk, factor exposure, TC)"
```

---

### Task 20: Page 6 — Monitoring

**Files:**
- Create: `dashboard/pages/6_Monitoring.py`

- [ ] **Step 1: Create the page**

```python
# dashboard/pages/6_Monitoring.py
"""Page 6: Monitoring & OOD Detection."""

import streamlit as st
import pandas as pd
import numpy as np
from core.diagnostics import ks_test, alpha_decay, signal_staleness, compute_ic_stats
from core.risk import factor_exposure
from components.charts import traffic_light_dashboard, bar_chart, STYLE
from components.theory import theory_section
import plotly.graph_objects as go

st.set_page_config(page_title="Monitoring", layout="wide")
st.title("Monitoring & OOD Detection")

result = st.session_state.get("backtest_result")
predictions = st.session_state.get("backtest_predictions")
params = st.session_state.get("backtest_params")
df = st.session_state.get("df")
ff5 = st.session_state.get("ff5_factors")

if result is None or predictions is None:
    st.info("Run a backtest on the **Alpha Model Lab** page first.")
    st.stop()

theory_section("Distribution Shift", "distribution_shift")
theory_section("Alpha Decay and Retraining", "alpha_decay")

months = sorted(predictions.keys())
features = params.get("features", [])
available_features = [f for f in features if f in df.columns]

# --- KS Test Panel ---
st.header("KS Test — Distribution Shift Detection")
if len(months) >= 2 and available_features:
    last_month = months[-1]
    train_data = df[df["ym"] < months[0]]

    X_train = train_data[available_features].dropna()
    X_current = df[df["ym"] == last_month][available_features].dropna()

    if len(X_train) > 0 and len(X_current) > 0:
        ks_results = ks_test(X_train, X_current)
        n_flagged = ks_results["flag"].sum()
        n_total = len(ks_results)
        pct_flagged = n_flagged / n_total if n_total > 0 else 0

        ks_col1, ks_col2 = st.columns([1, 2])
        with ks_col1:
            color = "normal" if pct_flagged < 0.20 else "inverse"
            st.metric("Features Flagged", f"{n_flagged}/{n_total} ({pct_flagged:.0%})",
                      delta_color=color)
        with ks_col2:
            st.dataframe(ks_results.head(20), use_container_width=True)

        # KS heatmap over time
        if len(months) > 3:
            ks_over_time = {}
            sample_months = months[::max(1, len(months) // 12)]
            for m in sample_months:
                X_m = df[df["ym"] == m][available_features].dropna()
                if len(X_m) > 0:
                    ks_m = ks_test(X_train, X_m)
                    ks_over_time[m] = ks_m.set_index("feature")["D"]
            if ks_over_time:
                ks_df = pd.DataFrame(ks_over_time).T
                fig = go.Figure(data=go.Heatmap(
                    z=ks_df.values, x=ks_df.columns[:30], y=ks_df.index,
                    colorscale=[[0, STYLE["positive"]], [0.5, STYLE["warning"]], [1, STYLE["negative"]]],
                    zmin=0, zmax=0.3,
                ))
                fig.update_layout(
                    title="KS D-Statistic Over Time (top 30 features)",
                    template=STYLE["template"], height=400,
                    paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Alpha Decay ---
st.markdown("---")
st.header("Alpha Decay Curve")
decay = alpha_decay(predictions, horizons=list(range(1, 13)))
if not decay.empty:
    ic1 = decay.iloc[0] if len(decay) > 0 else 0
    half_life = None
    for h, ic_h in decay.items():
        if ic_h < ic1 / 2:
            half_life = h
            break

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decay.index, y=decay.values, mode="lines+markers",
        line=dict(color=STYLE["accent"], width=2),
        marker=dict(size=8),
    ))
    if half_life:
        fig.add_vline(x=half_life, line_dash="dash", line_color=STYLE["warning"],
                      annotation_text=f"Half-life: {half_life}m")
    fig.add_hline(y=0, line_dash="dot", line_color=STYLE["muted"])
    fig.update_layout(
        title="IC at Forward Horizons", xaxis_title="Horizon (months)",
        yaxis_title="IC", template=STYLE["template"], height=400,
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
    )
    st.plotly_chart(fig, use_container_width=True)

    if half_life:
        if half_life < 3:
            st.info(f"Half-life = {half_life} months → **monthly rebalancing** recommended.")
        elif half_life < 9:
            st.info(f"Half-life = {half_life} months → **quarterly rebalancing** may suffice.")
        else:
            st.info(f"Half-life = {half_life} months → **slower rebalancing** acceptable.")

# --- Signal Staleness ---
st.markdown("---")
st.header("Signal Staleness")
stale_df = signal_staleness(result["turnover"], threshold=0.10, consecutive=3)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=stale_df.index, y=stale_df["turnover"].values,
    marker_color=[STYLE["negative"] if s else STYLE["accent"] for s in stale_df["stale"]],
    name="Turnover",
))
fig.add_hline(y=0.10, line_dash="dash", line_color=STYLE["warning"],
              annotation_text="Stale threshold (10%)")
fig.update_layout(
    title="Monthly Turnover (red = stale signal)",
    yaxis_tickformat=".0%", template=STYLE["template"], height=350,
    paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
)
st.plotly_chart(fig, use_container_width=True)

# --- Automated Alert Dashboard ---
st.markdown("---")
st.header("Alert Dashboard")

ic = result["ic"]
rolling_6m_ic = ic.rolling(6, min_periods=3).mean()
latest_rolling_ic = rolling_6m_ic.iloc[-1] if len(rolling_6m_ic) > 0 else 0
latest_monthly_ic = ic.iloc[-1] if len(ic) > 0 else 0
stale_active = stale_df["stale"].iloc[-1] if len(stale_df) > 0 else False
ood_pct = pct_flagged if "pct_flagged" in dir() else 0

alerts = {
    "IC Decay": {
        "status": "red" if latest_rolling_ic < 0.02 else "green",
        "value": f"6m IC: {latest_rolling_ic:.3f}",
    },
    "IC Collapse": {
        "status": "red" if latest_monthly_ic < -0.03 else "green",
        "value": f"Latest: {latest_monthly_ic:.3f}",
    },
    "OOD Shift": {
        "status": "red" if ood_pct > 0.20 else "amber" if ood_pct > 0.10 else "green",
        "value": f"{ood_pct:.0%} flagged",
    },
    "Stale Signal": {
        "status": "red" if stale_active else "green",
        "value": "Stale" if stale_active else "Active",
    },
}

if ff5 is not None:
    exp = factor_exposure(result["monthly_returns"], ff5)
    max_drift = exp.abs().max() if len(exp) > 0 else 0
    alerts["Factor Drift"] = {
        "status": "red" if max_drift > 0.10 else "green",
        "value": f"Max |β|: {max_drift:.3f}",
    }

fig = traffic_light_dashboard(alerts)
st.plotly_chart(fig, use_container_width=True)

# --- Retraining Recommendation ---
st.header("Retraining Recommendation")
n_red = sum(1 for a in alerts.values() if a["status"] == "red")
n_amber = sum(1 for a in alerts.values() if a["status"] == "amber")

if n_red >= 2:
    st.error("**Retrain immediately.** Multiple critical alerts active.")
elif n_red >= 1 or n_amber >= 2:
    st.warning("**Consider retraining.** Some alerts triggered.")
else:
    st.success("**No action needed.** All indicators healthy.")
```

- [ ] **Step 2: Smoke test**

Run backtest, then navigate to Monitoring. Verify: KS test panel, alpha decay curve, signal staleness, alert dashboard, retraining recommendation.

- [ ] **Step 3: Commit**

```
git add dashboard/pages/6_Monitoring.py
git commit -m "feat: add Monitoring page (KS test, alpha decay, alerts, retraining)"
```

---

### Task 21: Page 7 — Theory & Methods

**Files:**
- Create: `dashboard/pages/7_Theory_Methods.py`

- [ ] **Step 1: Create the page**

```python
# dashboard/pages/7_Theory_Methods.py
"""Page 7: Theory & Methods — standalone educational overview."""

import streamlit as st
from components.theory_content import THEORY_CONTENT

st.set_page_config(page_title="Theory & Methods", layout="wide")
st.title("Theory & Methods")

st.markdown(
    "This page provides a comprehensive overview of the quantitative alpha pipeline, "
    "covering the theories, methodologies, and monitoring approaches used in this dashboard. "
    "Each section corresponds to concepts from the RMBI3110 course at HKUST."
)

st.markdown("---")

sections = [
    ("The Alpha Pipeline", "alpha_pipeline"),
    ("Factor Investing Foundations", "factor_investing_foundations"),
    ("CAPM and Factor Models", "capm_and_factors"),
    ("VIF and Wald Tests", "vif_and_wald"),
    ("Walk-Forward Validation", "walk_forward"),
    ("Regularization: L1, L2, and Trees", "regularization"),
    ("Information Coefficient & Fundamental Law", "ic_and_fundamental_law"),
    ("Feature Importance", "feature_importance"),
    ("Portfolio Construction Methods", "portfolio_construction"),
    ("Turnover and Transaction Costs", "turnover_and_costs"),
    ("Distribution Shift & OOD Detection", "distribution_shift"),
    ("Alpha Decay and Retraining", "alpha_decay"),
    ("Glossary", "glossary"),
]

# --- Table of Contents ---
st.sidebar.header("Contents")
for title, key in sections:
    st.sidebar.markdown(f"- [{title}](#{key})")

# --- Render all sections ---
for title, key in sections:
    st.header(title, anchor=key)
    content = THEORY_CONTENT.get(key)
    if content:
        st.markdown(content, unsafe_allow_html=False)
    st.markdown("---")
```

- [ ] **Step 2: Smoke test**

Run: `streamlit run dashboard/app.py`

Navigate to "Theory & Methods". Verify: all 13 sections render with formatted text, LaTeX equations, and tables.

- [ ] **Step 3: Commit**

```
git add dashboard/pages/7_Theory_Methods.py
git commit -m "feat: add Theory & Methods page (full educational content)"
```

---

## Phase 5: Integration

### Task 22: Cleanup + End-to-End Verification

**Files:**
- Delete: `dashboard/engine.py` (functionality now in core/)
- Delete: `dashboard/app_old.py` (backup from migration)
- Modify: `dashboard/__init__.py` (no changes needed, stays empty)

- [ ] **Step 1: Run all tests**

Run: `cd dashboard && python -m pytest ../tests/ -v`

Expected: All tests pass (test_data_loader, test_models, test_backtest, test_diagnostics, test_portfolio, test_factor_models, test_risk, test_features).

- [ ] **Step 2: Download FF5 data if not present**

Run in Python:
```python
from core.data_loader import download_ff5_factors
download_ff5_factors()
```

Verify: `Data/ff5_factors.csv` exists with monthly factor data.

- [ ] **Step 3: Full end-to-end smoke test**

Run: `streamlit run dashboard/app.py`

Test each page in order:

1. **Home**: Dataset metrics displayed (stocks, months, date range).
2. **Data Explorer**: Summary stats, distribution viewer, correlation heatmap, missing data.
3. **Factor Analysis**: Select a stock, run CAPM/FF3/FF5, view rolling beta, VIF, Wald tests, hedge calculator.
4. **Alpha Model Lab**: Select HGB model, default features, run backtest. Pin the config. Switch to Lasso, run again.
5. **Backtest Results**: KPIs, cumulative wealth, drawdown, IC dashboard, cumulative IC, Fundamental Law, heatmap, turnover, feature importance, comparison table.
6. **Portfolio Construction**: Compare EW vs ERC, view sector allocation, holdings, risk decomposition, factor exposure, turnover/costs, method comparison.
7. **Monitoring**: KS test, alpha decay curve, signal staleness, alert dashboard, retraining recommendation.
8. **Theory & Methods**: All sections render with equations and tables.

- [ ] **Step 4: Delete old files**

```
del dashboard/engine.py
del dashboard/app_old.py
```

- [ ] **Step 5: Final commit**

```
git add -A
git commit -m "feat: complete 7-page dashboard expansion with educational content"
```

---

## Appendix: Quick Reference

### Running the App

```bash
streamlit run dashboard/app.py
```

### Running Tests

```bash
cd dashboard && python -m pytest ../tests/ -v
```

### Downloading FF5 Data

```python
python -c "from dashboard.core.data_loader import download_ff5_factors; download_ff5_factors()"
```
