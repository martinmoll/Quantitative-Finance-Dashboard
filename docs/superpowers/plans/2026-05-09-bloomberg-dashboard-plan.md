# Alpha Strategy Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit dashboard that wraps the existing walk-forward ML pipeline into an interactive tool with tunable parameters, two-tier caching, and full analytics (cumulative wealth, drawdown, holdings, IC, turnover, heatmap).

**Architecture:** Four Python modules under `dashboard/`: `features.py` (extracted feature engineering), `engine.py` (split walk-forward into prediction + portfolio stages for two-tier caching), `cache_manager.py` (hash-based disk cache), and `app.py` (Streamlit layout with top controls, KPI cards, and 2-column chart grid). All charts use Plotly.

**Tech Stack:** Streamlit, Plotly, pandas, numpy, scipy, scikit-learn, hashlib, pickle

**Karpathy guidelines apply:** Simplicity first, surgical extraction from notebook, no over-engineering, verify each step works before moving on.

---

## File Structure

```
dashboard/
  features.py         # Feature engineering (extracted from notebook cells)
  engine.py           # run_predictions() + build_portfolio() — the two-tier split
  cache_manager.py    # Hash-based disk cache for predictions and portfolios
  app.py              # Streamlit entry point — controls, KPIs, charts
  .cache/             # Auto-created cache directory (gitignored)
```

**Key design decision:** The notebook's `run_model()` is split into two functions:
- `run_predictions()` — trains model, returns per-stock predictions per month (expensive, cached by model params)
- `build_portfolio()` — takes predictions, applies K/vol_tilt/regime to produce returns + holdings (cheap, cached by portfolio params)

This enables instant portfolio-level param changes without rerunning the model.

---

### Task 1: Extract Feature Engineering into `dashboard/features.py`

**Files:**
- Create: `dashboard/features.py`
- Reference: `Code/MartinMoellenhus_Assignment2.ipynb` (cell `1p3z1u6w73i`)

- [ ] **Step 1: Create the dashboard directory**

```bash
mkdir dashboard
```

- [ ] **Step 2: Write `features.py` with all feature functions**

Extract these four functions verbatim from the notebook:
- `_get_existing(df_slice, cols)`
- `_build_engineered_features(df_slice)`
- `build_features_linear(df_slice)`
- `build_features_ensemble(df_slice)`

```python
import pandas as pd
import numpy as np


def _get_existing(df_slice, cols):
    return df_slice[[c for c in cols if c in df_slice.columns]]


def _build_engineered_features(df_slice):
    feat = pd.DataFrame(index=df_slice.index)

    def _col(name):
        return df_slice[name] if name in df_slice.columns else pd.Series(0.0, index=df_slice.index)

    feat['mom_x_quality'] = _col('ret_2_12_xs') * _col('gpa_xs')
    feat['mom_x_roe'] = _col('ret_2_12_xs') * _col('roe_xs')
    feat['val_x_lowvol'] = _col('bm_xs') * (-_col('vol_12m_xs'))
    feat['ep_x_lowvol'] = _col('ep_xs') * (-_col('vol_12m_xs'))
    feat['sue_x_lowdisp'] = _col('sue_xs') * (-_col('dispersion_xs'))
    feat['sue_x_revision'] = _col('sue_xs') * _col('revision_xs')
    feat['mom_x_lowivol'] = _col('ret_2_12_xs') * (-_col('ivol_xs'))
    feat['bm_x_roe'] = _col('bm_xs') * _col('roe_xs')
    feat['ep_x_gpa'] = _col('ep_xs') * _col('gpa_xs')

    for col in ['ret_2_12_xs', 'ret_1_xs', 'sue_xs', 'bm_xs', 'revision_xs']:
        raw = _col(col)
        feat[f'{col}_sq'] = raw ** 2 * np.sign(raw)

    earn_cols = ['sue_xs', 'sue_q_xs', 'rev_surp_xs', 'earn_growth_yoy_xs', 'beat_xs']
    feat['earnings_composite'] = _get_existing(df_slice, earn_cols).mean(axis=1)

    qual_cols = ['gpa_xs', 'roe_xs', 'roa_xs', 'earn_quality_xs', 'cfo_at_xs']
    feat['quality_composite'] = _get_existing(df_slice, qual_cols).mean(axis=1)

    val_cols = ['bm_xs', 'ep_xs', 'cfp_xs', 'sp_xs']
    feat['value_composite'] = _get_existing(df_slice, val_cols).mean(axis=1)

    mom_cols = ['ret_2_12_xs', 'ret_2_6_xs', 'ret_13_36_xs', 'prc_52w_high_xs']
    feat['momentum_composite'] = _get_existing(df_slice, mom_cols).mean(axis=1)

    tech_cols = ['rsi_14_xs', 'macd_hist_xs', 'bb_position_xs', 'roc_3_xs', 'roc_6_xs']
    feat['technical_composite'] = _get_existing(df_slice, tech_cols).mean(axis=1)

    analyst_cols = ['revision_xs', 'revision_ratio_xs', 'rec_chg_xs', 'n_analysts_xs']
    feat['analyst_composite'] = _get_existing(df_slice, analyst_cols).mean(axis=1)

    feat['sue_vs_peer'] = _col('sue_xs') - _col('peer_sue_xs')
    feat['revision_vs_peer'] = _col('revision_xs') - _col('peer_revision_xs')
    feat['reversal_mom_combo'] = _col('ret_2_12_xs') - _col('ret_1_xs')

    feat['earn_x_mom'] = feat['earnings_composite'] * feat['momentum_composite']
    feat['quality_x_value'] = feat['quality_composite'] * feat['value_composite']
    feat['earn_x_lowvol'] = feat['earnings_composite'] * (-_col('vol_12m_xs'))

    return feat.fillna(0.0)


def build_features_linear(df_slice):
    feat = pd.DataFrame(index=df_slice.index)
    core = [
        'ret_1_xs', 'ret_2_12_xs', 'ret_2_6_xs',
        'bm_xs', 'ep_xs', 'cfp_xs', 'sp_xs',
        'gpa_xs', 'roe_xs', 'roa_xs',
        'vol_12m_xs', 'ivol_xs', 'beta_xs',
        'log_me_xs',
        'sue_xs', 'revision_xs', 'beat_xs',
        'turnover_xs', 'illiq_12m_xs',
        'mom_x_size_xs', 'val_x_prof_xs', 'mom_x_vol_xs',
        'ret_vs_sector_xs', 'bm_vs_sector_xs', 'ret_vs_ind_xs',
        'bm_vs_size_xs',
    ]
    for c in core:
        if c in df_slice.columns:
            feat[c] = df_slice[c]
    engineered = _build_engineered_features(df_slice)
    feat = pd.concat([feat, engineered], axis=1)
    return feat.fillna(0.0)


def build_features_ensemble(df_slice):
    xs_cols = [c for c in df_slice.columns if c.endswith('_xs') and c != 'y_xs']
    feat = df_slice[xs_cols].copy()
    engineered = _build_engineered_features(df_slice)
    feat = pd.concat([feat, engineered], axis=1)
    return feat.fillna(0.0)
```

- [ ] **Step 3: Verify feature counts match notebook**

```bash
cd dashboard
python -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from features import build_features_linear, build_features_ensemble
df = pd.read_csv('../Data/alpha_dataset_v2.csv')
lin = build_features_linear(df)
ens = build_features_ensemble(df)
print(f'Tier 1: {lin.shape[1]} (expect 52)')
print(f'Tier 2: {ens.shape[1]} (expect 118)')
assert lin.shape[1] == 52, f'Tier 1 mismatch: {lin.shape[1]}'
assert ens.shape[1] == 118, f'Tier 2 mismatch: {ens.shape[1]}'
print('OK')
"
```

Expected: `Tier 1: 52`, `Tier 2: 118`, `OK`

- [ ] **Step 4: Commit**

```bash
git add dashboard/features.py
git commit -m "Extract feature engineering into dashboard/features.py"
```

---

### Task 2: Build the Walk-Forward Engine (`dashboard/engine.py`)

**Files:**
- Create: `dashboard/engine.py`
- Reference: `Code/MartinMoellenhus_Assignment2.ipynb` (cells `3df80422`, `apygdmev3ul`, `ul1mtfq4zfd`)

The key change from the notebook: split `run_model()` into `run_predictions()` (expensive) and `build_portfolio()` (cheap). `run_predictions()` returns per-stock prediction DataFrames. `build_portfolio()` applies K/vol_tilt/regime to produce returns, holdings, IC, and turnover.

- [ ] **Step 1: Write `engine.py`**

```python
import pandas as pd
import numpy as np
from sklearn.base import clone
from scipy.stats import spearmanr

TRAIN_TARGET = 'y_xs'
EVAL_TARGET = 'y_raw'
OOS_START = '2015-01'


def compute_market_monthly(df):
    market = df.groupby('ym')[['Mkt_RF', 'rf_ff']].first().sort_index()
    market['spy_ret'] = market['Mkt_RF'] + market['rf_ff']
    return market


def run_predictions(df, feature_builder, estimator, retrain_every=12,
                    progress_callback=None):
    """Walk-forward prediction generation. Returns per-month DataFrames.

    Returns: dict of {month_str: DataFrame with columns [permno, sector, pred, y_raw]}
    """
    all_months = sorted(df['ym'].unique())
    oos_months = [m for m in all_months if m >= OOS_START]
    retrain_months = set(oos_months[::retrain_every])

    predictions = {}
    model = None
    total = len(oos_months)

    for i, m in enumerate(oos_months):
        if m in retrain_months:
            train = df[df['ym'] < m].dropna(subset=[TRAIN_TARGET])
            model = clone(estimator)
            model.fit(feature_builder(train), train[TRAIN_TARGET])

        test = df[df['ym'] == m].copy()
        if len(test) < 20:
            continue

        test['pred'] = model.predict(feature_builder(test))

        keep_cols = ['permno', 'sector', 'pred', EVAL_TARGET]
        if 'vol_12m_xs' in test.columns:
            keep_cols.append('vol_12m_xs')
        predictions[m] = test[keep_cols].copy()

        if progress_callback:
            progress_callback((i + 1) / total)

    return predictions


def build_portfolio(predictions, K=10, vol_tilt=0.0, regime_lookback=6,
                    market_monthly=None):
    """Build portfolio from cached predictions. Cheap to recompute.

    Returns dict with keys:
        monthly_returns: pd.Series of monthly portfolio returns
        holdings: dict of {month: DataFrame of held stocks}
        ic: pd.Series of monthly IC values
        turnover: pd.Series of monthly turnover values
    """
    if regime_lookback > 0 and market_monthly is not None:
        trailing_spy = market_monthly['spy_ret'].rolling(regime_lookback).sum().shift(1)
        regime_on = trailing_spy >= 0
    else:
        regime_on = None

    monthly_returns = {}
    holdings = {}
    ic_series = {}
    turnover_series = {}
    prev_weights = None

    for m in sorted(predictions.keys()):
        month_df = predictions[m].copy()

        if len(month_df) < 2 * K:
            continue

        if vol_tilt > 0 and 'vol_12m_xs' in month_df.columns:
            month_df['pred'] = month_df['pred'] - vol_tilt * month_df['vol_12m_xs'].fillna(0)

        # IC before portfolio selection
        valid = month_df[['pred', EVAL_TARGET]].dropna()
        ic = spearmanr(valid['pred'], valid[EVAL_TARGET])[0] if len(valid) > 10 else np.nan
        ic_series[m] = ic

        top = month_df.nlargest(K, 'pred')
        ret = top[EVAL_TARGET].mean()

        # Apply regime filter
        if regime_on is not None and m in regime_on.index and not regime_on[m]:
            ret = 0.0

        monthly_returns[m] = ret
        holdings[m] = top

        # Turnover
        curr_weights = pd.Series(0.0, index=month_df['permno'].values)
        curr_weights[top['permno'].values] = 1.0 / K
        if prev_weights is not None:
            aligned = curr_weights.align(prev_weights, fill_value=0.0)
            turnover_series[m] = 0.5 * (aligned[0] - aligned[1]).abs().sum()
        prev_weights = curr_weights

    return {
        'monthly_returns': pd.Series(monthly_returns).sort_index(),
        'holdings': holdings,
        'ic': pd.Series(ic_series).sort_index(),
        'turnover': pd.Series(turnover_series).sort_index(),
    }


def compute_perf(monthly_returns, name=''):
    """Compute performance statistics from a monthly return series."""
    s = monthly_returns
    mu = s.mean() * 12
    vol = s.std() * np.sqrt(12)
    sr = mu / vol if vol > 0 else 0
    cum = (1 + s).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return {
        'Strategy': name,
        'SR': round(sr, 2),
        'Ann Return': round(mu * 100, 1),
        'Ann Vol': round(vol * 100, 1),
        'MDD': round(mdd * 100, 1),
        'Total Return': round((cum.iloc[-1] - 1) * 100, 0),
        'Mean IC': None,
        'Mean Turnover': None,
    }
```

- [ ] **Step 2: Verify engine produces same results as notebook**

```bash
cd dashboard
python -c "
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
from features import build_features_ensemble
from engine import run_predictions, build_portfolio, compute_market_monthly
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_csv('../Data/alpha_dataset_v2.csv')
market = compute_market_monthly(df)

hgb = HistGradientBoostingRegressor(
    max_iter=500, max_depth=2, learning_rate=0.05,
    min_samples_leaf=500, l2_regularization=0.1,
    early_stopping=False, random_state=42)

print('Running predictions (this takes ~30-60s)...')
preds = run_predictions(df, build_features_ensemble, hgb, retrain_every=12)
print(f'Got predictions for {len(preds)} months')

result = build_portfolio(preds, K=10, vol_tilt=0.05, regime_lookback=6, market_monthly=market)
sr = result['monthly_returns'].mean() / result['monthly_returns'].std() * np.sqrt(12)
cum = (1 + result['monthly_returns']).cumprod()
mdd = (cum / cum.cummax() - 1).min()
print(f'SR={sr:.2f}, MDD={mdd:.1%}')
print(f'Expected: SR~1.28, MDD~-27.6%')
assert abs(sr - 1.28) < 0.15, f'SR too far off: {sr:.2f}'
assert mdd > -0.35, f'MDD too large: {mdd:.1%}'
print('OK')
"
```

Expected: `SR~1.28`, `MDD~-27.6%`, `OK`

- [ ] **Step 3: Commit**

```bash
git add dashboard/engine.py
git commit -m "Add walk-forward engine with prediction/portfolio split"
```

---

### Task 3: Build the Cache Manager (`dashboard/cache_manager.py`)

**Files:**
- Create: `dashboard/cache_manager.py`

- [ ] **Step 1: Write `cache_manager.py`**

Two-tier disk cache: prediction cache (keyed on model params) and portfolio cache (keyed on prediction key + portfolio params).

```python
import hashlib
import json
import os
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).parent / '.cache'
PRED_DIR = CACHE_DIR / 'predictions'
PORT_DIR = CACHE_DIR / 'portfolios'


def _ensure_dirs():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PORT_DIR.mkdir(parents=True, exist_ok=True)


def _make_key(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def prediction_key(model_type, model_params, retrain_every):
    return _make_key({
        'model_type': model_type,
        'model_params': model_params,
        'retrain_every': retrain_every,
    })


def portfolio_key(pred_key, K, vol_tilt, regime_lookback):
    return _make_key({
        'pred_key': pred_key,
        'K': K,
        'vol_tilt': vol_tilt,
        'regime_lookback': regime_lookback,
    })


def get_predictions(key):
    _ensure_dirs()
    path = PRED_DIR / f'{key}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_predictions(key, predictions):
    _ensure_dirs()
    path = PRED_DIR / f'{key}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(predictions, f)


def get_portfolio(key):
    _ensure_dirs()
    path = PORT_DIR / f'{key}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_portfolio(key, portfolio):
    _ensure_dirs()
    path = PORT_DIR / f'{key}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(portfolio, f)
```

- [ ] **Step 2: Verify cache round-trips correctly**

```bash
cd dashboard
python -c "
import sys
sys.path.insert(0, '.')
from cache_manager import prediction_key, portfolio_key, save_predictions, get_predictions, save_portfolio, get_portfolio

pk = prediction_key('HGB', {'max_depth': 2}, 12)
print(f'Prediction key: {pk}')

# Save and load
save_predictions(pk, {'2015-01': 'test_data'})
loaded = get_predictions(pk)
assert loaded == {'2015-01': 'test_data'}, 'Prediction cache round-trip failed'

ppk = portfolio_key(pk, 10, 0.05, 6)
print(f'Portfolio key: {ppk}')
save_portfolio(ppk, {'returns': [1, 2, 3]})
loaded2 = get_portfolio(ppk)
assert loaded2 == {'returns': [1, 2, 3]}, 'Portfolio cache round-trip failed'

# Cache miss returns None
assert get_predictions('nonexistent') is None
print('OK')
"
```

Expected: Two hash keys printed, `OK`

- [ ] **Step 3: Add `.cache/` to `.gitignore`**

Append to the project's `.gitignore`:

```
dashboard/.cache/
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/cache_manager.py .gitignore
git commit -m "Add two-tier disk cache for predictions and portfolios"
```

---

### Task 4: Build the Streamlit App — Controls and Data Loading (`dashboard/app.py`)

**Files:**
- Create: `dashboard/app.py`

This task builds the control bar, data loading, and the backtest execution wiring. Charts come in the next tasks.

- [ ] **Step 1: Write the app skeleton with controls**

```python
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV

from features import build_features_linear, build_features_ensemble
from engine import run_predictions, build_portfolio, compute_market_monthly, compute_perf
import cache_manager as cache

st.set_page_config(page_title="Alpha Dashboard", layout="wide")
st.title("Alpha Strategy Dashboard")


@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / 'Data' / 'alpha_dataset_v2.csv'
    df = pd.read_csv(data_path)
    market = compute_market_monthly(df)
    spy_oos = market.loc[market.index >= '2015-01', 'spy_ret']
    return df, market, spy_oos


df, market_monthly, spy_oos = load_data()

# --- Top Control Bar ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    model_type = st.selectbox("Model", ["HGB", "Lasso"])
with col2:
    K = st.slider("K (stocks)", 5, 50, 10, step=5)
with col3:
    vol_tilt = st.slider("Vol tilt", 0.0, 0.50, 0.05, step=0.01)
with col4:
    regime_lookback = st.slider("Regime lookback (months)", 0, 12, 6)

# --- Expandable Model Hyperparams ---
with st.expander("Model Hyperparameters"):
    if model_type == "HGB":
        hgb_col1, hgb_col2, hgb_col3 = st.columns(3)
        with hgb_col1:
            max_depth = st.slider("max_depth", 1, 6, 2)
            learning_rate = st.slider("learning_rate", 0.01, 0.20, 0.05, step=0.01)
        with hgb_col2:
            min_samples_leaf = st.slider("min_samples_leaf", 100, 2000, 500, step=100)
            l2_reg = st.slider("l2_regularization", 0.0, 1.0, 0.1, step=0.05)
        with hgb_col3:
            max_iter = st.slider("max_iter", 100, 1000, 500, step=50)
        retrain_every = st.slider("Retrain frequency (months)", 3, 24, 12, step=3)

        model_params = {
            'max_depth': max_depth, 'learning_rate': learning_rate,
            'min_samples_leaf': min_samples_leaf, 'l2_regularization': l2_reg,
            'max_iter': max_iter, 'early_stopping': False, 'random_state': 42,
        }
    else:
        lasso_col1, lasso_col2 = st.columns(2)
        with lasso_col1:
            cv_folds = st.slider("CV folds", 3, 10, 5)
        with lasso_col2:
            lasso_max_iter = st.slider("max_iter", 1000, 10000, 5000, step=1000)
        retrain_every = st.slider("Retrain frequency (months)", 3, 24, 12, step=3)

        model_params = {
            'cv': cv_folds, 'max_iter': lasso_max_iter,
        }

# --- Run / Pin buttons ---
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)
with btn_col2:
    pin_clicked = st.button("Pin Config", use_container_width=True)

# --- Initialize session state ---
if 'pinned' not in st.session_state:
    st.session_state.pinned = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'current_params' not in st.session_state:
    st.session_state.current_params = None

# --- Run Backtest ---
if run_clicked:
    pred_key = cache.prediction_key(model_type, model_params, retrain_every)
    predictions = cache.get_predictions(pred_key)

    if predictions is None:
        if model_type == "HGB":
            estimator = HistGradientBoostingRegressor(**model_params)
            feature_builder = build_features_ensemble
        else:
            estimator = LassoCV(**model_params)
            feature_builder = build_features_linear

        progress = st.progress(0, text="Running walk-forward backtest...")
        predictions = run_predictions(
            df, feature_builder, estimator, retrain_every,
            progress_callback=lambda p: progress.progress(p, text=f"Training... {p:.0%}")
        )
        cache.save_predictions(pred_key, predictions)
        progress.empty()

    port_key = cache.portfolio_key(pred_key, K, vol_tilt, regime_lookback)
    portfolio = cache.get_portfolio(port_key)

    if portfolio is None:
        portfolio = build_portfolio(predictions, K, vol_tilt, regime_lookback, market_monthly)
        cache.save_portfolio(port_key, portfolio)

    st.session_state.current_result = portfolio
    st.session_state.current_params = {
        'model': model_type, 'K': K, 'vol_tilt': vol_tilt,
        'regime': regime_lookback, **model_params,
    }

# --- Pin Config ---
if pin_clicked and st.session_state.current_result is not None:
    if len(st.session_state.pinned) < 4:
        label = f"{st.session_state.current_params['model']} K={st.session_state.current_params['K']} vt={st.session_state.current_params['vol_tilt']}"
        st.session_state.pinned.append({
            'label': label,
            'result': st.session_state.current_result,
            'params': st.session_state.current_params,
        })

# --- Pinned Config Chips ---
if st.session_state.pinned:
    st.markdown("**Pinned configs:**")
    chip_cols = st.columns(len(st.session_state.pinned) + 1)
    to_remove = None
    for i, pinned in enumerate(st.session_state.pinned):
        with chip_cols[i]:
            if st.button(f"❌ {pinned['label']}", key=f"unpin_{i}"):
                to_remove = i
    if to_remove is not None:
        st.session_state.pinned.pop(to_remove)
        st.rerun()

# --- Charts rendered below (Tasks 5-8) ---
```

- [ ] **Step 2: Smoke test — app starts and controls render**

```bash
cd dashboard
streamlit run app.py --server.headless true &
sleep 3
curl -s http://localhost:8501 | head -5
# Should return HTML content (Streamlit app)
# Kill the server after check
kill %1
```

On Windows, just run `streamlit run app.py` and verify in browser that controls appear. Kill with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "Add Streamlit app with controls, data loading, and caching wiring"
```

---

### Task 5: Add KPI Cards and Core Performance Charts

**Files:**
- Modify: `dashboard/app.py` (append after the pinned chips section)

- [ ] **Step 1: Add chart helper imports and KPI rendering**

Append to `app.py` after the pinned chips section:

```python
import plotly.graph_objects as go

PIN_COLORS = ['#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6']

result = st.session_state.current_result
if result is not None:
    rets = result['monthly_returns']
    perf_stats = compute_perf(rets)
    perf_stats['Mean IC'] = round(result['ic'].mean(), 4) if len(result['ic']) > 0 else None
    perf_stats['Mean Turnover'] = round(result['turnover'].mean() * 100, 1) if len(result['turnover']) > 0 else None

    # --- Row 1: KPI Cards ---
    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    sr = perf_stats['SR']
    sr_color = "green" if sr > 1.0 else ("orange" if sr > 0.5 else "red")
    kpi1.metric("Sharpe Ratio", f"{sr:.2f}")

    ann_ret = perf_stats['Ann Return']
    kpi2.metric("Ann. Return", f"{ann_ret:.1f}%")

    ann_vol = perf_stats['Ann Vol']
    kpi3.metric("Ann. Volatility", f"{ann_vol:.1f}%")

    mdd = perf_stats['MDD']
    mdd_color = "red" if mdd < -40 else ("orange" if mdd < -30 else "green")
    kpi4.metric("Max Drawdown", f"{mdd:.1f}%")

    # --- Row 2: Cumulative Wealth + Drawdown ---
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        cum = (1 + rets).cumprod()
        spy_cum = (1 + spy_oos).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name="Strategy", line=dict(color='#3b82f6', width=2)))
        fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum.values, name="SPY", line=dict(color='gray', width=2, dash='dash')))

        for i, pinned in enumerate(st.session_state.pinned):
            p_cum = (1 + pinned['result']['monthly_returns']).cumprod()
            fig.add_trace(go.Scatter(x=p_cum.index, y=p_cum.values, name=pinned['label'], line=dict(color=PIN_COLORS[i % len(PIN_COLORS)], width=1.5)))

        fig.update_layout(title="Cumulative Wealth", yaxis_title="Growth of $1", template="plotly_dark", height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        cum = (1 + rets).cumprod()
        dd = cum / cum.cummax() - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill='tozeroy', name="Drawdown", line=dict(color='#ef4444', width=1), fillcolor='rgba(239,68,68,0.3)'))

        for i, pinned in enumerate(st.session_state.pinned):
            p_cum = (1 + pinned['result']['monthly_returns']).cumprod()
            p_dd = p_cum / p_cum.cummax() - 1
            fig.add_trace(go.Scatter(x=p_dd.index, y=p_dd.values, name=pinned['label'], line=dict(color=PIN_COLORS[i % len(PIN_COLORS)], width=1)))

        fig.update_layout(title="Drawdown", yaxis_title="Drawdown %", yaxis_tickformat='.0%', template="plotly_dark", height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: Test — run a backtest and verify KPIs + charts appear**

Run `streamlit run app.py`, click "Run Backtest" with default params, verify:
- 4 KPI cards show reasonable values (SR ~1.28, MDD ~-27.6%)
- Cumulative wealth chart shows strategy vs SPY
- Drawdown chart shows underwater curve

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "Add KPI cards, cumulative wealth, and drawdown charts"
```

---

### Task 6: Add Portfolio Composition Charts

**Files:**
- Modify: `dashboard/app.py` (append after Row 2)

- [ ] **Step 1: Add sector allocation and holdings table**

Append to `app.py`:

```python
    # --- Row 3: Portfolio Composition ---
    st.markdown("---")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        sector_data = {}
        for m, held in result['holdings'].items():
            counts = held['sector'].value_counts(normalize=True)
            sector_data[m] = counts.to_dict()

        if sector_data:
            sector_df = pd.DataFrame(sector_data).T.fillna(0).sort_index()
            fig = go.Figure()
            for col in sector_df.columns:
                fig.add_trace(go.Scatter(
                    x=sector_df.index, y=sector_df[col], stackgroup='one',
                    name=col, mode='lines'
                ))
            fig.update_layout(title="Sector Allocation Over Time", yaxis_title="Weight", yaxis_tickformat='.0%', template="plotly_dark", height=400, margin=dict(t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

    with comp_col2:
        last_month = sorted(result['holdings'].keys())[-1]
        last_holdings = result['holdings'][last_month][['permno', 'sector', 'pred', 'y_raw']].copy()
        last_holdings.columns = ['Permno', 'Sector', 'Predicted', 'Actual Return']
        last_holdings['Predicted'] = last_holdings['Predicted'].round(4)
        last_holdings['Actual Return'] = (last_holdings['Actual Return'] * 100).round(2).astype(str) + '%'
        st.markdown(f"**Holdings — {last_month}**")
        st.dataframe(last_holdings.reset_index(drop=True), use_container_width=True, height=380)
```

- [ ] **Step 2: Test — verify sector stacked area and holdings table render**

Run the app, click "Run Backtest", scroll to Row 3. Verify:
- Stacked area chart shows sector weights summing to ~100%
- Holdings table shows 10 rows (K=10) with permno, sector, predicted, actual return

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "Add sector allocation chart and holdings table"
```

---

### Task 7: Add Risk & Signal Charts

**Files:**
- Modify: `dashboard/app.py` (append after Row 3)

- [ ] **Step 1: Add rolling Sharpe and IC charts**

Append to `app.py`:

```python
    # --- Row 4: Risk & Signal ---
    st.markdown("---")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        rolling_sr = rets.rolling(12, min_periods=6).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_sr.index, y=rolling_sr.values, name="Strategy", line=dict(color='#3b82f6', width=2)))

        for i, pinned in enumerate(st.session_state.pinned):
            p_sr = pinned['result']['monthly_returns'].rolling(12, min_periods=6).apply(
                lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
            )
            fig.add_trace(go.Scatter(x=p_sr.index, y=p_sr.values, name=pinned['label'], line=dict(color=PIN_COLORS[i % len(PIN_COLORS)], width=1.5)))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Rolling 12-Month Sharpe Ratio", yaxis_title="Sharpe Ratio", template="plotly_dark", height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with risk_col2:
        ic = result['ic'].dropna()
        ic_rolling = ic.rolling(12, min_periods=6).mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=ic.index, y=ic.values, name="Monthly IC", marker_color='rgba(59,130,246,0.4)'))
        fig.add_trace(go.Scatter(x=ic_rolling.index, y=ic_rolling.values, name="12m Rolling Mean", line=dict(color='#1e3a5f', width=2)))
        fig.add_hline(y=ic.mean(), line_dash="dash", line_color="red", annotation_text=f"Mean={ic.mean():.3f}")

        fig.update_layout(title="Information Coefficient", yaxis_title="Spearman IC", template="plotly_dark", height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: Test — verify rolling SR and IC charts render correctly**

Run the app, click "Run Backtest", scroll to Row 4. Verify:
- Rolling SR line chart oscillates around expected range
- IC bar chart with rolling mean overlay, red dashed mean line

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "Add rolling Sharpe ratio and IC charts"
```

---

### Task 8: Add Diagnostics and Comparison Table

**Files:**
- Modify: `dashboard/app.py` (append after Row 4)

- [ ] **Step 1: Add monthly return heatmap and turnover chart**

Append to `app.py`:

```python
    # --- Row 5: Deeper Diagnostics ---
    st.markdown("---")
    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        rets_df = rets.to_frame('ret')
        rets_df['year'] = rets_df.index.str[:4]
        rets_df['month'] = rets_df.index.str[5:7].astype(int)
        heatmap_pivot = rets_df.pivot_table(values='ret', index='year', columns='month', aggfunc='first')
        heatmap_pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(heatmap_pivot.columns)]

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values * 100,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale=[[0, '#ef4444'], [0.5, '#1f2937'], [1, '#10b981']],
            zmid=0,
            text=np.round(heatmap_pivot.values * 100, 1),
            texttemplate='%{text:.1f}%',
            textfont=dict(size=10),
            hovertemplate='%{y} %{x}: %{z:.1f}%<extra></extra>',
        ))
        fig.update_layout(title="Monthly Returns Heatmap (%)", template="plotly_dark", height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with diag_col2:
        to = result['turnover'].dropna()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=to.index, y=to.values, name="Monthly Turnover", marker_color='rgba(59,130,246,0.5)'))
        fig.add_hline(y=to.mean(), line_dash="dash", line_color="red", annotation_text=f"Mean={to.mean():.1%}")
        fig.update_layout(title="Monthly Turnover", yaxis_title="Turnover", yaxis_tickformat='.0%', template="plotly_dark", height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 6: Comparison Table (only if pinned configs exist) ---
    if st.session_state.pinned:
        st.markdown("---")
        st.markdown("### Strategy Comparison")

        comp_rows = []
        current_perf = compute_perf(rets, "Current")
        current_perf['Mean IC'] = round(result['ic'].mean(), 4)
        current_perf['Mean Turnover'] = f"{result['turnover'].mean():.1%}"
        current_perf['Ann Return'] = f"{current_perf['Ann Return']}%"
        current_perf['Ann Vol'] = f"{current_perf['Ann Vol']}%"
        current_perf['MDD'] = f"{current_perf['MDD']}%"
        current_perf['Total Return'] = f"{current_perf['Total Return']:.0f}%"
        comp_rows.append(current_perf)

        for pinned in st.session_state.pinned:
            p = compute_perf(pinned['result']['monthly_returns'], pinned['label'])
            p['Mean IC'] = round(pinned['result']['ic'].mean(), 4)
            p['Mean Turnover'] = f"{pinned['result']['turnover'].mean():.1%}"
            p['Ann Return'] = f"{p['Ann Return']}%"
            p['Ann Vol'] = f"{p['Ann Vol']}%"
            p['MDD'] = f"{p['MDD']}%"
            p['Total Return'] = f"{p['Total Return']:.0f}%"
            comp_rows.append(p)

        comp_df = pd.DataFrame(comp_rows).set_index('Strategy')
        st.dataframe(comp_df, use_container_width=True)

else:
    st.info("Click **Run Backtest** to see results.")
```

- [ ] **Step 2: End-to-end test**

Run `streamlit run app.py`. Full test sequence:

1. Click "Run Backtest" with defaults (HGB, K=10, vt=0.05, regime=6) → verify all 6 rows of charts render
2. Click "Pin Config" → verify chip appears below controls
3. Change K to 20, click "Run Backtest" again → verify charts update, pinned config stays overlaid
4. Click "Pin Config" again → verify second chip, comparison table appears at bottom
5. Click ❌ on a pinned chip → verify it disappears

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "Add heatmap, turnover chart, and comparison table"
```

---

### Task 9: Final Polish and `.gitignore`

**Files:**
- Modify: `.gitignore`
- Create: `dashboard/__init__.py` (empty, for clean imports)

- [ ] **Step 1: Create empty `__init__.py`**

```bash
touch dashboard/__init__.py
```

- [ ] **Step 2: Update `.gitignore`**

Ensure these entries exist:

```
dashboard/.cache/
.superpowers/
```

- [ ] **Step 3: Full end-to-end verification**

Run `streamlit run dashboard/app.py` from the project root. Verify:

1. App loads without errors
2. Default params show in controls
3. "Run Backtest" produces all charts
4. Second run with same params is instant (cache hit)
5. Changing K/vol_tilt/regime and re-running is fast (prediction cache hit, only portfolio rebuilt)
6. Changing model hyperparams triggers full rerun with progress bar
7. Pinning and unpinning works
8. Comparison table shows when configs are pinned
9. All Plotly charts are interactive (hover, zoom, pan)

- [ ] **Step 4: Final commit**

```bash
git add dashboard/__init__.py .gitignore
git commit -m "Add init file and update gitignore for dashboard"
```

---

## Self-Review Checklist

- **Spec coverage:** All spec sections mapped to tasks: features (T1), engine (T2), cache (T3), controls (T4), KPIs + core charts (T5), composition (T6), risk/signal (T7), diagnostics + comparison (T8), polish (T9). Pin/compare system in T4+T5-T8. v2 yfinance explicitly out of scope.
- **Placeholder scan:** All steps contain complete code. No TBD/TODO.
- **Type consistency:** `run_predictions()` returns `dict[str, DataFrame]`, `build_portfolio()` returns dict with `monthly_returns`, `holdings`, `ic`, `turnover` — used consistently in T4-T8. `compute_perf()` returns dict with `SR`, `Ann Return`, `Ann Vol`, `MDD`, `Total Return`, `Mean IC`, `Mean Turnover` — used in T5 and T8. `prediction_key()` and `portfolio_key()` return `str` — used consistently in T4.
