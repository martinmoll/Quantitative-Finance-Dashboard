# Dashboard Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reposition the repo as a general-purpose quant finance dashboard by removing assignment framing, gitignoring notebooks, and adding long-short strategy support.

**Architecture:** Surgical changes to three layers — repo housekeeping (README, gitignore, doc archival), engine logic (new `strategy_type` + `K_short` params in `build_portfolio()`), and UI (new controls in `app.py`). No directory restructuring, no new files.

**Tech Stack:** Python, Streamlit, pandas, numpy, scikit-learn, plotly

---

### Task 1: Gitignore Notebooks

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add notebook pattern to .gitignore**

Add `Code/*.ipynb` to `.gitignore` after the existing entries:

```
# Jupyter notebooks (kept locally, not tracked)
Code/*.ipynb
```

- [ ] **Step 2: Remove notebooks from git tracking (keep files on disk)**

Run:
```bash
git rm --cached "Code/MartinMoellenhus_Assignment2.ipynb" "Code/html_transformation (2).ipynb" "Code/testingModel.ipynb"
```

Expected: files removed from index, still present on disk.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "Stop tracking notebooks, add to gitignore"
```

---

### Task 2: Archive Assignment-Specific Docs

**Files:**
- Move to `docs/archive/`: `progress.md`, `mdd_reduction_log.md`, `karpathy_guidelines_summary.md`, plus old specs and plans
- Modify: `learning/module4_overview.md` (also archive — assignment module notes)

- [ ] **Step 1: Create archive directory and move files**

Run:
```bash
mkdir -p docs/archive
git mv progress.md docs/archive/progress.md
git mv karpathy_guidelines_summary.md docs/archive/karpathy_guidelines_summary.md
git mv docs/mdd_reduction_log.md docs/archive/mdd_reduction_log.md
git mv docs/superpowers/specs/2026-04-25-alpha-strategy-design.md docs/archive/2026-04-25-alpha-strategy-design.md
git mv docs/superpowers/specs/2026-05-09-bloomberg-dashboard-design.md docs/archive/2026-05-09-bloomberg-dashboard-design.md
git mv docs/superpowers/plans/2026-04-25-alpha-strategy-plan.md docs/archive/2026-04-25-alpha-strategy-plan.md
git mv docs/superpowers/plans/2026-05-09-bloomberg-dashboard-plan.md docs/archive/2026-05-09-bloomberg-dashboard-plan.md
```

- [ ] **Step 2: Commit**

```bash
git commit -m "Archive assignment-specific docs"
```

---

### Task 3: Rewrite README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md with new content**

Full replacement. Keeps the technical substance (dataset, features, models, walk-forward design, experiment results) but removes all assignment framing (RMBI3110, HKUST, grading rubric, instructor, "submission strategy"). New structure:

```markdown
# Quantitative Finance Dashboard

A Streamlit-based interactive backtesting dashboard for ML-driven equity strategies.

## Overview

An ML pipeline that predicts cross-sectional stock returns and constructs portfolios that beat the S&P 500 benchmark on a risk-adjusted basis. Supports:

- **Long-only** or **long-short** strategies
- **Equal-weight** portfolio construction with configurable concentration (K stocks)
- **Walk-forward backtesting** with expanding training windows
- **Volatility tilt** to control drawdown risk
- **Momentum regime filter** to reduce bear-market exposure
- **Pin & compare** multiple strategy configurations side-by-side

## Quick Start

```bash
pip install streamlit plotly pandas numpy scipy scikit-learn
cd dashboard
streamlit run app.py
```

## Dataset

| Property | Value |
|---|---|
| Rows | 257,800 |
| Columns | 229 |
| Months | 407 |
| Unique stocks | 1,216 |

116 raw features across 10 categories (price/momentum, accounting/value, analyst, technical, options, peer/industry, quarterly fundamentals, interactions, sector/macro). Each raw feature has a `_xs` (cross-sectional standardized) version computed per month.

**Targets:**
- `y_xs` — cross-sectionally standardized forward monthly return (training target)
- `y_raw` — raw forward monthly return (portfolio evaluation)

## Feature Engineering

Two feature tiers:

- **Tier 1 (~52 features)** — Core factors + engineered features, used by Lasso
- **Tier 2 (~118 features)** — All `_xs` columns + engineered features, used by HGB

Engineered features include interaction terms (9), non-linear transforms (5), composite signals (6), relative signals (3), and composite interactions (3). All are pure arithmetic on per-month standardized values — no look-ahead risk.

## Models

| Model | Features | Key Settings |
|---|---|---|
| HistGradientBoosting | Tier 2 (118) | max_depth=2, min_samples_leaf=500, max_iter=500, l2_reg=0.1 |
| Lasso | Tier 1 (52) | LassoCV with 5-fold CV for alpha selection |

Conservative regularization throughout — overfitting is the dominant failure mode in financial prediction.

## Walk-Forward Backtesting

Expanding-window walk-forward with 12-month retraining:

1. Train on all data before the current OOS window
2. Generate predictions for all stocks in each OOS month
3. Apply volatility tilt (optional) to penalise high-vol stocks
4. Select top K stocks → long leg (and bottom K_short → short leg, if long-short)
5. Portfolio return = equal-weighted mean of selected stocks' forward returns

## Strategy Modes

**Long-only:** Portfolio return = mean return of top K stocks.

**Long-short:** Portfolio return = mean(top K returns) − mean(bottom K_short returns). Dollar-neutral when K = K_short.

## Dashboard Controls

| Control | Range | Default |
|---|---|---|
| Model | HGB / Lasso | HGB |
| Strategy | Long Only / Long-Short | Long Only |
| K (long leg) | 5–50 | 10 |
| K_short (short leg) | 5–50 | 10 |
| Vol tilt | 0.00–0.50 | 0.05 |
| Regime lookback | 0–12 months | 6 |

Plus expandable model hyperparameters (max_depth, learning_rate, etc.) and display controls (start date, starting value, cash flow per period).

## Repository Structure

```
dashboard/
  app.py              # Streamlit entry point
  engine.py           # Walk-forward prediction engine + portfolio construction
  features.py         # Feature engineering (Tier 1 and Tier 2)
  cache_manager.py    # Two-tier disk caching
Data/
  alpha_dataset_v2.csv  # Dataset (not tracked)
Code/
  *.ipynb               # Notebooks (not tracked)
docs/
  archive/              # Historical experiment logs
  superpowers/          # Design specs and plans
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Rewrite README as quant finance dashboard (remove assignment framing)"
```

---

### Task 4: Add Long-Short to `cache_manager.py`

**Files:**
- Modify: `dashboard/cache_manager.py:30-36`

- [ ] **Step 1: Update `portfolio_key()` to include strategy params**

Replace the current `portfolio_key` function:

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

- [ ] **Step 2: Commit**

```bash
git add dashboard/cache_manager.py
git commit -m "Add strategy_type and K_short to portfolio cache key"
```

---

### Task 5: Add Long-Short to `engine.py`

**Files:**
- Modify: `dashboard/engine.py:138-240`

- [ ] **Step 1: Update `build_portfolio()` signature**

Change the function signature at line 138 to add `strategy_type` and `K_short`:

```python
def build_portfolio(
    predictions: dict,
    K: int = 10,
    vol_tilt: float = 0.0,
    regime_lookback: int = 6,
    market_monthly: pd.DataFrame | None = None,
    strategy_type: str = "long_only",
    K_short: int = 10,
) -> dict:
```

Update the docstring Parameters section to add:

```
    strategy_type : str
        "long_only" (default) selects top K stocks.
        "long_short" selects top K as long leg and bottom K_short as short leg.
    K_short : int
        Number of stocks in the short leg (only used when strategy_type="long_short").
```

Update the Returns docstring to note:

```
        holdings        : dict        — {month: DataFrame of held stocks}
                                        Long-short mode adds a 'side' column ("long"/"short").
```

- [ ] **Step 2: Update stock selection and return logic inside the month loop**

Find the block (around lines 212-220) that currently does:

```python
        # ---- Select top-K ----
        top = df_m.nlargest(K, 'pred')
        port_ret = top[EVAL_TARGET].mean()

        # ---- Regime filter — go to cash when trailing SPY is negative ----
        if regime_on is not None and m in regime_on.index and not regime_on[m]:
            port_ret = 0.0

        monthly_returns[m] = port_ret
        holdings[m] = top
```

Replace with:

```python
        # ---- Select stocks ----
        top = df_m.nlargest(K, 'pred')

        if strategy_type == "long_short":
            bottom = df_m.nsmallest(K_short, 'pred')
            port_ret = top[EVAL_TARGET].mean() - bottom[EVAL_TARGET].mean()
            top = top.copy()
            bottom = bottom.copy()
            top['side'] = 'long'
            bottom['side'] = 'short'
            held = pd.concat([top, bottom], ignore_index=True)
        else:
            port_ret = top[EVAL_TARGET].mean()
            held = top

        # ---- Regime filter — go to cash when trailing SPY is negative ----
        if regime_on is not None and m in regime_on.index and not regime_on[m]:
            port_ret = 0.0

        monthly_returns[m] = port_ret
        holdings[m] = held
```

- [ ] **Step 3: Update turnover tracking for both legs**

Find the turnover block (around lines 224-230):

```python
        curr_weights = pd.Series(0.0, index=df_m['permno'].values)
        curr_weights[top['permno'].values] = 1.0 / K

        if prev_weights is not None:
            aligned_curr, aligned_prev = curr_weights.align(prev_weights, fill_value=0.0)
            turnover_vals[m] = 0.5 * (aligned_curr - aligned_prev).abs().sum()
        else:
            turnover_vals[m] = np.nan

        prev_weights = curr_weights
```

Replace with:

```python
        curr_weights = pd.Series(0.0, index=df_m['permno'].values)
        curr_weights[top['permno'].values] = 1.0 / K
        if strategy_type == "long_short":
            curr_weights[bottom['permno'].values] = -1.0 / K_short

        if prev_weights is not None:
            aligned_curr, aligned_prev = curr_weights.align(prev_weights, fill_value=0.0)
            turnover_vals[m] = 0.5 * (aligned_curr - aligned_prev).abs().sum()
        else:
            turnover_vals[m] = np.nan

        prev_weights = curr_weights
```

- [ ] **Step 4: Update the minimum stocks guard**

Find line ~200:

```python
        if len(df_m) < 2 * K:
```

Replace with:

```python
        min_required = K + K_short if strategy_type == "long_short" else 2 * K
        if len(df_m) < min_required:
```

- [ ] **Step 5: Verify by running the dashboard**

Run:
```bash
cd dashboard
streamlit run app.py
```

Open in browser, run a backtest with default (Long Only) settings. Confirm results match previous behavior (same SR, MDD, etc. for default params).

- [ ] **Step 6: Commit**

```bash
git add dashboard/engine.py
git commit -m "Add long-short strategy support to build_portfolio()"
```

---

### Task 6: Add Long-Short UI to `app.py`

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Add Strategy dropdown and K_short slider to top control bar**

Find the top control bar (lines 31-43):

```python
col1, col2, col3, col4 = st.columns(4)

with col1:
    model_type = st.selectbox("Model", ["HGB", "Lasso"])

with col2:
    K = st.slider("K (stocks)", min_value=5, max_value=50, step=5, value=10)

with col3:
    vol_tilt = st.slider("Vol tilt", min_value=0.0, max_value=0.50, step=0.01, value=0.05)

with col4:
    regime_lookback = st.slider("Regime lookback", min_value=0, max_value=12, value=6)
```

Replace with:

```python
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    model_type = st.selectbox("Model", ["HGB", "Lasso"])

with col2:
    strategy_type = st.selectbox("Strategy", ["Long Only", "Long-Short"])

with col3:
    K = st.slider("K (stocks)", min_value=5, max_value=50, step=5, value=10)

with col4:
    vol_tilt = st.slider("Vol tilt", min_value=0.0, max_value=0.50, step=0.01, value=0.05)

with col5:
    regime_lookback = st.slider("Regime lookback", min_value=0, max_value=12, value=6)

strategy_key = "long_short" if strategy_type == "Long-Short" else "long_only"

if strategy_key == "long_short":
    K_short = st.slider("K short (stocks)", min_value=5, max_value=50, step=5, value=K)
else:
    K_short = K
```

- [ ] **Step 2: Update portfolio cache key call**

Find (line ~202):

```python
    port_key = cache.portfolio_key(pred_key, K, vol_tilt, regime_lookback)
```

Replace with:

```python
    port_key = cache.portfolio_key(pred_key, K, vol_tilt, regime_lookback, strategy_key, K_short)
```

- [ ] **Step 3: Update `build_portfolio()` call**

Find (line ~205):

```python
    if portfolio is None:
        portfolio = build_portfolio(
            predictions,
            K=K,
            vol_tilt=vol_tilt,
            regime_lookback=regime_lookback,
            market_monthly=market_monthly,
        )
```

Replace with:

```python
    if portfolio is None:
        portfolio = build_portfolio(
            predictions,
            K=K,
            vol_tilt=vol_tilt,
            regime_lookback=regime_lookback,
            market_monthly=market_monthly,
            strategy_type=strategy_key,
            K_short=K_short,
        )
```

- [ ] **Step 4: Update session state params to include strategy**

Find (line ~217):

```python
    st.session_state.current_params = {
        "model_type": model_type,
        "model_params": model_params,
        "retrain_every": retrain_every,
        "K": K,
        "vol_tilt": vol_tilt,
        "regime_lookback": regime_lookback,
    }
```

Replace with:

```python
    st.session_state.current_params = {
        "model_type": model_type,
        "model_params": model_params,
        "retrain_every": retrain_every,
        "K": K,
        "vol_tilt": vol_tilt,
        "regime_lookback": regime_lookback,
        "strategy_type": strategy_key,
        "K_short": K_short,
    }
```

- [ ] **Step 5: Update pin label to show strategy**

Find (line ~237):

```python
        mt = params["model_type"]
        k_val = params["K"]
        vt_val = params["vol_tilt"]
        label = f"{mt} K={k_val} vt={vt_val}"
```

Replace with:

```python
        mt = params["model_type"]
        k_val = params["K"]
        vt_val = params["vol_tilt"]
        strat = params.get("strategy_type", "long_only")
        if strat == "long_short":
            ks_val = params.get("K_short", k_val)
            label = f"{mt} L/S K={k_val}/{ks_val} vt={vt_val}"
        else:
            label = f"{mt} K={k_val} vt={vt_val}"
```

- [ ] **Step 6: Update holdings table to show Side column**

Find the holdings table block (lines 368-387). Replace the display column construction:

```python
        last_month = sorted(holdings_filtered.keys())[-1]
        last_h = holdings_filtered[last_month].copy()
        display_cols = []
        if 'permno' in last_h.columns:
            display_cols.append('permno')
        if 'sector' in last_h.columns:
            display_cols.append('sector')
        if 'pred' in last_h.columns:
            display_cols.append('pred')
        if 'y_raw' in last_h.columns:
            display_cols.append('y_raw')
        show_df = last_h[display_cols].copy()
        show_df.columns = ['Permno', 'Sector', 'Predicted', 'Actual Return'][:len(display_cols)]
        if 'Predicted' in show_df.columns:
            show_df['Predicted'] = show_df['Predicted'].round(4)
        if 'Actual Return' in show_df.columns:
            show_df['Actual Return'] = (show_df['Actual Return'] * 100).round(2).astype(str) + '%'
        st.markdown(f"**Holdings — {last_month}**")
        st.dataframe(show_df.reset_index(drop=True), width='stretch', height=380)
```

With:

```python
        last_month = sorted(holdings_filtered.keys())[-1]
        last_h = holdings_filtered[last_month].copy()
        display_cols = []
        col_labels = []
        if 'side' in last_h.columns:
            display_cols.append('side')
            col_labels.append('Side')
        if 'permno' in last_h.columns:
            display_cols.append('permno')
            col_labels.append('Permno')
        if 'sector' in last_h.columns:
            display_cols.append('sector')
            col_labels.append('Sector')
        if 'pred' in last_h.columns:
            display_cols.append('pred')
            col_labels.append('Predicted')
        if 'y_raw' in last_h.columns:
            display_cols.append('y_raw')
            col_labels.append('Actual Return')
        show_df = last_h[display_cols].copy()
        show_df.columns = col_labels
        if 'Predicted' in show_df.columns:
            show_df['Predicted'] = show_df['Predicted'].round(4)
        if 'Actual Return' in show_df.columns:
            show_df['Actual Return'] = (show_df['Actual Return'] * 100).round(2).astype(str) + '%'
        st.markdown(f"**Holdings — {last_month}**")
        st.dataframe(show_df.reset_index(drop=True), width='stretch', height=380)
```

- [ ] **Step 7: Verify long-short in browser**

Run:
```bash
cd dashboard
streamlit run app.py
```

Test:
1. Select "Long-Short" strategy, set K=10, K_short=10
2. Click "Run Backtest"
3. Verify: wealth chart shows a different curve than Long Only (likely smoother since market-neutral)
4. Verify: holdings table shows "Side" column with "long" and "short" entries
5. Verify: KPIs display correctly
6. Switch back to "Long Only", run again — confirm previous behavior is intact

- [ ] **Step 8: Commit**

```bash
git add dashboard/app.py
git commit -m "Add long-short strategy UI controls and display"
```

---

### Task 7: Update Memory Files

**Files:**
- Modify: memory files (outside repo)

- [ ] **Step 1: Remove assignment-specific memories**

Delete these memory files (no longer relevant now that the project is a general dashboard):
- `project_alpha_challenge.md` (assignment-specific)
- `project_grading_rubric.md` (grading rubric)

Update `project_dashboard_idea.md` to reflect current state (dashboard is built, not a future idea).

- [ ] **Step 2: Update MEMORY.md index**

Remove the deleted entries and update the dashboard entry.

- [ ] **Step 3: No commit needed** (memory files are outside the repo)

---

### Task 8: Final Verification

- [ ] **Step 1: Run git status and verify clean state**

```bash
git status
git log --oneline -10
```

Verify: all changes committed, no untracked files except notebooks (now gitignored) and cache.

- [ ] **Step 2: Run dashboard end-to-end**

```bash
cd dashboard
streamlit run app.py
```

Test both strategies:
1. Long Only: HGB, K=10, vt=0.05, regime=6 → should match historical results (~SR 1.48)
2. Long-Short: HGB, K=10, K_short=10, vt=0.05, regime=6 → should run without errors, show both legs in holdings

- [ ] **Step 3: Commit any final fixes if needed**
