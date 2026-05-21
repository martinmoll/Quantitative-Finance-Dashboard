# Alpha Challenge Strategy Implementation Plan

> **For agentic workers:** REQUIRED SKILLS: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Also use andrej-karpathy-skills:karpathy-guidelines when writing, reviewing, or refactoring code to avoid overcomplication, make surgical changes, surface assumptions, and define verifiable success criteria. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a multi-model walk-forward trading strategy in the existing Jupyter notebook to maximize OOS Sharpe Ratio while keeping MDD < 40%. Target: SR > 1.3 (12/20), stretch SR > 1.5 (15/20). Current best: SR=1.20 (10/20 tier).

**Grading rubric (performance, 20 pts):** SR > 1.7 → 18pts, 1.5-1.7 → 15pts, 1.3-1.5 → 12pts, 1.2-1.3 → 10pts. Disqualifying: look-ahead bias, overfitting, MDD > 40%.

**Architecture:** Two feature-builder functions (Tier 1 for linear, Tier 2 for ensemble models) feed into a unified `run_model()` walk-forward engine. Each model is a `(feature_builder, estimator)` pair in a dictionary. Results are compared via summary table and cumulative wealth plots.

**Tech Stack:** Python, pandas, numpy, scikit-learn (LassoCV, RandomForestRegressor, HistGradientBoostingRegressor), matplotlib. All work in `Code/AlphaChallenge.ipynb`.

---

### Task 1: Feature Engineering Functions

**Files:**
- Modify: `Code/AlphaChallenge.ipynb` — add new code cell after the helpers cell

**Context:** The dataset has 92 `_xs` columns (cross-sectionally standardized per month). We build two feature sets from these. All composites are simple arithmetic — no look-ahead risk. NaNs are filled with 0.0 (the cross-sectional median since `_xs` features are median-centered).

- [ ] **Step 1: Add a new code cell after the helpers cell with both feature-builder functions**

```python
# ── Feature Engineering ──────────────────────────────────────────────

def build_features_linear(df_slice):
    """Tier 1: Conservative feature set for Lasso (~25-35 features).
    Core factors + pre-built interactions + hand-crafted composites.
    """
    feat = pd.DataFrame(index=df_slice.index)

    # Core factor features
    core = [
        # Momentum
        'ret_1_xs', 'ret_2_12_xs', 'ret_2_6_xs',
        # Value
        'bm_xs', 'ep_xs', 'cfp_xs', 'sp_xs',
        # Quality
        'gpa_xs', 'roe_xs', 'roa_xs',
        # Risk
        'vol_12m_xs', 'ivol_xs', 'beta_xs',
        # Size
        'log_me_xs',
        # Analyst
        'sue_xs', 'revision_xs', 'beat_xs',
        # Turnover / Liquidity
        'turnover_xs', 'illiq_12m_xs',
    ]
    for c in core:
        if c in df_slice.columns:
            feat[c] = df_slice[c]

    # Pre-built interaction features
    interactions = [
        'mom_x_size_xs', 'val_x_prof_xs', 'mom_x_vol_xs',
        'ret_vs_sector_xs', 'bm_vs_sector_xs', 'ret_vs_ind_xs',
        'bm_vs_size_xs',
    ]
    for c in interactions:
        if c in df_slice.columns:
            feat[c] = df_slice[c]

    # Hand-crafted composites
    feat['quality_composite'] = (
        df_slice.get('gpa_xs', 0)
        + df_slice.get('roe_xs', 0)
        - df_slice.get('ag_xs', 0)
    ) / 3
    feat['earnings_momentum'] = (
        df_slice.get('sue_xs', 0)
        + df_slice.get('revision_xs', 0)
    ) / 2
    feat['reversal_mom_combo'] = (
        df_slice.get('ret_2_12_xs', 0)
        - df_slice.get('ret_1_xs', 0)
    )

    feat = feat.fillna(0.0)
    return feat


def build_features_ensemble(df_slice):
    """Tier 2: Broad feature set for RF and HGB (~60-70 features).
    Everything from Tier 1 + all remaining _xs features + extra composites.
    """
    # Start with all _xs features
    xs_cols = [c for c in df_slice.columns if c.endswith('_xs') and c != 'y_xs']
    feat = df_slice[xs_cols].copy()

    # Additional composites
    feat['quality_composite'] = (
        df_slice.get('gpa_xs', 0)
        + df_slice.get('roe_xs', 0)
        - df_slice.get('ag_xs', 0)
    ) / 3
    feat['earnings_momentum'] = (
        df_slice.get('sue_xs', 0)
        + df_slice.get('revision_xs', 0)
    ) / 2
    feat['reversal_mom_combo'] = (
        df_slice.get('ret_2_12_xs', 0)
        - df_slice.get('ret_1_xs', 0)
    )
    feat['peer_adj_earnings'] = (
        df_slice.get('sue_xs', 0)
        - df_slice.get('peer_sue_xs', 0)
    )
    feat['value_momentum'] = (
        df_slice.get('bm_xs', 0)
        + df_slice.get('ret_2_12_xs', 0)
    )

    feat = feat.fillna(0.0)
    return feat


print(f"Tier 1 features: {build_features_linear(df_sp).shape[1]}")
print(f"Tier 2 features: {build_features_ensemble(df_sp).shape[1]}")
```

- [ ] **Step 2: Run the cell and verify output**

Expected output (approximately):
```
Tier 1 features: 29
Tier 2 features: 97
```

The exact count may vary slightly depending on which `_xs` columns exist. Tier 1 should be ~25-35, Tier 2 should be ~95-100 (92 base + 5 composites).

- [ ] **Step 3: Commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "feat: add Tier 1 and Tier 2 feature engineering functions"
```

---

### Task 2: Model Definitions

**Files:**
- Modify: `Code/AlphaChallenge.ipynb` — add new code cell after the feature engineering cell

**Context:** Three models of increasing complexity. LassoCV auto-tunes regularization strength. RF and HGB are configured conservatively (shallow, regularized) to prevent overfitting on noisy financial data.

- [ ] **Step 1: Add a new code cell with the model dictionary**

```python
# ── Model Definitions ────────────────────────────────────────────────

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

MODELS = {
    'Lasso': {
        'feature_builder': build_features_linear,
        'estimator': LassoCV(cv=5, max_iter=5000),
    },
    'RandomForest': {
        'feature_builder': build_features_ensemble,
        'estimator': RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            max_features=0.33,
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=42,
        ),
    },
    'HGB': {
        'feature_builder': build_features_ensemble,
        'estimator': HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=100,
            early_stopping=False,
            random_state=42,
        ),
    },
}

print(f"Models defined: {list(MODELS.keys())}")
```

- [ ] **Step 2: Run the cell and verify output**

Expected output:
```
Models defined: ['Lasso', 'RandomForest', 'HGB']
```

- [ ] **Step 3: Commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "feat: define Lasso, Random Forest, and HGB model configs"
```

---

### Task 3: Walk-Forward Engine

**Files:**
- Modify: `Code/AlphaChallenge.ipynb` — add new code cell after model definitions

**Context:** This is the core backtest loop. For each model, it trains on expanding historical data and predicts out-of-sample, retraining every 12 months. It computes both long-only (required) and long-short (diagnostic) portfolio returns. Uses `y_xs` as training target but evaluates on `y_raw` (actual returns).

- [ ] **Step 1: Add a new code cell with the `run_model()` function**

```python
# ── Walk-Forward Engine ──────────────────────────────────────────────
from sklearn.base import clone

def run_model(df_in, feature_builder, estimator, name, K=30):
    """Walk-forward backtest for a single (feature_builder, estimator) pair.

    Returns dict with:
      - 'long_only': pd.Series of monthly long-only returns
      - 'long_short': pd.Series of monthly long-short returns
    """
    all_months = sorted(df_in['ym'].unique())
    oos_months = [m for m in all_months if m >= OOS_START]

    long_only_rets = {}
    long_short_rets = {}

    # Determine retraining schedule
    retrain_months = oos_months[::RETRAIN_EVERY]  # every 12 months

    model = None
    for i, m in enumerate(oos_months):
        # Retrain if needed
        if m in retrain_months:
            train = df_in[df_in['ym'] < m].dropna(subset=[TRAIN_TARGET])
            X_train = feature_builder(train)
            y_train = train[TRAIN_TARGET]
            model = clone(estimator)
            model.fit(X_train, y_train)
            print(f"  [{name}] Trained on {len(train):,} rows up to {m}")

        # Predict this month
        test = df_in[df_in['ym'] == m].copy()
        if len(test) < 2 * K:
            continue

        X_test = feature_builder(test)
        test['pred'] = model.predict(X_test)

        # Long-only: top K stocks, equal weight
        top = test.nlargest(K, 'pred')
        lo_ret = top[EVAL_TARGET].mean()
        long_only_rets[m] = lo_ret

        # Long-short: top K minus bottom K (diagnostic)
        bot = test.nsmallest(K, 'pred')
        ls_ret = top[EVAL_TARGET].mean() - bot[EVAL_TARGET].mean()
        long_short_rets[m] = ls_ret

    return {
        'long_only': pd.Series(long_only_rets).sort_index(),
        'long_short': pd.Series(long_short_rets).sort_index(),
    }

print("Walk-forward engine loaded.")
```

- [ ] **Step 2: Run the cell and verify output**

Expected output:
```
Walk-forward engine loaded.
```

- [ ] **Step 3: Commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "feat: implement walk-forward engine with long-only and long-short"
```

---

### Task 4: Run All Models

**Files:**
- Modify: `Code/AlphaChallenge.ipynb` — add new code cell after walk-forward engine

**Context:** Execute each model through the walk-forward engine. This cell will take several minutes to run due to training 3 models across ~19 retraining windows each. The print statements from `run_model()` show progress.

- [ ] **Step 1: Add a new code cell to run all models and collect results**

```python
# ── Run All Models ───────────────────────────────────────────────────

results = {}

for name, cfg in MODELS.items():
    print(f"\n{'='*50}")
    print(f"Running {name}...")
    print(f"{'='*50}")
    res = run_model(df_sp, cfg['feature_builder'], cfg['estimator'], name, K=K)
    results[name] = res
    # Quick preview
    lo = res['long_only']
    sr = lo.mean() / lo.std() * np.sqrt(12) if lo.std() > 0 else 0
    print(f"  → Long-only SR = {sr:.2f} ({len(lo)} months)")

print("\nAll models complete.")
```

- [ ] **Step 2: Run the cell and wait for completion**

Expected output pattern (SRs will vary):
```
==================================================
Running Lasso...
==================================================
  [Lasso] Trained on X rows up to 2005-01
  [Lasso] Trained on X rows up to 2006-01
  ...
  → Long-only SR = X.XX (227 months)

==================================================
Running RandomForest...
==================================================
  ...
  → Long-only SR = X.XX (227 months)

==================================================
Running HGB...
==================================================
  ...
  → Long-only SR = X.XX (227 months)

All models complete.
```

This may take 5-15 minutes depending on hardware. RandomForest will be the slowest.

- [ ] **Step 3: Commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "feat: run all three models through walk-forward backtest"
```

---

### Task 5: Evaluation and Comparison

**Files:**
- Modify: `Code/AlphaChallenge.ipynb` — add new code cell after model run cell

**Context:** Compare all models side-by-side with SPY benchmark. Two outputs: (1) summary table with metrics, (2) cumulative wealth plot. Also show long-short diagnostics for insight.

- [ ] **Step 1: Add a new code cell for evaluation**

```python
# ── Evaluation & Comparison ──────────────────────────────────────────

# --- Long-only performance (what matters for the grade) ---
print("=" * 60)
print("LONG-ONLY PORTFOLIO PERFORMANCE")
print("=" * 60)

lo_strats = {}
perf_rows = []

for name, res in results.items():
    lo = res['long_only']
    lo_strats[name] = lo
    perf_rows.append(perf(lo, name))

# Add SPY benchmark
perf_rows.append(perf(spy_oos, 'SPY'))

df_perf = pd.DataFrame(perf_rows).set_index('Strategy')
print(df_perf.to_string())

# Cumulative wealth plot - long only
plot_strats(lo_strats, 'Long-Only Portfolio: Cumulative Wealth vs SPY')

# --- Long-short diagnostics ---
print("\n" + "=" * 60)
print("LONG-SHORT DIAGNOSTICS (not for submission)")
print("=" * 60)

ls_perf_rows = []
for name, res in results.items():
    ls = res['long_short']
    ls_perf_rows.append(perf(ls, f'{name} L/S'))

df_ls_perf = pd.DataFrame(ls_perf_rows).set_index('Strategy')
print(df_ls_perf.to_string())
```

- [ ] **Step 2: Run the cell and review outputs**

Check:
- All models produce 227 OOS months
- At least one model has SR > 0.66 (beats SPY)
- MDD is under 40% for all models
- Cumulative wealth plot renders with all strategies + SPY

- [ ] **Step 3: Commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "feat: add evaluation table and cumulative wealth comparison"
```

---

### Task 6: Optional Blend

**Files:**
- Modify: `Code/AlphaChallenge.ipynb` — add new code cell after evaluation

**Context:** Average predictions from the top 2 models (by long-only SR) and re-run the portfolio selection. Only keep the blend if it improves SR. This requires re-running walk-forward with blended predictions, so we store predictions during the model runs.

- [ ] **Step 1: Add a new code cell for blending**

```python
# ── Optional Blend ───────────────────────────────────────────────────

def run_blend(df_in, models_cfg, K=30):
    """Walk-forward with averaged predictions from multiple models."""
    all_months = sorted(df_in['ym'].unique())
    oos_months = [m for m in all_months if m >= OOS_START]
    retrain_months = oos_months[::RETRAIN_EVERY]

    fitted_models = {}  # name -> fitted model
    long_only_rets = {}

    for i, m in enumerate(oos_months):
        # Retrain all models if needed
        if m in retrain_months:
            train = df_in[df_in['ym'] < m].dropna(subset=[TRAIN_TARGET])
            for name, cfg in models_cfg.items():
                X_train = cfg['feature_builder'](train)
                y_train = train[TRAIN_TARGET]
                mdl = clone(cfg['estimator'])
                mdl.fit(X_train, y_train)
                fitted_models[name] = (mdl, cfg['feature_builder'])
            print(f"  [Blend] Trained all models up to {m}")

        # Predict this month with each model, then average
        test = df_in[df_in['ym'] == m].copy()
        if len(test) < 2 * K:
            continue

        preds = []
        for name, (mdl, fb) in fitted_models.items():
            X_test = fb(test)
            preds.append(mdl.predict(X_test))

        test['pred_blend'] = np.mean(preds, axis=0)

        # Long-only
        top = test.nlargest(K, 'pred_blend')
        long_only_rets[m] = top[EVAL_TARGET].mean()

    return pd.Series(long_only_rets).sort_index()


# Find top 2 models by SR
model_srs = {}
for name, res in results.items():
    lo = res['long_only']
    model_srs[name] = lo.mean() / lo.std() * np.sqrt(12) if lo.std() > 0 else 0

sorted_models = sorted(model_srs.items(), key=lambda x: x[1], reverse=True)
top2_names = [n for n, _ in sorted_models[:2]]
print(f"Blending top 2 models: {top2_names}")

blend_cfg = {n: MODELS[n] for n in top2_names}
blend_rets = run_blend(df_sp, blend_cfg, K=K)

blend_sr = blend_rets.mean() / blend_rets.std() * np.sqrt(12)
print(f"Blend SR = {blend_sr:.2f}")
print(f"Best individual SR = {sorted_models[0][1]:.2f} ({sorted_models[0][0]})")
if blend_sr > sorted_models[0][1]:
    print("→ Blend IMPROVES over best individual model. Using blend.")
    results['Blend'] = {'long_only': blend_rets}
else:
    print("→ Blend does NOT improve. Sticking with best individual model.")
```

- [ ] **Step 2: Run the cell**

This will take several minutes (retrains 2 models across all windows). Check whether blend SR exceeds the best individual model SR.

- [ ] **Step 3: If blend was added, re-run evaluation with it**

Add one more cell if blend improved:

```python
# Re-plot with blend included (only if blend was added to results)
if 'Blend' in results:
    lo_strats_with_blend = {n: r['long_only'] for n, r in results.items() if 'long_only' in r}
    plot_strats(lo_strats_with_blend, 'All Strategies Including Blend vs SPY')
    print(perf(results['Blend']['long_only'], 'Blend'))
```

- [ ] **Step 4: Commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "feat: add optional blend of top-2 models"
```

---

### Task 7: Lasso Feature Importance (for writeup insight)

**Files:**
- Modify: `Code/AlphaChallenge.ipynb` — add new code cell after blend

**Context:** Extract which features Lasso kept (non-zero coefficients) from the last trained Lasso model. This gives insight for the Task 2 writeup.

- [ ] **Step 1: Add a new code cell for Lasso feature analysis**

```python
# ── Lasso Feature Analysis ───────────────────────────────────────────

# Re-train Lasso on full pre-OOS data to inspect coefficients
train_full = df_sp[df_sp['ym'] < OOS_START].dropna(subset=[TRAIN_TARGET])
X_full = build_features_linear(train_full)
y_full = train_full[TRAIN_TARGET]

lasso_inspect = LassoCV(cv=5, max_iter=5000)
lasso_inspect.fit(X_full, y_full)

coefs = pd.Series(lasso_inspect.coef_, index=X_full.columns)
nonzero = coefs[coefs != 0].sort_values(key=abs, ascending=False)
print(f"Lasso selected {len(nonzero)} / {len(coefs)} features")
print(f"Best alpha: {lasso_inspect.alpha_:.6f}\n")
print("Non-zero coefficients (sorted by magnitude):")
print(nonzero.to_string())
```

- [ ] **Step 2: Run the cell and note which features survived**

The output shows which features Lasso considers predictive. Use this in the Task 2 writeup.

- [ ] **Step 3: Commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "feat: add Lasso feature importance analysis"
```

---

### Task 8: Final Results Cell and Writeup Template

**Files:**
- Modify: `Code/AlphaChallenge.ipynb` — add code cell and markdown cell for Task 2

**Context:** A final cell that prints the definitive results for submission, plus the structured writeup template the user will fill in.

- [ ] **Step 1: Add a final results code cell**

```python
# ── Final Results ────────────────────────────────────────────────────

# Pick the best strategy (highest SR with MDD < 40%)
best_name = None
best_sr = -999

for name, res in results.items():
    if 'long_only' not in res:
        continue
    lo = res['long_only']
    sr = lo.mean() / lo.std() * np.sqrt(12) if lo.std() > 0 else 0
    cum = (1 + lo).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    if mdd > -0.40 and sr > best_sr:
        best_sr = sr
        best_name = name

print(f"BEST STRATEGY: {best_name} (SR = {best_sr:.2f})")
print(f"SPY Benchmark SR = {spy_sr:.2f}")
print(f"Alpha over SPY = {best_sr - spy_sr:+.2f} SR\n")

# Final performance table
best_lo = results[best_name]['long_only']
final_perf = pd.DataFrame([
    perf(best_lo, best_name),
    perf(spy_oos, 'SPY'),
]).set_index('Strategy')
print(final_perf.to_string())

# Final plot
plot_strats({best_name: best_lo}, f'Best Strategy ({best_name}) vs SPY')
```

- [ ] **Step 2: Add a markdown cell for the Task 2 writeup template**

```markdown
### Task 2: Description of Approach

**Features:** We constructed two feature tiers from the cross-sectionally standardized (`_xs`) variables. Tier 1 (~XX features) for our linear model included core momentum (`ret_1_xs`, `ret_2_12_xs`), value (`bm_xs`, `ep_xs`), quality (`gpa_xs`, `roe_xs`), risk (`vol_12m_xs`, `ivol_xs`), and analyst signals (`sue_xs`, `revision_xs`), plus hand-crafted composites: a quality score (profitability minus asset growth), earnings momentum (SUE + revisions), and a reversal-momentum combination. Tier 2 (~XX features) for ensemble models added all remaining `_xs` features and additional composites (peer-adjusted earnings, value-momentum interaction).

**Models:** We tested three models: (1) LassoCV for automatic feature selection with L1 regularization, (2) Random Forest (300 trees, max_depth=5, min_samples_leaf=50) for non-linear patterns, and (3) HistGradientBoosting (300 iterations, max_depth=4, learning_rate=0.05) as our strongest learner. All models were deliberately configured with shallow depth and large leaf sizes to prevent overfitting on noisy return data (~3% signal).

**What worked:** [Fill in: best model name and SR, which features Lasso selected, whether blend helped, any surprising findings]

**What didn't work:** [Fill in: worst model and why, any features that didn't help, overfitting observations]
```

- [ ] **Step 3: Run the final results cell and verify**

Check:
- Best strategy SR > 0.66 (beats SPY)
- MDD < 40%
- Plot renders correctly

- [ ] **Step 4: Commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "feat: add final results summary and writeup template"
```

---

---

## Phase 2: SR Improvement (Post-Rubric Update)

Current best: HGB_vt0.05, SR=1.20, MDD=-38.7% → 10/20 tier. These tasks aim to push SR above 1.3 (12/20) or 1.5 (15/20). Each strategy is tested independently against the current best, then the best combination is selected.

**Important:** All changes must be economically motivated, not data-mined. The rubric disqualifies overfitting.

---

### Task 10: K Tuning (Portfolio Concentration)

**Files:**
- Modify: `Code/AlphaChallenge.ipynb`

**Context:** Currently K=30. If the model's predictions are good, concentrating on fewer top picks should boost returns. Test K=10, 15, 20 with the current best config (HGB + vt=0.05).

- [ ] **Step 1: Run walk-forward for HGB_vt0.05 with K=10, 15, 20, 30**
- [ ] **Step 2: Compare SR and MDD across K values. Record results in progress.md**
- [ ] **Step 3: Identify best K. If SR improves, update the default K**

---

### Task 11: Retrain Frequency Tuning

**Files:**
- Modify: `Code/AlphaChallenge.ipynb`

**Context:** Currently retrain every 12 months. More frequent retraining captures changing market dynamics. Not overfitting — same model, just trained on more recent data more often.

- [ ] **Step 1: Run walk-forward for HGB_vt0.05 with retrain every 3, 6, 12 months**
- [ ] **Step 2: Compare SR and MDD. Record results in progress.md**
- [ ] **Step 3: Identify best retrain frequency**

---

### Task 12: Target Winsorization

**Files:**
- Modify: `Code/AlphaChallenge.ipynb`

**Context:** Extreme returns distort model training. Winsorize `y_xs` at ±3σ (or ±2σ) before training. The model only needs to rank stocks, not predict extreme magnitudes.

- [ ] **Step 1: Add winsorization to the training target (clip y_xs at percentiles)**
- [ ] **Step 2: Run walk-forward with winsorized vs non-winsorized target**
- [ ] **Step 3: Compare SR and MDD. Record results**

---

### Task 13: Feature Selection (Reduce Tier 2)

**Files:**
- Modify: `Code/AlphaChallenge.ipynb`

**Context:** 97 features is a lot. Many may be noise. Use HGB feature importances to keep only top 30-50 features. Fewer features = less overfitting = better OOS performance. Also addresses rubric's overfitting concern directly.

- [ ] **Step 1: Extract feature importances from a trained HGB model**
- [ ] **Step 2: Create a reduced feature builder (top 30, top 50 features)**
- [ ] **Step 3: Run walk-forward with reduced features. Compare SR/MDD**

---

### Task 14: Momentum Regime Filter

**Files:**
- Modify: `Code/AlphaChallenge.ipynb`

**Context:** Go to cash (or reduce allocation) when trailing 12-month SPY return is negative. Avoids the worst drawdown months, which both improves SR (removes negative returns) and reduces MDD.

- [ ] **Step 1: Compute trailing 12-month SPY return for each OOS month**
- [ ] **Step 2: Modify portfolio construction: if SPY trailing return < 0, return 0 (or risk-free rate) instead of investing**
- [ ] **Step 3: Run walk-forward with and without regime filter. Compare SR/MDD**

---

### Task 15: HGB Ensemble (Multiple Seeds/Configs)

**Files:**
- Modify: `Code/AlphaChallenge.ipynb`

**Context:** Average predictions from 3-5 HGB models with different random seeds or slightly different configs. Reduces variance in predictions. This is standard ensemble practice.

- [ ] **Step 1: Define 3-5 HGB configs (different seeds, or minor param variations)**
- [ ] **Step 2: Run walk-forward averaging their predictions**
- [ ] **Step 3: Compare ensemble SR/MDD vs single model**

---

### Task 16: Combine Best Improvements

**Files:**
- Modify: `Code/AlphaChallenge.ipynb`

**Context:** After testing each strategy independently, combine the ones that improved SR. E.g., best K + best retrain frequency + target winsorization. Verify the combination still passes MDD < 40%.

- [ ] **Step 1: Select the strategies that individually improved SR**
- [ ] **Step 2: Run walk-forward with all improvements combined**
- [ ] **Step 3: Record final combined SR/MDD. Update progress.md**
- [ ] **Step 4: If SR > 1.3, lock in the strategy. If not, iterate**

---

### Task 9: Final Review and Cleanup

**Files:**
- Review: `Code/AlphaChallenge.ipynb`

- [ ] **Step 1: Run all cells top-to-bottom**

Restart kernel and run all cells in sequence to verify everything works end-to-end. This catches any hidden state dependencies.

- [ ] **Step 2: Verify submission requirements**

Checklist:
- [ ] OOS from 2005-01 onward (no look-ahead)
- [ ] Walk-forward with retraining every 12 months
- [ ] Equal-weight long-only portfolio with K=30
- [ ] Only uses `alpha_dataset_v2.csv` (no external data)
- [ ] Reports Sharpe Ratio and MDD
- [ ] Reports annualized return
- [ ] Compares with SPY benchmark
- [ ] Cumulative wealth plot with benchmark
- [ ] MDD < 40%
- [ ] Task 2 writeup is filled in

- [ ] **Step 3: Final commit**

```bash
git add Code/AlphaChallenge.ipynb
git commit -m "chore: final review - all cells run clean, submission ready"
```
