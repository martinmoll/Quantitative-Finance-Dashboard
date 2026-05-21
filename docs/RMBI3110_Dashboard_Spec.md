# RMBI3110 Alpha Dashboard — Claude Code Specification

## Project Overview

Build a **Bloomberg-like interactive dashboard** using **Python + Streamlit** that implements the full quantitative alpha pipeline from the HKUST RMBI3110 course (Risk Management & Business Intelligence). The dashboard uses a pre-existing **S&P 500 dataset with 116 cross-sectionally standardized features** (the Assignment 2 dataset from the course).

The goal: a user can configure alpha model parameters, run a walk-forward backtest, and see the full pipeline from data → model → portfolio → performance, with all relevant course diagnostics displayed.

---

## Tech Stack

- **Frontend/App**: Streamlit
- **Data**: Pre-loaded parquet/csv (S&P 500, ~500 stocks, ~20 years monthly, 116 features, forward returns as target)
- **ML Models**: scikit-learn (HistGradientBoostingRegressor, RandomForestRegressor, Lasso, ElasticNet, Ridge)
- **Regressions**: statsmodels (OLS with HAC SEs, Fama-MacBeth)
- **Factors**: pandas-datareader or bundled CSV (Fama-French 5 factors from Ken French library)
- **Plotting**: Plotly (interactive charts that match Bloomberg's dark aesthetic)
- **Style**: Dark theme throughout. Bloomberg terminal aesthetic — black/dark navy background, green/amber/white text, clean monospace numbers.

---

## Data Expectations

The Assignment 2 dataset has this structure:

```
Columns: date, permno (stock ID), ret_forward (target: next-month return),
          + 116 feature columns (cross-sectionally standardized each month)

Features include (among others):
- Momentum: ret_2_12, ret_1_1 (short-term reversal)
- Value: bm (book-to-market), ep (earnings/price), cfp (cash flow/price)
- Quality/Profitability: roe, roa, gpa, sue (earnings surprise)
- Size: mkt_cap, log_me
- Volatility: ivol (idiosyncratic vol), beta, vol_252d
- Technical: mom_x_size, val_x_prof (pre-computed interactions)
- Analyst: revision, disp (dispersion)

Each month has ~500 stocks. Features are already z-scored cross-sectionally.
```

The user should be able to upload or point to this dataset. Provide a file uploader widget AND a path input for local files.

---

## Dashboard Pages / Tabs

### Page 1: Data Explorer

**Purpose**: Understand the dataset before modeling.

**Components**:
- Summary statistics table: N stocks, N months, date range, feature count
- Feature distribution viewer: select a feature → histogram + box plot (cross-sectional for a chosen month)
- Correlation heatmap: top-N features pairwise correlation matrix (Spearman, not Pearson)
- Missing data summary: % NaN per feature, per month
- Time-series of universe size (stocks per month over time)

**Course concepts covered**: Module 1.1-1.2 (Pandas, data exploration, groupby, aggregation)

---

### Page 2: Factor Analysis (CAPM & FF5)

**Purpose**: Run CAPM and Fama-French regressions on individual stocks or portfolios.

**Components**:

**Stock-Level Regression Panel**:
- User selects a stock (by permno or ticker if available)
- Run CAPM: r_i - r_f = α + β(r_m - r_f) + ε
- Run FF3: add SMB, HML
- Run FF5: add RMW, CMA
- Display regression output table (matching statsmodels summary): coef, HAC SE, t-stat, p-value
- Highlight alpha significance (green if |t| > 1.96 on const, red otherwise)
- Display R², Adjusted R², Durbin-Watson, Jarque-Bera
- **Always use HAC standard errors**: `cov_type='HAC', cov_kwds={'maxlags': 5}`

**Rolling Beta Plot**:
- User selects W (window size) via slider: range 21 to 504, default 252
- Plot rolling beta over time with confidence bands
- Show Bloomberg-shrunk beta: β_adj = 0.67β̂ + 0.33
- Overlay the two (OLS vs shrunk)

**VIF Table**:
- For the FF5 regression, compute VIF for each regressor
- Color-code: green (VIF < 5), amber (5-10), red (> 10)

**Wald Test Panel**:
- Buttons to test joint hypotheses: "SMB=0, HML=0", "RMW=0, CMA=0", "SMB=0, HML=0, RMW=0, CMA=0"
- Display Wald statistic, p-value, and reject/fail decision

**Market-Neutral Hedge Calculator**:
- Input: portfolio beta
- Output: SPY hedge weight (w_m = -β_p / β_m), hedged return decomposition

**Course concepts covered**: Module 3.1 (CAPM, OLS, beta, rolling beta, hedge), Module 3.2 (FF5, VIF, Wald test, IR, OVB)

---

### Page 3: Alpha Model Lab

**Purpose**: Train, configure, and compare alpha models.

**Model Selector** (sidebar):
- Model type: HGBR, Random Forest, LASSO, Ridge, Elastic Net, Fama-MacBeth
- For HGBR: max_depth (1-6, default 2), learning_rate (0.01-0.3, default 0.05), max_iter (50-1000, default 300), min_samples_leaf (20-200, default 80), l2_regularization (0.0-10.0, default 1.0)
- For RF: n_estimators (50-500, default 200), max_depth (1-8, default 4), max_features (options: K/3, sqrt(K), K), min_samples_leaf (20-200, default 50)
- For LASSO/Ridge/ElasticNet: alpha slider (log scale 1e-5 to 10), l1_ratio for ElasticNet (0-1)
- For Fama-MacBeth: no hyperparameters (runs monthly cross-sectional OLS + averages slopes)

**Feature Selection Panel**:
- Checkbox list of all 116 features, grouped by category (momentum, value, quality, size, vol, technical, analyst)
- "Select All" / "Deselect All" buttons
- Quick presets: "Momentum only", "Value only", "All", "Kitchen sink", "Top-20 by univariate IC"

**Walk-Forward Configuration**:
- Training start: date selector
- OOS start: date selector (must be after training start)
- Retrain frequency: 6, 12, 24 months (default 12)
- Window type: Expanding (recommended) vs Rolling
- Top-K: slider 10-100, default 30
- Portfolio type: Long-Only top-K or Long-Short (top-K minus bottom-K)

**Run Backtest Button** → triggers the walk-forward loop:
```
For each rebalance date t:
  1. Train on all data before t (expanding) or last W months (rolling)
  2. Predict R̂_i,t+1 for all stocks at time t using X_i,t
  3. Rank stocks by R̂
  4. Long-only: equal-weight top K. L/S: EW top K minus EW bottom K.
  5. Compute realized portfolio return for month t+1
  6. Store IC_t = Spearman(R̂, R_realized) cross-sectionally
```

**Course concepts covered**: Module 3.3 (all alpha models, walk-forward, regularization, decision trees, HGBR)

---

### Page 4: Backtest Results & Diagnostics

**Purpose**: Evaluate the backtest comprehensively.

**Performance Panel** (top):
- Cumulative return chart (log scale) with SPY benchmark overlay
- Key metrics in large cards: Annualized Return, Annualized Vol, Sharpe Ratio, Max Drawdown, Calmar Ratio (SR/MDD)
- Monthly returns heatmap (year × month grid, color-coded)
- Drawdown chart over time

**IC Dashboard** (middle):
- Monthly IC time series plot
- Cumulative IC plot (ΣIC_t — should trend up)
- Rolling 12-month IC with ±1σ bands
- Summary: Mean IC, IC t-stat, ICIR, Hit Rate
- Color-code against thresholds: Mean IC > 0.03 (green), t > 2 (green), Hit > 55% (green), ICIR > 0.3 (green)

**Fundamental Law Panel**:
- Compute and display: IR_realized, IC_mean, BR_nominal (K × 12), BR_implied = (SR/IC)²
- Show IR upper bound = IC × √BR vs realized IR
- IC_required = (SR_target + Cost_SR) / √BR — user inputs SR_target and TC assumptions

**Feature Importance** (bottom):
- If tree-based model: bar chart of feature_importances_ (top 20)
- Permutation importance: compute IC(original) − IC(shuffled X_k) for top features
- Feature-level IC: Spearman(X_k, R_realized) for each feature (univariate signal strength)
- Feature importance drift: rank correlation of importance across retraining windows

**Course concepts covered**: Module 4.1 (IC, ICIR, Fundamental Law, feature importance, alpha decay)

---

### Page 5: Portfolio Construction & Risk

**Purpose**: Go from raw predictions to optimized portfolios.

**Construction Method Selector**:
- Equal-Weight Top-K (baseline)
- Score-Weighted: w_i = R̂_i / Σ R̂_j
- Inverse-Vol (ERC proxy): w_i ∝ 1/σ_i (using trailing realized vol)
- Full ERC: numerical optimization for equal risk contribution
- Mean-Variance (with Ledoit-Wolf shrinkage for Σ̂)

**Risk Decomposition Panel**:
- Risk contribution pie chart: RC_i for each position (or top-10 + "other")
- For 2 methods side by side: EW vs ERC showing how risk shifts
- Factor exposure table: run FF5 on portfolio returns, show β_M, s, h, r, c with alert thresholds (|β| > 0.10)
- Factor exposure time series: rolling FF5 exposures over the backtest

**Turnover & Cost Panel**:
- Monthly turnover time series
- Average monthly turnover, annualized turnover
- TC drag calculator: user inputs cost-per-trade in bps → computes TC_ann = TO_ann × c
- Net-of-cost Sharpe: SR_gross vs SR_net
- Cost_SR = TC_ann / σ_p

**Comparison Table**:
- Side-by-side metrics for all construction methods: SR, Ann Return, Ann Vol, MDD, Turnover, TC Drag, Net SR

**Course concepts covered**: Module 4.1 (ERC, risk contribution, turnover, TC, MVO, Ledoit-Wolf, factor monitoring)

---

### Page 6: Model Monitoring & OOD Detection

**Purpose**: Production monitoring dashboard.

**KS Test Panel**:
- For each feature, compute KS statistic between training distribution and most recent OOS month
- Table: feature name, D statistic, p-value, flag (D > 0.10)
- Summary: X of Y features flagged → alert if > 20%
- Heatmap: D statistic per feature over time (months × features)

**Alpha Decay Curve**:
- Plot IC(h) for h = 1, 2, ..., 12 months ahead
- Show half-life h* where IC drops to IC(1)/2
- Annotate: h* < 3 → "monthly rebalance", h* > 9 → "quarterly OK"

**Signal Staleness**:
- Plot monthly turnover over time
- Flag if TO < 10%/month for 3+ consecutive months
- Highlight periods where signal became stale

**Automated Alert Dashboard**:
- Traffic-light indicators for:
  - IC decay (rolling 6m IC < 0.02 → WARN)
  - IC collapse (monthly IC < -0.03 → CRITICAL)
  - OOD shift (>20% features D > 0.10 → WARN)
  - Factor drift (any |β_k| > 0.10 → WARN)
  - Stale signal (TO < 10%/mo for 3+ months → WARN)

**Retraining Recommendation**:
- Based on alerts, suggest: "No action needed" / "Consider retraining" / "Retrain immediately"
- Show whether scheduled (every 12 months) or triggered (IC-based) retraining would have caught the issue

**Course concepts covered**: Module 4.1 (KS test, alpha decay, monitoring cadence, retraining policy, feature importance drift)

---

## Design Requirements

### Visual Style
- **Dark theme everywhere** — Streamlit dark mode (`theme.base = "dark"` in .streamlit/config.toml)
- **Bloomberg aesthetic**: dark navy (#0A1628) background, monospace numbers, green (#00D26A) for positive, red (#FF4444) for negative, amber (#FFB800) for warnings, white (#FFFFFF) for labels
- **Plotly dark template**: `template="plotly_dark"` on all charts
- **Cards**: use `st.metric()` for key numbers, custom CSS for Bloomberg-style metric cards
- **Tables**: styled with alternating dark rows, monospace font for numbers

### Layout
- Sidebar: global controls (dataset path, date range, model selection)
- Main area: tabs or pages using `st.tabs()` or multipage app
- Responsive: should work on 1920×1080 and 1440×900

### Performance
- Cache expensive computations with `@st.cache_data` and `@st.cache_resource`
- Walk-forward backtest can take 30-60 seconds — show progress bar with `st.progress()`
- Pre-compute and store backtest results in session state so switching tabs doesn't re-run

---

## File Structure

```
rmbi3110-dashboard/
├── .streamlit/
│   └── config.toml          # dark theme config
├── app.py                   # main Streamlit app (multipage router)
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Factor_Analysis.py
│   ├── 3_Alpha_Model_Lab.py
│   ├── 4_Backtest_Results.py
│   ├── 5_Portfolio_Construction.py
│   └── 6_Monitoring.py
├── core/
│   ├── data_loader.py       # load and validate dataset
│   ├── factor_models.py     # CAPM, FF3, FF5 regressions (HAC)
│   ├── alpha_models.py      # HGBR, RF, LASSO, FM wrappers
│   ├── backtest.py          # walk-forward engine
│   ├── portfolio.py         # EW, score-weight, ERC, MVO construction
│   ├── diagnostics.py       # IC, ICIR, Fundamental Law, KS, decay
│   └── risk.py              # risk contribution, turnover, TC, factor exposure
├── data/
│   ├── README.md            # instructions for placing the dataset
│   └── ff5_factors.csv      # Fama-French 5 factors (bundled)
├── requirements.txt
└── README.md
```

---

## Key Implementation Details

### Walk-Forward Backtest Engine (`core/backtest.py`)

This is the most complex component. The function signature should be:

```python
def run_walk_forward(
    data: pd.DataFrame,         # full panel: date, permno, features, ret_forward
    model_class: str,           # "HGBR", "RF", "LASSO", "Ridge", "ElasticNet", "FM"
    model_params: dict,         # hyperparameters
    feature_cols: list[str],    # selected features
    oos_start: str,             # first OOS date
    retrain_freq: int,          # months between retrains (default 12)
    window_type: str,           # "expanding" or "rolling"
    top_k: int,                 # number of stocks in portfolio
    portfolio_type: str,        # "long_only" or "long_short"
    progress_callback=None,     # for Streamlit progress bar
) -> BacktestResult:
```

The `BacktestResult` dataclass should contain:
```python
@dataclass
class BacktestResult:
    monthly_returns: pd.Series       # portfolio returns indexed by date
    monthly_ic: pd.Series            # Spearman IC per month
    predictions: pd.DataFrame        # date, permno, y_hat, y_realized
    holdings: pd.DataFrame           # date, permno, weight
    feature_importance: pd.DataFrame  # feature, importance (if tree-based)
    train_dates: list                 # retrain boundary dates
    model_params: dict               # for reproducibility
```

### Critical: No Look-Ahead Bias

The walk-forward loop MUST:
1. Only train on data strictly BEFORE the prediction month
2. Only use features X_i,t (available at time t) to predict R_i,t+1
3. Never include the current month's return in training data
4. Retrain on expanding window (all data up to current point)

### Regression Requirements

All time-series regressions (CAPM, FF3, FF5) MUST use:
```python
model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
```

Never use classical standard errors for financial time-series data.

### IC Computation

```python
from scipy.stats import spearmanr
# Cross-sectional Spearman each month
ic_t = data.groupby('date').apply(
    lambda g: spearmanr(g['y_hat'], g['ret_forward']).correlation
)
```

### ERC Portfolio (Uncorrelated Approximation)

For the simplified version (uncorrelated assets):
```python
# w_i ∝ 1/σ_i
trailing_vol = returns.rolling(12).std().iloc[-1]  # trailing 12-month vol
raw_weights = 1.0 / trailing_vol
weights = raw_weights / raw_weights.sum()  # normalize to sum=1
```

### KS Test

```python
from scipy.stats import ks_2samp

def compute_ks(X_train, X_current, threshold=0.10):
    results = []
    for col in X_train.columns:
        D, pval = ks_2samp(X_train[col].dropna(), X_current[col].dropna())
        results.append({'feature': col, 'D': D, 'pval': pval, 'flag': D > threshold})
    return pd.DataFrame(results).sort_values('D', ascending=False)
```

---

## Course Module Coverage Checklist

Use this to verify every major concept from the course is represented somewhere in the dashboard:

### Module 1.0-1.2 (Python/Pandas, 30%)
- [ ] DataFrame operations, groupby, aggregation → Data Explorer page
- [ ] axis=0 vs axis=1 → used in portfolio return computation
- [ ] MultiIndex handling → if applicable to data structure
- [ ] Rolling, resampling → rolling beta, rolling IC, rolling vol

### Module 2 (Financial Markets, 5%)
- [ ] Simple & log returns → Data Explorer return histograms
- [ ] Annualization (√252, √12) → all performance metrics
- [ ] Sharpe ratio → Backtest Results
- [ ] Drawdown → Backtest Results
- [ ] Portfolio variance → Portfolio Construction risk decomposition

### Module 3.1 (CAPM/Beta, 10%)
- [ ] OLS regression with HAC → Factor Analysis page
- [ ] Beta computation → Factor Analysis
- [ ] Alpha significance (|t| > 1.96 on const) → highlighted in output
- [ ] Rolling beta with configurable W → Factor Analysis
- [ ] Bloomberg shrinkage → Factor Analysis
- [ ] Market-neutral hedge → Factor Analysis hedge calculator

### Module 3.2 (FF5/VIF/IR, 10%)
- [ ] FF5 regression → Factor Analysis
- [ ] Factor loading interpretation (s > 0 = small-cap, etc.) → Factor Analysis
- [ ] VIF computation and thresholds → Factor Analysis
- [ ] Wald test (HAC χ²) → Factor Analysis
- [ ] Information Ratio (α_ann / TE_ann) → Backtest Results
- [ ] Omitted variable bias concept → comparison of CAPM vs FF5 alpha

### Module 3.3 (Alpha Models, 20%)
- [ ] HGBR with all hyperparameters → Alpha Model Lab
- [ ] Random Forest → Alpha Model Lab
- [ ] LASSO, Ridge, Elastic Net → Alpha Model Lab
- [ ] Fama-MacBeth regression → Alpha Model Lab
- [ ] Walk-forward protocol (no look-ahead) → Backtest engine
- [ ] Decision tree mechanics (displayed if max_depth small) → optional tree visualization
- [ ] Gini / MSE split criteria → shown in model info panel
- [ ] Feature importance (tree gain, permutation, feature IC) → Backtest Results
- [ ] Spearman IC (not Pearson) → all IC computations

### Module 4.1 (Portfolio Construction, 25%)
- [ ] Equal-weight, Score-weight, ERC, MVO → Portfolio Construction
- [ ] Risk contribution (RC_i) → Portfolio Construction
- [ ] Inverse-vol weighting for ERC → Portfolio Construction
- [ ] Ledoit-Wolf shrinkage for Σ̂ → Portfolio Construction MVO
- [ ] Turnover computation (0.5 × Σ|Δw|) → Portfolio Construction
- [ ] Transaction cost drag → Portfolio Construction
- [ ] Cost_SR → Portfolio Construction
- [ ] Fundamental Law (IR = IC × √BR, implied BR, IC_req) → Backtest Results
- [ ] IC diagnostics (mean, t-stat, hit rate, ICIR) → Backtest Results
- [ ] Cumulative IC → Backtest Results
- [ ] KS test (D = sup|F_n - G_m|) with thresholds → Monitoring
- [ ] Alpha decay curve (IC at multiple horizons) → Monitoring
- [ ] Feature importance drift → Monitoring
- [ ] Stale signal detection (TO < 10%) → Monitoring
- [ ] Automated alert thresholds → Monitoring
- [ ] Monitoring cadence (daily/weekly/monthly/quarterly) → Monitoring
- [ ] Retraining policy (scheduled + triggered = hybrid) → Monitoring
- [ ] Factor exposure monitoring (|β_k| > 0.10 per factor) → Portfolio Construction
- [ ] Portfolio method comparison table → Portfolio Construction

---

## Dependencies (requirements.txt)

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
```

---

## Getting Started Instructions (for Claude Code)

1. Create the project structure above
2. Start with `core/data_loader.py` — handle dataset loading and validation
3. Build `core/backtest.py` — the walk-forward engine is the critical path
4. Build pages one at a time, starting with Page 3 (Alpha Model Lab) + Page 4 (Backtest Results) since those are the core functionality
5. Add Factor Analysis (Page 2) and Portfolio Construction (Page 5) next
6. Data Explorer (Page 1) and Monitoring (Page 6) last
7. Apply Bloomberg dark theme styling throughout
8. Test with the Assignment 2 dataset

The most important thing: **the walk-forward backtest must have zero look-ahead bias**. Every other bug is forgivable. Look-ahead bias is not.
