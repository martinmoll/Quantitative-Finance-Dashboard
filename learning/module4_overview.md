# RMBI3110 — Module 4.1 Comprehensive Overview

**Course:** RMBI3110 Risk Management and Business Intelligence — HKUST Business School, Spring 2026
**Instructor:** Professor Xuhu Wan
**Lecture:** 4.1 Portfolio Optimization — Alpha Diagnostics, Construction, and Production
**Source deck:** 86 slides total
**Module title:** Portfolio Construction and Risk Allocation

> **Purpose of this document.** This is a build-spec for an alpha forward-pass model. Every formula, threshold, and decision rule in the lecture is recorded here in a form Claude Code can implement directly. Concepts are grouped by pipeline stage, not by slide order, so a coding agent can read top-to-bottom and produce working components. Where a formula is approximate, I say so. Where a slide gives a numerical example, the exact numbers are reproduced so unit tests can be written against them.

> **Scope warning.** This deck is Lecture 4.1 only. Several conclusions in the deck (e.g., "MDD = −29% is too large to lever up") explicitly punt to Module 4.2 for the fix (Kelly criterion, volatility targeting, dynamic leverage). Module 4.2 is **not** in this overview because it is **not** in the source deck I was given. If your forward-pass model needs sub-5% drawdown control, you will need Module 4.2 separately.

---

## 0. Pipeline context — where Module 4.1 fits

Module 3 produced the model. Module 4.1 turns the model's output into a portfolio.

```
[Module 3.1–3.2]            [Module 3.3]
  Factor betas β̂_{i,t}        Alpha model α̂_{i,t}
  Covariance Σ                (HGBR predictor)
        \                    /
         \                  /
          v                v
        +----------------------+
        |  Portfolio Optimizer |   <-- THIS LECTURE (Module 4.1)
        |  (Section 2 here)    |
        +----------------------+
                  |
                  v
            Weights w_t
                  |
                  v
        +----------------------+
        | Execution + TC model |   <-- Section 3 here
        +----------------------+
                  |
                  v
        +----------------------+
        |   Risk monitor       |   <-- Section 1 here, runs continuously
        |   drawdown < 5%?     |       feeds back to optimizer
        +----------------------+
```

The deck's lecture order is Section 1 (Diagnostics) → Section 2 (Construction) → Section 3 (Implementation), but in **build order** Section 2 is the core engine, Section 3 wraps cost/execution around it, and Section 1 is the runtime/offline monitor. This document is organized in lecture order so it matches the slide deck, but the coder can implement Section 2 first.

The deck's running example uses an HGBR alpha model on the S&P 500 universe trained 2005–2023, monthly rebalance, K = 30 stocks per side for L/S, with EW, score-weighted, MVO, constrained MVO, MAXSER, and risk-parity portfolios all evaluated head-to-head. Numbers reproduced in this document are from that empirical setup.

---

## SECTION 1 — Alpha Diagnostics and Model Monitoring

The job of this section: detect when an alpha signal stops working, before the P&L tells you. The single most important diagnostic is the **Information Coefficient (IC)**.

### 1.1 Information Coefficient (IC) — the central metric

**Definition.** The IC at time `t` is the cross-sectional Spearman rank correlation between the model's predicted next-period returns and the realized next-period returns:

```
IC_t = Spearman( R̂_{1,t+1}, R̂_{2,t+1}, ..., R̂_{N,t+1};
                 R_{1,t+1},  R_{2,t+1},  ..., R_{N,t+1} )
```

For each rebalance period, compute one IC scalar. Stack across months → an IC time series.

**Why Spearman, not Pearson.** Portfolio construction picks the top-decile and shorts the bottom-decile — it is a **rank-based** operation. Pearson penalizes the model for outliers in the realized-return distribution (one stock with a +25% month) that don't matter to a rank-based strategy. Spearman is invariant to those outliers. The deck's worked example: 5 stocks with `ŷ = (1,2,3,4,5)` and realized `y = (1,2,3,4,25)` → Pearson ≈ 0.78 but Spearman = 1.00 (all ranks match exactly). Use Spearman.

**Interpretation table** (slide 5):

| IC value | Interpretation |
|---|---|
| −1.0 | Perfectly anti-predictive (flip the sign → perfect signal) |
| [−0.10, −0.02] | **Anti-predictive** — model loses money systematically; danger |
| 0 | No predictive power (random) |
| 0.02 | Weak but potentially useful |
| 0.05 | Good for a quant model |
| 0.10 | Excellent (rare) |
| 1.0 | Perfect foresight (unattainable) |

A real production ML alpha sits at IC ≈ 0.02–0.05. Persistently negative IC is not just unprofitable — it is **systematically losing money**; cut exposure first, investigate second.

**IC reporting metrics — always report all five:**

| Metric | Formula | What it measures |
|---|---|---|
| Mean IC | `IC̄ = (1/T) Σ_t IC_t` | Average signal quality |
| IC Std | `σ_IC = std(IC_t)` | Signal stability |
| IC t-stat | `t = IC̄ / (σ_IC / √T) = ICIR × √T` | Significance (`|t| > 2` required) |
| IC Hit Rate | fraction of months with `IC_t > 0` | How often signal is right |
| ICIR | `IC̄ / σ_IC` | IC Sharpe ratio (consistency) |

**Health rules of thumb:**
- Mean IC > 0.03, t-stat > 2, hit rate > 55% → viable alpha
- ICIR > 0.5 → good consistency
- High mean IC but hit rate < 50% → unreliable; works only in narrow regimes

### 1.2 Information Ratio (IR) and the Fundamental Law

**Information Ratio.** Risk-adjusted active return relative to a benchmark `B`:

```
IR = E[R_p − R_B] / σ(R_p − R_B) = active return / tracking error
```

For monthly data, annualize: multiply mean by 12, std by √12.

For a market-neutral L/S portfolio, IR = SR (the benchmark is zero).

**Worked example:** mean monthly active return 0.4% → annualized 4.8%; monthly TE 1.5% → annualized 5.20%; IR = 4.8/5.20 = 0.92.

**Grinold's Fundamental Law of Active Management:**

```
IR = IC × √BR
```

where:
- IR = information ratio (defined above)
- IC = Spearman correlation between predictions and returns
- BR = breadth = number of **independent** bets per year

**Critical caveat — BR counts independent bets, not raw trades.** Stocks are correlated, signals persist across months. Effective breadth `BR_eff << K × T` (number of forecasts × rebalance frequency). The formula is an **upper bound**, not a guarantee.

**Numerical example.** HGBR alpha, monthly rebalance, K=30 stock forecasts/month, T=12 → nominal `BR = 30 × 12 = 360`. With `IC = 0.03`: upper-bound `IR ≤ 0.03 × √360 = 0.57`. Reality check: in practice realized IR is often 0.3–0.5 of that ceiling because of correlation and signal persistence.

**Implied breadth** (back out from realized data):

```
BR_implied = (SR_realized / IC)²
```

Use this to detect whether your construction is converting IC into IR efficiently.

**Empirical numbers from the lab (slide 10), all four model types compared head-to-head with BR=360 nominal:**

| Model | Mean IC | Theo SR (BR=360) | Actual SR (OOS) | Implied BR = (SR/IC)² |
|---|---|---|---|---|
| FM | 0.0218 | 0.41 | 0.71 | 1,061 |
| LASSO | 0.0225 | 0.43 | 0.67 | 886 |
| RF | 0.0208 | 0.39 | 0.91 | 1,914 |
| HGBR | 0.0248 | 0.47 | 0.97 | 1,531 |

**Interpretation.** Implied BR > nominal BR for all four. The Fundamental Law systematically **underestimates** ML signals because Spearman IC measures only rank agreement and cannot capture the nonlinear interactions that tree models exploit. ML signals deliver more bang per IC-unit than the law predicts.

### 1.3 IC vs Wealth — five empirical insights (slide 12)

The deck's analysis of the four-model bake-off:

1. **IC is similar (≈0.02) but cumulative wealth is dramatically different** — FM ends at 6×, HGBR at 13×. IC alone doesn't determine performance; how the predictions translate into returns matters more.
2. **When rolling IC drops below zero, drawdowns follow within the next rebalance cycle** (visible in 2008, 2019, 2022 across all models). IC is a **leading indicator** for dynamic position sizing.
3. **Tree models recover faster after IC drops** — nonlinear signals capture mean-reversion that linear models miss.
4. **Fundamental Law underestimates tree models** — same IC but 2–3× higher wealth means trees exploit nonlinear info Spearman can't see.
5. **HGBR has the most stable IC, FM the most volatile.** Regularization (shallow trees, learning rate) stabilizes the signal.

### 1.4 Model decay — what kills alpha and how fast

**Definition.** Model decay = gradual loss of predictive power over time.

**Causes (slide 13):**
- **Crowding** — other funds find the same signal → alpha erodes
- **Regime change** — market structure shifts (COVID, rate hikes)
- **Data distribution shift** — feature distributions change
- **Overfitting** — model memorized training patterns that no longer hold
- **Market adaptation** — counterparties learn and front-run trades

**Half-life rules of thumb:**
- Momentum signals (e.g., 12-month minus 1-month): **half-life ≈ 6–12 months**
- Value signals: **half-life ≈ 2–5 years**
- ML signals (HGBR): depends on feature stability
- **Standard practice: retrain at least annually; monitor IC monthly.**

**The danger:** a decayed model doesn't just underperform — it can **reverse**. A momentum signal in a mean-reverting regime loses money systematically.

### 1.5 Detecting decay — the IC dashboard

Monitor monthly (slide 14):

| Indicator | Healthy | Warning | Action |
|---|---|---|---|
| Rolling 12-mo IC | > 0.02 | 0–0.02 | Review features |
| IC hit rate (12 mo) | > 55% | 45–55% | Reduce exposure |
| ICIR (12 mo) | > 0.3 | 0–0.3 | Consider retrain |
| Rolling 12-mo SR | > 0.5 | 0–0.5 | Scale down |
| Consecutive negative IC | < 4 months | 4–6 months | **Retrain immediately** |

**IC is a leading indicator of drawdowns. Watch IC before it shows up in P&L.**

### 1.6 Cumulative IC — visual diagnostic

```
CumIC_T = Σ_{t=1}^T IC_t
```

- **Rising line** → model consistently adds value (healthy)
- **Flat line** → no predictive power (neutral)
- **Declining line** → model is anti-predictive (actively losing money)
- **Slope change** → regime shift; model worked before but stopped

**Decision rules:**
- CumIC flattens for 6+ months → scheduled retrain won't fix it (structural decay)
- CumIC drops sharply → emergency: cut exposure first, investigate second
- Compare CumIC across models to identify which is decaying

CumIC is to alpha models what the equity curve is to portfolios.

### 1.7 Retraining policy — three options

**Scheduled (every 12 months).** Simple, predictable, low operational risk. Model may decay between retrains.

**Triggered (retrain when IC drops below threshold).** More adaptive, catches decay faster. Risk: overfitting to recent noise.

**Hybrid (recommended).** Annual scheduled retrain as baseline, plus trigger if **any** of:
1. Rolling 12-mo IC < 0 for 3+ months
2. Major regime event (e.g., VIX > 40)
3. Feature distribution shift detected (KS test, see §1.10)

**Never retrain in response to a single bad month** — that's chasing noise.

**Empirical comparison from lab (slide 18), HGBR with both policies, retrain trigger = rolling 6-mo IC < 0 for 3 consecutive months:**
- Scheduled (annual): 19 retrains over 19 years, SR = 0.29
- Hybrid (annual + IC-triggered): 23 retrains, SR = 0.36

Hybrid wins by ≈25%. The four extra retrains caught regime breaks early.

**Each cadence has a different question. Don't skip levels:**
- Daily catches **pipeline issues**
- Monthly catches **signal decay**
- Quarterly catches **structural change**
- Annually decides **the strategy's future**

### 1.8 Feature importance — three measures

**1. Tree gain importance (Gini / split gain), tree-only.**
```
I_k^gain = Σ_{splits on k} Δloss(split)
```
Available as `model.feature_importances_` for HGBR/RF. **Pro:** free, built-in. **Con:** biased toward high-cardinality features.

**2. Permutation importance, model-agnostic.**
```
I_k^perm = IC(original) − IC(shuffled X_k)
```
Shuffle column k in the test set, re-evaluate IC. Drop in IC → feature mattered. Tool: `sklearn.inspection.permutation_importance`. **Pro:** unbiased, simple. **Con:** O(K × T) runtime.

**3. Feature-level IC, model-free.**
```
I_k^IC = Spearman( X_{k,i,t}, R_{i,t+1} )
```
Bypass the model entirely — correlate each feature with realized returns. Tells you whether the **feature itself** predicts returns regardless of how the model uses it. **Pro:** model-independent diagnostic; catches feature decay even when the model masks it. **Con:** ignores feature interactions; nonlinear features may have low univariate IC.

**Use permutation importance for production monitoring. Use feature-level IC as a sanity check that doesn't depend on the model. If they disagree, investigate whether the model is exploiting interactions or masking decay.**

### 1.9 Feature importance drift detection

Compute one feature-importance vector per retraining window, then compare across retrains.

**Drift rule:** flag a feature when `|Recent IC − Mean IC| > 0.01` AND (sign flipped OR magnitude collapsed).

**Stable signal example:**
| Feature | 2022 | 2023 | Verdict |
|---|---|---|---|
| momentum | 0.15 | 0.14 | stable |
| profitability | 0.12 | 0.11 | stable |
| vol | 0.08 | 0.09 | stable |

**Drifting signal example (warning):**
| Feature | 2022 | 2023 | Verdict |
|---|---|---|---|
| momentum | 0.15 | 0.03 | **DRIFT (decay)** |
| profitability | 0.12 | 0.02 | **DRIFT (decay)** |
| sector_iv | 0.02 | 0.18 | drift up — regime change |

When the top features flip ranks, the regime has shifted.

**Actual feature-level IC drift table from the lab (HGBR, S&P 500):**

| Feature | Mean IC (long-run) | Recent IC | Verdict |
|---|---|---|---|
| ret_1 | −0.0006 | −0.0234 | DRIFT (decay) |
| ret_2_12 | 0.0080 | 0.0049 | stable |
| ret_2_6 | 0.0047 | −0.0126 | DRIFT (decay) |
| ret_13_36 | 0.0161 | −0.0307 | DRIFT (decay) |
| vol_12m | 0.0047 | 0.0111 | stable |
| max_ret_12m | 0.0094 | 0.0195 | improving |
| beta | 0.0026 | 0.0117 | stable |
| ivol | −0.0002 | −0.0009 | stable |
| turnover | −0.0038 | −0.0079 | stable |
| log_me | −0.0150 | −0.0107 | stable |

**Interpretation.** Three momentum features (`ret_1`, `ret_2_6`, `ret_13_36`) decaying simultaneously is the post-2015 momentum unwind. `max_ret_12m` improving suggests the lottery-stock effect is strengthening.

### 1.10 Out-of-distribution (OOD) detection — Kolmogorov-Smirnov test

ML models extrapolate poorly. If today's features look nothing like the training data, predictions are unreliable regardless of historical IC.

**KS test.** For each feature, compute the maximum vertical distance between the empirical CDFs of the training distribution `F_n` and the current month `G_m`:

```
D = sup_x | F_n(x) − G_m(x) |
```

- `n` = training observations (e.g., trailing 36 months × stocks_per_month)
- `m` = current month observations
- `D ∈ [0,1]`. `D=0` → identical distributions. `D=1` → disjoint.
- Distribution-free (no normality assumption).

**Practical thresholds for OOD monitoring (large n, m):**

| D value | Interpretation |
|---|---|
| < 0.05 | Distributions essentially identical |
| 0.05–0.10 | Minor shift (typical month-to-month noise) |
| > 0.10 | Meaningful shift — flag for review |
| > 0.20 | Major distribution shift — pause trading; investigate |

**Aggregate OOD flag:** if `> 20%` of features have `D > 0.10`, treat predictions as untrusted until investigated.

**Python implementation pattern (slide 34):**
```python
from scipy.stats import ks_2samp

def detect_ood(X_train, X_current, threshold=0.10):
    """KS statistic per feature; flag features with D > threshold."""
    results = []
    for feat in X_train.columns:
        train_vals = X_train[feat].dropna()
        cur_vals = X_current[feat].dropna()
        if len(train_vals) < 30 or len(cur_vals) < 30:
            continue
        ks_stat, pval = ks_2samp(train_vals, cur_vals)
        results.append({
            'feature': feat,
            'KS': ks_stat,
            'pvalue': pval,
            'flag': ks_stat > threshold
        })
    df = pd.DataFrame(results).sort_values('KS', ascending=False)
    n_flagged = df['flag'].sum()
    print(f"{n_flagged} / {len(df)} features show distribution shift (KS > {threshold})")
    return df

# Compare current month features vs trailing 36-month training window
ood = detect_ood(X_train, X_current, threshold=0.10)
```

**Empirical numbers (slide 35), HGBR S&P 500 2005–2023, 227 months:**
- Average fraction of features flagged: 24.9%
- Mean KS = 0.080, max KS = 0.726 (during 2008–09 crisis)
- Worst OOD month: 2009-02 (57% features flagged). Top shifted features: `size_peer_ret` (0.33), `earn_growth_yoy` (0.33), `delta_bm` (0.31), `beat` (0.30), `sue_chg` (0.30), `age` (0.27), `earn_quality` (0.27), `dp_ratio` (0.26), `illiq_12m` (0.25), `gm_chg` (0.24). All p-values < 10⁻²⁹.

### 1.11 Cross-model correlation — diversification check

Monitor rolling correlation between predictions from two distinct alpha models (e.g., FM and HGBR):

- ρ(R̂^FM, R̂^HGBR) < 0.5 → models see different things (good)
- ρ > 0.8 → models converge → ensemble adds no value
- Sudden ρ spike → both models may be overfitting to the same pattern

When ρ spikes: (1) check if a dominant factor (e.g., momentum) is driving both, (2) consider dropping the redundant model, (3) determine if the high ρ is regime-driven (temporary) or structural convergence (permanent).

### 1.12 Turnover diagnostics

**Definition (one-way turnover, summed over the union of stocks held in either month):**

```
TO_t = (1/2) × Σ_{i ∈ U_t} | w_{i,t} − w_{i,t-1} |
where U_t = { i : w_{i,t-1} ≠ 0 OR w_{i,t} ≠ 0 }
```

- New stock entering: `w_{i,t-1}=0` ⇒ contribution = `w_{i,t}`
- Old stock exiting: `w_{i,t}=0` ⇒ contribution = `w_{i,t-1}`
- Factor 1/2 avoids double counting (one sale funds one buy)

**Worked example, 5-stock universe (slide 25):** D exits, E enters.
| Stock | A | B | C | D | E | Sum |
|---|---|---|---|---|---|---|
| Last w_{t-1} | 0.40 | 0.30 | 0.20 | 0.10 | 0.00 | 1.00 |
| This w_t | 0.25 | 0.30 | 0.25 | 0.00 | 0.20 | 1.00 |
| `|Δw|` | 0.15 | 0.00 | 0.05 | 0.10 (exit) | 0.20 (entry) | 0.50 |

`TO_t = ½ × 0.50 = 0.25 = 25%` of capital traded.

**Annualization (monthly rebalance):** `TO_annual = 12 × TO_monthly_avg`. Example: 40% monthly avg → 480% annual (the portfolio's value is traded 4.8× per year, one-way).

**Long-short turnover:** sum across both legs separately:
```
TO_t^LS = ½ Σ_{i ∈ long} |w_{i,t} − w_{i,t-1}| + ½ Σ_{i ∈ short} |w_{i,t} − w_{i,t-1}|
```

**Healthy turnover:** 30–60% per month for monthly alpha. **Stale signal:** TO < 10%/month → model is coasting; either retrain or increase retraining frequency.

**Empirical (HGBR top-K long-only):** avg monthly TO = 41.9% → annualized 503%. Min/Max monthly: 20% / 70%. Mean stays comfortably above the 10% stale threshold; briefly crosses the 60% high-turnover threshold during regime shifts (2009, 2020).

### 1.13 Cost-aware alpha — the Required IC

**Cost in Sharpe units:**
```
Cost_annual ≈ TO_annual × c        (c = round-trip cost per dollar traded)
Cost SR     = Cost_annual / σ_p
```

**Required IC** (back out from Grinold's Law):
```
IC_required = (SR_target + Cost SR) / √BR
```

**Numerical example (slide 27).** TO = 600%, c = 10 bps, σ_p = 5%, BR = 360, target net SR = 0.5:
- Cost_annual = 6.0 × 0.0010 = 0.6%
- Cost SR = 0.6% / 5% = 0.12
- IC_req = (0.5 + 0.12) / √360 = 0.62 / 19.0 ≈ **0.033**

**Decision rule:** match turnover to your IC level. Higher turnover → higher IC needed to clear costs. If your model's IC < required IC for your TO, **slow down**.

### 1.14 Alpha decay curve — choose the rebalance frequency

**Compute IC at multiple forward horizons** using the same forecast made at time t:
```
IC(h) = Spearman( R̂_{i,t+1}, R_{i,t+h} ),    h = 1, 2, ..., 12 months
```

- **Fast decay** (momentum-like): IC drops 50% by month 3 → rebalance monthly
- **Slow decay** (value-like): IC drops 50% by month 12 → rebalance quarterly
- **HGBR observed:** mostly momentum-driven, expect fast decay → monthly rebalance correct

**HGBR empirical decay curve (slide 30):** IC(1m) = 0.0248, IC(2m) = 0.0209, ..., IC(9m) = 0.0119 (half of IC(1m)). Half-life ≈ 9 months. Slow decay implies monthly cadence may be over-trading; quarterly rebalance could capture similar IC at lower turnover.

### 1.15 P&L attribution — predicted vs realized

**Predicted P&L** (deterministic given weights and predictions):
```
PnL_t^pred = Σ_i w_{i,t} · R̂_{i,t+1}
```

**Realized P&L** (random):
```
PnL_t^real = Σ_i w_{i,t} · R_{i,t+1}
```

**Monitor:** rolling correlation `corr(PnL^pred, PnL^real)` over time.

| Correlation | Interpretation |
|---|---|
| > 0.3 | Model is useful |
| ≈ 0 | Model is noise |
| < 0 | **Model is anti-predictive — danger** |
| Divergence between predicted and realized | Execution problem or model decay |

**Empirical (HGBR 2005–2023, slide 37):** full-period correlation = **−0.067**. Predicted mean/std: +0.118 / 0.033. Realized mean/std: +0.020 / 0.066. Mean shortfall: −0.098. Months by rolling 12-mo correlation: anti-predictive 122, weak 68, useful 32.

**Critical takeaway from this slide:** the prediction *magnitude* is severely biased upward (HGBR over-promises), but the *rank ordering* (used by the L/S top-K portfolio) still has positive Spearman IC. This is **why we use Spearman, not Pearson**, to evaluate alpha quality.

### 1.16 Execution shortfall — paper vs live returns

```
Shortfall_t = R_t^backtest − R_t^live
Shortfall̄^(W) = (1/W) Σ_{s=t-W+1}^t Shortfall_s
```

- W = 12 months for stable estimates, 6 months for faster detection
- Single bad month is noise; persistent positive shortfall = real degradation

**Sources:** market impact (10–30 bps/trade), timing (close vs intraday), slippage, data adjustments.

**Rule of thumb:** expect 20–50% SR reduction backtest → live. If `SR_live / SR_backtest < 0.5`, investigate execution.

**Worked example.** 12-mo avg backtest return +0.85%/mo, 12-mo avg live return +0.55%/mo, shortfall 0.30%/mo = 3.6%/year. Backtest SR = 1.20, live SR = 0.75 → ratio 0.63 (acceptable but watch).

**Diagnostic actions when shortfall grows:**
1. Has AUM grown? Larger trades → rising market impact (capacity issue)
2. Have other funds adopted similar signals? (crowding)
3. Compare execution venues, order types, fill quality
4. Re-estimate cost model with recent fills; rerun backtest with updated `c`

### 1.17 Monitoring dashboard — frequency table

| Frequency | What to monitor | Tool |
|---|---|---|
| Daily | P&L vs predicted P&L | Scatter plot |
| Daily | Portfolio factor exposures | Factor attribution |
| Weekly | Realized vol vs target vol | Vol targeting check |
| Monthly | IC (Spearman) | IC time series |
| Monthly | Feature importance stability | Rank correlation |
| Monthly | Cumulative IC trend | CumIC plot |
| Monthly | Feature distribution shift | KS test |
| Quarterly | Full model retrain | Walk-forward backtest |
| Annually | Strategy review | IC, SR, MDD, Calmar |

### 1.18 Daily monitoring — implementation details

**Issue:** predictions `R̂_{i,t+1}` are *monthly* forward returns; daily realized returns are not directly comparable. Two work-arounds:

**1. Cumulative MTD vs monthly prediction.** Hold weights `w_{i,t}` fixed through the month. Each day `d` within month `t`, compute MTD realized:
```
R_{p,d}^MTD = Σ_i w_{i,t} · ( Π_{s=1}^d (1 + r_{i,s}) − 1 )
```
Compare to monthly predicted `Σ_i w_{i,t} · R̂_{i,t+1}`. Alert if `R^MTD` deviates > 3σ from expected linear path.

**2. Factor exposures (daily).** Regress daily portfolio returns on daily Fama-French 5 + Momentum (rolling 60-day window). Track each β over time. Alert if any factor `|β|` exceeds policy limit (e.g., `|β_mkt| > 0.1` for market-neutral mandate).

**Other daily checks (no prediction comparison needed):**
- Position drift: actual weights vs target weights (catches corporate actions, missed fills)
- Realized vol: 20-day annualized vs target σ_p
- Data feed: all features non-NaN; price/return outliers

**Why daily?** Pipeline failures (missing data, broken feed, position drift) need to be caught **before** the next rebalance. Implementation: cron job at 5pm post-close → dashboard → alert email.

### 1.19 Monthly monitoring — implementation details

Run after each rebalance (5 metrics, ~10 minutes total):

1. **IC (Spearman)** — compute on this month's predictions vs realized returns; append to time series; check rolling 12-month mean and t-stat
2. **Feature importance stability** — pull `feature_importances_` from latest retrain; compute Spearman rank correlation vs previous retrain; flag if `ρ < 0.7`
3. **Cumulative IC trend** — plot `Σ_s IC_s` from inception; visual check for flattening or downturn
4. **Feature distribution shift (KS test)** — per-feature KS statistic between current month and trailing 36 months; alert if `> 20%` of features have `D > 0.10`
5. **Turnover** — `½ Σ_i |w_{i,t} − w_{i,t-1}|`; flag if outside 10%–60% range

**Output:** one-page PDF report with all 5 charts, emailed to PM/risk on the 1st business day of each month. Status traffic-lights: **green** (all OK), **amber** (1–2 yellow flags), **red** (any red flag).

### 1.20 Quarterly and annual reviews

**Quarterly — full retrain.** Run walk-forward backtest with all data up to last quarter-end. Re-tune hyperparameters via CV (don't trust last year's tuning). Compare new model vs current production: out-of-sample IC, SR, hit rate. **Promote new model only if it beats production by a clear margin (e.g., ΔIC > 0.005).** Document what changed, why, validation results.

**Annual — strategy review.** Full performance attribution: IC, SR, MDD, Calmar, hit rate. Compare to peer strategies and benchmarks. Capacity analysis (AUM vs market impact). Re-evaluate cost model with current fills. Stress test under regime changes (rate hikes, COVID-style shocks). **Decision: continue / scale / wind down.**

### 1.21 Automated alerts

**Set alerts to fire automatically. Don't check manually.**

| Alert | Trigger | Severity | Action |
|---|---|---|---|
| IC decay | Rolling 6-mo IC < 0.02 | Warning | Schedule retrain |
| IC collapse | Monthly IC < −0.03 | **Critical** | Stop trading |
| Uniform signal | All universe predictions same sign | **Critical** | Pipeline broken |
| Feature NaN | > 20% NaN in any feature | Warning | Check data feed |
| Zero turnover | Model picks same stocks 3 months | Warning | Model may be stale |

**Code pattern (slide 45):**
```python
# After computing IC each month:
if rolling_ic_6m < 0.02:
    send_alert('IC Decay', f'Rolling IC = {rolling_ic_6m:.3f}')

if (predictions > 0).all() or (predictions < 0).all():
    send_alert('CRITICAL: Uniform Signal', 'All predictions same sign!')
```

**Critical alerts (IC collapse, uniform signal) mean stop trading immediately. Warning alerts mean investigate within 24 hours.**

### 1.22 Walk-forward alert simulation — empirical numbers

Slide 46, HGBR 2005–2023, 227 months simulated:
- **Total alerts fired: 323**
  - IC collapse (CRITICAL): 61
  - IC decay (WARN): 106
  - OOD shift (WARN): 156
- Silent months: 35 / 227 (15.4%)
- Months with ≥1 alert: 192 / 227

Tall stacks (3 alerts/month) cluster near regime breaks (2008–09, 2020, 2022). Persistent post-2018 alert frequency confirms post-COVID alpha decay also visible in cumulative IC.

### 1.23 The model lifecycle — five takeaways

```
Build (Module 3.3) → Deploy (this section's Section 2) → Monitor (this section)
        ↑                                                       │
        │  new data                                   IC < 0    │
        └─── Retrain (when IC decays) ←──────────────────────────┘
```

1. IC is the single most important monitoring metric.
2. Model decay is inevitable — plan for it.
3. Retrain on schedule + triggers, never on a single bad month.
4. Feature importance drift is an early warning of structural change.
5. CumIC is the alpha model's equity curve — watch it daily.

---

## SECTION 2 — Portfolio Construction Methods

The pipeline so far: features `X_{i,t}` → ML model `f_θ` → predictions `R̂_{i,t+1}` for every stock i. Today's question: given predictions, **how do we form the optimal portfolio?**

**Key takeaway up front:** alpha generation and portfolio construction are **separate problems**. A great signal can be destroyed by bad portfolio construction.

### 2.1 What Lecture 3.3 ignored

Lecture 3.3's portfolio rule:
- Pick top-K stocks by R̂_i
- Equal-weight: `w_i = 1/K` for `i ∈ Top-K`
- Compute portfolio return: `R_p = Σ_i w_i R_{i,t+1}`

What this ignores:
1. **Conviction levels** — stock ranked #1 gets the same weight as stock #K
2. **Correlations** — two highly correlated picks provide less diversification
3. **Risk** — no control on portfolio volatility or tail risk
4. **Constraints** — no position limits, sector caps, or turnover control

### 2.2 Ingredients for optimal portfolio construction

1. **Expected returns** `μ̂ = (R̂_1, ..., R̂_N)'` ← from ML model (Module 3.3)
2. **Covariance matrix** `Σ̂` ← from historical returns (Module 3.2)
3. **Constraints** ← long-only, position limits, sector caps, etc.

**Progression covered in this lecture:**
```
Equal-Weight  →  Score-Weighted  →  Mean-Variance  →  MAXSER
   (simplest)        (uses μ̂)        (uses μ̂, Σ̂)   (regression form)
                                          ↓
                                  Constrained MV  →  Risk Parity
                                  (add constraints)   (Σ̂ only, no μ̂)
```

- **MAXSER** (Maximum Sharpe via Regression) bypasses Σ̂⁻¹ — uses Lasso/Ridge on returns directly. Handles `N ≈ T`.
- **Risk Parity** drops μ̂ entirely; only uses risk information (Σ̂).

### 2.3 Equal-Weight Top-K

**Rule:** Each month, select the K stocks with the highest R̂_i, assign equal weight.
```
w_i = 1/K  if R̂_i ∈ Top-K, else 0
```

**Properties:**
- Maximum diversification *within* the selected set
- No dependence on the *scale* of R̂_i — only ranks matter
- Robust to prediction noise (stock #1 and stock #K get the same weight)

**Choice of K:**
| K | Concentration | Character |
|---|---|---|
| 5 | Very high | Best ideas, high tracking error |
| 10 | High | Focused conviction |
| 20 | Moderate | Balanced diversification |
| 50 | Low | Index-like with tilt |

### 2.4 Score-Weighted Top-K

**Rule:** weight proportional to predicted return.
```
w_i = R̂_i / Σ_{j ∈ Top-K} R̂_j,    i ∈ Top-K
```

**Compared to equal-weight:**
- Exploits the *magnitude* of predictions, not just ranks
- Allocates more capital to highest-conviction picks
- If predictions are well-calibrated: higher expected return
- **If predictions are noisy: concentrates on noise**

**Practical note:** for tree-based models (HGBR), predicted magnitudes are compressed — score-weighted often looks similar to equal-weight in practice.

### 2.5 Equal-Weight vs Score-Weighted — comparison

| Criterion | Equal-Weight | Score-Weighted |
|---|---|---|
| Simplicity | ✓✓ | ✓ |
| Diversification | Maximum in set | Lower (tilted) |
| Robust to noise | ✓✓ | Sensitive |
| Uses conviction | No | Yes |
| Turnover | Lower (only trade entries/exits) | Higher (weights shift with scores) |
| SR (empirical) | Solid baseline | Depends on calibration |

**Empirical L/S comparison (slide 54), 2005–2023:**
- EW L/S: SR = 0.97, MDD = −29% — strong baseline
- Score-Weighted L/S: SR = 0.93, MDD = −32% — slight degradation; concentrating in high-conviction picks slightly hurts

**Equal-weight is a strong baseline. Score-weighting helps only if prediction magnitudes are meaningful, not just ranks.**

### 2.6 Mean-Variance Optimization (MVO)

**Goal:** find weights `w_1, ..., w_N` that give the best tradeoff between expected return and risk.

```
max_w  w'μ̂ − (δ/2) w'Σ̂w
       └─return┘   └─risk┘
s.t.   Σ_i w_i = 1
```

`δ` = risk aversion parameter. Higher δ → more risk-averse → more diversified.

**The optimizer rewards stocks that:**
1. Have high predicted return μ̂_i → more weight
2. Have low volatility σ_i → more weight
3. Have low correlation with other holdings → more weight (diversification)

The optimizer balances all three **simultaneously**.

**The catch.** Estimating Σ̂ requires the covariance σ_ij between every pair of stocks. For N=60, that's `60 × 59 / 2 = 1,770` numbers. Estimation error is huge.

**Closed-form solution (ignoring the budget constraint):**

Take ∂/∂w of objective and set to zero:
```
∂/∂w (w'μ − (δ/2) w'Σw) = μ − δΣw = 0
```
Solve:
```
δΣw = μ  ⇒  w* = (1/δ) Σ⁻¹μ
```

**In plain English:**
- μ = predicted returns (which stocks are good?)
- Σ⁻¹ = inverse covariance (accounts for correlations and volatilities)
- Σ⁻¹μ = "give more weight to high-return stocks, **adjusted for** how correlated and volatile they are"
- 1/δ = how aggressively to bet (lower risk aversion → larger positions)

`w* ∝ Σ⁻¹μ` is the fundamental formula of portfolio theory. **The problem is that Σ⁻¹ amplifies estimation errors.** This is why we need Ledoit-Wolf shrinkage or MAXSER.

### 2.7 MVO 3-stock numerical example

3 stocks: NVDA (μ̂ = +2.0%, σ = 8%, correlation ≈ 0.6 with AAPL), AAPL (μ̂ = +1.5%, σ = 5%, moderate), XOM (μ̂ = +1.0%, σ = 6%, ρ ≈ 0.1 with tech).

- **EW gives:** w = (1/3, 1/3, 1/3). Ignores predictions and correlations.
- **MVO gives:** w ≈ (0.25, 0.45, 0.30).
  - AAPL gets the most: good return + low vol
  - NVDA gets less than expected: high return but high vol and correlated with AAPL
  - XOM gets a boost: lower return but **uncorrelated** with tech → diversification value

**MVO rewards stocks that add return AND reduce portfolio risk. Uncorrelated stocks are more valuable than their return alone suggests.**

### 2.8 The Optimal Portfolio — intuition

The MVO solution gives more weight to stocks that:
1. Have high predicted return μ̂_i
2. Have low volatility σ_i
3. Have low correlation with other holdings (diversification)

**Result:** MVO produces the portfolio with the **highest Sharpe ratio** among all possible weight combinations. (The textbook name is the "tangency portfolio" because it sits where the capital market line touches the efficient frontier.)

**Why it's the gold standard:**
- Provably the best risk-return tradeoff (given accurate inputs)
- Foundation of all portfolio theory

**Why it fails in practice.** The optimizer is only as good as its inputs. Even small errors in predicted returns or covariances are **amplified** by the inverse-covariance operation, producing extreme, unstable weights. This is why constraints are essential.

### 2.9 MVO empirical disaster — lab results

Unconstrained MVO on the HGBR L/S top-K portfolio, 2005–2023:
- **Unconstrained MVO: SR = 0.14** — much worse than EW (SR = 0.97)!
- MVO is extremely sensitive to covariance estimation errors
- The optimizer "chases" noise → unstable, poor OOS performance

Cumulative wealth: MVO ends roughly flat (≈$1.2 from $1) while EW L/S grows to ≈$13. This is the classic Markowitz error-maximization phenomenon.

### 2.10 The estimation problem — why raw MV fails

For N stocks and T months of history, we need to estimate `N(N+1)/2` parameters in Σ̂.

- S&P 500: N = 500 ⇒ **125,250 parameters**
- With T = 60 months of data: severely underdetermined
- When N > T: sample covariance matrix is **singular** (Σ̂ not invertible)
- Even when N < T: Σ̂⁻¹ amplifies estimation errors

**Consequences:**
- Extreme long/short positions (some weights ≫ 100%)
- Portfolio dominated by estimation error, not by true signal
- Markowitz "error-maximizing" property (Michaud, 1989): the optimizer **loves** the most mismeasured inputs

**Solution:** regularize Σ̂ before inverting.

### 2.11 Ledoit-Wolf shrinkage — fixing noisy covariance

**The problem:** sample covariance is noisy (too many parameters, too little data).
**The fix:** blend it with a simple, stable estimate.

```
Σ̂_shrunk = (1 − δ) · Σ̂_sample  +  δ · diag(Σ̂_sample)
            └trust data┘            └trust structure┘
```

- `δ ∈ [0, 1]`: shrinkage intensity (computed automatically by the Ledoit-Wolf formula)
- `δ = 0`: full sample covariance (noisy but unbiased)
- `δ = 1`: diagonal only (stable but ignores all correlations)
- Optimal δ is in between — keeps strong correlations, shrinks noisy ones

**Why it works:** pulls extreme covariance estimates toward zero ⇒ MVO weights become stable.

**Use Ledoit-Wolf whenever `N/T > 0.1`. Never invert raw sample covariance for optimization. In Python:**
```python
from sklearn.covariance import LedoitWolf
Sigma_hat = LedoitWolf().fit(returns).covariance_
```

### 2.12 MAXSER — Maximum Sharpe via Regression

**The clever trick:** find the max-Sharpe portfolio by regressing **1** on returns.

```
min_{w̃_1,...,w̃_N}  (1/T) Σ_t ( 1 − w̃_1 R_{1,t} − w̃_2 R_{2,t} − ... − w̃_N R_{N,t} )²
```

**Important:** the raw weights `w̃_i` do **not** sum to 1. To get actual portfolio weights:
```
w_i = w̃_i / Σ_j w̃_j
```

**Why it works.** The OLS solution is `ŵ = (X'X)⁻¹X'Y` with `Y = 1` and `X = R_t`. Compute:
- `(1/T) X'X = (1/T) Σ_t R_t R_t' ≈ Σ + μμ'` (second moment of returns)
- `(1/T) X'Y = (1/T) Σ_t R_t · 1 = μ` (average return)
- When returns are small: `Σ + μμ' ≈ Σ` (μμ' is negligible)

Therefore `ŵ ≈ Σ⁻¹μ` — **exactly the MVO tangency portfolio**, but obtained from a simple regression with no separate Σ estimation or inversion.

**Add Ridge penalty** `+λ Σ_i w̃_i²` for stabilization (prevents extreme positions). With only K=30 pre-selected stocks, Ridge is sufficient — no need for LASSO sparsity. Use `positive=True` for long-only.

**Lab implementation:** use top K = 30 stocks from HGBR, long-only (`w_i ≥ 0`). MAXSER turns portfolio optimization into a regression problem. Normalize w̃ to get actual weights.

### 2.13 MAXSER empirical results

Slide 64, lab results:
- **MAXSER Long: SR = 0.87** — optimal weighting within HGBR's top 30 stocks
- Beats Constrained MVO (SR = 0.53) and SPY (SR = 0.66)
- Close to EW L/S (SR = 0.97) but long-only — different risk profile

### 2.14 Long-only constraint

```
w_i ≥ 0 for all i
```

**Why:**
- Many institutional investors cannot short
- Short selling is costly (borrow fees, short squeezes, unlimited downside)
- Simplifies execution and risk management

**Impact on optimization:**
- **No closed-form solution** — must use numerical optimizer
- The efficient frontier shifts *inward* (lower SR achievable)
- Stocks with negative R̂_i get zero weight (useful regularization)

**Practical "clip-and-renormalize" heuristic:**
```
1. Compute unconstrained w* = Σ̂⁻¹μ̂
2. Set negative weights to zero
3. Renormalize: w_i ← w_i / Σ_j w_j
```

This is a rough approximation; for exact solution use a quadratic programming solver (e.g., `cvxpy`).

### 2.15 Position limits

```
w_i ≤ w_max for all i  (e.g., w_max = 5%)
```

**Why:**
- Prevent concentration in a single stock
- Regulatory limits (e.g., mutual funds: 5% rule for "diversified" funds)
- Risk management: no single stock can blow up the portfolio

**Combined with long-only:**
```
0 ≤ w_i ≤ w_max,    Σ_i w_i = 1
```

**Effect:** spreads weight more evenly. The constrained MV portfolio moves *toward* equal-weight as `w_max → 1/N`.

### 2.16 Constrained MVO empirical results

Constrained MVO (position limits + dollar neutral) on HGBR top-K, 2005–2023:
- **Constrained MVO: SR = 0.53, MDD = −28%**
- Much better than unconstrained MVO (SR = 0.14) — constraints act as regularization
- Still below EW L/S (SR = 0.97) — optimizer struggles with noisy covariance inputs

### 2.17 Summary: portfolio construction methods

| Property | Equal-Weight | Score-Wt | MV | Constrained MV | MAXSER | Risk Parity |
|---|---|---|---|---|---|---|
| Uses μ̂? | Ranks only | Magnitudes | Yes | Yes | Yes | **No** |
| Uses Σ̂? | No | No | Yes | Yes | Implicit | Yes |
| Closed-form? | Yes | Yes | Yes | Numerical | Regression | Numerical |
| Robust to noise? | High | Medium | **Low** | Medium-High | Medium | High |
| Diversification? | Max in set | Moderate | Can be poor | Controlled | Max-Sharpe | Equal risk |
| Turnover? | Low | Medium | High | Medium | Medium | Low |
| Implementation cost? | Trivial | Trivial | Moderate | Moderate | Low | Moderate |

**Empirical regularity (DeMiguel, Garlappi, Uppal, 2009):** equal-weight (1/N) is hard to beat OOS for most asset classes and sample sizes. The cited finding from the paper is that of 14 models evaluated, none consistently outperformed the 1/N rule.

### 2.18 Risk Parity — portfolios without alpha forecasts

**Problem:** expected return forecasts are noisy. Can we build portfolios using *only* risk information?

**Risk contribution.** For asset `i`, define:
```
(Σw)_i = Σ_{j=1}^N σ_{ij} w_j = Cov(R_i, R_p)
```
i.e., the covariance of asset i's return with the *whole portfolio's* return — how much asset i "co-moves" with the portfolio.

**Mini-example** (N=2, σ_1=20%, σ_2=10%, ρ_12 = 0.3, w = (0.5, 0.5)):
```
Σ = [[0.040, 0.006],
     [0.006, 0.010]]

Σw = [0.040(0.5) + 0.006(0.5),    [0.023,
      0.006(0.5) + 0.010(0.5)]  =  0.008]
```

**Risk Contribution** (share of total variance, sums to 1):
```
RC_i = w_i · (Σw)_i / σ_p² ,    Σ_i RC_i = 1
```

Continuing the example: `σ_p² = w'Σw = 0.5(0.023) + 0.5(0.008) = 0.0155`.
- RC_1 = (0.5 × 0.023) / 0.0155 = **74%**
- RC_2 = (0.5 × 0.008) / 0.0155 = **26%**

**Equal weights ≠ equal risk!** The high-vol asset dominates portfolio risk.

**Equal Risk Contribution (ERC):** choose w so that `RC_i = 1/N` for all i.

**Special case ρ = 0 (uncorrelated assets):** ERC condition reduces to `w_i² σ_i² = constant`, i.e.:
```
w_i ∝ 1 / σ_i      (inverse-vol weighting)
```
Higher-vol assets get less weight.

### 2.19 Risk Parity 2-asset numerical example

Stock (σ = 20%) and Bond (σ = 5%), correlation ρ = 0.

| | 60/40 Portfolio | Risk Parity |
|---|---|---|
| w_stock | 60% | 20% |
| w_bond | 40% | 80% |
| Portfolio variance σ_p² | 0.6²(0.20)² + 0.4²(0.05)² = 0.0148 | 0.2²(0.20)² + 0.8²(0.05)² = 0.0032 |
| Portfolio vol σ_p | √0.0148 = 12.2% | √0.0032 = 5.7% |
| Stock RC | 0.36×0.04 / 0.0148 = **97%** | 0.04×0.04 / 0.0032 = **50%** |
| Bond RC | 0.16×0.0025 / 0.0148 = 3% | 50% |
| Check (sum) | 97% + 3% = 100% ✓ | 50% + 50% = 100% ✓ |

In 60/40, **stock dominates risk (97%)** — bonds are decorative. Risk parity (w_s = 20%, w_b = 80%) equalizes risk contributions to 50/50. Lower vol (5.7% vs 12.2%) but needs **leverage (~2×)** to match return.

### 2.20 Risk Parity — drawdown control

**Advantages:**
- No noisy return forecast required
- Stable weights over time → low turnover
- Genuinely diversified across risk sources
- **Lower drawdowns** than concentrated portfolios

**Disadvantages:**
- Ignores alpha — purely risk-based
- Requires **leverage** for adequate return
- Fails when correlations spike (2022: bonds + equities fell together)

**Bridgewater All Weather (Ray Dalio, $80bn+).** Allocates risk equally across growth, recession, inflation, deflation. Exceptional performance 2000–2019. Then **2022: all four scenarios failed simultaneously** — a reminder that risk parity is not universal.

For our 5% drawdown goal: risk parity naturally limits drawdowns by preventing any single position from dominating risk. Combined with volatility targeting (Module 4.2), this is the foundation.

### 2.21 Risk Parity empirical results

Slide 73, lab:
- **Risk Parity Long: SR = 0.82** — competitive without any alpha forecast
- MAXSER Long: SR = 0.87 — optimal weighting adds a small edge
- EW L/S: SR = 0.97 — still the strongest (L/S captures short-side alpha too)

### 2.22 Grand comparison of all portfolio methods

Slide 75, full performance table on HGBR universe 2005–2023:

| Strategy | SR | Ann. Mean | Ann. Vol | MDD | Calmar |
|---|---|---|---|---|---|
| **MAXSER Long** | **0.91** | 23.3% | 25.6% | −41% | 0.58 |
| Risk Parity Long | 0.85 | 17.8% | 20.9% | −58% | 0.31 |
| EW L/S | 0.80 | 12.3% | 15.4% | −29% | 0.43 |
| Score-Weighted L/S | 0.76 | 13.1% | 17.3% | −32% | 0.41 |
| SPY Buy & Hold | 0.66 | 10.7% | 16.1% | −48% | 0.22 |
| Constrained MVO | 0.54 | 8.1% | 15.1% | −28% | 0.29 |
| MVO (unconstrained) | 0.17 | 1.7% | 10.2% | −26% | 0.07 |

**Key findings:**
- **MAXSER wins on SR (0.91)** — Σ⁻¹μ weighting within top-K stocks
- EW L/S (0.80) has the **lowest MDD (−29%)** — best Calmar among L/S
- **Unconstrained MVO fails (SR = 0.17)** — estimation error dominates
- **All methods have MDD > 25%** — none achieves the 5% drawdown target
- **Next (Module 4.2):** volatility targeting and Kelly to control drawdown

### 2.23 Drawdown comparison — all methods

The 5% drawdown target is the red dashed line. **No method stays below it.** All strategies experience 20–60% drawdowns at some point. Module 4.2 addresses this with volatility targeting and dynamic leverage.

---

## SECTION 3 — Implementation: Costs and Production

### 3.1 Why transaction costs wipe out alpha

Every trade costs:

| Cost component | What it is | Typical size |
|---|---|---|
| Bid-ask spread | Buy at ask, sell at bid | 3–10 bps |
| Market impact | Your trade moves the price | 5–20 bps |
| Commission | Broker fee | 1–5 bps |
| **Total round-trip** | Buy + sell | **~10–30 bps** |

1 bps = 0.01%. So 10 bps = 0.10% per round-trip trade.

**The arithmetic of cost drag:**
- Turnover = fraction of portfolio replaced per year. 200% = entire portfolio replaced twice.
- Annual cost drag = turnover × cost per trade = 200% × 10 bps = 200 bps = **2%/year**
- HGBR strategy earns ~15%/year gross. After 2% costs: ~13%/year net.
- A weaker strategy earning 4%/year gross → 2% net → **barely worth running**.

**The brutal truth.** Many strategies look great in backtests (before costs) and terrible live (after costs). TC modeling is **not optional** — it determines whether a strategy is profitable.

### 3.2 The TC model — two formulations

**Linear cost model (simple):**
```
TC_i = c_i × |Δw_i| × AUM
```
- `c_i` = cost per dollar traded (5–15 bps)
- `Δw_i = w_{i,new} − w_{i,old}`
- Total cost = Σ_i c_i |Δw_i|
- Good approximation for **small trades**

**What determines c_i:**
- Bid-ask spread (wider for illiquid stocks)
- Market cap: large-cap ≈ 5 bps, small-cap ≈ 30 bps
- Volume (low-volume stocks cost more)

**Square-root model (for large trades):**
```
TC_i = σ_i × √( |q_i| / ADV_i )
```
- σ_i = daily volatility of stock i
- q_i = number of shares traded
- ADV_i = average daily volume
- When you trade 10% of ADV, price moves against you

**Numerical example:** stock with σ = 2%/day, you trade 5% of ADV:
```
TC = 0.02 × √0.05 = 0.02 × 0.22 = 0.45%
```
Much larger than the 10 bps linear estimate.

**Decision rule:** linear model for diversified portfolios (many small positions). Square-root model for concentrated portfolios or large AUM where trades move prices.

### 3.3 Controlling transaction costs

How to reduce cost drag without destroying the signal:

**1. Turnover penalty in the optimizer:**
```
max_w  w'α̂  −  (δ/2) w'Σw  −  λ Σ_i c_i |w_i − w_{i,t-1}|
       └alpha┘   └risk┘         └─trading cost──┘
```
Larger λ = fewer trades = lower costs but staler positions.

**2. Rebalance less frequently:** monthly instead of daily. Trade-off: alpha signal decays.

**3. Buffer zones:** only trade stock i if `|w_{i,target} − w_{i,current}| > ε`. Ignore small deviations.

**4. Use liquid stocks only.** The lab dataset filters to large-cap (`ME ≥ $1B`) which have low trading costs.

### 3.4 Break-even IC — cost-aware version

Minimum signal quality to cover costs:
```
IC* = TC_ann / ( σ_p × √BR )
```
BR = breadth = number of independent bets (stocks traded) per year.

**Example:** σ_p = 10%, BR = 500 stocks, TC_ann = 2%:
```
IC* = 0.02 / (0.10 × √500) = 0.02 / 2.236 ≈ 0.009
```
Any IC above 0.9% is profitable after costs.

### 3.5 The complete quant pipeline

```
[Module 3.1–3.2]                   [Module 3.3]
β̂_{i,t}, Σ                          α̂_{i,t} (HGBR model)
     \                              /
      v                            v
   +--------------------+
   |  Portfolio         |  ←── feedback (drawdown)
   |  Optimizer         |
   |  (this lecture)    |
   +--------------------+
            |
            v
       Weights w_t
            |
            v
   +--------------------+    +-------------------+
   |  Execution         |    |  Risk monitor     |
   |  + TC model        |    |  drawdown < 5%?   |
   +--------------------+    +-------------------+
```

The optimizer takes alpha scores from Module 3.3 and covariance from Module 3.2, applies constraints (factor neutrality, position limits, turnover), and outputs weights. The risk monitor checks drawdown daily.

### 3.6 Walk-forward rebalancing timeline

```
   Train window           Optimize  Hold period
[─────────────────────][trade ─────────────────────]
t-252                  t-1   t                    t+h
```

1. Use data up to `t-1` to estimate α̂_{i,t}, Σ_t.
2. Solve constrained optimization at `t` ⇒ target weights `w_t*`.
3. Execute trades. Hold until next rebalance (`t+h`, monthly in our case).
4. **Never look ahead:** no data from `[t, t+h]` in step 1.

### 3.7 Live monitoring — the 5% drawdown target

Production portfolio requires daily monitoring:

| Metric | Target | Action if breached |
|---|---|---|
| Portfolio vol (20d realized) | Within risk budget | Reduce position sizes |
| **Drawdown** | **< 5%** | **Cut exposure by 50%** |
| Factor exposures B'w | ≈ 0 for all factors | Rebalance |
| Gross leverage `‖w‖_1` | ≤ L_max | Reduce positions |
| Daily turnover | ≤ TO budget | Widen buffer zone |

**The drawdown-return tradeoff.** Our HGBR L/S has SR = 0.97 but MDD = −29%. With a 5% drawdown limit, we must reduce leverage ⇒ lower return but survivable. **Module 4.2** covers **how**: Kelly criterion, volatility targeting, dynamic leverage.

### 3.8 Conclusion — Module 3 → Module 4.1 → Module 4.2

**What Module 3 built:**
- 3.1–3.2: Factor betas β̂_{i,t} and covariance Σ
- 3.3: Alpha model α̂_{i,t} via HGBR (SR = 0.97)
- Problem: MDD = −29% — too large to lever up

**What Module 4.1 adds (this lecture):**
- Turn (α̂, Σ) into optimal weights
- Mean-variance with practical constraints
- Risk parity as the forecast-free alternative
- Transaction cost integration
- Production pipeline with 5% drawdown target

**Next: Module 4.2 (NOT in this deck):**
- Kelly criterion: optimal bet sizing
- Volatility targeting: scale exposure to maintain constant risk
- Dynamic leverage: lever up in calm markets, delever in stress
- Goal: achieve 5% max drawdown while preserving most of the SR = 0.97 alpha signal

### 3.9 Fractional shares and small account sizing

**Problem:** with $100K and 30 stocks, target = $3,333 per stock. If stock price = $480 (e.g., MSFT), you can only buy 6 or 7 shares — **13% rounding error**.

**Solution 1: Fractional shares.** IBKR, Alpaca, Schwab support fractional shares. Buy exactly $3,333.33 of MSFT = 6.944 shares. Eliminates rounding error entirely. **Recommended for accounts < $500K.**

**Solution 2: Round intelligently.** Round to nearest whole share. Allocate leftover cash to the highest-conviction stock. Accept ~5–10% tracking error vs target weights. Only viable for accounts > $500K.

**Minimum viable capital.** With 30 stocks at $50–500/share, you need **≥$50K** for whole shares to work. Below $50K, fractional shares are mandatory or reduce to 15–20 stocks.

### 3.10 Minimum trade filter — when NOT to trade

**Small trades cost more than they're worth.** Before executing any trade, check:
```
Expected alpha gain > Round-trip transaction cost
```

**Example:** stock weight changes from 3.0% to 3.2%.
- Trade size = 0.2% × $100K = $200
- Commission = $1 (IBKR) + spread cost ≈ $0.50 = $1.50 round-trip
- Expected alpha on $200 for one month ≈ 0.2% × 1% = $0.004
- **Cost ($1.50) ≫ expected gain ($0.004). Don't trade.**

**Practical rule:**

| Account size | Minimum weight change to trade |
|---|---|
| $50K–$100K | `|Δw| > 1.0%` |
| $100K–$500K | `|Δw| > 0.5%` |
| $500K–$1M | `|Δw| > 0.3%` |

This is the buffer zone (`ε`) that should be implemented in the rebalance routine.

---

## APPENDIX A — Implementation roadmap for the alpha forward pass model

This is a recommended build order. It reorganizes the lecture content into the sequence Claude Code should implement.

### A.1 Required inputs (assumed from Module 3)
- `predictions_df`: DataFrame indexed by `(date, ticker)` with column `alpha` = α̂_{i,t}, monthly
- `returns_df`: DataFrame of realized monthly stock returns
- `features_df`: DataFrame of features used by the model (for KS test, feature-level IC)
- `model`: trained HGBR/RF estimator with `feature_importances_` attribute

### A.2 Core forward-pass components (build order)

**Step 1 — IC computation.**
```python
def compute_ic_t(pred_t: pd.Series, real_t: pd.Series) -> float:
    """Cross-sectional Spearman IC for one month."""
    df = pd.concat([pred_t, real_t], axis=1).dropna()
    return df.corr(method='spearman').iloc[0, 1]

def ic_time_series(predictions, realized) -> pd.Series:
    return pd.Series({t: compute_ic_t(predictions.loc[t], realized.loc[t])
                      for t in predictions.index.get_level_values('date').unique()})
```

**Step 2 — IC summary metrics.**
```python
def ic_summary(ic_series: pd.Series) -> dict:
    T = len(ic_series)
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'icir': mean_ic / std_ic if std_ic > 0 else np.nan,
        't_stat': mean_ic / (std_ic / np.sqrt(T)) if std_ic > 0 else np.nan,
        'hit_rate': (ic_series > 0).mean(),
    }
```

**Step 3 — Equal-weight top-K (the baseline that's hard to beat).**
```python
def equal_weight_topK(pred_t: pd.Series, K: int, long_short: bool = False) -> pd.Series:
    if long_short:
        top = pred_t.nlargest(K).index
        bot = pred_t.nsmallest(K).index
        w = pd.Series(0.0, index=pred_t.index)
        w.loc[top] = +1.0 / K
        w.loc[bot] = -1.0 / K
        return w
    top = pred_t.nlargest(K).index
    w = pd.Series(0.0, index=pred_t.index)
    w.loc[top] = 1.0 / K
    return w
```

**Step 4 — Score-weighted top-K.**
```python
def score_weighted_topK(pred_t: pd.Series, K: int) -> pd.Series:
    top = pred_t.nlargest(K)
    w = pd.Series(0.0, index=pred_t.index)
    w.loc[top.index] = top / top.sum()
    return w
```

**Step 5 — Ledoit-Wolf shrinkage (always use this for Σ̂).**
```python
from sklearn.covariance import LedoitWolf

def shrunk_cov(returns_window: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(returns_window).covariance_
```

**Step 6 — Constrained MVO (long-only, position limits) via cvxpy.**
```python
import cvxpy as cp

def mvo_constrained(mu, Sigma, w_max=0.05, long_only=True, delta=5.0):
    N = len(mu)
    w = cp.Variable(N)
    obj = cp.Maximize(mu @ w - (delta/2) * cp.quad_form(w, Sigma))
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    cons.append(w <= w_max)
    cp.Problem(obj, cons).solve()
    return w.value
```

**Step 7 — MAXSER (Maximum Sharpe via Regression).**
```python
from sklearn.linear_model import Ridge

def maxser(returns_window: pd.DataFrame, alpha=1.0, long_only=True) -> np.ndarray:
    """Regress 1 on returns, normalize to sum to 1."""
    X = returns_window.values     # T x N
    y = np.ones(X.shape[0])       # T x 1
    if long_only:
        from sklearn.linear_model import Lasso
        # Ridge with positive=True is not built-in; use NNLS or constrained QP
        from scipy.optimize import nnls
        w_tilde, _ = nnls(X, y)
    else:
        w_tilde = Ridge(alpha=alpha, fit_intercept=False).fit(X, y).coef_
    return w_tilde / w_tilde.sum()
```

**Step 8 — Risk parity (equal risk contribution).**
```python
def risk_parity(Sigma: np.ndarray) -> np.ndarray:
    """Solve for w such that w_i (Σw)_i is constant across i."""
    N = Sigma.shape[0]
    w = cp.Variable(N, nonneg=True)
    # Use the Spinu/Maillard convex reformulation:
    log_w = cp.log(w)
    obj = cp.Minimize(0.5 * cp.quad_form(w, Sigma) - (1/N) * cp.sum(log_w))
    cp.Problem(obj).solve()
    return w.value / w.value.sum()
```

**Step 9 — Turnover and TC drag.**
```python
def turnover(w_new: pd.Series, w_old: pd.Series) -> float:
    aligned = w_new.align(w_old, fill_value=0.0)
    return 0.5 * (aligned[0] - aligned[1]).abs().sum()

def cost_drag_annual(monthly_TO_avg, c_bps=10):
    return 12 * monthly_TO_avg * (c_bps / 10000)
```

**Step 10 — Buffer zone (don't trade tiny moves).**
```python
def apply_buffer(w_target: pd.Series, w_current: pd.Series, eps: float = 0.005) -> pd.Series:
    """If |w_target - w_current| < eps, hold w_current."""
    diff = (w_target - w_current).abs()
    keep_old = diff < eps
    w_executed = w_target.copy()
    w_executed[keep_old] = w_current[keep_old]
    # Renormalize to sum to 1
    return w_executed / w_executed.sum() if w_executed.sum() > 0 else w_executed
```

### A.3 Monitoring components (build after the optimizer works)

**Component 1 — KS-based OOD detection** (slide 34 reproduced verbatim above in §1.10)

**Component 2 — Feature-level IC drift** (§1.9)

**Component 3 — Cumulative IC plot** (§1.6)

**Component 4 — Walk-forward alert engine** (§1.21)

**Component 5 — Daily MTD vs predicted comparison** (§1.18)

### A.4 Decision thresholds — single source of truth

The lecture is full of magic numbers. Centralize them in one config:

```python
THRESHOLDS = {
    # IC dashboard
    'ic_healthy_min': 0.02,
    'ic_warning_max': 0.02,
    'ic_collapse_max': -0.03,
    'hit_rate_healthy_min': 0.55,
    'hit_rate_warning_min': 0.45,
    'icir_healthy_min': 0.3,
    'consecutive_neg_ic_retrain': 4,

    # Turnover
    'turnover_stale_max': 0.10,
    'turnover_high_min': 0.60,

    # OOD / KS
    'ks_minor_max': 0.05,
    'ks_meaningful_min': 0.10,
    'ks_major_min': 0.20,
    'ood_feature_pct_alert': 0.20,

    # PnL attribution
    'pnl_corr_useful_min': 0.30,
    'pnl_corr_anti_max': 0.0,

    # Execution shortfall
    'sr_live_to_backtest_min': 0.5,

    # Position / risk
    'position_limit_max': 0.05,
    'drawdown_kill_switch': 0.05,
    'feature_importance_corr_min': 0.7,
    'drift_flag_delta_ic': 0.01,

    # Trade filter buffer (account-size dependent)
    'min_dw_50to100k': 0.01,
    'min_dw_100to500k': 0.005,
    'min_dw_500kto1m': 0.003,
}
```

### A.5 Key empirical anchors for unit tests

These are the exact numbers from the deck. Useful for validating that your implementation reproduces the lab.

- **HGBR alpha decay** (slide 30): IC(1m) = 0.0248, IC(9m) = 0.0119, half-life ≈ 9 months
- **HGBR turnover** (slide 28): mean monthly TO = 41.9%, min 20%, max 70%, annualized 503%
- **HGBR P&L attribution** (slide 37): full-period correlation predicted vs realized = −0.067; 122 anti-predictive months, 68 weak, 32 useful
- **HGBR alert simulation** (slide 46): 323 total alerts over 227 months; 61 critical (IC collapse), 106 IC decay, 156 OOD; 35 silent months
- **OOD worst month** (slide 35): 2009-02 with 57% features flagged
- **Required IC numerical example** (slide 27): TO=600%, c=10bps, σ_p=5%, BR=360, target net SR=0.5 → IC_req ≈ 0.033
- **Risk parity 2-asset** (slide 71): w_stock=20%, w_bond=80% gives RC = 50/50 with portfolio vol 5.7%
- **MVO 3-stock** (slide 57): NVDA, AAPL, XOM → MVO weights ≈ (0.25, 0.45, 0.30)
- **Grand comparison** (slide 75) — full table reproduced in §2.22; MAXSER Long wins on SR (0.91), EW L/S wins on MDD (−29%)
- **Fundamental Law model bake-off** (slide 10): FM IC=0.0218 / SR=0.71 / impliedBR=1061; LASSO 0.0225/0.67/886; RF 0.0208/0.91/1914; HGBR 0.0248/0.97/1531

### A.6 Things this deck does NOT cover (and you'll need elsewhere)

- **Kelly criterion** for optimal bet sizing — Module 4.2
- **Volatility targeting** to scale exposure dynamically — Module 4.2
- **Dynamic leverage** (lever up calm, delever stress) — Module 4.2
- Black-Litterman for blending priors with views
- Robust optimization (Goldfarb-Iyengar uncertainty sets)
- Hierarchical Risk Parity (López de Prado, 2016)
- Implementation shortfall execution algorithms (TWAP/VWAP/IS)
- Tax-aware rebalancing (wash-sale rule, harvesting)
- Multi-period dynamic programming optimization
- Liquidity adjustments for portfolios approaching capacity

If your forward-pass model needs any of those, you cannot get them from this lecture.

---

## APPENDIX B — Slide index (for cross-reference)

| Slides | Topic |
|---|---|
| 1–2 | Title, outline |
| 3 | Section 1 cover |
| 4 | IC definition |
| 5 | IC interpretation scale |
| 6 | Spearman vs Pearson worked example |
| 7 | Information Ratio |
| 8 | Fundamental Law of Active Management |
| 9 | Fundamental Law numerical example, BR_implied formula |
| 10 | Theoretical vs actual SR, model bake-off table |
| 11 | IC reporting metrics |
| 12 | Five empirical insights from IC-Wealth analysis |
| 13 | Model decay causes and half-lives |
| 14 | IC dashboard with thresholds |
| 15 | HGBR rolling and cumulative IC empirical chart |
| 16 | Cumulative IC visual diagnostic |
| 17 | When to retrain — decision framework |
| 18 | Wealth comparison: scheduled vs hybrid retrain |
| 19 | Feature importance: tree gain, permutation |
| 20 | Feature importance: feature-level IC |
| 21 | Feature importance drift table (synthetic) |
| 22 | Feature-level IC drift over time empirical chart |
| 23 | Feature-level IC drift summary table (real) |
| 24 | Turnover decay |
| 25 | Computing turnover step-by-step worked example |
| 26 | Turnover annualization, L/S, basis points definition |
| 27 | Required IC formula and numerical example |
| 28 | HGBR turnover empirical chart |
| 29 | Alpha decay curve concept |
| 30 | HGBR alpha decay empirical curve, half-life ≈ 9 months |
| 31 | Cross-model correlation diversification check |
| 32 | OOD detection overview |
| 33 | KS test primer with thresholds |
| 34 | KS test Python implementation |
| 35 | KS test empirical: HGBR 2005–2023 |
| 36 | Predicted vs realized P&L definitions |
| 37 | P&L attribution empirical: HGBR scatter and rolling correlation |
| 38 | Execution shortfall: paper vs live |
| 39 | Execution shortfall how to use |
| 40 | Monitoring dashboard cadence overview table |
| 41 | Daily monitoring implementation |
| 42 | Monthly monitoring implementation |
| 43 | Quarterly and annual monitoring |
| 44 | Model lifecycle diagram and 5 takeaways |
| 45 | Automated alerts table and code |
| 46 | Walk-forward alert simulation empirical |
| 47 | Section 2 cover |
| 48 | Where we are: alpha → portfolio |
| 49 | What Lecture 3.3 ignored |
| 50 | What portfolio theory says — three ingredients |
| 51 | Equal-Weight Top-K |
| 52 | Score-Weighted Top-K |
| 53 | EW vs Score-Weighted comparison table |
| 54 | EW vs Score-Weighted L/S empirical wealth |
| 55 | MVO big idea and objective |
| 56 | MVO closed-form derivation `w* ∝ Σ⁻¹μ` |
| 57 | MVO 3-stock numerical example |
| 58 | Optimal portfolio intuition (tangency portfolio) |
| 59 | MVO empirical disaster (SR = 0.14) |
| 60 | Estimation problem: why raw MV fails |
| 61 | Ledoit-Wolf shrinkage |
| 62 | MAXSER: Maximum Sharpe via Regression |
| 63 | MAXSER derivation showing ŵ ≈ Σ⁻¹μ |
| 64 | MAXSER empirical results (SR = 0.87 long-only) |
| 65 | Long-only constraint and clip-and-renormalize |
| 66 | Position limits |
| 67 | Constrained MVO empirical (SR = 0.53) |
| 68 | Summary table of construction methods, DGU 2009 reference |
| 69 | Risk Parity intro and (Σw)_i definition |
| 70 | Risk contribution and ERC definition |
| 71 | Risk Parity 2-asset numerical example (stock 20% / bond 80%) |
| 72 | Risk Parity advantages, disadvantages, Bridgewater All Weather |
| 73 | Risk Parity empirical (SR = 0.82) |
| 74 | Grand comparison cumulative wealth chart |
| 75 | Grand comparison performance table |
| 76 | Drawdown comparison — none below 5% |
| 77 | Section 3 cover |
| 78 | Why TC wipes out alpha |
| 79 | TC model: linear and square-root |
| 80 | Controlling TC: turnover penalty, buffers, break-even IC |
| 81 | Complete quant pipeline diagram |
| 82 | Walk-forward rebalancing timeline |
| 83 | Live monitoring 5% drawdown target |
| 84 | Conclusion: from alpha to controlled risk; preview Module 4.2 |
| 85 | Fractional shares and small account sizing |
| 86 | Minimum trade filter (when not to trade) |

---

*End of overview. Total source coverage: 86/86 slides.*
