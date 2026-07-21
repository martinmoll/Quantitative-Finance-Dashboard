"""All educational text for theory sections, organized by topic key."""

THEORY_CONTENT = {
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

    "capm_and_factors": r"""
**Capital Asset Pricing Model (CAPM)**

The CAPM says a stock's expected return is proportional to its market risk:

$$r_i - r_f = \alpha_i + \beta_i (r_m - r_f) + \epsilon_i$$

where, for stock $i$ over a given period:

- $r_i$ — the **stock's return**.
- $r_f$ — the **risk-free rate** (e.g. a short-term Treasury yield). The quantity $r_i - r_f$ is the stock's *excess* return: what it earned beyond parking cash safely.
- $r_m$ — the **market return** (the whole market's return). The quantity $r_m - r_f$ is the *market risk premium* — the market's excess return.
- $\beta_i$ — the stock's **sensitivity to the market**. $\beta > 1$ amplifies market moves; $\beta < 1$ dampens them.
- $\alpha_i$ — the **intercept**: the average excess return *not* explained by market exposure. A significant positive alpha means the stock outperforms its risk level (genuine skill, not just beta).
- $\epsilon_i$ — the **residual**: the stock-specific, idiosyncratic wiggle left over each period, assumed to average zero.

We test alpha significance using the t-statistic on the intercept. **Always use HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors** because financial returns exhibit time-varying volatility and serial correlation. Classical standard errors would be too small, making alphas look significant when they aren't.

**Fama-French 5-Factor Model (FF5)**

Extends CAPM with four additional risk factors:

$$r_i - r_f = \alpha_i + \beta_i^M (r_m - r_f) + \beta_i^S \, SMB + \beta_i^H \, HML + \beta_i^R \, RMW + \beta_i^C \, CMA + \epsilon_i$$

Here $r_i$, $r_f$, $r_m$, $\alpha_i$, and $\epsilon_i$ mean the same as in CAPM above. Each
$\beta_i$ (with a superscript for the factor) is a **factor loading** — how strongly stock $i$
moves with that factor, estimated as a regression slope. $SMB$, $HML$, $RMW$, and $CMA$ are the
four factors themselves, each a long-short return spread:

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

    "jensens_alpha": r"""
**Jensen's Alpha: Do You Actually Have Skill?**

A portfolio can beat the market simply by taking more risk — holding high-beta stocks,
tilting toward small-caps, or loading on value. That's not alpha. That's factor exposure
you could get cheaper with an ETF.

**The FF5 regression strips out known risk premiums:**

$$r_p - r_f = \alpha + \beta_M (r_m - r_f) + \beta_S \cdot SMB + \beta_H \cdot HML + \beta_R \cdot RMW + \beta_C \cdot CMA + \epsilon$$

Whatever return is left after removing all five factor exposures is **Jensen's alpha** ($\alpha$) —
the intercept of the regression. It represents the monthly return your portfolio earns that
*no combination of passive factor exposures can explain*.

**How to read the results:**

| Metric | What it means | What's good |
|--------|--------------|-------------|
| Annualized Alpha | $\alpha_{monthly} \times 12$ — your yearly excess return beyond factors | Positive |
| t-statistic | Signal-to-noise ratio of alpha. Accounts for sample size | > 2.0 |
| p-value | Probability that alpha = 0 and you got lucky | < 0.05 |
| R-squared | How much of your returns are explained by factor exposures | Context-dependent |

**The four scenarios:**

1. **Positive alpha, significant (t > 2)** — Genuine skill. Your model finds something
   the five factors don't capture. This is rare and valuable.

2. **Positive alpha, not significant (t < 2)** — Could be skill, could be luck. You don't have
   enough months, or the alpha is too noisy, to tell. More data or a more consistent strategy
   would help. Don't trust this alpha in production.

3. **Near-zero alpha, high R-squared** — Your portfolio is a factor portfolio in disguise.
   The returns come entirely from tilting toward known premiums (value, momentum, etc.).
   This isn't necessarily bad — you're harvesting factor premiums efficiently — but you're
   not adding skill beyond what a smart factor ETF could replicate.

4. **Negative alpha** — After adjusting for factor risk, you're underperforming.
   The factor bets in your portfolio would have done better without your stock selection.

**Common pitfalls:**

- **Overfitting alpha**: In-sample alpha from backtests is always inflated.
   Walk-forward (out-of-sample) alpha is what matters.
- **Alpha vs. Beta confusion**: A portfolio with 1.3 market beta and 15% return in a 12% market year
   looks great — but the expected return was $1.3 \times 12\% = 15.6\%$. The alpha is actually *negative*.
- **Survivorship bias**: If your universe only includes stocks that survived to today, alpha is
   inflated because you're excluding the ones that went to zero.
- **HAC standard errors**: We use Newey-West (HAC) standard errors because monthly returns are
   heteroskedastic and can be autocorrelated. Classical standard errors would overstate significance.

**A practical benchmark:** In academic studies, monthly alpha of 0.3-0.5% (3.6-6.0%/year) with
t-stat > 2 is considered strong. Most published strategies lose significance out of sample.
""",

    "alpha_significance": r"""
**Is It Really Alpha? Measuring Statistical Confidence**

A positive alpha number is the beginning of the question, not the answer. The
estimate $\hat{\alpha}$ is a *random variable* — run the same strategy on a
different sample of months and you'd get a different number. What matters is how
much of the estimate is signal and how much is sampling noise. This section covers
the machinery for deciding how much to trust an alpha, and the traps that make
alphas look real when they aren't.

---

**1. The standard error is the whole game**

Every significance verdict flows from one ratio:

$$t = \frac{\hat{\alpha}}{\text{SE}(\hat{\alpha})}$$

The point estimate on top is the easy part. The **standard error** on the bottom is
where the honesty lives — and where the default OLS formula lies for financial
returns. Ordinary least squares assumes residuals are independent and identically
distributed. Monthly strategy returns violate both:

- **Heteroskedasticity** — volatility clusters (calm years, then a crisis), so the
  residual variance is not constant.
- **Autocorrelation** — this month's residual is correlated with last month's
  (momentum crashes, slow-moving factor regimes).

Both make the naive OLS standard error *too small*, which inflates $t$ and makes
noise look like skill. The fix is **HAC (Heteroskedasticity- and Autocorrelation-
Consistent) standard errors**, a.k.a. **Newey-West**. They widen the error bars to
account for the correlation structure. A common automatic lag choice is:

$$L = \left\lfloor 4 \left(\tfrac{T}{100}\right)^{2/9} \right\rfloor$$

where $T$ is the number of months. HAC SEs are almost always *larger* than OLS SEs —
if switching to HAC kills your significance, the significance was never there. This
dashboard reports HAC-based t-stats and p-values throughout.

---

**2. From the t-stat to a confidence interval**

- **p-value** — the probability of seeing an alpha at least this large *if the true
  alpha were zero*. $p < 0.05$ is the conventional bar; $p < 0.01$ is stronger. It is
  **not** the probability that your alpha is real — that's a common misreading.
- **Bootstrap confidence interval** — rather than trust a formula, resample the
  observed months (with replacement) thousands of times, re-estimate alpha on each
  resample, and take the 2.5th and 97.5th percentiles. This makes no assumption that
  $\hat{\alpha}$ is normally distributed, which matters for short, fat-tailed return
  series. **The single most useful habit: read the CI, not the point estimate.** If
  the 95% interval straddles zero, you cannot rule out that your true alpha is zero —
  regardless of how good the headline number looks. The Backtest Results page shows
  this interval directly under the FF5 Alpha card.

---

**3. The multiple-testing trap — the biggest threat to "real" alpha**

Significance thresholds assume you ran *one* test. You never do. Every model,
feature set, K, and construction method you try is another draw. Test 20 genuinely
worthless strategies at the 5% level and you *expect* one to print $t > 2$ purely by
luck. Keep the winner, and its t-stat is **selection-biased** upward — it looks
significant precisely because you chose it for being extreme.

This dashboard makes the trap concrete: the Alpha Model Lab lets you try
configurations and **pin several to compare**. The moment you compare $N$ strategies,
$t > 1.96$ is no longer the right bar. Three standard corrections:

| Method | Idea | Trade-off |
|--------|------|-----------|
| **Bonferroni** | Require $p < \alpha / N$ (equivalently a higher t-hurdle) to hold family-wise error at $\alpha$ across all $N$ tests. | Simple, but very conservative when tests are correlated. |
| **False Discovery Rate** (Benjamini-Hochberg) | Control the *expected fraction* of your "discoveries" that are false, rather than the chance of *any* false positive. | Less conservative; the modern default for many parallel tests. |
| **Deflated Sharpe Ratio** (Bailey & López de Prado) | Shrink the Sharpe (and, by analogy, alpha) for the number of trials, the variance across trials, and non-normal returns. Answers: "is this the best of many, or genuinely good?" | Needs a defensible count of how many configs you actually tried. |

The Bonferroni hurdle rises fast with the number of trials:

| Trials $N$ | Approx. t-hurdle (5% family-wise) |
|-----------|-----------------------------------|
| 1 | 1.96 |
| 5 | 2.58 |
| 20 | 3.02 |
| 100 | 3.48 |

This is why **Harvey, Liu & Zhu (2016)**, after cataloguing decades of published
factor "discoveries," argue the credible hurdle for a *new* anomaly is **$t > 3.0$,
not 2.0**. When you've tried many configurations, treat $t \approx 2$ as "interesting,
keep looking," not "found it."

---

**4. In-sample vs. out-of-sample alpha**

Alpha estimated on the same data used to build and tune the model is *optimistically
biased* — the model has partly memorized the sample. The walk-forward protocol makes
the reported returns genuinely out-of-sample, which is why the alpha here is more
trustworthy than a fitted backtest. But note the residual leak: even with clean
walk-forward returns, *if you picked the configuration by eyeballing its
out-of-sample alpha*, you have re-introduced selection bias through the back door.
True out-of-sample means data your choices never touched.

---

**5. Minimum detectable alpha — do you even have the sample size?**

Short backtests physically cannot certify small alphas, no matter the point estimate.
Setting the significance bar $t = 2$ and solving for the months needed:

$$T \approx \left(\frac{2}{IR_{monthly}}\right)^2, \qquad IR_{monthly} = \frac{\hat{\alpha}_{monthly}}{\sigma_{monthly}}$$

A strategy with a *monthly* information ratio of 0.2 (a respectable ~0.7 annualized
Sharpe) needs roughly $ (2/0.2)^2 = 100 $ months — over eight years — before a true
alpha reliably clears $t > 2$. If your backtest is 40 months and the alpha isn't
huge, an insignificant result may reflect **too little data**, not absence of skill.
Report the CI and move on; don't over-read either the significance or the lack of it.

---

**6. Statistical vs. economic significance**

A t-stat says the alpha is unlikely to be zero; it says nothing about whether the
alpha *survives contact with reality*:

- **Net of costs** — subtract turnover and transaction-cost drag. A significant gross
  alpha with a Cost-Sharpe that eats most of it is not investable.
- **Regime stability** — check the *rolling* alpha, not just the full-sample number.
  An alpha driven entirely by one lucky year (or one crisis month) is fragile, even if
  the pooled t-stat looks fine. A real edge shows up repeatedly across sub-periods.

---

**A confidence checklist**

Before calling something alpha, ask:

1. Is the t-stat computed with **HAC** standard errors? (Yes, here.)
2. Does the **bootstrap CI exclude zero**?
3. How many configurations did I try — and does the alpha clear the **multiple-testing
   hurdle** ($t > 3$ is a safe default), not just $t > 2$?
4. Is the alpha **out-of-sample**, and did I avoid choosing the config *by* its OOS result?
5. Do I have **enough months** to detect an alpha this size?
6. Does it survive **net of costs** and show up **across sub-periods**?

Passing one of these is easy and meaningless. Passing all six is rare — which is
exactly why genuine alpha is valuable. See the *Jensen's Alpha* section for how to
interpret the four alpha scenarios once you trust the number.
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
| Multiple Testing | Inflated false-positive risk from trying many strategies and keeping the best; corrected via Bonferroni / FDR |
| Deflated Sharpe | Sharpe ratio shrunk for the number of trials and non-normality (Bailey & López de Prado) |
| VIF | Variance Inflation Factor — measures multicollinearity |
| KS Test | Kolmogorov-Smirnov test for distribution shift |
| ERC | Equal Risk Contribution — portfolio where each position contributes equally to risk |
| MVO | Mean-Variance Optimization — maximize Sharpe ratio subject to constraints |
| Ledoit-Wolf | Covariance matrix shrinkage estimator; stabilizes noisy sample covariance |
| Walk-Forward | Time-respecting validation: only train on past data to predict future |
| Look-Ahead Bias | Using future information in training — the cardinal sin of backtesting |
| VaR | Value at Risk — maximum loss at a given confidence level over a time horizon |
| CVaR / ES | Conditional VaR / Expected Shortfall — expected loss given that loss exceeds VaR |
""",

    "var_and_tail_risk": r"""
**Value at Risk (VaR)**

Value at Risk is a single number that summarizes downside risk: it estimates the
largest loss a portfolio is likely to suffer over a set period, at a chosen level
of confidence. It compresses "how risky is this?" into one figure a risk manager
can budget around.

Every VaR number has three ingredients:

- **A time horizon** — the period the loss is measured over (here, one month).
- **A confidence level** ($\alpha$) — how sure you want to be (here, 95% or 99%).
- **A loss amount** — the number VaR reports.

Formally, VaR is the loss threshold that returns fall below only $(1-\alpha)$ of the time:

$$\text{VaR}_\alpha = -\inf\{x : P(R \leq x) > 1 - \alpha\}$$

So a **95% monthly VaR of 5%** reads as: *"In 19 months out of 20 we expect to lose
no more than 5%; in the remaining 1 month we expect to lose more than that."*

**What VaR does _not_ tell you:** it marks the threshold, not how bad things get once
you cross it — a 5% VaR is the same whether the worst 1-in-20 month loses 6% or 30%.
That blind spot is exactly what CVaR (below) is designed to fill.

**Three estimation approaches:**

| Method | Assumption | Strengths | Weaknesses |
|--------|-----------|-----------|------------|
| **Parametric** | Returns ~ Normal | Simple, closed-form | Underestimates tail risk (fat tails) |
| **Historical** | Past = future | No distribution assumption | Needs long history; ignores regime changes |
| **Monte Carlo** | Fitted model | Flexible; handles complex portfolios | Computationally expensive; model-dependent |

**Parametric VaR** assumes normally distributed returns:

$$\text{VaR}_\alpha = -(\mu + z_\alpha \cdot \sigma)$$

where $z_\alpha = \Phi^{-1}(1-\alpha)$ is the standard normal quantile (e.g., $z_{0.95} = -1.645$).

**Historical VaR** simply takes the empirical quantile of observed returns — no distribution assumed.

**Monte Carlo VaR** simulates thousands of return paths from a fitted distribution, then takes the quantile of the simulated distribution.

---

**Conditional VaR (CVaR) / Expected Shortfall**

VaR tells you the threshold; CVaR tells you *how bad it gets when VaR is breached*:

$$\text{CVaR}_\alpha = -E[R \mid R \leq -\text{VaR}_\alpha]$$

CVaR is a **coherent risk measure** (unlike VaR), meaning it satisfies subadditivity: the risk of a combined portfolio is never greater than the sum of individual risks. This makes CVaR preferable for portfolio optimization.

---

**Cornish-Fisher VaR**

Adjusts the normal quantile for skewness ($S$) and excess kurtosis ($K$):

$$z_{CF} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24} - \frac{(2z^3-5z)S^2}{36}$$

This captures two ways real returns depart from the normal bell curve:

- **Left-skewed** ($S < 0$): the distribution is lopsided, with a longer tail on the *loss* side than the gain side — crashes tend to be sharper and deeper than rallies are tall.
- **Fat-tailed** (excess kurtosis $K > 0$): extreme moves of *either* sign happen far more often than a normal distribution predicts, so a bell-curve model badly understates the odds of a large loss.

Because the normal model ignores both, parametric VaR reads too optimistic; Cornish-Fisher pushes the quantile further into the tail to compensate. When $S=0$ and $K=0$, it reduces to parametric VaR.

---

**Component VaR**

Decomposes portfolio VaR by position:

$$\text{CVaR}_i = w_i \cdot \frac{\partial \text{VaR}}{\partial w_i} = w_i \cdot z_\alpha \cdot \frac{(\Sigma \mathbf{w})_i}{\sigma_p}$$

Component VaRs sum to total VaR, so you can see which positions drive tail risk.

---

**Kupiec Proportion of Failures (POF) Test**

To validate a VaR model, count how many times actual losses exceeded VaR. Under a correct model, breaches should occur at rate $(1-\alpha)$. The Kupiec test uses a likelihood ratio:

$$LR_{POF} = -2\ln\frac{(1-p)^{n-x} p^x}{(1-\hat{p})^{n-x} \hat{p}^x} \sim \chi^2(1)$$

where $p$ is the expected breach rate, $\hat{p}$ is the observed rate, $x$ is the number of breaches, and $n$ is the number of observations. A low p-value means the VaR model is miscalibrated.
""",
}
