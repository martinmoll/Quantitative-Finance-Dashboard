# Project History: Alpha Strategy Dashboard

*Reconstructed from git history (82 commits, 2026-04-23 → 2026-06-29), archived planning docs recovered from git history, and the current codebase. Everything here is verifiable — sources are cited as commit hashes or file paths.*

---

## 1. Timeline of major phases

The project ran ~9.5 weeks (67 days) and splits cleanly into two halves: **an academic assignment** (Apr 23 – May 8, 21 commits) and **productization into a research platform** (May 9 – Jun 29, 61 commits). A nice detail visible in commit timezones: all commits through May 31 are UTC+8 (Hong Kong — the course was HKUST's RMBI3110), and from June 3 onward they are UTC+2 (Europe) — the project continued voluntarily after the semester ended.

### Phase 0 — Setup and learning (Apr 23–25)
The very first commit (`b775c7d`) contained exactly **two files**: `.gitattributes` and a 2-line README. The next commits added self-written ML/quant background notes and the assignment notebook `AlphaChallenge.ipynb`. Compare with today: 59 tracked files, ~8,600 lines of Python, a 9-page dashboard, a live data pipeline, and 79 tests.

### Phase 1 — Core modeling in one evening (Apr 25)
A striking burst: design spec committed at 17:29, implementation plan at 19:04, then **eight feature commits between 19:08 and 19:22** — feature engineering, three model configs (Lasso/RF/HGB), the walk-forward engine, backtests, evaluation tables, model blending, and Lasso feature importance. This worked because the design was fully specified *before* code was written (spec-first workflow — every later phase also has a design doc + implementation plan committed before the code).

### Phase 2 — The tuning campaign (Apr 27 – May 8)
The longest, hardest phase. The assignment rubric disqualified any strategy with max drawdown (MDD) worse than -40%, and **every baseline model failed it** (HGB: SR 1.09 but MDD -50.1%). Two archived documents — `mdd_reduction_log.md` and `progress.md` (recoverable via `git show 702837d^:docs/archive/...`) — record a systematic experimental campaign: sector caps (failed), feature fixes (partial), volatility tilt (worked), regime filter (worked, after a look-ahead bug was caught — see §3). Submission: May 8, "Final model done for submission", **SR 1.48, MDD -27.6%** vs SPY's SR 0.66.

### Phase 3 — Dashboard v1 (May 9–11)
The morning after submission (03:19 AM commit), a design spec for a "Bloomberg-style" Streamlit dashboard. Over two days: feature engineering extracted from the notebook into `dashboard/features.py`, a walk-forward engine with a prediction/portfolio split, a **two-tier disk cache**, and a single-page app with ~10 chart types. Merged as PR #1.

### Phase 4 — Repositioning + 7-page expansion (May 21–29)
Deliberate pivot from "class assignment" to "portfolio-grade project" (`4bdb434`: "remove assignment framing"): README rewritten, assignment docs archived, notebooks untracked, long-short strategy added. Then a massive one-night expansion (May 22, 00:40–01:36): the single-page app was **replaced** (old `app.py` and `engine.py` deleted) by a modular architecture — `core/` (models, backtest, portfolio, risk, factor models, diagnostics, data loader), `components/`, and 7 pages. The registry grew to 6 models, 5 portfolio construction methods, factor models (CAPM/FF3/FF5), and monitoring (KS tests, alpha decay). The implementation plan for this phase was 4,801 lines. Merged as PR #2.

### Phase 5 — Live data pipeline (May 29–31)
Until now the project ran on the course-provided dataset. A new `pipeline/` package made it self-sufficient: fetchers for prices/fundamentals (Yahoo Finance), FF5+momentum factors (Ken French's library), macro series (FRED: VIX, yield-curve slope, credit spread, EPU, financial stress), feature computation modules, and an assembler producing a monthly cross-sectional panel (116 raw feature columns defined in `pipeline/config.py`).

### Phase 6 — Risk analytics and model upgrades (Jun 3–8)
VaR/CVaR tail-risk page (parametric, historical, Monte Carlo, Cornish-Fisher, with Kupiec backtesting), auto-generation of the dataset on first run, repaired test imports + new test files (`bc0dd00`), workflow navigation, then (`77fdb51`) an **Ensemble model, hyperparameter auto-tuning via inner time-series CV, and transaction-cost-aware MVO**.

### Phase 7 — Hardening and polish (Jun 27–29)
After a ~3-week gap: centralized metric interpretations (523-line `interpretations.py` giving context-aware explanations of every metric), out-of-sample R² (Campbell & Thompson 2008), pipeline hardening (dedup, graceful FF5 fallback), a full design-system UI modernization, **bootstrap confidence intervals** for Sharpe and alpha (5,000 resamples), a rewritten README with an explicit *Limitations* section, and repo cleanup.

---

## 2. Key architectural / design decisions

**Walk-forward validation instead of random cross-validation.**
Decided in the very first design spec. The alternative — standard k-fold CV — randomly splits time-series data, letting a stock's 2020 return help predict its 2018 return (leakage). The engine (`dashboard/core/backtest.py`) trains only on months strictly before the prediction month, retrains on a configurable schedule (12-month found optimal), and supports expanding vs rolling windows. The project's own theory page calls this "the most critical design decision in financial ML."

**Models tiered by feature capacity.**
Two feature tiers: Tier 1 (~52 features) for linear models that overfit with too many inputs, Tier 2 (~118 features) for tree ensembles that handle them. Alternative: one feature set for all models. The registry pattern (each model = a `(feature_builder, estimator)` pair) was chosen explicitly so "adding, removing, or swapping models is a one-line change" — and it paid off: 3 models at submission grew to 7 (adding Ridge, ElasticNet, Fama-MacBeth, Ensemble) with no engine rewrite.

**Deliberately conservative hyperparameters.**
From the original design spec: "All models configured conservatively (shallow, regularized). Overfitting is the dominant failure mode in finance." Shallow trees (depth 4), large leaf sizes (min 100 samples), heavy regularization. The alternative — deep expressive models — was rejected up front, and the rubric explicitly disqualified "overfitting/excessive params."

**Regime-conditional volatility tilt (a documented change of mind).**
First attempt: unconditionally penalize high-volatility stocks in the ranking. The MDD log records the verdict: *"Unconditional vol tilt works for MDD but destroys too much return. Need a regime-conditional approach"* (annual return dropped 24.2% → ~12%). Second attempt: apply the tilt **only when trailing 3-month SPY volatility exceeds a threshold**, so the model's picks are trusted fully in calm markets. Key empirical finding: the threshold mattered far more than the tilt strength.

**Prediction/portfolio cache split (two-tier cache).**
Model predictions (expensive, minutes) and portfolio construction (cheap, seconds) are cached separately with SHA-256 keys over their exact parameter sets (`dashboard/cache_manager.py`). Changing K or vol-tilt reuses cached predictions and only rebuilds the portfolio. Notably, commit `a88e617` adds `strategy_type`/`K_short` to the cache key the same day long-short was added — new parameters must invalidate old cache entries or you silently serve wrong results.

**Notebook → modular package (full rewrite, not refactor).**
The dashboard was rebuilt twice: notebook → single-page app (May 9), then single-page app → multipage modular architecture (May 22, with the old `app_old.py`/`engine.py` deleted outright). Evidence the second design was right: five subsequent feature phases (pipeline, VaR, ensemble, interpretations, redesign) each slotted into existing modules.

**Course dataset → live pipeline.**
The assignment dataset was static and course-provided. The live pipeline (yfinance + Ken French + FRED) made the project reproducible by anyone — the app auto-builds the dataset on first run. Trade-off consciously documented in the README: using *current* S&P 500 constituents introduces survivorship bias vs point-in-time membership (which requires paid data).

**Streamlit as the framework.**
Alternative would have been a JS frontend (React + charting) or Jupyter widgets. Streamlit's multipage model maps one page per workflow stage (Data → Factors → Model Lab → Backtest → Portfolio → Monitoring → Tail Risk), and Python end-to-end means the research code *is* the app code.

---

## 3. Obstacles and friction points

**The look-ahead bias bug — the best story in the repo.**
During regime-filter tuning, results looked great: SR 1.53. Then the bug was found: the trailing SPY return used to decide "invest this month or go to cash" **included the current month's own return** — the strategy was peeking at the outcome it was trying to predict. The fix was one line — `.shift(1)` on the trailing series (still visible at `dashboard/core/portfolio.py:195`) — and SR dropped 1.53 → 1.41. Rather than deleting the invalid results, `progress.md` marks phases 2e–2g "**[BUGGY — no .shift(1)]** … preserved for historical context but **invalid**", then re-runs the entire lookback sweep. The corrected optimum (6-month lookback) recovered to SR 1.48. Look-ahead bias was a *disqualifying* offense under the rubric, so catching it was the difference between a grade and a zero.

**The MDD wall.**
Every baseline model violated the MDD < 40% constraint because a long-only portfolio carries full market beta — in 2008 everything fell 40–55%, diversification or not. The log shows the honest path: sector caps tried first and **abandoned** ("The drawdown is market-wide, not sector-specific"), before the vol-tilt + regime-filter combination worked.

**The composite-dilution feature bug.**
Composite signals originally averaged over a fixed column list, filling missing columns with zeros — so if only 3 of 5 earnings columns existed, the composite averaged 3 real values + 2 zeros, diluting the signal toward the mean. Fixed by averaging only columns present (`_get_existing()` in `dashboard/features.py:5`, with a comment explaining why). Fixing it raised HGB_500's SR from 0.98 to 1.09 in one change — a real "feature quality beats model tuning" moment.

**Non-monotonic tuning surprises.**
The vol-tilt sweep found tilt=0.05 gave SR 1.20 while both 0.0 (SR 1.09) and 0.3 (SR 0.89) were worse — a tiny nudge removes tail risk without overriding the model's alpha picks, but a strong one destroys returns. Similarly, K=10 beat K=15/20/30, and **feature selection was tested and rejected**: top-20/30/50/70 feature subsets all underperformed the full 118 (SR 1.11–1.33 vs 1.36).

**External data source fragility.**
The Ken French FF5 download would intermittently fail; commit `16f4be1` wraps it with a graceful fallback to cached factors. The same commit deduplicates dataset rows on (month, stock) — yfinance appends had produced duplicates that corrupted forward-return backfill.

**Test rot.**
Commit `bc0dd00` ("fix: repair test imports and add missing VaR/risk + data_loader tests") shows tests broke during the rapid June feature additions and had to be repaired and expanded — the test suite grew to cover the newly added risk module.

**Minor churn worth knowing about:** Streamlit page files are number-prefixed, so adding pages forced renames (Theory 7→8, Data Pipeline 8→9→0); a pandas deprecation (`applymap` → `map`) needed fixing; one commit was accidentally duplicated (`42580df`/`0c03b72`, identical message, May 9).

---

## 4. Scope evolution

**Grew far beyond the assignment.** The assignment needed: one notebook, three models, one performance table. Added afterward, none of it required: the dashboard itself, long-short support, 4 more models, 5 portfolio construction methods, factor regressions, VaR/CVaR with Kupiec backtesting, monitoring (KS drift tests, alpha decay), hyperparameter auto-tuning, live data pipeline, bootstrap CIs, and a 79-test suite.

**Deliberately repositioned.** The May 21 overhaul spec states the goal outright: "Reposition this repository from a class assignment (RMBI3110) to a general-purpose quantitative finance dashboard." Assignment docs were archived, not deleted — and then in the final cleanup (`702837d`) removed from the tree but intentionally left recoverable in git history.

**Planned but cut / not yet done (documented, not hidden):** the README's Limitations section explicitly lists capacity analysis ("not yet implemented"), point-in-time index membership (survivorship bias), market-impact-aware cost models (Almgren-Chriss named as the alternative to flat bps), and short-borrow constraints. MAXSER portfolio weighting was tested during the assignment but "falls back to equal-weight with K=10" and didn't survive into the dashboard's five methods.

**Simplified versions of grander ideas:** the tracked v2 dataset snapshot covers 101 stocks over 37 months (2023-06 → 2026-06) with 231 columns — a Nasdaq-100 slice, while `universe.py` supports S&P 500, Nasdaq-100, or both; the full S&P 500 build runs on first launch rather than being shipped.

---

## 5. Quality / rigor measures (with mechanics)

**Walk-forward train/test isolation** (`dashboard/core/backtest.py:59`). For each out-of-sample month *m*, the training set is literally `data[data["ym"] < m]` — a strict inequality on the month column, so the model never sees the month it predicts or anything after. Predictions then roll forward month by month, exactly as a live strategy would have operated.

**An explicit no-lookahead test** (`tests/test_backtest.py:49`, `test_no_lookahead`). Runs the walk-forward engine on a synthetic panel and asserts every predicted month is on or after the out-of-sample start date. It's a regression guard: if someone later breaks the date logic, this test fails.

**The `.shift(1)` regime lag** (`dashboard/core/portfolio.py:195`). The "should I be invested this month?" signal is trailing SPY returns *shifted by one month*, so the decision for month *m* uses only data through *m−1*. This is the exact line that fixed the look-ahead bug found during the assignment.

**Lagged peer features in the pipeline** (`pipeline/features/peer_features.py:38` and 3 more sites). Any feature built from other stocks' returns (sector momentum, size-peer returns, leader returns) is computed with `.shift(1)` — the feature for month *m* only reflects peers' behavior through *m−1*, because month-*m* peer returns aren't knowable when the month-*m* prediction is made.

**Chronology-respecting hyperparameter tuning** (`backtest.py:_tune_hyperparams`). When auto-tune is on, inner cross-validation folds are built so validation months always come *after* their training months (the last 12-month blocks of the training window). Random CV would leak future information even inside the training set; this doesn't.

**Bootstrap confidence intervals** (`dashboard/core/diagnostics.py:bootstrap_sharpe_ci`). Instead of reporting "Sharpe = 1.4" as a point estimate, monthly returns are resampled 5,000 times (with replacement, seeded for reproducibility), Sharpe recomputed each time, and the 2.5th/97.5th percentiles reported. This answers "could this Sharpe be luck?" — a wide interval spanning zero means yes.

**Content-hashed caching** (`dashboard/cache_manager.py`). Cache keys are SHA-256 hashes of the *complete* sorted parameter set (model type, hyperparameters, features, window type, retrain frequency…). Any parameter change produces a different key, so stale results can never be served for a new configuration — correctness enforced by construction, not discipline.

**Data integrity in the assembler** (`pipeline/assembler.py`). Rows are deduplicated on (month, stock) before forward returns are backfilled, because a duplicated stock row would corrupt the mapping of next month's return back onto this month's row (the prediction target).

**79 pytest tests** across 11 files covering the backtest engine, models, portfolio construction, risk (VaR/CVaR), factor models, diagnostics, features, data loading, and both pipeline stages.

**Separate train and eval targets** (`backtest.py:14-15`). Models train on the cross-sectionally standardized return (`y_xs`, comparable across regimes) but are *evaluated* on the raw return (`y_raw`) — you tune on a well-behaved target but score on the thing you'd actually earn.

**Documented limitations** (README). Survivorship bias, flat transaction costs, unlimited free borrow, single asset class — stated plainly with the note "acknowledging limitations is part of rigorous research."

---

## 6. The "almost done but not done" moment

**May 8, 2026, 23:51 — commit `89aecf2`: "Final model done for submission."** At that point the model worked: SR 1.48, MDD -27.6%, beating SPY's 0.66 by +0.82 Sharpe. The assignment was finished. By commit count, that was **21 of 82 commits — the remaining 61 commits (74%) came after the model "worked."**

What the last mile contained: extracting notebook code into a tested package, the entire dashboard (twice), the live data pipeline, all the risk analytics (VaR/CVaR, factor exposure, monitoring), the test suite, caching, statistical honesty (bootstrap CIs, R²_OOS), and UI/UX. In hours it's smaller than 74% (several phases were dense one-night pushes), but the pattern holds: *a working model is the beginning of a product, not the end.*

There was also a nested version of this moment inside the assignment itself: on April 25 the models trained and predicted fine (SR 1.09 — "done"?), but every configuration failed the MDD constraint. The two weeks from Apr 27 to May 8 — the tuning campaign, the bug hunts, the look-ahead fix — were the assignment's own last mile.

---

## Which dataset to talk about

The project has two data eras, and the distinction matters when presenting:

- **Assignment era (Apr–May):** ran on a locally downloaded course dataset (large panel, 227 out-of-sample months from 2005). The SR 1.48 / MDD -27.6% result, the IC statistics, and all the tuning-campaign numbers come from this data. That dataset was never committed and is not part of the product.
- **The product (what the repo is now):** built on the **live data pipeline** — yfinance for prices and fundamentals, Ken French's library for FF5 factors, FRED for macro. The app auto-builds its dataset on first run, and every performance metric in the dashboard is **computed fresh on live data, with bootstrap confidence intervals**, rather than being a fixed headline number.

**Why the live dataset only covers ~3 years.** The price fetcher deliberately requests `period="3y"` from yfinance (`pipeline/fetchers/prices.py:17`, specified in the May 29 design spec — not a bug). Three constraints of the free data source make a longer window pointless or misleading: (1) **fundamentals are the binding constraint** — yfinance's quarterly statements only return the last ~4–5 quarters, so value/quality/earnings features can't be computed further back regardless of price history (the design spec already hit this wall: `nsi` is hardcoded to 0.0 with the note "share count history unreliable from yfinance"); (2) **survivorship bias compounds with depth** — the universe is *today's* index constituents, so backfilling 15 years of prices for current members would produce a panel of proven winners; three years keeps the distortion modest; (3) **rate limits** — 500 tickers already download in batches of 50 with delays and retries. This is why the two-era split exists at all: deep-history validation belongs to point-in-time institutional data (the course dataset), while the live pipeline is built for *ongoing operation* — monthly appends, drift monitoring, current signals. One practical consequence to know: with a ~37-month panel, long-lookback features like `ret_13_36` (needing 36 months) are NaN in the earliest live months.

When discussing the project, lead with the live platform: its methodology (walk-forward validation, factor attribution, cost-aware construction, monitoring) is dataset-agnostic and is the actual engineering achievement. Cite the assignment-era numbers as *development history* — they are evidence of process (the look-ahead bug you caught, the drawdown campaign, feature-quality findings) and of what the methodology achieved on a long historical panel, not claims about current live performance. This framing is also the honest one: the live universe uses current index constituents (survivorship bias, documented in the README), so quoting the historical SR as if it were the product's live track record would overreach.

---

## Interview-ready facts

1. **82 commits over 67 days** (Apr 23 – Jun 29, 2026); first commit was 2 files and 4 lines, the final repo is ~8,600 lines of Python across 59 files (5,816 dashboard, 1,516 pipeline, 1,249 tests).

2. **The look-ahead bug:** my regime filter used trailing market returns that included the current month — the strategy was peeking at the outcome it predicted. The fix was one line (`.shift(1)`), and it cost 0.12 Sharpe (1.53 → 1.41); after re-tuning I recovered to **1.48**. I kept the invalid results in the log, explicitly marked "BUGGY … invalid."

3. **Assignment-era result (course dataset): Sharpe 1.48, max drawdown -27.6%**, vs the S&P 500 benchmark's Sharpe 0.66 over the same 227 out-of-sample months — +0.82 Sharpe over the market. Frame this as what the methodology achieved on a 2005-onward historical panel; the product itself recomputes all metrics live on pipeline data.

4. **The composite-dilution bug:** composite signals were zero-padding missing columns (averaging 3 real values + 2 zeros), diluting the signal. Fixing that one function raised HGB's Sharpe from 0.98 to 1.09 — feature quality beat any amount of model tuning.

5. **Sector caps failed before vol tilt worked:** I first tried capping per-sector holdings to cut drawdown; it did nothing because the 2008 drawdown was market-wide, not sector-specific. That diagnosis redirected me to volatility tilting.

6. **Non-monotonic tuning:** a volatility tilt of 0.05 gave Sharpe 1.20 while both zero (1.09) and 0.3 (0.89) were worse — a tiny nudge away from the most volatile names removes tail risk without overriding the model's picks.

7. **Feature selection was tested and rejected with data:** top-20/30/50/70 feature subsets all underperformed the full 118-feature set (SR 1.11–1.33 vs 1.36) — even low-importance features contribute through interactions in tree models.

8. **~118 features in 11 groups** (momentum, value, quality, earnings, analyst, technical, volatility, interactions, relative, nonlinear transforms…), each engineered feature justified by an economic-rationale comment in the code (e.g., value × low-vol isolates "safe value" from distress).

9. **7 alpha models, 5 portfolio construction methods, 9 dashboard pages, 79 automated tests** — including a dedicated `test_no_lookahead` regression test.

10. **Signal quality numbers (assignment-era, course dataset):** mean monthly IC 0.0225, IC t-stat 3.45 (statistically significant), 65.6% hit rate, ~56% monthly turnover costing ~0.67%/year at 10 bps — net Sharpe ~1.28 after costs. The dashboard computes the same diagnostics live on pipeline data.

11. **Two-tier caching:** predictions (minutes to compute) and portfolios (seconds) cache separately under SHA-256 hashes of their full parameter sets — changing portfolio parameters reuses model predictions, and stale cache hits are impossible by construction.

12. **The model was "done" at commit 21 of 82** — 74% of all commits came after the model worked: dashboard, live data pipeline (yfinance/Ken French/FRED), risk analytics, tests, and statistical honesty (bootstrap CIs with 5,000 resamples).

13. **Spec-first workflow:** every phase has a design spec and implementation plan committed *before* the code — the dashboard-expansion plan alone was 4,801 lines, and the core modeling engine went from spec to working backtests of three models in one evening (eight commits, 19:04–19:22).

14. **Two full rewrites by choice:** notebook → single-page dashboard → modular multipage package (old app deleted outright). The modular design then absorbed five later feature phases without an engine rewrite.

15. **The 3-year live window is a reasoned trade-off, not a gap:** yfinance only exposes ~4–5 quarters of fundamentals, so deeper price history would yield features that are mostly empty — and since the universe is today's constituents, going further back would only compound survivorship bias. So: methodology validated on a 20-year point-in-time panel (course data), live pipeline scoped to 3 years of current data with the trade-off documented.

16. **Honest limitations, documented:** survivorship bias from using current S&P 500 constituents, flat transaction costs vs market-impact models, unlimited short borrow — listed in the README because "acknowledging limitations is part of rigorous research."

---

### Sources for verification
- Commit history: `git log --reverse --stat`
- Archived tuning logs (removed in final cleanup, recoverable): `git show 702837d^:docs/archive/progress.md` and `git show 702837d^:docs/archive/mdd_reduction_log.md`
- Original design spec: `git show 702837d^:docs/archive/2026-04-25-alpha-strategy-design.md`
- Look-ahead fix in code: `dashboard/core/portfolio.py:195`; no-lookahead test: `tests/test_backtest.py:49`
- Composite-dilution fix: `dashboard/features.py:5`
