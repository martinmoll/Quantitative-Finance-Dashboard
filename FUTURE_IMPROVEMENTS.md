# Future Improvements

A running log of implementation ideas to review and build later. Add new ideas
here as they come up; keep each one concrete enough to pick up cold.

---

## Currency risk (NOK investor)

**Context.** A Norwegian investor funding USD stock purchases with kroner earns
the stock's local (USD) return *plus* the USDNOK move. The unhedged NOK
conversion and a reporting-currency toggle are already implemented
(`pipeline/fetchers/fx.py`, `dashboard/core/currency.py`, and the toggle on the
Backtest Results page). The items below are the remaining, more involved pieces.

### 1. Hedged NOK return series

Model rolling 1-month FX forwards so the currency exposure can be stripped out.
By covered interest parity the hedge cost ≈ the short-rate differential:

```
r_hedged ≈ r_local − (i_USD − i_NOK) / 12
```

- Needs USD and NOK short rates (FRED already wired in for US rf; add a NOK rate).
- Add "NOK (hedged)" as a third option on the reporting-currency toggle.
- Value: comparing hedged vs unhedged Sharpe shows whether the currency exposure
  helped or just added noise.
- Caveat: Yahoo `NOK=X` is a spot proxy, not a tradable forward curve — the hedge
  leg is an approximation.

### 2. FX risk decomposition / attribution

Answer "how much of my return and risk *is* currency":

```
Var(r_NOK) ≈ Var(r_local) + Var(r_fx) + 2·Cov(r_local, r_fx)
```

- Split cumulative return into a local-return leg and an FX leg (NBIM-style
  attribution chart).
- Report the variance breakdown and Cov(local, FX). NOK is oil-linked, so the
  covariance term is often non-trivial for an Oslo/energy-heavy book.
- Surface as a small panel on the Attribution tab.

### 3. Currency-aware factor attribution

Today the FF5 alpha regresses NOK-converted returns on USD factors, so in NOK
mode the alpha also absorbs currency swings (flagged in the UI banner). Proper
fix: either always run the factor regression on the local (USD) return series
regardless of reporting currency, or add an explicit FX factor to the regression.

---

## Industry roadmap items

See `ROADMAP.md` for the prioritized overview. Implementation notes for the
larger deferred pieces:

### Point-in-time universe + delisting returns (Tier 1, data-limited)

- yfinance drops delisted tickers, so survivorship can't be fully fixed on free
  data. Concrete steps that *are* feasible: freeze the constituent list per
  rebuild (a dated snapshot in `pipeline/cache`), and when a name disappears
  from later data, book a delisting return (−100% for bankruptcies, else last
  observed) rather than dropping the row.
- Quantification tool: run the same strategy on "today's constituents" vs a
  frozen older snapshot and report the Sharpe gap as the survivorship premium.
- The honest full fix is point-in-time index membership from paid data (CRSP).

### Cross-sectional factor risk model + constrained optimization (Tier 2)

- Build a structural risk model: regress stock returns on factor exposures
  (FF5 betas + industry dummies) → factor covariance `F` and diagonal specific
  risk `D`; stock covariance `Σ = B F Bᵀ + D`. Far more stable than the sample
  cov of the top-K used today in `portfolio.py:_get_cov_matrix`.
- Feed `Σ` into a constrained optimizer (cvxpy): maximize `wᵀα − λ wᵀΣw` s.t.
  Σw = 1, sector exposures = benchmark (sector-neutral), portfolio beta = target
  (beta-neutral), |w| ≤ cap, and a turnover cap vs previous weights.
- Add a risk-decomposition panel: % of portfolio variance from each factor vs
  specific risk.

### Deflated Sharpe Ratio + PBO (Tier 3)

- DSR (Bailey & López de Prado 2014): `DSR = Φ((SR − SR0)·√(n−1) / √(1 − γ3·SR +
  (γ4−1)/4·SR²))`, where `SR0` is the expected max Sharpe across `N` trials
  (grows with N) and γ3, γ4 are skew/kurtosis of returns. The pinned-config
  count is a defensible `N`. Surface next to the bootstrap Sharpe CI.
- PBO via CSCV: split the return matrix over configs into `S` combinatorial
  train/test halves; PBO = fraction of splits where the in-sample-best config
  underperforms the test-set median. Report as "probability this is overfit".

### Benchmark-relative framework (Tier 2)

- Add active return `r_p − r_b`, tracking error `std(active)·√12`, IR
  `mean(active)/te`, and active share `½Σ|w_p − w_b|`. Benchmark weights: cap-
  weighted from `me`, or equal-weight the universe.
- Reporting-only first; a min-tracking-error variant of the optimizer is a
  follow-up once the risk model exists.

### Purged & embargoed CV (Tier 3)

- In `backtest._tune_hyperparams`, drop training observations whose label window
  overlaps the validation fold (purge), and embargo a buffer of months after
  each fold, before scoring. Prevents leakage from overlapping forward returns.

### Standardized tear sheet + paper-trading loop (Tier 4)

- Tear sheet: assemble the existing metrics (rolling Sharpe, monthly/annual
  tables, drawdown, factor exposures over time) into one exportable HTML page.
- Paper trading: persist each month's target portfolio to disk; on the next run
  compare realized vs expected returns and accumulate a live IC series on the
  Monitoring page.

---

## Concentration control — follow-ups and negative results

Context: the July 2026 unwind. The equity curve falls between its May and June
2026 points; because the series is indexed by the *decision* month and books
`y_raw`, that loss was realized in **calendar July 2026**. The market did not
fall — cap-weighted universe +1.39%, only 38.7% of names down, S&P 500 +0.19%.
One theme fell: SNDK −46.6%, GLW −45.9%, KLAC −39.4%, INTC −35.4%, LRCX −32.4%,
AMAT −29.8%, MU −28.7%. The book held nine of ten names in that supply chain
and lost 30.8% gross. Book beta 2.76 × market +1.39% = +3.84% expected, actual
−30.8%, **residual −34.7%** — a theme shock, not market exposure.

Shipped: a sector count cap (`max_per_sector`), a bounded covariance window
(`cov_window`), and `diagnostics.effective_bets`.

### Rejected, with the measurement — do not retry without new evidence

- **Industry-level cap.** Measured at K=10: worst month −27.8% vs −20.2% for a
  sector cap, SR 1.34 vs 1.40. The theme spanned six `industry` labels
  (Semiconductors, Semiconductor Equipment & Materials, Computer Hardware,
  Electronic Components, Communication Equipment), so capping semicap at 3 just
  backfilled with semiconductors and hardware — the same trade. `industry` is
  too fine a granularity to bind on a cross-industry theme.
- **Correlation clustering of candidates.** Hierarchical clusters on trailing
  return correlation, capped per cluster: worst month −26.5%, SR 1.36. Worse
  than a plain sector cap and it puts a clustering step on the hot path.
- **EWMA / shorter covariance as an early warning.** The 2026-06 book moves only
  from the 10th to the 26th–33rd percentile of ex-ante risk — still below
  median — and every estimator tried (36m and 12m sample, EWMA halflife 6 and
  12) correlates *negatively* with next-month absolute return. Portfolio
  covariance on a ten-name book does not forecast this. `cov_window` was kept
  as an estimator fix for ERC/MVO, not as protection.
- **Binding on effective bets.** Under a sector cap the metric stops predicting:
  correlation with next-month return falls +0.20 → +0.04 and bottom-decile
  months go from −6.6% to +1.1% mean next-month return, because the cap already
  removed the exposure it detects. A second binding constraint would be
  redundant and would make the two impossible to attribute separately.

### Still open

- **A concentration cap does not fix the max drawdown** (−39.5% → −38.4%). MDD
  here is a multi-month 2022 path, not a single event. Fixing it is a different
  problem from fixing the worst month.
- **`prev_weights` should be a permno-indexed `pd.Series`, not a positional
  array.** `_mvo_weights` aligns `ref` by position, which is the only reason
  `build_portfolio_series` has to re-derive the book to build `mvo_prev`. If
  `_mvo_weights` did `selected["permno"].map(prev_weights).fillna(0.0)` the
  duplication would disappear and could never drift again. `test_mvo_tc_aware`
  passes an ndarray, so the parameter would have to accept both.
- **`5_Portfolio_Construction.py` risk decomposition uses `np.eye(n) * 0.01`.**
  Its "Risk Decomposition" pie is therefore normalized `|weight|`, not risk. It
  should take the same point-in-time covariance the diagnostic now builds.
- **Long-short passes the long leg's `prev_weights` to the short leg**
  (`portfolio.py`, `_compute_weights` for `bottom`). With `mvo` and
  `K_short != K` this misaligns.

## Dead feature columns

**47 of the 231 columns in `alpha_dataset_v2.parquet` are 100% NaN.**
`ind_crowding` is one of them, not a special case.

- **`_TIER1_CORE` contains `sue_xs`, `revision_xs` and `beat_xs`, all entirely
  NaN.** `backtest.py` filters features on column *existence* only, and both
  `features.py` and `backtest.py` blanket-`fillna(0.0)`, so these enter every
  Tier 1 fit as constant-zero columns. There is no variance or coverage guard
  anywhere. The Data Explorer's missing-data chart scans `_xs` columns only,
  which is why this was never visible. **A zero-variance / coverage filter
  alongside the existence check is cheap and independent of everything else.**
- **`ind_crowding` specifically** is in `RED_FEATURES` (`pipeline/config.py`) —
  consciously parked for want of a data feed — and also in `_NO_XS_SUFFIX`, so
  it has no `_xs` twin and never reaches the model matrix at all. Making it real
  needs a definition first; nothing in the repo says what it should measure. A
  turnover-based industry crowding measure is buildable from existing columns
  (`turnover` is 84,800/84,861 non-null) and would go in `peer_features.py`
  beside `ind_mom`, come out of both lists, and need a dataset rebuild.
  Caveat: median names per industry-month is 1 before 2020 and 3 in the 2020s,
  so industry-level statistics are near-degenerate on this panel — group on
  `sector` (12 values, dense) or require a minimum group size.

## Feature audit — what was fixed, what stays dead

Measured on `alpha_dataset_v2.parquet`, OOS 2016-07 onward.

### Fixed

- **The value/quality block was fabricated in two months out of every three.**
  `compute_fundamental_features` stamped each quarter to one month only
  (quarter-end + `REPORTING_LAG_MONTHS`), and the assembler left-joins on
  `["ticker", "ym"]`, so the two months between filings had no fundamentals at
  all. Per-month coverage of `bm` cycled 0.87 / 0.09 / 0.04 from 2025-08. A
  filing does not stop being the newest public figure the month after it lands,
  so it is now carried until the next one supersedes it, capped at the
  three-month filing cadence so a company that stops filing stops reporting.
  On the cached filings this turns 3,538 published filing-months into 11,793
  (x3.33) and gives every month from 2025-01 to 2026-10 about 575 names with a
  `bm` value instead of one month in three. **Needs a pipeline rebuild to reach
  the parquet.**
- **A no-variance guard at fit time.** `run_walk_forward` checked only that a
  feature column existed, then `fillna(0.0)`. For cross-sectionally
  standardized features zero means "exactly average", so an all-NaN column
  entered every fit as "average everything" for every stock. At the first Tier 1
  retrain 33 of 52 columns were constant this way; by 2026 it is 9 of 52.
  `_informative_features` now judges each training window, so a column that is
  empty early and real later — the fundamentals block — is picked up at the
  first retrain after it becomes real, with no look-ahead.
  Measured against the same run without the guard: **out-of-sample IC, annual
  return, Sharpe and worst month are unchanged** — HGB tier 2 (66 columns in,
  52 fitted at the last retrain) gives IC +0.0321, t +2.88, 64.3% a year,
  SR 1.35, worst month −27.5% either way; Lasso tier 1 (52 in, 43 fitted) is
  likewise identical. An HGB walk-forward **halves**, 284s to 141s. Pushing
  further to a 5% coverage floor, which also drops the sparse fundamentals,
  changes nothing either (IC +0.0323, SR 1.35) — so a coverage threshold buys
  nothing the variance check does not, and was left out rather than adding a
  number to tune. The gain is run time and an honest feature-importance
  report, not return. `predictions_version` bumped to 3
  because RandomForest samples `max_features` and Fama-MacBeth inverts the
  design matrix, so those two do change.
- **A partial FRED download is no longer silent.** `fetch_macro` raised only
  when *every* series failed, so one failure was a warning nobody read and the
  partial frame was cached and reused forever. The live cache holds
  `credit_spread`, `epu` and `fin_stress` but not `vix` or `yield_curve_slope`,
  which is why `macro_unc_1m`, `macro_unc_12m` and `mom_x_unc` have always been
  empty. It now names the missing series and the features they cost.
  **`FRED_API_KEY` is not set, so filling them needs a key and a rebuild.**

### New features prototyped and rejected — do not retry without new evidence

Rank IC and its t-statistic, OOS months only. Nothing cleared |t| > 2 on its
incremental content, and every candidate is another risk loading of the kind
survivorship bias inflates on this panel.

| candidate | mean IC | t | note |
|---|---|---|---|
| `ind_beta` (industry, ex-self, 24m) | +0.019 | +1.78 | |
| `sec_beta` (sector, ex-self, 24m) | +0.025 | +1.43 | |
| `ind_comovement` / `sec_corr` | +0.001…+0.009 | +0.08…+0.56 | no alpha at any window |
| downside beta (36m, down months) | +0.043 | +2.04 | mostly plain beta |
| downside beta − beta | +0.017 | +1.19 | the incremental part |
| seasonality (Heston-Sadka) | −0.010 | −0.77 | |

- **The ex-self correction matters.** An earlier pass measured `ind_beta` at
  t = +2.53 by regressing each stock on a peer mean that included itself. With
  the stock removed from its own peer group it is +1.78. Industry groups have a
  median of 2-3 names on this panel, so the contamination was large.
- **Comovement as a portfolio constraint was already rejected** (correlation
  clustering measured worse than a plain sector cap), and `effective_bets` is
  the portfolio-level version of the same quantity. There is no remaining slot
  for it.

### Still dead, and why

- **No data source in the repo:** the options/IV block (`iv_atm_30d`,
  `iv_atm_91d`, `iv_skew`, `pc_vol_ratio`, `pc_oi_ratio`, `vrp`,
  `iv_term_structure`, `sector_iv`, `sector_vrp`) and the analyst block
  (`beat`, `n_analysts`, `revision`, `dispersion`, `revision_ratio`,
  `rev_surp`, `peer_revision`). These are the `RED_FEATURES` list; it is
  accurate.
- **`ind_size_ret` and `ind_size_mom`** are assigned `np.nan` outright in
  `peer_features._add_lagged_diffs`. An industry x size double sort is
  degenerate here (median 2-3 names per industry-month before the split), so
  reviving them means grouping on `sector`, not `industry`.
- **The fundamentals themselves are still too short to test.** Even carried
  forward, the cache holds a median of 7 quarter-ends and only 16 OOS months
  have enough coverage to measure. Every value/quality IC over those months is
  noise (|t| < 2.5, mostly negative). The carry-forward is a point-in-time
  correctness fix that will pay off as history accumulates, not a signal today.
- **Momentum has no IC on this panel** (`ret_2_12_xs` t = +0.25,
  `ret_1_xs` t = −0.05), partly because the daily price cache starts 2021-09:
  25 of 123 OOS months have zero momentum coverage. The measurable edge is
  `illiq_12m_xs` (t = +3.02), `log_me_xs` (t = −2.96) and `ivol_xs` (t = +1.73)
  — illiquid, small and volatile, which is exactly what a survivor-only
  universe inflates. Treat the level of that edge as unproven until the
  point-in-time universe item above is done.
