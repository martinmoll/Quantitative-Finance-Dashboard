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
