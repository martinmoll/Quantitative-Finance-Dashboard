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
