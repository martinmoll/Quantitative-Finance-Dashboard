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
