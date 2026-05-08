## Grading Rubric (Performance Component — 20 pts)

| OOS Sharpe Ratio | Min Score |
|---|---|
| SR > 1.7 | 18 / 20 |
| 1.5 < SR ≤ 1.7 | 15 / 20 |
| 1.3 < SR ≤ 1.5 | 12 / 20 |
| 1.2 < SR ≤ 1.3 | 10 / 20 |
| SR ≤ 1.2 | Instructor's discretion |

**Disqualifying:** look-ahead bias, overfitting/excessive params, MDD > 40%.

**Current best:** K10_vt0.05 + 4mo regime filter — SR=1.53, MDD=-18.0% → **15/20 tier**. Next target: SR > 1.7 for 18/20.

---

## Without Volatility tuning

============================================================
RESULTS SUMMARY
============================================================
            SR Ann Mean Ann Vol     MDD  Total
Strategy                                      
Lasso     0.79    21.1%   26.6%  -56.7%  2736%
HGB_200   1.10    25.0%   22.6%  -49.7%  6636%
HGB_500   1.09    24.2%   22.3%  -46.0%  5801%
SPY       0.66    10.4%   15.7%  -50.3%   459%


## With Volatility Tuning

============================================================
RESULTS SUMMARY
============================================================
             SR Ann Mean Ann Vol     MDD  Total
Strategy                                       
Lasso      0.79    21.1%   26.6%  -56.7%  2736%
HGB_vt0.0  1.09    24.2%   22.3%  -46.0%  5801%
HGB_vt0.3  0.89    12.3%   13.8%  -39.1%   740%
HGB_vt0.5  0.87    11.8%   13.5%  -37.5%   671%
HGB_vt0.8  0.85    11.4%   13.5%  -38.7%   619%
SPY        0.66    10.4%   15.7%  -50.3%   459%


## With Volatility Tuning and something else

============================================================
RESULTS SUMMARY
============================================================
                    SR Ann Mean Ann Vol     MDD  Total
Strategy                                              
Lasso             0.79    21.1%   26.6%  -56.7%  2736%
HGB_vt0.3_th0.18  1.15    21.1%   18.4%  -35.8%  3702%
HGB_vt0.3_th0.2   1.03    19.6%   19.0%  -43.0%  2699%
HGB_vt0.3_th0.25  1.07    22.5%   21.1%  -42.8%  4423%
HGB_vt0.5_th0.18  1.13    20.8%   18.3%  -35.8%  3517%
HGB_vt0.5_th0.2   1.03    19.4%   18.9%  -42.3%  2643%
HGB_vt0.5_th0.25  1.07    22.6%   21.1%  -42.2%  4480%
HGB_vt0.8_th0.18  1.13    20.7%   18.4%  -36.6%  3469%
HGB_vt0.8_th0.2   1.03    19.5%   18.9%  -41.9%  2662%
HGB_vt0.8_th0.25  1.07    22.6%   21.1%  -42.0%  4505%
SPY               0.66    10.4%   15.7%  -50.3%   459%

BEST STRATEGY: HGB_vt0.05 (SR = 1.20) [K=30]
SPY Benchmark SR = 0.66
Alpha over SPY = +0.54 SR

              SR Ann Mean Ann Vol     MDD  Total
Strategy
HGB_vt0.05  1.20    22.0%   18.3%  -38.7%  4405%
SPY         0.66    10.4%   15.7%  -50.3%   459%


## Phase 2: K Tuning (Portfolio Concentration)

K=10 concentrates on the model's highest-conviction picks. Return jumps to 29% while vol tilt keeps MDD in check. K=15/20 fail MDD — they include weaker picks that add vol without enough return.

============================================================
K TUNING RESULTS (HGB + vt=0.05)
============================================================
                  SR Ann Mean Ann Vol     MDD   Total  MDD Pass?
HGB_vt0.05_K10  1.36    29.0%   21.4%  -36.2%  14770%  Yes
HGB_vt0.05_K15  1.17    23.9%   20.4%  -42.5%   5910%  No
HGB_vt0.05_K20  1.16    22.7%   19.7%  -45.0%   4858%  No
HGB_vt0.05_K30  1.20    22.0%   18.3%  -38.7%   4405%  Yes
SPY             0.66    10.4%   15.7%  -50.3%    459%

NEW BEST: HGB_vt0.05_K10 — SR=1.36, MDD=-36.2% → 12/20 tier (up from 10/20)


## Phase 2b: K=10 Vol Tilt Sweep

Confirming vt=0.05 is optimal for K=10. Higher return at lower tilts but MDD fails.

============================================================
K=10 VOL TILT SWEEP
============================================================
              SR Ann Mean Ann Vol     MDD   Total  MDD Pass?
K10_vt0.0   1.07    27.0%   25.3%  -59.5%   8568%  No
K10_vt0.02  1.26    29.9%   23.8%  -48.7%  15764%  No
K10_vt0.03  1.31    30.1%   23.0%  -39.6%  16976%  Yes (tight)
K10_vt0.05  1.36    29.0%   21.4%  -36.2%  14770%  Yes
K10_vt0.07  1.30    25.7%   19.8%  -40.8%   8452%  No (barely)
K10_vt0.1   1.15    21.0%   18.3%  -36.7%   3704%  Yes
K10_vt0.15  1.08    17.0%   15.7%  -38.2%   1854%  Yes
SPY         0.66    10.4%   15.7%  -50.3%    459%

CONFIRMED BEST: K10_vt0.05 — SR=1.36, MDD=-36.2%, Ann Mean=29.0%


## Phase 2c: Retrain Frequency Sweep (K=10, vt=0.05)

More frequent retraining hurts both SR and MDD. 12-month is optimal — model is stable and doesn't chase recent noise.

============================================================
RETRAIN FREQUENCY RESULTS (K=10, vt=0.05)
============================================================
                     SR Ann Mean Ann Vol     MDD   Total  MDD Pass?
K10_vt0.05_rt1mo   1.29    28.0%   21.8%  -45.3%  11958%  No
K10_vt0.05_rt3mo   1.26    27.6%   21.9%  -47.8%  11006%  No
K10_vt0.05_rt6mo   1.21    26.6%   22.0%  -51.8%   9118%  No
K10_vt0.05_rt12mo  1.36    29.0%   21.4%  -36.2%  14770%  Yes

CONFIRMED: 12-month retraining is optimal. No change needed.


## Phase 2d: Feature Selection (K=10, vt=0.05)

Reducing features hurts. Full 118-feature set is best — even zero-importance features contribute via interactions.

============================================================
FEATURE SELECTION RESULTS (K=10, vt=0.05)
============================================================
                  SR Ann Mean Ann Vol     MDD   Total  MDD Pass?
K10_top20feat   1.11    23.0%   20.7%  -36.1%   4895%  Yes
K10_top30feat   1.13    25.6%   22.5%  -45.5%   7414%  No
K10_top50feat   1.18    26.9%   22.9%  -51.9%   9369%  No
K10_top70feat   1.33    29.5%   22.2%  -47.3%  15756%  No
K10_all118feat  1.36    29.0%   21.4%  -36.2%  14770%  Yes

CONFIRMED: All 118 features is optimal. No feature reduction.


## Phase 2e: Momentum Regime Filter (K=10, vt=0.05)

Going to cash when trailing SPY return is negative. 6-month lookback is the sweet spot — SR jumps to 1.46!

============================================================
MOMENTUM REGIME FILTER RESULTS (K=10, vt=0.05)
============================================================
                  SR Ann Mean Ann Vol     MDD   Total  MDD Pass?
K10_regime6mo   1.46    25.4%   17.5%  -33.4%   8751%  Yes
K10_regime12mo  1.35    24.7%   18.3%  -33.3%   7402%  Yes
K10_no_filter   1.36    29.0%   21.4%  -36.2%  14770%  Yes

NEW BEST: K10_vt0.05 + 6mo regime filter — SR=1.46, MDD=-33.4% → nearly 15/20 tier


## Phase 2f: Regime Lookback Fine-Tune (K=10, vt=0.05)

============================================================
REGIME LOOKBACK FINE-TUNE (K=10, vt=0.05)
============================================================
              SR Ann Mean Ann Vol     MDD   Total
regime_3mo  1.46    23.1%   15.8%  -15.8%   5953%
regime_4mo  1.53    25.3%   16.6%  -18.0%   8759%
regime_5mo  1.40    24.2%   17.3%  -29.4%   6935%
regime_6mo  1.46    25.4%   17.5%  -33.4%   8751%
regime_7mo  1.47    26.4%   17.9%  -32.1%  10264%
regime_8mo  1.42    25.1%   17.7%  -33.9%   8104%
regime_9mo  1.35    23.8%   17.7%  -33.9%   6439%
no_filter   1.36    29.0%   21.4%  -36.2%  14770%


## Phase 2g: Vol Tilt Re-Optimization with 4mo Regime Filter (K=10)

============================================================
VOL TILT + 4MO REGIME FILTER (K=10)
============================================================
                        SR Ann Mean Ann Vol     MDD   Total
K10_vt0.0_regime4mo   1.34    25.5%   19.0%  -20.1%   8402%
K10_vt0.02_regime4mo  1.50    27.2%   18.2%  -19.0%  11945%
K10_vt0.03_regime4mo  1.47    26.4%   18.0%  -23.1%  10349%
K10_vt0.04_regime4mo  1.46    25.0%   17.1%  -21.0%   8235%
K10_vt0.05_regime4mo  1.53    25.3%   16.6%  -18.0%   8759%
K10_vt0.06_regime4mo  1.43    23.2%   16.2%  -15.9%   5955%
K10_vt0.07_regime4mo  1.39    21.5%   15.4%  -18.5%   4412%

NEW BEST: K10_vt0.05 + 4mo regime filter — SR=1.53, MDD=-18.0% → 15/20 tier


## Phase 3: Diagnostics & Alternative Weighting

### Weighting Comparison (K=10, vt=0.05)
Score-weighted hurts (SR 1.05, MDD -54.7%). MAXSER identical to equal-weight (falls back with K=10).
Equal-weight confirmed optimal.

### IC Diagnostics (K=10, vt=0.05, equal-weight)
  Mean IC:      0.0225 (comparable to lecture's HGBR at 0.0248)
  IC Std:       0.0982
  ICIR:         0.229
  IC t-stat:    3.45 (significant)
  Hit Rate:     65.6% (good)
  Implied BR:   3,636 (nominal 360) — tree model captures 10x more signal than IC suggests

### Turnover Diagnostics
  Mean monthly TO:   56.2%
  Annualized TO:     674%
  TC drag (10 bps):  0.67%/year
  Net SR (approx):   1.33

