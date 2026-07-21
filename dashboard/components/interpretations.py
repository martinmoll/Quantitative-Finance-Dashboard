"""Inline theory-grounded interpretations for every key metric.

Each function takes a metric value (and optional context) and returns a dict
with 'level' (success/info/warning/error) and 'text' (the interpretation).
Pages call ``render_interpretation()`` to display the banner.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
from components.metrics import banner


def render_interpretation(interp: dict | None):
    if interp is None:
        return
    level = interp.get("level", "info")
    text = interp.get("text", "")
    if not text:
        return
    sentences = text.split(". ", 1)
    headline = sentences[0].rstrip(".")
    detail = sentences[1] if len(sentences) > 1 else None
    banner(level, headline, detail)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def interpret_sharpe(sr: float, ir_upper: float | None = None) -> dict:
    if np.isnan(sr):
        return {"level": "info", "text": "Sharpe Ratio unavailable (zero volatility)."}

    base = f"**Sharpe Ratio = {sr:.2f}.**"
    tc_note = ""
    if ir_upper is not None and not np.isnan(ir_upper) and ir_upper > 0:
        tc = sr / ir_upper
        tc_note = (
            f" The Fundamental Law upper bound is IR = {ir_upper:.2f}, giving "
            f"an implied Transfer Coefficient of TC = SR / IR = {tc:.2f} "
            f"(1.0 = no signal lost to constraints; <0.5 = significant leakage "
            f"from position limits, long-only constraints, or turnover costs)."
        )

    if sr >= 1.5:
        return {"level": "success", "text": (
            f"{base} Exceptional risk-adjusted return. SRs this high in live "
            f"trading are rare and warrant scrutiny for overfitting or look-ahead "
            f"bias — most published academic strategies have out-of-sample SRs "
            f"of 0.5–1.0.{tc_note}"
        )}
    if sr >= 1.0:
        return {"level": "success", "text": (
            f"{base} Strong risk-adjusted return. An SR above 1.0 means each "
            f"unit of volatility is more than compensated by return — this is "
            f"the threshold many allocators consider institutional-grade.{tc_note}"
        )}
    if sr >= 0.5:
        return {"level": "info", "text": (
            f"{base} Moderate risk-adjusted return. SRs of 0.5–1.0 are typical "
            f"for well-constructed equity long-short strategies. The Fundamental "
            f"Law suggests improving either IC (forecast quality) or breadth "
            f"(number of independent bets) to push higher.{tc_note}"
        )}
    if sr >= 0:
        return {"level": "warning", "text": (
            f"{base} Weak positive return per unit risk. After transaction costs, "
            f"the net SR may be near zero. Consider whether the alpha signal is "
            f"strong enough to justify trading costs.{tc_note}"
        )}
    return {"level": "error", "text": (
        f"{base} Negative risk-adjusted return — the strategy loses money on "
        f"average. A negative SR means the model's predictions are either "
        f"uncorrelated or inversely correlated with realized returns.{tc_note}"
    )}


def interpret_max_drawdown(mdd: float, calmar: float) -> dict:
    if np.isnan(mdd):
        return {"level": "info", "text": "Max drawdown unavailable."}

    mdd_pct = abs(mdd)
    base = f"**Max Drawdown = {mdd:.1%}, Calmar = {calmar:.2f}.**"

    calmar_note = ""
    if not np.isnan(calmar):
        recovery_years = 1 / calmar if calmar > 0 else float("inf")
        if recovery_years < 10:
            calmar_note = (
                f" Calmar Ratio = Ann. Return / |MDD|; at current returns it "
                f"would take ~{recovery_years:.1f} years of expected returns to "
                f"recover from this drawdown."
            )

    if mdd_pct <= 0.10:
        return {"level": "success", "text": (
            f"{base} Shallow drawdown — strong capital preservation. "
            f"Drawdowns under 10% are typical for low-volatility or "
            f"well-hedged strategies.{calmar_note}"
        )}
    if mdd_pct <= 0.20:
        return {"level": "info", "text": (
            f"{base} Moderate drawdown, consistent with a diversified equity "
            f"strategy. For context, the S&P 500's typical intra-year drawdown "
            f"is ~14%.{calmar_note}"
        )}
    if mdd_pct <= 0.35:
        return {"level": "warning", "text": (
            f"{base} Significant drawdown. Losses of this magnitude test "
            f"investor discipline — behaviorally, most allocators reduce "
            f"exposure after a 20–25% drawdown, locking in losses.{calmar_note}"
        )}
    return {"level": "error", "text": (
        f"{base} Severe drawdown exceeding 35%. This level of capital loss "
        f"requires a {mdd_pct / (1 - mdd_pct):.0%} gain just to break even "
        f"(the asymmetry of losses). Risk management or position sizing "
        f"should be reviewed.{calmar_note}"
    )}


def interpret_r_squared(r2: float, context: str = "factor") -> dict:
    if np.isnan(r2):
        return {"level": "info", "text": "R² unavailable."}

    idio = (1 - r2) * 100
    if context == "factor":
        base = f"**R² = {r2:.3f}.**"
        return {"level": "info", "text": (
            f"{base} {r2:.0%} of return variation is explained by factor "
            f"exposures; the remaining {idio:.0f}% is idiosyncratic. "
            f"This idiosyncratic component is where alpha models operate — "
            f"a high R² means most returns come from factor tilts (replicable "
            f"cheaply via ETFs), while a low R² means the portfolio is driven "
            f"by stock-specific bets."
        )}
    return {"level": "info", "text": f"**R² = {r2:.3f}** — {r2:.0%} of variance explained."}


def interpret_ff5_alpha(
    annual_alpha: float, t_stat: float, p_value: float, r2: float,
) -> dict:
    base = f"**FF5 Alpha = {annual_alpha:.2%}/year (t={t_stat:.2f}, p={p_value:.4f}).**"
    if p_value < 0.05 and annual_alpha > 0:
        return {"level": "success", "text": (
            f"{base} Statistically significant positive alpha — returns exceed "
            f"what the five Fama-French factors explain. This is the textbook "
            f"definition of manager skill: returns that cannot be replicated by "
            f"any passive combination of market, size, value, profitability, and "
            f"investment factors. Academic benchmark: monthly alpha of 0.3–0.5% "
            f"(3.6–6.0%/year) with t > 2 is considered strong."
        )}
    if annual_alpha > 0 and p_value >= 0.05:
        return {"level": "warning", "text": (
            f"{base} Positive but not statistically significant. With t < 2.0, "
            f"we cannot distinguish this alpha from sampling noise at the 5% "
            f"level. More out-of-sample months or a more consistent signal "
            f"would help. Do not rely on this alpha for allocation decisions."
        )}
    if annual_alpha <= 0 and p_value < 0.05:
        return {"level": "error", "text": (
            f"{base} Statistically significant *negative* alpha. After removing "
            f"factor exposures, the portfolio underperforms. The factor bets "
            f"embedded in the portfolio (R² = {r2:.2f}) would have done better "
            f"alone — the stock selection is destroying value."
        )}
    return {"level": "info", "text": (
        f"{base} Alpha is statistically indistinguishable from zero. "
        f"The portfolio's returns are fully explained by its factor exposures "
        f"(R² = {r2:.2f}). This isn't necessarily bad — you may be efficiently "
        f"harvesting factor premiums — but there is no evidence of "
        f"stock-selection skill."
    )}


# ---------------------------------------------------------------------------
# Turnover & costs
# ---------------------------------------------------------------------------

def interpret_turnover(
    monthly_to: float,
    cost_bps: float,
    gross_sr: float,
    ic_mean: float | None = None,
    K: int | None = None,
) -> dict:
    annual_to = monthly_to * 12
    annual_cost = monthly_to * cost_bps / 10000 * 2 * 12
    base = (
        f"**Avg monthly turnover = {monthly_to:.1%} "
        f"(~{annual_to:.0%} annualized).**"
    )
    cost_note = (
        f" At {cost_bps:.0f} bps/trade, annual cost drag ≈ {annual_cost:.2%}. "
    )

    fl_note = ""
    if ic_mean is not None and K is not None and ic_mean > 0:
        port_vol = 0.15
        cost_sr = annual_cost / port_vol
        br = K * 12
        ic_required = (gross_sr + cost_sr) / np.sqrt(br) if br > 0 else 0
        fl_note = (
            f"The Fundamental Law requires IC > {ic_required:.4f} to justify "
            f"this turnover (your IC = {ic_mean:.4f})."
        )
        if ic_mean < ic_required:
            return {"level": "warning", "text": (
                f"{base}{cost_note}{fl_note} Your IC is below the break-even "
                f"threshold — costs may erode the entire alpha."
            )}

    if monthly_to <= 0.15:
        return {"level": "success", "text": (
            f"{base}{cost_note}Low turnover — cost drag is minimal. "
            f"{fl_note}"
        )}
    if monthly_to <= 0.35:
        return {"level": "info", "text": (
            f"{base}{cost_note}Moderate turnover, typical for monthly-rebalanced "
            f"strategies. {fl_note}"
        )}
    return {"level": "warning", "text": (
        f"{base}{cost_note}High turnover. Consider score-weighted or "
        f"constrained optimization to reduce unnecessary trading. {fl_note}"
    )}


# ---------------------------------------------------------------------------
# VaR & tail risk
# ---------------------------------------------------------------------------

def interpret_var_comparison(
    parametric_var: float,
    historical_var: float,
    cf_var: float,
    skewness: float,
    excess_kurtosis: float,
) -> dict:
    spread = abs(cf_var - parametric_var) / parametric_var if parametric_var > 0 else 0

    base = (
        f"**Parametric VaR = {parametric_var:.2%} vs "
        f"Cornish-Fisher VaR = {cf_var:.2%}.**"
    )

    if abs(skewness) < 0.3 and abs(excess_kurtosis) < 1:
        return {"level": "success", "text": (
            f"{base} Returns are approximately normal (skew = {skewness:.2f}, "
            f"excess kurtosis = {excess_kurtosis:.2f}), so all VaR methods "
            f"agree. Parametric VaR is reliable here."
        )}

    tail_note = ""
    if skewness < -0.5:
        tail_note = (
            f"Negative skewness ({skewness:.2f}) means large losses are more "
            f"frequent than a normal distribution predicts. "
        )
    if excess_kurtosis > 1:
        tail_note += (
            f"Fat tails (excess kurtosis = {excess_kurtosis:.2f}) mean extreme "
            f"events — both gains and losses — are more likely than the normal "
            f"assumption implies. "
        )

    if spread > 0.20:
        return {"level": "warning", "text": (
            f"{base} The {spread:.0%} gap between methods signals significant "
            f"non-normality. {tail_note}Parametric VaR underestimates true tail "
            f"risk — prefer Cornish-Fisher or Historical VaR for risk limits."
        )}
    return {"level": "info", "text": (
        f"{base} {tail_note}The methods are broadly consistent, but monitor "
        f"for periods where non-normality increases (crisis regimes)."
    )}


def interpret_kupiec(breach_rate: float, expected_rate: float, p_value: float) -> dict:
    base = (
        f"**Breach rate = {breach_rate:.1%} vs expected {expected_rate:.1%} "
        f"(Kupiec p = {p_value:.3f}).**"
    )
    if p_value > 0.05:
        return {"level": "success", "text": (
            f"{base} The VaR model is well-calibrated — actual breach frequency "
            f"is statistically consistent with the target confidence level. "
            f"This means the model neither underestimates nor overestimates "
            f"tail risk."
        )}
    if breach_rate > expected_rate:
        return {"level": "error", "text": (
            f"{base} Too many breaches — the model underestimates risk. "
            f"In practice this means the portfolio experiences unexpected losses "
            f"more often than the VaR level promises. Consider using "
            f"Cornish-Fisher adjustment or increasing the confidence level."
        )}
    return {"level": "warning", "text": (
        f"{base} Too few breaches — the model is overly conservative. "
        f"While this is safer, it means capital reserves are larger than "
        f"necessary, reducing capital efficiency."
    )}


def interpret_skewness_kurtosis(skewness: float, excess_kurtosis: float) -> dict:
    issues = []
    if skewness < -0.5:
        issues.append(
            f"negative skew ({skewness:.2f}) — left tail is heavier, meaning "
            f"large losses are more frequent than large gains"
        )
    elif skewness > 0.5:
        issues.append(
            f"positive skew ({skewness:.2f}) — right tail is heavier"
        )

    if excess_kurtosis > 3:
        issues.append(
            f"high kurtosis ({excess_kurtosis:.2f}) — extreme events are "
            f"~{excess_kurtosis / 3:.1f}× more likely than a normal distribution"
        )
    elif excess_kurtosis > 1:
        issues.append(
            f"moderate kurtosis ({excess_kurtosis:.2f}) — tails are fatter "
            f"than normal"
        )

    if not issues:
        return {"level": "info", "text": (
            f"**Distribution shape:** approximately normal (skew = {skewness:.2f}, "
            f"excess kurtosis = {excess_kurtosis:.2f}). Standard parametric "
            f"risk measures are appropriate."
        )}

    return {"level": "warning", "text": (
        f"**Distribution shape:** {'; '.join(issues)}. "
        f"These departures from normality mean parametric VaR (which assumes "
        f"a bell curve) will underestimate tail risk. Use Cornish-Fisher or "
        f"Historical VaR for more accurate risk budgets."
    )}


# ---------------------------------------------------------------------------
# Factor exposure on portfolio
# ---------------------------------------------------------------------------

def interpret_factor_exposure(exposures: dict[str, float]) -> dict:
    """Interpret portfolio-level factor loadings from FF5 regression."""
    if not exposures:
        return {"level": "info", "text": "No factor exposures available."}

    tilts = []
    for factor, beta in exposures.items():
        if abs(beta) < 0.10:
            continue
        direction = "positive" if beta > 0 else "negative"
        label = _factor_label(factor, beta)
        tilts.append(f"{factor} = {beta:.3f} ({label})")

    if not tilts:
        return {"level": "success", "text": (
            "**Factor exposures are all within ±0.10.** The portfolio is "
            "approximately factor-neutral — returns are driven by "
            "stock-specific (idiosyncratic) bets rather than systematic "
            "factor tilts. This is the ideal for a pure alpha strategy."
        )}

    return {"level": "info", "text": (
        f"**Non-trivial factor tilts detected:** {'; '.join(tilts)}. "
        f"These exposures mean part of your return comes from factor "
        f"premiums you could harvest more cheaply via ETFs. To isolate "
        f"pure alpha, consider hedging out these tilts or using a "
        f"factor-neutral portfolio construction method."
    )}


def _factor_label(factor: str, beta: float) -> str:
    labels = {
        "Mkt-RF": ("levered market exposure" if beta > 0
                    else "defensive / low-beta tilt"),
        "SMB": ("small-cap tilt — historically compensated but "
                "carries liquidity risk" if beta > 0
                else "large-cap tilt"),
        "HML": ("value tilt — long cheap, short expensive" if beta > 0
                else "growth tilt"),
        "RMW": ("quality/profitability tilt" if beta > 0
                else "tilt toward low-profitability firms"),
        "CMA": ("conservative investment tilt" if beta > 0
                else "aggressive growth / high-investment tilt"),
    }
    return labels.get(factor, f"{'positive' if beta > 0 else 'negative'} loading")


# ---------------------------------------------------------------------------
# R² out-of-sample
# ---------------------------------------------------------------------------

def interpret_r2_oos(r2_oos: float) -> dict:
    base = f"**R²_OOS = {r2_oos:.4f} ({r2_oos * 100:.2f}%).**"

    if np.isnan(r2_oos):
        return {"level": "info", "text": "R²_OOS unavailable."}

    scale_note = (
        "R²_OOS is measured on the *standardized* return the model forecasts, so "
        "it is on the same scale as IC² — cross-sectional return R² is tiny even "
        "for good models."
    )

    if r2_oos < -0.02:
        return {"level": "warning", "text": (
            f"{base} The forecast *magnitudes* are miscalibrated — predictions are "
            f"more extreme than realized returns justify, so squared errors exceed "
            f"the naive 'predict the mean' forecast. {scale_note} Crucially this "
            f"does **not** invalidate a positive IC: the model can rank stocks "
            f"correctly (what the portfolio actually uses) while getting the scale "
            f"of its predictions wrong. Judge selection skill by IC; read R²_OOS as "
            f"a magnitude-calibration check."
        )}
    if r2_oos < 0:
        return {"level": "info", "text": (
            f"{base} Roughly matches the naive mean forecast — typical for equity "
            f"returns, whose magnitudes are close to unpredictable month to month. "
            f"{scale_note} A positive IC can still coexist with this; ranking, not "
            f"magnitude, is what drives the portfolio."
        )}
    if r2_oos < 0.02:
        return {"level": "success", "text": (
            f"{base} Positive out-of-sample — the forecast genuinely beats "
            f"predicting the cross-sectional mean, and its magnitudes are "
            f"reasonably calibrated. {scale_note} 0–2% corresponds to a forecast "
            f"correlation up to ~0.14, which is strong for monthly equity returns."
        )}
    return {"level": "success", "text": (
        f"{base} Unusually high for cross-sectional equity prediction — it implies "
        f"a forecast correlation above ~0.14. Sanity-check for look-ahead bias or a "
        f"leaked target before trusting it. {scale_note}"
    )}


# ---------------------------------------------------------------------------
# Monitoring metrics
# ---------------------------------------------------------------------------

def interpret_ks_shift(pct_flagged: float, n_flagged: int, n_total: int) -> dict:
    base = (
        f"**{n_flagged}/{n_total} features flagged "
        f"({pct_flagged:.0%}) with KS D > 0.10.**"
    )
    if pct_flagged < 0.05:
        return {"level": "success", "text": (
            f"{base} Feature distributions are stable — the current data "
            f"looks similar to training data. The model is operating within "
            f"its training domain."
        )}
    if pct_flagged < 0.20:
        return {"level": "info", "text": (
            f"{base} Some features have shifted, but fewer than the 20% "
            f"threshold for concern. Monitor these features — if they are "
            f"among the model's most important inputs, even a small shift "
            f"can degrade predictions."
        )}
    return {"level": "warning", "text": (
        f"{base} Exceeds the 20% threshold — the model is likely operating "
        f"out-of-distribution. KS D > 0.10 means the max distance between "
        f"training and current CDFs is large enough that model assumptions "
        f"may break down. Consider retraining on recent data or reducing "
        f"position sizes until the shift resolves."
    )}


def interpret_alpha_decay(half_life: int | None, ic_1m: float) -> dict:
    if half_life is None:
        if ic_1m > 0:
            return {"level": "success", "text": (
                f"**No half-life detected** — the signal's IC does not decay "
                f"below 50% of its 1-month value within 12 months. This "
                f"suggests a persistent alpha source, possibly tied to slow-moving "
                f"fundamentals rather than short-term momentum."
            )}
        return {"level": "info", "text": "Alpha decay analysis inconclusive (IC at 1m is near zero)."}

    base = f"**Signal half-life = {half_life} month(s).**"
    if half_life <= 2:
        return {"level": "warning", "text": (
            f"{base} Very fast decay — the signal loses half its predictive "
            f"power within {half_life} months. Monthly rebalancing is essential, "
            f"and even weekly might be warranted. Fast-decaying signals tend to "
            f"be momentum or sentiment driven."
        )}
    if half_life <= 6:
        return {"level": "info", "text": (
            f"{base} Moderate decay rate. Quarterly rebalancing captures most "
            f"of the signal value. The decay rate suggests a mix of fundamental "
            f"and technical drivers."
        )}
    return {"level": "success", "text": (
        f"{base} Slow decay — the signal retains value for many months, "
        f"consistent with fundamental (value, quality) rather than technical "
        f"(momentum) alpha sources. Less frequent rebalancing is acceptable, "
        f"which reduces turnover and costs."
    )}


# ---------------------------------------------------------------------------
# VIF interpretation
# ---------------------------------------------------------------------------

def interpret_vif(factor: str, vif_value: float) -> str:
    shared_pct = (1 - 1 / vif_value) * 100 if vif_value > 1 else 0
    if vif_value < 5:
        return f"VIF = {vif_value:.1f} — acceptable ({shared_pct:.0f}% variance shared)."
    if vif_value < 10:
        return (
            f"VIF = {vif_value:.1f} — moderate multicollinearity. "
            f"{shared_pct:.0f}% of {factor}'s variance is explained by other factors, "
            f"inflating its standard error and potentially masking a true loading."
        )
    return (
        f"VIF = {vif_value:.1f} — severe multicollinearity. "
        f"{shared_pct:.0f}% of {factor}'s variance is shared, making its "
        f"coefficient estimate unreliable. Consider dropping correlated factors."
    )
