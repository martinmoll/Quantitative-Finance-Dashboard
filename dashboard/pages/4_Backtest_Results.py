# dashboard/pages/4_Backtest_Results.py
"""Page 4: Backtest Results & Diagnostics — tabbed layout."""

import re

import streamlit as st
import pandas as pd
import numpy as np
from core.diagnostics import (
    compute_performance_metrics, compute_ic_stats, fundamental_law, feature_ic,
    compute_r2_oos, bootstrap_sharpe_ci, bootstrap_alpha_ci, multiple_testing_hurdle,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio,
    probability_of_backtest_overfitting, survivorship_premium,
    return_by_vol_decile, effective_bets,
)
from core.risk import factor_alpha
from components.charts import (
    cumulative_wealth_chart, drawdown_chart, monthly_heatmap, bar_chart,
)
from components.theory import theory_section
from components.interpretations import (
    interpret_sharpe, interpret_max_drawdown,
    interpret_ff5_alpha, interpret_r2_oos, interpret_turnover,
)
from components.workflow import STAGES, render_empty_state, render_next_steps, config_label
from components.theme import inject_theme, inject_ledger_theme, LEDGER as L
from core.currency import to_nok_unhedged
from core.data_loader import region_for_label
from core.portfolio import net_returns_at_cost
import plotly.graph_objects as go

st.set_page_config(page_title="Backtest Results", layout="wide")


# ---------------------------------------------------------------------------
# Ledger render helpers. This page is the pilot for the Ledger design, so the
# helpers stay here until other pages use them too.
# ---------------------------------------------------------------------------
_BANNER = {
    "success": ("&#10003;", L["positive"]),
    "info": ("i", L["primary"]),
    "warning": ("!", L["warning"]),
    "error": ("&#10005;", L["negative"]),
}

# Pinned configs. No orange, which is the benchmark color.
_PIN_LINES = ["#5B2A86", "#0B6E4F", "#3B4048", "#8A5A00"]


def _banner(kind, headline_html, detail=None):
    """Ledger version of components.metrics.banner.

    The headline goes inside raw HTML, where Streamlit does not parse
    Markdown. Interpretation text marks bold with **, so convert it here.
    """
    icon, accent = _BANNER.get(kind, _BANNER["info"])
    headline_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", headline_html)
    st.markdown(
        f'<div style="background:{L["surface"]};border:1px solid {L["hairline"]};border-radius:4px;'
        f'padding:10px 14px;display:flex;align-items:center;gap:10px;margin:8px 0;">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;'
        f'width:22px;height:22px;border-radius:3px;background:{accent};color:#FFFFFF;'
        f'font-size:12px;font-weight:700;">{icon}</span>'
        f'<span style="font-size:14px;line-height:1.45;color:{L["text"]};">{headline_html}</span></div>',
        unsafe_allow_html=True,
    )
    if detail:
        with st.expander("Why this matters"):
            st.markdown(detail)


def _interp(interp):
    """Ledger version of components.interpretations.render_interpretation."""
    if not interp or not interp.get("text"):
        return
    sentences = interp["text"].split(". ", 1)
    _banner(interp.get("level", "info"), sentences[0].rstrip("."),
            sentences[1] if len(sentences) > 1 else None)


def _kpi_grid(cards):
    """Hairline grid of figures. Each card has label and value, optional sub and color."""
    tiles = "".join(
        f'<div style="background:{L["surface"]};padding:16px 18px;display:flex;flex-direction:column;gap:4px;">'
        f'<span style="font-size:12px;font-weight:500;letter-spacing:.04em;text-transform:uppercase;'
        f'color:{L["text_secondary"]};">{c["label"]}</span>'
        f'<span class="mono" style="font-size:26px;font-weight:500;letter-spacing:-.02em;line-height:1.15;'
        f'color:{c.get("color", L["text"])};">{c["value"]}</span>'
        f'<span class="mono" style="font-size:12px;color:{L["text_secondary"]};">{c.get("sub") or "&nbsp;"}</span>'
        f'</div>'
        for c in cards
    )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;'
        f'background:{L["hairline"]};border:1px solid {L["hairline"]};margin:4px 0 8px;">{tiles}</div>',
        unsafe_allow_html=True,
    )


def _workflow_steps(current_stage):
    """Ledger version of components.workflow.render_workflow_status.

    Same stages and completion checks. Only the colors change, because the
    navy-theme colors fail 4.5:1 contrast on the light background.
    """
    done = [s["complete_check"]() for s in STAGES]
    parts = []
    for i, stage in enumerate(STAGES):
        if stage["key"] == current_stage:
            circle = f'background:{L["primary"]};color:#FFFFFF;'
            label = f'color:{L["primary"]};font-weight:600;'
            symbol = stage["icon"]
        elif done[i]:
            circle = f'background:{L["positive"]};color:#FFFFFF;'
            label = f'color:{L["positive"]};'
            symbol = "&#10003;"
        else:
            circle = f'background:{L["surface"]};color:{L["text_secondary"]};border:1px solid {L["hairline"]};'
            label = f'color:{L["text_secondary"]};'
            symbol = stage["icon"]
        parts.append(
            f'<div style="text-align:center;flex:1;">'
            f'<div style="width:30px;height:30px;border-radius:50%;{circle}display:inline-flex;'
            f'align-items:center;justify-content:center;font-size:13px;font-weight:600;">{symbol}</div>'
            f'<div style="font-size:12px;margin-top:4px;{label}">{stage["label"]}</div></div>'
        )
        if i < len(STAGES) - 1:
            line = L["positive"] if done[i] and done[i + 1] else L["hairline"]
            parts.append(f'<div style="flex:0.5;height:2px;background:{line};align-self:center;margin-top:-10px;"></div>')
    st.markdown(
        f'<nav aria-label="Workflow" style="display:flex;align-items:flex-start;justify-content:space-between;'
        f'padding:10px 0 14px;margin-bottom:8px;">{"".join(parts)}</nav>',
        unsafe_allow_html=True,
    )


# Same formats as components.metrics.comparison_table.
_COMPARE_FMT = {
    "SR": "{:.2f}", "Ann Return": "{:.1%}", "Ann Vol": "{:.1%}", "MDD": "{:.1%}",
    "Calmar": "{:.2f}", "Total Return": "{:.0%}", "Mean IC": "{:.4f}",
    "Mean Turnover": "{:.1%}", "TC Drag": "{:.2%}", "Net SR": "{:.2f}",
}


def _table(df, formats=None):
    """HTML table that Ledger CSS can style.

    st.dataframe draws its cells on a canvas with the app-wide theme, which
    page CSS cannot reach. The trade-off: no sorting, search or download.
    """
    df = df.copy()
    for col, fmt in (formats or {}).items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "-")
    st.table(df)


def _card_head(title, sub=None):
    sub_html = f'<p style="margin:2px 0 0;font-size:13px;color:{L["text_secondary"]};">{sub}</p>' if sub else ""
    st.markdown(
        f'<div><h3 style="margin:0;padding:0;font-size:18px;font-weight:600;color:{L["text"]};">{title}</h3>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _plot(fig, key):
    """Plot with the Ledger template.

    Even with theme=None, the Streamlit frontend writes the config.toml colors
    into the layout's backgrounds and font when the figure leaves them unset.
    Those win over the template, so they are set here explicitly.
    """
    fig.update_layout(
        template="ledger", paper_bgcolor=L["surface"], plot_bgcolor=L["surface"],
        font=dict(family="DM Mono, monospace", size=11, color=L["text_secondary"]),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def _ledger_bars(fig, series, signed=True):
    """Recolor a components.charts.bar_chart figure for Ledger."""
    colors = [L["primary"] if v >= 0 else L["negative"] for v in series.values] if signed else L["neutral"]
    fig.update_traces(marker_color=colors, opacity=1)
    fig.update_shapes(line_color=L["text"])
    fig.update_annotations(font_color=L["text_secondary"])
    return fig


def _ledger_lines(fig, fill_first=False):
    """Recolor a wealth or drawdown figure: strategy blue, SPY orange, gross dashed."""
    pin_i = 0
    for tr in fig.data:
        if tr.name == "Strategy":
            tr.line = dict(color=L["primary"], width=2.25 if not fill_first else 1.5)
            if fill_first:
                tr.fillcolor = "rgba(31,63,191,0.16)"
        elif tr.name == "Gross":
            tr.line = dict(color=L["primary"], width=1.5, dash="dash")
            tr.opacity = 0.5
        elif tr.name == "SPY":
            tr.line = dict(color=L["benchmark"], width=1.75)
        else:
            tr.line = dict(color=_PIN_LINES[pin_i % len(_PIN_LINES)], width=1.5)
            pin_i += 1
    fig.update_layout(title=None, margin=dict(t=36, b=36, l=64, r=16))
    return fig


def _heatmap(rets):
    fig = monthly_heatmap(rets)
    fig.update_traces(colorscale=[[0.0, L["negative"]], [0.5, "#F7F7F5"], [1.0, L["primary"]]])
    return fig

# The shared empty state has inline navy-theme colors, which fail contrast on
# the Ledger canvas. Keep the navy theme until there is a result to show.
if render_empty_state("results"):
    inject_theme()
    st.stop()
inject_ledger_theme()

result = st.session_state.get("backtest_result")
market = st.session_state.get("market_monthly")
pinned = st.session_state.get("pinned_configs", [])


@st.cache_data(show_spinner=False)
def _load_usdnok():
    """Monthly USDNOK return series, cached across reruns."""
    from pipeline.fetchers.fx import fetch_usdnok_monthly
    return fetch_usdnok_monthly()

# --- Config Selector ---
# Selected by index (0 = current run, 1..N = pinned) so two pins with identical
# labels never collide the way a label-keyed map does.
current_config = {
    "result": result,
    "params": st.session_state.get("backtest_params", {}),
    "predictions": st.session_state.get("backtest_predictions"),
}
configs = [current_config] + list(pinned)


def _config_name(i):
    return "Current Run" if i == 0 else pinned[i - 1]["label"]


# --- Header ---
active_idx = 0
params = st.session_state.get("backtest_params", {})

head_left, head_right = st.columns([2, 1], vertical_alignment="bottom")
with head_left:
    region = st.session_state.get("active_region", "")
    eyebrow = "Backtest" + (f" &middot; {region}" if region else "")
    model_name = params.get("model_name", "—")
    strategy = params.get("strategy_type", "long_only").replace("_", "-")
    window = params.get("window_type", "expanding")
    rets_all = result["monthly_returns"]
    subtitle = f"{model_name} &middot; {strategy} &middot; {window} &middot; {rets_all.index.min()} &rarr; {rets_all.index.max()}"
    st.markdown(
        f'<p class="mono" style="margin:0;font-size:12px;letter-spacing:.06em;text-transform:uppercase;'
        f'color:{L["text_secondary"]};">{eyebrow}</p>'
        f'<h1 style="margin:4px 0 0;padding:0;font-size:44px;font-weight:600;letter-spacing:-.025em;'
        f'line-height:1.05;color:{L["text"]};">Backtest Results</h1>'
        f'<p style="margin:6px 0 0;font-size:14px;color:{L["text_secondary"]};">{subtitle}</p>',
        unsafe_allow_html=True,
    )
with head_right:
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        if len(configs) > 1:
            active_idx = st.selectbox(
                "View config", range(len(configs)), format_func=_config_name,
                label_visibility="collapsed",
            )
    with r_col2:
        pin_clicked = st.button("Pin to compare", type="primary")

_workflow_steps("results")

# --- Survivorship warning -------------------------------------------------
# The universe is today's index constituents projected backwards, so every
# number on this page is measured on a book that could not have been held at
# the time. Show the size of that head start before any performance figure.
_panel = st.session_state.get("df")
_oos = st.session_state.get("backtest_params", {}).get("oos_start")
if _panel is not None:
    _surv = survivorship_premium(_panel, start=_oos)
    if _surv:
        _banner(
            "warning",
            f"<b>Survivorship bias: +{_surv['gap_ann']:.1%} per year.</b> "
            f"Holding <i>every</i> name in this universe returned "
            f"{_surv['panel_ann']:.1%}/yr against a benchmark of "
            f"{_surv['bench_ann']:.1%}/yr, before any model. "
            f"Universe: {_surv['names_first']} names at the start, "
            f"{_surv['names_last']} at the end.",
            detail=(
                "The universe is **today's** index constituents projected "
                "backwards. Companies that were dropped, acquired or went "
                "bankrupt are absent, so the backtest picks only from names "
                "that survived. No investor could have held this list at the "
                "time.\n\n"
                "Treat the gap above as a **floor**, not a point estimate. It "
                "understates the true effect, because the same surviving names "
                "also dominate the training data. Subtract it from any Sharpe "
                "or return figure on this page before you act on it.\n\n"
                "Fixing this properly needs point-in-time index membership and "
                "delisting returns — see `ROADMAP.md` Tier 1."
            ),
        )

        with st.expander("Where the bias lives: return by volatility decile"):
            _dec = return_by_vol_decile(
                _panel,
                holdings=(result or {}).get("holdings"),
                start=_oos,
            )
            if not _dec.empty:
                st.caption(
                    "Deciles are formed within each month, so the split itself "
                    "carries no look-ahead. In a point-in-time universe the "
                    "highest-volatility decile should **not** reliably "
                    "outperform — those are the names that blow up, and the "
                    "ones that did are missing here. A large positive spread "
                    "is measuring the survivor filter, not a risk premium."
                )
                _show = _dec.copy()
                _show["ann_return"] = (_show["ann_return"] * 100).round(1)
                if "share_of_book" in _show.columns:
                    _show["share_of_book"] = (_show["share_of_book"] * 100).round(1)
                _table(
                    _show.rename(columns={
                        "ann_return": "Universe return %/yr",
                        "n_obs": "Observations",
                        "share_of_book": "Share of book %",
                    }),
                    {"Universe return %/yr": "{:.1f}", "Share of book %": "{:.1f}"},
                )
                _spread = (_dec["ann_return"].iloc[-1] - _dec["ann_return"].iloc[0]) * 100
                st.caption(f"Top decile minus bottom: **{_spread:.1f}** pct pts/yr.")

# --- Concentration: effective number of bets ------------------------------
# The position count cannot see this. In July 2026 the book held nine of ten
# names in the semiconductor supply chain and lost 30.8% gross while the market
# rose 1.4%. It read 1.73 effective bets at the decision point, the 2nd
# percentile of its own history — a warning that was available in advance.
_enb = effective_bets(
    (result or {}).get("holdings", {}), st.session_state.get("returns_history"),
)
_enb = _enb.dropna() if len(_enb) else _enb
if len(_enb) >= 12:
    _floor = _enb.quantile(0.10)
    _latest, _latest_m = _enb.iloc[-1], _enb.index[-1]
    if _latest <= _floor:
        _banner(
            "warning",
            f"<b>Concentration: {_latest:.1f} effective bets</b> in "
            f"<span class='mono'>{_latest_m}</span>, the bottom decile of this "
            f"backtest (median <span class='mono'>{_enb.median():.1f}</span>). "
            f"The book holds more positions than it holds bets.",
            detail=(
                "Effective bets is `(mean asset vol / portfolio vol)²` on the "
                "held book, from the point-in-time covariance. Ten names that "
                "move as one is one bet; the position count still says ten.\n\n"
                "Read it as verification, not forecast. With a sector cap "
                "active this metric stops predicting returns (correlation with "
                "the next month falls from +0.20 to +0.04) because the "
                "constraint has already removed the exposure it detects. A low "
                "reading *while a cap is on* means the book has found a way to "
                "concentrate that the cap does not catch."
            ),
        )
    with st.expander("Concentration over time: effective number of bets"):
        st.caption(
            "Estimated only on returns realized up to and including each "
            "month, so the reading is available **at** the decision, not after "
            "it. The flat line is the bottom decile of this backtest."
        )
        _fig = _ledger_bars(bar_chart(_enb, name="Effective bets", mean_line=True), _enb, signed=False)
        _fig.add_hline(y=_floor, line_dash="dot", line_color=L["negative"],
                       annotation_text=f"10th pct = {_floor:.1f}",
                       annotation_font_color=L["negative"])
        _plot(_fig, key="overview_effective_bets")

active = configs[active_idx]
active_result = active["result"]
active_params = active.get("params", {})
is_current_run = active_idx == 0

rets = active_result["monthly_returns"]
ic = active_result["ic"]
turnover = active_result["turnover"]

# --- Display Controls (sidebar) ---
st.sidebar.header("Display Controls")
oos_months = sorted(rets.index)
display_start = st.sidebar.select_slider("Display from", options=oos_months, value=oos_months[0])
start_value = st.sidebar.number_input("Starting value ($)", min_value=1, value=10000, step=1000)
cash_flow = st.sidebar.number_input("Cash flow/period ($)", value=0, step=100)

# Reporting currency — a Norwegian investor funding USD purchases with NOK also
# earns the USDNOK move. Only offered for the USD (US) book; the Oslo book is
# already in NOK. Hedged/attribution follow-ups live in FUTURE_IMPROVEMENTS.md.
fx_ret = None
if region_for_label(st.session_state.get("active_region", "")) == "US":
    currency_mode = st.sidebar.radio(
        "Reporting currency",
        ["USD", "NOK (unhedged)"],
        help="Restate returns in NOK by compounding each month's USD return with "
             "the USDNOK move. Unhedged — no forward hedging applied.",
    )
    if currency_mode.startswith("NOK"):
        try:
            fx_ret = _load_usdnok()
        except Exception as e:  # network fetch can fail — degrade to USD
            st.sidebar.warning(f"USDNOK fetch failed ({e}); showing USD.")

rets = rets[rets.index >= display_start]
ic = ic[ic.index >= display_start]
turnover = turnover[turnover.index >= display_start]

if fx_ret is not None:
    rets = to_nok_unhedged(rets, fx_ret)

# --- KPI data ---
perf = compute_performance_metrics(rets)
ff5 = st.session_state.get("ff5_factors")
alpha_res = factor_alpha(rets, ff5) if ff5 is not None else None

# --- Bootstrap CIs ---
sr_ci = bootstrap_sharpe_ci(rets)
alpha_ci = bootstrap_alpha_ci(rets, ff5) if ff5 is not None else None

# --- Tabs ---
tab_overview, tab_signal, tab_attribution, tab_costs, tab_compare = st.tabs(
    ["Overview", "Signal quality", "Attribution", "Turnover & costs", "Compare"]
)

# =========================================================================
# OVERVIEW TAB
# =========================================================================
with tab_overview:
    if fx_ret is not None:
        _banner("info",
                'Returns shown in <span class="mono">NOK</span> — unhedged',
                "Restated for a Norwegian investor: each month compounds the USD "
                "return with the USDNOK move. The FF5 alpha still regresses on USD "
                "factors, so in NOK mode it also absorbs currency swings — read it "
                "as indicative. Hedged NOK and full FX attribution are planned "
                "(see FUTURE_IMPROVEMENTS.md).")

    spy = None
    if market is not None:
        spy = market.loc[market.index >= display_start, "spy_ret"]
        if fx_ret is not None:
            spy = to_nok_unhedged(spy, fx_ret)
    bench = compute_performance_metrics(spy) if spy is not None else None

    # KPI grid
    psr = probabilistic_sharpe_ratio(rets)
    mean_to = turnover.dropna().mean()
    kpi_cards = [
        {"label": "Sharpe", "value": f"{perf['SR']:.2f}",
         "color": L["positive"] if perf["SR"] >= 1.0 else L["primary"] if perf["SR"] >= 0.5 else L["warning"],
         "sub": f"95% CI [{sr_ci['lo']:.2f}, {sr_ci['hi']:.2f}]" if not np.isnan(sr_ci["lo"]) else None},
        {"label": "Ann. Return", "value": f"{perf['Ann Return']:+.1%}",
         "color": L["positive"] if perf["Ann Return"] > 0 else L["negative"],
         "sub": f"SPY {bench['Ann Return']:+.1%}" if bench else None},
    ]
    if bench:
        excess = perf["Ann Return"] - bench["Ann Return"]
        kpi_cards.append({"label": "Excess / yr", "value": f"{excess:+.1%}", "sub": "vs SPY",
                          "color": L["positive"] if excess >= 0 else L["negative"]})
    kpi_cards += [
        {"label": "Ann. Vol", "value": f"{perf['Ann Vol']:.1%}",
         "sub": f"SPY {bench['Ann Vol']:.1%}" if bench else None},
        {"label": "Max DD", "value": f"{perf['MDD']:.1%}", "color": L["negative"],
         "sub": f"SPY {bench['MDD']:.1%}" if bench else None},
        {"label": "Calmar", "value": f"{perf['Calmar']:.2f}" if not np.isnan(perf['Calmar']) else "N/A",
         "color": L["positive"] if perf["Calmar"] >= 1.0 else L["warning"]},
        {"label": "Turnover / yr", "value": f"{mean_to * 12:.0%}" if not np.isnan(mean_to) else "N/A",
         "sub": "one-way"},
    ]
    if alpha_res is not None:
        sig = "**" if alpha_res["p_value"] < 0.01 else "*" if alpha_res["p_value"] < 0.05 else ""
        alpha_ci_str = None
        if alpha_ci is not None and not np.isnan(alpha_ci["lo"]):
            alpha_ci_str = f"95% CI [{alpha_ci['lo']:.1%}, {alpha_ci['hi']:.1%}]"
        kpi_cards.append({
            "label": "FF5 Alpha", "value": f"{alpha_res['annual_alpha']:.1%}{sig}", "sub": alpha_ci_str,
            "color": L["positive"] if alpha_res["annual_alpha"] > 0 and alpha_res["p_value"] < 0.05 else L["text"],
        })
    kpi_cards.append({
        "label": "Prob. Sharpe > 0", "value": f"{psr:.0%}" if not np.isnan(psr) else "N/A",
        "sub": "P(true Sharpe > 0)",
        "color": L["positive"] if (not np.isnan(psr) and psr > 0.95) else L["warning"],
    })
    if pinned:
        trial_srs = []
        for cfg in [current_config] + list(pinned):
            cr = cfg["result"]["monthly_returns"]
            cr = cr[cr.index >= display_start].dropna()
            if cr.std() > 0:
                trial_srs.append(cr.mean() / cr.std())
        dsr_res = deflated_sharpe_ratio(rets, trial_srs)
        dsr = dsr_res["dsr"]
        kpi_cards.append({
            "label": "Deflated Sharpe", "value": f"{dsr:.0%}" if not np.isnan(dsr) else "N/A",
            "sub": f"deflated for {dsr_res['n_trials']} trials",
            "color": L["positive"] if (not np.isnan(dsr) and dsr > 0.95) else L["warning"],
        })
    _kpi_grid(kpi_cards)

    # Compact interpretation banner — include CI in headline
    sr_val = perf["SR"]
    ci_note = f' <span class="mono">[{sr_ci["lo"]:.2f}, {sr_ci["hi"]:.2f}]</span>' if not np.isnan(sr_ci["lo"]) else ""
    interp = interpret_sharpe(sr_val)
    if interp and interp.get("text"):
        sentences = interp["text"].split(". ", 1)
        headline = sentences[0].rstrip(".") + ci_note
        detail = sentences[1] if len(sentences) > 1 else None
        zero_in_ci = not np.isnan(sr_ci["lo"]) and sr_ci["lo"] <= 0
        if zero_in_ci and detail:
            detail += (" Note: the 95% bootstrap CI includes zero, meaning the positive "
                       "Sharpe is not statistically distinguishable from chance at this sample size.")
        _banner(interp["level"], headline, detail)

    if alpha_res is not None:
        # Multiple-testing nudge: comparing pinned configs inflates false positives.
        if pinned:
            n_trials = len(pinned) + 1  # pinned configs + the current run
            hurdle = multiple_testing_hurdle(n_trials)
            t_obs = abs(alpha_res["t_stat"])
            if t_obs >= hurdle:
                _banner("success",
                        f'Comparing <span class="mono">{n_trials}</span> configs — alpha t-stat '
                        f'<span class="mono">{alpha_res["t_stat"]:.2f}</span> still clears the '
                        f'multiple-testing hurdle (|t| &gt; <span class="mono">{hurdle:.2f}</span>)',
                        "Bonferroni raises the bar when you pick the best of several strategies. "
                        "This alpha survives the correction — evidence it is not just the luckiest "
                        "of the configs you tried.")
            else:
                _banner("warning",
                        f'Comparing <span class="mono">{n_trials}</span> configs — alpha t-stat '
                        f'<span class="mono">{alpha_res["t_stat"]:.2f}</span> is below the '
                        f'multiple-testing hurdle (|t| &gt; <span class="mono">{hurdle:.2f}</span>)',
                        "Every config you pin is another test, so the naive |t| &gt; 1.96 bar no "
                        "longer holds. Bonferroni requires this tighter hurdle to keep the "
                        "family-wise false-positive rate at 5% — this alpha may be the luckiest "
                        "draw rather than genuine skill. See the theory below.")
        theory_section("Is It Really Alpha? Measuring Statistical Confidence", "alpha_significance")
        theory_section("Jensen's Alpha — Do You Have Skill?", "jensens_alpha")

    # build_portfolio_series stores the gross series. An older pinned config
    # may not have it; then the gross line and the cost panel are left out.
    run_bps = active_result.get("cost_bps")
    gross_usd = active_result.get("monthly_returns_gross")
    if gross_usd is not None:
        gross_usd = gross_usd[gross_usd.index >= display_start]

    returns_dict = {"Strategy": rets}
    if gross_usd is not None and run_bps:
        returns_dict["Gross"] = to_nok_unhedged(gross_usd, fx_ret) if fx_ret is not None else gross_usd
    if spy is not None:
        returns_dict["SPY"] = spy
    for p in pinned:
        p_rets = p["result"]["monthly_returns"]
        p_rets = p_rets[p_rets.index >= display_start]
        if fx_ret is not None:
            p_rets = to_nok_unhedged(p_rets, fx_ret)
        returns_dict[p["label"]] = p_rets

    eq_col, cost_col = st.columns([2, 1], gap="medium")
    with eq_col:
        with st.container(key="ledger-card-equity"):
            net_note = f"net of {run_bps:g} bps one-way" if run_bps is not None else "net of costs"
            _card_head("Equity curve vs benchmark", f"Growth of ${start_value:,}, {net_note}")
            fig = _ledger_lines(cumulative_wealth_chart(returns_dict, start_value, cash_flow))
            fig.update_layout(height=380)
            _plot(fig, key="overview_wealth")

    with cost_col:
        with st.container(key="ledger-card-cost"):
            _card_head("Cost-adjusted Sharpe", "Annualised. One-way cost per unit traded.")
            if gross_usd is None:
                st.caption("Unavailable for this configuration: no gross return series is stored.")
            else:
                turnover_usd = active_result["turnover"]
                turnover_usd = turnover_usd[turnover_usd.index >= display_start]

                def _perf_at(bps):
                    net = net_returns_at_cost(gross_usd, turnover_usd, bps)
                    return compute_performance_metrics(to_nok_unhedged(net, fx_ret) if fx_ret is not None else net)

                base_bps = float(run_bps or 0.0)
                levels = sorted({0.0, 5.0, 10.0, 20.0, 30.0, 50.0, base_bps})
                # One key per config. Streamlit ignores a new default on a key it
                # already holds, so a shared key would keep the last config's level.
                chosen = st.segmented_control(
                    "Cost level", levels, default=base_bps,
                    format_func=lambda b: f"{b:g} bps", key=f"cost_level_{active_idx}",
                    label_visibility="collapsed",
                )
                chosen = base_bps if chosen is None else chosen
                srs = {b: _perf_at(b)["SR"] for b in levels}
                gross_perf, chosen_perf = _perf_at(0.0), _perf_at(chosen)
                top = max((v for v in srs.values() if not np.isnan(v)), default=np.nan)

                def _num(v, fmt):
                    # Zero volatility or an empty series gives a NaN Sharpe.
                    return "N/A" if np.isnan(v) else format(v, fmt)

                rows = ""
                for b, v in srs.items():
                    width = max(0.0, v / top * 100) if (top > 0 and not np.isnan(v)) else 0.0
                    sel = b == chosen
                    rows += (
                        f'<div style="display:grid;grid-template-columns:64px minmax(0,1fr) 48px;align-items:center;'
                        f'gap:12px;padding:6px 10px;border-radius:3px;background:{"#EEF1FC" if sel else "transparent"};">'
                        f'<span class="mono" style="font-size:13px;color:{L["text_secondary"]};">{b:g} bps</span>'
                        f'<span style="height:8px;background:#F0F0EC;"><span style="display:block;height:100%;'
                        f'width:{width:.1f}%;background:{L["primary"] if sel else "#B9BCC4"};"></span></span>'
                        f'<span class="mono" style="font-size:13px;text-align:right;color:{L["text"]};">'
                        f'{_num(v, ".2f")}</span></div>'
                    )
                # Break-even is the one-way cost that takes the mean gross return
                # to zero. It is a property of the trades, so it is computed in
                # the base currency also in NOK mode.
                traded = turnover_usd.reindex(gross_usd.index).fillna(1.0)
                g_mean = gross_usd.mean()
                break_even = f"{g_mean / (2 * traded.mean()) * 1e4:.0f} bps" if g_mean > 0 else "none (gross ≤ 0)"
                if fx_ret is not None:
                    break_even += " (USD)"
                drag = gross_perf["Ann Return"] - chosen_perf["Ann Return"]
                st.markdown(
                    f'<div style="display:flex;align-items:baseline;gap:12px;margin:4px 0 8px;">'
                    f'<span class="mono" style="font-size:40px;font-weight:500;letter-spacing:-.03em;'
                    f'color:{L["primary"]};">{_num(chosen_perf["SR"], ".2f")}</span>'
                    f'<span class="mono" style="font-size:13px;color:{L["text_secondary"]};">'
                    f'net · gross {_num(gross_perf["SR"], ".2f")}</span></div>'
                    f'<div style="display:flex;flex-direction:column;gap:2px;">{rows}</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;'
                    f'background:{L["hairline"]};border:1px solid {L["hairline"]};margin-top:14px;">'
                    f'<div style="background:{L["surface"]};padding:12px 14px;">'
                    f'<div style="font-size:12px;color:{L["text_secondary"]};">Cost drag / yr</div>'
                    f'<div class="mono" style="font-size:18px;color:{L["text"]};">{_num(drag, ".2%")}</div></div>'
                    f'<div style="background:{L["surface"]};padding:12px 14px;">'
                    f'<div style="font-size:12px;color:{L["text_secondary"]};">Break-even cost</div>'
                    f'<div class="mono" style="font-size:18px;color:{L["text"]};">{break_even}</div></div></div>',
                    unsafe_allow_html=True,
                )

    dd_col, to_col = st.columns(2, gap="medium")
    with dd_col:
        with st.container(key="ledger-card-drawdown"):
            dd = (1 + rets).cumprod()
            dd = dd / dd.cummax() - 1
            _card_head("Drawdown", f"Max {perf['MDD']:.1%} · trough {dd.idxmin()}" if len(dd) else None)
            dd_dict = {k: v for k, v in returns_dict.items() if k != "Gross"}
            fig = _ledger_lines(drawdown_chart(dd_dict, start_value), fill_first=True)
            fig.update_layout(height=280, yaxis_title=None)
            _plot(fig, key="overview_dd")
    with to_col:
        with st.container(key="ledger-card-turnover"):
            to_sub = (f"One-way per rebalance · mean {mean_to:.0%} · {mean_to * 12:.0%} / yr"
                      if not np.isnan(mean_to) else None)
            _card_head("Turnover", to_sub)
            to_clean = turnover.dropna()
            fig = _ledger_bars(bar_chart(to_clean, name="Monthly turnover"), to_clean, signed=False)
            fig.update_layout(title=None, height=280, yaxis_title=None, yaxis_tickformat=".0%",
                              margin=dict(t=36, b=36, l=64, r=16))
            _plot(fig, key="overview_turnover")

    hm_col, fi_col = st.columns([2, 1], gap="medium")
    with hm_col:
        with st.container(key="ledger-card-heatmap"):
            _plot(_heatmap(rets), key="overview_heatmap")
    with fi_col:
        importance = active.get("feature_importance", st.session_state.get("backtest_feature_importance"))
        if importance is not None and len(importance) > 0:
            with st.container(key="ledger-card-importance"):
                top_imp = importance.nlargest(15, "importance")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_imp["importance"].values, y=top_imp.index,
                    orientation="h", marker_color=L["primary"],
                ))
                fig.update_layout(
                    title="Feature Importance (Top 15)",
                    height=400, margin=dict(t=40, b=40, l=150),
                    yaxis=dict(autorange="reversed"),
                )
                _plot(fig, key="overview_feat_imp")

# =========================================================================
# SIGNAL QUALITY TAB
# =========================================================================
with tab_signal:
    theory_section("Information Coefficient & Fundamental Law", "ic_and_fundamental_law")

    ic_col1, ic_col2 = st.columns(2)
    with ic_col1:
        ic_clean = ic.dropna()
        fig = _ledger_bars(bar_chart(ic_clean, name="Monthly IC"), ic_clean)
        _plot(fig, key="signal_ic_bar")
    with ic_col2:
        cum_ic = ic.dropna().cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_ic.index, y=cum_ic.values, name="Cumulative IC",
            line=dict(color=L["primary"], width=2),
        ))
        fig.update_layout(
            title="Cumulative IC", height=400,
            margin=dict(t=40, b=40),
        )
        _plot(fig, key="signal_cum_ic")

    ic_stats = compute_ic_stats(ic)
    _kpi_grid([
        {"label": "Mean IC", "value": f"{ic_stats['mean_ic']:.4f}",
         "color": L["positive"] if ic_stats["mean_ic"] > 0.03 else L["warning"]},
        {"label": "IC t-stat", "value": f"{ic_stats['ic_tstat']:.2f}",
         "color": L["positive"] if ic_stats["ic_tstat"] > 2 else L["warning"]},
        {"label": "ICIR", "value": f"{ic_stats['icir']:.3f}",
         "color": L["positive"] if ic_stats["icir"] > 0.3 else L["warning"]},
        {"label": "Hit Rate", "value": f"{ic_stats['hit_rate']:.1%}",
         "color": L["positive"] if ic_stats["hit_rate"] > 0.55 else L["warning"]},
    ])

    t = ic_stats["ic_tstat"]
    if abs(t) >= 2.58:
        _banner("success",
                f'IC t-stat = <span class="mono">{t:.2f}</span> — highly significant (p < 0.01)',
                "The model's predictive signal is statistically reliable and very unlikely "
                "to be due to chance. This is strong evidence of genuine forecasting ability.")
    elif abs(t) >= 1.96:
        _banner("info",
                f'IC t-stat = <span class="mono">{t:.2f}</span> — significant at 5% level',
                "The model's signal is statistically distinguishable from zero, suggesting "
                "real predictive content. Verify with out-of-sample performance before relying on it.")
    elif abs(t) >= 1.65:
        _banner("warning",
                f'IC t-stat = <span class="mono">{t:.2f}</span> — marginally significant',
                "There is weak evidence of a signal. The IC could plausibly be zero — "
                "consider increasing the backtest window or simplifying the model.")
    else:
        _banner("error",
                f'IC t-stat = <span class="mono">{t:.2f}</span> — not significant',
                "The model's predictions are statistically indistinguishable from random. "
                "Consider changing the feature set, model, or retraining frequency.")

    # Fundamental Law
    st.markdown("---")
    st.subheader("Fundamental Law of Active Management")
    fl_col1, fl_col2 = st.columns(2)
    with fl_col1:
        sr_target = st.number_input("SR Target", value=1.0, step=0.1, key="fl_sr_target")
        tc_bps = st.number_input("Transaction Cost (bps)", value=10.0, step=5.0, key="fl_tc_bps")
    with fl_col2:
        K_val = active_params.get("K", 10)
        fl = fundamental_law(ic_stats["mean_ic"], K_val, sr_target=sr_target, tc_bps=tc_bps)
        _kpi_grid([
            {"label": "Breadth", "value": f"{fl['BR_nominal']}"},
            {"label": "IR Upper Bound", "value": f"{fl['IR_upper_bound']:.3f}"},
            {"label": "IC Required", "value": f"{fl['IC_required']:.4f}"},
            {"label": "Cost SR", "value": f"{fl['Cost_SR']:.3f}"},
        ])

    # R2 OOS
    st.markdown("---")
    st.subheader("Out-of-Sample R² (Campbell & Thompson 2008)")
    predictions = active.get("predictions")
    if predictions:
        r2_oos = compute_r2_oos(predictions)
        r2_col1, r2_col2 = st.columns([1, 3])
        with r2_col1:
            _kpi_grid([{"label": "R²_OOS", "value": f"{r2_oos:.4f}" if not np.isnan(r2_oos) else "N/A",
                        "color": L["positive"] if r2_oos > 0 else L["warning"]}])
        with r2_col2:
            _interp(interpret_r2_oos(r2_oos))
    else:
        st.caption("R²_OOS is unavailable for this configuration (no stored predictions).")

# =========================================================================
# ATTRIBUTION TAB
# =========================================================================
with tab_attribution:
    _interp(interpret_max_drawdown(perf["MDD"], perf["Calmar"]))
    if alpha_res is not None:
        _interp(interpret_ff5_alpha(
            alpha_res["annual_alpha"], alpha_res["t_stat"],
            alpha_res["p_value"], alpha_res["r_squared"],
        ))

    # Feature Importance (full view)
    theory_section("Feature Importance", "feature_importance")
    importance = active.get("feature_importance", st.session_state.get("backtest_feature_importance"))
    if not is_current_run and importance is None:
        st.info("Feature importance is only available for the most recent run.")
    if importance is not None and len(importance) > 0:
        st.subheader("Feature Importance (Top 20)")
        top_imp = importance.nlargest(20, "importance")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_imp["importance"].values, y=top_imp.index,
            orientation="h", marker_color=L["primary"],
        ))
        fig.update_layout(
            title="Feature Importance",
            height=500, margin=dict(t=40, b=40, l=150),
            yaxis=dict(autorange="reversed"),
        )
        _plot(fig, key="attrib_feat_imp")

# =========================================================================
# TURNOVER & COSTS TAB
# =========================================================================
with tab_costs:
    hm_col1, hm_col2 = st.columns(2)
    with hm_col1:
        _plot(_heatmap(rets), key="costs_heatmap")
    with hm_col2:
        to_clean = turnover.dropna()
        fig = _ledger_bars(bar_chart(to_clean, name="Monthly Turnover"), to_clean, signed=False)
        _plot(fig, key="costs_turnover")

    mean_to = turnover.dropna().mean()
    _interp(interpret_turnover(
        mean_to, 10.0, perf["SR"],
        ic_mean=compute_ic_stats(ic)["mean_ic"],
        K=active_params.get("K", 10),
    ))

# =========================================================================
# COMPARE TAB
# =========================================================================
with tab_compare:
    if pinned:
        st.subheader("Strategy Comparison")
        configs = []

        cur_result = st.session_state.get("backtest_result")
        cur_rets = cur_result["monthly_returns"]
        cur_rets = cur_rets[cur_rets.index >= display_start]
        if fx_ret is not None:
            cur_rets = to_nok_unhedged(cur_rets, fx_ret)
        cur_ic = cur_result["ic"]
        cur_ic = cur_ic[cur_ic.index >= display_start]
        cur_perf = compute_performance_metrics(cur_rets)
        cur_perf["Strategy"] = "Current Run"
        cur_perf["Mean IC"] = cur_ic.mean()
        cur_perf["Mean Turnover"] = cur_result["turnover"].dropna().mean()
        configs.append(cur_perf)

        for p in pinned:
            p_rets = p["result"]["monthly_returns"]
            p_rets = p_rets[p_rets.index >= display_start]
            if fx_ret is not None:
                p_rets = to_nok_unhedged(p_rets, fx_ret)
            p_perf = compute_performance_metrics(p_rets)
            p_perf["Strategy"] = p["label"]
            p_ic = p["result"]["ic"]
            p_perf["Mean IC"] = p_ic[p_ic.index >= display_start].mean()
            p_perf["Mean Turnover"] = p["result"]["turnover"].dropna().mean()
            configs.append(p_perf)

        _table(pd.DataFrame(configs).set_index("Strategy"), _COMPARE_FMT)

        # --- Probability of Backtest Overfitting (CSCV) ---
        ret_cols = {}
        _cur = st.session_state.get("backtest_result")["monthly_returns"]
        ret_cols["Current Run"] = _cur[_cur.index >= display_start]
        for p in pinned:
            pr = p["result"]["monthly_returns"]
            ret_cols[p["label"]] = pr[pr.index >= display_start]
        pbo = probability_of_backtest_overfitting(pd.DataFrame(ret_cols))

        st.markdown("---")
        st.subheader("Probability of Backtest Overfitting")
        if not np.isnan(pbo):
            _kpi_grid([{
                "label": "PBO", "value": f"{pbo:.0%}",
                "color": L["negative"] if pbo > 0.5 else L["positive"],
                "sub": "P(backtest winner disappoints out-of-sample)",
            }])
            _banner("warning" if pbo > 0.5 else "success",
                    f'PBO = <span class="mono">{pbo:.0%}</span>',
                    "Across combinatorial in-sample / out-of-sample splits, how often the "
                    "in-sample-best config landed below the out-of-sample median. Above 50% "
                    "means picking the backtest winner is likely overfitting. Most reliable "
                    "with several configs and a long history.")
        else:
            st.caption("Need at least 2 configs and enough history to estimate PBO.")
    else:
        st.info("Pin configurations from the Alpha Model Lab to compare them here.")

# --- Pin action ---
if pin_clicked:
    cur_result = st.session_state.get("backtest_result")
    cur_params = st.session_state.get("backtest_params")
    cur_pinned = st.session_state.get("pinned_configs", [])
    if cur_result is not None and len(cur_pinned) < 4:
        label = config_label(cur_params)
        cur_pinned.append({
            "label": label, "result": cur_result, "params": cur_params,
            "predictions": st.session_state.get("backtest_predictions"),
        })
        st.session_state.pinned_configs = cur_pinned
        st.rerun()

render_next_steps("results")
