# dashboard/pages/4_Backtest_Results.py
"""Page 4: Backtest Results & Diagnostics."""

import streamlit as st
import pandas as pd
import numpy as np
from core.diagnostics import (
    compute_performance_metrics, compute_ic_stats, fundamental_law, feature_ic,
)
from core.risk import factor_alpha
from components.charts import (
    cumulative_wealth_chart, drawdown_chart, monthly_heatmap,
    rolling_metric_chart, bar_chart, STYLE, PIN_COLORS,
)
from components.metrics import metric_row, comparison_table
from components.theory import theory_section
import plotly.graph_objects as go

st.set_page_config(page_title="Backtest Results", layout="wide")
st.title("Backtest Results & Diagnostics")

result = st.session_state.get("backtest_result")
if result is None:
    st.info("No backtest results yet. Run a backtest on the **Alpha Model Lab** page.")
    st.stop()

market = st.session_state.get("market_monthly")
pinned = st.session_state.get("pinned_configs", [])

rets = result["monthly_returns"]
ic = result["ic"]
turnover = result["turnover"]

# --- Display Controls ---
disp_col1, disp_col2, disp_col3 = st.columns(3)
with disp_col1:
    oos_months = sorted(rets.index)
    display_start = st.select_slider("Display from", options=oos_months, value=oos_months[0])
with disp_col2:
    start_value = st.number_input("Starting value ($)", min_value=1, value=10000, step=1000)
with disp_col3:
    cash_flow = st.number_input("Cash flow/period ($)", value=0, step=100)

rets = rets[rets.index >= display_start]
ic = ic[ic.index >= display_start]
turnover = turnover[turnover.index >= display_start]

# --- KPI Cards ---
perf = compute_performance_metrics(rets)
ff5 = st.session_state.get("ff5_factors")
alpha_res = factor_alpha(rets, ff5) if ff5 is not None else None

kpi_cards = [
    {"label": "Sharpe Ratio", "value": f"{perf['SR']:.2f}"},
    {"label": "Ann. Return", "value": f"{perf['Ann Return']:.1%}"},
    {"label": "Ann. Volatility", "value": f"{perf['Ann Vol']:.1%}"},
    {"label": "Max Drawdown", "value": f"{perf['MDD']:.1%}"},
    {"label": "Calmar", "value": f"{perf['Calmar']:.2f}" if not np.isnan(perf['Calmar']) else "N/A"},
]
if alpha_res is not None:
    sig = "*" if alpha_res["p_value"] < 0.05 else ""
    kpi_cards.append(
        {"label": f"FF5 Alpha{sig}", "value": f"{alpha_res['annual_alpha']:.2%}"}
    )
metric_row(kpi_cards)

if alpha_res is not None:
    theory_section("Jensen's Alpha — Do You Have Skill?", "jensens_alpha")

# --- Cumulative Wealth + Drawdown ---
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

returns_dict = {"Strategy": rets}
if market is not None:
    spy = market.loc[market.index >= display_start, "spy_ret"]
    returns_dict["SPY"] = spy
for p in pinned:
    p_rets = p["result"]["monthly_returns"]
    returns_dict[p["label"]] = p_rets[p_rets.index >= display_start]

with chart_col1:
    fig = cumulative_wealth_chart(returns_dict, start_value, cash_flow)
    st.plotly_chart(fig, use_container_width=True)
with chart_col2:
    fig = drawdown_chart(returns_dict, start_value)
    st.plotly_chart(fig, use_container_width=True)

# --- IC Dashboard ---
st.markdown("---")
theory_section("Information Coefficient & Fundamental Law", "ic_and_fundamental_law")

ic_col1, ic_col2 = st.columns(2)
with ic_col1:
    fig = bar_chart(ic.dropna(), name="Monthly IC")
    st.plotly_chart(fig, use_container_width=True)
with ic_col2:
    cum_ic = ic.dropna().cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_ic.index, y=cum_ic.values, name="Cumulative IC",
        line=dict(color=STYLE["accent"], width=2),
    ))
    fig.update_layout(
        title="Cumulative IC", template=STYLE["template"],
        height=400, margin=dict(t=40, b=40),
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
    )
    st.plotly_chart(fig, use_container_width=True)

ic_stats = compute_ic_stats(ic)
ic_metric_col1, ic_metric_col2, ic_metric_col3, ic_metric_col4 = st.columns(4)
ic_metric_col1.metric("Mean IC", f"{ic_stats['mean_ic']:.4f}",
                       delta="Good" if ic_stats["mean_ic"] > 0.03 else "Low")
ic_metric_col2.metric("IC t-stat", f"{ic_stats['ic_tstat']:.2f}",
                       delta="Sig" if ic_stats["ic_tstat"] > 2 else "Insig")
ic_metric_col3.metric("ICIR", f"{ic_stats['icir']:.3f}",
                       delta="Good" if ic_stats["icir"] > 0.3 else "Low")
ic_metric_col4.metric("Hit Rate", f"{ic_stats['hit_rate']:.1%}",
                       delta="Good" if ic_stats["hit_rate"] > 0.55 else "Low")

# --- Fundamental Law ---
st.markdown("---")
st.header("Fundamental Law of Active Management")
fl_col1, fl_col2 = st.columns(2)
with fl_col1:
    sr_target = st.number_input("SR Target", value=1.0, step=0.1)
    tc_bps = st.number_input("Transaction Cost (bps)", value=10.0, step=5.0)
with fl_col2:
    params = st.session_state.get("backtest_params", {})
    K_val = params.get("K", 10)
    fl = fundamental_law(ic_stats["mean_ic"], K_val, sr_target=sr_target, tc_bps=tc_bps)
    st.metric("Breadth (nominal)", f"{fl['BR_nominal']}")
    st.metric("IR Upper Bound", f"{fl['IR_upper_bound']:.3f}")
    st.metric("IC Required", f"{fl['IC_required']:.4f}")
    st.metric("Cost SR", f"{fl['Cost_SR']:.3f}")

# --- Monthly Heatmap + Turnover ---
st.markdown("---")
hm_col1, hm_col2 = st.columns(2)
with hm_col1:
    fig = monthly_heatmap(rets)
    st.plotly_chart(fig, use_container_width=True)
with hm_col2:
    fig = bar_chart(turnover.dropna(), name="Monthly Turnover")
    st.plotly_chart(fig, use_container_width=True)

# --- Feature Importance ---
st.markdown("---")
theory_section("Feature Importance", "feature_importance")
importance = st.session_state.get("backtest_feature_importance")
if importance is not None and len(importance) > 0:
    st.header("Feature Importance (Top 20)")
    top_imp = importance.nlargest(20, "importance")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_imp["importance"].values, y=top_imp.index,
        orientation="h", marker_color=STYLE["accent"],
    ))
    fig.update_layout(
        title="Feature Importance", template=STYLE["template"],
        height=500, margin=dict(t=40, b=40, l=150),
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Comparison Table ---
if pinned:
    st.markdown("---")
    st.header("Strategy Comparison")
    configs = []
    current_perf = compute_performance_metrics(rets)
    current_perf["Strategy"] = "Current"
    current_perf["Mean IC"] = ic_stats["mean_ic"]
    current_perf["Mean Turnover"] = turnover.dropna().mean()
    configs.append(current_perf)

    for p in pinned:
        p_rets = p["result"]["monthly_returns"]
        p_rets = p_rets[p_rets.index >= display_start]
        p_perf = compute_performance_metrics(p_rets)
        p_perf["Strategy"] = p["label"]
        p_ic = p["result"]["ic"]
        p_perf["Mean IC"] = p_ic[p_ic.index >= display_start].mean()
        p_perf["Mean Turnover"] = p["result"]["turnover"].dropna().mean()
        configs.append(p_perf)

    comparison_table(configs)
