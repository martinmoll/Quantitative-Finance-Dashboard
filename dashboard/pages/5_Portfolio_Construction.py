# dashboard/pages/5_Portfolio_Construction.py
"""Page 5: Portfolio Construction & Risk Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
from core.portfolio import build_portfolio_series
from core.diagnostics import compute_performance_metrics
from core.risk import (
    risk_contribution, factor_exposure, rolling_factor_exposure,
    transaction_cost_drag,
)
from components.charts import (
    sector_allocation_chart, risk_pie_chart, bar_chart, STYLE,
)
from components.metrics import metric_row, comparison_table
from components.theory import theory_section
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Construction", layout="wide")
st.title("Portfolio Construction & Risk")

result = st.session_state.get("backtest_result")
predictions = st.session_state.get("backtest_predictions")
params = st.session_state.get("backtest_params")
ff5 = st.session_state.get("ff5_factors")
market = st.session_state.get("market_monthly")

if result is None or predictions is None:
    st.info("Run a backtest on the **Alpha Model Lab** page first.")
    st.stop()

theory_section("Portfolio Construction Methods", "portfolio_construction")

# --- Method Comparison ---
st.header("Compare Construction Methods")
methods = ["equal_weight", "score_weight", "inverse_vol", "erc", "mvo"]
selected_methods = st.multiselect("Methods to compare", methods, default=["equal_weight", "inverse_vol"])

method_results = {}
for method in selected_methods:
    port = build_portfolio_series(
        predictions=predictions, method=method,
        K=params["K"], strategy_type=params["strategy_type"],
        K_short=params["K_short"], vol_tilt=params["vol_tilt"],
        regime_lookback=params["regime_lookback"], market_monthly=market,
    )
    method_results[method] = port

# --- Sector Allocation ---
st.header("Sector Allocation")
if result["holdings"]:
    fig = sector_allocation_chart(result["holdings"], params["strategy_type"])
    st.plotly_chart(fig, use_container_width=True)

# --- Holdings Table ---
st.header("Current Holdings")
if result["holdings"]:
    last_month = sorted(result["holdings"].keys())[-1]
    held = result["holdings"][last_month].copy()
    display_cols = []
    for c in ["side", "permno", "sector", "pred", "y_raw", "weight"]:
        if c in held.columns:
            display_cols.append(c)
    show = held[display_cols].copy()
    if "pred" in show.columns:
        show["pred"] = show["pred"].round(4)
    if "y_raw" in show.columns:
        show["y_raw"] = (show["y_raw"] * 100).round(2).astype(str) + "%"
    if "weight" in show.columns:
        show["weight"] = (show["weight"] * 100).round(2).astype(str) + "%"
    st.markdown(f"**{last_month}** ({len(held)} positions)")
    st.dataframe(show.reset_index(drop=True), use_container_width=True)

# --- Risk Decomposition ---
if len(selected_methods) >= 1:
    st.markdown("---")
    st.header("Risk Decomposition")
    rc_cols = st.columns(len(selected_methods))
    for i, method in enumerate(selected_methods):
        with rc_cols[i]:
            st.subheader(method)
            port = method_results[method]
            if port["holdings"]:
                last_m = sorted(port["holdings"].keys())[-1]
                held = port["holdings"][last_m]
                if "weight" in held.columns:
                    w = held["weight"].abs().values
                    n = len(w)
                    cov = np.eye(n) * 0.01
                    rc = risk_contribution(w / w.sum(), cov)
                    rc.index = held["permno"].values
                    fig = risk_pie_chart(rc)
                    st.plotly_chart(fig, use_container_width=True)

# --- Factor Exposure ---
if ff5 is not None:
    st.markdown("---")
    st.header("Factor Exposure")
    theory_section("Turnover and Transaction Costs", "turnover_and_costs")

    rets = result["monthly_returns"]
    exposure = factor_exposure(rets, ff5)
    if len(exposure) > 0:
        exp_cols = st.columns(len(exposure))
        for i, (factor, beta) in enumerate(exposure.items()):
            with exp_cols[i]:
                color = "normal" if abs(beta) < 0.10 else "inverse"
                st.metric(factor, f"{beta:.3f}", delta_color=color)

    rolling_exp = rolling_factor_exposure(rets, ff5, window=24)
    if not rolling_exp.dropna(how="all").empty:
        fig = go.Figure()
        for col in rolling_exp.columns:
            fig.add_trace(go.Scatter(
                x=rolling_exp.index, y=rolling_exp[col].values, name=col,
            ))
        fig.add_hline(y=0.10, line_dash="dash", line_color=STYLE["warning"])
        fig.add_hline(y=-0.10, line_dash="dash", line_color=STYLE["warning"])
        fig.update_layout(
            title="Rolling 24-Month Factor Exposures",
            template=STYLE["template"], height=400,
            paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Turnover & Costs ---
st.markdown("---")
st.header("Turnover & Transaction Costs")
tc_col1, tc_col2 = st.columns(2)
with tc_col1:
    fig = bar_chart(result["turnover"].dropna(), name="Monthly Turnover")
    st.plotly_chart(fig, use_container_width=True)
with tc_col2:
    cost_bps = st.number_input("Cost per trade (bps)", value=10.0, step=5.0, key="tc_bps")
    ann_vol = result["monthly_returns"].std() * np.sqrt(12)
    tc = transaction_cost_drag(result["turnover"], cost_bps, ann_vol)
    perf = compute_performance_metrics(result["monthly_returns"])
    gross_sr = perf["SR"]
    net_sr = gross_sr - tc["Cost_SR"]
    metric_row([
        {"label": "Avg Monthly TO", "value": f"{tc['mean_monthly_turnover']:.1%}"},
        {"label": "TC Drag (annual)", "value": f"{tc['TC_annual']:.2%}"},
        {"label": "Gross SR", "value": f"{gross_sr:.2f}"},
        {"label": "Net SR", "value": f"{net_sr:.2f}"},
    ])

# --- Method Comparison Table ---
if len(method_results) >= 2:
    st.markdown("---")
    st.header("Method Comparison")
    comp = []
    for method, port in method_results.items():
        p = compute_performance_metrics(port["monthly_returns"])
        p["Strategy"] = method
        p["Mean Turnover"] = port["turnover"].dropna().mean()
        ann_vol_m = port["monthly_returns"].std() * np.sqrt(12)
        tc_m = transaction_cost_drag(port["turnover"], cost_bps, ann_vol_m)
        p["TC Drag"] = tc_m["TC_annual"]
        p["Net SR"] = p["SR"] - tc_m["Cost_SR"]
        comp.append(p)
    comparison_table(comp)
