# dashboard/pages/2_Factor_Analysis.py
"""Page 2: Factor Analysis — CAPM and Fama-French regressions."""

import streamlit as st
import pandas as pd
import numpy as np
from core.factor_models import (
    run_regression, rolling_beta, bloomberg_shrink_beta,
    compute_vif, wald_test, hedge_ratio,
)
from components.metrics import regression_table, vif_table, metric_card, metric_card_row
from components.charts import STYLE, rolling_metric_chart
from components.theory import theory_section
from components.interpretations import (
    render_interpretation, interpret_r_squared, interpret_vif,
)
from components.workflow import render_workflow_status, render_next_steps
from components.theme import inject_theme, COLORS, FONT_SANS
import plotly.graph_objects as go

st.set_page_config(page_title="Factor Analysis", layout="wide")
inject_theme()

C = COLORS
st.markdown(
    f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
    f'color:{C["text"]};margin:0;">Factor Analysis (CAPM & FF5)</h1>',
    unsafe_allow_html=True,
)
render_workflow_status("explore")

df = st.session_state.get("df")
ff5 = st.session_state.get("ff5_factors")
market = st.session_state.get("market_monthly")

if df is None:
    st.error("Dataset not loaded.")
    st.stop()

theory_section("CAPM and Factor Models", "capm_and_factors")

# --- Stock Selector ---
st.sidebar.header("Stock Selection")
sectors = sorted(df["sector"].dropna().unique()) if "sector" in df.columns else []
selected_sector = st.sidebar.selectbox("Filter by Sector", ["All"] + sectors)

if selected_sector != "All":
    filtered_permnos = df[df["sector"] == selected_sector]["permno"].unique()
else:
    filtered_permnos = df["permno"].unique()

selected_permno = st.sidebar.selectbox("Stock (permno)", sorted(filtered_permnos))

# --- Compute stock excess returns ---
stock_data = df[df["permno"] == selected_permno][["ym", "y_raw"]].dropna()
stock_data = stock_data.set_index("ym").sort_index()
stock_returns = stock_data["y_raw"]

if ff5 is not None:
    rf = ff5["RF"] if "RF" in ff5.columns else pd.Series(0, index=ff5.index)
    common_idx = stock_returns.index.intersection(ff5.index)
    excess_returns = stock_returns.loc[common_idx] - rf.loc[common_idx]
    factors = ff5.loc[common_idx]
else:
    excess_returns = stock_returns
    if market is not None:
        common_idx = stock_returns.index.intersection(market.index)
        excess_returns = stock_returns.loc[common_idx]
        factors = pd.DataFrame({"Mkt-RF": market.loc[common_idx, "Mkt_RF"]})
    else:
        st.warning("No factor data available.")
        st.stop()

# --- Regression Panel ---
st.header("Regression Results")
model_types = ["CAPM"]
if ff5 is not None:
    model_types.extend(["FF3", "FF5"])

reg_type = st.selectbox("Regression Model", model_types)

if len(excess_returns) > 20:
    reg_result = run_regression(excess_returns, factors, model_type=reg_type)
    regression_table(reg_result)
    render_interpretation(interpret_r_squared(reg_result["r_squared"], "factor"))
else:
    st.warning("Not enough data for regression (need > 20 months).")

# --- Rolling Beta ---
st.header("Rolling Beta")
window = st.slider("Rolling Window (months)", min_value=12, max_value=120, value=36)

if "Mkt-RF" in factors.columns and len(excess_returns) > window:
    beta_df = rolling_beta(excess_returns, factors["Mkt-RF"], window=window)
    shrunk = bloomberg_shrink_beta(beta_df["beta"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=beta_df.index, y=beta_df["upper"].values, mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=beta_df.index, y=beta_df["lower"].values, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(91,155,255,0.15)",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=beta_df.index, y=beta_df["beta"].values, name="OLS Beta",
        line=dict(color=C["primary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=beta_df.index, y=shrunk.values, name="Bloomberg Shrunk",
        line=dict(color=C["warning"], width=2, dash="dash"),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=C["text_muted"])
    fig.update_layout(
        title=f"Rolling {window}-Month Beta",
        yaxis_title="Beta", template="alpha",
        height=400, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- VIF Table ---
if ff5 is not None and reg_type == "FF5":
    st.header("Variance Inflation Factors")
    theory_section("VIF and Wald Tests", "vif_and_wald")
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    available = [c for c in factor_cols if c in factors.columns]
    if len(available) >= 2:
        vif_df = compute_vif(factors[available].dropna())
        vif_table(vif_df)
        high_vif = vif_df[vif_df["VIF"] >= 5]
        if not high_vif.empty:
            for _, row in high_vif.iterrows():
                st.caption(interpret_vif(row["feature"], row["VIF"]))

    # --- Wald Tests ---
    st.header("Wald Tests (Joint Significance)")
    wald_col1, wald_col2, wald_col3 = st.columns(3)

    with wald_col1:
        if st.button("Test SMB = HML = 0"):
            result = wald_test(excess_returns, factors, ["SMB", "HML"])
            verdict = "Reject H₀" if result["reject"] else "Fail to reject"
            st.metric("Wald Statistic", f"{result['statistic']:.2f}")
            st.metric("p-value", f"{result['pvalue']:.4f}")
            st.markdown(f"**{verdict}** at 5% level")

    with wald_col2:
        if st.button("Test RMW = CMA = 0"):
            result = wald_test(excess_returns, factors, ["RMW", "CMA"])
            verdict = "Reject H₀" if result["reject"] else "Fail to reject"
            st.metric("Wald Statistic", f"{result['statistic']:.2f}")
            st.metric("p-value", f"{result['pvalue']:.4f}")
            st.markdown(f"**{verdict}** at 5% level")

    with wald_col3:
        if st.button("Test All Four = 0"):
            result = wald_test(excess_returns, factors, ["SMB", "HML", "RMW", "CMA"])
            verdict = "Reject H₀" if result["reject"] else "Fail to reject"
            st.metric("Wald Statistic", f"{result['statistic']:.2f}")
            st.metric("p-value", f"{result['pvalue']:.4f}")
            st.markdown(f"**{verdict}** at 5% level")

# --- Hedge Calculator ---
st.header("Market-Neutral Hedge Calculator")
hedge_col1, hedge_col2 = st.columns(2)
with hedge_col1:
    port_beta = st.number_input("Portfolio Beta", value=1.0, step=0.1)
with hedge_col2:
    hedge_w = hedge_ratio(port_beta)
    st.metric("SPY Hedge Weight", f"{hedge_w:.2f}")
    st.markdown(
        f"To neutralize a portfolio with β={port_beta:.1f}, "
        f"short **{abs(hedge_w):.1%}** of portfolio value in SPY."
    )

render_next_steps("explore", custom_message="Ready to build a predictive model? Continue to the Alpha Model Lab.")
