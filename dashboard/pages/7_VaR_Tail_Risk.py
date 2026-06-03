# dashboard/pages/7_VaR_Tail_Risk.py
"""Page 7: Value at Risk & Tail Risk Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.risk import (
    compute_all_var, parametric_var, historical_var, monte_carlo_var,
    cornish_fisher_var, component_var, rolling_var, var_backtest,
)
from components.charts import STYLE, _base_layout
from components.metrics import metric_row
from components.theory import theory_section

st.set_page_config(page_title="VaR & Tail Risk", layout="wide")
st.title("Value at Risk & Tail Risk")

result = st.session_state.get("backtest_result")
params = st.session_state.get("backtest_params")

if result is None:
    st.info("Run a backtest on the **Alpha Model Lab** page first.")
    st.stop()

rets = result["monthly_returns"].dropna()
if len(rets) < 6:
    st.warning("Not enough return observations for VaR analysis.")
    st.stop()

theory_section("VaR, CVaR & Tail Risk", "var_and_tail_risk")

# --- Configuration ---
st.sidebar.header("VaR Settings")
confidence = st.sidebar.selectbox("Confidence Level", [0.95, 0.99], index=0)
rolling_window = st.sidebar.slider("Rolling Window (months)", 12, 48, 24)
n_sims = st.sidebar.number_input("Monte Carlo Simulations", value=10_000, step=5000)

# --- Compute all VaR methods ---
var_summary = compute_all_var(rets, confidence, n_sims)
cf = cornish_fisher_var(rets, confidence)

# --- KPI Row ---
st.header("VaR & CVaR Summary")
st.caption(f"Monthly figures at {confidence:.0%} confidence | {len(rets)} months of data")

metric_row([
    {"label": "Parametric VaR", "value": f"{var_summary.loc['Parametric', 'VaR']:.2%}"},
    {"label": "Historical VaR", "value": f"{var_summary.loc['Historical', 'VaR']:.2%}"},
    {"label": "Monte Carlo VaR", "value": f"{var_summary.loc['Monte Carlo', 'VaR']:.2%}"},
    {"label": "Cornish-Fisher VaR", "value": f"{var_summary.loc['Cornish-Fisher', 'VaR']:.2%}"},
])

metric_row([
    {"label": "Parametric CVaR", "value": f"{var_summary.loc['Parametric', 'CVaR']:.2%}"},
    {"label": "Historical CVaR", "value": f"{var_summary.loc['Historical', 'CVaR']:.2%}"},
    {"label": "Monte Carlo CVaR", "value": f"{var_summary.loc['Monte Carlo', 'CVaR']:.2%}"},
    {"label": "Skewness", "value": f"{cf['skewness']:.2f}"},
])

# --- Comparison Table ---
st.markdown("---")
st.header("Method Comparison")
display_df = var_summary[["VaR", "CVaR"]].copy()
display_df["VaR"] = display_df["VaR"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
display_df["CVaR"] = display_df["CVaR"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
display_df.loc["Cornish-Fisher", "Skew"] = f"{cf['skewness']:.2f}"
display_df.loc["Cornish-Fisher", "Excess Kurt"] = f"{cf['excess_kurtosis']:.2f}"
display_df = display_df.fillna("")
st.dataframe(display_df, use_container_width=True)

# --- Return Distribution with VaR/CVaR lines ---
st.markdown("---")
st.header("Return Distribution")

param = parametric_var(rets, confidence)
hist = historical_var(rets, confidence)

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=rets.values, nbinsx=40, name="Monthly Returns",
    marker_color=STYLE["accent"], opacity=0.7,
))

fig.add_vline(x=-param["VaR"], line_dash="solid", line_color=STYLE["negative"],
              annotation_text=f"Param VaR ({param['VaR']:.1%})",
              annotation_position="top left",
              annotation_font_color=STYLE["negative"])

fig.add_vline(x=-hist["VaR"], line_dash="dash", line_color=STYLE["warning"],
              annotation_text=f"Hist VaR ({hist['VaR']:.1%})",
              annotation_position="top right",
              annotation_font_color=STYLE["warning"])

fig.add_vline(x=-param["CVaR"], line_dash="dot", line_color="#FF6B6B",
              annotation_text=f"CVaR ({param['CVaR']:.1%})",
              annotation_position="bottom left",
              annotation_font_color="#FF6B6B")

fig.update_layout(
    **_base_layout(
        title=f"Monthly Return Distribution with VaR/CVaR ({confidence:.0%})",
        xaxis_title="Monthly Return",
        yaxis_title="Frequency",
        showlegend=False,
    )
)
st.plotly_chart(fig, use_container_width=True)

# --- Rolling VaR ---
st.markdown("---")
st.header("Rolling VaR")
st.caption(f"{rolling_window}-month rolling window at {confidence:.0%} confidence")

roll = rolling_var(rets, rolling_window, confidence)
if not roll.empty:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=roll.index, y=(-rets.loc[roll.index]).values,
        mode="markers", name="Actual Loss",
        marker=dict(size=5, color=STYLE["muted"], opacity=0.6),
    ))

    fig.add_trace(go.Scatter(
        x=roll.index, y=roll["Parametric"].values,
        mode="lines", name="Parametric VaR",
        line=dict(color=STYLE["negative"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=roll.index, y=roll["Historical"].values,
        mode="lines", name="Historical VaR",
        line=dict(color=STYLE["warning"], width=2, dash="dash"),
    ))

    breach_mask = (-rets.loc[roll.index]) > roll["Parametric"]
    breach_dates = roll.index[breach_mask]
    if len(breach_dates) > 0:
        fig.add_trace(go.Scatter(
            x=breach_dates,
            y=(-rets.loc[breach_dates]).values,
            mode="markers", name="VaR Breach",
            marker=dict(size=10, color=STYLE["negative"], symbol="x"),
        ))

    fig.update_layout(**_base_layout(
        title=f"Rolling {rolling_window}-Month VaR vs Actual Losses",
        xaxis_title="Month", yaxis_title="Loss (positive = loss)",
    ))
    st.plotly_chart(fig, use_container_width=True)

# --- VaR Backtest (Kupiec) ---
st.markdown("---")
st.header("VaR Model Validation (Kupiec Test)")

bt = var_backtest(rets, rolling_window, confidence)
if bt["total"] > 0:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("VaR Breaches", f"{bt['breaches']} / {bt['total']}")
    with col2:
        st.metric("Breach Rate", f"{bt['breach_rate']:.1%}")
    with col3:
        st.metric("Expected Rate", f"{bt['expected_rate']:.1%}")
    with col4:
        st.metric("Kupiec p-value", f"{bt['kupiec_p']:.4f}")

    if bt["kupiec_p"] > 0.05:
        st.success(
            f"VaR model passes Kupiec test (p={bt['kupiec_p']:.3f} > 0.05). "
            f"Observed breach rate ({bt['breach_rate']:.1%}) is consistent with "
            f"the expected {bt['expected_rate']:.0%}."
        )
    elif bt["breach_rate"] > bt["expected_rate"]:
        st.error(
            f"VaR model rejected by Kupiec test (p={bt['kupiec_p']:.3f}). "
            f"Too many breaches ({bt['breach_rate']:.1%} vs expected "
            f"{bt['expected_rate']:.0%}) — the model underestimates risk."
        )
    else:
        st.warning(
            f"VaR model rejected by Kupiec test (p={bt['kupiec_p']:.3f}). "
            f"Too few breaches ({bt['breach_rate']:.1%} vs expected "
            f"{bt['expected_rate']:.0%}) — the model may be overly conservative."
        )

# --- Component VaR ---
st.markdown("---")
st.header("Component VaR (Current Holdings)")

holdings = result.get("holdings", {})
if holdings:
    last_month = sorted(holdings.keys())[-1]
    held = holdings[last_month]
    if "weight" in held.columns and len(held) > 0:
        w = held["weight"].values
        n = len(w)
        cov = np.eye(n) * 0.01
        comp = component_var(w / np.abs(w).sum(), cov, confidence)

        labels = held["permno"].astype(str).values if "permno" in held.columns else [f"Pos {i}" for i in range(n)]
        comp.index = labels

        comp_pct = (comp / comp.sum() * 100).round(1)
        top_n = min(20, len(comp_pct))
        top = comp_pct.abs().nlargest(top_n)
        top_vals = comp_pct.loc[top.index]

        fig = go.Figure(go.Bar(
            x=top_vals.index.astype(str),
            y=top_vals.values,
            marker_color=[
                STYLE["negative"] if v > 0 else STYLE["positive"]
                for v in top_vals.values
            ],
        ))
        fig.update_layout(**_base_layout(
            title=f"Component VaR Contribution — Top {top_n} Holdings ({last_month})",
            xaxis_title="Position", yaxis_title="% of Total VaR",
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Component VaR decomposes the total portfolio VaR into per-position contributions. "
            "Positions with larger contributions drive more of the portfolio's tail risk."
        )
    else:
        st.info("No holdings with weights available for component VaR.")
else:
    st.info("No holdings data available.")

# --- Tail Risk Statistics ---
st.markdown("---")
st.header("Tail Risk Statistics")
from scipy import stats as sp_stats

col1, col2, col3, col4 = st.columns(4)
with col1:
    skew = float(sp_stats.skew(rets.dropna()))
    st.metric("Skewness", f"{skew:.3f}")
    if skew < -0.5:
        st.caption("Left-skewed: large losses more frequent than large gains")
    elif skew > 0.5:
        st.caption("Right-skewed: large gains more frequent")
    else:
        st.caption("Approximately symmetric")
with col2:
    kurt = float(sp_stats.kurtosis(rets.dropna()))
    st.metric("Excess Kurtosis", f"{kurt:.3f}")
    if kurt > 1:
        st.caption("Fat-tailed: extreme events more likely than normal")
    else:
        st.caption("Near-normal tail behavior")
with col3:
    jb_stat, jb_p = sp_stats.jarque_bera(rets.dropna())
    st.metric("Jarque-Bera stat", f"{jb_stat:.2f}")
    st.caption(f"p={jb_p:.4f} {'(reject normality)' if jb_p < 0.05 else '(consistent with normal)'}")
with col4:
    sortino_denom = rets[rets < 0].std() * np.sqrt(12)
    ann_ret = rets.mean() * 12
    sortino = ann_ret / sortino_denom if sortino_denom > 0 else np.nan
    st.metric("Sortino Ratio", f"{sortino:.2f}")
    st.caption("Return per unit of downside risk")
