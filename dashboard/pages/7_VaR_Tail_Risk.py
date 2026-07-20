# dashboard/pages/7_VaR_Tail_Risk.py
"""Page 7: Value at Risk & Tail Risk Analysis — tabbed layout."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.risk import (
    compute_all_var, parametric_var, historical_var, monte_carlo_var,
    cornish_fisher_var, component_var, rolling_var, var_backtest,
)
from components.charts import STYLE
from components.metrics import metric_card, metric_card_row, banner
from components.theory import theory_section
from components.interpretations import (
    render_interpretation, interpret_var_comparison, interpret_kupiec,
    interpret_skewness_kurtosis,
)
from components.workflow import render_workflow_status, render_empty_state
from components.theme import inject_theme, COLORS, FONT_MONO, FONT_SANS, base_layout

st.set_page_config(page_title="VaR & Tail Risk", layout="wide")
inject_theme()

if render_empty_state("monitor"):
    st.stop()

result = st.session_state.get("backtest_result")
pinned = st.session_state.get("pinned_configs", [])

# --- Config selector ---
config_options = ["Current Run"]
config_map = {"Current Run": {"result": result}}
for p in pinned:
    config_options.append(p["label"])
    config_map[p["label"]] = p

active_label = "Current Run"
if len(config_options) > 1:
    active_label = st.sidebar.selectbox("View config", config_options)
result = config_map[active_label]["result"]

rets = result["monthly_returns"].dropna()
if len(rets) < 6:
    st.warning("Not enough return observations for VaR analysis.")
    st.stop()

C = COLORS

# --- Header ---
st.markdown(
    f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
    f'color:{C["text"]};margin:0;">Value at Risk & Tail Risk</h1>',
    unsafe_allow_html=True,
)

# --- Sidebar controls ---
st.sidebar.header("VaR Settings")
confidence = st.sidebar.selectbox("Confidence Level", [0.95, 0.99], index=0)
rolling_window = st.sidebar.slider("Rolling Window (months)", 12, 48, 24)
n_sims = st.sidebar.number_input("Monte Carlo Simulations", value=10_000, step=5000)

subtitle = f"{confidence:.0%} confidence &middot; 1-month horizon &middot; {n_sims:,} sims"
st.markdown(
    f'<p style="font-family:{FONT_SANS};font-size:13px;color:{C["text_secondary"]};'
    f'margin:4px 0 0;">{subtitle}</p>',
    unsafe_allow_html=True,
)

render_workflow_status("monitor")

theory_section("VaR, CVaR & Tail Risk", "var_and_tail_risk")

# --- Compute all VaR methods ---
var_summary = compute_all_var(rets, confidence, n_sims)
cf = cornish_fisher_var(rets, confidence)

# --- Tabs ---
tab_summary, tab_dist, tab_bt, tab_decomp, tab_tail = st.tabs(
    ["Summary", "Distribution", "Backtest", "Decomposition", "Tail stats"]
)

# =========================================================================
# SUMMARY TAB
# =========================================================================
with tab_summary:
    st.subheader("VaR & CVaR Summary")
    metric_card_row([
        {"label": "Historical VaR", "value": f"{var_summary.loc['Historical', 'VaR']:.2%}",
         "accent": C["negative"], "variant": "bar"},
        {"label": "Parametric VaR", "value": f"{var_summary.loc['Parametric', 'VaR']:.2%}",
         "accent": C["negative"], "variant": "bar"},
        {"label": "Cornish-Fisher VaR", "value": f"{var_summary.loc['Cornish-Fisher', 'VaR']:.2%}",
         "accent": C["negative"], "variant": "bar"},
        {"label": "Monte Carlo VaR", "value": f"{var_summary.loc['Monte Carlo', 'VaR']:.2%}",
         "accent": C["negative"], "variant": "bar"},
    ])
    metric_card_row([
        {"label": "Historical CVaR", "value": f"{var_summary.loc['Historical', 'CVaR']:.2%}",
         "accent": C["negative"], "variant": "bar"},
        {"label": "Parametric CVaR", "value": f"{var_summary.loc['Parametric', 'CVaR']:.2%}",
         "accent": C["negative"], "variant": "bar"},
        {"label": "Monte Carlo CVaR", "value": f"{var_summary.loc['Monte Carlo', 'CVaR']:.2%}",
         "accent": C["negative"], "variant": "bar"},
        {"label": "Skewness", "value": f"{cf['skewness']:.2f}",
         "accent": C["warning"] if abs(cf["skewness"]) > 0.5 else C["primary"], "variant": "bar"},
    ])

    st.markdown("---")
    st.subheader("Method Comparison")
    display_df = var_summary[["VaR", "CVaR"]].copy()
    display_df["VaR"] = display_df["VaR"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    display_df["CVaR"] = display_df["CVaR"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    display_df.loc["Cornish-Fisher", "Skew"] = f"{cf['skewness']:.2f}"
    display_df.loc["Cornish-Fisher", "Excess Kurt"] = f"{cf['excess_kurtosis']:.2f}"
    display_df = display_df.fillna("")
    st.dataframe(display_df, use_container_width=True)

    render_interpretation(interpret_var_comparison(
        var_summary.loc["Parametric", "VaR"],
        var_summary.loc["Historical", "VaR"],
        var_summary.loc["Cornish-Fisher", "VaR"],
        cf["skewness"],
        cf["excess_kurtosis"],
    ))

# =========================================================================
# DISTRIBUTION TAB
# =========================================================================
with tab_dist:
    st.subheader("Return Distribution")
    param = parametric_var(rets, confidence)
    hist = historical_var(rets, confidence)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets.values, nbinsx=40, name="Monthly Returns",
        marker_color=C["primary"], opacity=0.7,
    ))
    fig.add_vline(x=-param["VaR"], line_dash="dash", line_color=C["warning"],
                  annotation_text=f"Param VaR ({param['VaR']:.1%})",
                  annotation_position="top left",
                  annotation_font_color=C["warning"])
    fig.add_vline(x=-hist["VaR"], line_dash="dash", line_color=C["warning"],
                  annotation_text=f"Hist VaR ({hist['VaR']:.1%})",
                  annotation_position="top right",
                  annotation_font_color=C["warning"])
    fig.add_vline(x=-param["CVaR"], line_dash="solid", line_color=C["negative"],
                  annotation_text=f"CVaR ({param['CVaR']:.1%})",
                  annotation_position="bottom left",
                  annotation_font_color=C["negative"])
    fig.update_layout(**base_layout(
        title=f"Monthly Return Distribution with VaR/CVaR ({confidence:.0%})",
        xaxis_title="Monthly Return",
        yaxis_title="Frequency",
        showlegend=False,
    ))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# BACKTEST TAB
# =========================================================================
with tab_bt:
    st.subheader("Rolling VaR")
    st.caption(f"{rolling_window}-month rolling window at {confidence:.0%} confidence")

    roll = rolling_var(rets, rolling_window, confidence)
    if not roll.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roll.index, y=(-rets.loc[roll.index]).values,
            mode="markers", name="Actual Loss",
            marker=dict(size=5, color=C["text_muted"], opacity=0.6),
        ))
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll["Parametric"].values,
            mode="lines", name="Parametric VaR",
            line=dict(color=C["warning"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll["Historical"].values,
            mode="lines", name="Historical VaR",
            line=dict(color=C["warning"], width=2, dash="dash"),
        ))
        breach_mask = (-rets.loc[roll.index]) > roll["Parametric"]
        breach_dates = roll.index[breach_mask]
        if len(breach_dates) > 0:
            fig.add_trace(go.Scatter(
                x=breach_dates,
                y=(-rets.loc[breach_dates]).values,
                mode="markers", name="VaR Breach",
                marker=dict(size=10, color=C["negative"], symbol="x"),
            ))
        fig.update_layout(**base_layout(
            title=f"Rolling {rolling_window}-Month VaR vs Actual Losses",
            xaxis_title="Month", yaxis_title="Loss (positive = loss)",
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("VaR Model Validation (Kupiec Test)")
    bt = var_backtest(rets, rolling_window, confidence)
    if bt["total"] > 0:
        metric_card_row([
            {"label": "VaR Breaches", "value": f"{bt['breaches']} / {bt['total']}",
             "accent": C["negative"] if bt["breach_rate"] > bt["expected_rate"] * 1.5 else C["positive"], "variant": "bar"},
            {"label": "Breach Rate", "value": f"{bt['breach_rate']:.1%}",
             "accent": C["negative"] if bt["breach_rate"] > bt["expected_rate"] * 1.5 else C["positive"], "variant": "bar"},
            {"label": "Expected Rate", "value": f"{bt['expected_rate']:.1%}",
             "accent": C["primary"], "variant": "bar"},
            {"label": "Kupiec p-value", "value": f"{bt['kupiec_p']:.4f}",
             "accent": C["positive"] if bt["kupiec_p"] > 0.05 else C["negative"], "variant": "bar"},
        ])
        render_interpretation(interpret_kupiec(
            bt["breach_rate"], bt["expected_rate"], bt["kupiec_p"],
        ))

# =========================================================================
# DECOMPOSITION TAB
# =========================================================================
with tab_decomp:
    st.subheader("Component VaR (Current Holdings)")
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
                    C["negative"] if v > 0 else C["positive"]
                    for v in top_vals.values
                ],
            ))
            fig.update_layout(**base_layout(
                title=f"Component VaR Contribution — Top {top_n} Holdings ({last_month})",
                xaxis_title="Position", yaxis_title="% of Total VaR",
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No holdings with weights available for component VaR.")
    else:
        st.info("No holdings data available.")

# =========================================================================
# TAIL STATS TAB
# =========================================================================
with tab_tail:
    st.subheader("Tail Risk Statistics")
    from scipy import stats as sp_stats

    skew = float(sp_stats.skew(rets.dropna()))
    kurt = float(sp_stats.kurtosis(rets.dropna()))
    jb_stat, jb_p = sp_stats.jarque_bera(rets.dropna())
    sortino_denom = rets[rets < 0].std() * np.sqrt(12)
    ann_ret = rets.mean() * 12
    sortino = ann_ret / sortino_denom if sortino_denom > 0 else np.nan

    skew_accent = C["warning"] if abs(skew) > 0.5 else C["primary"]
    kurt_accent = C["warning"] if kurt > 1 else C["primary"]
    jb_accent = C["negative"] if jb_p < 0.05 else C["positive"]

    metric_card_row([
        {"label": "Skewness", "value": f"{skew:.3f}", "accent": skew_accent, "variant": "bar"},
        {"label": "Excess Kurtosis", "value": f"{kurt:.3f}", "accent": kurt_accent, "variant": "bar"},
        {"label": "Jarque-Bera", "value": f"{jb_stat:.2f}",
         "delta": f"p={jb_p:.4f} {'(reject normality)' if jb_p < 0.05 else '(normal)'}",
         "delta_color": C["negative"] if jb_p < 0.05 else C["positive"],
         "accent": jb_accent, "variant": "bar"},
        {"label": "Sortino Ratio", "value": f"{sortino:.2f}",
         "delta": "Return per unit downside risk",
         "delta_color": C["text_secondary"],
         "accent": C["positive"] if sortino > 1 else C["primary"], "variant": "bar"},
    ])

    render_interpretation(interpret_skewness_kurtosis(skew, kurt))
