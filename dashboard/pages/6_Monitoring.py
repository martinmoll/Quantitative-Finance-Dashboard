# dashboard/pages/6_Monitoring.py
"""Page 6: Monitoring & OOD Detection."""

import streamlit as st
import pandas as pd
import numpy as np
from core.diagnostics import ks_test, alpha_decay, signal_staleness, compute_ic_stats
from core.risk import factor_exposure
from components.charts import traffic_light_dashboard, bar_chart, STYLE
from components.theory import theory_section
from components.workflow import render_workflow_status, render_empty_state
import plotly.graph_objects as go

st.set_page_config(page_title="Monitoring", layout="wide")
st.title("Monitoring & OOD Detection")
render_workflow_status("monitor")

if render_empty_state("monitor"):
    st.stop()

result = st.session_state.get("backtest_result")
predictions = st.session_state.get("backtest_predictions")
params = st.session_state.get("backtest_params")
df = st.session_state.get("df")
ff5 = st.session_state.get("ff5_factors")

theory_section("Distribution Shift", "distribution_shift")
theory_section("Alpha Decay and Retraining", "alpha_decay")

months = sorted(predictions.keys())
features = params.get("features", [])
available_features = [f for f in features if f in df.columns]

# --- KS Test Panel ---
st.header("KS Test — Distribution Shift Detection")
if len(months) >= 2 and available_features:
    last_month = months[-1]
    train_data = df[df["ym"] < months[0]]

    X_train = train_data[available_features].dropna()
    X_current = df[df["ym"] == last_month][available_features].dropna()

    if len(X_train) > 0 and len(X_current) > 0:
        ks_results = ks_test(X_train, X_current)
        n_flagged = ks_results["flag"].sum()
        n_total = len(ks_results)
        pct_flagged = n_flagged / n_total if n_total > 0 else 0

        ks_col1, ks_col2 = st.columns([1, 2])
        with ks_col1:
            color = "normal" if pct_flagged < 0.20 else "inverse"
            st.metric("Features Flagged", f"{n_flagged}/{n_total} ({pct_flagged:.0%})",
                      delta_color=color)
        with ks_col2:
            st.dataframe(ks_results.head(20), use_container_width=True)

        # KS heatmap over time
        if len(months) > 3:
            ks_over_time = {}
            sample_months = months[::max(1, len(months) // 12)]
            for m in sample_months:
                X_m = df[df["ym"] == m][available_features].dropna()
                if len(X_m) > 0:
                    ks_m = ks_test(X_train, X_m)
                    ks_over_time[m] = ks_m.set_index("feature")["D"]
            if ks_over_time:
                ks_df = pd.DataFrame(ks_over_time).T
                fig = go.Figure(data=go.Heatmap(
                    z=ks_df.values, x=ks_df.columns[:30], y=ks_df.index,
                    colorscale=[[0, STYLE["positive"]], [0.5, STYLE["warning"]], [1, STYLE["negative"]]],
                    zmin=0, zmax=0.3,
                ))
                fig.update_layout(
                    title="KS D-Statistic Over Time (top 30 features)",
                    template=STYLE["template"], height=400,
                    paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Alpha Decay ---
st.markdown("---")
st.header("Alpha Decay Curve")
decay = alpha_decay(predictions, horizons=list(range(1, 13)))
if not decay.empty:
    ic1 = decay.iloc[0] if len(decay) > 0 else 0
    half_life = None
    for h, ic_h in decay.items():
        if ic_h < ic1 / 2:
            half_life = h
            break

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decay.index, y=decay.values, mode="lines+markers",
        line=dict(color=STYLE["accent"], width=2),
        marker=dict(size=8),
    ))
    if half_life:
        fig.add_vline(x=half_life, line_dash="dash", line_color=STYLE["warning"],
                      annotation_text=f"Half-life: {half_life}m")
    fig.add_hline(y=0, line_dash="dot", line_color=STYLE["muted"])
    fig.update_layout(
        title="IC at Forward Horizons", xaxis_title="Horizon (months)",
        yaxis_title="IC", template=STYLE["template"], height=400,
        paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
    )
    st.plotly_chart(fig, use_container_width=True)

    if half_life:
        if half_life < 3:
            st.info(f"Half-life = {half_life} months → **monthly rebalancing** recommended.")
        elif half_life < 9:
            st.info(f"Half-life = {half_life} months → **quarterly rebalancing** may suffice.")
        else:
            st.info(f"Half-life = {half_life} months → **slower rebalancing** acceptable.")

# --- Signal Staleness ---
st.markdown("---")
st.header("Signal Staleness")
stale_df = signal_staleness(result["turnover"], threshold=0.10, consecutive=3)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=stale_df.index, y=stale_df["turnover"].values,
    marker_color=[STYLE["negative"] if s else STYLE["accent"] for s in stale_df["stale"]],
    name="Turnover",
))
fig.add_hline(y=0.10, line_dash="dash", line_color=STYLE["warning"],
              annotation_text="Stale threshold (10%)")
fig.update_layout(
    title="Monthly Turnover (red = stale signal)",
    yaxis_tickformat=".0%", template=STYLE["template"], height=350,
    paper_bgcolor=STYLE["bg"], plot_bgcolor=STYLE["bg"],
)
st.plotly_chart(fig, use_container_width=True)

# --- Automated Alert Dashboard ---
st.markdown("---")
st.header("Alert Dashboard")

ic = result["ic"]
rolling_6m_ic = ic.rolling(6, min_periods=3).mean()
latest_rolling_ic = rolling_6m_ic.iloc[-1] if len(rolling_6m_ic) > 0 else 0
latest_monthly_ic = ic.iloc[-1] if len(ic) > 0 else 0
stale_active = stale_df["stale"].iloc[-1] if len(stale_df) > 0 else False
ood_pct = pct_flagged if "pct_flagged" in dir() else 0

alerts = {
    "IC Decay": {
        "status": "red" if latest_rolling_ic < 0.02 else "green",
        "value": f"6m IC: {latest_rolling_ic:.3f}",
    },
    "IC Collapse": {
        "status": "red" if latest_monthly_ic < -0.03 else "green",
        "value": f"Latest: {latest_monthly_ic:.3f}",
    },
    "OOD Shift": {
        "status": "red" if ood_pct > 0.20 else "amber" if ood_pct > 0.10 else "green",
        "value": f"{ood_pct:.0%} flagged",
    },
    "Stale Signal": {
        "status": "red" if stale_active else "green",
        "value": "Stale" if stale_active else "Active",
    },
}

if ff5 is not None:
    exp = factor_exposure(result["monthly_returns"], ff5)
    max_drift = exp.abs().max() if len(exp) > 0 else 0
    alerts["Factor Drift"] = {
        "status": "red" if max_drift > 0.10 else "green",
        "value": f"Max |β|: {max_drift:.3f}",
    }

fig = traffic_light_dashboard(alerts)
st.plotly_chart(fig, use_container_width=True)

# --- Retraining Recommendation ---
st.header("Retraining Recommendation")
n_red = sum(1 for a in alerts.values() if a["status"] == "red")
n_amber = sum(1 for a in alerts.values() if a["status"] == "amber")

if n_red >= 2:
    st.error("**Retrain immediately.** Multiple critical alerts active.")
elif n_red >= 1 or n_amber >= 2:
    st.warning("**Consider retraining.** Some alerts triggered.")
else:
    st.success("**No action needed.** All indicators healthy.")
