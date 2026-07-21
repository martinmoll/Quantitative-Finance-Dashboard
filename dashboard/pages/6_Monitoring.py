# dashboard/pages/6_Monitoring.py
"""Page 6: Monitoring & OOD Detection — alert-first layout."""

import streamlit as st
import pandas as pd
import numpy as np
from core.diagnostics import (
    feature_drift, recent_training_window, alpha_decay, signal_staleness,
    compute_ic_stats,
)
from core.risk import factor_exposure
from components.charts import traffic_light_dashboard, bar_chart, STYLE
from components.theory import theory_section
from components.metrics import metric_card, metric_card_row, banner
from components.interpretations import (
    render_interpretation, interpret_ks_shift, interpret_alpha_decay,
)
from components.workflow import render_workflow_status, render_empty_state
from components.theme import inject_theme, COLORS, FONT_MONO, FONT_SANS
import plotly.graph_objects as go

st.set_page_config(page_title="Monitoring", layout="wide")
inject_theme()

if render_empty_state("monitor"):
    st.stop()

pinned = st.session_state.get("pinned_configs", [])

# --- Config selector ---
config_options = ["Current Run"]
config_map = {"Current Run": {
    "result": st.session_state.get("backtest_result"),
    "predictions": st.session_state.get("backtest_predictions"),
    "params": st.session_state.get("backtest_params"),
}}
for p in pinned:
    config_options.append(p["label"])
    config_map[p["label"]] = p

active_label = "Current Run"
if len(config_options) > 1:
    active_label = st.sidebar.selectbox("View config", config_options)
active = config_map[active_label]

result = active["result"]
predictions = active.get("predictions")
params = active.get("params")
df = st.session_state.get("df")
ff5 = st.session_state.get("ff5_factors")

if predictions is None:
    st.info(
        "This pinned configuration has no stored predictions, so monitoring "
        "diagnostics are unavailable. Re-run and re-pin it to enable them."
    )
    st.stop()

C = COLORS

# --- Precompute alert data ---
months = sorted(predictions.keys())
features = params.get("features", [])
available_features = [f for f in features if f in df.columns]

# Drift baseline = the model's most recent training window (what it actually
# learned from), not the earliest pre-OOS history. Otherwise a rolling model is
# compared against data far older than anything it trained on, overstating drift.
_oos_start = params.get("oos_start")
_baseline_months = (
    recent_training_window(
        df["ym"].unique(), _oos_start, params.get("retrain_freq", 12),
        params.get("window_type", "expanding"), params.get("rolling_window"),
        months[-1],
    )
    if _oos_start is not None and months else None
)
drift_train_data = (
    df[df["ym"].isin(_baseline_months)] if _baseline_months
    else (df[df["ym"] < months[0]] if months else df.iloc[0:0])
)

ic = result["ic"]
rolling_6m_ic = ic.rolling(6, min_periods=3).mean()
latest_rolling_ic = rolling_6m_ic.iloc[-1] if len(rolling_6m_ic) > 0 else 0
latest_monthly_ic = ic.iloc[-1] if len(ic) > 0 else 0

stale_df = signal_staleness(result["turnover"], threshold=0.10, consecutive=3)
stale_active = stale_df["stale"].iloc[-1] if len(stale_df) > 0 else False

pct_flagged = 0
n_flagged = 0
n_total = 0
ks_results = None
if len(months) >= 2 and available_features:
    last_month = months[-1]
    train_data = drift_train_data
    current_data = df[df["ym"] == last_month]
    drift = feature_drift(train_data, current_data, available_features)
    if not drift.empty:
        ks_results = drift
        n_flagged = int(ks_results["flag"].sum())
        n_total = len(ks_results)
        pct_flagged = n_flagged / n_total if n_total > 0 else 0

decay = alpha_decay(predictions, horizons=list(range(1, 13)))
ic1 = decay.iloc[0] if len(decay) > 0 else 0
half_life = None
for h, ic_h in decay.items():
    if ic_h < ic1 / 2:
        half_life = h
        break

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
        "status": "red" if pct_flagged > 0.20 else "amber" if pct_flagged > 0.10 else "green",
        "value": f"{pct_flagged:.0%} flagged",
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

n_red = sum(1 for a in alerts.values() if a["status"] == "red")
n_amber = sum(1 for a in alerts.values() if a["status"] == "amber")

# --- Header ---
st.markdown(
    f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
    f'color:{C["text"]};margin:0;">Monitoring & OOD Detection</h1>',
    unsafe_allow_html=True,
)
render_workflow_status("monitor")

# --- Hero status banner ---
if n_red >= 2:
    hero_accent = C["negative"]
    hero_label = "Retrain now"
    hero_icon = "&#10005;"
elif n_red >= 1 or n_amber >= 2:
    hero_accent = C["warning"]
    hero_label = "Watch"
    hero_icon = "!"
else:
    hero_accent = C["positive"]
    hero_label = "Healthy"
    hero_icon = "&#10003;"

from components.metrics import _hex_to_rgb
hero_bg = f"rgba({_hex_to_rgb(hero_accent)},0.08)"
hero_border = f"rgba({_hex_to_rgb(hero_accent)},0.22)"

hl_str = f"t½ {half_life}m" if half_life else "no decay"
summary_parts = [f"drift KS {pct_flagged:.2f}", hl_str]
if stale_active:
    summary_parts.append("stale signal")
summary_mono = " · ".join(summary_parts)

st.markdown(
    f'<div style="background:{hero_bg};border:1px solid {hero_border};border-radius:12px;'
    f'padding:18px 22px;display:flex;align-items:center;gap:16px;margin-bottom:16px;">'
    f'<span style="display:inline-flex;align-items:center;justify-content:center;'
    f'width:36px;height:36px;border-radius:8px;background:rgba({_hex_to_rgb(hero_accent)},0.14);'
    f'color:{hero_accent};font-size:18px;font-weight:700;flex-shrink:0;">{hero_icon}</span>'
    f'<div>'
    f'<span style="font-family:{FONT_SANS};font-size:18px;font-weight:700;color:{hero_accent};">'
    f'{hero_label}</span>'
    f'<span style="font-family:{FONT_MONO};font-size:12px;color:{C["text_secondary"]};'
    f'margin-left:14px;">{summary_mono}</span>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

@st.dialog("What to do when monitoring flags “Retrain now”")
def _retrain_guide(active_alerts):
    reds = [name for name, a in active_alerts.items() if a["status"] == "red"]
    if reds:
        st.markdown("**Red right now:** " + ", ".join(reds))
    st.markdown(
        """
This appears when **2 or more alerts are red** — the model looks stale or
out-of-distribution, so its predictions are less trustworthy than usual. What to do:

**1. Find out which alert fired** — use the tabs below:
- **Drift (KS):** which features shifted (usually momentum / interaction features).
- **Alpha decay:** how fast the signal fades (the half-life).
- **Signal staleness:** turnover collapsing — the model has stopped changing its bets.

**2. Retrain on recent data.** Go to the **Alpha Model Lab** and re-run. To keep the
model fresher, shorten the *retrain frequency*, shrink the *rolling window*, or move
the *OOS start* closer to today so training tracks the current regime.

**3. If drift is the trigger.** Retraining rolls the training window forward so today's
data is in-distribution again. For fat-tailed momentum features you can also try the
winsorized (`_w`) feature variants — but winsorizing tames tails, it does **not** remove
a genuine regime shift.

**4. De-risk until it clears.** When operating out-of-distribution, lower K / position
sizes or lean on the regime filter — the signal is noisier than the backtest implies.

**Caveat:** on a short panel (e.g. Oslo's ~37 months) high drift can reflect limited
history and volatile momentum rather than a broken model. Confirm it persists across
several months before over-reacting.
        """
    )
    if st.button("Go to Alpha Model Lab →", type="primary", use_container_width=True):
        st.switch_page("pages/3_Alpha_Model_Lab.py")


if n_red >= 2:
    if st.button("What to do →", type="primary"):
        _retrain_guide(alerts)

theory_section("Distribution Shift", "distribution_shift")
theory_section("Alpha Decay and Retraining", "alpha_decay")

# --- Tabs ---
tab_drift, tab_decay, tab_stale = st.tabs(["Drift (KS)", "Alpha decay", "Signal staleness"])

# =========================================================================
# DRIFT TAB
# =========================================================================
with tab_drift:
    st.subheader("KS Test — Distribution Shift Detection")
    if _baseline_months:
        st.caption(
            f"Baseline = the model's most recent training window "
            f"({_baseline_months[0]} → {_baseline_months[-1]}, {len(_baseline_months)} months), "
            "compared against the latest month."
        )
    if ks_results is not None:
        metric_card_row([
            {"label": "Features Flagged", "value": f"{n_flagged}/{n_total} ({pct_flagged:.0%})",
             "accent": C["negative"] if pct_flagged > 0.20 else C["positive"], "variant": "bar"},
        ])
        render_interpretation(interpret_ks_shift(pct_flagged, n_flagged, n_total))
        st.dataframe(ks_results.head(20), use_container_width=True)

        if len(months) > 3 and available_features:
            train_data = drift_train_data
            ks_over_time = {}
            sample_months = months[::max(1, len(months) // 12)]
            for m in sample_months:
                month_data = df[df["ym"] == m]
                ks_m = feature_drift(train_data, month_data, available_features)
                if not ks_m.empty:
                    ks_over_time[m] = ks_m.set_index("feature")["D"]
            if ks_over_time:
                ks_df = pd.DataFrame(ks_over_time).T
                fig = go.Figure(data=go.Heatmap(
                    z=ks_df.values, x=ks_df.columns[:30], y=ks_df.index,
                    colorscale=[
                        [0, C["canvas"]], [0.33, C["primary"]],
                        [0.66, C["warning"]], [1, C["negative"]],
                    ],
                    zmin=0, zmax=0.3,
                ))
                fig.update_layout(
                    template="alpha",
                    title="KS D-Statistic Over Time (top 30 features)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Not enough overlapping feature data between the training window "
            "and the current month to compute distribution drift. This happens "
            "when the pre-out-of-sample window is empty or its features have no "
            "history yet."
        )

# =========================================================================
# ALPHA DECAY TAB
# =========================================================================
with tab_decay:
    st.subheader("Alpha Decay Curve")
    if not decay.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=decay.index, y=decay.values, mode="lines+markers",
            line=dict(color=C["primary"], width=2),
            marker=dict(size=8),
        ))
        if half_life:
            fig.add_vline(x=half_life, line_dash="dash", line_color=C["warning"],
                          annotation_text=f"t½ = {half_life}m",
                          annotation_font=dict(family="IBM Plex Mono", color=C["warning"]))
        fig.add_hline(y=0, line_dash="dot", line_color=C["text_muted"])
        fig.update_layout(
            template="alpha",
            title="IC at Forward Horizons", xaxis_title="Horizon (months)",
            yaxis_title="IC", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        render_interpretation(interpret_alpha_decay(half_life, ic1))

# =========================================================================
# SIGNAL STALENESS TAB
# =========================================================================
with tab_stale:
    st.subheader("Signal Staleness")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stale_df.index, y=stale_df["turnover"].values,
        marker_color=[C["negative"] if s else C["primary"] for s in stale_df["stale"]],
        name="Turnover",
    ))
    fig.add_hline(y=0.10, line_dash="dash", line_color=C["warning"],
                  annotation_text="Stale threshold (10%)")
    fig.update_layout(
        template="alpha",
        title="Monthly Turnover (red = stale signal)",
        yaxis_tickformat=".0%", height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
