# dashboard/pages/1_Data_Explorer.py
"""Page 1: Data Explorer — understand the dataset before modeling."""

import streamlit as st
import pandas as pd
import numpy as np
from components.charts import correlation_heatmap, bar_chart, STYLE
from components.theory import theory_section
from components.workflow import render_workflow_status, render_next_steps
from components.theme import inject_theme, COLORS, FONT_SANS
from components.metrics import metric_card_row
import plotly.graph_objects as go

st.set_page_config(page_title="Data Explorer", layout="wide")
inject_theme()

C = COLORS
st.markdown(
    f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
    f'color:{C["text"]};margin:0;">Data Explorer</h1>',
    unsafe_allow_html=True,
)
render_workflow_status("explore")

df = st.session_state.get("df")
if df is None:
    st.error("Dataset not loaded. Return to the home page.")
    st.stop()

theory_section("Cross-Sectional Standardization", "cross_sectional_standardization")

# --- Summary Statistics ---
st.header("Dataset Summary")
xs_cols = [c for c in df.columns if c.endswith("_xs") and c != "y_xs"]
metric_card_row([
    {"label": "Stocks", "value": f"{df['permno'].nunique():,}", "accent": C["primary"], "variant": "bar"},
    {"label": "Months", "value": f"{df['ym'].nunique()}", "accent": C["primary"], "variant": "bar"},
    {"label": "Date Range", "value": f"{df['ym'].min()} to {df['ym'].max()}", "accent": C["primary"], "variant": "bar"},
    {"label": "Features", "value": f"{len(xs_cols)}", "accent": C["primary"], "variant": "bar"},
])

# --- Universe Size Over Time ---
st.header("Universe Size Over Time")
universe = df.groupby("ym")["permno"].nunique().sort_index()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=universe.index, y=universe.values, mode="lines",
    line=dict(color=STYLE["accent"], width=2),
))
fig.update_layout(
    template="alpha", height=300, margin=dict(t=20, b=30),
    yaxis_title="Number of Stocks",
)
st.plotly_chart(fig, use_container_width=True)

# --- Feature Distribution Viewer ---
st.header("Feature Distribution")
dist_col1, dist_col2 = st.columns([1, 3])

with dist_col1:
    selected_feature = st.selectbox("Feature", xs_cols)
    months = sorted(df["ym"].unique())
    selected_month = st.selectbox("Month", months, index=len(months) - 1)

with dist_col2:
    month_data = df[df["ym"] == selected_month][selected_feature].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=month_data.values, nbinsx=50, name="Distribution",
        marker_color=STYLE["accent"], opacity=0.7,
    ))
    fig.update_layout(
        title=f"{selected_feature} — {selected_month} (n={len(month_data)})",
        template="alpha", height=350, margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Correlation Heatmap ---
st.header("Feature Correlations (Spearman)")
n_features = st.slider("Number of features", min_value=5, max_value=40, value=20)

top_features = xs_cols[:n_features]
sample_month = df["ym"].max()
corr_data = df[df["ym"] == sample_month][top_features].dropna()
if len(corr_data) > 10:
    corr_matrix = corr_data.corr(method="spearman")
    fig = correlation_heatmap(corr_matrix)
    st.plotly_chart(fig, use_container_width=True)

# --- Missing Data Summary ---
st.header("Missing Data Summary")
missing_pct = df[xs_cols].isnull().mean().sort_values(ascending=False)
missing_nonzero = missing_pct[missing_pct > 0]
if len(missing_nonzero) > 0:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=missing_nonzero.index[:30], y=missing_nonzero.values[:30] * 100,
        marker_color=STYLE["warning"],
    ))
    fig.update_layout(
        title="% Missing by Feature (top 30)",
        yaxis_title="% Missing", template="alpha",
        height=350, margin=dict(t=40, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("No missing data in feature columns.")

render_next_steps("explore")
