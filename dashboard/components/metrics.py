"""Reusable Streamlit metric display components."""

import streamlit as st
import pandas as pd
import numpy as np


def metric_row(metrics: list[dict]):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            delta = m.get("delta")
            delta_color = m.get("delta_color", "normal")
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=delta,
                delta_color=delta_color,
            )


def comparison_table(configs: list[dict], highlight: bool = True):
    df = pd.DataFrame(configs)
    if "Strategy" in df.columns:
        df = df.set_index("Strategy")

    format_map = {
        "SR": "{:.2f}",
        "Ann Return": "{:.1%}",
        "Ann Vol": "{:.1%}",
        "MDD": "{:.1%}",
        "Calmar": "{:.2f}",
        "Total Return": "{:.0%}",
        "Mean IC": "{:.4f}",
        "Mean Turnover": "{:.1%}",
        "TC Drag": "{:.2%}",
        "Net SR": "{:.2f}",
    }

    for col, fmt in format_map.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "-")

    st.dataframe(df, use_container_width=True)


def regression_table(result: dict):
    rows = []
    for var in result["coefficients"]:
        coef = result["coefficients"][var]
        se = result["hac_se"][var]
        t = result["t_stats"][var]
        p = result["p_values"][var]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        rows.append({
            "Variable": var,
            "Coefficient": f"{coef:.4f}",
            "HAC SE": f"{se:.4f}",
            "t-stat": f"{t:.2f}",
            "p-value": f"{p:.4f}",
            "Sig": sig,
        })

    df = pd.DataFrame(rows).set_index("Variable")

    alpha_t = abs(result["t_stats"].get("const", 0))
    if alpha_t > 1.96:
        st.success(f"Alpha significant (|t| = {alpha_t:.2f} > 1.96)")
    else:
        st.warning(f"Alpha not significant (|t| = {alpha_t:.2f} < 1.96)")

    st.dataframe(df, use_container_width=True)

    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.metric("R²", f"{result['r_squared']:.4f}")
        st.metric("Adj R²", f"{result['adj_r_squared']:.4f}")
    with stats_col2:
        st.metric("Durbin-Watson", f"{result['durbin_watson']:.2f}")
        jb = result.get("jarque_bera", {})
        st.metric("Jarque-Bera", f"{jb.get('statistic', 0):.2f} (p={jb.get('pvalue', 0):.3f})")


def vif_table(vif_df: pd.DataFrame):
    def color_vif(val):
        if val < 5:
            return "background-color: rgba(0, 210, 106, 0.3)"
        elif val < 10:
            return "background-color: rgba(255, 184, 0, 0.3)"
        else:
            return "background-color: rgba(255, 68, 68, 0.3)"

    styled = vif_df.style.map(color_vif, subset=["VIF"])
    st.dataframe(styled, use_container_width=True)
