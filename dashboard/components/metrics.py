"""Reusable Streamlit metric display components — modernized."""

import streamlit as st
import pandas as pd
import numpy as np
from components.theme import COLORS, FONT_MONO, FONT_SANS


# ---------------------------------------------------------------------------
# metric_card — replaces st.metric rows with sentiment-colored HTML cards
# ---------------------------------------------------------------------------

_CARD_BASE = (
    "background:{bg};border:1px solid {border};border-radius:10px;"
    "padding:13px 16px;min-height:90px;position:relative;"
)

_CARD_BAR = "border-left:4px solid {accent};"

_LABEL_STYLE = (
    f"font-family:{FONT_SANS};font-size:10px;font-weight:500;"
    f"text-transform:uppercase;letter-spacing:.05em;color:{COLORS['text_secondary']};"
    "margin:0 0 6px;"
)

_VALUE_STYLE = (
    f"font-family:{FONT_MONO};font-variant-numeric:tabular-nums;"
    f"font-size:24px;font-weight:500;color:{COLORS['text']};margin:0;line-height:1.2;"
)

_DELTA_STYLE = (
    f"font-family:{FONT_MONO};font-size:11px;margin:4px 0 0;"
)

_CHIP_STYLE = (
    "position:absolute;top:10px;right:10px;font-size:10px;font-weight:600;"
    f"font-family:{FONT_SANS};padding:2px 8px;border-radius:6px;"
)


def metric_card(
    label: str,
    value: str,
    *,
    accent: str = COLORS["primary"],
    delta: str | None = None,
    delta_color: str | None = None,
    chip: str | None = None,
    variant: str = "bar",
):
    border = COLORS["hairline"]
    bg = COLORS["surface"]

    style = _CARD_BASE.format(bg=bg, border=border)
    if variant == "bar":
        style += _CARD_BAR.format(accent=accent)

    chip_html = ""
    if chip:
        chip_bg = accent.replace(")", ",0.14)").replace("rgb", "rgba") if "rgb" in accent else accent + "24"
        chip_html = (
            f'<div style="{_CHIP_STYLE}background:{chip_bg};color:{accent};">'
            f'{chip}</div>'
        )

    delta_html = ""
    if delta:
        d_color = delta_color or COLORS["text_secondary"]
        delta_html = f'<p style="{_DELTA_STYLE}color:{d_color};">{delta}</p>'

    html = (
        f'<div style="{style}">'
        f'{chip_html}'
        f'<p style="{_LABEL_STYLE}">{label}</p>'
        f'<p style="{_VALUE_STYLE}">{value}</p>'
        f'{delta_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def metric_card_row(cards: list[dict]):
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            metric_card(**card)


# Legacy wrapper — keeps old call sites working during migration
def metric_row(metrics: list[dict]):
    cards = []
    for m in metrics:
        accent = COLORS["primary"]
        delta_str = None
        delta_color = None
        d = m.get("delta")
        if d is not None:
            delta_str = str(d)
            dc = m.get("delta_color", "normal")
            if dc == "normal":
                delta_color = COLORS["positive"]
            elif dc == "inverse":
                delta_color = COLORS["negative"]
            else:
                delta_color = COLORS["text_secondary"]
        cards.append({
            "label": m["label"],
            "value": m["value"],
            "accent": accent,
            "delta": delta_str,
            "delta_color": delta_color,
            "variant": "bar",
        })
    metric_card_row(cards)


# ---------------------------------------------------------------------------
# banner — compact interpretation with optional expander
# ---------------------------------------------------------------------------

_BANNER_ICONS = {
    "success": ("&#10003;", COLORS["positive"]),
    "info": ("i", COLORS["primary"]),
    "warning": ("!", COLORS["warning"]),
    "error": ("&#10005;", COLORS["negative"]),
}


def banner(kind: str, headline_html: str, detail: str | None = None):
    icon_char, accent = _BANNER_ICONS.get(kind, ("i", COLORS["primary"]))
    bg = f"rgba({_hex_to_rgb(accent)},0.08)"
    border = f"rgba({_hex_to_rgb(accent)},0.22)"

    icon_html = (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:22px;height:22px;border-radius:6px;background:rgba({_hex_to_rgb(accent)},0.14);'
        f'color:{accent};font-size:12px;font-weight:700;margin-right:10px;flex-shrink:0;">'
        f'{icon_char}</span>'
    )

    st.markdown(
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
        f'padding:10px 14px;display:flex;align-items:center;margin:8px 0;">'
        f'{icon_html}'
        f'<span style="font-family:{FONT_SANS};font-size:13px;color:{COLORS["text"]};'
        f'line-height:1.4;">{headline_html}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if detail:
        with st.expander("Why this matters"):
            st.markdown(detail)


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# ---------------------------------------------------------------------------
# Tables — comparison & regression (kept, with updated VIF colors)
# ---------------------------------------------------------------------------

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
        banner("success",
               f'Alpha significant — <span class="mono">|t| = {alpha_t:.2f}</span> > 1.96')
    else:
        banner("warning",
               f'Alpha not significant — <span class="mono">|t| = {alpha_t:.2f}</span> < 1.96')

    st.dataframe(df, use_container_width=True)

    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        metric_card("R²", f"{result['r_squared']:.4f}", variant="minimal")
        metric_card("Adj R²", f"{result['adj_r_squared']:.4f}", variant="minimal")
    with stats_col2:
        metric_card("Durbin-Watson", f"{result['durbin_watson']:.2f}", variant="minimal")
        jb = result.get("jarque_bera", {})
        metric_card("Jarque-Bera", f"{jb.get('statistic', 0):.2f} (p={jb.get('pvalue', 0):.3f})", variant="minimal")


def vif_table(vif_df: pd.DataFrame):
    def color_vif(val):
        if val < 5:
            return f"background-color: rgba({_hex_to_rgb(COLORS['positive'])},0.15)"
        elif val < 10:
            return f"background-color: rgba({_hex_to_rgb(COLORS['warning'])},0.15)"
        else:
            return f"background-color: rgba({_hex_to_rgb(COLORS['negative'])},0.15)"

    styled = vif_df.style.map(color_vif, subset=["VIF"])
    st.dataframe(styled, use_container_width=True)
