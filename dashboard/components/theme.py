"""Shared design tokens, Plotly template, and theme injection."""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "canvas": "#0B121F",
    "surface": "#141E2E",
    "raised": "#1B2738",
    "hairline": "rgba(255,255,255,0.07)",
    "grid": "rgba(255,255,255,0.06)",
    "zeroline": "rgba(255,255,255,0.1)",
    "text": "#E9EEF6",
    "text_secondary": "#93A1B5",
    "text_muted": "#64748B",
    "primary": "#5B9BFF",
    "positive": "#34E0A1",
    "negative": "#F46A6A",
    "warning": "#F5B13D",
}

PIN_COLORS = ["#A78BFA", "#2DD4BF", "#F472B6", "#FB923C"]

COLORWAY = ["#5B9BFF", "#34E0A1", "#F5B13D", "#A78BFA", "#2DD4BF", "#F472B6"]

FONT_SANS = "'IBM Plex Sans', sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"

# ---------------------------------------------------------------------------
# Plotly template — register once at import time
# ---------------------------------------------------------------------------
_TEMPLATE_LAYOUT = dict(
    paper_bgcolor=COLORS["canvas"],
    plot_bgcolor=COLORS["canvas"],
    font=dict(family="IBM Plex Mono, monospace", size=12, color=COLORS["text_secondary"]),
    margin=dict(t=40, b=40, l=50, r=20),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["zeroline"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["zeroline"]),
    colorway=COLORWAY,
)
pio.templates["alpha"] = go.layout.Template(layout=_TEMPLATE_LAYOUT)


def base_layout(**overrides) -> dict:
    layout = dict(
        template="alpha",
        height=400,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Theme injection — call at top of every page
# ---------------------------------------------------------------------------
_THEME_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.mono {
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
}
</style>
"""


def inject_theme():
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
