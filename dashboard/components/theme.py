"""Shared design tokens, Plotly template, and theme injection."""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "canvas": "#0B375D",
    "surface": "rgba(79,195,232,0.20)",  # cyan-tinted glass over the canvas
    "raised": "#1C6299",
    "hairline": "rgba(255,255,255,0.10)",
    "grid": "rgba(255,255,255,0.08)",
    "zeroline": "rgba(255,255,255,0.15)",
    "text": "#FFFFFF",
    "text_secondary": "#AFC6DD",
    "text_muted": "#7C97B4",
    "primary": "#4FC3E8",
    "positive": "#34E0A1",
    "negative": "#F46A6A",
    "warning": "#F5B13D",
}

PIN_COLORS = ["#A78BFA", "#2DD4BF", "#F472B6", "#FB923C"]

COLORWAY = ["#4FC3E8", "#34E0A1", "#F5B13D", "#A78BFA", "#2DD4BF", "#F472B6"]

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


# Force the app background to the navy canvas via CSS. config.toml's
# backgroundColor is unreliable (needs a full restart and can be missed), so we
# set it deterministically here instead of depending on the base theme.
_APP_BG_CSS = (
    "<style>"
    f'.stApp, [data-testid="stAppViewContainer"] {{ background-color: {COLORS["canvas"]}; }}'
    '[data-testid="stHeader"] { background-color: transparent; }'
    "</style>"
)


def inject_theme():
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    st.markdown(_APP_BG_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Ledger — light design system. Only page 4 (Backtest Results) uses it for
# now. The other pages keep COLORS and the "alpha" template until the
# redesign is approved for them.
# ---------------------------------------------------------------------------
LEDGER = {
    "canvas": "#F4F4F1",
    "surface": "#FFFFFF",
    "hairline": "#DEDFDB",
    "grid": "#ECEDEA",
    "text": "#121417",
    "text_secondary": "#5A606B",
    "primary": "#1F3FBF",
    "benchmark": "#C2600E",
    "positive": "#0B6E4F",
    "negative": "#B4232C",
    "warning": "#8A5A00",
    "neutral": "#9AA0AB",
}

LEDGER_SANS = "'Instrument Sans', sans-serif"
LEDGER_MONO = "'DM Mono', monospace"

pio.templates["ledger"] = go.layout.Template(layout=dict(
    paper_bgcolor=LEDGER["surface"],
    plot_bgcolor=LEDGER["surface"],
    font=dict(family="DM Mono, monospace", size=11, color=LEDGER["text_secondary"]),
    title=dict(font=dict(family="Instrument Sans, sans-serif", size=15, color=LEDGER["text"])),
    margin=dict(t=40, b=40, l=56, r=16),
    xaxis=dict(gridcolor=LEDGER["grid"], zerolinecolor=LEDGER["hairline"], linecolor=LEDGER["hairline"]),
    yaxis=dict(gridcolor=LEDGER["grid"], zerolinecolor=LEDGER["hairline"], linecolor=LEDGER["hairline"]),
    colorway=[LEDGER["primary"], LEDGER["benchmark"], "#5B2A86", "#0B6E4F", "#8A5A00", "#3B4048"],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=LEDGER["text"])),
    hoverlabel=dict(bgcolor=LEDGER["surface"], bordercolor=LEDGER["hairline"],
                    font=dict(family="DM Mono, monospace", color=LEDGER["text"])),
))

# The app runs on Streamlit's dark base theme. These rules make the main area
# light and leave the sidebar dark. Inline styles in page HTML still win.
# Containers created with a key that starts with "ledger-card" become cards.
_LEDGER_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color: {LEDGER["canvas"]}; }}
[data-testid="stHeader"] {{ background-color: transparent; }}
[data-testid="stMain"] {{ color: {LEDGER["text"]}; font-family: {LEDGER_SANS}; }}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] :is(p, li, h1, h2, h3, h4, strong, em, td, th),
[data-testid="stMain"] [data-testid="stHeading"] :is(h1, h2, h3, h4),
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stAlertContainer"] p,
[data-testid="stMain"] [data-testid="stExpander"] summary p {{ color: {LEDGER["text"]}; font-family: {LEDGER_SANS}; }}
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {{ color: {LEDGER["text_secondary"]}; }}
[data-testid="stMain"] [data-testid="stExpander"] details {{ background: {LEDGER["surface"]}; border: 1px solid {LEDGER["hairline"]}; border-radius: 4px; }}
[data-testid="stMain"] [data-testid="stExpander"] summary svg {{ fill: {LEDGER["text"]}; color: {LEDGER["text"]}; }}
[data-testid="stMain"] hr {{ border-color: {LEDGER["hairline"]}; }}
[data-testid="stMain"] [data-baseweb="tab-list"] button p {{ color: {LEDGER["text_secondary"]}; font-size: 14px; }}
[data-testid="stMain"] [data-baseweb="tab-list"] button[aria-selected="true"] p {{ color: {LEDGER["text"]}; font-weight: 600; }}
[data-testid="stMain"] [data-baseweb="tab-highlight"] {{ background-color: {LEDGER["text"]}; }}
[data-testid="stMain"] [data-baseweb="tab-border"] {{ background-color: {LEDGER["hairline"]}; }}
[data-testid="stMain"] [data-testid="stBaseButton-segmented_control"] {{
    background: {LEDGER["surface"]}; border-color: {LEDGER["hairline"]}; color: {LEDGER["text_secondary"]};
}}
[data-testid="stMain"] [data-testid="stBaseButton-segmented_controlActive"] {{
    background: #EEF1FC; border-color: {LEDGER["primary"]}; color: {LEDGER["primary"]};
}}
[data-testid="stMain"] [data-testid^="stBaseButton-segmented_control"] p {{ color: inherit; }}
[data-testid="stMain"] [data-baseweb="input"],
[data-testid="stMain"] [data-baseweb="base-input"],
[data-testid="stMain"] [data-baseweb="select"] > div,
[data-testid="stMain"] [data-testid="stNumberInputContainer"] {{
    background: {LEDGER["surface"]}; border-color: {LEDGER["hairline"]};
}}
[data-testid="stMain"] [data-baseweb="input"] input,
[data-testid="stMain"] [data-baseweb="select"] div,
[data-testid="stMain"] [data-baseweb="select"] input {{ color: {LEDGER["text"]}; }}
[data-testid="stMain"] [data-baseweb="select"] svg {{ color: {LEDGER["text_secondary"]}; fill: {LEDGER["text_secondary"]}; }}
[data-testid="stMain"] :is([data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"]) {{
    background: {LEDGER["surface"]}; color: {LEDGER["text"]}; border-color: {LEDGER["hairline"]};
}}
[data-testid="stMain"] [data-testid="stTable"] > div {{ background: {LEDGER["surface"]}; border-color: {LEDGER["hairline"]}; }}
[data-testid="stMain"] [data-testid="stTableStyledTable"] {{ color: {LEDGER["text"]}; font-family: {LEDGER_MONO}; }}
[data-testid="stMain"] [data-testid="stTableStyledTable"] :is(th, td) {{ border-color: {LEDGER["hairline"]}; }}
[data-testid="stMain"] [data-testid="stTableStyledTable"] th,
[data-testid="stMain"] [data-testid="stTableStyledTable"] th [data-testid="stMarkdownContainer"] {{
    color: {LEDGER["text_secondary"]}; font-family: {LEDGER_SANS};
}}
[data-testid="stHeader"] [data-testid="stToolbar"],
[data-testid="stHeader"] [data-testid="stToolbar"] * {{ color: {LEDGER["text_secondary"]}; }}
[data-testid="stMain"] [class*="st-key-ledger-card"] {{
    background: {LEDGER["surface"]}; border: 1px solid {LEDGER["hairline"]};
    border-radius: 4px; padding: 20px 22px;
}}
.mono {{ font-family: {LEDGER_MONO}; font-variant-numeric: tabular-nums; }}
</style>
"""


def inject_ledger_theme():
    st.markdown(_LEDGER_CSS, unsafe_allow_html=True)
