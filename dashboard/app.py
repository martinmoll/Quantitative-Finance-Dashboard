# dashboard/app.py
"""Alpha Strategy Dashboard — multipage Streamlit app."""

import streamlit as st
from pathlib import Path
from core.data_loader import load_dataset, compute_market_monthly, load_ff5_factors
from components.workflow import render_workflow_status
from components.theme import inject_theme, COLORS, FONT_MONO, FONT_SANS

st.set_page_config(
    page_title="Alpha Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

DATA_DIR = Path(__file__).parent.parent / "Data"
_dataset_exists = (DATA_DIR / "alpha_dataset_v2.parquet").exists() or \
                  (DATA_DIR / "alpha_dataset_v2.csv").exists()


@st.cache_data
def _load_all_data():
    df = load_dataset(auto_generate=True)

    from features import precompute_features
    df = precompute_features(df)

    market = compute_market_monthly(df)

    ff5_path = Path(__file__).parent.parent / "Data" / "ff5_factors.csv"
    ff5 = None
    if ff5_path.exists():
        ff5 = load_ff5_factors(ff5_path)

    return df, market, ff5


if not _dataset_exists:
    st.info(
        "No dataset found. Fetching live market data and building the dataset — "
        "this may take a few minutes on first run..."
    )

df, market_monthly, ff5_factors = _load_all_data()

if "df" not in st.session_state:
    st.session_state.df = df
if "market_monthly" not in st.session_state:
    st.session_state.market_monthly = market_monthly
if "ff5_factors" not in st.session_state:
    st.session_state.ff5_factors = ff5_factors
if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None
if "backtest_params" not in st.session_state:
    st.session_state.backtest_params = None
if "pinned_configs" not in st.session_state:
    st.session_state.pinned_configs = []
if "portfolio_weights" not in st.session_state:
    st.session_state.portfolio_weights = None

C = COLORS

# --- Sidebar ---
st.sidebar.title("Alpha Strategy Dashboard")
st.sidebar.markdown("---")
render_workflow_status("data")

# --- Header row: eyebrow + title + dataset chips ---
header_left, header_right = st.columns([2, 1])
with header_left:
    st.markdown(
        f'<p style="font-family:{FONT_MONO};font-size:12px;font-weight:500;'
        f'color:{C["primary"]};text-transform:uppercase;letter-spacing:.05em;'
        f'margin:0 0 4px;">RMBI3110 &middot; Walk-forward backtester</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h1 style="font-family:{FONT_SANS};font-size:30px;font-weight:700;'
        f'color:{C["text"]};margin:0 0 8px;">Alpha Strategy Dashboard</h1>',
        unsafe_allow_html=True,
    )

with header_right:
    n_stocks = f"{df['permno'].nunique():,}"
    n_months = f"{df['ym'].nunique()}"
    date_range = f"{df['ym'].min()} &rarr; {df['ym'].max()}"
    chip_style = (
        f"background:{C['surface']};border:1px solid {C['hairline']};border-radius:8px;"
        f"padding:8px 14px;text-align:center;"
    )
    label_s = (
        f"font-family:{FONT_SANS};font-size:10px;font-weight:500;"
        f"text-transform:uppercase;letter-spacing:.05em;color:{C['text_secondary']};"
        "margin:0;"
    )
    value_s = (
        f"font-family:{FONT_MONO};font-size:18px;font-weight:500;"
        f"color:{C['text']};margin:2px 0 0;"
    )
    st.markdown(
        f'<div style="display:flex;gap:10px;justify-content:flex-end;margin-top:8px;">'
        f'<div style="{chip_style}"><p style="{label_s}">Stocks</p><p style="{value_s}">{n_stocks}</p></div>'
        f'<div style="{chip_style}"><p style="{label_s}">Months</p><p style="{value_s}">{n_months}</p></div>'
        f'<div style="{chip_style}"><p style="{label_s}">Range</p><p style="{value_s}">{date_range}</p></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# --- Stepper ---
render_workflow_status("data")

# --- Two-column body ---
col_left, col_right = st.columns([1.55, 1])

with col_left:
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#15243c,#101a2c);'
        f'border:1px solid rgba(91,155,255,0.22);border-radius:14px;'
        f'padding:28px 24px;">'
        f'<p style="font-family:{FONT_SANS};font-size:11px;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:.06em;color:{C["primary"]};'
        f'margin:0 0 10px;">Start here</p>'
        f'<h2 style="font-family:{FONT_SANS};font-size:22px;font-weight:700;'
        f'color:{C["text"]};margin:0 0 8px;">Explore your data, then build a model</h2>'
        f'<p style="font-family:{FONT_SANS};font-size:13px;color:{C["text_secondary"]};'
        f'margin:0 0 20px;line-height:1.5;">'
        f'{n_stocks} stocks across {n_months} months are loaded and clean. '
        f'Jump into the Data Explorer or skip ahead to the Alpha Model Lab.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    btn_l, btn_r, _ = st.columns([1, 1, 2])
    with btn_l:
        if st.button("Explore data →", type="primary"):
            st.switch_page("pages/1_Data_Explorer.py")
    with btn_r:
        if st.button("Build a model"):
            st.switch_page("pages/3_Alpha_Model_Lab.py")

with col_right:
    result = st.session_state.get("backtest_result")
    params = st.session_state.get("backtest_params")
    if result is not None:
        rets = result["monthly_returns"]
        cum = (1 + rets).cumprod()
        final_wealth = cum.iloc[-1] * 10000
        total_ret = cum.iloc[-1] - 1
        sr = rets.mean() / rets.std() * (12 ** 0.5) if rets.std() > 0 else 0
        mdd_cum = cum / cum.cummax() - 1
        mdd = mdd_cum.min()

        model_name = params.get("model_name", "—") if params else "—"

        chip_html = (
            f'<span style="background:rgba(91,155,255,0.14);color:{C["primary"]};'
            f'font-family:{FONT_SANS};font-size:10px;font-weight:600;padding:2px 8px;'
            f'border-radius:6px;position:absolute;top:14px;right:14px;">{model_name}</span>'
        )

        st.markdown(
            f'<div style="background:{C["surface"]};border:1px solid {C["hairline"]};'
            f'border-radius:14px;padding:20px 22px;position:relative;">'
            f'{chip_html}'
            f'<p style="font-family:{FONT_SANS};font-size:13px;font-weight:600;'
            f'color:{C["text_secondary"]};margin:0 0 6px;">Last run</p>'
            f'<p style="font-family:{FONT_MONO};font-size:28px;font-weight:500;'
            f'color:{C["text"]};margin:0;">${final_wealth:,.0f}</p>'
            f'<p style="font-family:{FONT_MONO};font-size:12px;color:{C["positive"]};'
            f'margin:4px 0 14px;">+{total_ret:.0%}</p>'
            f'<div style="display:flex;gap:20px;">'
            f'<span style="font-family:{FONT_SANS};font-size:12px;color:{C["text_secondary"]};">'
            f'Sharpe <span class="mono" style="color:{C["text"]};font-weight:500;">{sr:.2f}</span></span>'
            f'<span style="font-family:{FONT_SANS};font-size:12px;color:{C["text_secondary"]};">'
            f'MDD <span class="mono" style="color:{C["negative"]};font-weight:500;">{mdd:.1%}</span></span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("View full results →"):
            st.switch_page("pages/4_Backtest_Results.py")
    else:
        st.markdown(
            f'<div style="background:{C["surface"]};border:1px solid {C["hairline"]};'
            f'border-radius:14px;padding:28px 22px;text-align:center;">'
            f'<p style="font-family:{FONT_SANS};font-size:13px;font-weight:600;'
            f'color:{C["text_secondary"]};margin:0 0 8px;">Last run</p>'
            f'<p style="font-family:{FONT_SANS};font-size:13px;color:{C["text_muted"]};'
            f'margin:0;">No backtest run yet. Build a model to see results here.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
