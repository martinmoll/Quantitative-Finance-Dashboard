# dashboard/app.py
"""Alpha Strategy Dashboard — multipage Streamlit app.

This is the entry point. It handles:
- Page configuration and dark theme
- One-time data loading into session state
- Session state initialization

All UI lives in the pages/ directory.
"""

import streamlit as st
from pathlib import Path
from core.data_loader import load_dataset, compute_market_monthly, load_ff5_factors
from components.workflow import render_workflow_status, render_next_steps

st.set_page_config(
    page_title="Alpha Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

st.sidebar.title("Alpha Strategy Dashboard")
st.sidebar.markdown("---")
render_workflow_status("data")

st.title("Alpha Strategy Dashboard")
st.markdown(
    "Welcome to the Alpha Strategy Dashboard. Use the sidebar to navigate between pages."
)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Stocks in Dataset", f"{df['permno'].nunique():,}")
with col2:
    st.metric("Months", f"{df['ym'].nunique()}")
with col3:
    date_range = f"{df['ym'].min()} → {df['ym'].max()}"
    st.metric("Date Range", date_range)

render_next_steps("data", custom_message="Start by exploring your data or jump straight to building a model.")
