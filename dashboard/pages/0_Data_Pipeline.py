"""Data Pipeline page — manual data refresh from live sources."""

import sys
import os
import streamlit as st
from pathlib import Path

# Add project root to path so pipeline package is importable
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from components.workflow import render_workflow_status, render_next_steps
from components.theme import inject_theme, COLORS, FONT_SANS
from components.metrics import metric_card_row

inject_theme()

C = COLORS
st.markdown(
    f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
    f'color:{C["text"]};margin:0;">Data Pipeline</h1>',
    unsafe_allow_html=True,
)
render_workflow_status("data")
st.markdown("Fetch the latest market data and update the dataset.")

df = st.session_state.get("df")
if df is not None:
    metric_card_row([
        {"label": "Stocks", "value": f"{df['permno'].nunique():,}", "accent": C["primary"], "variant": "bar"},
        {"label": "Months", "value": f"{df['ym'].nunique()}", "accent": C["primary"], "variant": "bar"},
        {"label": "Date Range", "value": f"{df['ym'].min()} to {df['ym'].max()}", "accent": C["primary"], "variant": "bar"},
    ])
    st.markdown("---")

universe = st.selectbox(
    "Stock Universe",
    ["S&P 500", "Nasdaq-100", "S&P 500 + Nasdaq-100", "Oslo Børs"],
    help="Which constituents to fetch. Oslo Børs builds a separate Norway dataset "
         "(CAPM vs OSEBX, no Fama-French factors).",
)
region = "NO" if universe == "Oslo Børs" else "US"

period = st.selectbox(
    "History Window",
    ["1y", "2y", "3y", "5y", "10y", "max"],
    index=2,
    help="How far back to download daily prices. Longer windows give more "
         "history for beta/volatility features but take longer to fetch.",
)

fred_key = os.environ.get("FRED_API_KEY", "")
if not fred_key:
    fred_key = st.text_input(
        "FRED API Key (optional — needed for macro features)",
        type="password",
        help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
    )
    if fred_key:
        os.environ["FRED_API_KEY"] = fred_key

st.markdown("---")

universe_info = {
    "S&P 500": ("~500 stocks", "10-20 minutes"),
    "Nasdaq-100": ("~100 stocks", "3-5 minutes"),
    "S&P 500 + Nasdaq-100": ("~530 stocks (deduplicated)", "10-25 minutes"),
    "Oslo Børs": ("~40 stocks", "2-4 minutes"),
}
count, est_time = universe_info[universe]

st.markdown(
    "**Data sources:** Yahoo Finance (prices, fundamentals), "
    "Ken French Library (FF5 factors), FRED (VIX, yield curve, EPU)"
)
st.markdown(
    f"**Universe:** {universe} ({count})  \n"
    f"**Estimated time:** {est_time}"
)

if st.button("Refresh Data", type="primary", use_container_width=True):
    st.markdown("---")
    status = st.status("Running pipeline...", expanded=True)
    progress_bar = st.progress(0)
    log_container = st.empty()
    logs = []

    stages = {
        "universe": 0.05,
        "prices": 0.30,
        "fundamentals": 0.60,
        "factors": 0.70,
        "macro": 0.75,
        "features": 0.90,
        "assembly": 1.0,
    }

    def progress_callback(stage, detail=""):
        pct = stages.get(stage, 0)
        progress_bar.progress(pct)
        msg = f"**{stage.title()}**: {detail}"
        logs.append(msg)
        status.update(label=f"Running pipeline... ({stage})")
        log_container.markdown("\n\n".join(logs[-10:]))

    try:
        from pipeline import run_pipeline

        result = run_pipeline(
            fred_api_key=fred_key or None,
            universe=universe,
            region=region,
            period=period,
            progress_callback=progress_callback,
        )

        progress_bar.progress(1.0)

        if result.success:
            status.update(label="Pipeline complete!", state="complete")
            st.success(
                f"Added {len(result.months_added)} month(s): "
                f"{', '.join(result.months_added[:5])}"
                f"{'...' if len(result.months_added) > 5 else ''}. "
                f"{result.tickers_fetched} tickers fetched. "
                f"Dataset now has {result.total_rows:,} rows."
            )
            if result.tickers_failed:
                st.warning(
                    f"{len(result.tickers_failed)} tickers failed: "
                    f"{', '.join(result.tickers_failed[:10])}"
                    f"{'...' if len(result.tickers_failed) > 10 else ''}"
                )

            from core.data_loader import (
                load_dataset, compute_market_monthly, load_ff5_factors,
                label_for_region,
            )
            from features import precompute_features
            from pipeline.config import dataset_paths

            pq_path, csv_path = dataset_paths(region)
            new_path = pq_path if pq_path.exists() else csv_path
            new_df = load_dataset(path=new_path)
            new_df = precompute_features(new_df)
            st.session_state.df = new_df
            st.session_state.market_monthly = compute_market_monthly(new_df)

            if region == "US":
                ff5_path = Path(__file__).parent.parent.parent / "Data" / "ff5_factors.csv"
                st.session_state.ff5_factors = (
                    load_ff5_factors(ff5_path) if ff5_path.exists() else None
                )
            else:
                # No regional Fama-French factors — CAPM only (vs the local market).
                st.session_state.ff5_factors = None

            # Switching universe invalidates any prior backtest/pins.
            for k in ["backtest_result", "backtest_predictions", "backtest_params",
                      "backtest_feature_importance"]:
                st.session_state[k] = None
            st.session_state.pinned_configs = []
            st.session_state["active_region"] = label_for_region(region)

            st.cache_data.clear()
            st.info("Session data reloaded. Navigate to other pages to use updated data.")
        else:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline failed: {result.error}")
            if result.tickers_failed:
                st.warning(
                    f"Failed tickers: {', '.join(result.tickers_failed[:20])}"
                )

    except Exception as e:
        status.update(label="Pipeline error", state="error")
        st.error(f"Unexpected error: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info(
        "Click **Refresh Data** to fetch the latest market data from Yahoo Finance, "
        "FRED, and Ken French's data library. This may take 10-20 minutes."
    )

render_next_steps("data")
