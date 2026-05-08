import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV

from features import build_features_linear, build_features_ensemble
from engine import run_predictions, build_portfolio, compute_market_monthly, compute_perf
import cache_manager as cache

st.set_page_config(page_title="Alpha Dashboard", layout="wide")
st.title("Alpha Strategy Dashboard")


@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / 'Data' / 'alpha_dataset_v2.csv'
    df = pd.read_csv(data_path)
    market = compute_market_monthly(df)
    spy_oos = market.loc[market.index >= '2015-01', 'spy_ret']
    return df, market, spy_oos


df, market_monthly, spy_oos = load_data()


# ---------------------------------------------------------------------------
# Top control bar (always visible)
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    model_type = st.selectbox("Model", ["HGB", "Lasso"])

with col2:
    K = st.slider("K (stocks)", min_value=5, max_value=50, step=5, value=10)

with col3:
    vol_tilt = st.slider("Vol tilt", min_value=0.0, max_value=0.50, step=0.01, value=0.05)

with col4:
    regime_lookback = st.slider("Regime lookback", min_value=0, max_value=12, value=6)


# ---------------------------------------------------------------------------
# Expandable model hyperparameters
# ---------------------------------------------------------------------------

with st.expander("Model Hyperparameters", expanded=False):
    if model_type == "HGB":
        hp_col1, hp_col2, hp_col3 = st.columns(3)
        with hp_col1:
            max_depth = st.slider("max_depth", min_value=1, max_value=6, value=2)
            learning_rate = st.slider(
                "learning_rate", min_value=0.01, max_value=0.20, step=0.01, value=0.05
            )
        with hp_col2:
            min_samples_leaf = st.slider(
                "min_samples_leaf", min_value=100, max_value=2000, step=100, value=500
            )
            l2_regularization = st.slider(
                "l2_regularization", min_value=0.0, max_value=1.0, step=0.05, value=0.1
            )
        with hp_col3:
            max_iter = st.slider(
                "max_iter", min_value=100, max_value=1000, step=50, value=500
            )
        model_params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            "max_iter": max_iter,
        }
    else:  # Lasso
        lasso_col1, lasso_col2 = st.columns(2)
        with lasso_col1:
            cv_folds = st.slider("cv folds", min_value=3, max_value=10, value=5)
        with lasso_col2:
            lasso_max_iter = st.slider(
                "max_iter", min_value=1000, max_value=10000, step=1000, value=5000
            )
        model_params = {
            "cv": cv_folds,
            "max_iter": lasso_max_iter,
        }

    retrain_every = st.slider(
        "retrain_every", min_value=3, max_value=24, step=3, value=12
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "pinned" not in st.session_state:
    st.session_state.pinned = []  # list of {label, result, params}

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "current_params" not in st.session_state:
    st.session_state.current_params = None


# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------

btn_col1, btn_col2 = st.columns([1, 1])

with btn_col1:
    run_clicked = st.button("Run Backtest", type="primary")

with btn_col2:
    pin_clicked = st.button("Pin Config")


# ---------------------------------------------------------------------------
# Run Backtest logic
# ---------------------------------------------------------------------------

if run_clicked:
    # --- Step 1: prediction cache ---
    pred_key = cache.prediction_key(model_type, model_params, retrain_every)
    predictions = cache.get_predictions(pred_key)

    if predictions is None:
        # Build estimator
        if model_type == "HGB":
            estimator = HistGradientBoostingRegressor(
                max_depth=model_params["max_depth"],
                learning_rate=model_params["learning_rate"],
                min_samples_leaf=model_params["min_samples_leaf"],
                l2_regularization=model_params["l2_regularization"],
                max_iter=model_params["max_iter"],
                random_state=42,
            )
            feature_builder = build_features_ensemble
        else:
            estimator = LassoCV(
                cv=model_params["cv"],
                max_iter=model_params["max_iter"],
                random_state=42,
            )
            feature_builder = build_features_linear

        progress = st.progress(0, text="Running walk-forward backtest...")
        predictions = run_predictions(
            df,
            feature_builder,
            estimator,
            retrain_every,
            progress_callback=lambda step, total, month: progress.progress(
                step / total,
                text=f"Training... {month} ({step}/{total})",
            ),
        )
        progress.empty()
        cache.save_predictions(pred_key, predictions)

    # --- Step 2: portfolio cache ---
    port_key = cache.portfolio_key(pred_key, K, vol_tilt, regime_lookback)
    portfolio = cache.get_portfolio(port_key)

    if portfolio is None:
        portfolio = build_portfolio(
            predictions,
            K=K,
            vol_tilt=vol_tilt,
            regime_lookback=regime_lookback,
            market_monthly=market_monthly,
        )
        cache.save_portfolio(port_key, portfolio)

    # --- Step 3: store in session state ---
    st.session_state.current_result = portfolio
    st.session_state.current_params = {
        "model_type": model_type,
        "model_params": model_params,
        "retrain_every": retrain_every,
        "K": K,
        "vol_tilt": vol_tilt,
        "regime_lookback": regime_lookback,
    }


# ---------------------------------------------------------------------------
# Pin Config logic
# ---------------------------------------------------------------------------

if pin_clicked:
    result = st.session_state.current_result
    params = st.session_state.current_params
    if result is not None and len(st.session_state.pinned) < 4:
        mt = params["model_type"]
        k_val = params["K"]
        vt_val = params["vol_tilt"]
        label = f"{mt} K={k_val} vt={vt_val}"
        st.session_state.pinned.append(
            {"label": label, "result": result, "params": params}
        )


# ---------------------------------------------------------------------------
# Pinned config chips
# ---------------------------------------------------------------------------

if st.session_state.pinned:
    st.markdown("**Pinned configs:**")
    chip_cols = st.columns(len(st.session_state.pinned))
    to_remove = None
    for i, pinned_item in enumerate(st.session_state.pinned):
        with chip_cols[i]:
            if st.button(f"❌ {pinned_item['label']}", key=f"remove_pin_{i}"):
                to_remove = i
    if to_remove is not None:
        st.session_state.pinned.pop(to_remove)
        st.rerun()


# ---------------------------------------------------------------------------
# Results area
# ---------------------------------------------------------------------------

result = st.session_state.current_result

if result is not None:
    pass  # --- Charts rendered below (Tasks 5-8) ---
else:
    st.info("Click **Run Backtest** to see results.")
