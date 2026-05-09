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
    return df, market


df, market_monthly = load_data()


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
# Display controls
# ---------------------------------------------------------------------------

all_months = sorted(df['ym'].unique())
oos_months = [m for m in all_months if m >= '2015-01']

disp_col1, disp_col2, disp_col3 = st.columns(3)
with disp_col1:
    display_start = st.select_slider(
        "Calculation start date", options=oos_months, value=oos_months[0])
with disp_col2:
    start_value = st.number_input(
        "Starting portfolio value ($)", min_value=1, value=10000, step=1000)
with disp_col3:
    cash_flow = st.number_input(
        "Cash flow per period ($)", value=0, step=100,
        help="Amount added before each rebalance. Negative = withdrawal.")


def filter_from(series, start):
    """Filter a Series or dict to only include months >= start."""
    if isinstance(series, pd.Series):
        return series[series.index >= start]
    return {m: v for m, v in series.items() if m >= start}


def compute_wealth(returns, start_val, cf):
    """Compute wealth path: each period adds cf before applying the return."""
    values = []
    bal = start_val
    for r in returns:
        bal = (bal + cf) * (1 + r)
        values.append(bal)
    return pd.Series(values, index=returns.index)


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
                early_stopping=False,
                random_state=42,
            )
            feature_builder = build_features_ensemble
        else:
            estimator = LassoCV(
                cv=model_params["cv"],
                max_iter=model_params["max_iter"],
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
    import plotly.graph_objects as go

    PIN_COLORS = ['#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6']
    rets = filter_from(result['monthly_returns'], display_start)
    ic_filtered = filter_from(result['ic'], display_start)
    turnover_filtered = filter_from(result['turnover'], display_start)
    holdings_filtered = filter_from(result['holdings'], display_start)
    spy_oos = filter_from(market_monthly.loc[market_monthly.index >= '2015-01', 'spy_ret'], display_start)

    # === Row 1: KPI Cards ===
    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    perf_stats = compute_perf(rets, ic=ic_filtered, turnover=turnover_filtered)
    sr = perf_stats['SR']
    kpi1.metric("Sharpe Ratio", f"{sr:.2f}")
    kpi2.metric("Ann. Return", f"{perf_stats['Ann Return']:.1%}")
    kpi3.metric("Ann. Volatility", f"{perf_stats['Ann Vol']:.1%}")
    mdd = perf_stats['MDD']
    kpi4.metric("Max Drawdown", f"{mdd:.1%}")

    # === Row 2: Cumulative Wealth + Drawdown ===
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        cum = compute_wealth(rets, start_value, cash_flow)
        spy_cum = compute_wealth(spy_oos, start_value, cash_flow)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name="Strategy",
            line=dict(color='#3b82f6', width=2)))
        fig.add_trace(go.Scatter(
            x=spy_cum.index, y=spy_cum.values, name="SPY",
            line=dict(color='gray', width=2, dash='dash')))

        for i, p in enumerate(st.session_state.pinned):
            p_rets = filter_from(p['result']['monthly_returns'], display_start)
            p_cum = compute_wealth(p_rets, start_value, cash_flow)
            fig.add_trace(go.Scatter(
                x=p_cum.index, y=p_cum.values, name=p['label'],
                line=dict(color=PIN_COLORS[i % len(PIN_COLORS)], width=1.5)))

        cf_label = f", ${cash_flow:+,}/mo" if cash_flow != 0 else ""
        fig.update_layout(
            title="Cumulative Wealth",
            yaxis_title=f"Portfolio Value (${start_value:,} start{cf_label})",
            yaxis_tickprefix="$", yaxis_tickformat=",.0f",
            template="plotly_dark", height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, width='stretch')

    with chart_col2:
        wealth = compute_wealth(rets, start_value, cash_flow)
        dd = wealth / wealth.cummax() - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, fill='tozeroy', name="Drawdown",
            line=dict(color='#ef4444', width=1),
            fillcolor='rgba(239,68,68,0.3)'))

        for i, p in enumerate(st.session_state.pinned):
            p_rets = filter_from(p['result']['monthly_returns'], display_start)
            p_wealth = compute_wealth(p_rets, start_value, cash_flow)
            p_dd = p_wealth / p_wealth.cummax() - 1
            fig.add_trace(go.Scatter(
                x=p_dd.index, y=p_dd.values, name=p['label'],
                line=dict(color=PIN_COLORS[i % len(PIN_COLORS)], width=1)))

        fig.update_layout(
            title="Drawdown", yaxis_title="Drawdown %",
            yaxis_tickformat='.0%', template="plotly_dark",
            height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, width='stretch')

    # === Row 3: Sector Allocation + Holdings Table ===
    st.markdown("---")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        sector_data = {}
        for m, held in holdings_filtered.items():
            if 'sector' in held.columns:
                counts = held['sector'].value_counts(normalize=True)
                sector_data[m] = counts.to_dict()

        if sector_data:
            sector_df = pd.DataFrame(sector_data).T.fillna(0).sort_index()
            fig = go.Figure()
            for col in sector_df.columns:
                fig.add_trace(go.Scatter(
                    x=sector_df.index, y=sector_df[col],
                    stackgroup='one', name=col, mode='lines'))
            fig.update_layout(
                title="Sector Allocation Over Time",
                yaxis_title="Weight", yaxis_tickformat='.0%',
                template="plotly_dark", height=400, margin=dict(t=40, b=40))
            st.plotly_chart(fig, width='stretch')

    with comp_col2:
        last_month = sorted(holdings_filtered.keys())[-1]
        last_h = holdings_filtered[last_month].copy()
        display_cols = []
        if 'permno' in last_h.columns:
            display_cols.append('permno')
        if 'sector' in last_h.columns:
            display_cols.append('sector')
        if 'pred' in last_h.columns:
            display_cols.append('pred')
        if 'y_raw' in last_h.columns:
            display_cols.append('y_raw')
        show_df = last_h[display_cols].copy()
        show_df.columns = ['Permno', 'Sector', 'Predicted', 'Actual Return'][:len(display_cols)]
        if 'Predicted' in show_df.columns:
            show_df['Predicted'] = show_df['Predicted'].round(4)
        if 'Actual Return' in show_df.columns:
            show_df['Actual Return'] = (show_df['Actual Return'] * 100).round(2).astype(str) + '%'
        st.markdown(f"**Holdings — {last_month}**")
        st.dataframe(show_df.reset_index(drop=True), width='stretch', height=380)

    # === Row 4: Rolling Sharpe + IC ===
    st.markdown("---")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        rolling_sr = rets.rolling(12, min_periods=6).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sr.index, y=rolling_sr.values, name="Strategy",
            line=dict(color='#3b82f6', width=2)))

        for i, p in enumerate(st.session_state.pinned):
            p_rets_sr = filter_from(p['result']['monthly_returns'], display_start)
            p_sr = p_rets_sr.rolling(12, min_periods=6).apply(
                lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0)
            fig.add_trace(go.Scatter(
                x=p_sr.index, y=p_sr.values, name=p['label'],
                line=dict(color=PIN_COLORS[i % len(PIN_COLORS)], width=1.5)))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Rolling 12-Month Sharpe Ratio",
            yaxis_title="Sharpe Ratio", template="plotly_dark",
            height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, width='stretch')

    with risk_col2:
        ic = ic_filtered.dropna()
        ic_rolling = ic.rolling(12, min_periods=6).mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ic.index, y=ic.values, name="Monthly IC",
            marker_color='rgba(59,130,246,0.4)'))
        fig.add_trace(go.Scatter(
            x=ic_rolling.index, y=ic_rolling.values,
            name="12m Rolling Mean",
            line=dict(color='#1e3a5f', width=2)))
        fig.add_hline(
            y=ic.mean(), line_dash="dash", line_color="red",
            annotation_text=f"Mean={ic.mean():.3f}")

        fig.update_layout(
            title="Information Coefficient",
            yaxis_title="Spearman IC", template="plotly_dark",
            height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, width='stretch')

    # === Row 5: Monthly Returns Heatmap + Turnover ===
    st.markdown("---")
    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        rets_df = rets.to_frame('ret')
        rets_df['year'] = rets_df.index.str[:4]
        rets_df['month'] = rets_df.index.str[5:7].astype(int)
        heatmap_pivot = rets_df.pivot_table(
            values='ret', index='year', columns='month', aggfunc='first')
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heatmap_pivot.columns = [month_labels[c - 1] for c in heatmap_pivot.columns]

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values * 100,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale=[[0, '#ef4444'], [0.5, '#1f2937'], [1, '#10b981']],
            zmid=0,
            text=np.round(heatmap_pivot.values * 100, 1),
            texttemplate='%{text:.1f}%',
            textfont=dict(size=10),
            hovertemplate='%{y} %{x}: %{z:.1f}%<extra></extra>'))
        fig.update_layout(
            title="Monthly Returns Heatmap (%)",
            template="plotly_dark", height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, width='stretch')

    with diag_col2:
        to = turnover_filtered.dropna()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=to.index, y=to.values, name="Monthly Turnover",
            marker_color='rgba(59,130,246,0.5)'))
        fig.add_hline(
            y=to.mean(), line_dash="dash", line_color="red",
            annotation_text=f"Mean={to.mean():.1%}")
        fig.update_layout(
            title="Monthly Turnover", yaxis_title="Turnover",
            yaxis_tickformat='.0%', template="plotly_dark",
            height=400, margin=dict(t=40, b=40))
        st.plotly_chart(fig, width='stretch')

    # === Row 6: Comparison Table (only when configs are pinned) ===
    if st.session_state.pinned:
        st.markdown("---")
        st.markdown("### Strategy Comparison")

        comp_rows = []
        current_perf = compute_perf(rets, "Current", ic=ic_filtered, turnover=turnover_filtered)
        comp_rows.append(current_perf)

        for p in st.session_state.pinned:
            p_rets_c = filter_from(p['result']['monthly_returns'], display_start)
            p_ic_c = filter_from(p['result']['ic'], display_start)
            p_to_c = filter_from(p['result']['turnover'], display_start)
            p_perf = compute_perf(p_rets_c, p['label'], ic=p_ic_c, turnover=p_to_c)
            comp_rows.append(p_perf)

        comp_df = pd.DataFrame(comp_rows).set_index('Strategy')
        comp_df['SR'] = comp_df['SR'].map('{:.2f}'.format)
        comp_df['Ann Return'] = comp_df['Ann Return'].map('{:.1%}'.format)
        comp_df['Ann Vol'] = comp_df['Ann Vol'].map('{:.1%}'.format)
        comp_df['MDD'] = comp_df['MDD'].map('{:.1%}'.format)
        comp_df['Total Return'] = comp_df['Total Return'].map('{:.0%}'.format)
        comp_df['Mean IC'] = comp_df['Mean IC'].map(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
        comp_df['Mean Turnover'] = comp_df['Mean Turnover'].map(lambda x: f'{x:.1%}' if pd.notna(x) else '-')
        st.dataframe(comp_df, width='stretch')

else:
    st.info("Click **Run Backtest** to see results.")
