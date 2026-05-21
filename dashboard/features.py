import pandas as pd
import numpy as np


def _get_existing(df_slice, cols):
    """Return DataFrame of only the columns that actually exist.
    Avoids zero-padding missing columns, which would dilute composite averages.
    """
    return df_slice[[c for c in cols if c in df_slice.columns]]


def _build_engineered_features(df_slice):
    """Build all engineered features (shared by both tiers)."""
    feat = pd.DataFrame(index=df_slice.index)

    # For individual features (not composites), missing columns default to 0.0
    def _col(name):
        return df_slice[name] if name in df_slice.columns else pd.Series(0.0, index=df_slice.index)

    # --- 1. Interaction terms ---
    # Factor premia are not independent since they interact in economically meaningful ways.
    # A linear model cannot capture these interactions, so they are constructed explicitly.
    # For HGB these are redundant since the trees learn interactions natively, but they still
    # help by making the signal more accessible at shallow tree depths.


    # High-quality firms with high profitability have their momentum driven more by fundamentals
    # than sentiment, making their momentum more persistent and less prone to a reversal.
    feat['mom_x_quality'] = _col('ret_2_12_xs') * _col('gpa_xs')
    feat['mom_x_roe'] = _col('ret_2_12_xs') * _col('roe_xs')

    # Cheap stocks can be cheap for good reason such as financial distres. High-vol value stocks
    # often reflect distress risk rather than genuine undervaluation. Interacting value with
    # negative volatility isolates the "safe value" premium, attempting at least to isolate cheap firms that are NOT risky.
    feat['val_x_lowvol'] = _col('bm_xs') * (-_col('vol_12m_xs'))
    feat['ep_x_lowvol'] = _col('ep_xs') * (-_col('vol_12m_xs'))

    # Earnings surprises (SUE) are more informative when analysts broadly agree on expectations.
    # High dispersion means analysts disagree on what earnings should be, so a "surprise" relative
    # to consensus may just reflect a noisy consensus rather than genuine new information.
    feat['sue_x_lowdisp'] = _col('sue_xs') * (-_col('dispersion_xs'))

    # When both surprise and revision point the same direction, the earnings signal is
    # twice confirmed. The company beat expectations AND analysts are revising upward.
    feat['sue_x_revision'] = _col('sue_xs') * _col('revision_xs')

    # Momentum in low-ivol stocks is more likely to reflect genuine
    # information diffusion rather than speculative overshooting that will reverse.
    feat['mom_x_lowivol'] = _col('ret_2_12_xs') * (-_col('ivol_xs'))

    # Classic "quality at a reasonable price": buying cheap stocks that are also profitable
    # avoids the value trap — firms that screen as cheap but are actually deteriorating.
    # Novy-Marx (2013) showed profitability and value are complementary signals.
    feat['bm_x_roe'] = _col('bm_xs') * _col('roe_xs')
    feat['ep_x_gpa'] = _col('ep_xs') * _col('gpa_xs')

    # --- 2. Non-linear transforms ---
    # Lasso is restricted to linear relationships, but factor premia often exhibit
    # diminishing or accelerating effects. For example, moderate momentum is predictive
    # but extreme momentum may indicate overreaction. Signed squares (x^2 * sign(x))
    # preserve direction while allowing the linear model to capture curvature.
    for col in ['ret_2_12_xs', 'ret_1_xs', 'sue_xs', 'bm_xs', 'revision_xs']:
        raw = _col(col)
        feat[f'{col}_sq'] = raw ** 2 * np.sign(raw)

    # --- 3. Composite signals ---
    # Individual features are noisy month-to-month. Averaging multiple related measures
    # that capture the same underlying economic concept reduces idiosyncratic measurement
    # error. Only columns that actually exist are averaged — including missing columns
    # as zeros would dilute the signal toward the cross-sectional mean.

    # Earnings momentum: multiple views of the same question — "is this company
    # delivering better-than-expected results?" Each measure captures it differently
    # (quarterly surprise, annual growth, analyst beats), so averaging reduces noise.
    earn_cols = ['sue_xs', 'sue_q_xs', 'rev_surp_xs', 'earn_growth_yoy_xs', 'beat_xs']
    feat['earnings_composite'] = _get_existing(df_slice, earn_cols).mean(axis=1)

    # Profitability quality: firms with high and sustainable profitability across
    # multiple accounting measures tend to outperform. Using multiple measures guards
    # against any single metric being manipulated or sector-biased.
    qual_cols = ['gpa_xs', 'roe_xs', 'roa_xs', 'earn_quality_xs', 'cfo_at_xs']
    feat['quality_composite'] = _get_existing(df_slice, qual_cols).mean(axis=1)

    # Value: different valuation ratios have different strengths — book-to-market
    # works well for asset-heavy firms, earnings yield for profitable firms, cash
    # flow yield for capital-intensive firms. Averaging smooths across these biases.
    val_cols = ['bm_xs', 'ep_xs', 'cfp_xs', 'sp_xs']
    feat['value_composite'] = _get_existing(df_slice, val_cols).mean(axis=1)

    # Momentum at multiple horizons: 2-12 month (classic), 2-6 month (intermediate),
    # 13-36 month (long-run), and proximity to 52-week high (George & Hwang 2004).
    # Combining horizons captures both intermediate trend and longer-term drift.
    mom_cols = ['ret_2_12_xs', 'ret_2_6_xs', 'ret_13_36_xs', 'prc_52w_high_xs']
    feat['momentum_composite'] = _get_existing(df_slice, mom_cols).mean(axis=1)

    # Technical indicators: individually noisy, but when multiple indicators agree
    # (RSI, MACD, Bollinger band position all pointing the same direction),
    # the signal is more likely to reflect genuine price momentum than noise.
    tech_cols = ['rsi_14_xs', 'macd_hist_xs', 'bb_position_xs', 'roc_3_xs', 'roc_6_xs']
    feat['technical_composite'] = _get_existing(df_slice, tech_cols).mean(axis=1)

    # Analyst sentiment: revision activity, direction, and coverage breadth.
    # Analysts are slow to update (anchoring bias), so revisions contain
    # predictive information beyond the level of current estimates.
    analyst_cols = ['revision_xs', 'revision_ratio_xs', 'rec_chg_xs', 'n_analysts_xs']
    feat['analyst_composite'] = _get_existing(df_slice, analyst_cols).mean(axis=1)

    # --- 4. Relative / differential signals ---
    # Subtracting the peer/industry average isolates the firm-specific component.
    # If the entire industry had a positive earnings surprise, a firm's SUE is less
    # informative — the surprise may reflect an industry-wide tailwind, not firm alpha.
    feat['sue_vs_peer'] = _col('sue_xs') - _col('peer_sue_xs')
    feat['revision_vs_peer'] = _col('revision_xs') - _col('peer_revision_xs')

    # Long-term momentum minus short-term return. Short-term returns (1 month)
    # exhibit reversal (Jegadeesh 1990), while 2-12 month returns exhibit continuation.
    # Subtracting the reversal component from momentum isolates the "pure" trend signal.
    feat['reversal_mom_combo'] = _col('ret_2_12_xs') - _col('ret_1_xs')

    # --- 5. Composite interactions ---
    # Interactions between composite signals — these capture higher-order relationships
    # like "firms with improving earnings AND positive price momentum" (earnings-momentum
    # confirmation) or "cheap firms that are also high-quality" (quality value).
    feat['earn_x_mom'] = feat['earnings_composite'] * feat['momentum_composite']
    feat['quality_x_value'] = feat['quality_composite'] * feat['value_composite']
    feat['earn_x_lowvol'] = feat['earnings_composite'] * (-_col('vol_12m_xs'))

    return feat.fillna(0.0)


def build_features_linear(df_slice):
    """Tier 1: Conservative feature set for Lasso."""
    feat = pd.DataFrame(index=df_slice.index)

    core = [
        'ret_1_xs', 'ret_2_12_xs', 'ret_2_6_xs',
        'bm_xs', 'ep_xs', 'cfp_xs', 'sp_xs',
        'gpa_xs', 'roe_xs', 'roa_xs',
        'vol_12m_xs', 'ivol_xs', 'beta_xs',
        'log_me_xs',
        'sue_xs', 'revision_xs', 'beat_xs',
        'turnover_xs', 'illiq_12m_xs',
        'mom_x_size_xs', 'val_x_prof_xs', 'mom_x_vol_xs',
        'ret_vs_sector_xs', 'bm_vs_sector_xs', 'ret_vs_ind_xs',
        'bm_vs_size_xs',
    ]
    for c in core:
        if c in df_slice.columns:
            feat[c] = df_slice[c]

    engineered = _build_engineered_features(df_slice)
    feat = pd.concat([feat, engineered], axis=1)
    return feat.fillna(0.0)


def build_features_ensemble(df_slice):
    """Tier 2: All raw _xs features + engineered features for HGB."""
    xs_cols = [c for c in df_slice.columns if c.endswith('_xs') and c != 'y_xs']
    feat = df_slice[xs_cols].copy()

    engineered = _build_engineered_features(df_slice)
    feat = pd.concat([feat, engineered], axis=1)
    return feat.fillna(0.0)


# ---------------------------------------------------------------------------
# Feature metadata and precomputation (added for multipage dashboard)
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "momentum": [
        "ret_1_xs", "ret_2_12_xs", "ret_2_6_xs", "ret_13_36_xs",
        "prc_52w_high_xs", "momentum_composite", "reversal_mom_combo",
    ],
    "value": [
        "bm_xs", "ep_xs", "cfp_xs", "sp_xs", "value_composite",
    ],
    "quality": [
        "gpa_xs", "roe_xs", "roa_xs", "earn_quality_xs", "cfo_at_xs",
        "quality_composite",
    ],
    "size": ["log_me_xs"],
    "volatility": [
        "vol_12m_xs", "ivol_xs", "beta_xs",
    ],
    "technical": [
        "rsi_14_xs", "macd_hist_xs", "bb_position_xs", "roc_3_xs", "roc_6_xs",
        "technical_composite",
    ],
    "analyst": [
        "revision_xs", "dispersion_xs", "beat_xs",
        "revision_ratio_xs", "rec_chg_xs", "n_analysts_xs",
        "analyst_composite",
    ],
    "earnings": [
        "sue_xs", "sue_q_xs", "rev_surp_xs", "earn_growth_yoy_xs",
        "earnings_composite",
    ],
    "interactions": [
        "mom_x_quality", "mom_x_roe", "val_x_lowvol", "ep_x_lowvol",
        "sue_x_lowdisp", "sue_x_revision", "mom_x_lowivol", "bm_x_roe",
        "ep_x_gpa", "earn_x_mom", "quality_x_value", "earn_x_lowvol",
        "mom_x_size_xs", "val_x_prof_xs", "mom_x_vol_xs",
    ],
    "relative": [
        "sue_vs_peer", "revision_vs_peer",
        "ret_vs_sector_xs", "bm_vs_sector_xs", "ret_vs_ind_xs", "bm_vs_size_xs",
    ],
    "nonlinear": [
        "ret_2_12_xs_sq", "ret_1_xs_sq", "sue_xs_sq", "bm_xs_sq", "revision_xs_sq",
    ],
}


def get_tier_defaults(tier: int) -> list[str]:
    """Return the default feature column list for a given tier.

    Tier 1 (~52 features): conservative set for linear models.
    Tier 2 (~118+ features): all _xs columns + engineered for tree models.
    """
    if tier == 1:
        core = [
            "ret_1_xs", "ret_2_12_xs", "ret_2_6_xs",
            "bm_xs", "ep_xs", "cfp_xs", "sp_xs",
            "gpa_xs", "roe_xs", "roa_xs",
            "vol_12m_xs", "ivol_xs", "beta_xs",
            "log_me_xs",
            "sue_xs", "revision_xs", "beat_xs",
            "turnover_xs", "illiq_12m_xs",
            "mom_x_size_xs", "val_x_prof_xs", "mom_x_vol_xs",
            "ret_vs_sector_xs", "bm_vs_sector_xs", "ret_vs_ind_xs",
            "bm_vs_size_xs",
        ]
        engineered = []
        for group in FEATURE_GROUPS.values():
            for f in group:
                if not f.endswith("_xs"):
                    engineered.append(f)
        return core + sorted(set(engineered))

    # Tier 2: all _xs columns + all engineered
    all_features = []
    for group in FEATURE_GROUPS.values():
        all_features.extend(group)
    return sorted(set(all_features))


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered feature columns to the DataFrame.

    Call once after loading the dataset. Subsequent operations can select
    features by column name without re-computing.
    """
    result = df.copy()
    engineered = _build_engineered_features(df)
    for col in engineered.columns:
        result[col] = engineered[col]
    return result
