import numpy as np
import pandas as pd
import pytest
from features import (
    precompute_features,
    FEATURE_GROUPS,
    get_tier_defaults,
    build_features_linear,
    build_features_ensemble,
    _winsorize_xs,
    WINSORIZE_FEATURES,
)


@pytest.fixture
def sample_panel():
    """Create a minimal sample panel DataFrame for testing."""
    np = pytest.importorskip("numpy")
    n_rows = 100
    data = {
        # Core price/return features
        "ret_1_xs": np.random.randn(n_rows),
        "ret_2_12_xs": np.random.randn(n_rows),
        "ret_2_6_xs": np.random.randn(n_rows),
        "ret_13_36_xs": np.random.randn(n_rows),
        "prc_52w_high_xs": np.random.randn(n_rows),

        # Value features
        "bm_xs": np.random.randn(n_rows),
        "ep_xs": np.random.randn(n_rows),
        "cfp_xs": np.random.randn(n_rows),
        "sp_xs": np.random.randn(n_rows),

        # Quality features
        "gpa_xs": np.random.randn(n_rows),
        "roe_xs": np.random.randn(n_rows),
        "roa_xs": np.random.randn(n_rows),
        "earn_quality_xs": np.random.randn(n_rows),
        "cfo_at_xs": np.random.randn(n_rows),

        # Risk/Size features
        "vol_12m_xs": np.random.randn(n_rows),
        "ivol_xs": np.random.randn(n_rows),
        "beta_xs": np.random.randn(n_rows),
        "log_me_xs": np.random.randn(n_rows),

        # Technical features
        "rsi_14_xs": np.random.randn(n_rows),
        "macd_hist_xs": np.random.randn(n_rows),
        "bb_position_xs": np.random.randn(n_rows),
        "roc_3_xs": np.random.randn(n_rows),
        "roc_6_xs": np.random.randn(n_rows),

        # Analyst features
        "revision_xs": np.random.randn(n_rows),
        "revision_ratio_xs": np.random.randn(n_rows),
        "dispersion_xs": np.random.randn(n_rows),
        "beat_xs": np.random.randn(n_rows),
        "rec_chg_xs": np.random.randn(n_rows),
        "n_analysts_xs": np.random.randn(n_rows),

        # Earnings features
        "sue_xs": np.random.randn(n_rows),
        "sue_q_xs": np.random.randn(n_rows),
        "rev_surp_xs": np.random.randn(n_rows),
        "earn_growth_yoy_xs": np.random.randn(n_rows),

        # Peer features
        "peer_sue_xs": np.random.randn(n_rows),
        "peer_revision_xs": np.random.randn(n_rows),

        # Relative features
        "ret_vs_sector_xs": np.random.randn(n_rows),
        "bm_vs_sector_xs": np.random.randn(n_rows),
        "ret_vs_ind_xs": np.random.randn(n_rows),
        "bm_vs_size_xs": np.random.randn(n_rows),

        # Other features
        "turnover_xs": np.random.randn(n_rows),
        "illiq_12m_xs": np.random.randn(n_rows),
        "mom_x_size_xs": np.random.randn(n_rows),
        "val_x_prof_xs": np.random.randn(n_rows),
        "mom_x_vol_xs": np.random.randn(n_rows),
    }
    return pd.DataFrame(data)


def test_feature_groups_is_dict():
    assert isinstance(FEATURE_GROUPS, dict)
    assert "momentum" in FEATURE_GROUPS
    assert "value" in FEATURE_GROUPS
    assert "quality" in FEATURE_GROUPS
    for key, cols in FEATURE_GROUPS.items():
        assert isinstance(cols, list)
        assert len(cols) > 0


def test_get_tier_defaults():
    t1 = get_tier_defaults(1)
    t2 = get_tier_defaults(2)
    assert isinstance(t1, list)
    assert isinstance(t2, list)
    assert len(t2) >= len(t1)


def test_precompute_features_adds_columns(sample_panel):
    original_cols = set(sample_panel.columns)
    result = precompute_features(sample_panel)
    new_cols = set(result.columns) - original_cols
    assert "earnings_composite" in new_cols
    assert "quality_composite" in new_cols
    assert "momentum_composite" in new_cols
    assert "mom_x_quality" in new_cols


def test_winsorized_group_registered():
    assert "interactions_winsorized" in FEATURE_GROUPS
    assert "mom_x_vol_xs_w" in FEATURE_GROUPS["interactions_winsorized"]


def test_tier_defaults_exclude_winsorized():
    # winsorized variants are opt-in, never in the default feature set
    for tier in (1, 2):
        assert not any(f.endswith("_w") for f in get_tier_defaults(tier))


def test_winsorize_xs_clips_per_month():
    ym = pd.Series(["2020-01"] * 100 + ["2020-02"] * 100)
    x = pd.Series(list(np.linspace(-1, 1, 99)) + [100.0]
                  + list(np.linspace(-1, 1, 100)))
    w = _winsorize_xs(x, ym, limit=3.0)
    m1 = x[:100]
    assert w[:100].max() < 100                                  # outlier pulled in
    assert w[:100].max() <= m1.mean() + 3 * m1.std() + 1e-9     # within ±3σ of month 1
    assert np.allclose(w[100:].values, x[100:].values)          # clean month untouched


def test_precompute_adds_winsorized_variants():
    n = 60
    df = pd.DataFrame({
        "ym": ["2020-01"] * 30 + ["2020-02"] * 30,
        "ret_2_12_xs": np.random.randn(n),
        "ivol_xs": np.random.randn(n),
        "mom_x_vol_xs": np.random.randn(n),
    })
    out = precompute_features(df)
    assert "mom_x_vol_xs_w" in out.columns    # raw drift-prone, winsorized
    assert "mom_x_lowivol_w" in out.columns   # engineered interaction, winsorized


def test_backward_compat_linear(sample_panel):
    result = build_features_linear(sample_panel)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_panel)


def test_backward_compat_ensemble(sample_panel):
    result = build_features_ensemble(sample_panel)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_panel)
