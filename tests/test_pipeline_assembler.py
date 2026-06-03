"""Tests for pipeline assembler module."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.assembler import (
    cross_sectional_standardize,
    fill_red_features,
    validate_schema,
)


@pytest.fixture
def sample_raw_features():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)] * 2,
        "ym": ["2025-06"] * n + ["2025-07"] * n,
        "ret_1": np.random.randn(n * 2) * 0.05,
        "bm": np.random.uniform(0.2, 2.0, n * 2),
        "vol_12m": np.random.uniform(0.1, 0.6, n * 2),
    })


def test_cross_sectional_standardize(sample_raw_features):
    cols = ["ret_1", "bm", "vol_12m"]
    result = cross_sectional_standardize(sample_raw_features, cols)
    for col in cols:
        xs_col = f"{col}_xs"
        assert xs_col in result.columns
        for ym in result["ym"].unique():
            month_vals = result.loc[result["ym"] == ym, xs_col].dropna()
            if len(month_vals) > 1:
                assert abs(month_vals.mean()) < 0.01
                assert abs(month_vals.std() - 1.0) < 0.15


def test_fill_red_features(sample_raw_features):
    result = fill_red_features(sample_raw_features)
    assert "iv_atm_30d" in result.columns
    assert "vrp" in result.columns
    assert result["iv_atm_30d"].isna().all()
    assert result["vrp"].isna().all()


def test_validate_schema_identifies_missing():
    cols = ["permno", "ym", "ret_1"]
    df = pd.DataFrame(columns=cols)
    missing = validate_schema(df)
    assert "y_xs" in missing
    assert "y_raw" in missing
    assert "Mkt_RF" in missing
    assert "rf_ff" in missing


def test_validate_schema_passes():
    cols = ["permno", "ym", "y_xs", "y_raw", "Mkt_RF", "rf_ff"]
    df = pd.DataFrame(columns=cols)
    missing = validate_schema(df)
    assert len(missing) == 0
