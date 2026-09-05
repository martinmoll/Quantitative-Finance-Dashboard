import pandas as pd
import pytest

import pipeline.fetchers.macro as macro_mod


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(macro_mod, "MACRO_CACHE", tmp_path)
    return tmp_path


def _write_cache(path):
    df = pd.DataFrame({"vix": [20.0, 21.0]},
                      index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"]))
    df.to_parquet(path / "fred_data.parquet")
    return df


def test_no_key_uses_cache_when_present(tmp_cache, monkeypatch):
    monkeypatch.setattr(macro_mod, "FRED_API_KEY", "")
    _write_cache(tmp_cache)
    out = macro_mod.fetch_macro(api_key=None)
    assert "vix" in out.columns and len(out) == 2


def test_no_key_and_no_cache_raises(tmp_cache, monkeypatch):
    monkeypatch.setattr(macro_mod, "FRED_API_KEY", "")
    with pytest.raises(ValueError):
        macro_mod.fetch_macro(api_key=None)


def test_fetch_failure_falls_back_to_cache(tmp_cache):
    _write_cache(tmp_cache)

    class _BadFred:
        def __init__(self, api_key=None):
            pass

        def get_series(self, *a, **k):
            raise RuntimeError("network down")

    # patch the class the fetcher imports, so every series download fails
    import fredapi
    fredapi.Fred, orig = _BadFred, fredapi.Fred
    try:
        out = macro_mod.fetch_macro(api_key="dummy-key")
    finally:
        fredapi.Fred = orig
    assert "vix" in out.columns and len(out) == 2


def test_fetch_failure_no_cache_raises_runtime(tmp_cache):
    class _BadFred:
        def __init__(self, api_key=None):
            pass

        def get_series(self, *a, **k):
            raise RuntimeError("network down")

    import fredapi
    fredapi.Fred, orig = _BadFred, fredapi.Fred
    try:
        with pytest.raises(RuntimeError):
            macro_mod.fetch_macro(api_key="dummy-key")
    finally:
        fredapi.Fred = orig


# --- partial-download visibility -------------------------------------------
# The cached frame holds credit_spread, epu and fin_stress but not vix or
# yield_curve_slope, so macro_unc_1m, macro_unc_12m and mom_x_unc have been
# empty in every dataset build. fetch_macro only raised when *every* series
# failed, so the gap never reached the caller.

class _PartialFred:
    """Serves one series and fails the rest, like a partial FRED outage."""

    def __init__(self, api_key=None):
        pass

    def get_series(self, code, **k):
        if code != "BAA10Y":
            raise RuntimeError(f"no data for {code}")
        return pd.Series([1.0, 2.0],
                         index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"]))


def _with_fred(cls, fn):
    import fredapi
    fredapi.Fred, orig = cls, fredapi.Fred
    try:
        return fn()
    finally:
        fredapi.Fred = orig


def test_partial_download_names_the_missing_series(tmp_cache, caplog):
    with caplog.at_level("WARNING"):
        out = _with_fred(_PartialFred,
                         lambda: macro_mod.fetch_macro(api_key="dummy-key"))
    assert list(out.columns) == ["credit_spread"]
    assert "vix" in caplog.text and "yield_curve_slope" in caplog.text


def test_cached_frame_reports_its_own_gaps(tmp_cache, monkeypatch, caplog):
    monkeypatch.setattr(macro_mod, "FRED_API_KEY", "")
    pd.DataFrame({"credit_spread": [1.0]},
                 index=pd.DatetimeIndex(["2020-01-31"])).to_parquet(
        tmp_cache / "fred_data.parquet")
    with caplog.at_level("WARNING"):
        macro_mod.fetch_macro(api_key=None)
    assert "vix" in caplog.text
