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
