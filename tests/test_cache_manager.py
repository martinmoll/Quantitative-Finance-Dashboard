import pandas as pd
import cache_manager as cache


def _panel(permnos, n_months=3):
    rows = []
    for i in range(n_months):
        for p in permnos:
            rows.append({"ym": f"2020-{i + 1:02d}", "permno": p})
    return pd.DataFrame(rows)


def test_dataset_fingerprint_stable():
    df = _panel([1, 2, 3])
    assert cache.dataset_fingerprint(df) == cache.dataset_fingerprint(df.copy())


def test_dataset_fingerprint_changes_with_universe():
    a = _panel([1, 2, 3])
    b = _panel([1, 2, 4])  # different permno set (different universe)
    assert cache.dataset_fingerprint(a) != cache.dataset_fingerprint(b)


def test_dataset_fingerprint_changes_with_rows():
    a = _panel([1, 2, 3], n_months=3)
    b = _panel([1, 2, 3], n_months=4)  # more history
    assert cache.dataset_fingerprint(a) != cache.dataset_fingerprint(b)


def test_prediction_key_includes_dataset():
    base = dict(model_type="HGB", model_params={"depth": 4}, retrain_every=12,
                feature_cols=["a", "b"], window_type="expanding", auto_tune=False)
    k1 = cache.prediction_key(**base, data_fingerprint="aaaa")
    k2 = cache.prediction_key(**base, data_fingerprint="bbbb")
    assert k1 != k2
