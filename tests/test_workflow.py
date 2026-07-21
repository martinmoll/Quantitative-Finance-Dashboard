from components.workflow import config_label

BASE = dict(
    model_name="HGB", construction_method="score_weight", K=10,
    window_type="expanding", retrain_freq=12, vol_tilt=0.05,
    regime_lookback=6, strategy_type="long_only",
    features=["a", "b"], model_params={"max_depth": 3},
)


def test_config_label_shows_key_params():
    lbl = config_label(BASE)
    assert "HGB" in lbl and "score_weight" in lbl and "K=10" in lbl
    assert "12m" in lbl                       # retrain frequency
    assert "vt=0.05" in lbl and "rl=6" in lbl


def test_config_label_differs_on_model_hyperparams():
    # the exact bug: two runs identical except model hyperparameters
    a = config_label(BASE)
    b = config_label(dict(BASE, model_params={"max_depth": 5}))
    assert a != b


def test_config_label_differs_on_retrain_window_and_features():
    assert config_label(BASE) != config_label(dict(BASE, retrain_freq=6))
    assert config_label(BASE) != config_label(
        dict(BASE, window_type="rolling", rolling_window=60))
    assert config_label(BASE) != config_label(dict(BASE, features=["a", "c"]))


def test_config_label_is_feature_order_independent():
    assert config_label(dict(BASE, features=["a", "b"])) == \
        config_label(dict(BASE, features=["b", "a"]))
