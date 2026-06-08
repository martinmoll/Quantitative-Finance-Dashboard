import numpy as np
import pandas as pd
from core.models import (
    get_model,
    get_default_params,
    get_param_ranges,
    get_feature_tier,
    list_models,
)


def test_list_models():
    models = list_models()
    assert set(models) == {"HGB", "RF", "Lasso", "Ridge", "ElasticNet", "FamaMacBeth", "Ensemble"}


def test_get_feature_tier():
    assert get_feature_tier("HGB") == 2
    assert get_feature_tier("RF") == 2
    assert get_feature_tier("Lasso") == 1
    assert get_feature_tier("Ridge") == 1
    assert get_feature_tier("ElasticNet") == 1
    assert get_feature_tier("FamaMacBeth") == 1
    assert get_feature_tier("Ensemble") == 2


def test_get_default_params():
    for name in list_models():
        params = get_default_params(name)
        assert isinstance(params, dict)


def test_get_param_ranges():
    for name in list_models():
        ranges = get_param_ranges(name)
        assert isinstance(ranges, dict)
        for key, spec in ranges.items():
            assert "min" in spec
            assert "max" in spec
            assert "default" in spec


def test_model_fit_predict():
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    y_train = pd.Series(np.random.randn(200))
    X_test = pd.DataFrame(np.random.randn(50, 5), columns=[f"f{i}" for i in range(5)])

    for name in list_models():
        model = get_model(name, get_default_params(name))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == 50
        assert not np.any(np.isnan(preds))


def test_model_feature_importance():
    np.random.seed(42)
    cols = [f"f{i}" for i in range(5)]
    X = pd.DataFrame(np.random.randn(200, 5), columns=cols)
    y = pd.Series(np.random.randn(200))

    tree_model = get_model("HGB", get_default_params("HGB"))
    tree_model.fit(X, y)
    imp = tree_model.get_feature_importance()
    assert imp is not None
    assert len(imp) == 5

    linear_model = get_model("Lasso", get_default_params("Lasso"))
    linear_model.fit(X, y)
    imp = linear_model.get_feature_importance()
    assert imp is not None
    assert len(imp) == 5
