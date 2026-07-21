import numpy as np
import pandas as pd
from core.currency import to_nok_unhedged


def test_compounds_return_with_fx():
    rets = pd.Series({"2020-01": 0.10, "2020-02": -0.05})
    fx = pd.Series({"2020-01": 0.02, "2020-02": 0.03})

    nok = to_nok_unhedged(rets, fx)

    # (1+0.10)(1+0.02)-1 = 0.122 ; (1-0.05)(1+0.03)-1 = -0.0215
    np.testing.assert_allclose(nok["2020-01"], 0.122)
    np.testing.assert_allclose(nok["2020-02"], -0.0215)


def test_zero_fx_is_identity():
    rets = pd.Series({"2020-01": 0.10, "2020-02": -0.05})
    fx = pd.Series({"2020-01": 0.0, "2020-02": 0.0})

    nok = to_nok_unhedged(rets, fx)

    pd.testing.assert_series_equal(nok, rets)


def test_missing_fx_month_is_dropped():
    rets = pd.Series({"2020-01": 0.10, "2020-02": -0.05})
    fx = pd.Series({"2020-01": 0.02})  # no Feb observation

    nok = to_nok_unhedged(rets, fx)

    assert list(nok.index) == ["2020-01"]
