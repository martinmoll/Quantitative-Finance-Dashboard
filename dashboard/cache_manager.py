import hashlib
import json
import os
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).parent / '.cache'
PRED_DIR = CACHE_DIR / 'predictions'
PORT_DIR = CACHE_DIR / 'portfolios'


def _ensure_dirs():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PORT_DIR.mkdir(parents=True, exist_ok=True)


def _make_key(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def prediction_key(model_type, model_params, retrain_every, feature_cols=None, window_type="expanding", auto_tune=False):
    return _make_key({
        'model_type': model_type,
        'model_params': model_params,
        'retrain_every': retrain_every,
        'feature_cols': sorted(feature_cols) if feature_cols else None,
        'window_type': window_type,
        'auto_tune': auto_tune,
    })


def portfolio_key(pred_key, K, vol_tilt, regime_lookback, strategy_type="long_only", K_short=10, construction_method="equal_weight", tc_bps=0.0):
    return _make_key({
        'pred_key': pred_key,
        'K': K,
        'vol_tilt': vol_tilt,
        'regime_lookback': regime_lookback,
        'strategy_type': strategy_type,
        'K_short': K_short,
        'construction_method': construction_method,
        'tc_bps': tc_bps,
    })


def get_predictions(key):
    _ensure_dirs()
    path = PRED_DIR / f'{key}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_predictions(key, predictions):
    _ensure_dirs()
    path = PRED_DIR / f'{key}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(predictions, f)


def get_portfolio(key):
    _ensure_dirs()
    path = PORT_DIR / f'{key}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_portfolio(key, portfolio):
    _ensure_dirs()
    path = PORT_DIR / f'{key}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(portfolio, f)
