"""Pipeline configuration: paths, constants, column schema."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
CACHE_DIR = Path(__file__).parent / "cache"
PRICES_CACHE = CACHE_DIR / "prices"
FUNDAMENTALS_CACHE = CACHE_DIR / "fundamentals"
FACTORS_CACHE = CACHE_DIR / "factors"
MACRO_CACHE = CACHE_DIR / "macro"

PARQUET_PATH = DATA_DIR / "alpha_dataset_v2.parquet"
CSV_PATH = DATA_DIR / "alpha_dataset_v2.csv"

# Per-region settings. US uses Ken French FF5 + SPY; other regions build a
# single-factor CAPM frame against a local market proxy (no regional FF5 exists).
REGION_CONFIG = {
    "US": {
        "market_ticker": "SPY",
        "parquet": PARQUET_PATH,
        "csv": CSV_PATH,
        "factor_source": "kenfrench",
    },
    "NO": {
        "market_ticker": "^OSEAX",       # Oslo Børs All-Share
        "parquet": DATA_DIR / "alpha_dataset_no.parquet",
        "csv": DATA_DIR / "alpha_dataset_no.csv",
        "factor_source": "capm",
        "rf_fred_series": "IR3TIB01NOM156N",  # Norway 3-month interbank rate
    },
}


def region_config(region: str = "US") -> dict:
    return REGION_CONFIG.get(region, REGION_CONFIG["US"])


def dataset_paths(region: str = "US"):
    cfg = region_config(region)
    return cfg["parquet"], cfg["csv"]

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

PRICE_BATCH_SIZE = 50
PRICE_BATCH_DELAY = 2.0
FUNDAMENTALS_DELAY = 0.5
FUNDAMENTALS_RETRIES = 3
MIN_SUCCESS_RATE = 0.5

FRED_SERIES = {
    "VIXCLS": "vix",
    "T10Y2Y": "yield_curve_slope",
    "BAA10Y": "credit_spread",
    "USEPUINDXM": "epu",
    "STLFSI4": "fin_stress",
}

METADATA_COLS = [
    "permno", "ym", "date", "in_sp500", "exchcd", "siccd", "sector",
    "me", "prc_abs",
]

RAW_FEATURE_COLS = [
    "ret_1", "ret_2_12", "ret_2_6", "ret_13_36", "vol_12m", "max_ret_12m",
    "beta", "ivol", "turnover", "log_me", "prc_52w_high", "age",
    "bm", "ep", "cfp", "sp", "ag", "gpa", "roe", "roa", "acc", "nsi",
    "lev", "cash_at", "sgr", "ato", "dp_ratio",
    "sue", "beat", "n_analysts", "revision", "dispersion", "revision_ratio",
    "rsi_14", "macd_hist", "bb_position", "prc_ma6", "prc_ma12", "prc_ma24",
    "roc_3", "roc_6", "vol_ratio", "skew_12m", "kurt_12m", "illiq_12m",
    "sue_chg",
    "iv_atm_30d", "iv_atm_91d", "iv_skew", "pc_vol_ratio", "pc_oi_ratio", "vrp",
    "ret_vs_sector", "bm_vs_sector", "mom_x_size", "val_x_prof", "mom_x_vol",
    "peer_sue", "peer_revision", "size_peer_ret", "val_peer_ret",
    "leader_ret_lag1", "peer_ret_1",
    "sue_q", "rev_surp", "rev_growth_qq", "earn_growth_yoy",
    "gm_q", "gm_chg", "op_margin_q", "op_margin_chg",
    "sga_chg", "acc_q", "roe_q", "roe_chg", "inv_chg", "rec_chg",
    "rd_intensity", "cfo_at", "earn_quality", "oi_growth_yoy", "ato_q",
    "ret_vs_ind", "bm_vs_size",
    "mom_x_unc", "val_x_finunc", "beta_x_disp",
    "mom_accel", "delta_vol", "delta_bm", "ret_streak", "mom_of_mom",
    "sector_ret_avg", "sector_ret_dispersion", "mkt_ret_dispersion",
    "sector_mom_lag1",
    "iv_term_structure", "sector_iv", "sector_vrp",
    "leader_ret", "sector_rel_mom",
    "ind_mom", "ind_dispersion", "ind_crowding", "ind_sue",
    "size_grp_mom", "size_grp_disp", "beta_grp_ret", "val_grp_mom",
    "vol_grp_ret", "ind_size_ret", "ind_size_mom",
    "macro_unc_1m", "macro_unc_12m", "fin_unc_1m", "fin_unc_12m",
]

RED_FEATURES = [
    "iv_atm_30d", "iv_atm_91d", "iv_skew", "pc_vol_ratio", "pc_oi_ratio",
    "vrp", "iv_term_structure", "sector_iv", "sector_vrp",
    "beat", "n_analysts", "revision", "dispersion", "revision_ratio",
    "rev_surp",
    "peer_revision", "ind_crowding",
]

TARGET_COLS = ["y_raw", "y_xs", "ret_excess", "ret_adj"]
FACTOR_COLS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom", "rf_ff", "spy_ret"]

_NO_XS_SUFFIX = {
    "sector_ret_avg", "sector_ret_dispersion", "mkt_ret_dispersion",
    "sector_mom_lag1", "iv_term_structure", "sector_iv", "sector_vrp",
    "leader_ret", "sector_rel_mom", "ind_mom", "ind_dispersion",
    "ind_crowding", "ind_sue", "size_grp_mom", "size_grp_disp",
    "beta_grp_ret", "val_grp_mom", "vol_grp_ret", "ind_size_ret",
    "ind_size_mom", "macro_unc_1m", "macro_unc_12m", "fin_unc_1m",
    "fin_unc_12m",
}

XS_FEATURE_COLS = [c for c in RAW_FEATURE_COLS if c not in _NO_XS_SUFFIX]


def ensure_cache_dirs():
    for d in [PRICES_CACHE, FUNDAMENTALS_CACHE, FACTORS_CACHE, MACRO_CACHE]:
        d.mkdir(parents=True, exist_ok=True)
