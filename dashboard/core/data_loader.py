"""Dataset and factor data loading utilities.

All functions are pure (no Streamlit imports).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "Data"
FF5_PATH = DATA_DIR / "ff5_factors.csv"


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the alpha dataset."""
    if path is None:
        path = DATA_DIR / "alpha_dataset_v2.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    required = {"ym", "permno", "y_xs", "y_raw", "Mkt_RF", "rf_ff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def compute_market_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly market returns from the panel data."""
    market = df.groupby("ym")[["Mkt_RF", "rf_ff"]].first().sort_index()
    market["spy_ret"] = market["Mkt_RF"] + market["rf_ff"]
    return market


def load_ff5_factors(path: str | Path | None = None) -> pd.DataFrame:
    """Load Fama-French 5-factor monthly data from CSV."""
    if path is None:
        path = FF5_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FF5 factor data not found at {path}. "
            "Run download_ff5_factors() or place ff5_factors.csv in Data/."
        )
    df = pd.read_csv(path, index_col=0)
    df.index.name = "date"
    return df


def download_ff5_factors(save_path: str | Path | None = None) -> pd.DataFrame:
    """Download Fama-French 5-factor data from Ken French's library."""
    import pandas_datareader.data as web

    if save_path is None:
        save_path = FF5_PATH

    raw = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3", "famafrench", start="1963-01-01"
    )
    df = raw[0]
    df = df / 100.0
    df.index = df.index.astype(str).str[:7]
    df.index.name = "date"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    return df
