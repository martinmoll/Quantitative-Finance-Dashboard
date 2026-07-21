"""Dataset and factor data loading utilities.

All functions are pure (no Streamlit imports).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "Data"
FF5_PATH = DATA_DIR / "ff5_factors.csv"

# Regional datasets the dashboard can load. US uses Ken French FF5; other regions
# have no FF5 (CAPM only), so ff5 is not loaded for them — see app.py.
DATASET_FILES = {
    "US (S&P 500 / Nasdaq)": ("alpha_dataset_v2", "US"),
    "Norway (Oslo Børs)": ("alpha_dataset_no", "NO"),
}


def available_datasets() -> dict:
    """Map each dataset label to its existing file path (parquet preferred)."""
    out = {}
    for label, (base, _region) in DATASET_FILES.items():
        parquet = DATA_DIR / f"{base}.parquet"
        csv = DATA_DIR / f"{base}.csv"
        if parquet.exists():
            out[label] = parquet
        elif csv.exists():
            out[label] = csv
    return out


def region_for_label(label: str) -> str:
    return DATASET_FILES.get(label, (None, "US"))[1]


def label_for_region(region: str) -> str:
    for label, (_base, r) in DATASET_FILES.items():
        if r == region:
            return label
    return "US (S&P 500 / Nasdaq)"


def generate_dataset(progress_callback=None) -> pd.DataFrame:
    """Run the pipeline to generate the dataset from scratch."""
    import os
    import sys
    sys.path.insert(0, str(DATA_DIR.parent))
    from pipeline import run_pipeline

    fred_key = os.environ.get("FRED_API_KEY", "")
    result = run_pipeline(
        fred_api_key=fred_key or None,
        progress_callback=progress_callback,
    )
    if not result.success:
        raise RuntimeError(
            f"Pipeline failed to generate dataset: {result.error}"
        )
    return load_dataset()


def load_dataset(path: str | Path | None = None, auto_generate: bool = False,
                 progress_callback=None) -> pd.DataFrame:
    """Load and validate the alpha dataset. Tries Parquet first, then CSV.

    If auto_generate is True and no dataset file exists, runs the pipeline
    to fetch data and build the dataset from scratch.
    """
    if path is None:
        parquet_path = DATA_DIR / "alpha_dataset_v2.parquet"
        csv_path = DATA_DIR / "alpha_dataset_v2.csv"
        if parquet_path.exists():
            path = parquet_path
        elif csv_path.exists():
            path = csv_path
        elif auto_generate:
            return generate_dataset(progress_callback=progress_callback)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}. "
                "Run with auto_generate=True or execute the pipeline first."
            )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
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


def permno_ticker_map(df: pd.DataFrame) -> dict:
    """Map each permno to its ticker symbol for display.

    Returns an empty dict if the dataset has no ticker column (e.g. the
    legacy course dataset), so callers fall back to showing permno.
    """
    if "ticker" not in df.columns:
        return {}
    return df.groupby("permno")["ticker"].last().to_dict()


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
    import io
    import zipfile
    import urllib.request

    if save_path is None:
        save_path = FF5_PATH

    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8")

    lines = raw.strip().split("\n")
    header_idx = next(i for i, line in enumerate(lines) if "Mkt-RF" in line)
    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = df["date"].astype(str).str.strip()
    df = df[df["date"].str.len() == 6]
    df["date"] = df["date"].str[:4] + "-" + df["date"].str[4:6]
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df.index.name = "date"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    return df
