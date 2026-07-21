"""Batched OHLCV download from yfinance."""

import time
import logging
import pandas as pd
import yfinance as yf

from pipeline.config import (
    PRICES_CACHE, PRICE_BATCH_SIZE, PRICE_BATCH_DELAY, ensure_cache_dirs,
)

logger = logging.getLogger(__name__)


def fetch_prices(
    tickers: list[str],
    period: str = "3y",
    progress_callback=None,
) -> pd.DataFrame:
    ensure_cache_dirs()
    batches = [
        tickers[i : i + PRICE_BATCH_SIZE]
        for i in range(0, len(tickers), PRICE_BATCH_SIZE)
    ]
    all_frames = []
    total_batches = len(batches)

    for idx, batch in enumerate(batches):
        if progress_callback:
            progress_callback(idx + 1, total_batches)
        try:
            df = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if df is not None and not df.empty:
                all_frames.append(df)
        except Exception as e:
            logger.warning(f"Batch {idx + 1} failed: {e}")

        if idx < total_batches - 1:
            time.sleep(PRICE_BATCH_DELAY)

    if not all_frames:
        raise RuntimeError("All price download batches failed")

    combined = pd.concat(all_frames, axis=1)
    cache_path = PRICES_CACHE / "daily_prices.parquet"
    combined.to_parquet(cache_path)
    return combined


def load_cached_prices() -> pd.DataFrame | None:
    cache_path = PRICES_CACHE / "daily_prices.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None
