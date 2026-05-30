"""Live data pipeline for Alpha Strategy Dashboard."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.config import ensure_cache_dirs, MIN_SUCCESS_RATE

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool
    months_added: list[str] = field(default_factory=list)
    tickers_fetched: int = 0
    tickers_failed: list[str] = field(default_factory=list)
    total_rows: int = 0
    error: str | None = None


def run_pipeline(
    fred_api_key: str | None = None,
    progress_callback=None,
) -> PipelineResult:
    from pipeline.universe import get_sp500_tickers
    from pipeline.fetchers.prices import fetch_prices
    from pipeline.fetchers.fundamentals import fetch_fundamentals
    from pipeline.fetchers.factors import fetch_factors
    from pipeline.fetchers.macro import fetch_macro
    from pipeline.features.price_features import compute_price_features
    from pipeline.features.fundamental_features import compute_fundamental_features
    from pipeline.features.peer_features import compute_peer_features
    from pipeline.features.macro_features import compute_macro_features
    from pipeline.assembler import assemble_month, append_to_dataset

    ensure_cache_dirs()

    def _report(stage, detail=""):
        if progress_callback:
            progress_callback(stage, detail)

    try:
        _report("universe", "Fetching S&P 500 tickers...")
        tickers = get_sp500_tickers()
        _report("universe", f"Got {len(tickers)} tickers")

        _report("prices", "Downloading daily prices...")
        prices = fetch_prices(
            tickers,
            progress_callback=lambda done, total: _report(
                "prices", f"Batch {done}/{total}"
            ),
        )

        _report("prices", "Extracting SPY market returns...")
        if "SPY" in prices.columns.get_level_values(0):
            market_daily = prices["SPY"]["Close"]
        else:
            import yfinance as yf
            spy = yf.download("SPY", period="3y", progress=False)
            market_daily = spy["Close"]

        _report("fundamentals", "Downloading quarterly financials...")
        fund_data = fetch_fundamentals(
            tickers,
            progress_callback=lambda done, total, t: _report(
                "fundamentals", f"{done}/{total} ({t})"
            ),
        )
        failed_tickers = [t for t in tickers if t not in fund_data]

        if len(fund_data) / max(len(tickers), 1) < MIN_SUCCESS_RATE:
            return PipelineResult(
                success=False,
                tickers_failed=failed_tickers,
                error=f"Only {len(fund_data)}/{len(tickers)} tickers succeeded "
                      f"(below {MIN_SUCCESS_RATE:.0%} threshold)",
            )

        _report("factors", "Downloading FF5 + Momentum factors...")
        factors = fetch_factors()

        _report("macro", "Downloading FRED macro data...")
        try:
            macro = fetch_macro(api_key=fred_api_key)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Macro data skipped: {e}")
            macro = pd.DataFrame()

        _report("features", "Computing price features...")
        shares = {}
        sectors = {}
        industries = {}
        for t, data in fund_data.items():
            shares[t] = data.get("shares_outstanding") or 0
            sectors[t] = data.get("sector", "Unknown")
            industries[t] = data.get("industry", "Unknown")

        price_feats = compute_price_features(prices, market_daily, shares)

        _report("features", "Computing fundamental features...")
        fund_input = {}
        for t, data in fund_data.items():
            quarters = []
            if data["income"] is not None and not data["income"].empty:
                quarters = sorted(data["income"].columns)
            if not quarters:
                continue

            fund_input[t] = {
                "quarters": pd.DatetimeIndex(quarters),
                "income": _stmt_to_dict(data["income"], quarters),
                "balance": _stmt_to_dict(data["balance"], quarters) if data["balance"] is not None and not data["balance"].empty else {},
                "cashflow": _stmt_to_dict(data["cashflow"], quarters) if data["cashflow"] is not None and not data["cashflow"].empty else {},
                "market_cap": data.get("market_cap", 0),
            }

        fund_feats = compute_fundamental_features(fund_input) if fund_input else pd.DataFrame()

        if not fund_feats.empty:
            merge_cols = [c for c in fund_feats.columns
                          if c not in price_feats.columns and c not in ["ticker", "ym"]]
            if merge_cols:
                price_feats = price_feats.merge(
                    fund_feats[["ticker", "ym"] + merge_cols],
                    on=["ticker", "ym"],
                    how="left",
                )

        _report("features", "Computing peer features...")
        peer_feats = compute_peer_features(price_feats, sectors, industries)

        target_months = sorted(price_feats["ym"].unique())
        _report("features", "Computing macro features...")
        macro_feats = (
            compute_macro_features(macro, target_months)
            if not macro.empty
            else pd.DataFrame({"ym": target_months})
        )

        _report("assembly", "Assembling dataset...")
        assembled = assemble_month(
            price_features=price_feats,
            fundamental_features=fund_feats,
            peer_features=peer_feats,
            macro_features=macro_feats,
            factors=factors,
        )

        _report("assembly", "Appending to dataset...")
        final = append_to_dataset(assembled)

        months_added = sorted(assembled["ym"].unique())
        return PipelineResult(
            success=True,
            months_added=months_added,
            tickers_fetched=len(fund_data),
            tickers_failed=failed_tickers,
            total_rows=len(final),
        )

    except Exception as e:
        logger.exception("Pipeline failed")
        return PipelineResult(success=False, error=str(e))


def _stmt_to_dict(stmt_df: pd.DataFrame, quarters) -> dict:
    result = {}
    for field in stmt_df.index:
        vals = []
        for q in quarters:
            val = stmt_df.loc[field, q]
            vals.append(float(val) if pd.notna(val) else np.nan)
        result[field] = vals
    return result
