# Alpha Strategy Dashboard

Interactive Streamlit dashboard for ML-driven equity strategy backtesting. Supports walk-forward backtesting with 6 alpha models, 5 portfolio construction methods, factor analysis, and risk diagnostics.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
```

Optional: set a FRED API key for macro features (VIX, credit spread, EPU, etc.):
```bash
export FRED_API_KEY=your_key_here        # Linux/Mac
set FRED_API_KEY=your_key_here           # Windows
```

## Run

```bash
python -m streamlit run dashboard/app.py
```

On first run, the app auto-fetches S&P 500 data via yfinance and builds the dataset. This takes a few minutes.

## Tests

```bash
python -m pytest tests/
```

## Project Structure

```
dashboard/
  app.py                  # Streamlit entry point
  features.py             # Feature engineering (Tier 1 / Tier 2)
  core/
    models.py             # 6 alpha models (HGB, RF, Lasso, Ridge, ElasticNet, Fama-MacBeth)
    backtest.py           # Walk-forward prediction engine
    portfolio.py          # Portfolio construction (equal-weight, score, inv-vol, ERC, MVO)
    diagnostics.py        # IC analysis, Fundamental Law, KS test, alpha decay
    risk.py               # VaR/CVaR, factor exposure, risk contribution
    factor_models.py      # CAPM / FF3 / FF5 regressions, rolling beta
    data_loader.py        # Dataset loading and generation
  components/
    charts.py             # Plotly chart builders
    theory.py, metrics.py # UI components
  pages/                  # Streamlit multipage views (data, factors, backtest, portfolio, risk)
pipeline/
  __init__.py             # Pipeline orchestrator (yfinance → features → dataset)
  config.py               # Paths, column schema, constants
  fetchers/               # Price, fundamental, factor, macro data fetchers
  features/               # Price, fundamental, peer, macro feature computation
  assembler.py            # Monthly assembly and dataset append
tests/                    # pytest suite
```
