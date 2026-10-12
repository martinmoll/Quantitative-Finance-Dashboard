"""Behaviour tests for the cost-adjusted Sharpe panel on the Backtest Results page."""

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from core.portfolio import net_returns_at_cost

PAGE = str(Path(__file__).parent.parent / "dashboard" / "pages" / "4_Backtest_Results.py")
MONTHS = [f"{y}-{m:02d}" for y in range(2015, 2020) for m in range(1, 13)]


def _result(gross, cost_bps):
    turnover = pd.Series(0.4, index=MONTHS)
    turnover.iloc[0] = np.nan
    return {
        "monthly_returns": net_returns_at_cost(gross, turnover, cost_bps),
        "monthly_returns_gross": gross,
        "turnover": turnover,
        "ic": pd.Series(0.02, index=MONTHS),
        "holdings": {},
        "cost_bps": cost_bps,
    }


def _page(result, pinned=()):
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.session_state["backtest_result"] = result
    at.session_state["backtest_params"] = {"model_name": "HGB", "K": 10}
    at.session_state["pinned_configs"] = list(pinned)
    return at.run()


def _cost_panel_html(at):
    return next(m.value for m in at.markdown if "Break-even cost" in m.value)


def test_cost_level_follows_the_viewed_config():
    """A pinned run at 15 bps must open the panel at 15 bps, not at the current run's 10."""
    gross = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, len(MONTHS)), index=MONTHS)
    pin = {"label": "Pin 15 bps", "result": _result(gross, 15.0), "params": {}}
    at = _page(_result(gross, 10.0), pinned=[pin])
    assert at.button_group[0].value == 10.0

    at.selectbox[0].set_value(1).run()

    assert not at.exception, at.exception
    assert at.button_group[0].value == 15.0


def test_cost_panel_shows_na_for_undefined_sharpe():
    """Zero volatility makes the Sharpe undefined. The panel must say N/A, not nan."""
    gross = pd.Series(0.0, index=MONTHS)
    at = _page(_result(gross, 10.0))

    assert not at.exception, at.exception
    assert "nan" not in _cost_panel_html(at)


def test_sharpe_banner_renders_bold_not_literal_asterisks():
    """Interpretation text marks bold with **. The banner is raw HTML, where
    Markdown is not parsed, so the markers must become <b> tags."""
    gross = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, len(MONTHS)), index=MONTHS)
    at = _page(_result(gross, 10.0))

    assert not at.exception, at.exception
    banner = next(m.value for m in at.markdown if "Sharpe Ratio =" in m.value)
    assert "**" not in banner
    assert "<b>Sharpe Ratio =" in banner


def test_compare_table_is_html_with_formatted_values():
    """The dataframe grid draws on a canvas that page CSS cannot style, so the
    Ledger page shows the Compare table as an HTML table. Formats must survive."""
    gross = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, len(MONTHS)), index=MONTHS)
    pin = {"label": "Pin 15 bps", "result": _result(gross, 15.0), "params": {}}
    at = _page(_result(gross, 10.0), pinned=[pin])

    assert not at.exception, at.exception
    assert len(at.dataframe) == 0
    compare = at.table[0].value
    assert list(compare.index) == ["Current Run", "Pin 15 bps"]
    assert compare.loc["Current Run", "Ann Return"].endswith("%")


def test_empty_state_keeps_the_navy_theme():
    """The shared empty state has inline navy-theme colors, which measure 1.6:1
    on the Ledger canvas. Without a backtest the page must stay navy."""
    at = AppTest.from_file(PAGE, default_timeout=120).run()

    assert not at.exception, at.exception
    css = " ".join(m.value for m in at.markdown)
    assert "No Backtest Results Yet" in css
    assert "background-color: #0B375D" in css
    assert "Instrument+Sans" not in css
