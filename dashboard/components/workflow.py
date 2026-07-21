"""Workflow navigation components for the Alpha Strategy Dashboard."""

from __future__ import annotations
import streamlit as st
from typing import Optional
from components.theme import COLORS

STAGES = [
    {
        "key": "data",
        "label": "Data",
        "icon": "1",
        "description": "Load market data",
        "page": "pages/0_Data_Pipeline.py",
        "complete_check": lambda: st.session_state.get("df") is not None,
    },
    {
        "key": "explore",
        "label": "Explore",
        "icon": "2",
        "description": "Explore data & factors",
        "page": "pages/1_Data_Explorer.py",
        "complete_check": lambda: st.session_state.get("df") is not None,
    },
    {
        "key": "model",
        "label": "Build",
        "icon": "3",
        "description": "Configure & run backtest",
        "page": "pages/3_Alpha_Model_Lab.py",
        "complete_check": lambda: st.session_state.get("backtest_result") is not None,
    },
    {
        "key": "results",
        "label": "Results",
        "icon": "4",
        "description": "Review backtest diagnostics",
        "page": "pages/4_Backtest_Results.py",
        "complete_check": lambda: st.session_state.get("backtest_result") is not None,
    },
    {
        "key": "portfolio",
        "label": "Portfolio",
        "icon": "5",
        "description": "Construction & risk analysis",
        "page": "pages/5_Portfolio_Construction.py",
        "complete_check": lambda: (
            st.session_state.get("backtest_result") is not None
            and st.session_state.get("backtest_predictions") is not None
        ),
    },
    {
        "key": "monitor",
        "label": "Monitor",
        "icon": "6",
        "description": "OOD detection & tail risk",
        "page": "pages/6_Monitoring.py",
        "complete_check": lambda: st.session_state.get("backtest_result") is not None,
    },
]

_STAGE_INDEX = {s["key"]: i for i, s in enumerate(STAGES)}

_REQUIRED_KEYS: dict[str, list[str]] = {
    "data": [],
    "explore": ["df"],
    "model": ["df", "market_monthly"],
    "results": ["backtest_result"],
    "portfolio": ["backtest_result", "backtest_predictions"],
    "monitor": ["backtest_result"],
}


def render_workflow_status(current_stage: str) -> None:
    """Render a refined horizontal workflow stepper."""
    C = COLORS
    steps_html = []
    for stage in STAGES:
        is_complete = stage["complete_check"]()
        is_current = stage["key"] == current_stage

        if is_complete and not is_current:
            circle = f"background:{C['positive']};color:{C['canvas']};"
            label_style = f"color:{C['positive']};"
            symbol = "&#10003;"
        elif is_current:
            circle = (
                f"background:{C['primary']};color:#fff;"
                f"box-shadow:0 0 0 4px rgba(79,195,232,0.18);"
            )
            label_style = f"color:{C['primary']};font-weight:600;"
            symbol = stage["icon"]
        else:
            circle = (
                f"background:{C['raised']};color:{C['text_muted']};"
                f"border:1px solid {C['hairline']};"
            )
            label_style = f"color:{C['text_muted']};"
            symbol = stage["icon"]

        steps_html.append(
            f'<div style="text-align:center;flex:1;">'
            f'<div style="width:30px;height:30px;border-radius:50%;{circle}'
            f"display:inline-flex;align-items:center;justify-content:center;"
            f'font-size:13px;font-weight:600;font-family:\'IBM Plex Sans\',sans-serif;">'
            f'{symbol}</div>'
            f'<div style="font-size:11px;margin-top:4px;{label_style}'
            f"font-family:'IBM Plex Sans',sans-serif;\">"
            f"{stage['label']}</div></div>"
        )

    connectors = []
    for i in range(len(STAGES) - 1):
        done_cur = STAGES[i]["complete_check"]()
        done_next = STAGES[i + 1]["complete_check"]()
        is_next_current = STAGES[i + 1]["key"] == current_stage

        if done_cur and (done_next or is_next_current):
            if is_next_current and not done_next:
                bg = f"linear-gradient(90deg,{C['positive']},{C['primary']})"
            else:
                bg = C["positive"]
        else:
            bg = C["raised"]

        connectors.append(
            f'<div style="flex:0.5;height:2px;background:{bg};'
            f'align-self:center;margin-top:-10px;border-radius:1px;"></div>'
        )

    interleaved = []
    for i, step in enumerate(steps_html):
        interleaved.append(step)
        if i < len(connectors):
            interleaved.append(connectors[i])

    html = (
        '<div style="display:flex;align-items:flex-start;'
        'justify-content:space-between;padding:10px 0 14px;'
        f'margin-bottom:8px;">{"".join(interleaved)}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_next_steps(
    current_stage: str, custom_message: Optional[str] = None,
) -> None:
    """Render 'Next Steps' navigation at the bottom of a page."""
    idx = _STAGE_INDEX.get(current_stage)
    if idx is None or idx >= len(STAGES) - 1:
        return

    next_stage = STAGES[idx + 1]
    required = _REQUIRED_KEYS.get(next_stage["key"], [])
    is_ready = all(st.session_state.get(k) is not None for k in required)

    st.markdown("---")
    msg = custom_message or (
        f"Continue to **{next_stage['label']}**: {next_stage['description']}"
    )
    st.markdown(msg)

    if is_ready:
        if st.button(f"Go to {next_stage['label']}  →", type="primary"):
            st.switch_page(next_stage["page"])
    else:
        st.button(
            f"Go to {next_stage['label']}  →",
            disabled=True,
            help="Complete this step first",
        )


def _config_signature(params: dict) -> str:
    """Short hash of the full config, so runs differing only in model
    hyperparameters (or anything else) get a distinct label."""
    import hashlib
    import json
    payload = {k: v for k, v in params.items() if k != "features"}
    payload["features"] = sorted(params.get("features", []) or [])
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:4]


def config_label(params: dict) -> str:
    """Human-readable, unique-per-config label for a pinned run.

    The old label (model + method + K) collided whenever two runs differed only
    in retrain/window/tilts/model hyperparameters — and the label is used as a
    dict key for the config selector and the chart overlays, so collisions made
    all but one run disappear. This adds the result-affecting parameters plus a
    short signature of the full config so distinct runs get distinct labels.
    """
    window = params.get("window_type", "expanding")
    if window == "rolling":
        rw = params.get("rolling_window")
        win = f"roll{rw}" if rw else "roll"
    else:
        win = "exp"
    parts = [
        str(params.get("model_name", "?")),
        str(params.get("construction_method", "?")),
        f"K={params.get('K', '?')}",
        f"{win}/{params.get('retrain_freq', '?')}m",
    ]
    if params.get("strategy_type") == "long_short":
        parts.append("L/S")
    if params.get("vol_tilt"):
        parts.append(f"vt={params['vol_tilt']}")
    if params.get("regime_lookback") is not None:
        parts.append(f"rl={params['regime_lookback']}")
    if params.get("auto_tune"):
        parts.append("tuned")
    return " · ".join(parts) + f" · #{_config_signature(params)}"


def render_empty_state(current_stage: str) -> bool:
    """Render a rich empty state when required data is missing.

    Returns True if dependencies are missing (caller should ``st.stop()``).
    """
    C = COLORS
    required = _REQUIRED_KEYS.get(current_stage, [])
    missing = [k for k in required if st.session_state.get(k) is None]
    if not missing:
        return False

    st.markdown(
        f'<div style="text-align:center;padding:60px 20px;">'
        f'<div style="font-size:48px;margin-bottom:16px;">&#128300;</div>'
        f'<h3 style="color:{C["text_secondary"]};">No Backtest Results Yet</h3>'
        f'<p style="color:{C["text_muted"]};max-width:500px;margin:0 auto 24px;'
        f"font-family:'IBM Plex Sans',sans-serif;\">"
        f"This page requires a completed backtest. Configure your model "
        f"and run a walk-forward backtest in the Alpha Model Lab.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Required data:**")
    for key in required:
        present = st.session_state.get(key) is not None
        icon = ":white_check_mark:" if present else ":x:"
        status = "Available" if present else "Missing"
        st.markdown(f"- {icon} `{key}` — {status}")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Go to Alpha Model Lab  →", type="primary"):
            st.switch_page("pages/3_Alpha_Model_Lab.py")

    return True
