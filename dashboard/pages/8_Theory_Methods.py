# dashboard/pages/8_Theory_Methods.py
"""Page 8: Theory & Methods — standalone educational overview."""

import streamlit as st
from components.theory_content import THEORY_CONTENT
from components.theme import inject_theme, COLORS, FONT_SANS

st.set_page_config(page_title="Theory & Methods", layout="wide")
inject_theme()

C = COLORS
st.markdown(
    f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
    f'color:{C["text"]};margin:0;">Theory & Methods</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    "This page provides a comprehensive overview of the quantitative alpha pipeline, "
    "covering the theories, methodologies, and monitoring approaches used in this dashboard. "
    "Each section corresponds to concepts from the RMBI3110 course at HKUST."
)

st.markdown("---")

sections = [
    ("The Alpha Pipeline", "alpha_pipeline"),
    ("Factor Investing Foundations", "factor_investing_foundations"),
    ("CAPM and Factor Models", "capm_and_factors"),
    ("VIF and Wald Tests", "vif_and_wald"),
    ("Walk-Forward Validation", "walk_forward"),
    ("Regularization: L1, L2, and Trees", "regularization"),
    ("Information Coefficient & Fundamental Law", "ic_and_fundamental_law"),
    ("Feature Importance", "feature_importance"),
    ("Portfolio Construction Methods", "portfolio_construction"),
    ("Turnover and Transaction Costs", "turnover_and_costs"),
    ("VaR, CVaR & Tail Risk", "var_and_tail_risk"),
    ("Distribution Shift & OOD Detection", "distribution_shift"),
    ("Alpha Decay and Retraining", "alpha_decay"),
    ("Glossary", "glossary"),
]

# --- Table of Contents ---
st.sidebar.header("Contents")
for title, key in sections:
    st.sidebar.markdown(f"- [{title}](#{key})")

# --- Render all sections ---
for title, key in sections:
    st.header(title, anchor=key)
    content = THEORY_CONTENT.get(key)
    if content:
        st.markdown(content, unsafe_allow_html=False)
    st.markdown("---")
