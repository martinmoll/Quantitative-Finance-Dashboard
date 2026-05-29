"""Expandable theory sections for educational content on each page."""

import streamlit as st
from components.theory_content import THEORY_CONTENT


def theory_section(title: str, content_key: str):
    content = THEORY_CONTENT.get(content_key)
    if content is None:
        return
    with st.expander(f"Theory: {title}", expanded=False):
        st.markdown(content, unsafe_allow_html=False)
