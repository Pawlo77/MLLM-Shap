"""Main entry point for the mllm_shapx Streamlit dashboard.

Run with:
    cd experiments/mllm_shapx/gui && streamlit run app.py
Or:
    make monitor  (from experiments/mllm_shapx/)
"""

import streamlit as st

from components import render_sidebar
from views import comparison, run_detail, run_monitor, runs_overview

_TABS: list[tuple[str, callable]] = [
    ("⚡ Monitor", run_monitor.render),
    ("📋 Runs Overview", runs_overview.render),
    ("🔍 Run Detail", run_detail.render),
    ("⚖️ Comparison", comparison.render),
]


def main() -> None:
    """Main function to render the Streamlit app."""
    st.set_page_config(
        page_title="MLflow Dashboard",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar()

    st.markdown(
        """
        <style>
            .block-container { padding-top: 3.5rem; }
            [data-testid="stAppViewBlockContainer"] { padding-top: 3.5rem; }
            header[data-testid="stHeader"] {
                background: #0E1117 !important;
            }
            [role="tab"] {
                color: #FAFAFA !important;
                font-size: 1rem !important;
                font-weight: 500 !important;
            }
            [role="tab"][aria-selected="true"] {
                color: #6C63FF !important;
            }
            [role="tab"] p {
                color: inherit !important;
                font-size: 1rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top navigation via tabs
    tabs = st.tabs([label for label, _ in _TABS])
    for tab, (_, render_fn) in zip(tabs, _TABS):
        with tab:
            render_fn()


if __name__ == "__main__":
    main()
