"""Sidebar component – connection settings & experiment selector."""

from __future__ import annotations

import streamlit as st

from state import AppState, get_client, get_experiment_id


def render_sidebar() -> None:
    """Render the sidebar with connection and filtering controls."""
    state = AppState.get()

    with st.sidebar:
        st.header("⚙️ Connection")
        state.tracking_uri = st.text_input(
            "MLflow Tracking URI",
            value=state.tracking_uri,
            help="URI of your MLflow tracking server",
        )
        state.token = st.text_input(
            "Token",
            value=state.token,
            type="password",
            help="Bearer token for MLflow auth (MLFLOW_TRACKING_TOKEN)",
        )

        # Auto-detect experiments and show as dropdown
        try:
            client = get_client(state.tracking_uri, state.token)
            experiments = client.search_experiments()
            exp_names = sorted(
                [e.name for e in experiments if e.name != "Default"],
            )
            if not exp_names:
                exp_names = ["Default"]
        except Exception:  # noqa: BLE001
            exp_names = [state.experiment_name] if state.experiment_name else []
            client = None

        if exp_names:
            default_idx = (
                exp_names.index(state.experiment_name)
                if state.experiment_name in exp_names
                else 0
            )
            state.experiment_name = st.selectbox(
                "Experiment",
                options=exp_names,
                index=default_idx,
            )

        state.max_runs = st.slider(
            "Max runs", min_value=10, max_value=500, value=state.max_runs, step=10
        )

        # Connection status indicator
        try:
            if client is None:
                client = get_client(state.tracking_uri, state.token)
            exp_id = get_experiment_id(client, state.experiment_name)
            if exp_id:
                st.success(f"Connected · `{state.experiment_name}`")
            else:
                st.warning(f"Experiment '{state.experiment_name}' not found")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Connection failed: {exc}")
