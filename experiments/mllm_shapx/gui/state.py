"""Shared session-state helpers for the Streamlit GUI."""

import os
from dataclasses import dataclass

import streamlit as st
from mlflow.tracking import MlflowClient


@dataclass
class AppState:
    """Lightweight wrapper around Streamlit session_state for typed access."""

    tracking_uri: str = ""
    experiment_name: str = "mllm_shapx"
    max_runs: int = 100
    token: str = ""

    @classmethod
    def get(cls) -> "AppState":
        """Retrieve or initialize the global AppState."""
        if "app_state" not in st.session_state:
            default_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
            default_token = os.environ.get("MLFLOW_TRACKING_TOKEN", "")
            st.session_state["app_state"] = cls(
                tracking_uri=default_uri, token=default_token
            )
        return st.session_state["app_state"]


def _apply_auth(token: str) -> None:
    """Set MLflow auth environment variables from the provided token."""
    if token.strip():
        os.environ["MLFLOW_TRACKING_TOKEN"] = token.strip()
    else:
        os.environ.pop("MLFLOW_TRACKING_TOKEN", None)


def get_client(tracking_uri: str, token: str = "") -> MlflowClient:
    """Create an MLflowClient, caching per URI+token within the session."""
    _apply_auth(token)
    uri = tracking_uri.strip() or None
    cache_key = f"_mlflow_client_{uri}_{bool(token.strip())}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = MlflowClient(tracking_uri=uri)
    return st.session_state[cache_key]


def get_experiment_id(client: MlflowClient, name: str) -> str | None:
    """Get experiment ID by name, returning None if not found."""
    exp = client.get_experiment_by_name(name)
    return exp.experiment_id if exp else None
