"""Metric chart helpers."""

from typing import Dict, List

import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


def plot_metric_history(
    client: MlflowClient,
    run_id: str,
    metric_keys: List[str],
    title: str = "Metrics",
) -> None:
    """Fetch metric histories and plot them as a line chart."""
    frames: List[pd.DataFrame] = []
    for key in metric_keys:
        hist = client.get_metric_history(run_id, key)
        if not hist:
            continue
        df = pd.DataFrame(
            {
                "step": [h.step for h in hist],
                "value": [h.value for h in hist],
                "metric": key,
            }
        )
        frames.append(df)

    if not frames:
        st.info("No metric history for the selected keys.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    pivot = all_df.pivot_table(
        index="step", columns="metric", values="value", aggfunc="last"
    )
    st.subheader(title)
    st.line_chart(pivot)


def metric_summary_cards(metrics: Dict[str, float], cols: int = 4) -> None:
    """Render a row of metric summary cards."""
    if not metrics:
        return
    keys = sorted(metrics.keys())
    columns = st.columns(min(cols, len(keys)))
    for i, key in enumerate(keys):
        col = columns[i % len(columns)]
        label = key.split("/")[-1]
        col.metric(label=label, value=f"{metrics[key]:.4g}")


def compare_metrics_chart(
    client: MlflowClient,
    run_ids: List[str],
    metric_key: str,
    run_labels: Dict[str, str] | None = None,
) -> None:
    """Plot a single metric across multiple runs for comparison."""
    frames: List[pd.DataFrame] = []
    for rid in run_ids:
        hist = client.get_metric_history(rid, metric_key)
        if not hist:
            continue
        label = (run_labels or {}).get(rid, rid[:8])
        df = pd.DataFrame(
            {
                "step": [h.step for h in hist],
                "value": [h.value for h in hist],
                "run": label,
            }
        )
        frames.append(df)

    if not frames:
        st.info(f"No history for '{metric_key}' across selected runs.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    pivot = all_df.pivot_table(
        index="step", columns="run", values="value", aggfunc="last"
    )
    st.line_chart(pivot)
