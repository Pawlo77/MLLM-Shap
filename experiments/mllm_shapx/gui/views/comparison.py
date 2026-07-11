"""Run Comparison page - compare metrics across multiple runs."""

import pandas as pd
import streamlit as st

from components.metrics import compare_metrics_chart
from state import AppState, get_client, get_experiment_id


def render() -> None:
    """Render the comparison page."""
    st.header("⚖️ Run Comparison")

    state = AppState.get()
    try:
        client = get_client(state.tracking_uri, state.token)
        exp_id = get_experiment_id(client, state.experiment_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Connection error: {exc}")
        return

    if not exp_id:
        return

    # Fetch all runs for multi-selection
    all_runs = client.search_runs(
        experiment_ids=[exp_id],
        order_by=["attributes.start_time DESC"],
        max_results=state.max_runs,
    )
    if not all_runs:
        st.info("No runs found.")
        return

    run_labels = {r.info.run_id: r.info.run_name or r.info.run_id[:8] for r in all_runs}
    selected_ids = st.multiselect(
        "Select runs to compare",
        options=[r.info.run_id for r in all_runs],
        format_func=lambda rid: run_labels.get(rid, rid[:8]),
    )
    if len(selected_ids) < 2:
        st.info("Select **2 or more** runs to compare.")
        return

    runs = [r for r in all_runs if r.info.run_id in selected_ids]
    run_ids = selected_ids

    # --- Summary table ---
    st.subheader("Summary")
    all_metric_keys: set[str] = set()
    for r in runs:
        all_metric_keys.update(r.data.metrics.keys())
    sorted_keys = sorted(all_metric_keys)

    rows = []
    for r in runs:
        row = {"Run": run_labels[r.info.run_id]}
        for k in sorted_keys:
            row[k] = r.data.metrics.get(k)
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Run")
    st.dataframe(summary_df, use_container_width=True)

    # --- Per-metric comparison charts ---
    st.subheader("Metric History Comparison")
    chosen_metrics = st.multiselect(
        "Metrics to compare",
        options=sorted_keys,
        default=sorted_keys[:3],
    )

    for metric_key in chosen_metrics:
        st.markdown(f"**{metric_key}**")
        compare_metrics_chart(client, run_ids, metric_key, run_labels)
