"""Run Detail page - deep-dive into a single run."""

import streamlit as st

from components.metrics import metric_summary_cards, plot_metric_history
from state import AppState, get_client, get_experiment_id


def render() -> None:
    """Render the run detail page."""
    st.header("🔍 Run Detail")

    state = AppState.get()
    try:
        client = get_client(state.tracking_uri, state.token)
        exp_id = get_experiment_id(client, state.experiment_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Connection error: {exc}")
        return

    if not exp_id:
        return

    # Fetch all runs for selection
    all_runs = client.search_runs(
        experiment_ids=[exp_id],
        order_by=["attributes.start_time DESC"],
        max_results=state.max_runs,
    )
    if not all_runs:
        st.info("No runs found.")
        return

    run_labels = {r.info.run_id: r.info.run_name or r.info.run_id[:8] for r in all_runs}
    run_id = st.selectbox(
        "Select run",
        options=[r.info.run_id for r in all_runs],
        format_func=lambda rid: run_labels.get(rid, rid[:8]),
    )
    if not run_id:
        return

    run = next(r for r in all_runs if r.info.run_id == run_id)

    # --- Info panel ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", run.info.status)
    col2.metric("Run Name", run.info.run_name or "—")
    if run.info.end_time and run.info.start_time:
        col3.metric(
            "Duration", f"{(run.info.end_time - run.info.start_time) / 1000:.0f}s"
        )

    # --- Tags ---
    with st.expander("🏷️ Tags", expanded=False):
        st.json(dict(run.data.tags))

    # --- Parameters ---
    with st.expander("📦 Parameters", expanded=False):
        if run.data.params:
            st.json(dict(run.data.params))
        else:
            st.caption("No parameters logged.")

    # --- Metrics summary ---
    st.subheader("Metrics (latest values)")
    metric_summary_cards(run.data.metrics)

    # --- Metric history plots ---
    metric_keys = sorted(run.data.metrics.keys())
    if not metric_keys:
        return

    progress_keys = [k for k in metric_keys if "progress" in k or "sample" in k]
    timing_keys = [k for k in metric_keys if "time" in k or "runtime" in k]
    other_keys = [
        k for k in metric_keys if k not in progress_keys and k not in timing_keys
    ]

    chosen = st.multiselect(
        "Metrics to plot",
        options=metric_keys,
        default=progress_keys[:2] + timing_keys[:1] + other_keys[:2],
    )
    if chosen:
        plot_metric_history(client, run_id, chosen)
