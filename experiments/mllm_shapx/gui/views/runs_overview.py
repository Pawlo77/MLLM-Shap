"""Runs Overview page - tabular summary of all experiment runs."""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from state import AppState, get_client, get_experiment_id


def render() -> None:
    """Render the runs overview page."""
    st.header("📋 Runs Overview")

    state = AppState.get()
    try:
        client = get_client(state.tracking_uri, state.token)
        exp_id = get_experiment_id(client, state.experiment_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Connection error: {exc}")
        return

    if not exp_id:
        st.warning(
            f"Experiment '{state.experiment_name}' not found. "
            "Check your connection settings in the sidebar."
        )
        return

    runs = client.search_runs(
        experiment_ids=[exp_id],
        order_by=["attributes.start_time DESC"],
        max_results=state.max_runs,
    )
    if not runs:
        st.info("No runs found in this experiment.")
        return

    # Build a summary table
    rows = []
    for r in runs:
        start = r.info.start_time
        start_dt = (
            datetime.fromtimestamp(start / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
            if start
            else "—"
        )
        duration_s = (
            (r.info.end_time - r.info.start_time) / 1000
            if r.info.end_time and r.info.start_time
            else None
        )
        rows.append(
            {
                "Run Name": r.info.run_name or "—",
                "Run ID": r.info.run_id[:12],
                "Status": r.info.status,
                "Started": start_dt,
                "Duration (s)": f"{duration_s:.1f}" if duration_s else "—",
                "Metrics": len(r.data.metrics),
                "Params": len(r.data.params),
            }
        )

    df = pd.DataFrame(rows)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "Filter by status",
            options=df["Status"].unique().tolist(),
            default=df["Status"].unique().tolist(),
        )
    with col2:
        search_term = st.text_input("Search run name", "")

    mask = df["Status"].isin(status_filter)
    if search_term:
        mask &= df["Run Name"].str.contains(search_term, case=False, na=False)
    filtered = df[mask]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Run ID": st.column_config.TextColumn(width="small"),
            "Status": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(f"Showing {len(filtered)} of {len(runs)} runs")
