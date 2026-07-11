"""Run Monitor page - live progress, speed, ETA, and resource usage."""

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

from state import AppState, get_client, get_experiment_id


def _get_active_runs(client: MlflowClient, exp_id: str) -> List[Any]:
    """Fetch currently running (RUNNING) and recently finished runs."""
    running = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="attributes.status = 'RUNNING'",
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    finished = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    return list(running) + list(finished)


def _compute_progress(run: Any, client: MlflowClient) -> Dict[str, Any]:
    """Compute progress stats from metric history."""
    run_id = run.info.run_id
    params = run.data.params

    # Total target from params
    max_samples = None
    for key in ("selection.max_samples", "max_samples"):
        if key in params:
            try:
                max_samples = int(params[key])
            except (ValueError, TypeError):
                pass
            break

    # Completed samples from metric history
    try:
        history = client.get_metric_history(run_id, "progress/sample_index")
        completed = len(history) if history else 0
    except Exception:  # noqa: BLE001
        completed = 0

    # Timing data
    try:
        timing_hist = client.get_metric_history(run_id, "timing/runtime_sec")
        runtimes = [m.value for m in timing_hist] if timing_hist else []
    except Exception:  # noqa: BLE001
        runtimes = []

    # Speed calculations
    avg_time_per_sample = sum(runtimes) / len(runtimes) if runtimes else None
    total_runtime = sum(runtimes) if runtimes else 0.0
    speed_samples_per_sec = completed / total_runtime if total_runtime > 0 else None

    # Wall-clock speed (more accurate for throughput)
    start_ms = run.info.start_time
    end_ms = run.info.end_time
    now_ms = int(time.time() * 1000)
    wall_elapsed_sec = ((end_ms or now_ms) - start_ms) / 1000 if start_ms else None
    wall_speed = (
        completed / wall_elapsed_sec
        if wall_elapsed_sec and wall_elapsed_sec > 0 and completed > 0
        else None
    )

    # ETA
    remaining = max_samples - completed if max_samples and completed else None
    eta_sec = (
        remaining * avg_time_per_sample
        if remaining is not None and avg_time_per_sample
        else None
    )
    eta_wall = remaining / wall_speed if remaining is not None and wall_speed else None

    # Recent speed trend (last 10 samples)
    recent_runtimes = runtimes[-10:] if len(runtimes) > 10 else runtimes
    recent_avg = (
        sum(recent_runtimes) / len(recent_runtimes) if recent_runtimes else None
    )

    return {
        "run_id": run_id,
        "run_name": run.info.run_name or run_id[:8],
        "status": run.info.status,
        "max_samples": max_samples,
        "completed": completed,
        "progress_pct": (
            completed / max_samples * 100 if max_samples and max_samples > 0 else None
        ),
        "avg_sec_per_sample": avg_time_per_sample,
        "recent_avg_sec": recent_avg,
        "total_compute_sec": total_runtime,
        "wall_elapsed_sec": wall_elapsed_sec,
        "speed_samples_per_sec": speed_samples_per_sec,
        "wall_speed": wall_speed,
        "eta_sec": eta_sec,
        "eta_wall_sec": eta_wall,
        "runtimes": runtimes,
    }


def _format_duration(seconds: float | None) -> str:
    """Format seconds to human-readable duration."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _render_run_card(stats: Dict[str, Any]) -> None:
    """Render a single run's progress card."""
    status_emoji = "🟢" if stats["status"] == "RUNNING" else "✅"
    st.markdown(f"### {status_emoji} {stats['run_name']}")

    # Progress bar
    pct = stats["progress_pct"]
    if pct is not None:
        st.progress(
            min(pct / 100, 1.0),
            text=f"{stats['completed']} / {stats['max_samples']} samples ({pct:.1f}%)",
        )
    else:
        st.caption(f"{stats['completed']} samples completed (target unknown)")

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "⚡ Speed",
        f"{stats['wall_speed']:.3f}/s" if stats["wall_speed"] else "—",
        help="Samples per second (wall-clock)",
    )
    c2.metric(
        "⏱️ Avg/sample",
        _format_duration(stats["avg_sec_per_sample"]),
        delta=(
            f"{stats['recent_avg_sec'] - stats['avg_sec_per_sample']:+.1f}s trend"
            if stats["recent_avg_sec"] and stats["avg_sec_per_sample"]
            else None
        ),
        delta_color="inverse",
    )
    c3.metric(
        "🏁 ETA",
        _format_duration(stats["eta_wall_sec"]),
        help="Estimated time remaining (wall-clock speed)",
    )
    c4.metric(
        "🕐 Elapsed",
        _format_duration(stats["wall_elapsed_sec"]),
    )

    # Second row
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Compute", _format_duration(stats["total_compute_sec"]))
    c6.metric(
        "Recent Avg",
        _format_duration(stats["recent_avg_sec"]),
        help="Avg time per sample (last 10)",
    )
    if stats["eta_wall_sec"] and stats["status"] == "RUNNING":
        finish_time = datetime.now(tz=timezone.utc) + timedelta(
            seconds=stats["eta_wall_sec"]
        )
        c7.metric("Est. Finish", finish_time.strftime("%H:%M %b %d"))
    else:
        c7.metric("Est. Finish", "—")
    c8.metric(
        "Compute Speed",
        f"{stats['speed_samples_per_sec']:.3f}/s"
        if stats["speed_samples_per_sec"]
        else "—",
    )

    # Per-sample timing chart
    if stats["runtimes"]:
        with st.expander("📈 Per-sample timing", expanded=False):
            timing_df = pd.DataFrame(
                {"sample": range(len(stats["runtimes"])), "seconds": stats["runtimes"]}
            )
            st.line_chart(timing_df, x="sample", y="seconds")

            # Distribution
            rt = pd.Series(stats["runtimes"])
            st.caption(
                f"min={rt.min():.1f}s | median={rt.median():.1f}s | "
                f"max={rt.max():.1f}s | std={rt.std():.1f}s"
            )


def _render_system_metrics(client: MlflowClient, run_id: str) -> None:
    """Render system resource metrics if available."""
    system_keys = [
        "system/cpu_utilization_percentage",
        "system/gpu_utilization_percentage",
        "system/system_memory_usage_megabytes",
        "system/gpu_memory_usage_megabytes",
        "system/gpu_memory_usage_percentage",
        "system/disk_usage_percentage",
        "system/network_receive_megabytes",
    ]
    frames = []
    for key in system_keys:
        try:
            hist = client.get_metric_history(run_id, key)
            if hist:
                short_name = key.split("/")[-1].replace("_", " ").title()
                df = pd.DataFrame(
                    {
                        "step": [h.step for h in hist],
                        short_name: [h.value for h in hist],
                    }
                ).set_index("step")
                frames.append(df)
        except Exception:  # noqa: BLE001
            continue

    if not frames:
        st.caption(
            "No system metrics available (enable `system_metrics_enabled` in config)."
        )
        return

    combined = pd.concat(frames, axis=1)
    # Split into CPU/Memory and GPU charts
    cpu_cols = [c for c in combined.columns if "gpu" not in c.lower()]
    gpu_cols = [c for c in combined.columns if "gpu" in c.lower()]

    if cpu_cols:
        st.markdown("**CPU / Memory / Disk**")
        st.line_chart(combined[cpu_cols])
    if gpu_cols:
        st.markdown("**GPU**")
        st.line_chart(combined[gpu_cols])


def render() -> None:
    """Render the run monitor page."""
    st.header("⚡ Run Monitor")

    state = AppState.get()
    try:
        client = get_client(state.tracking_uri, state.token)
        exp_id = get_experiment_id(client, state.experiment_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Connection error: {exc}")
        return

    if not exp_id:
        st.warning(f"Experiment '{state.experiment_name}' not found.")
        return

    # Auto-refresh toggle
    col_r1, col_r2 = st.columns([3, 1])
    with col_r2:
        refresh_interval = st.selectbox(
            "Refresh",
            options=[0, 15, 30, 60],
            format_func=lambda x: "Off" if x == 0 else f"{x}s",
        )
    if refresh_interval:
        # Use fragment-based rerun with sleep in a placeholder
        import streamlit.components.v1 as components

        components.html(
            f"<script>setTimeout(() => window.parent.postMessage({{isStreamlitMessage: true, type: 'streamlit:rerun'}}, '*'), {refresh_interval * 1000});</script>",
            height=0,
        )

    runs = _get_active_runs(client, exp_id)
    if not runs:
        st.info("No active or recent runs found.")
        return

    # Separate running vs finished
    running = [r for r in runs if r.info.status == "RUNNING"]
    finished = [r for r in runs if r.info.status != "RUNNING"]

    if running:
        st.subheader(f"🟢 Active Runs ({len(running)})")
        for run in running:
            stats = _compute_progress(run, client)
            _render_run_card(stats)
            with st.expander("🖥️ System Resources", expanded=False):
                _render_system_metrics(client, run.info.run_id)
            st.divider()

    if finished:
        st.subheader(f"✅ Recently Completed ({len(finished)})")
        for run in finished[:5]:
            stats = _compute_progress(run, client)
            _render_run_card(stats)
            st.divider()
