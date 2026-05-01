"""Generate a static HTML dashboard for rebuttal SLURM runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_LISTS: list[Path] = [
    Path("experiments/interspeech/configs/rebuttal_stage1_1k_sgpa_configs.txt"),
    Path("experiments/interspeech/configs/rebuttal_stage2_500_original_configs.txt"),
    Path("experiments/interspeech/configs/rebuttal_stage3_1k_raw_configs.txt"),
]


@dataclass(frozen=True)
class ConfigStatus:
    """Progress summary for one experiment config."""

    stage: str
    config_path: Path
    experiment_set_id: str
    run_slug: str
    input_modality: str
    dataset_subset: str
    segmentation: str
    expected_samples: int
    completed_samples: int
    run_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_config_lists(config_lists: list[Path]) -> list[tuple[str, Path]]:
    configs: list[tuple[str, Path]] = []
    for list_path in config_lists:
        stage_name = list_path.stem.replace("rebuttal_", "").replace("_configs", "")
        if not list_path.exists():
            continue
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            configs.append((stage_name, Path(line)))
    return configs


def _variant_slug(config: dict[str, Any]) -> str:
    """Generate a slug for the run variant based on its config.
    This is used to identify the run directory and display it in the dashboard.
    The slug is based on the explainer type and key parameters that differentiate runs,
    such as linearity, number of samples, or fraction of samples.
    The exact format can be adjusted as needed to ensure uniqueness and readability."""
    variant = config["experiments"][0]
    name = variant.get("name") or variant["explainer_type"]
    if variant.get("linear"):
        linear = str(float(variant["linear"][0])).replace(".", "_")
        return f"{name}_{variant['explainer_type']}_lin{linear}"
    if variant.get("num_samples"):
        return f"{name}_{variant['explainer_type']}_ns{int(variant['num_samples'][0])}"
    if variant.get("fractions"):
        frac = str(float(variant["fractions"][0])).replace(".", "_")
        return f"{name}_{variant['explainer_type']}_frac{frac}"
    return f"{name}_{variant['explainer_type']}"


def _expected_samples(config: dict[str, Any], default: int) -> int:
    """Determine the expected number of samples for a run based on its config,
    falling back to a default if not specified."""
    selection = config.get("selection", {})
    return int(selection.get("max_samples") or default)


def _count_samples(run_dir: Path) -> int:
    """Count the number of completed sample result files in the run directory."""
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return 0
    return len(list(samples_dir.glob("sample_*_result.json")))


def _config_status(
    stage: str,
    config_path: Path,
    project_dir: Path,
    expected_default: int,
) -> ConfigStatus:
    """Load the config and count completed samples to build a ConfigStatus summary."""
    config = _read_json(project_dir / config_path)
    run_slug = _variant_slug(config)
    run_dir = (
        project_dir
        / config.get("output_root", "experiments_output")
        / config["experiment_set_id"]
        / run_slug
    )
    return ConfigStatus(
        stage=stage,
        config_path=config_path,
        experiment_set_id=config["experiment_set_id"],
        run_slug=run_slug,
        input_modality=config["modality"]["input_modality"],
        dataset_subset=config["dataset"]["subset"],
        segmentation=config.get("audio_segmentation", {}).get("method", "raw"),
        expected_samples=_expected_samples(config, expected_default),
        completed_samples=_count_samples(run_dir),
        run_dir=run_dir,
    )


def _run_command(command: list[str]) -> str:
    """Run a command and return its output, or an error message if it fails. Designed to be robust for dashboard use, so it won't raise exceptions or crash the dashboard if something goes wrong with the command execution."""
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001 - dashboard should not crash.
        return f"{type(exc).__name__}: {exc}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return output.strip()


def _sacct_rows(job_ids: list[str]) -> list[dict[str, str]]:
    """Get sacct rows for the given job IDs, or return an empty list if no job IDs are provided."""
    if not job_ids:
        return []
    output = _run_command(
        [
            "sacct",
            "-P",
            "-n",
            "-j",
            ",".join(job_ids),
            "--format=JobID,JobName,State,Elapsed,ExitCode",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        rows.append(
            {
                "job_id": parts[0],
                "job_name": parts[1],
                "state": parts[2],
                "elapsed": parts[3],
                "exit_code": parts[4],
            }
        )
    return rows


def _squeue_output(job_ids: list[str]) -> str:
    """Get the raw squeue output for the given job IDs, or return an empty string if no job IDs are provided."""
    if not job_ids:
        return ""
    return _run_command(["squeue", "-j", ",".join(job_ids)])


def _state_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count the occurrences of each SLURM state in the sacct rows."""
    counts: dict[str, int] = {}
    for row in rows:
        # Count array tasks only, not .batch/.extern/.0 steps.
        job_id = row["job_id"]
        if "." in job_id:
            continue
        state = row["state"].split()[0]
        counts[state] = counts.get(state, 0) + 1
    return counts


def _pct(done: int, total: int) -> float:
    """Calculate percentage completion, handling edge cases."""
    return 0.0 if total <= 0 else min(100.0, 100.0 * done / total)


def _esc(value: object) -> str:
    """Escape a value for safe HTML rendering."""
    return html.escape(str(value))


def _render_progress_bar(done: int, total: int) -> str:
    """Render an HTML progress bar for the given completion status."""
    pct = _pct(done, total)
    return (
        '<div class="bar">'
        f'<div class="fill" style="width:{pct:.1f}%"></div>'
        "</div>"
        f'<span class="pct">{pct:.1f}%</span>'
    )


def _render_html(
    statuses: list[ConfigStatus],
    sacct_rows: list[dict[str, str]],
    squeue_text: str,
    job_ids: list[str],
    refresh_seconds: int,
) -> str:
    """Render the HTML dashboard content."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_expected = sum(status.expected_samples for status in statuses)
    total_completed = sum(status.completed_samples for status in statuses)
    state_counts = _state_counts(sacct_rows)

    grouped: dict[str, list[ConfigStatus]] = {}
    for status in statuses:
        grouped.setdefault(status.stage, []).append(status)

    stage_sections = []
    for stage, items in grouped.items():
        rows = []
        for item in items:
            rows.append(
                "<tr>"
                f"<td>{_esc(stage)}</td>"
                f"<td>{_esc(item.dataset_subset)}</td>"
                f"<td>{_esc(item.input_modality)}</td>"
                f"<td>{_esc(item.segmentation)}</td>"
                f"<td>{_esc(item.completed_samples)}/{_esc(item.expected_samples)}</td>"
                f"<td>{_render_progress_bar(item.completed_samples, item.expected_samples)}</td>"
                f"<td><code>{_esc(item.run_dir.relative_to(Path.cwd()) if item.run_dir.is_relative_to(Path.cwd()) else item.run_dir)}</code></td>"
                "</tr>"
            )
        stage_sections.append(
            f"<h2>{_esc(stage)}</h2>"
            "<table><thead><tr><th>Stage</th><th>Dataset</th><th>Input</th>"
            "<th>Segmentation</th><th>Samples</th><th>Progress</th><th>Run dir</th>"
            "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"
        )

    state_badges = " ".join(
        f'<span class="badge">{_esc(k)}: {v}</span>'
        for k, v in sorted(state_counts.items())
    )
    if not state_badges:
        state_badges = '<span class="muted">No sacct rows yet.</span>'

    sacct_table = "\n".join(
        "<tr>"
        f"<td>{_esc(row['job_id'])}</td>"
        f"<td>{_esc(row['state'])}</td>"
        f"<td>{_esc(row['elapsed'])}</td>"
        f"<td>{_esc(row['exit_code'])}</td>"
        "</tr>"
        for row in sacct_rows[:300]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{int(refresh_seconds)}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rebuttal Runs Dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #222; background: #fafafa; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 28px; }}
    .muted {{ color: #666; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }}
    .stat {{ border: 1px solid #ddd; background: white; border-radius: 8px; padding: 12px; }}
    .stat .value {{ font-size: 26px; font-weight: 700; }}
    .stat .label {{ color: #666; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #ddd; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: middle; }}
    th {{ background: #f2f2f2; font-size: 13px; }}
    code {{ font-size: 12px; }}
    .bar {{ position: relative; height: 10px; width: 160px; background: #eee; border-radius: 999px; display: inline-block; margin-right: 8px; overflow: hidden; }}
    .fill {{ height: 100%; background: #2f6fdd; border-radius: 999px; }}
    .pct {{ font-size: 12px; color: #555; }}
    .badge {{ display: inline-block; padding: 4px 8px; margin: 2px; border: 1px solid #ddd; background: white; border-radius: 999px; font-size: 13px; }}
    pre {{ background: #111; color: #eee; padding: 12px; border-radius: 8px; overflow: auto; max-height: 340px; }}
  </style>
</head>
<body>
  <h1>Rebuttal Runs Dashboard</h1>
  <div class="muted">Last generated: {_esc(now)}. Auto-refresh: {int(refresh_seconds)}s.</div>
  <div class="grid">
    <div class="stat"><div class="value">{_esc(total_completed)}/{_esc(total_expected)}</div><div class="label">completed samples</div></div>
    <div class="stat"><div class="value">{_pct(total_completed, total_expected):.1f}%</div><div class="label">overall progress</div></div>
    <div class="stat"><div class="value">{_esc(len(job_ids))}</div><div class="label">tracked jobs</div></div>
    <div class="stat"><div class="value">{_esc(len(statuses))}</div><div class="label">configs</div></div>
  </div>
  <h2>SLURM State</h2>
  <div>{state_badges}</div>
  {"".join(stage_sections)}
  <h2>squeue</h2>
  <pre>{_esc(squeue_text or "No squeue output.")}</pre>
  <h2>sacct Rows</h2>
  <table><thead><tr><th>JobID</th><th>State</th><th>Elapsed</th><th>ExitCode</th></tr></thead><tbody>{sacct_table}</tbody></table>
</body>
</html>
"""


def _write_csv(statuses: list[ConfigStatus], output_path: Path) -> None:
    """Write a CSV summary of the config statuses alongside the HTML dashboard."""
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "dataset_subset",
                "input_modality",
                "segmentation",
                "expected_samples",
                "completed_samples",
                "run_dir",
            ],
        )
        writer.writeheader()
        for status in statuses:
            writer.writerow(
                {
                    "stage": status.stage,
                    "dataset_subset": status.dataset_subset,
                    "input_modality": status.input_modality,
                    "segmentation": status.segmentation,
                    "expected_samples": status.expected_samples,
                    "completed_samples": status.completed_samples,
                    "run_dir": status.run_dir,
                }
            )


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser for the dashboard script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config-list",
        type=Path,
        action="append",
        dest="config_lists",
        default=None,
        help="Config list file. Can be passed multiple times.",
    )
    parser.add_argument("--job-id", action="append", dest="job_ids", default=[])
    parser.add_argument("--output", type=Path, default=Path("rebuttal_status.html"))
    parser.add_argument("--expected-default", type=int, default=100)
    parser.add_argument("--refresh-seconds", type=int, default=30)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    project_dir = args.project_dir.resolve()
    config_lists = args.config_lists or [
        project_dir / path for path in DEFAULT_CONFIG_LISTS
    ]
    config_entries = _expand_config_lists(config_lists)
    statuses = [
        _config_status(
            stage=stage,
            config_path=config_path,
            project_dir=project_dir,
            expected_default=args.expected_default,
        )
        for stage, config_path in config_entries
    ]
    sacct_rows = _sacct_rows(args.job_ids)
    squeue_text = _squeue_output(args.job_ids)
    html_text = _render_html(
        statuses=statuses,
        sacct_rows=sacct_rows,
        squeue_text=squeue_text,
        job_ids=args.job_ids,
        refresh_seconds=args.refresh_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    _write_csv(statuses, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
