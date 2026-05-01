"""File I/O, config loading, and dataset access for faithfulness evaluation."""

import json
from pathlib import Path
from typing import Any

from experiments.mllm_shapx.src.config import ExperimentSet
from experiments.mllm_shapx.src.data import (
    apply_filters,
    choose_prompt_text_column,
    iter_balanced_token_count_rows,
    iter_rows_for_selection,
    load_df,
)


def load_spec(run_dir: Path, spec_path: Path | None = None) -> dict[str, Any]:
    """Load an mllm_shapx experiment spec from disk."""
    spec_path = spec_path or (run_dir / "spec.json")
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing mllm_shapx spec: {spec_path}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def experiment_set_from_spec(spec: dict[str, Any]) -> ExperimentSet:
    """Build an ``ExperimentSet`` from a persisted run spec.

    The faithfulness experiment consumes only the subset required for replay,
    and explicitly disables W&B side effects.
    """
    raw = {
        "experiment_set_id": spec["experiment_set_id"],
        "output_root": "experiments_output",
        "device": spec.get("device"),
        "connector": spec["connector"],
        "dataset": spec["dataset"],
        "selection": spec["selection"],
        "generation": spec["generation"],
        "modality": spec["modality"],
        "audio_segmentation": spec.get("audio_segmentation") or {},
        "shap": spec["shap"],
        "embedding": spec.get("embedding") or {},
        "experiments": [],
        "wandb": {"enabled": False},
    }
    return ExperimentSet.model_validate(raw)


def load_selected_rows(
    cfg: ExperimentSet, max_samples: int | None
) -> dict[int, dict[str, Any]]:
    """Load and index dataset rows selected by the original experiment config."""
    df = load_df(cfg.dataset)
    if cfg.selection.filters:
        df = apply_filters(df, cfg.selection.filters)

    text_col = choose_prompt_text_column(df)
    selected: dict[int, dict[str, Any]] = {}

    if cfg.selection.balanced_token_counts:
        row_iter = iter_balanced_token_count_rows(
            df=df,
            token_counts=cfg.selection.balanced_token_counts,
            samples_per_token_count=int(cfg.selection.samples_per_token_count or 0),
            start_index=cfg.selection.start_index,
            max_samples=max_samples or cfg.selection.max_samples,
            shuffle_seed=cfg.selection.shuffle_seed,
            allow_partial_buckets=cfg.selection.allow_partial_token_count_buckets,
            token_count_col=cfg.dataset.column_mapping.token_count,
        )
    else:
        row_iter = iter_rows_for_selection(
            df=df,
            start_index=cfg.selection.start_index,
            max_samples=max_samples,
            shuffle_seed=cfg.selection.shuffle_seed,
        )

    for row_idx, row in row_iter:
        row_dict = dict(row)
        row_dict["_text_col"] = text_col
        selected[int(row_idx)] = row_dict
    return selected


def sample_paths(run_dir: Path, max_samples: int | None) -> list[Path]:
    """Return sorted sample result files from a run directory."""
    paths = sorted((run_dir / "samples").glob("sample_*_result.json"))
    if max_samples is not None:
        paths = paths[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No sample JSON files found in {run_dir / 'samples'}")
    return paths


def parse_sample_id(sample_path: Path) -> int:
    """Parse integer sample id from a ``sample_<id>_result.json`` path."""
    return int(sample_path.name.split("_")[1])
