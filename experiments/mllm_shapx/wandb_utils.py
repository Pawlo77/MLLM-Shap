"""Thin wrappers around Weights & Biases to keep imports isolated."""

from __future__ import annotations

import os
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import WandBConfig
from .constants import WandbMode, InputModality, OutputModality


def wandb_init_if_enabled(
    cfg: WandBConfig, run_name: str, run_config: Dict[str, Any]
) -> Optional[Any]:
    """Initialize a W&B run if enabled; return the run object or None."""
    if not cfg.enabled or cfg.mode == WandbMode.DISABLED.value:
        return None
    if cfg.mode:
        os.environ["WANDB_MODE"] = cfg.mode

    wb = importlib.import_module("wandb")
    init = getattr(wb, "init")
    return init(
        project=cfg.project,
        entity=cfg.entity,
        group=cfg.group,
        name=run_name,
        tags=cfg.tags,
        config=run_config,
    )


def wandb_log_artifact(
    run: Optional[Any],
    local_path: Path,
    artifact_name: str,
    artifact_type: str,
    metadata: Dict[str, Any],
) -> None:
    """Upload a file as an artifact to the given run, if any."""
    if run is None:
        return

    wb = importlib.import_module("wandb")
    artifact = getattr(wb, "Artifact")
    art = artifact(name=artifact_name, type=artifact_type, metadata=metadata)
    art.add_file(str(local_path))
    run.log_artifact(art)


def log_metrics(run: Optional[Any], metrics: Dict[str, Any]) -> None:
    """Log a metrics dict to W&B if a run object is present."""
    if run is None:
        return
    wb = importlib.import_module("wandb")
    log = getattr(wb, "log")
    log(metrics)


def wandb_log_dir_incremental(
    run: Optional[Any],
    dir_path: Path,
    artifact_name: str,
    artifact_type: str,
    metadata: Dict[str, Any],
) -> None:
    """Snapshot a whole directory into a *single* artifact name after each sample.
    Each call creates a new artifact version under the same name; W&B deduplicates unchanged files.
    """
    if run is None:
        return
    wb = importlib.import_module("wandb")
    artifact = getattr(wb, "Artifact")
    art = artifact(name=artifact_name, type=artifact_type, metadata=metadata)
    art.add_dir(str(dir_path))
    run.log_artifact(art, aliases=["latest"])


def log_audio_artifacts(
    run: Optional[Any],
    audio_dir: Path,
    input_modality: InputModality,
    output_modality: OutputModality,
    sample_id: str,
    sample_rate: int = 24_000,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log audio artifacts to WandB.

    Args:
        run: WandB run instance.
        audio_dir: Directory containing audio files.
        input_modality: Input modality used.
        output_modality: Output modality used.
        sample_id: Sample identifier.
        sample_rate: Audio sample rate.
        metadata: Additional metadata to log.
    """
    if run is None:
        return

    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        return

    wb = importlib.import_module("wandb")
    audio_cls = getattr(wb, "Audio")
    log_fn = getattr(wb, "log")

    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    for audio_file in audio_files:
        try:
            audio = audio_cls(
                str(audio_file),
                caption=f"{sample_id}: {audio_file.stem}",
                sample_rate=sample_rate,
            )

            # Determine if input or output
            is_input = "input" in audio_file.stem.lower()
            key = f"audio/{sample_id}/{'input' if is_input else 'output'}"

            log_fn({key: audio})
        except Exception:
            print("Failed to log audio file:", audio_file)
            continue

    # Log metadata
    if metadata:
        log_fn(
            {
                f"audio_metadata/{sample_id}": {
                    "input_modality": input_modality.value,
                    "output_modality": output_modality.value,
                    **metadata,
                }
            }
        )


def log_audio_files(
    run: Optional[Any],
    audio_files: List[Path],
    sample_id: str,
    sample_rate: int = 24_000,
) -> None:
    """
    Log specific audio files to WandB.

    Args:
        run: WandB run instance.
        audio_files: List of audio file paths.
        sample_id: Sample identifier.
        sample_rate: Audio sample rate.
    """
    if run is None:
        return

    wb = importlib.import_module("wandb")
    audio_cls = getattr(wb, "Audio")
    log_fn = getattr(wb, "log")

    for audio_file in audio_files:
        audio_file = Path(audio_file)
        if not audio_file.exists():
            continue

        try:
            audio = audio_cls(
                str(audio_file),
                caption=f"{sample_id}: {audio_file.stem}",
                sample_rate=sample_rate,
            )

            is_input = "input" in audio_file.stem.lower()
            key = f"audio/{sample_id}/{'input' if is_input else 'output'}"

            log_fn({key: audio})
        except Exception:
            print("Failed to log audio file:", audio_file)
            continue


def create_audio_artifact(
    run: Optional[Any],
    audio_dir: Path,
    artifact_name: str,
    artifact_type: str = "audio",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Create a WandB artifact containing audio files.

    Args:
        run: WandB run instance.
        audio_dir: Directory containing audio files.
        artifact_name: Name for the artifact.
        artifact_type: Type of artifact.
        metadata: Optional metadata for the artifact.

    Returns:
        WandB Artifact or None if not available.
    """
    if run is None:
        return None

    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        return None

    wb = importlib.import_module("wandb")
    artifact_cls = getattr(wb, "Artifact")

    art = artifact_cls(name=artifact_name, type=artifact_type, metadata=metadata or {})
    art.add_dir(str(audio_dir))
    run.log_artifact(art)

    return art
