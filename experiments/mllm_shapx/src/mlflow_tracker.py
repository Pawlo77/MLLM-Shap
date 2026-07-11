"""MLflow tracking wrapper for experiment runs."""

import json
import logging
import os
import mlflow
import mlflow.system_metrics
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Set

LOGGER = logging.getLogger(__name__)

_MLLM_SHAPX_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_TRACKING_URI = f"sqlite:///{_MLLM_SHAPX_DIR / 'mlflow.db'}"


def _ensure_tracking_uri() -> None:
    """Set the default MLflow tracking URI if none is configured."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(_DEFAULT_TRACKING_URI)


def _flatten_params(
    prefix: str, obj: Any, out: Dict[str, str], limit: int = 90
) -> None:
    """Flatten nested parameters for MLflow logging."""
    if len(out) >= limit:
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[key[:250]] = "" if v is None else str(v)[:500]
            elif isinstance(v, (list, dict)) and len(json.dumps(v, default=str)) < 400:
                out[key[:250]] = json.dumps(v, default=str)[:500]
            if len(out) >= limit:
                return
            elif isinstance(v, Mapping):
                _flatten_params(key, v, out, limit)


def _find_existing_run(experiment_name: str, run_name: str) -> str | None:
    """Find an existing MLflow run by experiment + run name. Returns run_id or None."""
    _ensure_tracking_uri()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f'tags.mlflow.runName = "{run_name}"',
        output_format="list",
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id
    return None


def _get_completed_indices_from_run(run_id: str) -> Set[int]:
    """Query MLflow metric history to find completed sample indices."""
    client = mlflow.tracking.MlflowClient()
    try:
        history = client.get_metric_history(run_id, "progress/sample_index")
        return {int(m.value) for m in history}
    except Exception:  # noqa: BLE001
        return set()


class MlflowTracker:
    """MLflow run lifecycle, metric/artifact/dict logging, and resume support."""

    def __init__(self, cfg: Any, run_name: str, experiment_set_id: str) -> None:
        self._cfg = cfg
        self._run_name = run_name
        self._experiment_set_id = experiment_set_id
        self._active = False
        self._run_id: str | None = None

    def __enter__(self) -> "MlflowTracker":
        if self._cfg.tracking_uri:
            mlflow.set_tracking_uri(self._cfg.tracking_uri)
        else:
            _ensure_tracking_uri()
        mlflow.set_experiment(self._cfg.experiment_name)
        if self._cfg.system_metrics_enabled:
            os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true")
            try:
                mlflow.system_metrics.enable_system_metrics_logging()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("System metrics not enabled: %s", exc)
        tags: Dict[str, str] = {
            f"tag_{i}": str(t)[:500] for i, t in enumerate(self._cfg.tags)
        }
        tags["experiment_set_id"] = self._experiment_set_id[:500]
        self._run = mlflow.start_run(
            run_id=self._run_id,
            run_name=self._run_name,
            nested=self._cfg.nested_per_variant,
            tags=tags,
        )
        self._active = True
        return self

    def __exit__(self, *args: Any) -> None:
        if self._active:
            mlflow.end_run()
            self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def try_resume(self) -> Set[int]:
        """Find existing run and return completed sample indices. Sets run_id for reuse."""
        existing_id = _find_existing_run(self._cfg.experiment_name, self._run_name)
        if existing_id is None:
            return set()
        self._run_id = existing_id
        return _get_completed_indices_from_run(existing_id)

    def log_params_shallow(self, spec: Mapping[str, Any]) -> None:
        """Log shallow parameters to MLflow."""
        if not self._active:
            return
        mlflow.log_param("experiment_set_id", self._experiment_set_id)
        flat: Dict[str, str] = {}
        _flatten_params("", dict(spec), flat)
        for k, v in flat.items():
            try:
                mlflow.log_param(k, v)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("skip param %s: %s", k, exc)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log metrics to MLflow."""
        if not self._active:
            return
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v), step=step)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("skip metric %s: %s", k, exc)

    def log_dict(self, data: Any, artifact_file: str) -> None:
        """Log a Python dict/list as a JSON artifact without persistent disk writes."""
        if not self._active:
            return

        try:
            mlflow.log_dict(data, artifact_file=artifact_file)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("log_dict failed for %s: %s", artifact_file, exc)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Log a local file as an artifact to MLflow."""
        if not self._active:
            return

        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("log_artifact failed for %s: %s", path, exc)

    def log_bytes_artifact(
        self, data: bytes, filename: str, artifact_path: str | None = None
    ) -> None:
        """Log binary data as an artifact using a temp file."""
        if not self._active:
            return
        with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            self.log_artifact(Path(tmp.name), artifact_path=artifact_path)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set tags for the current MLflow run."""
        if not self._active:
            return

        for k, v in tags.items():
            try:
                mlflow.set_tag(k, str(v)[:5000])
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("skip tag %s: %s", k, exc)


def start_mlflow_run(cfg: Any, run_name: str, experiment_set_id: str) -> MlflowTracker:
    """Convenience function to create an MlflowTracker."""
    return MlflowTracker(cfg, run_name, experiment_set_id)
