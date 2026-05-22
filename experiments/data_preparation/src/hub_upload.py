"""Upload local builder parquets to the published Hugging Face dataset repo.

Utilities to build a safe upload plan and commit parquet artifacts to the
Hugging Face Hub. Handles token checks, simple dry-run formatting and
error translation for common Hub failures.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from .constants import (
    HUB_PUBLISH_TARGETS,
    HUB_README_PATH,
    HUB_README_PATH_IN_REPO,
    HUB_REPO_ID,
    HubPublishTarget,
    hub_parquet_path_in_repo,
)


@dataclass(frozen=True)
class UploadPlanItem:
    """Resolved upload entry ready for the Hub API."""

    target: HubPublishTarget
    """The original publish target this item was built from, containing the Hub config, local parquet path and description."""
    path_in_repo: str
    """Target path inside the Hub repo, e.g. "single_sentence__voice_bench/test/0000.parquet"."""
    parquet_path: Path
    """Resolved local path to the parquet file to upload. Should exist and be a file, but not checked until the upload step."""
    size_bytes: int
    """Size of the parquet file in bytes, used for dry-run summaries and upload progress estimation."""


@dataclass(frozen=True)
class UploadResult:
    """Outcome of one Hub commit."""

    repo_id: str
    """Hub repo id where the files were uploaded, e.g. "Pawlo77/mllm-shap"."""
    revision: str
    """New commit hash created by the upload, which can be used to reference the exact dataset version containing the uploaded files. Should be a 40-character hexadecimal string."""
    commit_url: str
    """URL to the commit on the Hugging Face Hub, useful for verification and sharing. May be empty if not provided by the API response."""
    uploaded: tuple[UploadPlanItem, ...]
    """Tuple of the items that were successfully uploaded in this commit, including their original targets, resolved local paths and Hub repo paths."""


def list_publish_targets(
    hub_configs: set[str] | None = None,
) -> list[HubPublishTarget]:
    """Return publish targets, optionally filtered by Hub config name."""
    targets = list(HUB_PUBLISH_TARGETS)
    if hub_configs is None:
        return targets
    return [t for t in targets if t.hub_config in hub_configs]


def build_upload_plan(
    targets: list[HubPublishTarget],
    require_files: bool = True,
) -> list[UploadPlanItem]:
    """Resolve parquet paths and Hub paths for the given targets."""
    plan: list[UploadPlanItem] = []
    missing: list[Path] = []

    for target in targets:
        parquet_path = target.parquet_path.resolve()
        if not parquet_path.is_file():
            missing.append(parquet_path)
            continue
        plan.append(
            UploadPlanItem(
                target=target,
                path_in_repo=hub_parquet_path_in_repo(target.hub_config, target.split),
                parquet_path=parquet_path,
                size_bytes=parquet_path.stat().st_size,
            )
        )

    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        if require_files:
            raise FileNotFoundError(
                "Missing parquet file(s). Run the builder notebook(s) first:\n"
                f"{missing_list}"
            )
    return plan


def format_upload_plan(
    plan: list[UploadPlanItem],
    *,
    repo_id: str = HUB_REPO_ID,
    readme_path: Path | None = None,
) -> str:
    """Human-readable summary for dry runs."""
    readme = readme_path.resolve() if readme_path is not None else None
    file_count = len(plan) + (1 if readme is not None else 0)
    lines = [f"Repo: {repo_id}", f"Files: {file_count}"]
    for item in plan:
        size_mb = item.size_bytes / (1024 * 1024)
        lines.append(
            f"  {item.target.hub_config}: {item.parquet_path} "
            f"-> {item.path_in_repo} ({size_mb:.1f} MiB)"
        )
        if item.target.description:
            lines.append(f"      {item.target.description}")
    if readme is not None:
        size_kb = readme.stat().st_size / 1024
        lines.append(
            f"  README: {readme} -> {HUB_README_PATH_IN_REPO} ({size_kb:.1f} KiB)"
        )
    return "\n".join(lines)


def resolve_hub_readme_path(readme_path: Path | None = None) -> Path:
    """Return the local Hub README path, raising if it is missing."""
    path = (readme_path or HUB_README_PATH).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Hub README not found: {path}. Expected dataset card at "
            f"{HUB_README_PATH.relative_to(HUB_README_PATH.parent.parent)}."
        )
    return path


def _ensure_write_token() -> None:
    """Check for a Hugging Face token with write access and raise a clear error if not found."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return
    try:
        HfApi().whoami()
    except Exception as exc:  # noqa: BLE001 — surface any auth failure clearly
        raise RuntimeError(
            "No Hugging Face write token found. Set HF_TOKEN (recommended) or run "
            "`hf auth login` with a token that has write access to the dataset repo."
        ) from exc


def upload_readme_to_hub(
    readme_path: Path | None = None,
    repo_id: str = HUB_REPO_ID,
    commit_message: str | None = None,
    create_pr: bool = False,
) -> UploadResult:
    """Upload the dataset card README to the Hub repo root."""
    return upload_targets_to_hub(
        [],
        readme_path=readme_path,
        repo_id=repo_id,
        commit_message=commit_message,
        create_pr=create_pr,
    )


def upload_targets_to_hub(
    targets: list[HubPublishTarget],
    repo_id: str = HUB_REPO_ID,
    commit_message: str | None = None,
    create_pr: bool = False,
    readme_path: Path | None = None,
) -> UploadResult:
    """Upload parquets and/or the Hub README in a single commit."""
    include_readme = readme_path is not None
    if not targets and not include_readme:
        raise ValueError("No publish targets or README selected.")

    plan = build_upload_plan(targets, require_files=bool(targets)) if targets else []
    resolved_readme = resolve_hub_readme_path(readme_path) if include_readme else None
    _ensure_write_token()

    api = HfApi()
    if commit_message:
        message = commit_message
    elif plan and resolved_readme is not None:
        config_names = ", ".join(item.target.hub_config for item in plan)
        message = f"Update dataset configs and README: {config_names}"
    elif plan:
        config_names = ", ".join(item.target.hub_config for item in plan)
        message = f"Update dataset configs: {config_names}"
    else:
        message = "Update dataset README"

    operations = [
        CommitOperationAdd(
            path_in_repo=item.path_in_repo,
            path_or_fileobj=str(item.parquet_path),
        )
        for item in plan
    ]
    if resolved_readme is not None:
        operations.append(
            CommitOperationAdd(
                path_in_repo=HUB_README_PATH_IN_REPO,
                path_or_fileobj=str(resolved_readme),
            )
        )

    try:
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=message,
            create_pr=create_pr,
        )
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"Dataset repo not found or not accessible: {repo_id}. "
            "Check the repo id and that your token has write access."
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Hub upload failed for {repo_id}: {exc}") from exc

    return UploadResult(
        repo_id=repo_id,
        revision=commit.oid,
        commit_url=commit.commit_url or "",
        uploaded=tuple(plan),
    )


def get_latest_hub_revision(repo_id: str = HUB_REPO_ID) -> str:
    """Return the current commit hash on the default branch."""
    info = HfApi().repo_info(repo_id, repo_type="dataset")
    sha = info.sha
    if not sha:
        raise RuntimeError(f"Could not resolve revision for {repo_id}")
    return sha
