"""Tests for Hub upload commit assembly (mocked API)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.constants import HUB_README_PATH_IN_REPO, HubPublishTarget
from src.hub_upload import upload_readme_to_hub, upload_targets_to_hub


@pytest.fixture
def mock_hf_commit() -> MagicMock:
    commit = MagicMock()
    commit.oid = "a" * 40
    commit.commit_url = "https://huggingface.co/datasets/demo/commit/abc"
    return commit


def test_upload_readme_only(tmp_path: Path, mock_hf_commit: MagicMock) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")

    with (
        patch("src.hub_upload._ensure_write_token"),
        patch("src.hub_upload.HfApi") as api_cls,
    ):
        api_cls.return_value.create_commit.return_value = mock_hf_commit
        result = upload_readme_to_hub(readme_path=readme, repo_id="user/demo")

    assert result.revision == "a" * 40
    api_cls.return_value.create_commit.assert_called_once()
    operations = api_cls.return_value.create_commit.call_args.kwargs["operations"]
    assert len(operations) == 1
    assert operations[0].path_in_repo == HUB_README_PATH_IN_REPO


def test_upload_parquet_and_readme_same_commit(
    tmp_path: Path, mock_hf_commit: MagicMock
) -> None:
    parquet = tmp_path / "demo.parquet"
    parquet.write_bytes(b"PAR1")
    readme = tmp_path / "README.md"
    readme.write_text("# Demo", encoding="utf-8")
    target = HubPublishTarget(hub_config="demo", parquet_path=parquet)

    with (
        patch("src.hub_upload._ensure_write_token"),
        patch("src.hub_upload.HfApi") as api_cls,
    ):
        api_cls.return_value.create_commit.return_value = mock_hf_commit
        result = upload_targets_to_hub(
            [target],
            readme_path=readme,
            repo_id="user/demo",
            commit_message="sync",
        )

    assert len(result.uploaded) == 1
    operations = api_cls.return_value.create_commit.call_args.kwargs["operations"]
    assert len(operations) == 2
    paths = {op.path_in_repo for op in operations}
    assert paths == {"demo/test/0000.parquet", HUB_README_PATH_IN_REPO}
    assert (
        api_cls.return_value.create_commit.call_args.kwargs["commit_message"] == "sync"
    )
