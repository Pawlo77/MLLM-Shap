"""Tests for Hub upload path planning (no network)."""

from pathlib import Path

import pytest

from src.constants import (
    EXPERIMENTS_ROOT_DIR,
    HUB_README_PATH,
    MULTI_LINGUAL__INFINITY_INSTRUCT,
    SINGLE_SENTENCE__VOICE_BENCH,
    hub_parquet_path_in_repo,
)
from src.hub_upload import (
    build_upload_plan,
    format_upload_plan,
    list_publish_targets,
    resolve_hub_readme_path,
)
from src.constants import HubPublishTarget


def test_hub_parquet_path_in_repo() -> None:
    assert (
        hub_parquet_path_in_repo("multi_sentence__voice_bench", "test")
        == "multi_sentence__voice_bench/test/0000.parquet"
    )


def test_list_publish_targets_filter() -> None:
    targets = list_publish_targets(
        hub_configs={SINGLE_SENTENCE__VOICE_BENCH, MULTI_LINGUAL__INFINITY_INSTRUCT}
    )
    names = {t.hub_config for t in targets}
    assert names == {SINGLE_SENTENCE__VOICE_BENCH, MULTI_LINGUAL__INFINITY_INSTRUCT}


def test_build_upload_plan_missing_file(tmp_path: Path) -> None:
    target = HubPublishTarget(
        hub_config="demo",
        parquet_path=tmp_path / "missing.parquet",
    )
    with pytest.raises(FileNotFoundError, match="Missing parquet"):
        build_upload_plan([target], require_files=True)


def test_build_upload_plan_and_format(tmp_path: Path) -> None:
    parquet = tmp_path / "demo.parquet"
    parquet.write_bytes(b"PAR1")
    target = HubPublishTarget(hub_config="demo", parquet_path=parquet)
    plan = build_upload_plan([target])
    assert len(plan) == 1
    assert plan[0].path_in_repo == "demo/test/0000.parquet"
    summary = format_upload_plan(plan, repo_id="user/ds")
    assert "user/ds" in summary
    assert "demo/test/0000.parquet" in summary


def test_hub_readme_path_exists() -> None:
    assert HUB_README_PATH == EXPERIMENTS_ROOT_DIR / "hf" / "README.md"
    assert HUB_README_PATH.is_file()


def test_format_upload_plan_includes_readme() -> None:
    readme = resolve_hub_readme_path()
    summary = format_upload_plan([], readme_path=readme, repo_id="user/ds")
    assert "README.md" in summary
    assert str(readme) in summary


def test_resolve_hub_readme_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Hub README not found"):
        resolve_hub_readme_path(tmp_path / "missing.md")
