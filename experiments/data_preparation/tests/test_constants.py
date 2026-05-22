"""Tests for Hub naming constants and publish target registry."""

from src.constants import (
    HUB_DEFAULT_SPLIT,
    HUB_PUBLISH_TARGETS,
    HUB_REPO_ID,
    HUB_SAMPLES_PER_LANGUAGE,
    HUB_TARGET_SAMPLES,
    MULTI_LINGUAL__INFINITY_INSTRUCT,
    MULTI_SENTENCE__VOICE_BENCH,
    SINGLE_SENTENCE__LIBRISPEECH_ASR,
    SINGLE_SENTENCE__VOICE_BENCH,
    hub_parquet_path_in_repo,
)


def test_hub_config_names_use_task_source_pattern() -> None:
    for name in (
        SINGLE_SENTENCE__VOICE_BENCH,
        MULTI_SENTENCE__VOICE_BENCH,
        SINGLE_SENTENCE__LIBRISPEECH_ASR,
        MULTI_LINGUAL__INFINITY_INSTRUCT,
    ):
        assert "__" in name
        assert name == name.lower()


def test_hub_samples_per_language_derived_from_target() -> None:
    assert HUB_SAMPLES_PER_LANGUAGE == HUB_TARGET_SAMPLES // 3


def test_publish_targets_match_config_constants() -> None:
    configs = {t.hub_config for t in HUB_PUBLISH_TARGETS}
    assert configs == {
        SINGLE_SENTENCE__VOICE_BENCH,
        MULTI_SENTENCE__VOICE_BENCH,
        SINGLE_SENTENCE__LIBRISPEECH_ASR,
        MULTI_LINGUAL__INFINITY_INSTRUCT,
    }
    for target in HUB_PUBLISH_TARGETS:
        assert target.parquet_path.name == f"{target.hub_config}.parquet"
        assert (
            hub_parquet_path_in_repo(target.hub_config, target.split)
            == f"{target.hub_config}/{HUB_DEFAULT_SPLIT}/0000.parquet"
        )


def test_hub_repo_id() -> None:
    assert HUB_REPO_ID == "Pawlo77/mllm-shap"
