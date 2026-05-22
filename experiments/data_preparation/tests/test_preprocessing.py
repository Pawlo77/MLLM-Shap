"""Tests for shared DataFrame preprocessing."""

import pandas as pd

from src.preprocessing import (
    add_datasets_combined,
    add_sentence_columns,
    dedupe_by_prompt,
    filter_multi_sentence,
    filter_single_sentence,
    non_english_prompts,
)


class _FakeClassifier:
    def is_english(self, text: str) -> bool:
        return "english" in text.lower() or text.startswith("First")


def test_add_sentence_columns_splits_prompt(voicebench_like_df: pd.DataFrame) -> None:
    result = add_sentence_columns(
        voicebench_like_df.drop(columns=["sentences", "sentences__num"])
    )
    assert "sentences" in result.columns
    assert result["sentences__num"].min() >= 1


def test_dedupe_by_prompt_merges_dataset_tags(voicebench_like_df: pd.DataFrame) -> None:
    result = dedupe_by_prompt(voicebench_like_df)
    assert len(result) == 2
    merged = result.loc[result["prompt"].str.startswith("First"), "datasets"].iloc[0]
    assert set(merged) == {"BBH", "WildVoice"}


def test_dedupe_by_prompt_keeps_audio_columns() -> None:
    df = pd.DataFrame({
        "prompt": ["same", "same"],
        "dataset": ["A", "B"],
        "sentences": [["same"], ["same"]],
        "sentences__num": [1, 1],
        "audio__original": [[b"a"], [b"b"]],
        "audio__original__duration": [[1.0], [2.0]],
    })
    result = dedupe_by_prompt(df, keep_audio=True)
    assert len(result) == 1
    assert result["audio__original"].iloc[0] == [b"a"]


def test_sentence_count_filters() -> None:
    df = pd.DataFrame({"sentences__num": [1, 2, 3]})
    assert len(filter_single_sentence(df)) == 1
    assert len(filter_multi_sentence(df)) == 2


def test_add_datasets_combined() -> None:
    df = pd.DataFrame({"datasets": [["BBH", "MMSU"], ["WildVoice"]]})
    result = add_datasets_combined(df)
    assert result["datasets__combined"].iloc[0] == "BBH MMSU"


def test_non_english_prompts() -> None:
    df = pd.DataFrame({
        "prompt": ["English text", "français seulement"],
        "is_english": [True, False],
    })
    non_en = non_english_prompts(df)
    assert len(non_en) == 1
    assert "français" in non_en.iloc[0]


def test_add_english_flag() -> None:
    from src.preprocessing import add_english_flag

    df = pd.DataFrame({"prompt": ["English sentence here.", "Texte en français."]})
    result = add_english_flag(df, _FakeClassifier())
    assert result["is_english"].tolist() == [True, False]
