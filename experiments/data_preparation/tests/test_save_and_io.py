"""Tests for save helpers and I/O utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.constants import LIBRISPEECH_ASR__CONFIG, VOICE_BENCH__CONFIG
from src.io import ensure_dir, load_dataset, save_json, save_parquet
from src.librispeech_loaders import _resolve_clean_splits
from src.save import prepare_for_save, save_single_sentence


def test_prepare_for_save_drops_temp_columns_and_orders_audio() -> None:
    df = pd.DataFrame({
        "prompt": ["hello"],
        "interestingness_score": [0.9],
        "sentences__num": [1],
        "audio__male": [[b"wav"]],
        "audio__male__duration": [[1.0]],
    })
    result = prepare_for_save(df)
    assert "interestingness_score" not in result.columns
    assert "sentences__num" not in result.columns
    assert list(result.columns)[-2:] == [
        "audio__male",
        "audio__male__duration",
    ]


def test_save_single_sentence_writes_parquet_and_sample(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "prompt": ["Alpha prompt.", "Beta prompt."],
        "datasets": [["BBH"], ["MMSU"]],
        "token_count": [5, 8],
    })
    save_single_sentence(df, tmp_path, name="demo_single")
    assert (tmp_path / "demo_single.parquet").is_file()
    assert (tmp_path / "demo_single__text__sample.json").is_file()
    loaded = pd.read_parquet(tmp_path / "demo_single.parquet")
    assert "prompt" not in loaded.columns
    assert loaded["sentences"].iloc[0] == ["Alpha prompt."]


def test_save_parquet_rejects_empty_dataframe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        save_parquet(pd.DataFrame(), tmp_path / "empty.parquet")


def test_save_json_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1], "b": ["text"]})
    path = tmp_path / "out.json"
    save_json(df, path)
    loaded = pd.read_json(path)
    assert loaded["a"].iloc[0] == 1


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    assert ensure_dir(target) == target
    assert target.is_dir()


def test_load_dataset_rejects_non_hash_revision() -> None:
    bad = VOICE_BENCH__CONFIG.model_copy(update={"revision": "main"})
    with pytest.raises(ValueError, match="Unsafe Hugging Face revision"):
        load_dataset("BBH", bad)


def test_resolve_clean_splits_finds_aliases() -> None:
    repo_files = [
        "clean/train.100/0000.parquet",
        "clean/train.360/0000.parquet",
        "clean/dev.clean/0000.parquet",
        "clean/test.clean/0000.parquet",
        "README.md",
    ]
    resolved = _resolve_clean_splits(repo_files)
    assert resolved == ["train.100", "train.360", "dev.clean", "test.clean"]


def test_resolve_clean_splits_raises_when_none_found() -> None:
    with pytest.raises(ValueError, match="No target clean splits"):
        _resolve_clean_splits(["other/train.parquet"])


def test_librispeech_config_has_valid_revision_hash() -> None:
    assert len(LIBRISPEECH_ASR__CONFIG.revision) == 40
