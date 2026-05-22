"""Tests for NLP text helpers and statistics utilities."""

import pandas as pd

from src.nlp import count_sentences, split_into_sentences
from src.statistics import get_df_stats, get_sample_df


def test_split_into_sentences() -> None:
    text = "First sentence. Second sentence! Third?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("First")


def test_count_sentences_matches_split() -> None:
    text = "One. Two. Three."
    assert count_sentences(text) == len(split_into_sentences(text))


def test_get_df_stats_basic_metrics() -> None:
    df = pd.DataFrame({
        "prompt": ["ab", "abcd"],
        "sentences": [["a", "b"], ["a", "b", "c", "d"]],
    })
    stats = get_df_stats(df, include_audio=False)
    assert stats["rows__num"] == 2
    assert stats["total_sentences__num"] == 6
    assert stats["unique_entries__num"] == 2


def test_get_sample_df_picks_longest_per_group() -> None:
    df = pd.DataFrame({
        "datasets": [["A"], ["A"], ["B"]],
        "sentences": [
            ["one"],
            ["one", "two", "three"],
            ["solo"],
        ],
    })
    sample = get_sample_df(df)
    assert len(sample) == 2
    sentences = sample["sentences"].tolist()
    assert ["one", "two", "three"] in sentences
    assert ["one"] not in sentences
