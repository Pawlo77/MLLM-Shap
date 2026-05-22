"""Tests for data module — loading, filtering, text extraction, row iteration."""

import numpy as np
import pandas as pd
import pytest

from ..src.config import FilterPredicate
from ..src.data import (
    apply_filters,
    choose_prompt_text_column,
    extract_text_from_row,
    extract_texts_from_row,
    iter_balanced_token_count_rows,
    iter_rows_for_selection,
)


class TestApplyFilters:
    def test_filter_in(self) -> None:
        df = pd.DataFrame({"lang": ["en", "de", "fr", "en"], "x": [1, 2, 3, 4]})
        result = apply_filters(
            df, [FilterPredicate(column="lang", op="in", value=["en"])]
        )
        assert len(result) == 2
        assert list(result["lang"]) == ["en", "en"]

    def test_filter_not_in(self) -> None:
        df = pd.DataFrame({"lang": ["en", "de", "fr"], "x": [1, 2, 3]})
        result = apply_filters(
            df, [FilterPredicate(column="lang", op="not_in", value=["en"])]
        )
        assert len(result) == 2
        assert "en" not in result["lang"].values

    def test_filter_eq(self) -> None:
        df = pd.DataFrame({"score": [1, 2, 3, 4]})
        result = apply_filters(df, [FilterPredicate(column="score", op="==", value=3)])
        assert len(result) == 1
        assert result.iloc[0]["score"] == 3

    def test_filter_neq(self) -> None:
        df = pd.DataFrame({"score": [1, 2, 3]})
        result = apply_filters(df, [FilterPredicate(column="score", op="!=", value=2)])
        assert len(result) == 2

    def test_filter_lt(self) -> None:
        df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
        result = apply_filters(df, [FilterPredicate(column="score", op="<", value=3)])
        assert len(result) == 2

    def test_filter_lte(self) -> None:
        df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
        result = apply_filters(df, [FilterPredicate(column="score", op="<=", value=3)])
        assert len(result) == 3

    def test_filter_gt(self) -> None:
        df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
        result = apply_filters(df, [FilterPredicate(column="score", op=">", value=3)])
        assert len(result) == 2

    def test_filter_gte(self) -> None:
        df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
        result = apply_filters(df, [FilterPredicate(column="score", op=">=", value=3)])
        assert len(result) == 3

    def test_filter_between(self) -> None:
        df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
        result = apply_filters(
            df, [FilterPredicate(column="score", op="between", value=[2, 4])]
        )
        assert len(result) == 3
        assert list(result["score"]) == [2, 3, 4]

    def test_missing_column_skipped(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = apply_filters(
            df, [FilterPredicate(column="missing", op="==", value=1)]
        )
        assert len(result) == 3  # no filtering applied

    def test_multiple_filters_chained(self) -> None:
        df = pd.DataFrame({"lang": ["en", "de", "en", "fr"], "score": [1, 2, 3, 4]})
        filters = [
            FilterPredicate(column="lang", op="in", value=["en", "de"]),
            FilterPredicate(column="score", op=">", value=1),
        ]
        result = apply_filters(df, filters)
        assert len(result) == 2  # de(2), en(3)

    def test_empty_filters_noop(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = apply_filters(df, [])
        assert len(result) == 3


class TestChoosePromptTextColumn:
    def test_auto_detect_sentences(self) -> None:
        df = pd.DataFrame({"sentences": ["a", "b"], "other": [1, 2]})
        assert choose_prompt_text_column(df) == "sentences"

    def test_auto_detect_prompt(self) -> None:
        df = pd.DataFrame({"prompt": ["a", "b"], "other": [1, 2]})
        assert choose_prompt_text_column(df) == "prompt"

    def test_sentences_takes_priority(self) -> None:
        df = pd.DataFrame({"sentences": ["a"], "prompt": ["b"]})
        assert choose_prompt_text_column(df) == "sentences"

    def test_override_used(self) -> None:
        df = pd.DataFrame({"custom_col": ["a", "b"], "prompt": ["c", "d"]})
        assert choose_prompt_text_column(df, override="custom_col") == "custom_col"

    def test_override_missing_raises(self) -> None:
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(KeyError, match="custom"):
            choose_prompt_text_column(df, override="custom")

    def test_no_known_column_raises(self) -> None:
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(KeyError, match="Neither"):
            choose_prompt_text_column(df)


class TestExtractTexts:
    def test_string_input(self) -> None:
        assert extract_texts_from_row("hello") == ["hello"]

    def test_list_input(self) -> None:
        assert extract_texts_from_row(["a", "b"]) == ["a", "b"]

    def test_none_input(self) -> None:
        assert extract_texts_from_row(None) == [""]

    def test_empty_string(self) -> None:
        assert extract_texts_from_row("   ") == [""]

    def test_numpy_array(self) -> None:
        arr = np.array(["x", "y"])
        assert extract_texts_from_row(arr) == ["x", "y"]

    def test_list_with_none(self) -> None:
        assert extract_texts_from_row(["a", None, "b"]) == ["a", "b"]

    def test_extract_text_concatenates(self) -> None:
        assert extract_text_from_row(["hello", "world"]) == "hello world"

    def test_extract_text_custom_separator(self) -> None:
        assert extract_text_from_row(["a", "b"], separator="|") == "a|b"


class TestIterRowsForSelection:
    def test_basic_iteration(self) -> None:
        df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
        rows = list(
            iter_rows_for_selection(df, start_index=0, max_samples=3, shuffle_seed=None)
        )
        assert len(rows) == 3
        assert rows[0] == (0, {"x": 10})

    def test_start_index(self) -> None:
        df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
        rows = list(
            iter_rows_for_selection(df, start_index=2, max_samples=2, shuffle_seed=None)
        )
        assert len(rows) == 2
        assert rows[0][0] == 2

    def test_shuffle_seed_deterministic(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        rows1 = list(
            iter_rows_for_selection(df, start_index=0, max_samples=5, shuffle_seed=42)
        )
        rows2 = list(
            iter_rows_for_selection(df, start_index=0, max_samples=5, shuffle_seed=42)
        )
        assert rows1 == rows2

    def test_different_seeds_different_order(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        rows1 = list(
            iter_rows_for_selection(df, start_index=0, max_samples=10, shuffle_seed=1)
        )
        rows2 = list(
            iter_rows_for_selection(df, start_index=0, max_samples=10, shuffle_seed=2)
        )
        # Extremely unlikely to be the same with different seeds
        values1 = [r[1]["x"] for r in rows1]
        values2 = [r[1]["x"] for r in rows2]
        assert values1 != values2

    def test_max_samples_none_returns_all(self) -> None:
        df = pd.DataFrame({"x": range(5)})
        rows = list(
            iter_rows_for_selection(
                df, start_index=0, max_samples=None, shuffle_seed=None
            )
        )
        assert len(rows) == 5


class TestIterBalancedTokenCountRows:
    def test_basic_balanced(self) -> None:
        df = pd.DataFrame({
            "text": ["a", "bb", "ccc", "dd", "e", "fff"],
            "token_count": [1, 2, 3, 2, 1, 3],
        })
        rows = iter_balanced_token_count_rows(
            df,
            token_counts=[1, 2],
            samples_per_token_count=2,
            start_index=0,
            max_samples=None,
            shuffle_seed=None,
            allow_partial_buckets=False,
        )
        assert len(rows) == 4  # 2 per bucket * 2 buckets

    def test_partial_buckets_raises_if_not_allowed(self) -> None:
        df = pd.DataFrame({"text": ["a"], "token_count": [1]})
        with pytest.raises(ValueError, match="Only 1 rows"):
            iter_balanced_token_count_rows(
                df,
                token_counts=[1],
                samples_per_token_count=5,
                start_index=0,
                max_samples=None,
                shuffle_seed=None,
                allow_partial_buckets=False,
            )

    def test_partial_buckets_allowed(self) -> None:
        df = pd.DataFrame({"text": ["a"], "token_count": [1]})
        rows = iter_balanced_token_count_rows(
            df,
            token_counts=[1],
            samples_per_token_count=5,
            start_index=0,
            max_samples=None,
            shuffle_seed=None,
            allow_partial_buckets=True,
        )
        assert len(rows) == 1

    def test_missing_token_count_column_raises(self) -> None:
        df = pd.DataFrame({"text": ["a"]})
        with pytest.raises(KeyError, match="token_count"):
            iter_balanced_token_count_rows(
                df,
                token_counts=[1],
                samples_per_token_count=1,
                start_index=0,
                max_samples=None,
                shuffle_seed=None,
                allow_partial_buckets=False,
            )

    def test_custom_token_count_col(self) -> None:
        df = pd.DataFrame({"text": ["a", "b"], "n_tokens": [1, 2]})
        rows = iter_balanced_token_count_rows(
            df,
            token_counts=[1],
            samples_per_token_count=1,
            start_index=0,
            max_samples=None,
            shuffle_seed=None,
            allow_partial_buckets=False,
            token_count_col="n_tokens",
        )
        assert len(rows) == 1
