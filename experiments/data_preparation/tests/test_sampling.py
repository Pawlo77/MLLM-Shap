"""Tests for stratified sampling helpers."""

import pandas as pd

from src.sampling import (
    sample_fraction_by_group,
    sample_n_per_group,
    stratified_sample,
)


def test_stratified_sample_returns_all_when_pool_smaller_than_target() -> None:
    pool = pd.DataFrame({"token_count": [1, 2, 3], "text": ["a", "b", "c"]})
    result = stratified_sample(pool, n_target=10, random_state=0)
    assert len(result) == 3
    assert set(result["text"]) == {"a", "b", "c"}


def test_stratified_sample_respects_target_size() -> None:
    pool = pd.DataFrame({
        "token_count": [1] * 5 + [2] * 5,
        "id": range(10),
    })
    result = stratified_sample(pool, n_target=6, random_state=0)
    assert len(result) == 6
    assert result["token_count"].nunique() == 2


def test_stratified_sample_is_deterministic() -> None:
    pool = pd.DataFrame({"token_count": list(range(6)), "id": range(6)})
    a = stratified_sample(pool, n_target=4, random_state=42)
    b = stratified_sample(pool, n_target=4, random_state=42)
    pd.testing.assert_frame_equal(a, b)


def test_sample_n_per_group_caps_per_language() -> None:
    df = pd.DataFrame({
        "language": ["en"] * 4 + ["fr"] * 2,
        "value": range(6),
    })
    result = sample_n_per_group(df, "language", n=2, random_state=0)
    assert len(result) == 4
    assert result.groupby("language").size().max() == 2


def test_sample_fraction_by_group() -> None:
    df = pd.DataFrame({"group": ["a"] * 10 + ["b"] * 10, "value": range(20)})
    result = sample_fraction_by_group(df, "group", frac=0.5, random_state=0)
    assert 8 <= len(result) <= 12
