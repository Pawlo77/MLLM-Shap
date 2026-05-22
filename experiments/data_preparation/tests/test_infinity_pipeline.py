"""Tests for Infinity-Instruct pipeline helpers (no SemHash / TTS)."""

import pandas as pd
from datasets import Dataset

from src.infinity_instruct_pipeline import (
    build_infinity_dataframe,
    build_multilingual_base,
    filter_by_allowed_abilities,
    filter_infinity_languages,
    filter_max_sentence_length,
    filter_max_turns,
    finalize_multilingual_for_save,
    split_conversation_into_sentences,
    verify_non_english_languages,
)


def _sample_infinity_rows() -> list[dict]:
    return [
        {
            "conversations": [
                {"from": "human", "value": "Short question one."},
                {"from": "gpt", "value": "Short answer one."},
            ],
            "label": {"ability_en": ["translation", "dialogue"]},
            "langdetect": "en",
            "source": "src-a",
            "id": "id-1",
        },
        {
            "conversations": [
                {"from": "human", "value": "Question deux en français."},
                {"from": "gpt", "value": "Réponse courte."},
            ],
            "label": {"ability_en": ["translation"]},
            "langdetect": "fr",
            "source": "src-b",
            "id": "id-2",
        },
    ]


def test_build_infinity_dataframe() -> None:
    train = Dataset.from_list(_sample_infinity_rows())
    df = build_infinity_dataframe(train)
    assert list(df.columns) == ["conversation", "labels", "language", "source", "id"]
    assert len(df) == 2
    assert df.loc[0, "language"] == "en"


def test_filter_infinity_languages() -> None:
    train = Dataset.from_list(_sample_infinity_rows())
    df = build_infinity_dataframe(train)
    filtered = filter_infinity_languages(df, {"en", "fr"})
    assert set(filtered["language"]) <= {"en", "fr"}
    assert "id" not in filtered.columns


def test_filter_max_turns_drops_long_conversations() -> None:
    df = pd.DataFrame({
        "conversation": [
            [{"from": "human", "value": "a"}],
            [{"from": "human", "value": "b"}] * 10,
        ]
    })
    result = filter_max_turns(df, max_turns=2)
    assert len(result) == 1


def test_filter_by_allowed_abilities() -> None:
    df = pd.DataFrame({
        "labels": [
            ["translation", "dialogue"],
            ["mathematics"],
        ]
    })
    allowed = {"translation", "dialogue"}
    result = filter_by_allowed_abilities(df, allowed)
    assert len(result) == 1


def test_split_conversation_into_sentences() -> None:
    df = pd.DataFrame({
        "conversation": [
            [
                {
                    "from": "human",
                    "value": "Hello there. Second sentence.",
                }
            ]
        ]
    })
    result = split_conversation_into_sentences(df)
    assert result["sentences__num"].iloc[0] >= 2


def test_filter_max_sentence_length() -> None:
    df = pd.DataFrame({
        "conversation": [
            [{"from": "human", "value": ["short"]}],
            [{"from": "human", "value": ["x" * 50]}],
        ]
    })
    result = filter_max_sentence_length(df, max_length=20)
    assert len(result) == 1


class _StubClassifier:
    def classify_language(self, text: str) -> str:
        return "fr" if "français" in text else "en"


def test_verify_non_english_languages_drops_mismatches() -> None:
    df = pd.DataFrame({
        "language": ["fr", "fr"],
        "conversation__joined": [
            "Texte en français correct.",
            "This text is actually English.",
        ],
    })
    result = verify_non_english_languages(df, _StubClassifier(), english_code="en")
    assert len(result) == 1
    assert result["language"].iloc[0] == "fr"


def test_build_multilingual_base_and_finalize() -> None:
    df = pd.DataFrame({
        "language": ["en", "en", "fr", "fr", "es"],
        "conversation": [
            [{"from": "human", "value": ["English one."]}],
            [{"from": "human", "value": ["English two."]}],
            [{"from": "human", "value": ["French one."]}],
            [{"from": "human", "value": ["French two."]}],
            [{"from": "human", "value": ["Spanish one."]}],
        ],
        "n_messages": [2, 2, 2, 2, 2],
    })
    base = build_multilingual_base(df, per_language_n=2, random_state=0)
    assert len(base) == 5
    assert "conversation__joined" in base.columns

    finalized = finalize_multilingual_for_save(base)
    assert "prompt" not in finalized.columns
    assert "conversation__joined" not in finalized.columns
