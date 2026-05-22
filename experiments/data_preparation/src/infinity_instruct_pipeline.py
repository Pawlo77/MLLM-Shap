"""Infinity-Instruct multilingual dataset preparation."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset
from semhash import SemHash
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
from .audio import calculate_audio_duration
from .reporting import value_counts_at_least
from .constants import TTSConfig
from .languages import LanguageClassifier, LanguageTranslator
from .nlp import TTS, split_into_sentences
from .sampling import sample_n_per_group
from .tokens import compute_multi_turn_token_counts


def build_infinity_dataframe(train_split: Dataset) -> pd.DataFrame:
    """Build the raw Infinity-Instruct table from the HF train split."""
    return pd.DataFrame(
        [
            [
                e["conversations"],
                e["label"].get("ability_en") if isinstance(e["label"], dict) else None,
                e["langdetect"],
                e["source"],
                e["id"],
            ]
            for e in train_split
        ],
        columns=["conversation", "labels", "language", "source", "id"],
    )


def languages_with_min_population(
    df: pd.DataFrame,
    language_col: str = "language",
    minimum: int = 1000,
) -> pd.DataFrame:
    """Return language value counts with at least *minimum* rows."""
    return value_counts_at_least(df[language_col], minimum)


def filter_infinity_languages(
    df: pd.DataFrame,
    languages: set[str],
) -> pd.DataFrame:
    """Keep configured languages and drop id/source metadata columns."""
    out = df[df["language"].isin(languages)].reset_index(drop=True)
    out["speakers"] = out["conversation"].apply(lambda x: {el["from"] for el in x})
    print(
        f"Percentage of unique IDs: "
        f"{(len(out.drop_duplicates(subset=['id'])) / len(out)) * 100:.2f}%"
    )
    print(f"Max speakers in a conversation: {out['speakers'].map(len).max()}")
    print(f"Number of unique sources: {out['source'].nunique()}")
    print(f"Dataframe size after language filtering: {len(out)}.")
    return out.drop(columns=["speakers", "id", "source"])


def message_count_distribution(
    df: pd.DataFrame,
    messages_col: str = "n_messages",
    minimum: int = 50,
) -> pd.DataFrame:
    """Return message-count histogram tail (counts >= *minimum*)."""
    return value_counts_at_least(df[messages_col].astype(int), minimum)


def filter_max_turns(df: pd.DataFrame, max_turns: int) -> pd.DataFrame:
    """Drop conversations exceeding *max_turns* user/assistant pairs."""
    out = df.copy()
    out["n_messages"] = out["conversation"].apply(len)
    before = len(out)
    out = out[out["n_messages"] <= 2 * max_turns].reset_index(drop=True)
    print(
        f"Remaining samples after filtering long prompts: {len(out)} (removed {before - len(out)})"
    )
    return out


def plot_ability_exclusion_curve(
    unique_keys: np.ndarray,
    abilities_to_exclude: set[str],
    embedding_model: SentenceTransformer,
    elbow_threshold: float = 0.4,
) -> dict[float, set[str]]:
    """Plot label exclusion counts by threshold; return keys to exclude per threshold."""
    term_embeddings: Tensor = embedding_model.encode(
        unique_keys, convert_to_tensor=True
    )
    exclude_embeddings: Tensor = embedding_model.encode(
        list(abilities_to_exclude), convert_to_tensor=True
    )
    similarity_matrix: Tensor = util.cos_sim(term_embeddings, exclude_embeddings)

    keys_to_exclude__by_threshold: dict[float, set[str]] = {}
    thresholds: list[float] = np.arange(0.1, 0.91, 0.1).round(2).tolist()

    for threshold in thresholds:
        keys_to_exclude__by_threshold[threshold] = set(
            str(x)
            for x in unique_keys[
                np.where(similarity_matrix.max(dim=1).values.cpu().numpy() > threshold)[
                    0
                ]
            ]
        )

    counts = [len(keys) for keys in keys_to_exclude__by_threshold.values()]
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=thresholds, y=counts, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Number of keys to exclude")
    plt.title("Keys Exclusion by Threshold")
    plt.axvline(x=elbow_threshold, color="red", linestyle="--", label="Elbow")
    plt.legend()
    plt.show()
    return keys_to_exclude__by_threshold


def filter_by_allowed_abilities(
    df: pd.DataFrame,
    keys_to_include: set[str],
) -> pd.DataFrame:
    """Keep rows whose label lists only contain allowed ability keys."""
    before = len(df)
    out = df[
        df["labels"].apply(
            lambda x, allowed=keys_to_include: all(el in allowed for el in x)
        )
    ].reset_index(drop=True)
    print(
        f"Remaining samples after filtering abilities: {len(out)} "
        f"(removed {before - len(out)}, {((before - len(out)) / before) * 100:.2f}%)"
    )
    return out


def dedupe_conversations_semhash(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and filter outliers on joined conversation text."""
    out = df.copy()
    out["conversation__joined"] = (
        out["conversation"].apply(lambda x: [el["value"] for el in x]).str.join(" ")
    )

    semhash = SemHash.from_records(records=out["conversation__joined"].tolist())
    dedup_texts = set(semhash.self_deduplicate().selected)
    filtered_texts = {e["text"] for e in semhash.self_filter_outliers().selected}
    representative_texts = dedup_texts.intersection(filtered_texts)

    before = len(out)
    out = out[out["conversation__joined"].isin(representative_texts)].reset_index(
        drop=True
    )
    print(
        f"Number of unique texts after deduplication: {len(representative_texts)} "
        f"(removed {before - len(representative_texts)})"
    )
    return out


def verify_non_english_languages(
    df: pd.DataFrame,
    classifier: LanguageClassifier,
    english_code: str = "en",
    sample_chars: int = 500,
) -> pd.DataFrame:
    """Drop rows where langdetect disagrees with a classifier on non-English text."""
    out = df.copy()
    out["detected_lang"] = None
    non_en_mask = out["language"] != english_code
    out.loc[non_en_mask, "detected_lang"] = out.loc[
        non_en_mask, "conversation__joined"
    ].progress_apply(lambda x: classifier.classify_language(x[:sample_chars]))

    incorrect_mask = (out["language"] != out["detected_lang"]) & out[
        "detected_lang"
    ].notna()
    print(
        f"Number of incorrect language detections: {incorrect_mask.sum()} "
        f"({(incorrect_mask.sum() / non_en_mask.sum()) * 100:.2f}%)"
    )
    return out.drop(index=out.index[incorrect_mask]).drop(columns=["detected_lang"])


def split_conversation_into_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Split each turn value into sentences; add ``sentences__num``."""
    out = df.copy()
    out["conversation"] = out["conversation"].apply(
        lambda x: [
            {"from": e["from"], "value": split_into_sentences(e["value"].strip())}
            for e in x
            if e["value"].strip() != ""
        ]
    )
    out["sentences__num"] = out["conversation"].apply(
        lambda x: sum(len(e["value"]) for e in x)
    )
    return out


def filter_max_sentence_length(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """Drop conversations containing a sentence longer than *max_length* characters."""
    before = len(df)
    out = df.copy()
    out["max_sentence_length"] = out["conversation"].apply(
        lambda x: (
            max(len(sentence) for turn in x for sentence in turn["value"]) if x else 0
        )
    )
    out = out[out["max_sentence_length"] <= max_length].reset_index(drop=True)
    out.drop(columns=["max_sentence_length"], inplace=True)
    print(
        f"Remaining samples after max sentence length filter: {len(out)} "
        f"(removed {before - len(out)}, {((before - len(out)) / before) * 100:.2f}%)"
    )
    return out


def build_multilingual_base(
    df: pd.DataFrame,
    per_language_n: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample per language and reshape to multilingual schema."""
    out = sample_n_per_group(
        df, "language", per_language_n, random_state=random_state
    ).rename(columns={"conversation": "prompt"})
    out["sentences"] = out["prompt"].apply(lambda x: x[0]["value"])
    out["original_language"] = out["language"]
    out["n_messages"] -= 1
    out["sentences__num"] = out["sentences"].apply(len)
    out["conversation__joined"] = out["sentences"].str.join(" ")
    print(
        f"Number of unique text entries and size of the multilingual dataset: {len(out)}"
    )
    return out


def filter_by_token_count(
    df: pd.DataFrame,
    model: Any,
    max_token_count: int,
    sentences_column: str = "sentences",
) -> pd.DataFrame:
    """Add token counts and keep rows within budget."""
    out = df.copy()
    out["token_count"] = compute_multi_turn_token_counts(
        out, model=model, sentences_column=sentences_column
    )
    print(f"Computed token counts for {len(out)} entries.")
    out = out[out["token_count"] <= max_token_count].copy()
    print(f"Candidates with token_count <= {max_token_count}: {len(out)}")
    return out


async def augment_with_translations(
    df: pd.DataFrame,
    translator: LanguageTranslator,
    languages: set[str],
) -> pd.DataFrame:
    """Add cross-language translated copies of each language subset."""
    translated_parts: list[pd.DataFrame] = []

    for language in languages:
        print(f"Translating from {language}...")
        to_translate = df[df["language"] == language].copy()

        for target_language in languages:
            if target_language == language:
                continue
            part = to_translate.copy()
            part["language"] = target_language
            part = await translator.translate_df(part, target_language)
            part["sentences"] = part["conversation__joined"].apply(split_into_sentences)
            part["sentences__num"] = part["sentences"].apply(len)
            translated_parts.append(part)

    if not translated_parts:
        return df
    return pd.concat([df, *translated_parts]).reset_index(drop=True)


async def synthesize_multilingual_voices(
    df: pd.DataFrame,
    tts: TTS,
    tts_configs: dict[str, dict[str, TTSConfig]],
    sentences_column: str = "sentences",
) -> pd.DataFrame:
    """Synthesize male/female audio per language row."""
    out = df.copy()
    out["audio__male"] = None
    out["audio__female"] = None

    for language, configs in tts_configs.items():
        mask = out["language"] == language
        if not mask.any():
            continue

        part = out.loc[mask, [sentences_column]].copy()
        for name, config in (("male", configs["male"]), ("female", configs["female"])):
            part = await tts.synthesize_df_from_config(
                part,
                config,
                column_to_synthesize=sentences_column,
                target_column=f"audio__{name}",
            )
            out.loc[mask, f"audio__{name}"] = part[f"audio__{name}"]

        del part

    out["audio__male__duration"] = out["audio__male"].progress_apply(
        calculate_audio_duration
    )
    out["audio__female__duration"] = out["audio__female"].progress_apply(
        calculate_audio_duration
    )
    return out


def finalize_multilingual_for_save(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed in the published parquet."""
    return df.drop(columns=["conversation__joined", "prompt"])
