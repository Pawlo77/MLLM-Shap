"""POS tagging helpers for the sanity/insight test suite.

This module is intentionally strict: it does not attempt to install packages or
models at runtime (that breaks offline runs and triggers Bandit/pylint issues).
"""

# pylint: disable=too-many-locals
# pylint: disable=magic-value-comparison

from __future__ import annotations

from typing import Any

import pandas as pd
import spacy
from spacy.util import is_package

from sanity_suite_core import canonicalize_pos


_POS_LANG_TO_SPACY_MODEL: dict[str, str] = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "pl": "pl_core_news_sm",
}


def _load_spacy_model_for_language(lang: str):
    """Return a spaCy pipeline for `lang`.

    Raises a RuntimeError with a manual install hint if missing.
    """

    lang_s = str(lang).strip().lower()
    model = _POS_LANG_TO_SPACY_MODEL.get(lang_s)
    if not model:
        raise RuntimeError(
            f"No spaCy model mapping for language={lang!r}. Add it to _POS_LANG_TO_SPACY_MODEL in sanity_suite_pos.py"
        )

    if not is_package(model):
        raise RuntimeError(
            f"spaCy model '{model}' is not installed (needed for language={lang_s}). "
            f"Run: python -m spacy download {model}"
        )

    return spacy.load(model)


def ensure_pos_tags(df_tokens: pd.DataFrame) -> pd.DataFrame:
    """Populate df_tokens['pos'] for current text using spaCy.

    Tags at word-level (based on word_id/word) then copies POS back to tokens.
    """

    df = df_tokens.copy()
    if "pos" not in df.columns:
        df["pos"] = None

    mask = (df["modality"].eq("text")) & (df["turn"].eq("current"))
    if not mask.any():
        return df

    if "word_id" not in df.columns or "word" not in df.columns:
        raise ValueError("ensure_pos_tags requires word_id/word; call add_word_features first")

    langs = sorted([x for x in df.loc[mask, "language"].dropna().astype(str).unique().tolist() if x])
    cache = {lang: _load_spacy_model_for_language(lang) for lang in langs}

    for (lang, _case, _sample_id), group in df[mask].groupby(["language", "case", "sample_id"], sort=False):
        lang_s = str(lang).lower() if lang is not None else ""
        if not lang_s or lang_s not in cache:
            continue

        nlp = cache[lang_s]
        words_df = (
            group.dropna(subset=["word_id", "word"])
            .sort_values("position")
            .drop_duplicates(subset=["word_id"], keep="first")
        )
        if words_df.empty:
            continue

        words = words_df["word"].astype(str).tolist()
        doc = nlp(" ".join(words))

        # If tokenization doesn't match our word list, skip this sample.
        if len(doc) != len(words):
            continue

        word_ids = words_df["word_id"].astype(int).tolist()
        pos_by_word_id: dict[int, Any] = {
            int(word_ids[i]): canonicalize_pos(doc[i].pos_) for i in range(len(word_ids))
        }

        idx = group.index
        mapped = df.loc[idx, "word_id"].astype("Int64").map(pos_by_word_id)
        df.loc[idx, "pos"] = mapped

    df["pos"] = df["pos"].map(canonicalize_pos)
    return df
