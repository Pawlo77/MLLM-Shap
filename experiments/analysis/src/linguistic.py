"""Linguistic features definitions and utilities."""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import spacy

ROOT_DEP: str = "ROOT"
"""Dependency label for the root of the sentence."""


LINGUISTIC_FEATURES_DF: pd.DataFrame = pd.DataFrame(
    [
        # --- Core meaning (highest importance) ---
        {"dep": ROOT_DEP, "importance": 1, "group": "core_structure"},
        {"dep": "nsubj", "importance": 1, "group": "core_structure"},
        {"dep": "nsubjpass", "importance": 1, "group": "core_structure"},
        {"dep": "dobj", "importance": 1, "group": "core_structure"},
        {"dep": "agent", "importance": 1, "group": "core_structure"},
        # --- Clause structure (very important) ---
        {"dep": "ccomp", "importance": 2, "group": "clausal_structure"},
        {"dep": "xcomp", "importance": 2, "group": "clausal_structure"},
        {"dep": "advcl", "importance": 2, "group": "clausal_structure"},
        {"dep": "relcl", "importance": 2, "group": "clausal_structure"},
        {"dep": "acl", "importance": 2, "group": "clausal_structure"},
        {"dep": "pcomp", "importance": 2, "group": "clausal_structure"},
        # --- Nominal / verbal complements ---
        {"dep": "attr", "importance": 3, "group": "arguments"},
        {"dep": "acomp", "importance": 3, "group": "arguments"},
        {"dep": "dative", "importance": 3, "group": "arguments"},
        {"dep": "pobj", "importance": 3, "group": "arguments"},
        {"dep": "oprd", "importance": 3, "group": "arguments"},
        # --- Modifiers (less critical, descriptive) ---
        {"dep": "amod", "importance": 4, "group": "modifiers"},
        {"dep": "advmod", "importance": 4, "group": "modifiers"},
        {"dep": "npadvmod", "importance": 4, "group": "modifiers"},
        {"dep": "nmod", "importance": 4, "group": "modifiers"},
        {"dep": "compound", "importance": 4, "group": "modifiers"},
        {"dep": "poss", "importance": 4, "group": "modifiers"},
        {"dep": "appos", "importance": 4, "group": "modifiers"},
        # --- Coordination / structure helpers ---
        {"dep": "conj", "importance": 4, "group": "coordination"},
        {"dep": "cc", "importance": 5, "group": "coordination"},
        # --- Function / grammar glue (lowest importance) ---
        {"dep": "det", "importance": 5, "group": "function_words"},
        {"dep": "case", "importance": 5, "group": "function_words"},
        {"dep": "aux", "importance": 5, "group": "function_words"},
        {"dep": "auxpass", "importance": 5, "group": "function_words"},
        {"dep": "mark", "importance": 5, "group": "function_words"},
        {"dep": "prep", "importance": 5, "group": "function_words"},
        {"dep": "punct", "importance": 6, "group": "punctuation"},
        # --- Generic / unknown ---
        {"dep": "dep", "importance": 4, "group": "generic"},
    ]
).set_index("dep")
"""DataFrame defining linguistic features with their importance levels and groups."""


@lru_cache(maxsize=3)
def get_nlp(language: str):
    """Load the spaCy language model based on the specified language."""
    if language == "en":
        return spacy.load("en_core_web_sm")
    if language == "fr":
        return spacy.load("fr_core_news_sm")
    if language == "es":
        return spacy.load("es_core_news_sm")
    raise ValueError(f"Unsupported language: {language}")


def get_linguistic_stats(prompt: str, language: str = "en") -> dict[str, Any]:
    """Get linguistic stats for prompt using spacy."""
    nlp_prompt = get_nlp(language)(prompt)

    return [
        {
            "text": t.text,
            "dep": t.dep_,
            "pos": t.pos_,
            "children": len(list(t.children)),
        }
        for t in nlp_prompt
    ]


def map_subwords_to_tokens(
    subwords: list[str], tokens_text: list[str], language: str = "en"
) -> list[int]:
    """Map subwords to token indices using cosine similarity of embeddings."""
    nlp = get_nlp(language=language)

    # Compute embeddings
    processed_tokens = [nlp(t.strip()) for t in tokens_text]
    token_vecs = np.array(
        [
            pt.vector if pt.vector.shape[0] > 0 else np.zeros(96) + 1e-9
            for pt in processed_tokens
        ]
    )
    processed_subwords = [nlp(s.strip()) for s in subwords]
    subword_vecs = np.array(
        [
            ps.vector if ps.vector.shape[0] > 0 else np.zeros(96) + 1e-9
            for ps in processed_subwords
        ]
    )

    # Normalize for cosine similarity
    token_vecs_norm = token_vecs / np.linalg.norm(token_vecs, axis=1, keepdims=True)
    subword_vecs_norm = subword_vecs / np.linalg.norm(
        subword_vecs, axis=1, keepdims=True
    )

    # Compute cosine similarity matrix (subwords x tokens)
    sim_matrix = np.dot(subword_vecs_norm, token_vecs_norm.T)

    # Map each subword to token with highest similarity
    mapping = np.argmax(sim_matrix, axis=1)

    return mapping.tolist()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich DataFrame with linguistic feature importance and group."""
    df = df.copy()

    df["stats_matched"] = df.apply(
        lambda row: [row["linguistic_stats"][i] for i in row["stats_mapping"]], axis=1
    )
    df["deps"] = df["stats_matched"].apply(
        lambda stats: [stat["dep"] for stat in stats]
    )
    df["pos"] = df["stats_matched"].apply(lambda stats: [stat["pos"] for stat in stats])
    df["children"] = df["stats_matched"].apply(
        lambda stats: [stat["children"] for stat in stats]
    )

    df.drop(
        columns=["linguistic_stats", "stats_mapping", "stats_matched"], inplace=True
    )

    exploded_df = (
        df.apply(
            lambda row: [
                {
                    "prompt": row["inputs"],
                    "row_index": row["row_index"],
                    "mode": row["mode"],
                    "language": row["language"],
                    "token": token,
                    "id": i,
                    "sv": sv,
                    "dep": dep,
                    "pos": pos,
                    "children": children,
                }
                for i, (token, sv, dep, pos, children) in enumerate(
                    zip(
                        row["tokens"],
                        row["sv"],
                        row["deps"],
                        row["pos"],
                        row["children"],
                    )
                )
            ],
            axis=1,
        )
        .explode()
        .apply(pd.Series)
        .reset_index(drop=True)
    )

    return exploded_df.join(
        LINGUISTIC_FEATURES_DF,
        on="dep",
        how="left",
    )
