"""Multilingual experiments analysis utilities."""

# pylint: disable=duplicate-code

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
import spacy
from spacy.util import is_package

from analysis_common import LoadedCase, build_units_dataframe, ensure_language_columns, load_case_results

LANGUAGE_COL: str = "language"
ORIGINAL_LANGUAGE_COL: str = "original_language"


DEFAULT_RUN: str = "multi_lingual_2026_01_03"

CASES: dict[str, str] = {
    # Keep names aligned with `single_sentence.py` where possible.
    "T2T": "text_text_limited_neyman_lin3_0",
    "SM2T": "audio_male_text_limited_neyman_lin3_0",
    "SF2T": "audio_female_text_limited_neyman_lin3_0",
}


def load_experiments_results(case: Literal["T2T", "SM2T", "SF2T"], run: str = DEFAULT_RUN) -> pd.DataFrame:
    """Load multilingual experiment results for a given case."""

    loaded: LoadedCase = load_case_results(run_name=run, case_dir=CASES[case])
    df = loaded.df.copy()
    df["case"] = case
    df["run"] = run

    # Multilingual runs should have these; keep as nullable columns if missing.
    df = ensure_language_columns(df)
    _ = (LANGUAGE_COL, ORIGINAL_LANGUAGE_COL)
    return df


def build_units_df(case: Literal["T2T", "SM2T", "SF2T"], run: str = DEFAULT_RUN) -> pd.DataFrame:
    """Build a long-form explainable-units DataFrame."""

    results_df = load_experiments_results(case=case, run=run)
    units_df = build_units_dataframe(results_df)
    units_df["case"] = case
    units_df["run"] = run
    return units_df


LANGS: list[str] = ["en", "es", "fr"]
POS_ORDER: list[str] = ["Noun", "Verb", "Adj", "Det", "Punct"]

LANG_EN: str = "en"
LANG_ES: str = "es"
LANG_FR: str = "fr"

UNIT_TYPE_TEXT: str = "text"

POS_NOUN: str = "Noun"
POS_VERB: str = "Verb"
POS_ADJ: str = "Adj"
POS_DET: str = "Det"
POS_PUNCT: str = "Punct"


def _load_pos_tagger(language: str):
    """Load a spaCy POS tagger for a language, falling back to multilingual model when available."""

    model_by_lang = {
        LANG_EN: "en_core_web_sm",
        LANG_ES: "es_core_news_sm",
        LANG_FR: "fr_core_news_sm",
    }
    name = model_by_lang.get(language)
    if name and is_package(name):
        return spacy.load(name, disable=["ner", "lemmatizer", "attribute_ruler"])
    if is_package("xx_sent_ud_sm"):
        return spacy.load("xx_sent_ud_sm", disable=["ner", "lemmatizer", "attribute_ruler"])
    return None


DET_BY_LANG: dict[str, set[str]] = {
    "en": {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"},
    "es": {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "este",
        "esta",
        "estos",
        "estas",
        "mi",
        "tu",
        "su",
        "nuestro",
        "nuestra",
        "vuestro",
        "vuestra",
    },
    "fr": {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "ce",
        "cet",
        "cette",
        "ces",
        "mon",
        "ma",
        "mes",
        "ton",
        "ta",
        "tes",
        "son",
        "sa",
        "ses",
        "notre",
        "votre",
        "leur",
    },
}


AUX_BY_LANG: dict[str, set[str]] = {
    "en": {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
    },
    "es": {"es", "son", "fue", "fueron", "ser", "estar", "ha", "han", "haber", "puede", "poder", "debe", "deber"},
    "fr": {"est", "sont", "être", "a", "ont", "avoir", "peut", "pouvoir", "doit", "devoir"},
}


PUNCT_RE = re.compile(r"^\\W+$", re.UNICODE)
ALPHA_RE = re.compile(r"^[\\W\\d_]+$", re.UNICODE)

EN_ADJ_SUFFIX: tuple[str, ...] = ("y", "ful", "ous", "ive", "al", "ic", "less", "able", "ible", "ish", "ary")
ES_ADJ_SUFFIX: tuple[str, ...] = (
    "oso",
    "osa",
    "osos",
    "osas",
    "able",
    "ables",
    "ible",
    "ibles",
    "al",
    "ales",
    "ico",
    "ica",
    "icos",
    "icas",
    "ario",
    "aria",
    "arios",
    "arias",
    "ante",
    "entes",
    "ente",
)
FR_ADJ_SUFFIX: tuple[str, ...] = (
    "eux",
    "euse",
    "euses",
    "ables",
    "able",
    "ibles",
    "ible",
    "al",
    "ale",
    "ales",
    "ique",
    "iques",
    "if",
    "ive",
    "ifs",
    "ives",
    "ant",
    "ante",
    "ants",
    "antes",
)


def _pos_group_from_spacy_pos(pos_: str) -> str | None:
    mapping = {
        "NOUN": POS_NOUN,
        "PROPN": POS_NOUN,
        "VERB": POS_VERB,
        "AUX": POS_VERB,
        "ADJ": POS_ADJ,
        "DET": POS_DET,
        "PUNCT": POS_PUNCT,
    }
    return mapping.get(pos_)


def _looks_like_verb(token: str, language: str) -> bool:
    if language == LANG_EN:
        return token.endswith("ing") or token.endswith("ed")
    if language == LANG_ES:
        return token.endswith(("ar", "er", "ir", "ando", "iendo"))
    if language == LANG_FR:
        return token.endswith(("er", "ir", "re", "ant"))
    return False


def _looks_like_adj(token: str, language: str) -> bool:
    if language == LANG_EN:
        return token.endswith(EN_ADJ_SUFFIX)
    if language == LANG_ES:
        return token.endswith(ES_ADJ_SUFFIX)
    if language == LANG_FR:
        return token.endswith(FR_ADJ_SUFFIX)
    return False


def _pos_group_heuristic(token: str, language: str) -> str | None:
    cleaned = token.strip().lower()
    if not cleaned:
        return None

    pos_group: str | None
    if PUNCT_RE.match(cleaned):
        pos_group = POS_PUNCT
    elif cleaned in DET_BY_LANG.get(language, set()):
        pos_group = POS_DET
    elif cleaned in AUX_BY_LANG.get(language, set()):
        pos_group = POS_VERB
    elif ALPHA_RE.match(cleaned):
        pos_group = None
    elif _looks_like_verb(cleaned, language):
        pos_group = POS_VERB
    elif _looks_like_adj(cleaned, language):
        pos_group = POS_ADJ
    else:
        pos_group = POS_NOUN

    return pos_group


def _decode_pieces(token_pieces: list[str]) -> str:
    return "".join(token_pieces)


def _piece_char_spans(token_pieces: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cur = 0
    for piece in token_pieces:
        start = cur
        cur += len(piece)
        spans.append((start, cur))
    return spans


def _iter_pos_rows_heuristic(
    language: str,
    language_str: str,
    token_pieces: list[str],
    sv_values: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {"language": language, "pos": pos_group, "abs_sv": float(abs(value))}
        for piece, value in zip(token_pieces, sv_values)
        for pos_group in [_pos_group_heuristic(piece, language_str)]
        if pos_group is not None and pos_group in POS_ORDER
    ]


def _iter_pos_rows_spacy(
    language: str,
    token_pieces: list[str],
    sv_values: np.ndarray,
    nlp_pos,
) -> list[dict[str, object]]:
    doc = nlp_pos(_decode_pieces(token_pieces))

    rows_local: list[dict[str, object]] = []
    for (start, end), value in zip(_piece_char_spans(token_pieces), sv_values):
        tok = next(
            (
                t
                for t in doc
                if t.idx <= ((start + end) // 2) < (t.idx + len(t))
            ),
            None,
        )
        if tok is None:
            continue
        pos_group = _pos_group_from_spacy_pos(tok.pos_)
        if pos_group is None or pos_group not in POS_ORDER:
            continue
        rows_local.append({"language": language, "pos": pos_group, "abs_sv": float(abs(value))})

    return rows_local


def _iter_pos_rows_for_sample(row: pd.Series, *, langs: set[str]) -> list[dict[str, object]]:
    language = row.get(LANGUAGE_COL)
    if language not in langs:
        return []

    language_str = str(language)
    token_pieces = list(row["tokens"])
    sv_values = np.asarray(row["sv"], dtype=float)
    nlp_pos = _load_pos_tagger(language_str)

    if nlp_pos is None:
        return _iter_pos_rows_heuristic(language, language_str, token_pieces, sv_values)

    return _iter_pos_rows_spacy(language, token_pieces, sv_values, nlp_pos)


def compute_pos_breakdown(samples_df: pd.DataFrame, *, langs: list[str] | None = None) -> pd.DataFrame:
    """Compute mean $|SV|$ by coarse POS group for EN/ES/FR text tokens.

    Expects a `samples_df` with at least columns: `unit_type`, `language`, `tokens`, `sv`.
    """

    langs_set = set(langs or LANGS)
    rows_local: list[dict[str, object]] = []

    text_rows = samples_df[samples_df["unit_type"] == UNIT_TYPE_TEXT]
    for _, row in text_rows.iterrows():
        rows_local.extend(_iter_pos_rows_for_sample(row, langs=langs_set))

    pos_df = pd.DataFrame(rows_local)
    if pos_df.empty:
        raise RuntimeError("POS breakdown produced no rows. Check language filters and token decoding.")

    pos_summary_df = pos_df.groupby(["pos", "language"], as_index=False).agg(
        mean_abs_sv=("abs_sv", "mean"),
        n=("abs_sv", "size"),
    )
    pos_summary_df["pos"] = pd.Categorical(pos_summary_df["pos"], categories=POS_ORDER, ordered=True)
    pos_summary_df["language"] = pos_summary_df["language"].astype(str).str.upper()

    return pos_summary_df.sort_values(["pos", "language"])
