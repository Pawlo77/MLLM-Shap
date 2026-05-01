"""Similarity computation and model inference utilities for faithfulness evaluation."""

import difflib
from typing import Any

from mllm_shap.connectors.config import ModelConfig
from mllm_shap.shap.embeddings import MeanReducer
from mllm_shap.shap.similarity import CosineSimilarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine

from experiments.mllm_shapx.src.constants import InputModality
from experiments.mllm_shapx.src.factory import build_chat


def embedding_similarities(
    model: Any, base: Any, others: list[Any]
) -> tuple[float, list[float]]:
    """Compute cosine similarities in model embedding space against a base output."""
    embeddings = model.get_static_embeddings([base, *others])
    reduced = MeanReducer()(embeddings)
    sims = CosineSimilarity()(base=reduced[0], other=reduced)
    return float(sims[0].item()), [float(s.item()) for s in sims[1:]]


def tfidf_similarities(
    model: Any, base: Any, others: list[Any]
) -> tuple[float, list[float]]:
    """Compute TF-based cosine similarity between base and other responses.

    Uses use_idf=False to avoid meaningless IDF on a micro-corpus of only a
    few documents. Uses a word-level tokenizer with a pattern that strips
    subword artifacts (e.g., SentencePiece '▁' prefixes).
    """
    chat = model.get_new_chat()
    texts = [
        chat.decode_text(text_tokens=resp.generated_text_tokens)
        for resp in [base, *others]
    ]
    vectorizer = TfidfVectorizer(
        use_idf=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return 1.0, [1.0] * len(others)
    sims = _sklearn_cosine(matrix[0:1], matrix[1:])[0]
    return 1.0, [float(s) for s in sims]


def sequence_match_similarities(
    model: Any, base: Any, others: list[Any]
) -> tuple[float, list[float]]:
    """Compute normalized sequence-matching similarity (SequenceMatcher ratio).

    This is an independent metric from cosine similarity — it measures the
    longest common subsequence ratio between decoded text outputs, breaking
    the circularity of using the same embedding-based metric for both SHAP
    computation and faithfulness evaluation.
    """
    chat = model.get_new_chat()
    base_text = chat.decode_text(text_tokens=base.generated_text_tokens)
    other_texts = [
        chat.decode_text(text_tokens=resp.generated_text_tokens) for resp in others
    ]
    sims = [
        difflib.SequenceMatcher(None, base_text, other_text).ratio()
        for other_text in other_texts
    ]
    return 1.0, sims


def generate_response(
    model: Any,
    audio_bytes: bytes,
    input_modality: InputModality,
    max_new_tokens: int,
    text_temperature: float,
    user_texts: list[str] | None = None,
) -> Any:
    """Generate one model response for provided audio bytes and prompt texts."""
    chat = build_chat(
        model,
        user_texts=user_texts,
        audio_bytes_list=[audio_bytes],
        input_modality=input_modality,
    )
    return model.generate(
        chat=chat,
        max_new_tokens=max_new_tokens,
        model_config=ModelConfig(text_temperature=float(text_temperature)),
        keep_history=False,
    )
