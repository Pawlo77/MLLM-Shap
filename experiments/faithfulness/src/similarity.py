"""Similarity computation and model inference utilities for faithfulness evaluation."""

import difflib
from copy import deepcopy
from typing import Any, cast

import torch
import torch.nn.functional as F
from mllm_shap.connectors.config import ModelConfig
from mllm_shap.connectors.enums import Role
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


@torch.no_grad()
def text_logprob_score(
    model: Any,
    audio_bytes: bytes,
    input_modality: InputModality,
    target_text_tokens: torch.Tensor,
    user_texts: list[str] | None = None,
) -> float:
    """Mean per-token log-probability of a fixed reference text response,
    conditioned on the provided (possibly perturbed) audio prompt.

    This is a *graded*, non-saturating utility for faithfulness: it teacher-forces
    the reference response's text tokens through the LFM2 backbone (mirroring
    ``generate_sequential``) and returns ``logP(reference_text | audio)`` averaged
    over tokens (in nats). Removing salient audio should lower this score.
    """
    target = target_text_tokens.detach().reshape(-1)
    if target.numel() == 0:
        raise ValueError("Reference response has no text tokens to score.")

    chat = build_chat(
        model,
        user_texts=user_texts,
        audio_bytes_list=[audio_bytes],
        input_modality=input_modality,
    )
    chat = deepcopy(chat)
    chat.new_turn(Role.ASSISTANT)

    backbone = model.model  # LFM2AudioModel
    device = model.device
    in_emb = backbone._prefill(**cast(dict[str, Any], chat))
    cache = None
    total_logprob = 0.0
    for tok in target.tolist():
        lfm_out = backbone.lfm(
            inputs_embeds=in_emb, past_key_values=cache, use_cache=True
        )
        cache = lfm_out.past_key_values
        logits = F.linear(
            lfm_out.last_hidden_state[0, -1], backbone.lfm.embed_tokens.weight
        )
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        total_logprob += float(log_probs[int(tok)])
        next_token = torch.tensor([int(tok)], device=device)
        in_emb = backbone.lfm.embed_tokens(next_token)[None, :]

    return total_logprob / int(target.numel())
