"""Quality and deduplication filters.

Provides rule-based and embedding-based filtering utilities used during
dataset curation: simple heuristic filters, interestingness scoring and
greedy semantic deduplication.
"""

import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

ANCHOR_SENTENCES: list[str] = [
    "What is the capital of France?",
    "Can you help me write a short story about a robot?",
    "Explain the concept of machine learning in simple terms.",
    "How do I bake a chocolate cake from scratch?",
    "Tell me a joke about scientists.",
    "What are the main causes of climate change?",
    "Summarise the plot of Romeo and Juliet.",
    "How does the stock market work?",
    "Give me a recipe for a healthy breakfast.",
    "What programming language should I learn first?",
]
"""Anchor sentences used for interestingness scoring.

These are generic, high-quality prompts covering a range of topics and
styles to define a broad "interestingness" direction in the embedding space
for filtering.
"""


def _resolve_encoding_devices(device: str, text_count: int) -> list[str] | None:
    """Choose worker devices for multi-process encoding.

    Args:
        device: Requested execution device from the caller.
        text_count: Number of texts to encode.

    Returns:
        A list of worker devices for multi-process encoding, or ``None``
        when the caller should fall back to single-device inference.
    """
    if device.startswith("cuda:") or device == "mps":
        return None

    if device == "cuda" and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            return [f"cuda:{index}" for index in range(gpu_count)]
        return None

    if device == "cpu":
        cpu_count = os.cpu_count() or 1
        worker_count = min(cpu_count, 8)
        if text_count >= 256 and worker_count > 1:
            return ["cpu"] * worker_count

    return None


def _encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    device: str,
) -> np.ndarray:
    """Encode texts with a device-aware batch or multi-process strategy."""
    worker_devices = _resolve_encoding_devices(device, len(texts))
    if worker_devices:
        pool = model.start_multi_process_pool(target_devices=worker_devices)
        try:
            chunk_size = max(64, len(texts) // (len(worker_devices) * 8))
            return model.encode_multi_process(texts, pool, chunk_size=chunk_size)
        finally:
            model.stop_multi_process_pool(pool)

    encode_kwargs: dict[str, object] = {
        "show_progress_bar": True,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "device": device,
        "batch_size": 1024 if device.startswith("cuda") else 256,
    }
    return model.encode(texts, **encode_kwargs)


def is_interesting_rule_based(text: str) -> bool:
    """Return True for candidate sentences that pass basic quality checks.

    The heuristic checks that the text contains a minimum number of words,
    is not excessively long, does not start with common URL/code markers,
    and has a minimum alphabetic-character ratio.

    Args:
        text (str): Candidate sentence.

    Returns:
        bool: True if the sentence passes the rule-based quality checks.
    """
    text = text.strip()
    words = text.split()
    if len(words) < 4:
        return False
    if len(text) > 300:
        return False
    if text.startswith(("http", "www", "<", "{", "[")):
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.55:
        return False
    return True


def semantic_dedup(
    embs: np.ndarray, threshold: float, scores: np.ndarray
) -> np.ndarray:
    """Perform greedy semantic deduplication based on cosine similarity.

    The algorithm sorts examples by ``scores`` (descending) and keeps the
    highest-scoring example from each similarity cluster where pairwise
    cosine similarity exceeds ``threshold``.

    Args:
        embs (np.ndarray): Array of shape (N, D) with L2-normalised embeddings.
        threshold (float): Cosine-similarity threshold above which two
            examples are considered duplicates.
        scores (np.ndarray): Array of length N with interestingness scores
            used to define the greedy selection order.

    Returns:
        np.ndarray: Sorted indices of the surviving examples.
    """
    order = np.argsort(-scores)
    kept: list[int] = []
    emb_dim = embs.shape[1]
    kept_count = 0
    kept_embs = np.empty((min(len(order), 64), emb_dim), dtype=embs.dtype)

    for idx in tqdm(order, desc="semantic dedup"):
        emb = embs[idx]
        if kept_count:
            sims = kept_embs[:kept_count] @ emb
            if sims.max() >= threshold:
                continue

        if kept_count == kept_embs.shape[0]:
            next_size = max(64, kept_embs.shape[0] * 2)
            expanded = np.empty((next_size, emb_dim), dtype=embs.dtype)
            expanded[:kept_count] = kept_embs[:kept_count]
            kept_embs = expanded

        kept_embs[kept_count] = emb
        kept_count += 1
        kept.append(idx)

    return np.sort(kept)


def nlp_quality_filter(
    df: pd.DataFrame,
    device: str,
    text_col: str = "prompt",
    interestingness_percentile: int = 20,
    similarity_threshold: float = 0.5,
    st_model_name: str = "all-MiniLM-L6-v2",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run a full NLP quality pipeline on a candidate DataFrame.

    The pipeline applies a rule-based filter, encodes the remaining text with
    a SentenceTransformer, scores examples by projection on an anchor
    centroid, removes low-scoring examples and finally performs greedy
    semantic deduplication.

    Args:
        df (pd.DataFrame): Input DataFrame containing text in ``text_col``.
        device (str): Device string for the sentence-transformer (e.g. "cuda").
        text_col (str): Name of the column containing text to filter.
        interestingness_percentile (int): Bottom percentile to drop by
            anchor-projection score.
        similarity_threshold (float): Cosine-similarity threshold for dedup.
        st_model_name (str): Name of the SentenceTransformer model to use.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: The filtered DataFrame and the final
        embeddings array corresponding to kept rows.
    """
    # 1. Rule-based filter
    before = len(df)
    df = df[df[text_col].apply(is_interesting_rule_based)].copy().reset_index(drop=True)
    print(f"Rule-based filter: {len(df)} / {before} entries retained")

    # 2. Encode
    st_model = SentenceTransformer(st_model_name)
    texts = df[text_col].tolist()
    embeddings = _encode_texts(st_model, texts, device=device)
    print(f"Embeddings shape: {embeddings.shape}")

    # 3. Interestingness via anchor projection
    anchor_embeddings = st_model.encode(
        ANCHOR_SENTENCES,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    )
    anchor_centroid = anchor_embeddings.mean(axis=0)
    anchor_centroid /= np.linalg.norm(anchor_centroid)

    interestingness_scores = embeddings @ anchor_centroid
    df["interestingness_score"] = interestingness_scores

    threshold = np.percentile(interestingness_scores, interestingness_percentile)
    mask = interestingness_scores >= threshold

    before = len(df)
    df = df[mask].copy().reset_index(drop=True)
    embeddings = embeddings[mask]
    print(
        f"Interestingness filter (bottom {interestingness_percentile}% removed): "
        f"{len(df)} / {before} entries retained"
    )

    # 4. Semantic dedup
    kept_indices = semantic_dedup(
        embs=embeddings,
        threshold=similarity_threshold,
        scores=interestingness_scores[mask],
    )
    before = len(df)
    df = df.iloc[kept_indices].copy().reset_index(drop=True)
    embeddings = embeddings[kept_indices]
    print(
        f"Semantic dedup (threshold={similarity_threshold}): "
        f"{len(df)} / {before} entries retained"
    )

    del st_model
    return df, embeddings
