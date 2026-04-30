"""Quality and deduplication filters."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def is_interesting_rule_based(text: str) -> bool:
    """Reject clearly low-quality sentences before embedding.

    Criteria:
    - At least 4 words
    - At most 300 characters
    - Does not start with URL/code patterns
    - Alpha ratio >= 0.55
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
    embs: np.ndarray,
    threshold: float,
    scores: np.ndarray,
) -> np.ndarray:
    """Greedy semantic deduplication.

    Parameters
    ----------
    embs      : (N, D) float32 array of L2-normalised embeddings.
    threshold : Cosine similarity above which two sentences are considered duplicates.
    scores    : (N,) interestingness scores used to sort candidates before the pass.

    Returns
    -------
    kept_indices : Sorted original indices of the surviving entries.
    """
    order = np.argsort(-scores)
    kept: list[int] = []
    kept_embs: list[np.ndarray] = []

    for idx in tqdm(order, desc="semantic dedup"):
        emb = embs[idx]
        if kept_embs:
            sims = np.array(kept_embs) @ emb
            if sims.max() >= threshold:
                continue
        kept.append(idx)
        kept_embs.append(emb)

    return np.sort(kept)


def nlp_quality_filter(
    df: pd.DataFrame,
    device: str,
    *,
    text_col: str = "prompt",
    interestingness_percentile: int = 20,
    similarity_threshold: float = 0.92,
    st_model_name: str = "all-MiniLM-L6-v2",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Full NLP quality pipeline: rule-based filter → interestingness → semantic dedup.

    Parameters
    ----------
    df : DataFrame with a *text_col* column.
    device : Device string for the sentence transformer (e.g. "cuda", "cpu").
    text_col : Column containing text to filter.
    interestingness_percentile : Bottom N% to remove by anchor similarity.
    similarity_threshold : Cosine threshold for semantic deduplication.
    st_model_name : SentenceTransformer model name.

    Returns
    -------
    (filtered_df, embeddings) — The filtered DataFrame and its final embeddings.
    """
    # 1. Rule-based filter
    before = len(df)
    df = df[df[text_col].apply(is_interesting_rule_based)].copy().reset_index(drop=True)
    print(f"Rule-based filter: {len(df)} / {before} entries retained")

    # 2. Encode
    st_model = SentenceTransformer(st_model_name, device=device)
    texts = df[text_col].tolist()
    embeddings = st_model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # 3. Interestingness via anchor projection
    anchor_embeddings = st_model.encode(
        ANCHOR_SENTENCES,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
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
