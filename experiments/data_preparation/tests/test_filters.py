"""Tests for quality and deduplication filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import filters
from src.filters import is_interesting_rule_based, semantic_dedup


def test_nlp_quality_filter_uses_multi_gpu_encoding(monkeypatch) -> None:
    """Use multiple CUDA devices when the hardware exposes them."""
    df = pd.DataFrame({
        "prompt": [f"A valid prompt number {i} with enough words." for i in range(300)]
    })

    monkeypatch.setattr(filters, "is_interesting_rule_based", lambda text: True)
    monkeypatch.setattr(
        filters,
        "semantic_dedup",
        lambda embs, threshold, scores: np.arange(len(scores)),
    )
    monkeypatch.setattr(filters.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(filters.torch.cuda, "device_count", lambda: 2)

    encode_calls: list[dict[str, object]] = []
    pool_calls: list[list[str]] = []

    class _FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def encode(self, texts, **kwargs):
            encode_calls.append(kwargs)
            return np.ones((len(texts), 384), dtype=np.float32)

        def start_multi_process_pool(self, target_devices):
            pool_calls.append(list(target_devices))
            return {"devices": list(target_devices)}

        def encode_multi_process(self, texts, pool, chunk_size):
            encode_calls.append({
                "pool": pool,
                "chunk_size": chunk_size,
                "multi_process": True,
            })
            return np.ones((len(texts), 384), dtype=np.float32)

        def stop_multi_process_pool(self, pool):
            return None

    monkeypatch.setattr(filters, "SentenceTransformer", _FakeSentenceTransformer)

    out_df, embeddings = filters.nlp_quality_filter(df, device="cuda")

    assert len(out_df) == len(df)
    assert embeddings.shape == (len(df), 384)
    assert pool_calls == [["cuda:0", "cuda:1"]]
    assert encode_calls[0]["multi_process"] is True
    assert encode_calls[0]["chunk_size"] >= 64


def test_is_interesting_rule_based_rejects_short_text() -> None:
    assert is_interesting_rule_based("too short") is False


def test_is_interesting_rule_based_rejects_urls() -> None:
    assert is_interesting_rule_based("http://example.com/path/to/resource") is False


def test_is_interesting_rule_based_accepts_normal_prompt() -> None:
    text = "Explain how gradient descent optimizes neural network weights."
    assert is_interesting_rule_based(text) is True


def test_semantic_dedup_keeps_highest_scoring_representative() -> None:
    embs = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.5, 0.9, 0.8])
    kept = semantic_dedup(embs, threshold=0.95, scores=scores)
    assert 1 in kept
    assert 2 in kept
    assert len(kept) == 2
