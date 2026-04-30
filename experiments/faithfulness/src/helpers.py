"""Helper functions for faithfulness evaluation."""

import difflib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mllm_shap.connectors.base.audio import AudioSegment
from mllm_shap.connectors.config import ModelConfig
from mllm_shap.shap.embeddings import MeanReducer
from mllm_shap.shap.similarity import CosineSimilarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine

from experiments.mllm_shapx import (
    ExperimentSet,
    InputModality,
    build_chat,
    load_df,
)
from experiments.mllm_shapx.src import (
    choose_prompt_text_column,
    iter_rows_for_selection,
)

EPS: float = 1e-9


def as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def load_spec(run_dir: Path, spec_path: Path | None = None) -> dict[str, Any]:
    spec_path = spec_path or (run_dir / "spec.json")
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing mllm_shapx spec: {spec_path}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def experiment_set_from_spec(spec: dict[str, Any]) -> ExperimentSet:
    raw = {
        "experiment_set_id": spec["experiment_set_id"],
        "output_root": "experiments_output",
        "device": spec.get("device"),
        "connector": spec["connector"],
        "dataset": spec["dataset"],
        "selection": spec["selection"],
        "generation": spec["generation"],
        "modality": spec["modality"],
        "shap": spec["shap"],
        "embedding": spec.get("embedding") or {},
        "experiments": [],
        "wandb": {"enabled": False},
    }
    return ExperimentSet.model_validate(raw)


def load_selected_rows(
    cfg: ExperimentSet, max_samples: int | None
) -> dict[int, dict[str, Any]]:
    df = load_df(
        cfg.dataset.repo_id,
        cfg.dataset.subset,
        cfg.dataset.split,
        cfg.dataset.revision,
        use_parquet=cfg.dataset.use_parquet,
        trust_remote_code=cfg.dataset.trust_remote_code,
    )
    text_col = choose_prompt_text_column(df)
    selected: dict[int, dict[str, Any]] = {}
    for row_idx, row in iter_rows_for_selection(
        df=df,
        start_index=cfg.selection.start_index,
        max_samples=max_samples,
        shuffle_seed=cfg.selection.shuffle_seed,
    ):
        row_dict = dict(row)
        row_dict["_text_col"] = text_col
        selected[int(row_idx)] = row_dict
    return selected


def extract_audio_sv(sample_json: dict[str, Any]) -> list[float]:
    for turn in sample_json.get("conversation", []):
        for entry in turn:
            if entry.get("content_type") != 1:
                continue
            values: list[float] = []
            for value in entry.get("shap_values") or []:
                if value is None:
                    continue
                value_f = float(value)
                if math.isfinite(value_f):
                    values.append(value_f)
            if values:
                return values
    raise ValueError("No audio SHAP values found in sample JSON.")


def aggregate_sv_to_segments(
    sv_values: list[float],
    segments: list[AudioSegment],
    total_samples: int,
) -> tuple[list[float], list[tuple[int, int]]]:
    """Map per-codec-token SVs to word segments using actual temporal alignment.

    Each codec token covers a fixed duration (total_samples / n_tokens samples).
    For each word segment [start_sample, end_sample], we compute which token
    indices fall within that range and average their SVs.

    Uses mean aggregation (not sum) to avoid biasing toward longer segments.
    """
    segment_count = len(segments)
    if segment_count <= 0:
        raise ValueError("segment_count must be positive.")
    if not sv_values:
        raise ValueError("No audio SHAP values found.")

    n_tokens = len(sv_values)
    if n_tokens == segment_count:
        return list(sv_values), [(i, i + 1) for i in range(segment_count)]

    values = np.asarray(sv_values, dtype=float)
    hop_size = total_samples / n_tokens

    aggregated: list[float] = []
    bins: list[tuple[int, int]] = []
    for seg in segments:
        start_sample = seg.start_sample if seg.start_sample is not None else 0
        end_sample = seg.end_sample if seg.end_sample is not None else total_samples
        start_token = int(start_sample / hop_size)
        end_token = int(np.ceil(end_sample / hop_size))
        start_token = max(0, min(start_token, n_tokens))
        end_token = max(start_token + 1, min(end_token, n_tokens))
        token_slice = values[start_token:end_token]
        aggregated.append(float(token_slice.mean()))
        bins.append((start_token, end_token))
    return aggregated, bins


def sample_paths(run_dir: Path, max_samples: int | None) -> list[Path]:
    paths = sorted((run_dir / "samples").glob("sample_*_result.json"))
    if max_samples is not None:
        paths = paths[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No sample JSON files found in {run_dir / 'samples'}")
    return paths


def parse_sample_id(sample_path: Path) -> int:
    return int(sample_path.name.split("_")[1])


def remove_interval(waveform: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Remove an interval from the waveform by concatenating the parts before and after.

    This matches the SHAP computation's masking paradigm (segment removal via
    concatenation) rather than silence insertion.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    start = max(0, min(int(start), waveform.size(-1)))
    end = max(start, min(int(end), waveform.size(-1)))
    return torch.cat([waveform[:, :start], waveform[:, end:]], dim=1)


def segment_interval(seg: AudioSegment) -> tuple[int, int]:
    if seg.start_sample is None or seg.end_sample is None:
        raise ValueError("Segment is missing sample indices.")
    return int(seg.start_sample), int(seg.end_sample)


def embedding_similarities(
    model: Any, base: Any, others: list[Any]
) -> tuple[float, list[float]]:
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


def rank_abs_sv(segment_sv_values: list[float]) -> dict[str, Any]:
    values = np.asarray(segment_sv_values, dtype=float)
    abs_values = np.abs(values)
    order = np.argsort(-abs_values, kind="mergesort")
    ranks = np.empty(len(abs_values), dtype=int)
    ranks[order] = np.arange(1, len(abs_values) + 1)

    total_abs = float(abs_values.sum())
    shares = abs_values / (total_abs + EPS)
    top_abs = float(abs_values[order[0]]) if len(order) else 0.0
    second_abs = float(abs_values[order[1]]) if len(order) > 1 else None
    top1_top2_gap = top_abs - second_abs if second_abs is not None else None
    top1_top2_ratio = top_abs / (second_abs + EPS) if second_abs is not None else None
    top1_share = float(shares[order[0]]) if len(order) else 0.0

    positive_shares = shares[shares > 0]
    entropy_norm = None
    if len(shares) > 1 and len(positive_shares):
        entropy = -float(np.sum(positive_shares * np.log(positive_shares)))
        entropy_norm = entropy / float(np.log(len(shares)))

    sorted_abs = np.sort(abs_values)
    if len(sorted_abs) == 0 or total_abs <= EPS:
        gini = 0.0
    else:
        index = np.arange(1, len(sorted_abs) + 1)
        gini = float(
            (2 * np.sum(index * sorted_abs)) / (len(sorted_abs) * total_abs)
            - (len(sorted_abs) + 1) / len(sorted_abs)
        )

    return {
        "order": order,
        "ranks": ranks,
        "abs_values": abs_values,
        "shares": shares,
        "top_abs": top_abs,
        "top1_top2_gap": float(top1_top2_gap) if top1_top2_gap is not None else None,
        "top1_top2_ratio": float(top1_top2_ratio)
        if top1_top2_ratio is not None
        else None,
        "top1_share": top1_share,
        "abs_sv_entropy_norm": float(entropy_norm)
        if entropy_norm is not None
        else None,
        "abs_sv_gini": gini,
    }


def generate_response(
    model: Any,
    audio_bytes: bytes,
    input_modality: InputModality,
    max_new_tokens: int,
    text_temperature: float,
    user_texts: list[str] | None = None,
) -> Any:
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
