"""Deletion-based faithfulness validation for SGPA Shapley values.

The experiment consumes existing mllm_shapx SGPA runs, identifies the highest
absolute-SV word segment per sample, and tests whether deleting that segment
changes the model response more than deleting a random equal-duration segment.
With --all-rank-deletions, it reuses the same phase-1 SHAP sample JSONs and
deletes every SGPA segment once to evaluate whether deletion impact tracks
absolute-SV rank.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors.base.audio import AudioSegment, SpectrogramGuidedAligner
from mllm_shap.connectors.config import ModelConfig
from mllm_shap.shap.similarity import TfIdfCosineSimilarity
from mllm_shap.utils.audio import TorchAudioHandler
from scipy import stats
from tqdm.auto import tqdm

from experiments.mllm_shapx.config import ExperimentSet, parse_experiment_set
from experiments.mllm_shapx.constants import InputModality, OutputModality
from experiments.mllm_shapx.data import (
    choose_prompt_text_column,
    extract_texts_from_row,
    iter_rows_for_selection,
    load_df,
)
from experiments.mllm_shapx.factory import _build_model, build_chat


DEFAULT_MALE_RUN_DIR = Path(
    "experiments_output/single_sentence_2026_01_03/"
    "audio_male_audio_limited_neyman_lin3_0"
)
DEFAULT_FEMALE_RUN_DIR = Path(
    "experiments_output/single_sentence_2026_01_03/"
    "audio_female_audio_limited_neyman_lin3_0"
)
DEFAULT_OUTPUT_DIR = Path("experiments/interspeech/outputs/faithfulness_deletion")
EPS = 1e-9


@dataclass(frozen=True)
class FaithfulnessResult:
    """Per-sample deletion faithfulness result."""

    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    n_segments: int
    top_segment_idx: int
    random_segment_idx: int
    top_segment_token: str
    random_segment_token: str
    top_abs_sv: float
    top_sv: float
    original_similarity: float
    top_similarity: float
    random_similarity: float
    top_drop: float
    random_drop: float
    drop_difference: float
    top_start_sec: float
    top_end_sec: float
    random_start_sec: float
    random_end_sec: float
    mask_duration_sec: float
    runtime_sec: float


@dataclass(frozen=True)
class RankwiseDeletionResult:
    """Per-segment deletion result for all-rank faithfulness validation."""

    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    n_segments: int
    segment_idx: int
    segment_rank_abs_sv: int
    segment_token: str
    segment_sv: float
    segment_abs_sv: float
    segment_abs_sv_share: float
    top_abs_sv: float
    top1_top2_gap: float | None
    top1_top2_ratio: float | None
    top1_share: float
    abs_sv_entropy_norm: float | None
    abs_sv_gini: float
    original_similarity: float
    deleted_similarity: float
    deletion_drop: float
    segment_start_sec: float
    segment_end_sec: float
    mask_duration_sec: float
    runtime_sec: float


@dataclass(frozen=True)
class FailureResult:
    """Failed sample metadata."""

    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    error_type: str
    error_message: str


def _as_list(value: Any) -> list[Any]:
    """Normalize parquet/datasets list-like cells."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _load_spec(run_dir: Path, spec_path: Path | None = None) -> dict[str, Any]:
    spec_path = spec_path or (run_dir / "spec.json")
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Missing mllm_shapx spec: {spec_path}. "
            "Pass --spec-path for committed HP-1 specs when experiments_output is incomplete."
        )
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _experiment_set_from_spec(spec: dict[str, Any]) -> ExperimentSet:
    """Rehydrate the subset of ExperimentSet needed for dataset/model setup."""
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
    return parse_experiment_set(raw)


def _load_selected_rows(
    cfg: ExperimentSet, max_samples: int | None
) -> dict[int, dict[str, Any]]:
    """Load rows using the same ordering policy as mllm_shapx."""
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


def _extract_audio_sv(sample_json: dict[str, Any]) -> list[float]:
    """Extract non-null audio SHAP values from a serialized mllm_shapx sample."""
    values: list[float] = []
    for turn in sample_json.get("conversation", []):
        for entry in turn:
            if entry.get("content_type") != 1:
                continue
            for value in entry.get("shap_values") or []:
                if value is None:
                    continue
                value_f = float(value)
                if math.isfinite(value_f):
                    values.append(value_f)
            if values:
                return values
    raise ValueError("No audio SHAP values found in sample JSON.")


def _aggregate_sv_to_segments(
    sv_values: list[float], segment_count: int
) -> tuple[list[float], list[tuple[int, int]]]:
    """Map serialized audio attribution values onto SGPA-aligned segments.

    The saved conversation can contain finer-grained audio attribution entries than
    the SGPA alignment emits. For the deletion test, aggregate contiguous SV bins
    onto the available aligned segments and select the segment with largest
    absolute aggregate contribution.
    """
    if segment_count <= 0:
        raise ValueError("segment_count must be positive.")
    if not sv_values:
        raise ValueError("No audio SHAP values found in sample JSON.")
    if len(sv_values) == segment_count:
        return list(sv_values), [(i, i + 1) for i in range(segment_count)]

    values = np.asarray(sv_values, dtype=float)
    aggregated: list[float] = []
    bins: list[tuple[int, int]] = []
    for segment_idx in range(segment_count):
        start = int(np.floor(segment_idx * len(values) / segment_count))
        end = int(np.floor((segment_idx + 1) * len(values) / segment_count))
        end = max(end, start + 1)
        end = min(end, len(values))
        if start >= len(values):
            start = len(values) - 1
            end = len(values)
        chunk = values[start:end]
        aggregated.append(float(chunk.sum()))
        bins.append((start, end))
    return aggregated, bins


def _sample_paths(run_dir: Path, max_samples: int | None) -> list[Path]:
    paths = sorted((run_dir / "samples").glob("sample_*_result.json"))
    if max_samples is not None:
        paths = paths[:max_samples]
    if not paths:
        raise FileNotFoundError(
            f"No sample JSON files found in {run_dir / 'samples'}. "
            "Generate HP-1 SHAP inputs first with "
            "`python -m experiments.mllm_shapx.cli run --config "
            "experiments/interspeech/configs/hp1_shap_inputs_audio_male.json "
            "--max-samples 1 --resume` "
            "(or the female config for audio__female)."
        )
    return paths


def _parse_sample_id(sample_path: Path) -> int:
    return int(sample_path.name.split("_")[1])


def _mask_interval(
    waveform: torch.Tensor,
    start_sample: int,
    end_sample: int,
) -> torch.Tensor:
    """Silence one interval while preserving duration."""
    out = waveform.clone()
    if out.dim() == 1:
        out = out.unsqueeze(0)
    start_sample = max(0, min(int(start_sample), out.size(-1)))
    end_sample = max(start_sample, min(int(end_sample), out.size(-1)))
    out[:, start_sample:end_sample] = 0.0
    return out


def _equal_duration_interval(
    center_sample: int,
    duration_samples: int,
    total_samples: int,
) -> tuple[int, int]:
    """Build an equal-duration interval centered near a reference segment."""
    duration_samples = max(1, min(int(duration_samples), int(total_samples)))
    start = int(center_sample - duration_samples // 2)
    start = max(0, min(start, total_samples - duration_samples))
    return start, start + duration_samples


def _segment_interval(seg: AudioSegment) -> tuple[int, int]:
    if seg.start_sample is None or seg.end_sample is None:
        raise ValueError("Segment is missing sample indices.")
    return int(seg.start_sample), int(seg.end_sample)


def _response_similarity(
    base: Any, top: Any, random: Any
) -> tuple[float, float, float]:
    """Compute the same token-level TF-IDF cosine payoff used in the SGPA runs."""
    sims = TfIdfCosineSimilarity()(base=base, other=[base, top, random])
    return float(sims[0].item()), float(sims[1].item()), float(sims[2].item())


def _response_similarities(base: Any, others: list[Any]) -> tuple[float, list[float]]:
    """Compute base self-similarity and similarities for multiple responses."""
    sims = TfIdfCosineSimilarity()(base=base, other=[base, *others])
    return float(sims[0].item()), [float(value.item()) for value in sims[1:]]


def _rank_abs_sv(segment_sv_values: list[float]) -> dict[str, Any]:
    """Rank segment SVs by absolute magnitude and compute concentration metrics."""
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


def _preflight_one_sample(
    *,
    sample_path: Path,
    row: dict[str, Any],
    aligner: SpectrogramGuidedAligner,
    audio_column: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Validate data, SV extraction, alignment, interval selection and audio encoding."""
    sample_id = _parse_sample_id(sample_path)
    sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
    row_index = int(sample_json.get("row_index", sample_id))
    transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
    audio_values = _as_list(row[audio_column])
    if not audio_values:
        raise ValueError(f"No audio bytes in column {audio_column}")
    audio_bytes = audio_values[0]

    waveform, sample_rate = TorchAudioHandler.from_bytes(
        audio_bytes, audio_format="wav"
    )
    segments = aligner(
        transcript=transcript,
        waveform=waveform,
        original_sr=int(sample_rate),
        audio_format="wav",
        attach_audio=False,
    )
    if not segments:
        raise ValueError("Alignment produced no segments.")
    sv_values = _extract_audio_sv(sample_json)
    segment_sv_values, sv_bins = _aggregate_sv_to_segments(sv_values, len(segments))
    top_segment_idx = int(np.argmax(np.abs(np.asarray(segment_sv_values, dtype=float))))

    candidate_indices = [i for i in range(len(segments)) if i != top_segment_idx]
    random_segment_idx = int(rng.choice(candidate_indices))
    top_start, top_end = _segment_interval(segments[top_segment_idx])
    random_start_raw, random_end_raw = _segment_interval(segments[random_segment_idx])
    random_start, random_end = _equal_duration_interval(
        center_sample=(random_start_raw + random_end_raw) // 2,
        duration_samples=max(1, top_end - top_start),
        total_samples=waveform.size(-1),
    )
    top_audio = TorchAudioHandler.to_bytes(
        _mask_interval(waveform, top_start, top_end),
        sample_rate=int(sample_rate),
        audio_format="wav",
    )
    random_audio = TorchAudioHandler.to_bytes(
        _mask_interval(waveform, random_start, random_end),
        sample_rate=int(sample_rate),
        audio_format="wav",
    )
    return {
        "sample_id": sample_id,
        "row_index": row_index,
        "audio_column": audio_column,
        "transcript": transcript,
        "sv_count": len(sv_values),
        "segment_count": len(segments),
        "top_segment_idx": top_segment_idx,
        "top_sv_bin": sv_bins[top_segment_idx],
        "random_segment_idx": random_segment_idx,
        "top_token": segments[top_segment_idx].token,
        "random_token": segments[random_segment_idx].token,
        "top_audio_bytes": len(top_audio),
        "random_audio_bytes": len(random_audio),
    }


def _generate_response(
    model: Any,
    audio_bytes: bytes,
    input_modality: InputModality,
    max_new_tokens: int,
    text_temperature: float,
) -> Any:
    chat = build_chat(
        model,
        user_texts=None,
        audio_bytes_list=[audio_bytes],
        input_modality=input_modality,
    )
    return model.generate(
        chat=chat,
        max_new_tokens=max_new_tokens,
        model_config=ModelConfig(text_temperature=float(text_temperature)),
        keep_history=False,
    )


def _run_one_sample(
    *,
    sample_path: Path,
    row: dict[str, Any],
    model: Any,
    aligner: SpectrogramGuidedAligner,
    input_modality: InputModality,
    audio_column: str,
    max_new_tokens: int,
    text_temperature: float,
    rng: np.random.Generator,
) -> FaithfulnessResult:
    sample_id = _parse_sample_id(sample_path)
    sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
    row_index = int(sample_json.get("row_index", sample_id))
    transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
    audio_values = _as_list(row[audio_column])
    if not audio_values:
        raise ValueError(f"No audio bytes in column {audio_column}")
    audio_bytes = audio_values[0]

    waveform, sample_rate = TorchAudioHandler.from_bytes(
        audio_bytes, audio_format="wav"
    )
    segments = aligner(
        transcript=transcript,
        waveform=waveform,
        original_sr=int(sample_rate),
        audio_format="wav",
        attach_audio=False,
    )
    if not segments:
        raise ValueError("Alignment produced no segments.")
    sv_values = _extract_audio_sv(sample_json)
    segment_sv_values, _sv_bins = _aggregate_sv_to_segments(sv_values, len(segments))
    top_segment_idx = int(np.argmax(np.abs(np.asarray(segment_sv_values, dtype=float))))
    top_sv = float(segment_sv_values[top_segment_idx])

    candidate_indices = [i for i in range(len(segments)) if i != top_segment_idx]
    if not candidate_indices:
        raise ValueError("Need at least two segments for random baseline.")
    random_segment_idx = int(rng.choice(candidate_indices))

    top_seg = segments[top_segment_idx]
    random_seg = segments[random_segment_idx]
    top_start, top_end = _segment_interval(top_seg)
    mask_duration = max(1, top_end - top_start)
    random_start_raw, random_end_raw = _segment_interval(random_seg)
    random_center = (random_start_raw + random_end_raw) // 2
    random_start, random_end = _equal_duration_interval(
        center_sample=random_center,
        duration_samples=mask_duration,
        total_samples=waveform.size(-1),
    )

    top_waveform = _mask_interval(waveform, top_start, top_end)
    random_waveform = _mask_interval(waveform, random_start, random_end)
    top_audio = TorchAudioHandler.to_bytes(
        top_waveform, sample_rate=int(sample_rate), audio_format="wav"
    )
    random_audio = TorchAudioHandler.to_bytes(
        random_waveform, sample_rate=int(sample_rate), audio_format="wav"
    )

    t0 = time.perf_counter()
    base_response = _generate_response(
        model, audio_bytes, input_modality, max_new_tokens, text_temperature
    )
    top_response = _generate_response(
        model, top_audio, input_modality, max_new_tokens, text_temperature
    )
    random_response = _generate_response(
        model, random_audio, input_modality, max_new_tokens, text_temperature
    )
    original_sim, top_sim, random_sim = _response_similarity(
        base_response, top_response, random_response
    )

    top_drop = original_sim - top_sim
    random_drop = original_sim - random_sim
    return FaithfulnessResult(
        sample_id=sample_id,
        row_index=row_index,
        audio_column=audio_column,
        transcript=transcript,
        n_segments=len(segments),
        top_segment_idx=top_segment_idx,
        random_segment_idx=random_segment_idx,
        top_segment_token=top_seg.token,
        random_segment_token=random_seg.token,
        top_abs_sv=abs(top_sv),
        top_sv=top_sv,
        original_similarity=original_sim,
        top_similarity=top_sim,
        random_similarity=random_sim,
        top_drop=top_drop,
        random_drop=random_drop,
        drop_difference=top_drop - random_drop,
        top_start_sec=float(top_start / sample_rate),
        top_end_sec=float(top_end / sample_rate),
        random_start_sec=float(random_start / sample_rate),
        random_end_sec=float(random_end / sample_rate),
        mask_duration_sec=float(mask_duration / sample_rate),
        runtime_sec=float(time.perf_counter() - t0),
    )


def _run_one_sample_rankwise(
    *,
    sample_path: Path,
    row: dict[str, Any],
    model: Any,
    aligner: SpectrogramGuidedAligner,
    input_modality: InputModality,
    audio_column: str,
    max_new_tokens: int,
    text_temperature: float,
) -> list[RankwiseDeletionResult]:
    """Delete every aligned SGPA segment once and record its SV rank."""
    sample_id = _parse_sample_id(sample_path)
    sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
    row_index = int(sample_json.get("row_index", sample_id))
    transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
    audio_values = _as_list(row[audio_column])
    if not audio_values:
        raise ValueError(f"No audio bytes in column {audio_column}")
    audio_bytes = audio_values[0]

    waveform, sample_rate = TorchAudioHandler.from_bytes(
        audio_bytes, audio_format="wav"
    )
    segments = aligner(
        transcript=transcript,
        waveform=waveform,
        original_sr=int(sample_rate),
        audio_format="wav",
        attach_audio=False,
    )
    if not segments:
        raise ValueError("Alignment produced no segments.")

    sv_values = _extract_audio_sv(sample_json)
    segment_sv_values, _sv_bins = _aggregate_sv_to_segments(sv_values, len(segments))
    rank_info = _rank_abs_sv(segment_sv_values)

    deleted_audio: list[bytes] = []
    intervals: list[tuple[int, int]] = []
    for segment in segments:
        start, end = _segment_interval(segment)
        intervals.append((start, end))
        deleted_audio.append(
            TorchAudioHandler.to_bytes(
                _mask_interval(waveform, start, end),
                sample_rate=int(sample_rate),
                audio_format="wav",
            )
        )

    t0 = time.perf_counter()
    base_response = _generate_response(
        model, audio_bytes, input_modality, max_new_tokens, text_temperature
    )
    deleted_responses = [
        _generate_response(
            model,
            segment_audio,
            input_modality,
            max_new_tokens,
            text_temperature,
        )
        for segment_audio in deleted_audio
    ]
    original_sim, deleted_sims = _response_similarities(
        base_response, deleted_responses
    )
    runtime_sec = float(time.perf_counter() - t0)

    results: list[RankwiseDeletionResult] = []
    for segment_idx, (segment, deleted_sim) in enumerate(zip(segments, deleted_sims)):
        start, end = intervals[segment_idx]
        segment_sv = float(segment_sv_values[segment_idx])
        deletion_drop = original_sim - deleted_sim
        results.append(
            RankwiseDeletionResult(
                sample_id=sample_id,
                row_index=row_index,
                audio_column=audio_column,
                transcript=transcript,
                n_segments=len(segments),
                segment_idx=segment_idx,
                segment_rank_abs_sv=int(rank_info["ranks"][segment_idx]),
                segment_token=segment.token,
                segment_sv=segment_sv,
                segment_abs_sv=float(rank_info["abs_values"][segment_idx]),
                segment_abs_sv_share=float(rank_info["shares"][segment_idx]),
                top_abs_sv=float(rank_info["top_abs"]),
                top1_top2_gap=rank_info["top1_top2_gap"],
                top1_top2_ratio=rank_info["top1_top2_ratio"],
                top1_share=float(rank_info["top1_share"]),
                abs_sv_entropy_norm=rank_info["abs_sv_entropy_norm"],
                abs_sv_gini=float(rank_info["abs_sv_gini"]),
                original_similarity=original_sim,
                deleted_similarity=deleted_sim,
                deletion_drop=deletion_drop,
                segment_start_sec=float(start / sample_rate),
                segment_end_sec=float(end / sample_rate),
                mask_duration_sec=float(max(1, end - start) / sample_rate),
                runtime_sec=runtime_sec / max(1, len(segments)),
            )
        )
    return results


def _summarize(results_df: pd.DataFrame, failures_df: pd.DataFrame) -> dict[str, Any]:
    if results_df.empty:
        return {
            "completed_samples": 0,
            "failed_samples": int(len(failures_df)),
        }

    top = results_df["top_drop"].to_numpy(dtype=float)
    random = results_df["random_drop"].to_numpy(dtype=float)
    diff = top - random
    has_pairwise_stats = len(results_df) >= 2
    t_stat, p_value = (
        stats.ttest_rel(top, random) if has_pairwise_stats else (None, None)
    )
    return {
        "completed_samples": int(len(results_df)),
        "failed_samples": int(len(failures_df)),
        "mean_top_drop": float(np.mean(top)),
        "std_top_drop": float(np.std(top, ddof=1)) if has_pairwise_stats else None,
        "mean_random_drop": float(np.mean(random)),
        "std_random_drop": float(np.std(random, ddof=1))
        if has_pairwise_stats
        else None,
        "mean_drop_difference": float(np.mean(diff)),
        "median_drop_difference": float(np.median(diff)),
        "paired_t_stat": float(t_stat) if t_stat is not None else None,
        "paired_p_value": float(p_value) if p_value is not None else None,
        "cohen_dz": (
            float(np.mean(diff) / (np.std(diff, ddof=1) + EPS))
            if has_pairwise_stats
            else None
        ),
        "top_greater_than_random_rate": float(np.mean(diff > 0)),
        "mean_runtime_sec": float(results_df["runtime_sec"].mean()),
    }


def _safe_spearman(x: pd.Series, y: pd.Series) -> float | None:
    """Compute Spearman correlation when both vectors have enough variation."""
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return None
    corr = stats.spearmanr(x.to_numpy(dtype=float), y.to_numpy(dtype=float)).statistic
    return float(corr) if math.isfinite(float(corr)) else None


def _summarize_rankwise(
    results_df: pd.DataFrame, failures_df: pd.DataFrame
) -> dict[str, Any]:
    """Summarize all-rank deletion outputs."""
    if results_df.empty:
        return {
            "completed_deletions": 0,
            "completed_samples": 0,
            "failed_samples": int(len(failures_df)),
        }

    per_rank = (
        results_df.groupby("segment_rank_abs_sv", as_index=True)
        .agg(
            n=("deletion_drop", "size"),
            mean_drop=("deletion_drop", "mean"),
            median_drop=("deletion_drop", "median"),
            mean_abs_sv_share=("segment_abs_sv_share", "mean"),
            mean_abs_sv=("segment_abs_sv", "mean"),
        )
        .sort_index()
    )
    per_rank_summary = {
        str(int(rank)): {
            "n": int(row["n"]),
            "mean_drop": float(row["mean_drop"]),
            "median_drop": float(row["median_drop"]),
            "mean_abs_sv_share": float(row["mean_abs_sv_share"]),
            "mean_abs_sv": float(row["mean_abs_sv"]),
        }
        for rank, row in per_rank.iterrows()
    }

    top_rows = results_df[results_df["segment_rank_abs_sv"] == 1]
    non_top_rows = results_df[results_df["segment_rank_abs_sv"] > 1]
    sample_count = int(
        results_df[["audio_column", "sample_id"]].drop_duplicates().shape[0]
    )

    per_sample_corrs: list[float] = []
    for _key, group in results_df.groupby(["audio_column", "sample_id"]):
        corr = _safe_spearman(group["segment_abs_sv"], group["deletion_drop"])
        if corr is not None:
            per_sample_corrs.append(corr)

    global_corr = _safe_spearman(
        results_df["segment_abs_sv"], results_df["deletion_drop"]
    )
    global_rank_corr = _safe_spearman(
        -results_df["segment_rank_abs_sv"], results_df["deletion_drop"]
    )
    top_minus_non_top = (
        float(top_rows["deletion_drop"].mean() - non_top_rows["deletion_drop"].mean())
        if not top_rows.empty and not non_top_rows.empty
        else None
    )

    return {
        "completed_deletions": int(len(results_df)),
        "completed_samples": sample_count,
        "failed_samples": int(len(failures_df)),
        "mean_deletion_drop": float(results_df["deletion_drop"].mean()),
        "mean_top_rank_drop": float(top_rows["deletion_drop"].mean())
        if not top_rows.empty
        else None,
        "mean_non_top_rank_drop": float(non_top_rows["deletion_drop"].mean())
        if not non_top_rows.empty
        else None,
        "mean_top_minus_non_top_drop": top_minus_non_top,
        "spearman_abs_sv_vs_drop": global_corr,
        "spearman_negative_rank_vs_drop": global_rank_corr,
        "mean_within_sample_spearman_abs_sv_vs_drop": (
            float(np.mean(per_sample_corrs)) if per_sample_corrs else None
        ),
        "median_within_sample_spearman_abs_sv_vs_drop": (
            float(np.median(per_sample_corrs)) if per_sample_corrs else None
        ),
        "within_sample_spearman_n": len(per_sample_corrs),
        "mean_top1_share": float(
            results_df[["audio_column", "sample_id", "top1_share"]]
            .drop_duplicates()["top1_share"]
            .mean()
        ),
        "mean_top1_top2_gap": float(
            results_df[["audio_column", "sample_id", "top1_top2_gap"]]
            .drop_duplicates()["top1_top2_gap"]
            .dropna()
            .mean()
        ),
        "mean_abs_sv_entropy_norm": float(
            results_df[["audio_column", "sample_id", "abs_sv_entropy_norm"]]
            .drop_duplicates()["abs_sv_entropy_norm"]
            .dropna()
            .mean()
        ),
        "per_rank": per_rank_summary,
        "mean_runtime_sec_per_deletion": float(results_df["runtime_sec"].mean()),
    }


def combine_partition_outputs(
    output_dir: Path, all_rank_deletions: bool = False
) -> dict[str, Any]:
    """Combine partition CSVs and write voice-level plus combined summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    combined_frames: list[pd.DataFrame] = []
    combined_failure_frames: list[pd.DataFrame] = []
    results_tag = "rankwise_results" if all_rank_deletions else "results"
    summary_tag = "rankwise_summary" if all_rank_deletions else "summary"
    combined_results_name = (
        "combined_rankwise_results.csv"
        if all_rank_deletions
        else "combined_results.csv"
    )
    combined_summary_name = (
        "combined_rankwise_summary.json"
        if all_rank_deletions
        else "combined_summary.json"
    )

    for audio_column in ("audio__male", "audio__female"):
        paths = sorted(output_dir.glob(f"{audio_column}_part*-of*_{results_tag}.csv"))
        if not paths:
            continue
        df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        if all_rank_deletions:
            df = df.drop_duplicates(subset=["sample_id", "segment_idx"], keep="last")
            df = df.sort_values(["sample_id", "segment_rank_abs_sv", "segment_idx"])
        else:
            df = df.drop_duplicates(subset=["sample_id"], keep="last")
            df = df.sort_values("sample_id")
        result_path = output_dir / f"{audio_column}_combined_{results_tag}.csv"
        summary_path = output_dir / f"{audio_column}_combined_{summary_tag}.json"
        df.to_csv(result_path, index=False)
        failures_paths = sorted(
            output_dir.glob(f"{audio_column}_part*-of*_failures.csv")
        )
        failures_df = (
            pd.concat([pd.read_csv(path) for path in failures_paths], ignore_index=True)
            if failures_paths
            else pd.DataFrame()
        )
        if not failures_df.empty:
            failures_df = failures_df.drop_duplicates(
                subset=["sample_id", "audio_column", "error_type", "error_message"],
                keep="last",
            )
            combined_failure_frames.append(failures_df)
        summary = (
            _summarize_rankwise(df, failures_df)
            if all_rank_deletions
            else _summarize(df, failures_df)
        )
        summary.update({"audio_column": audio_column, "results_csv": str(result_path)})
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[audio_column] = summary
        combined_frames.append(df)

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        sort_columns = (
            ["audio_column", "sample_id", "segment_rank_abs_sv", "segment_idx"]
            if all_rank_deletions
            else ["audio_column", "sample_id"]
        )
        combined_df = combined_df.sort_values(sort_columns)
        combined_path = output_dir / combined_results_name
        combined_summary_path = output_dir / combined_summary_name
        combined_df.to_csv(combined_path, index=False)
        combined_failures_df = (
            pd.concat(combined_failure_frames, ignore_index=True)
            if combined_failure_frames
            else pd.DataFrame()
        )
        combined_summary = (
            _summarize_rankwise(combined_df, combined_failures_df)
            if all_rank_deletions
            else _summarize(combined_df, combined_failures_df)
        )
        combined_summary.update({"results_csv": str(combined_path)})
        combined_summary_path.write_text(
            json.dumps(combined_summary, indent=2), encoding="utf-8"
        )
        summaries["combined"] = combined_summary

    return summaries


def run_faithfulness(
    *,
    run_dir: Path,
    spec_path: Path | None,
    output_dir: Path,
    max_samples: int | None,
    device: str,
    aligner_device: str,
    seed: int,
    partition_index: int | None,
    num_partitions: int | None,
    resume: bool,
    preflight_only: bool = False,
    all_rank_deletions: bool = False,
    max_new_tokens_override: int | None = None,
    text_temperature_override: float | None = None,
) -> dict[str, Any]:
    """Run deletion faithfulness for one mllm_shapx run directory."""
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = _load_spec(run_dir, spec_path=spec_path)
    cfg = _experiment_set_from_spec(spec)
    input_modality = cfg.modality.get_input_modality()
    output_modality = cfg.modality.get_output_modality()
    if output_modality != OutputModality.AUDIO:
        raise ValueError("HP-1 faithfulness currently expects audio-output runs.")
    audio_column = cfg.modality.input_modality

    sample_paths = _sample_paths(run_dir, max_samples=max_samples)
    # Load the full deterministic row index map. max_samples limits the number of
    # sample JSONs to evaluate, not the maximum row_index those JSONs may refer to.
    rows = _load_selected_rows(cfg, max_samples=None)
    if partition_index is not None or num_partitions is not None:
        if partition_index is None or num_partitions is None:
            raise ValueError("Provide both partition_index and num_partitions.")
        sample_paths = [
            path
            for i, path in enumerate(sample_paths)
            if i % int(num_partitions) == int(partition_index)
        ]

    suffix = audio_column
    if partition_index is not None:
        suffix += f"_part{partition_index}-of-{num_partitions}"
    results_name = "rankwise_results" if all_rank_deletions else "results"
    summary_name = "rankwise_summary" if all_rank_deletions else "summary"
    results_path = output_dir / f"{suffix}_{results_name}.csv"
    failures_path = output_dir / f"{suffix}_failures.csv"
    summary_path = output_dir / f"{suffix}_{summary_name}.json"

    existing_ids: set[int] = set()
    results: list[FaithfulnessResult | RankwiseDeletionResult] = []
    failures: list[FailureResult] = []
    if resume and results_path.exists():
        existing = pd.read_csv(results_path)
        existing_ids = set(existing["sample_id"].astype(int).tolist())

    aligner = SpectrogramGuidedAligner(device=torch.device(aligner_device))
    rng = np.random.default_rng(seed)

    if preflight_only:
        checks: list[dict[str, Any]] = []
        for sample_path in sample_paths:
            sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
            row_index = int(sample_json.get("row_index", _parse_sample_id(sample_path)))
            checks.append(
                _preflight_one_sample(
                    sample_path=sample_path,
                    row=rows[row_index],
                    aligner=aligner,
                    audio_column=audio_column,
                    rng=rng,
                )
            )
        preflight_path = output_dir / f"{suffix}_preflight.json"
        summary = {
            "preflight_only": True,
            "checked_samples": len(checks),
            "run_dir": str(run_dir),
            "audio_column": audio_column,
            "preflight_json": str(preflight_path),
            "checks": checks,
        }
        preflight_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    torch_device = torch.device(device)
    model = _build_model(
        device=torch_device,
        connector=cfg.connector,
        output_modality=output_modality,
    )
    max_new_tokens = (
        int(max_new_tokens_override)
        if max_new_tokens_override is not None
        else int(cfg.generation.max_new_tokens)
    )
    text_temperature = (
        float(text_temperature_override)
        if text_temperature_override is not None
        else float(cfg.generation.text_temperature)
    )

    for sample_path in tqdm(sample_paths, desc=f"faithfulness ({audio_column})"):
        sample_id = _parse_sample_id(sample_path)
        if sample_id in existing_ids:
            continue
        row_index = sample_id
        transcript = ""
        try:
            sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
            row_index = int(sample_json.get("row_index", sample_id))
            row = rows[row_index]
            transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
            if all_rank_deletions:
                results.extend(
                    _run_one_sample_rankwise(
                        sample_path=sample_path,
                        row=row,
                        model=model,
                        aligner=aligner,
                        input_modality=input_modality,
                        audio_column=audio_column,
                        max_new_tokens=max_new_tokens,
                        text_temperature=text_temperature,
                    )
                )
            else:
                result = _run_one_sample(
                    sample_path=sample_path,
                    row=row,
                    model=model,
                    aligner=aligner,
                    input_modality=input_modality,
                    audio_column=audio_column,
                    max_new_tokens=max_new_tokens,
                    text_temperature=text_temperature,
                    rng=rng,
                )
                results.append(result)
        except Exception as exc:  # noqa: BLE001 - keep long cluster runs alive.
            failures.append(
                FailureResult(
                    sample_id=sample_id,
                    row_index=row_index,
                    audio_column=audio_column,
                    transcript=transcript,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

        if results:
            new_df = pd.DataFrame([asdict(r) for r in results])
            if resume and results_path.exists():
                old_df = pd.read_csv(results_path)
                new_df = pd.concat([old_df, new_df], ignore_index=True)
                duplicate_subset = (
                    ["sample_id", "segment_idx"]
                    if all_rank_deletions
                    else ["sample_id"]
                )
                new_df = new_df.drop_duplicates(subset=duplicate_subset, keep="last")
            sort_columns = (
                ["sample_id", "segment_rank_abs_sv", "segment_idx"]
                if all_rank_deletions
                else ["sample_id"]
            )
            new_df.sort_values(sort_columns).to_csv(results_path, index=False)
        if failures:
            pd.DataFrame([asdict(f) for f in failures]).to_csv(
                failures_path, index=False
            )

    final_results = (
        pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    )
    final_failures = (
        pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    )
    summary = (
        _summarize_rankwise(final_results, final_failures)
        if all_rank_deletions
        else _summarize(final_results, final_failures)
    )
    summary.update(
        {
            "run_dir": str(run_dir),
            "audio_column": audio_column,
            "all_rank_deletions": all_rank_deletions,
            "results_csv": str(results_path),
            "failures_csv": str(failures_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=None,
        help="Path to committed mllm_shapx spec JSON; falls back to RUN_DIR/spec.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aligner-device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--partition-index", type=int, default=None)
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override spec generation.max_new_tokens, useful for one-sample smoke tests.",
    )
    parser.add_argument(
        "--text-temperature",
        type=float,
        default=None,
        help="Override spec generation.text_temperature.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--all-rank-deletions",
        action="store_true",
        help=(
            "Delete every SGPA segment once and evaluate deletion drop by abs-SV "
            "rank, using existing phase-1 sample JSONs."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate data/SV/alignment/masking/audio encoding without loading LFM2.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Combine partition CSVs in --output-dir and write aggregate summaries.",
    )
    return parser


def main() -> None:
    """Run CLI."""
    args = build_argparser().parse_args()
    if args.combine_only:
        summary = combine_partition_outputs(
            args.output_dir, all_rank_deletions=args.all_rank_deletions
        )
        print(json.dumps(summary, indent=2))
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --combine-only is used.")

    summary = run_faithfulness(
        run_dir=args.run_dir,
        spec_path=args.spec_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
        aligner_device=args.aligner_device,
        seed=args.seed,
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
        resume=args.resume,
        preflight_only=args.preflight_only,
        all_rank_deletions=args.all_rank_deletions,
        max_new_tokens_override=args.max_new_tokens,
        text_temperature_override=args.text_temperature,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
