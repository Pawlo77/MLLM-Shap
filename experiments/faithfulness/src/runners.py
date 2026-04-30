"""Single-sample faithfulness evaluation runners."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler

from experiments.mllm_shapx import InputModality
from experiments.mllm_shapx.src import extract_texts_from_row

from .helpers import (
    aggregate_sv_to_segments,
    as_list,
    embedding_similarities,
    extract_audio_sv,
    generate_response,
    parse_sample_id,
    rank_abs_sv,
    remove_interval,
    segment_interval,
    sequence_match_similarities,
    tfidf_similarities,
)
from .models import FaithfulnessResult, RankwiseDeletionResult


def run_one_sample(
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
    sample_id = parse_sample_id(sample_path)
    sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
    row_index = int(sample_json.get("row_index", sample_id))
    user_texts = extract_texts_from_row(row[row["_text_col"]])
    transcript = " ".join(user_texts)
    audio_bytes = as_list(row[audio_column])[0]

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

    sv_values = extract_audio_sv(sample_json)
    segment_sv_values, _ = aggregate_sv_to_segments(
        sv_values, segments, total_samples=waveform.size(-1)
    )
    top_idx = int(np.argmax(np.abs(np.asarray(segment_sv_values, dtype=float))))
    top_sv = float(segment_sv_values[top_idx])

    candidate_indices = [i for i in range(len(segments)) if i != top_idx]
    if not candidate_indices:
        raise ValueError("Need at least two segments for random baseline.")

    top_start, top_end = segment_interval(segments[top_idx])
    mask_duration = max(1, top_end - top_start)

    top_audio = TorchAudioHandler.to_bytes(
        remove_interval(waveform, top_start, top_end),
        sample_rate=int(sample_rate),
        audio_format="wav",
    )
    # Generate perturbed audio for all non-top segments (stable random baseline)
    random_audios = []
    for idx in candidate_indices:
        s, e = segment_interval(segments[idx])
        random_audios.append(
            TorchAudioHandler.to_bytes(
                remove_interval(waveform, s, e),
                sample_rate=int(sample_rate),
                audio_format="wav",
            )
        )

    t0 = time.perf_counter()
    base_resp = generate_response(
        model,
        audio_bytes,
        input_modality,
        max_new_tokens,
        text_temperature,
        user_texts=user_texts,
    )
    top_resp = generate_response(
        model,
        top_audio,
        input_modality,
        max_new_tokens,
        text_temperature,
        user_texts=user_texts,
    )
    rand_resps = [
        generate_response(
            model,
            a,
            input_modality,
            max_new_tokens,
            text_temperature,
            user_texts=user_texts,
        )
        for a in random_audios
    ]

    orig_sim, other_sims = embedding_similarities(
        model, base_resp, [top_resp, *rand_resps]
    )
    top_sim = other_sims[0]
    rand_sims = other_sims[1:]
    mean_rand_sim = float(np.mean(rand_sims))

    tfidf_orig, tfidf_other = tfidf_similarities(
        model, base_resp, [top_resp, *rand_resps]
    )
    tfidf_top = tfidf_other[0]
    tfidf_rand_sims = tfidf_other[1:]
    tfidf_mean_rand = float(np.mean(tfidf_rand_sims))

    top_drop = orig_sim - top_sim
    mean_random_drop = orig_sim - mean_rand_sim

    _, seqmatch_other = sequence_match_similarities(
        model, base_resp, [top_resp, *rand_resps]
    )
    seqmatch_top = seqmatch_other[0]
    seqmatch_rand_sims = seqmatch_other[1:]
    seqmatch_mean_rand = float(np.mean(seqmatch_rand_sims))

    return FaithfulnessResult(
        sample_id=sample_id,
        row_index=row_index,
        audio_column=audio_column,
        transcript=transcript,
        n_segments=len(segments),
        top_segment_idx=top_idx,
        top_segment_token=segments[top_idx].token,
        top_abs_sv=abs(top_sv),
        top_sv=top_sv,
        original_similarity=orig_sim,
        top_similarity=top_sim,
        mean_random_similarity=mean_rand_sim,
        top_drop=top_drop,
        mean_random_drop=mean_random_drop,
        drop_difference=top_drop - mean_random_drop,
        tfidf_original_sim=tfidf_orig,
        tfidf_top_sim=tfidf_top,
        tfidf_mean_random_sim=tfidf_mean_rand,
        tfidf_top_drop=tfidf_orig - tfidf_top,
        tfidf_mean_random_drop=tfidf_orig - tfidf_mean_rand,
        tfidf_drop_difference=(tfidf_orig - tfidf_top) - (tfidf_orig - tfidf_mean_rand),
        seqmatch_top_sim=seqmatch_top,
        seqmatch_mean_random_sim=seqmatch_mean_rand,
        seqmatch_top_drop=1.0 - seqmatch_top,
        seqmatch_mean_random_drop=1.0 - seqmatch_mean_rand,
        seqmatch_drop_difference=(1.0 - seqmatch_top) - (1.0 - seqmatch_mean_rand),
        top_start_sec=float(top_start / sample_rate),
        top_end_sec=float(top_end / sample_rate),
        top_mask_duration_sec=float(mask_duration / sample_rate),
        n_random_segments=len(candidate_indices),
        runtime_sec=float(time.perf_counter() - t0),
    )


def run_one_sample_rankwise(
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
    sample_id = parse_sample_id(sample_path)
    sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
    row_index = int(sample_json.get("row_index", sample_id))
    user_texts = extract_texts_from_row(row[row["_text_col"]])
    transcript = " ".join(user_texts)
    audio_bytes = as_list(row[audio_column])[0]

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

    sv_values = extract_audio_sv(sample_json)
    segment_sv_values, _ = aggregate_sv_to_segments(
        sv_values, segments, total_samples=waveform.size(-1)
    )
    rank_info = rank_abs_sv(segment_sv_values)

    intervals = [segment_interval(seg) for seg in segments]
    deleted_audio = [
        TorchAudioHandler.to_bytes(
            remove_interval(waveform, s, e),
            sample_rate=int(sample_rate),
            audio_format="wav",
        )
        for s, e in intervals
    ]

    # Cumulative deletion: remove segments in descending abs-SV order
    rank_order = rank_info["order"]  # indices sorted by descending abs SV
    cumulative_audios: list[bytes] = []
    for k in range(1, len(segments) + 1):
        # Remove top-k segments by abs SV
        removed_indices = set(int(rank_order[i]) for i in range(k))
        pieces = [
            waveform[:, s:e]
            for idx, (s, e) in enumerate(intervals)
            if idx not in removed_indices
        ]
        if pieces:
            cum_waveform = torch.cat(pieces, dim=1)
        else:
            cum_waveform = torch.zeros(1, 1)
        cumulative_audios.append(
            TorchAudioHandler.to_bytes(
                cum_waveform,
                sample_rate=int(sample_rate),
                audio_format="wav",
            )
        )

    t0 = time.perf_counter()
    base_resp = generate_response(
        model,
        audio_bytes,
        input_modality,
        max_new_tokens,
        text_temperature,
        user_texts=user_texts,
    )
    del_responses = [
        generate_response(
            model,
            a,
            input_modality,
            max_new_tokens,
            text_temperature,
            user_texts=user_texts,
        )
        for a in deleted_audio
    ]
    cum_responses = [
        generate_response(
            model,
            a,
            input_modality,
            max_new_tokens,
            text_temperature,
            user_texts=user_texts,
        )
        for a in cumulative_audios
    ]
    orig_sim, del_sims = embedding_similarities(model, base_resp, del_responses)
    _, cum_sims = embedding_similarities(model, base_resp, cum_responses)
    tfidf_orig, tfidf_del_sims = tfidf_similarities(model, base_resp, del_responses)
    _, tfidf_cum_sims = tfidf_similarities(model, base_resp, cum_responses)
    _, seqmatch_del_sims = sequence_match_similarities(model, base_resp, del_responses)
    _, seqmatch_cum_sims = sequence_match_similarities(model, base_resp, cum_responses)
    runtime = float(time.perf_counter() - t0)

    # Build a mapping: for each segment, what is its cumulative position?
    # rank_order[0] is the segment with rank 1 (highest abs SV).
    # After removing top-k segments, the cumulative similarity is cum_sims[k-1].
    # So for segment at rank r, cumulative = cum_sims[r-1] (removing top-r).
    results: list[RankwiseDeletionResult] = []
    for idx, (seg, del_sim, tfidf_del, seqmatch_del) in enumerate(
        zip(segments, del_sims, tfidf_del_sims, seqmatch_del_sims)
    ):
        start, end = intervals[idx]
        rank = int(rank_info["ranks"][idx])  # 1-indexed
        cum_sim = cum_sims[rank - 1]
        tfidf_cum_sim = tfidf_cum_sims[rank - 1]
        seqmatch_cum = seqmatch_cum_sims[rank - 1]
        results.append(
            RankwiseDeletionResult(
                sample_id=sample_id,
                row_index=row_index,
                audio_column=audio_column,
                transcript=transcript,
                n_segments=len(segments),
                segment_idx=idx,
                segment_rank_abs_sv=rank,
                segment_token=seg.token,
                segment_sv=float(segment_sv_values[idx]),
                segment_abs_sv=float(rank_info["abs_values"][idx]),
                segment_abs_sv_share=float(rank_info["shares"][idx]),
                top_abs_sv=float(rank_info["top_abs"]),
                top1_top2_gap=rank_info["top1_top2_gap"],
                top1_top2_ratio=rank_info["top1_top2_ratio"],
                top1_share=float(rank_info["top1_share"]),
                abs_sv_entropy_norm=rank_info["abs_sv_entropy_norm"],
                abs_sv_gini=float(rank_info["abs_sv_gini"]),
                original_similarity=orig_sim,
                deleted_similarity=del_sim,
                deletion_drop=orig_sim - del_sim,
                cumulative_similarity=cum_sim,
                cumulative_drop=orig_sim - cum_sim,
                cumulative_n_deleted=rank,
                tfidf_original_sim=tfidf_orig,
                tfidf_deleted_sim=tfidf_del,
                tfidf_deletion_drop=tfidf_orig - tfidf_del,
                tfidf_cumulative_sim=tfidf_cum_sim,
                tfidf_cumulative_drop=tfidf_orig - tfidf_cum_sim,
                seqmatch_deleted_sim=seqmatch_del,
                seqmatch_deletion_drop=1.0 - seqmatch_del,
                seqmatch_cumulative_sim=seqmatch_cum,
                seqmatch_cumulative_drop=1.0 - seqmatch_cum,
                segment_start_sec=float(start / sample_rate),
                segment_end_sec=float(end / sample_rate),
                mask_duration_sec=float(max(1, end - start) / sample_rate),
                runtime_sec=runtime / max(1, len(segments)),
            )
        )
    return results
