"""Single-sample faithfulness evaluation runners."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler
from scipy import stats

from experiments.mllm_shapx.src.constants import InputModality
from experiments.mllm_shapx.src.data import extract_texts_from_row

from .helpers import (
    aggregate_sv_to_segments,
    as_list,
    embedding_similarities,
    extract_audio_sv,
    generate_response,
    parse_sample_id,
    quantile_bins,
    rank_abs_sv,
    response_drop,
    remove_interval,
    sample_random_set_matching_targets,
    segment_interval,
    sequence_match_similarities,
    tfidf_similarities,
)
from .models import FaithfulnessResult, RankwiseDeletionResult


def run_one_sample(
    sample_path: Path,
    row: dict[str, Any],
    model: Any,
    aligner: SpectrogramGuidedAligner,
    input_modality: InputModality,
    audio_column: str,
    max_new_tokens: int,
    text_temperature: float,
    rng: np.random.Generator,
    random_draws: int,
    strat_duration_bins: int,
    strat_position_bins: int,
    comprehensiveness_k: int,
    sufficiency_k: int,
) -> FaithfulnessResult:
    """Run faithfulness tests for a single sample.

    This executes the HP-1 deletion protocol for one SGPA sample. It:
    - aligns audio segments,
    - aggregates Shapley values to segments and selects top positive/negative
        segments using raw signed SV,
    - computes model responses for deletions/keeps and for random baselines
        (uniform and stratified),
    - computes embedding/tfidf/sequence similarity drops and monotonicity
        statistics, and
    - returns a `FaithfulnessResult` dataclass with per-sample measurements.

    Parameters mirror the CLI wiring in `run.py`.
    """
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
    sv_arr = np.asarray(segment_sv_values, dtype=float)
    top_pos_idx = int(np.argmax(sv_arr))
    top_neg_idx = int(np.argmin(sv_arr))
    top_pos_sv = float(sv_arr[top_pos_idx])
    top_neg_sv = float(sv_arr[top_neg_idx])

    if len(segments) < 2:
        raise ValueError("Need at least two segments for random baseline.")

    intervals = [segment_interval(seg) for seg in segments]
    durations = np.asarray([max(1, e - s) for s, e in intervals], dtype=float)
    centers = np.asarray([(s + e) / 2.0 for s, e in intervals], dtype=float)
    duration_bins = quantile_bins(durations, strat_duration_bins)
    position_bins = quantile_bins(centers, strat_position_bins)

    def _audio_from_mask(mode: str, indices: tuple[int, ...]) -> bytes:
        """Generate modified audio bytes by either deleting or keeping specified
        segments based on the mode and indices."""
        if mode not in {"delete", "keep"}:
            raise ValueError(f"Unsupported mode {mode!r}")
        selected = set(int(i) for i in indices)
        if mode == "delete":
            kept = [
                waveform[:, s:e]
                for idx, (s, e) in enumerate(intervals)
                if idx not in selected
            ]
        else:
            kept = [
                waveform[:, s:e]
                for idx, (s, e) in enumerate(intervals)
                if idx in selected
            ]
        masked = torch.cat(kept, dim=1) if kept else torch.zeros(1, 1)
        return TorchAudioHandler.to_bytes(
            masked,
            sample_rate=int(sample_rate),
            audio_format="wav",
        )

    audio_cache: dict[tuple[str, tuple[int, ...]], bytes] = {}
    response_cache: dict[tuple[str, tuple[int, ...]], Any] = {}

    def _get_response(mode: str, indices: list[int]) -> Any:
        """Get the model response for a given modification mode and segment indices,
        using caching to avoid redundant computations."""
        key = (mode, tuple(sorted(set(int(i) for i in indices))))
        if key not in audio_cache:
            audio_cache[key] = _audio_from_mask(key[0], key[1])
        if key not in response_cache:
            response_cache[key] = generate_response(
                model,
                audio_cache[key],
                input_modality,
                max_new_tokens,
                text_temperature,
                user_texts=user_texts,
            )
        return response_cache[key]

    def _compute_metric_sims(
        specs: list[tuple[str, list[int]]],
    ) -> dict[str, tuple[float, list[float]]]:
        """Compute original and modified similarities for a list of modification specifications,
        where each spec is a tuple of (mode, segment_indices). Returns a dictionary mapping
        metric names to tuples of (original_similarity, list_of_modified_similarities)."""
        responses = [_get_response(mode, idxs) for mode, idxs in specs]
        emb_orig, emb_sims = embedding_similarities(model, base_resp, responses)
        tfidf_orig, tfidf_sims = tfidf_similarities(model, base_resp, responses)
        seq_orig, seq_sims = sequence_match_similarities(model, base_resp, responses)
        return {
            "embedding": (emb_orig, emb_sims),
            "tfidf": (tfidf_orig, tfidf_sims),
            "seqmatch": (seq_orig, seq_sims),
        }

    def _mean_random_drops(
        mode: str,
        target_indices: list[int],
        baseline_type: str,
        draws: int,
        exclude: set[int],
    ) -> dict[str, float]:
        """Compute mean similarity drops for random modifications matching the target segments,
        using the specified baseline type and number of draws.
        This samples random segment sets that match the duration and position
        bins of the target segments, computes their responses and similarities,
        and returns the average drops for each metric.
        """
        if draws <= 0:
            raise ValueError("random_draws must be positive.")
        specs: list[tuple[str, list[int]]] = []
        for _ in range(draws):
            rand_set = sample_random_set_matching_targets(
                target_indices=target_indices,
                n=len(segments),
                duration_bins=duration_bins,
                position_bins=position_bins,
                rng=rng,
                baseline_type=baseline_type,
                global_exclude=exclude,
            )
            specs.append((mode, rand_set))

        sims = _compute_metric_sims(specs)
        out: dict[str, float] = {}
        for metric_name, (orig_val, sim_vals) in sims.items():
            drops = [response_drop(orig_val, s) for s in sim_vals]
            out[metric_name] = float(np.mean(drops))
        return out

    t0 = time.perf_counter()
    base_resp = generate_response(
        model,
        audio_bytes,
        input_modality,
        max_new_tokens,
        text_temperature,
        user_texts=user_texts,
    )

    pos_specs = [("delete", [top_pos_idx])]
    neg_specs = [("delete", [top_neg_idx])]

    positive_sorted = [
        int(idx) for idx in np.argsort(-sv_arr) if float(sv_arr[idx]) > 0.0
    ]
    if not positive_sorted:
        positive_sorted = [top_pos_idx]
    comp_k = max(
        1, min(int(comprehensiveness_k), len(positive_sorted), len(segments) - 1)
    )
    suff_k = max(1, min(int(sufficiency_k), len(positive_sorted), len(segments) - 1))
    top_comp_indices = positive_sorted[:comp_k]
    top_suff_indices = positive_sorted[:suff_k]

    comp_specs = [("delete", top_comp_indices)]
    suff_specs = [("keep", top_suff_indices)]

    rank_order = np.argsort(-sv_arr).tolist()
    mono_specs = [("delete", [int(idx)]) for idx in rank_order]

    pos_metrics = _compute_metric_sims(pos_specs)
    neg_metrics = _compute_metric_sims(neg_specs)
    comp_metrics = _compute_metric_sims(comp_specs)
    suff_metrics = _compute_metric_sims(suff_specs)
    mono_metrics = _compute_metric_sims(mono_specs)

    pos_uniform_random = _mean_random_drops(
        mode="delete",
        target_indices=[top_pos_idx],
        baseline_type="uniform_random",
        draws=random_draws,
        exclude={top_pos_idx},
    )
    pos_stratified_random = _mean_random_drops(
        mode="delete",
        target_indices=[top_pos_idx],
        baseline_type="stratified_random",
        draws=random_draws,
        exclude={top_pos_idx},
    )
    neg_uniform_random = _mean_random_drops(
        mode="delete",
        target_indices=[top_neg_idx],
        baseline_type="uniform_random",
        draws=random_draws,
        exclude={top_neg_idx},
    )
    neg_stratified_random = _mean_random_drops(
        mode="delete",
        target_indices=[top_neg_idx],
        baseline_type="stratified_random",
        draws=random_draws,
        exclude={top_neg_idx},
    )
    comp_uniform_random = _mean_random_drops(
        mode="delete",
        target_indices=top_comp_indices,
        baseline_type="uniform_random",
        draws=random_draws,
        exclude=set(top_comp_indices),
    )
    comp_stratified_random = _mean_random_drops(
        mode="delete",
        target_indices=top_comp_indices,
        baseline_type="stratified_random",
        draws=random_draws,
        exclude=set(top_comp_indices),
    )
    suff_uniform_random = _mean_random_drops(
        mode="keep",
        target_indices=top_suff_indices,
        baseline_type="uniform_random",
        draws=random_draws,
        exclude=set(top_suff_indices),
    )
    suff_stratified_random = _mean_random_drops(
        mode="keep",
        target_indices=top_suff_indices,
        baseline_type="stratified_random",
        draws=random_draws,
        exclude=set(top_suff_indices),
    )

    def _mono_stats(
        orig_val: float, sims: list[float]
    ) -> tuple[float | None, float | None, float | None]:
        drops = np.asarray([response_drop(orig_val, s) for s in sims], dtype=float)
        if drops.size < 2:
            return None, None, None
        ranks = np.arange(1, drops.size + 1, dtype=float)
        rho = stats.spearmanr(ranks, drops).statistic
        rho = float(rho) if np.isfinite(rho) else None
        score = float(-rho) if rho is not None else None
        violation_rate = (
            float(np.mean(drops[1:] > drops[:-1])) if drops.size > 1 else None
        )
        return rho, score, violation_rate

    emb_orig = float(pos_metrics["embedding"][0])
    emb_pos_sim = float(pos_metrics["embedding"][1][0])
    emb_neg_sim = float(neg_metrics["embedding"][1][0])
    emb_comp_sim = float(comp_metrics["embedding"][1][0])
    emb_suff_sim = float(suff_metrics["embedding"][1][0])

    tfidf_orig = float(pos_metrics["tfidf"][0])
    tfidf_pos_sim = float(pos_metrics["tfidf"][1][0])
    tfidf_neg_sim = float(neg_metrics["tfidf"][1][0])
    tfidf_comp_sim = float(comp_metrics["tfidf"][1][0])
    tfidf_suff_sim = float(suff_metrics["tfidf"][1][0])

    seq_orig = float(pos_metrics["seqmatch"][0])
    seq_pos_sim = float(pos_metrics["seqmatch"][1][0])

    top_drop = response_drop(emb_orig, emb_pos_sim)
    mean_random_drop = float(pos_uniform_random["embedding"])
    drop_difference = float(top_drop - mean_random_drop)
    pos_stratified_mean_random_drop = float(pos_stratified_random["embedding"])
    pos_stratified_drop_difference = float(top_drop - pos_stratified_mean_random_drop)

    neg_drop = response_drop(emb_orig, emb_neg_sim)
    neg_mean_random_drop = float(neg_uniform_random["embedding"])
    neg_drop_improvement = float(neg_mean_random_drop - neg_drop)
    neg_stratified_mean_random_drop = float(neg_stratified_random["embedding"])
    neg_stratified_drop_improvement = float(neg_stratified_mean_random_drop - neg_drop)

    comp_drop = response_drop(emb_orig, emb_comp_sim)
    comp_mean_random_drop = float(comp_uniform_random["embedding"])
    comp_drop_diff = float(comp_drop - comp_mean_random_drop)
    comp_strat_mean_random_drop = float(comp_stratified_random["embedding"])
    comp_strat_drop_diff = float(comp_drop - comp_strat_mean_random_drop)

    suff_drop = response_drop(emb_orig, emb_suff_sim)
    suff_mean_random_drop = float(suff_uniform_random["embedding"])
    suff_adv = float(suff_mean_random_drop - suff_drop)
    suff_strat_mean_random_drop = float(suff_stratified_random["embedding"])
    suff_strat_adv = float(suff_strat_mean_random_drop - suff_drop)

    tfidf_top_drop = response_drop(tfidf_orig, tfidf_pos_sim)
    tfidf_mean_random_drop = float(pos_uniform_random["tfidf"])
    tfidf_drop_difference = float(tfidf_top_drop - tfidf_mean_random_drop)
    tfidf_pos_strat_random_drop = float(pos_stratified_random["tfidf"])
    tfidf_pos_strat_drop_difference = float(
        tfidf_top_drop - tfidf_pos_strat_random_drop
    )

    tfidf_neg_drop = response_drop(tfidf_orig, tfidf_neg_sim)
    tfidf_neg_mean_random_drop = float(neg_uniform_random["tfidf"])
    tfidf_neg_drop_improvement = float(tfidf_neg_mean_random_drop - tfidf_neg_drop)
    tfidf_neg_stratified_mean_random_drop = float(neg_stratified_random["tfidf"])
    tfidf_neg_stratified_drop_improvement = float(
        tfidf_neg_stratified_mean_random_drop - tfidf_neg_drop
    )

    tfidf_comp_drop = response_drop(tfidf_orig, tfidf_comp_sim)
    tfidf_comp_mean_random_drop = float(comp_uniform_random["tfidf"])
    tfidf_comp_drop_difference = float(tfidf_comp_drop - tfidf_comp_mean_random_drop)
    tfidf_comp_strat_mean_random_drop = float(comp_stratified_random["tfidf"])
    tfidf_comp_strat_drop_difference = float(
        tfidf_comp_drop - tfidf_comp_strat_mean_random_drop
    )

    tfidf_suff_drop = response_drop(tfidf_orig, tfidf_suff_sim)
    tfidf_suff_mean_random_drop = float(suff_uniform_random["tfidf"])
    tfidf_suff_advantage = float(tfidf_suff_mean_random_drop - tfidf_suff_drop)
    tfidf_suff_strat_mean_random_drop = float(suff_stratified_random["tfidf"])
    tfidf_suff_strat_advantage = float(
        tfidf_suff_strat_mean_random_drop - tfidf_suff_drop
    )

    seqmatch_mean_random_drop = float(pos_uniform_random["seqmatch"])
    seqmatch_top_drop = response_drop(seq_orig, seq_pos_sim)
    seqmatch_drop_difference = float(seqmatch_top_drop - seqmatch_mean_random_drop)

    mono_emb = _mono_stats(mono_metrics["embedding"][0], mono_metrics["embedding"][1])
    mono_tfidf = _mono_stats(mono_metrics["tfidf"][0], mono_metrics["tfidf"][1])

    top_start, top_end = intervals[top_pos_idx]
    mask_duration = max(1, top_end - top_start)

    return FaithfulnessResult(
        sample_id=sample_id,
        row_index=row_index,
        audio_column=audio_column,
        transcript=transcript,
        n_segments=len(segments),
        top_segment_idx=top_pos_idx,
        top_segment_token=segments[top_pos_idx].token,
        top_abs_sv=abs(top_pos_sv),
        top_sv=top_pos_sv,
        top_positive_idx=top_pos_idx,
        top_positive_sv=top_pos_sv,
        top_negative_idx=top_neg_idx,
        top_negative_sv=top_neg_sv,
        random_draws=int(random_draws),
        strat_duration_bins=int(strat_duration_bins),
        strat_position_bins=int(strat_position_bins),
        comprehensiveness_k=int(comp_k),
        sufficiency_k=int(suff_k),
        original_similarity=emb_orig,
        top_similarity=emb_pos_sim,
        mean_random_similarity=emb_orig - mean_random_drop,
        top_drop=top_drop,
        mean_random_drop=mean_random_drop,
        drop_difference=drop_difference,
        pos_stratified_mean_random_similarity=emb_orig
        - pos_stratified_mean_random_drop,
        pos_stratified_mean_random_drop=pos_stratified_mean_random_drop,
        pos_stratified_drop_difference=pos_stratified_drop_difference,
        neg_similarity=emb_neg_sim,
        neg_drop=neg_drop,
        neg_mean_random_drop=neg_mean_random_drop,
        neg_drop_improvement=neg_drop_improvement,
        neg_stratified_mean_random_drop=neg_stratified_mean_random_drop,
        neg_stratified_drop_improvement=neg_stratified_drop_improvement,
        comprehensiveness_similarity=emb_comp_sim,
        comprehensiveness_drop=comp_drop,
        comprehensiveness_mean_random_drop=comp_mean_random_drop,
        comprehensiveness_drop_difference=comp_drop_diff,
        comprehensiveness_stratified_mean_random_drop=comp_strat_mean_random_drop,
        comprehensiveness_stratified_drop_difference=comp_strat_drop_diff,
        sufficiency_similarity=emb_suff_sim,
        sufficiency_drop=suff_drop,
        sufficiency_mean_random_drop=suff_mean_random_drop,
        sufficiency_advantage=suff_adv,
        sufficiency_stratified_mean_random_drop=suff_strat_mean_random_drop,
        sufficiency_stratified_advantage=suff_strat_adv,
        monotonicity_spearman=mono_emb[0],
        monotonicity_score=mono_emb[1],
        monotonicity_violation_rate=mono_emb[2],
        tfidf_original_sim=tfidf_orig,
        tfidf_top_sim=tfidf_pos_sim,
        tfidf_mean_random_sim=tfidf_orig - tfidf_mean_random_drop,
        tfidf_top_drop=tfidf_top_drop,
        tfidf_mean_random_drop=tfidf_mean_random_drop,
        tfidf_drop_difference=tfidf_drop_difference,
        tfidf_pos_stratified_mean_random_sim=tfidf_orig - tfidf_pos_strat_random_drop,
        tfidf_pos_stratified_mean_random_drop=tfidf_pos_strat_random_drop,
        tfidf_pos_stratified_drop_difference=tfidf_pos_strat_drop_difference,
        tfidf_neg_similarity=tfidf_neg_sim,
        tfidf_neg_drop=tfidf_neg_drop,
        tfidf_neg_mean_random_drop=tfidf_neg_mean_random_drop,
        tfidf_neg_drop_improvement=tfidf_neg_drop_improvement,
        tfidf_neg_stratified_mean_random_drop=tfidf_neg_stratified_mean_random_drop,
        tfidf_neg_stratified_drop_improvement=tfidf_neg_stratified_drop_improvement,
        tfidf_comprehensiveness_similarity=tfidf_comp_sim,
        tfidf_comprehensiveness_drop=tfidf_comp_drop,
        tfidf_comprehensiveness_mean_random_drop=tfidf_comp_mean_random_drop,
        tfidf_comprehensiveness_drop_difference=tfidf_comp_drop_difference,
        tfidf_comprehensiveness_stratified_mean_random_drop=tfidf_comp_strat_mean_random_drop,
        tfidf_comprehensiveness_stratified_drop_difference=tfidf_comp_strat_drop_difference,
        tfidf_sufficiency_similarity=tfidf_suff_sim,
        tfidf_sufficiency_drop=tfidf_suff_drop,
        tfidf_sufficiency_mean_random_drop=tfidf_suff_mean_random_drop,
        tfidf_sufficiency_advantage=tfidf_suff_advantage,
        tfidf_sufficiency_stratified_mean_random_drop=tfidf_suff_strat_mean_random_drop,
        tfidf_sufficiency_stratified_advantage=tfidf_suff_strat_advantage,
        tfidf_monotonicity_spearman=mono_tfidf[0],
        tfidf_monotonicity_score=mono_tfidf[1],
        tfidf_monotonicity_violation_rate=mono_tfidf[2],
        seqmatch_top_sim=seq_pos_sim,
        seqmatch_mean_random_sim=seq_orig - seqmatch_mean_random_drop,
        seqmatch_top_drop=seqmatch_top_drop,
        seqmatch_mean_random_drop=seqmatch_mean_random_drop,
        seqmatch_drop_difference=seqmatch_drop_difference,
        top_start_sec=float(top_start / sample_rate),
        top_end_sec=float(top_end / sample_rate),
        top_mask_duration_sec=float(mask_duration / sample_rate),
        n_random_segments=len(segments) - 1,
        runtime_sec=float(time.perf_counter() - t0),
    )


def run_one_sample_rankwise(
    sample_path: Path,
    row: dict[str, Any],
    model: Any,
    aligner: SpectrogramGuidedAligner,
    input_modality: InputModality,
    audio_column: str,
    max_new_tokens: int,
    text_temperature: float,
) -> list[RankwiseDeletionResult]:
    """Compute rank-wise (cumulative) deletion results for one sample.

    Removes segments cumulatively in descending absolute-SV order and records
    embedding/tfidf/sequence similarity for each single-segment deletion and
    for each cumulative removal (top-k removed). Returns a list of
    `RankwiseDeletionResult` entries (one per original segment).
    """
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
