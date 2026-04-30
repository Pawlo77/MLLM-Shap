"""Dataclasses for faithfulness evaluation results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FaithfulnessResult:
    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    n_segments: int
    top_segment_idx: int
    top_segment_token: str
    top_abs_sv: float
    top_sv: float
    original_similarity: float
    top_similarity: float
    mean_random_similarity: float
    top_drop: float
    mean_random_drop: float
    drop_difference: float
    tfidf_original_sim: float
    tfidf_top_sim: float
    tfidf_mean_random_sim: float
    tfidf_top_drop: float
    tfidf_mean_random_drop: float
    tfidf_drop_difference: float
    seqmatch_top_sim: float
    seqmatch_mean_random_sim: float
    seqmatch_top_drop: float
    seqmatch_mean_random_drop: float
    seqmatch_drop_difference: float
    top_start_sec: float
    top_end_sec: float
    top_mask_duration_sec: float
    n_random_segments: int
    runtime_sec: float


@dataclass(frozen=True)
class RankwiseDeletionResult:
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
    cumulative_similarity: float
    cumulative_drop: float
    cumulative_n_deleted: int
    tfidf_original_sim: float
    tfidf_deleted_sim: float
    tfidf_deletion_drop: float
    tfidf_cumulative_sim: float
    tfidf_cumulative_drop: float
    seqmatch_deleted_sim: float
    seqmatch_deletion_drop: float
    seqmatch_cumulative_sim: float
    seqmatch_cumulative_drop: float
    segment_start_sec: float
    segment_end_sec: float
    mask_duration_sec: float
    runtime_sec: float


@dataclass(frozen=True)
class FailureResult:
    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    error_type: str
    error_message: str
