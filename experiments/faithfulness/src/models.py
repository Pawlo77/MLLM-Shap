"""Dataclasses for faithfulness evaluation results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FaithfulnessResult:
    """Per-sample output for faithfulness checks (top/random, k-set and monotonicity)."""

    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    n_segments: int
    top_segment_idx: int
    top_segment_token: str
    top_abs_sv: float
    top_sv: float
    top_positive_idx: int
    top_positive_sv: float
    top_negative_idx: int
    top_negative_sv: float
    random_draws: int
    strat_duration_bins: int
    strat_position_bins: int
    comprehensiveness_k: int
    sufficiency_k: int
    original_similarity: float
    top_similarity: float
    mean_random_similarity: float
    top_drop: float
    mean_random_drop: float
    drop_difference: float
    pos_stratified_mean_random_similarity: float
    pos_stratified_mean_random_drop: float
    pos_stratified_drop_difference: float
    neg_similarity: float
    neg_drop: float
    neg_mean_random_drop: float
    neg_drop_improvement: float
    neg_stratified_mean_random_drop: float
    neg_stratified_drop_improvement: float
    comprehensiveness_similarity: float
    comprehensiveness_drop: float
    comprehensiveness_mean_random_drop: float
    comprehensiveness_drop_difference: float
    comprehensiveness_stratified_mean_random_drop: float
    comprehensiveness_stratified_drop_difference: float
    sufficiency_similarity: float
    sufficiency_drop: float
    sufficiency_mean_random_drop: float
    sufficiency_advantage: float
    sufficiency_stratified_mean_random_drop: float
    sufficiency_stratified_advantage: float
    monotonicity_spearman: float | None
    monotonicity_score: float | None
    monotonicity_violation_rate: float | None
    tfidf_original_sim: float
    tfidf_top_sim: float
    tfidf_mean_random_sim: float
    tfidf_top_drop: float
    tfidf_mean_random_drop: float
    tfidf_drop_difference: float
    tfidf_pos_stratified_mean_random_sim: float
    tfidf_pos_stratified_mean_random_drop: float
    tfidf_pos_stratified_drop_difference: float
    tfidf_neg_similarity: float
    tfidf_neg_drop: float
    tfidf_neg_mean_random_drop: float
    tfidf_neg_drop_improvement: float
    tfidf_neg_stratified_mean_random_drop: float
    tfidf_neg_stratified_drop_improvement: float
    tfidf_comprehensiveness_similarity: float
    tfidf_comprehensiveness_drop: float
    tfidf_comprehensiveness_mean_random_drop: float
    tfidf_comprehensiveness_drop_difference: float
    tfidf_comprehensiveness_stratified_mean_random_drop: float
    tfidf_comprehensiveness_stratified_drop_difference: float
    tfidf_sufficiency_similarity: float
    tfidf_sufficiency_drop: float
    tfidf_sufficiency_mean_random_drop: float
    tfidf_sufficiency_advantage: float
    tfidf_sufficiency_stratified_mean_random_drop: float
    tfidf_sufficiency_stratified_advantage: float
    tfidf_monotonicity_spearman: float | None
    tfidf_monotonicity_score: float | None
    tfidf_monotonicity_violation_rate: float | None
    seqmatch_top_sim: float
    seqmatch_mean_random_sim: float
    seqmatch_top_drop: float
    seqmatch_mean_random_drop: float
    seqmatch_drop_difference: float
    seqmatch_neg_sim: float
    seqmatch_neg_drop: float
    seqmatch_neg_mean_random_drop: float
    seqmatch_neg_drop_improvement: float
    seqmatch_comprehensiveness_sim: float
    seqmatch_comprehensiveness_drop: float
    seqmatch_comprehensiveness_mean_random_drop: float
    seqmatch_comprehensiveness_drop_difference: float
    seqmatch_sufficiency_sim: float
    seqmatch_sufficiency_drop: float
    seqmatch_sufficiency_mean_random_drop: float
    seqmatch_sufficiency_advantage: float
    seqmatch_monotonicity_spearman: float | None
    seqmatch_monotonicity_score: float | None
    seqmatch_monotonicity_violation_rate: float | None
    pos_stratified_strict_match_rate: float | None
    neg_stratified_strict_match_rate: float | None
    comp_stratified_strict_match_rate: float | None
    suff_stratified_strict_match_rate: float | None
    top_start_sec: float
    top_end_sec: float
    top_mask_duration_sec: float
    n_random_segments: int
    runtime_sec: float


@dataclass(frozen=True)
class RankwiseDeletionResult:
    """Per-segment output for rank-wise deletion diagnostics."""

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
    """Failure record capturing sample-level errors during evaluation."""

    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    error_type: str
    error_message: str
