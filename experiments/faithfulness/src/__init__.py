"""Faithfulness evaluation source package."""

from .audio import (
    aggregate_sv_to_segments,
    extract_audio_sv,
    remove_interval,
    segment_interval,
)
from .io import (
    experiment_set_from_spec,
    load_selected_rows,
    load_spec,
    parse_sample_id,
    sample_paths,
)
from .models import FailureResult, FaithfulnessResult, RankwiseDeletionResult
from .plot import plot_deletion, plot_rankwise
from .run import (
    DEFAULT_OUTPUT_DIR,
    build_argparser,
    combine_partition_outputs,
    main,
    run_faithfulness,
)
from .sampling import (
    quantile_bins,
    sample_random_set_matching_targets,
    sample_stratified_index,
    sample_uniform_index,
)
from .similarity import (
    embedding_similarities,
    generate_response,
    sequence_match_similarities,
    tfidf_similarities,
)
from .summarize import summarize, summarize_rankwise

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "FailureResult",
    "FaithfulnessResult",
    "RankwiseDeletionResult",
    "aggregate_sv_to_segments",
    "build_argparser",
    "combine_partition_outputs",
    "embedding_similarities",
    "experiment_set_from_spec",
    "extract_audio_sv",
    "generate_response",
    "load_selected_rows",
    "load_spec",
    "main",
    "parse_sample_id",
    "plot_deletion",
    "plot_rankwise",
    "quantile_bins",
    "remove_interval",
    "run_faithfulness",
    "sample_paths",
    "sample_random_set_matching_targets",
    "sample_stratified_index",
    "sample_uniform_index",
    "segment_interval",
    "sequence_match_similarities",
    "summarize",
    "summarize_rankwise",
    "tfidf_similarities",
]
