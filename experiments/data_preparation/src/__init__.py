"""Convenience re-exports for the data-preparation package.

Expose commonly used helpers for the notebooks so they can be imported
via ``from experiments.data_preparation import ...`` in the demos.
"""

from .audio import (
    calculate_audio_duration,
    normalize_original_audio_entry,
    synthesize_en_voices,
    synthesize_voices,
    to_audio_bytes_and_duration,
)
from .audio_alignment_demo import (
    create_sgpa_demo,
    display_random_alignment_example,
    display_token_alignment,
)
from .google_tts_demo import (
    build_configured_samples_dataframe,
    display_all_tts_samples,
    display_tts_sample_row,
    filter_voices_for_demo,
    list_voices_dataframe,
)
from .hub_overview import (
    inspect_random_test_row,
    load_hub_dataset,
    pick_random_config,
    play_first_audio_clip,
    show_prompt,
)
from .constants import (
    DATA_DIR,
    EXPERIMENTS_ROOT_DIR,
    INFINITY_INSTRUCT__CONFIG,
    LIBRISPEECH_ASR__CONFIG,
    TTS_CONFIGS,
    VOICE_BENCH__CONFIG,
    DatasetConfig,
    TTSConfig,
)
from .filters import (
    is_interesting_rule_based,
    nlp_quality_filter,
    semantic_dedup,
)
from .infinity_instruct_pipeline import (
    augment_with_translations,
    build_infinity_dataframe,
    build_multilingual_base,
    dedupe_conversations_semhash,
    filter_by_allowed_abilities,
    filter_by_token_count,
    filter_infinity_languages,
    filter_max_sentence_length,
    filter_max_turns,
    finalize_multilingual_for_save,
    languages_with_min_population,
    message_count_distribution,
    plot_ability_exclusion_curve,
    split_conversation_into_sentences,
    synthesize_multilingual_voices,
    verify_non_english_languages,
)
from .io import ensure_dir, load_dataset, save_json, save_parquet
from .languages import LanguageClassifier, LanguageTranslator
from .librispeech_loaders import attach_librispeech_audio, load_librispeech_text_pool
from .multi_sentence_pipeline import (
    prepare_multi_sentence_candidates,
    sample_multi_sentence,
)
from .nlp import TTS, split_into_sentences
from .preprocessing import (
    add_datasets_combined,
    add_english_flag,
    add_sentence_columns,
    dedupe_by_prompt,
    filter_multi_sentence,
    filter_single_sentence,
    non_english_prompts,
)
from .reporting import (
    plot_interestingness_distribution,
    plot_token_sampling_stages,
    print_datasets_list_length_counts,
    report_dataset_stats,
    value_counts_at_least,
)
from .sampling import (
    sample_fraction_by_group,
    sample_fraction_by_groups,
    sample_n_per_group,
    stratified_sample,
)
from .save import (
    prepare_for_save,
    save_dataset_and_sample,
    save_multi_sentence,
    save_single_sentence,
)
from .setup import configure_notebook_environment, get_device, get_token_model
from .single_sentence_pipeline import (
    prepare_candidates,
    run_nlp_quality_filter,
    run_token_count_filter,
    sample_single_sentence_100,
    sample_single_sentence_nk,
)
from .statistics import (
    compute_budgets,
    get_df_stats,
    get_df_stats__by_source,
    get_sample_df,
    plot_token_count_comparison,
)
from .tokens import (
    add_token_counts_and_filter,
    compute_multi_turn_token_counts,
    compute_token_counts,
)
from .voice_bench_loaders import load_voicebench_dataframe

__all__ = [
    # audio
    "calculate_audio_duration",
    "normalize_original_audio_entry",
    "synthesize_en_voices",
    "synthesize_voices",
    "to_audio_bytes_and_duration",
    # audio alignment demo
    "create_sgpa_demo",
    "display_random_alignment_example",
    "display_token_alignment",
    # google tts demo
    "build_configured_samples_dataframe",
    "display_all_tts_samples",
    "display_tts_sample_row",
    "filter_voices_for_demo",
    "list_voices_dataframe",
    # hub overview
    "inspect_random_test_row",
    "load_hub_dataset",
    "pick_random_config",
    "play_first_audio_clip",
    "show_prompt",
    # constants
    "DATA_DIR",
    "EXPERIMENTS_ROOT_DIR",
    "INFINITY_INSTRUCT__CONFIG",
    "LIBRISPEECH_ASR__CONFIG",
    "TTS_CONFIGS",
    "VOICE_BENCH__CONFIG",
    "DatasetConfig",
    "TTSConfig",
    # filters
    "is_interesting_rule_based",
    "nlp_quality_filter",
    "semantic_dedup",
    # infinity instruct pipeline
    "augment_with_translations",
    "build_infinity_dataframe",
    "build_multilingual_base",
    "dedupe_conversations_semhash",
    "filter_by_allowed_abilities",
    "filter_by_token_count",
    "filter_infinity_languages",
    "filter_max_sentence_length",
    "filter_max_turns",
    "finalize_multilingual_for_save",
    "languages_with_min_population",
    "message_count_distribution",
    "plot_ability_exclusion_curve",
    "split_conversation_into_sentences",
    "synthesize_multilingual_voices",
    "verify_non_english_languages",
    # io
    "ensure_dir",
    "load_dataset",
    "save_json",
    "save_parquet",
    # languages
    "LanguageClassifier",
    "LanguageTranslator",
    # librispeech loaders
    "attach_librispeech_audio",
    "load_librispeech_text_pool",
    # multi sentence pipeline
    "prepare_multi_sentence_candidates",
    "sample_multi_sentence",
    # nlp
    "TTS",
    "split_into_sentences",
    # preprocessing
    "add_datasets_combined",
    "add_english_flag",
    "add_sentence_columns",
    "dedupe_by_prompt",
    "filter_multi_sentence",
    "filter_single_sentence",
    "non_english_prompts",
    # reporting
    "plot_interestingness_distribution",
    "plot_token_sampling_stages",
    "print_datasets_list_length_counts",
    "report_dataset_stats",
    "value_counts_at_least",
    # sampling
    "sample_fraction_by_group",
    "sample_fraction_by_groups",
    "sample_n_per_group",
    "stratified_sample",
    # save
    "prepare_for_save",
    "save_dataset_and_sample",
    "save_multi_sentence",
    "save_single_sentence",
    # setup
    "configure_notebook_environment",
    "get_device",
    "get_token_model",
    # single sentence pipeline
    "prepare_candidates",
    "run_nlp_quality_filter",
    "run_token_count_filter",
    "sample_single_sentence_100",
    "sample_single_sentence_nk",
    # statistics
    "compute_budgets",
    "get_df_stats",
    "get_df_stats__by_source",
    "get_sample_df",
    "plot_token_count_comparison",
    # tokens
    "add_token_counts_and_filter",
    "compute_multi_turn_token_counts",
    "compute_token_counts",
    # voice bench loaders
    "load_voicebench_dataframe",
]
