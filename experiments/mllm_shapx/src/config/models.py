"""Configuration models and parsing logic for experiment execution."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ..constants import (
    DEFAULT_SPLIT,
    DEFAULT_SUBSET,
    DatasetSource,
    InputModality,
    OutputModality,
    TokenFilterType,
)
from ..discovery import (
    get_available_explainer_types,
    is_supported_explainer,
)
from .registry import (
    ALLOWED_HIERARCHICAL_MODES,
    ALLOWED_SHAP_MODES,
    NORMALIZER_MAP,
    REDUCER_MAP,
    SIMILARITY_MAP,
)


class MlflowConfig(BaseModel):
    """MLflow Tracking configuration."""

    tracking_uri: str | None = None
    """MLflow server URI. None uses the default local file store."""
    experiment_name: str = "mllm_shapx"
    """Name of the MLflow experiment to log runs under."""
    nested_per_variant: bool = False
    """Whether to create nested child runs for each explainer variant."""
    system_metrics_enabled: bool = True
    """Log system metrics (CPU, GPU, memory) during experiment runs."""
    tags: List[str] = Field(default_factory=list)
    """Additional tags to attach to each MLflow run."""


class FilterPredicate(BaseModel):
    """A generic row-level filter predicate for dataset selection."""

    column: str
    """Name of the dataset column to filter on."""
    op: str  # "in", "not_in", "==", "!=", "<", "<=", ">", ">=", "between"
    """Comparison operator: in, not_in, ==, !=, <, <=, >, >=, or between."""
    value: Any
    """Value(s) to compare against. Type depends on the chosen operator."""

    @field_validator("op")
    @classmethod
    def _validate_op(cls, value: str) -> str:
        allowed = {"in", "not_in", "==", "!=", "<", "<=", ">", ">=", "between"}
        if value not in allowed:
            raise ValueError(f"filter op must be one of: {sorted(allowed)}")
        return value


class ColumnMapping(BaseModel):
    """Configurable column name mapping for datasets with non-standard schemas."""

    text: str | None = None
    """Column name containing text input. None uses the dataset default."""
    audio: str | None = None
    """Column name containing audio input. None uses the dataset default."""
    language: str = "language"
    """Column name containing the language label."""
    original_language: str = "original_language"
    """Column name containing the original (pre-translation) language label."""
    token_count: str = "token_count"
    """Column name containing pre-computed token counts."""


class DatasetConfig(BaseModel):
    """Dataset location and loading configuration."""

    model_config = {"populate_by_name": True}

    source: DatasetSource = DatasetSource.HF_PARQUET
    """Backend for loading the dataset (HuggingFace parquet, local CSV, etc.)."""
    subset: str = DEFAULT_SUBSET
    """Dataset subset/configuration name."""
    split: str = DEFAULT_SPLIT
    """Dataset split to load (e.g. train, test, validation)."""
    revision: str = "6c046d6c94a76ddb2bb9e5577fd51e7fb77bb691"
    """Git revision (commit SHA or tag) for the HuggingFace dataset."""
    repo_id: str = "Pawlo77/mllm-shap"
    """HuggingFace repository ID for the dataset."""
    trust_remote_code: bool = True
    """Whether to trust and execute remote code from the dataset repository."""
    path: str | None = None
    """Local filesystem path. Required for LOCAL_PARQUET and LOCAL_CSV sources."""
    column_mapping: ColumnMapping = Field(default_factory=ColumnMapping)
    """Custom column name mappings for non-standard dataset schemas."""

    @model_validator(mode="after")
    def _validate_source(self) -> "DatasetConfig":
        if self.source in (DatasetSource.LOCAL_PARQUET, DatasetSource.LOCAL_CSV):
            if not self.path:
                raise ValueError(f"dataset.path is required when source={self.source}")
        return self


class SelectionConfig(BaseModel):
    """Row selection parameters."""

    max_samples: int | None = None
    """Maximum number of samples to process. None means no limit."""
    shuffle_seed: int | None = 0
    """Random seed for shuffling rows. None disables shuffling."""
    start_index: int = 0
    """Zero-based index to start sampling from (after shuffling)."""
    max_prompt_tokens: int | None = None
    """Upper bound on prompt token count for row inclusion."""
    min_prompt_tokens: int | None = None
    """Lower bound on prompt token count for row inclusion."""
    balanced_token_counts: List[int] | None = None
    """Specific token counts to balance across when sampling."""
    samples_per_token_count: int | None = None
    """Number of samples to draw per token-count bucket."""
    allow_partial_token_count_buckets: bool = False
    """Allow buckets with fewer samples than samples_per_token_count."""
    filters: List[FilterPredicate] = Field(default_factory=list)
    """Row-level filter predicates applied before sampling."""

    @field_validator("max_samples")
    @classmethod
    def _validate_max_samples(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("selection.max_samples must be positive or null.")
        return value

    @field_validator("start_index")
    @classmethod
    def _validate_start_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("selection.start_index must be >= 0.")
        return value


class GenerationConfig(BaseModel):
    """Model text/audio generation knobs; mirrors mllm_shap ModelConfig."""

    max_new_tokens: int = 32
    """Maximum number of new tokens to generate per forward pass."""
    text_temperature: float = 0.2
    """Sampling temperature for text token generation."""
    text_top_k: int | None = None
    """Top-k filtering for text generation. None disables top-k."""
    audio_temperature: float | None = None
    """Sampling temperature for audio token generation. None uses text value."""
    audio_top_k: int | None = None
    """Top-k filtering for audio generation. None uses text value."""


class ChatConfig(BaseModel):
    """Chat construction configuration."""

    system_roles_setup: str = "SYSTEM_ASSISTANT"
    """Chat template role setup identifier (e.g. SYSTEM_ASSISTANT, USER_ONLY)."""
    system_prompt: str = "You are a helpful assistant."
    """System message content prepended to the conversation."""
    assistant_prefill: str | None = None
    """Optional text to prefill the assistant response with."""


class ModalityConfig(BaseModel):
    """Configuration for input/output modalities."""

    input_modality: InputModality = InputModality.TEXT
    """Modality of the input data (TEXT, AUDIO, or MULTI_MODAL)."""
    output_modality: OutputModality = OutputModality.TEXT
    """Modality of the model output (TEXT or AUDIO)."""


class AudioSegmentationConfig(BaseModel):
    """Audio segmentation policy for audio SHAP tokens."""

    method: str = "raw"
    """Segmentation method: 'raw' (fixed-size chunks) or 'sgpa' (phoneme-aligned)."""
    aligner_device: str = "cpu"
    """Device for the forced-alignment model used by SGPA segmentation."""

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        if value not in ("raw", "sgpa"):
            raise ValueError("audio_segmentation.method must be one of: raw | sgpa.")
        return value


class ShapConfig(BaseModel):
    """SHAP-wide knobs shared across explainers."""

    mode: str = "CONTEXTUAL"
    """SHAP computation mode (e.g. CONTEXTUAL, MARGINAL)."""
    normalizer: str = "AbsSumNormalizer"
    """Normalizer class for SHAP value post-processing."""
    reducer: str = "MeanReducer"
    """Reducer class for aggregating per-token SHAP values."""
    similarity: str = "TfIdfCosineSimilarity"
    """Similarity metric class for comparing model outputs."""
    allow_mask_duplicates: bool = False
    """Whether to allow duplicate tokens in SHAP masking coalitions."""
    token_filter: TokenFilterType = TokenFilterType.EXCLUDE_PUNCTUATION
    """Token filtering strategy to exclude certain tokens from attribution."""

    @field_validator("normalizer")
    @classmethod
    def _validate_normalizer(cls, value: str) -> str:
        if value not in NORMALIZER_MAP:
            raise ValueError(f"Unknown shap.normalizer: {value}")
        return value

    @field_validator("reducer")
    @classmethod
    def _validate_reducer(cls, value: str) -> str:
        if value not in REDUCER_MAP:
            raise ValueError(f"Unknown shap.reducer: {value}")
        return value

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: str) -> str:
        """Normalize SHAP mode identifier to uppercase enum name."""
        if not isinstance(value, str):
            raise ValueError("shap.mode must be a string")
        return value.upper()

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        """Validate SHAP mode against package-defined Mode enum members."""
        if value not in ALLOWED_SHAP_MODES:
            raise ValueError(
                f"Unknown shap.mode: {value}. Available: {sorted(ALLOWED_SHAP_MODES)}"
            )
        return value

    @field_validator("similarity")
    @classmethod
    def _validate_similarity(cls, value: str) -> str:
        """Validate similarity class name against runtime-discovered implementations."""
        if value not in SIMILARITY_MAP:
            raise ValueError(
                f"Unknown shap.similarity: {value}. Available: {sorted(SIMILARITY_MAP)}"
            )
        return value


class HierarchicalConfig(BaseModel):
    """Hierarchical explainer sub-configuration."""

    model_config = {"populate_by_name": True}

    ks: List[int] = Field(default_factory=lambda: [10])
    """List of k values (number of top tokens per layer) to evaluate."""
    shap_type: str = "limited_neyman"
    """Explainer type for the second-layer SHAP computation."""
    shap_num_samples: List[int] | None = None
    """Number of samples for the second-layer explainer. Overrides fractions."""
    shap_fractions: List[float] | None = Field(default=None, alias="shap_fraction")
    """Fraction of full sample budget for the second-layer explainer."""
    first_layer_type: str | None = None
    """Explainer type for the first layer. None defaults to shap_type."""
    first_layer_num_samples: List[int] | None = None
    """Number of samples for the first-layer explainer. Overrides fractions."""
    first_layer_fractions: List[float] | None = Field(
        default=None, alias="first_layer_fraction"
    )
    """Fraction of full sample budget for the first-layer explainer."""
    use_importance_sampling: bool = True
    """Enable importance sampling for more efficient coalition allocation."""
    importance_min_fractions: List[float] | None = None
    """Minimum fraction of samples allocated to low-importance tokens."""
    mode: str = "MULTI_MODAL_MULTI_USER"
    """Hierarchical grouping mode (e.g. MULTI_MODAL_MULTI_USER, SINGLE_MODAL)."""

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: str) -> str:
        """Normalize hierarchical mode identifier to uppercase enum name."""
        if not isinstance(value, str):
            raise ValueError("hierarchical.mode must be a string")
        return value.upper()

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        """Validate hierarchical mode against package-defined enum members."""
        if value not in ALLOWED_HIERARCHICAL_MODES:
            raise ValueError(
                "Unknown hierarchical.mode: "
                f"{value}. Available: {sorted(ALLOWED_HIERARCHICAL_MODES)}"
            )
        return value


class ExplainerVariant(BaseModel):
    """One experiment variant with optional per-variant overrides."""

    model_config = {"populate_by_name": True}

    explainer_type: str = "limited_mc"
    """Canonical name of the SHAP explainer algorithm to use."""
    num_samples: List[int] | None = None
    """List of sample counts to sweep over for this variant."""
    linear: List[float] | None = None
    """Linear scaling factors for sample budget allocation."""
    fractions: List[float] | None = None
    """Fractions of the full combinatorial sample budget to evaluate."""
    name: str | None = None
    """Optional human-readable name for this variant in logs and MLflow."""
    hierarchical: HierarchicalConfig | None = None
    """Hierarchical explainer config. Required when explainer_type is hierarchical."""
    shap_override: Dict[str, Any] | None = None
    """Per-variant overrides for ShapConfig fields."""
    generation_override: Dict[str, Any] | None = None
    """Per-variant overrides for GenerationConfig fields."""
    embedding_override: Dict[str, Any] | None = None
    """Per-variant overrides for EmbeddingConfig fields."""

    @field_validator("explainer_type", mode="before")
    @classmethod
    def _normalize_explainer_type(cls, value: str) -> str:
        """Normalize explainer type ids to lower-case canonical strings."""
        if not isinstance(value, str):
            raise ValueError("explainer_type must be a string")
        value = value.lower()
        # Config-level aliases → canonical names
        _ALIASES = {"precise": "exact"}
        return _ALIASES.get(value, value)

    @field_validator("explainer_type")
    @classmethod
    def _validate_supported_explainer_type(cls, value: str) -> str:
        """Ensure explainer_type is discoverable in the installed mllm_shap package."""
        if not is_supported_explainer(value):
            raise ValueError(
                f"Unsupported explainer_type '{value}'. Use one of {sorted(get_available_explainer_types())}."
            )
        return value


class EmbeddingConfig(BaseModel):
    """Optional external embedding model (CustomEmbedding)."""

    model_id: str | None = None
    """HuggingFace model ID for the embedding model. None disables custom embeddings."""
    revision: str | None = None
    """Git revision (commit SHA or tag) for the embedding model."""
    max_length: int = 64
    """Maximum input token length for the embedding model."""
    batch_size: int = 64
    """Batch size for embedding inference."""
    l2_normalize: bool = True
    """Whether to L2-normalize embedding vectors."""
    local_files_only: bool = False
    """Only load model from local cache, do not download."""
    device: str | None = None
    """Device for embedding computation. None uses the experiment-level device."""


class LmStudioConfigModel(BaseModel):
    """LM Studio managed model lifecycle configuration."""

    enabled: bool = False
    """Whether to use LM Studio for model lifecycle management."""
    model_key: str = ""
    """LM Studio model identifier key (required when enabled=True)."""
    context_length: int | None = None
    """Context window size. None uses the model's default."""
    context_length_gap: int = 64
    """Token gap reserved between context length and max generation tokens."""
    max_concurrency: int = 1
    """Maximum number of concurrent inference requests."""
    seed: int | None = None
    """Random seed for deterministic generation. None uses non-deterministic."""
    gpu_offload: float | None = None
    """Fraction of model layers to offload to GPU (0.0-1.0)."""
    cpu_threads: int | None = None
    """Number of CPU threads for inference. None uses system default."""
    flash_attention: bool = True
    """Enable flash attention optimization for faster inference."""
    ttl: int | None = 3600
    """Time-to-live in seconds before unloading idle model. None keeps forever."""
    quantization_preference: str | None = None
    """Preferred quantization format (e.g. q4_0, q8_0). None uses original."""
    api_host: str = "127.0.0.1:1234"
    """Host and port for the LM Studio API server."""
    keep_model_in_memory: bool = True
    """Keep the model loaded in memory between requests."""

    @field_validator("model_key")
    @classmethod
    def _validate_model_key(cls, value: str, info: Any) -> str:
        # Only enforce non-empty when enabled (checked at ExperimentSet level)
        return value


class RuntimeConfig(BaseModel):
    """Runtime behavior knobs."""

    verbose: bool = True
    """Enable verbose logging during experiment execution."""
    progress_bar: bool = True
    """Display a progress bar for sample processing."""
    gc_after_each_sample: bool = True
    """Run garbage collection periodically during processing."""
    gc_interval: int = 1
    """Run gc.collect() every N samples (only when gc_after_each_sample=True).
    Set >1 to reduce GC pauses between samples at cost of higher peak memory."""
    cuda_empty_cache: bool = False
    """Call torch.cuda.empty_cache() after garbage collection."""
    n_generator_jobs: int = 1
    """Number of parallel generation jobs per sample."""


class ExperimentSet(BaseModel):
    """Top-level config for a set of experiment variants."""

    experiment_set_id: str
    """Unique identifier for this experiment set (used in output paths and MLflow)."""
    output_root: str = "experiments_output"
    """Root directory for saving experiment results and artifacts."""
    device: str | None = None
    """Torch device for model inference (e.g. 'cuda', 'cpu'). None auto-detects."""
    connector: str = "liquid_audio"
    """Model connector type (e.g. liquid_audio, transformers_text, mock)."""
    connector_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments passed to the connector constructor."""
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    """Dataset loading and source configuration."""
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    """Row selection, filtering, and sampling configuration."""
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    """MLflow tracking and logging configuration."""
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    """Text/audio generation parameters for the model."""
    chat: ChatConfig = Field(default_factory=ChatConfig)
    """Chat template and system prompt configuration."""
    modality: ModalityConfig = Field(default_factory=ModalityConfig)
    """Input and output modality specification."""
    audio_segmentation: AudioSegmentationConfig = Field(
        default_factory=AudioSegmentationConfig
    )
    """Audio segmentation policy for audio SHAP tokens."""
    shap: ShapConfig = Field(default_factory=ShapConfig)
    """SHAP computation parameters shared across all variants."""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    """External embedding model configuration for similarity computation."""
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    """Runtime behavior knobs (GC, verbosity, parallelism)."""
    lm_studio: LmStudioConfigModel = Field(default_factory=LmStudioConfigModel)
    """LM Studio model lifecycle management configuration."""
    experiments: List[ExplainerVariant] = Field(default_factory=list)
    """List of explainer variants to run in this experiment set."""

    @field_validator("connector", mode="before")
    @classmethod
    def _normalize_connector(cls, value: str) -> str:
        """Normalize connector ids to lower-case canonical strings."""
        if not isinstance(value, str):
            raise ValueError("connector must be a string")
        return value.lower()

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentSet":
        """Load ExperimentSet from JSON file with optional base inheritance."""
        raw = _load_json_with_inheritance(Path(path))
        return cls.model_validate(raw)

    def get_effective_shap(self, variant: ExplainerVariant) -> "ShapConfig":
        """Return ShapConfig with per-variant overrides applied."""
        if not variant.shap_override:
            return self.shap
        merged = self.shap.model_dump()
        merged.update(variant.shap_override)
        return ShapConfig.model_validate(merged)

    def get_effective_generation(self, variant: ExplainerVariant) -> "GenerationConfig":
        """Return GenerationConfig with per-variant overrides applied."""
        if not variant.generation_override:
            return self.generation
        merged = self.generation.model_dump()
        merged.update(variant.generation_override)
        return GenerationConfig.model_validate(merged)

    def get_effective_embedding(self, variant: ExplainerVariant) -> "EmbeddingConfig":
        """Return EmbeddingConfig with per-variant overrides applied."""
        if not variant.embedding_override:
            return self.embedding
        merged = self.embedding.model_dump()
        merged.update(variant.embedding_override)
        return EmbeddingConfig.model_validate(merged)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base dict with override precedence."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_json_with_inheritance(path: Path) -> Dict[str, Any]:
    """Load JSON config with optional base key for config inheritance."""
    with open(path, "r", encoding="utf-8") as file_obj:
        raw = json.load(file_obj)

    base_path = raw.pop("base", None)
    if base_path is not None:
        base_resolved = (path.parent / base_path).resolve()
        if not base_resolved.exists():
            raise FileNotFoundError(
                f"Base config not found: {base_resolved} (referenced from {path})"
            )
        base_raw = _load_json_with_inheritance(base_resolved)
        raw = _deep_merge(base_raw, raw)

    return raw
