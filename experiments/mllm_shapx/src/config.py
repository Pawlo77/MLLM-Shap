"""Configuration models, registries, parsing and validation using Pydantic."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from mllm_shap.shap.base.embeddings import BaseEmbeddingReducer
from mllm_shap.shap.embeddings import (
    FirstReducer,
    MaxReducer,
    MeanReducer,
    MinReducer,
    SumReducer,
    ZeroReducer,
)
from mllm_shap.shap.normalizers import (
    AbsSumNormalizer,
    IdentityNormalizer,
    MinMaxNormalizer,
    PowerShiftNormalizer,
)
from mllm_shap.shap.similarity import (
    CosineSimilarity,
    EuclideanSimilarity,
    TfIdfCosineSimilarity,
)

from .constants import (
    DEFAULT_SPLIT,
    DEFAULT_SUBSET,
    ConnectorType,
    DatasetSource,
    DatasetType,
    ExplainerType,
    HierarchicalModeType,
    InputModality,
    MC_LIKE_EXPLAINERS,
    ModeType,
    OutputModality,
    SimilarityType,
    TokenFilterType,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------
# REGISTRIES
# ---------------------------

NORMALIZER_MAP: Dict[str, Any] = {
    "AbsSumNormalizer": AbsSumNormalizer,
    "IdentityNormalizer": IdentityNormalizer,
    "PowerShiftNormalizer": PowerShiftNormalizer,
    "MinMaxNormalizer": MinMaxNormalizer,
}

REDUCER_MAP: Mapping[str, Callable[[], BaseEmbeddingReducer]] = {
    "MeanReducer": MeanReducer,
    "MaxReducer": MaxReducer,
    "MinReducer": MinReducer,
    "SumReducer": SumReducer,
    "FirstReducer": FirstReducer,
    "ZeroReducer": ZeroReducer,
}

SIMILARITY_MAP: Dict[str, Any] = {
    SimilarityType.COSINE.value: CosineSimilarity,
    SimilarityType.TFIDF_COSINE.value: TfIdfCosineSimilarity,
    SimilarityType.EUCLIDEAN.value: EuclideanSimilarity,
}


# ---------------------------
# CONFIG MODELS (Pydantic)
# ---------------------------


class WandBConfig(BaseModel):
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "mllm-shap"
    entity: Optional[str] = None
    group: Optional[str] = None
    mode: Optional[str] = None  # "online" | "offline" | "disabled"
    tags: List[str] = Field(default_factory=list)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("online", "offline", "disabled"):
            raise ValueError("wandb.mode must be one of: online | offline | disabled.")
        return v


class FilterPredicate(BaseModel):
    """A generic row-level filter predicate for dataset selection."""

    column: str
    op: str  # "in", "not_in", "==", "!=", "<", "<=", ">", ">=", "between"
    value: Any

    @field_validator("op")
    @classmethod
    def _validate_op(cls, v: str) -> str:
        allowed = {"in", "not_in", "==", "!=", "<", "<=", ">", ">=", "between"}
        if v not in allowed:
            raise ValueError(f"filter op must be one of: {sorted(allowed)}")
        return v


class ColumnMapping(BaseModel):
    """Configurable column name mapping for datasets with non-standard schemas."""

    text: Optional[str] = None  # defaults to auto-detect (sentences/prompt)
    audio: Optional[str] = None  # defaults to modality-based resolution
    language: str = "language"
    original_language: str = "original_language"
    token_count: str = "token_count"


class DatasetConfig(BaseModel):
    """Dataset location and loading configuration."""

    model_config = {"populate_by_name": True}

    source: DatasetSource = DatasetSource.HF_PARQUET
    # HuggingFace Hub fields
    subset: str = DEFAULT_SUBSET
    split: str = DEFAULT_SPLIT
    revision: str = "refs/convert/parquet"
    repo_id: str = "Pawlo77/mllm-shap"
    trust_remote_code: bool = True
    # Local file fields
    path: Optional[str] = None  # path for local_parquet or local_csv
    # Column mapping
    column_mapping: ColumnMapping = Field(default_factory=ColumnMapping)

    @model_validator(mode="before")
    @classmethod
    def _migrate_use_parquet(cls, data: Any) -> Any:
        """Backward compat: migrate legacy use_parquet field to source."""
        if isinstance(data, dict) and "use_parquet" in data:
            use_parquet = data.pop("use_parquet")
            if "source" not in data:
                if use_parquet:
                    data["source"] = DatasetSource.HF_PARQUET.value
                else:
                    data["source"] = DatasetSource.HF_DATASETS.value
        return data

    @model_validator(mode="after")
    def _validate_source(self) -> "DatasetConfig":
        if self.source in (DatasetSource.LOCAL_PARQUET, DatasetSource.LOCAL_CSV):
            if not self.path:
                raise ValueError(f"dataset.path is required when source={self.source}")
        return self


class SelectionConfig(BaseModel):
    """Row selection parameters."""

    max_samples: Optional[int] = None
    shuffle_seed: Optional[int] = 0
    start_index: int = 0
    max_prompt_tokens: Optional[int] = None
    min_prompt_tokens: Optional[int] = None
    balanced_token_counts: Optional[List[int]] = None
    samples_per_token_count: Optional[int] = None
    allow_partial_token_count_buckets: bool = False
    filters: List[FilterPredicate] = Field(default_factory=list)

    @field_validator("max_samples")
    @classmethod
    def _validate_max_samples(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("selection.max_samples must be positive or null.")
        return v

    @field_validator("start_index")
    @classmethod
    def _validate_start_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError("selection.start_index must be >= 0.")
        return v


class GenerationConfig(BaseModel):
    """Model text/audio generation knobs — exposes full mllm_shap ModelConfig."""

    max_new_tokens: int = 32
    text_temperature: float = 0.2
    text_top_k: Optional[int] = None
    audio_temperature: Optional[float] = None
    audio_top_k: Optional[int] = None


class ChatConfig(BaseModel):
    """Chat construction configuration."""

    system_roles_setup: str = "SYSTEM_ASSISTANT"
    system_prompt: str = "You are a helpful assistant."
    assistant_prefill: Optional[str] = None


class ModalityConfig(BaseModel):
    """Configuration for input/output modalities."""

    input_modality: InputModality = InputModality.TEXT
    output_modality: OutputModality = OutputModality.TEXT


class AudioSegmentationConfig(BaseModel):
    """Audio segmentation policy for audio SHAP tokens."""

    method: str = "raw"  # raw | sgpa
    aligner_device: str = "cpu"

    @field_validator("method")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        if v not in ("raw", "sgpa"):
            raise ValueError("audio_segmentation.method must be one of: raw | sgpa.")
        return v


class ShapConfig(BaseModel):
    """SHAP-wide knobs shared across explainers."""

    mode: ModeType = ModeType.CONTEXTUAL
    normalizer: str = "AbsSumNormalizer"
    reducer: str = "MeanReducer"
    similarity: SimilarityType = SimilarityType.TFIDF_COSINE
    allow_mask_duplicates: bool = False
    token_filter: TokenFilterType = TokenFilterType.EXCLUDE_PUNCTUATION

    @field_validator("normalizer")
    @classmethod
    def _validate_normalizer(cls, v: str) -> str:
        if v not in NORMALIZER_MAP:
            raise ValueError(f"Unknown shap.normalizer: {v}")
        return v

    @field_validator("reducer")
    @classmethod
    def _validate_reducer(cls, v: str) -> str:
        if v not in REDUCER_MAP:
            raise ValueError(f"Unknown shap.reducer: {v}")
        return v


class HierarchicalConfig(BaseModel):
    """Hierarchical explainer sub-configuration."""

    model_config = {"populate_by_name": True}

    ks: List[int] = Field(default_factory=lambda: [10])
    shap_type: str = "limited_neyman"
    shap_num_samples: Optional[List[int]] = None
    shap_fractions: Optional[List[float]] = Field(default=None, alias="shap_fraction")
    first_layer_type: Optional[str] = None
    first_layer_num_samples: Optional[List[int]] = None
    first_layer_fractions: Optional[List[float]] = Field(
        default=None, alias="first_layer_fraction"
    )
    use_importance_sampling: bool = True
    importance_min_fractions: Optional[List[float]] = None
    mode: HierarchicalModeType = HierarchicalModeType.MULTI_MODAL_MULTI_USER


class ExplainerVariant(BaseModel):
    """One experiment variant with optional per-variant overrides."""

    model_config = {"populate_by_name": True}

    explainer_type: ExplainerType = ExplainerType.LIMITED_MC
    num_samples: Optional[List[int]] = None
    linear: Optional[List[float]] = None
    fractions: Optional[List[float]] = None
    name: Optional[str] = None
    # Hierarchical sub-config
    hierarchical: Optional[HierarchicalConfig] = None
    # Per-variant overrides (merged with global configs at runtime)
    shap_override: Optional[Dict[str, Any]] = None
    generation_override: Optional[Dict[str, Any]] = None
    embedding_override: Optional[Dict[str, Any]] = None

    @field_validator("explainer_type", mode="before")
    @classmethod
    def _normalize_explainer_type(cls, v: str) -> str:
        """Normalize shorthand explainer type names to canonical values."""
        aliases = {
            "mc": ExplainerType.LIMITED_MC.value,
            "cc": ExplainerType.LIMITED_CC.value,
            "neyman": ExplainerType.LIMITED_NEYMAN.value,
            "complementary": ExplainerType.LIMITED_CC.value,
        }
        if isinstance(v, str):
            return aliases.get(v.lower(), v.lower())
        return v


class EmbeddingConfig(BaseModel):
    """Optional external embedding model (CustomEmbedding)."""

    model_id: Optional[str] = None
    revision: Optional[str] = None
    max_length: int = 64
    batch_size: int = 64
    l2_normalize: bool = True
    local_files_only: bool = False
    device: Optional[str] = None  # if None, uses model device


class RuntimeConfig(BaseModel):
    """Runtime behavior knobs."""

    verbose: bool = True
    progress_bar: bool = True
    gc_after_each_sample: bool = True
    cuda_empty_cache: bool = True


class ExperimentSet(BaseModel):
    """Top-level config for a set of experiment variants."""

    experiment_set_id: str
    output_root: str = "experiments_output"
    device: Optional[str] = None
    connector: ConnectorType = ConnectorType.LIQUID_AUDIO
    connector_kwargs: Dict[str, Any] = Field(default_factory=dict)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    modality: ModalityConfig = Field(default_factory=ModalityConfig)
    audio_segmentation: AudioSegmentationConfig = Field(
        default_factory=AudioSegmentationConfig
    )
    shap: ShapConfig = Field(default_factory=ShapConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    experiments: List[ExplainerVariant] = Field(default_factory=list)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentSet":
        """Load ExperimentSet from a JSON file, supporting config inheritance via 'base' key."""
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


# ---------------------------
# CONFIG INHERITANCE
# ---------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base dict (override wins)."""
    result = deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = deepcopy(val)
    return result


def _load_json_with_inheritance(path: Path) -> Dict[str, Any]:
    """Load JSON config with optional 'base' key for config inheritance."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    base_path = raw.pop("base", None)
    if base_path is not None:
        # Resolve relative to the current config's directory
        base_resolved = (path.parent / base_path).resolve()
        if not base_resolved.exists():
            raise FileNotFoundError(
                f"Base config not found: {base_resolved} (referenced from {path})"
            )
        base_raw = _load_json_with_inheritance(base_resolved)
        raw = _deep_merge(base_raw, raw)

    return raw


# ---------------------------
# VALIDATION (beyond Pydantic)
# ---------------------------


def validate_config(cfg: ExperimentSet) -> List[str]:
    """Return a list of human-readable problems (empty = valid)."""
    errs: List[str] = []

    # Dataset warnings (not errors) for unknown subsets
    known_subsets = {dt.value for dt in DatasetType}
    if cfg.dataset.subset not in known_subsets:
        LOGGER.warning(
            "Unknown dataset subset '%s': not in known subsets %s. "
            "Proceeding anyway; this may indicate a typo or custom subset. "
            "Known subsets: %s",
            cfg.dataset.subset,
            sorted(known_subsets),
            ", ".join(sorted(known_subsets)),
        )

    # Selection validation
    if cfg.selection.balanced_token_counts is not None:
        if not cfg.selection.balanced_token_counts:
            errs.append("selection.balanced_token_counts must be non-empty.")
        bad_counts = [
            c
            for c in cfg.selection.balanced_token_counts
            if not isinstance(c, int) or c <= 0
        ]
        if bad_counts:
            errs.append(
                f"selection.balanced_token_counts entries must be positive ints; bad: {bad_counts}"
            )
        if (
            cfg.selection.samples_per_token_count is None
            or cfg.selection.samples_per_token_count <= 0
        ):
            errs.append(
                "selection.samples_per_token_count must be positive when balanced_token_counts is set."
            )

    # Modality / connector compat
    if cfg.connector == ConnectorType.TRANSFORMERS_TEXT:
        if cfg.modality.output_modality == OutputModality.AUDIO:
            errs.append(
                "TransformersCausalText connector does not support audio output."
            )
        if cfg.modality.input_modality != InputModality.TEXT:
            errs.append("TransformersCausalText connector only supports text input.")

    # Audio segmentation compat
    if (
        cfg.audio_segmentation.method == "sgpa"
        and cfg.modality.input_modality == InputModality.TEXT
    ):
        errs.append("audio_segmentation.method='sgpa' requires audio input.")

    # Experiments
    if not cfg.experiments:
        errs.append("experiments must contain at least one variant.")
        return errs

    for i, exp in enumerate(cfg.experiments):
        t = exp.explainer_type

        # MC-like knobs
        if t in MC_LIKE_EXPLAINERS:
            if not exp.num_samples and not exp.fractions and not exp.linear:
                errs.append(
                    f"experiments[{i}]: MC-like explainer requires num_samples, fractions, or linear."
                )
            if exp.num_samples is not None:
                bad_ns = [
                    ns
                    for ns in exp.num_samples
                    if not isinstance(ns, int) or (ns != -1 and ns <= 0)
                ]
                if bad_ns:
                    errs.append(
                        f"experiments[{i}].num_samples entries must be -1 or positive ints; bad: {bad_ns}"
                    )
            if exp.fractions is not None:
                bad = [f for f in exp.fractions if not 0.0 < float(f) <= 1.0]
                if bad:
                    errs.append(
                        f"experiments[{i}].fractions must be in (0,1]; bad: {bad}"
                    )
            if exp.linear is not None:
                bad = [lin for lin in exp.linear if not 0.0 < float(lin) <= 10.0]
                if bad:
                    errs.append(
                        f"experiments[{i}].linear must be in (0,10]; bad: {bad}"
                    )

        # Hierarchical
        if t == ExplainerType.HIERARCHICAL:
            h = exp.hierarchical
            if h is None:
                errs.append(
                    f"experiments[{i}]: hierarchical explainer requires 'hierarchical' config block."
                )
                continue

            if not h.ks:
                errs.append(f"experiments[{i}]: hierarchical.ks must be non-empty.")
            else:
                badk = [k for k in h.ks if not isinstance(k, int) or k < 2]
                if badk:
                    errs.append(
                        f"experiments[{i}]: hierarchical.ks values must be >= 2; bad: {badk}"
                    )

            allowed_inner = {
                "precise",
                "limited_mc",
                "limited_cc",
                "standard_cc",
                "limited_neyman",
                "standard_neyman",
                # Shorthands (backward compat)
                "neyman",
                "mc",
                "cc",
            }
            if h.shap_type.lower() not in allowed_inner:
                errs.append(
                    f"experiments[{i}]: hierarchical.shap_type must be one of {sorted(allowed_inner)}."
                )

            if not h.use_importance_sampling:
                errs.append(
                    f"experiments[{i}]: hierarchical.use_importance_sampling must be True."
                )

            if h.importance_min_fractions:
                bad_imp = [
                    f for f in h.importance_min_fractions if not 0.0 < float(f) <= 1.0
                ]
                if bad_imp:
                    errs.append(
                        f"experiments[{i}]: hierarchical.importance_min_fractions must be in (0,1]; bad: {bad_imp}"
                    )

            if h.first_layer_type is not None:
                allowed_first = {
                    "precise",
                    "limited_mc",
                    "limited_cc",
                    "standard_cc",
                    "limited_neyman",
                    "standard_neyman",
                    # Shorthands (backward compat)
                    "neyman",
                    "mc",
                    "cc",
                }
                if h.first_layer_type.lower() not in allowed_first:
                    errs.append(
                        f"experiments[{i}]: hierarchical.first_layer_type must be one of {sorted(allowed_first)}."
                    )

            # Warning for normalizer
            if cfg.shap.normalizer != "MinMaxNormalizer":
                LOGGER.warning(
                    "HierarchicalExplainer typically uses MinMaxNormalizer for better hierarchical attribution; "
                    "you configured %s instead. Verify this is intentional.",
                )

    return errs
