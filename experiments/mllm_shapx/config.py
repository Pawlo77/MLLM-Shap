"""Configuration models, registries, parsing and validation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

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
    ExplainerType,
    SimilarityType,
    ModeType,
    InputModality,
    OutputModality,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------
# CONFIG MODEL (dataclasses)
# ---------------------------


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "mllm-shap"
    entity: Optional[str] = None
    group: Optional[str] = None
    mode: Optional[str] = None  # "online" | "offline" | "disabled"
    tags: List[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Dataset location on Hugging Face Hub."""

    subset: str = "single_sentence"
    split: str = "test"
    revision: str = "refs/convert/parquet"
    repo_id: str = "Pawlo77/mllm-shap"


@dataclass
class SelectionConfig:
    """Row selection parameters."""

    max_samples: Optional[int] = None
    shuffle_seed: Optional[int] = 0
    start_index: int = 0
    max_prompt_tokens: Optional[int] = None
    min_prompt_tokens: Optional[int] = None


@dataclass
class GenerationConfig:
    """Model text generation knobs."""

    max_new_tokens: int = 32
    text_temperature: float = 0.2


@dataclass
class ModalityConfig:
    """Configuration for input/output modalities."""

    input_modality: str = InputModality.TEXT.value  # text, audio__male, audio__female
    output_modality: str = OutputModality.TEXT.value  # text, audio

    def get_input_modality(self) -> InputModality:
        """Get InputModality enum from string value."""
        return InputModality(self.input_modality)

    def get_output_modality(self) -> OutputModality:
        """Get OutputModality enum from string value."""
        return OutputModality(self.output_modality)


@dataclass
class ShapConfig:
    """SHAP-wide knobs that are shared across explainers."""

    mode: str = ModeType.CONTEXTUAL.value
    normalizer: str = "AbsSumNormalizer"
    reducer: str = "MeanReducer"
    similarity: str = SimilarityType.TFIDF_COSINE.value


@dataclass
class ExplainerVariant:  # pylint: disable=too-many-instance-attributes
    """
    One experiment variant.

    - explainer_type: 'exact' | 'limited_mc' | 'standard_mc' | 'complementary' | 'neyman' | 'hierarchical'
    - MC-like explainers ('limited_mc', 'standard_mc', 'complementary'):
        * num_samples: list[int] (each entry yields a run)
        * fractions:   list[float] in (0, 1] (each entry yields a run)
    - 'neyman': ignores num_samples/fractions (auto mode).
    """

    explainer_type: str = ExplainerType.LIMITED_MC.value
    num_samples: Optional[List[int]] = None
    linear: Optional[List[float]] = None
    fractions: Optional[List[float]] = None
    name: Optional[str] = None
    # hierarchical
    hierarchical_ks: Optional[List[int]] = None  # list of k to sweep
    # which inner SHAP to use at deeper levels
    hierarchical_shap_type: Optional[str] = None
    hierarchical_shap_num_samples: Optional[List[int]] = None
    hierarchical_shap_fractions: Optional[List[float]] = None
    # optional first-layer explainer
    hierarchical_first_layer_type: Optional[str] = None
    hierarchical_first_layer_num_samples: Optional[List[int]] = None
    hierarchical_first_layer_fractions: Optional[List[float]] = None
    # hierarchical sampling knobs
    hierarchical_use_importance_sampling: bool = True
    hierarchical_importance_min_fractions: Optional[List[float]] = None


@dataclass
class EmbeddingConfig:
    """Optional external embedding model (CustomEmbedding)."""

    model_id: Optional[str] = None
    revision: Optional[str] = None
    max_length: int = 64
    batch_size: int = 64
    l2_normalize: bool = True
    local_files_only: bool = False


@dataclass
class ExperimentSet:
    """Top-level config for a set of experiment variants."""

    # pylint: disable=too-many-instance-attributes
    experiment_set_id: str
    output_root: str = "experiments_output"
    device: Optional[str] = None  # "cuda"|"cpu"|None (auto)
    connector: str = ConnectorType.LIQUID_AUDIO.value
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    experiments: List[ExplainerVariant] = field(default_factory=list)

    @staticmethod
    def from_json(path: Union[str, Path]) -> "ExperimentSet":
        """Load ExperimentSet from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return parse_experiment_set(raw)


# ---------------------------
# REGISTRIES
# ---------------------------


NORMALIZER_MAP = {
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

SIMILARITY_MAP = {
    SimilarityType.COSINE.value: CosineSimilarity,
    SimilarityType.TFIDF_COSINE.value: TfIdfCosineSimilarity,
    SimilarityType.EUCLIDEAN.value: EuclideanSimilarity,
}

# ---------------------------
# PARSING & VALIDATION
# ---------------------------


def _subdict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key, {})
    return v if isinstance(v, dict) else {}


def parse_experiment_set(raw: Dict[str, Any]) -> ExperimentSet:
    """Tolerant dict→dataclasses parser with sane defaults."""
    ds = _subdict(raw, "dataset")
    sel = _subdict(raw, "selection")
    wb = _subdict(raw, "wandb")
    gen = _subdict(raw, "generation")
    mod = _subdict(raw, "modality")
    shp = _subdict(raw, "shap")
    emb = _subdict(raw, "embedding")

    experiments_raw = raw.get("experiments", []) or []
    exps: List[ExplainerVariant] = []
    for e in experiments_raw:
        h = e.get("hierarchical", {}) or {}
        exps.append(
            ExplainerVariant(
                explainer_type=e.get("explainer_type", ExplainerType.LIMITED_MC.value),
                num_samples=e.get("num_samples"),
                fractions=e.get("fractions"),
                linear=e.get("linear"),
                name=e.get("name"),
                hierarchical_ks=e.get("hierarchical_ks", h.get("ks")),
                hierarchical_shap_type=e.get(
                    "hierarchical_shap_type", h.get("shap_type")
                ),
                hierarchical_shap_num_samples=e.get(
                    "hierarchical_shap_num_samples", h.get("shap_num_samples")
                ),
                hierarchical_shap_fractions=e.get(
                    "hierarchical_shap_fractions", h.get("shap_fractions")
                ),
                hierarchical_first_layer_type=e.get(
                    "hierarchical_first_layer_type", h.get("first_layer_type")
                ),
                hierarchical_first_layer_num_samples=e.get(
                    "hierarchical_first_layer_num_samples",
                    h.get("first_layer_num_samples"),
                ),
                hierarchical_first_layer_fractions=e.get(
                    "hierarchical_first_layer_fractions", h.get("first_layer_fractions")
                ),
                hierarchical_use_importance_sampling=bool(
                    e.get(
                        "hierarchical_use_importance_sampling",
                        h.get("use_importance_sampling", True),
                    )
                ),
                hierarchical_importance_min_fractions=e.get(
                    "hierarchical_importance_min_fractions",
                    h.get("importance_min_fractions"),
                ),
            )
        )

    return ExperimentSet(
        experiment_set_id=raw["experiment_set_id"],
        output_root=raw.get("output_root", "experiments_output"),
        device=raw.get("device"),
        connector=raw.get("connector", ConnectorType.LIQUID_AUDIO.value),
        dataset=DatasetConfig(
            subset=ds.get("subset", DEFAULT_SUBSET),
            split=ds.get("split", DEFAULT_SPLIT),
            revision=ds.get("revision", "e1a6f11d58749529f10cc520dfaeb2138fcfc0bf"),
            repo_id=ds.get("repo_id", "Pawlo77/mllm-shap"),
        ),
        selection=SelectionConfig(
            max_samples=sel.get("max_samples"),
            shuffle_seed=sel.get("shuffle_seed", 0),
            start_index=sel.get("start_index", 0),
            max_prompt_tokens=sel.get("max_prompt_tokens"),
            min_prompt_tokens=sel.get("min_prompt_tokens"),
        ),
        wandb=WandBConfig(
            enabled=wb.get("enabled", True),
            project=wb.get("project", "mllm-shap"),
            entity=wb.get("entity"),
            group=wb.get("group"),
            mode=wb.get("mode"),
            tags=wb.get("tags", []),
        ),
        generation=GenerationConfig(
            max_new_tokens=gen.get("max_new_tokens", 32),
            text_temperature=gen.get("text_temperature", 0.2),
        ),
        modality=ModalityConfig(
            input_modality=mod.get("input_modality", InputModality.TEXT.value),
            output_modality=mod.get("output_modality", OutputModality.TEXT.value),
        ),
        shap=ShapConfig(
            mode=shp.get("mode", ModeType.CONTEXTUAL.value),
            normalizer=shp.get("normalizer", "AbsSumNormalizer"),
            reducer=shp.get("reducer", "MeanReducer"),
            similarity=shp.get("similarity", SimilarityType.TFIDF_COSINE.value),
        ),
        embedding=EmbeddingConfig(
            model_id=emb.get("model_id"),
            revision=emb.get("revision"),
            max_length=int(emb.get("max_length", 64)),
            batch_size=int(emb.get("batch_size", 64)),
            l2_normalize=bool(emb.get("l2_normalize", True)),
            local_files_only=bool(emb.get("local_files_only", False)),
        ),
        experiments=exps,
    )


def validate_config(cfg: ExperimentSet) -> List[str]:  # pylint: disable=too-many-statements
    """Return a list of human-readable problems (empty = valid)."""
    errs: List[str] = []

    def _validate_dataset() -> None:
        if cfg.dataset.subset != DEFAULT_SUBSET:
            errs.append(f"Only dataset.subset='{DEFAULT_SUBSET}' is supported.")
        if cfg.dataset.split != DEFAULT_SPLIT:
            errs.append(f"Only dataset.split='{DEFAULT_SPLIT}' is supported.")

    def _validate_selection() -> None:
        if cfg.selection.max_samples is not None and cfg.selection.max_samples <= 0:
            errs.append("selection.max_samples must be positive or null.")
        if cfg.selection.start_index < 0:
            errs.append("selection.start_index must be >= 0.")
        if (
            cfg.selection.max_prompt_tokens is not None
            and cfg.selection.max_prompt_tokens <= 0
        ):
            errs.append("selection.max_prompt_tokens must be positive if provided.")
        if (
            cfg.selection.min_prompt_tokens is not None
            and cfg.selection.min_prompt_tokens <= 0
        ):
            errs.append("selection.min_prompt_tokens must be positive if provided.")

    def _validate_wandb() -> None:
        if cfg.wandb.mode is not None and cfg.wandb.mode not in (
            "online",
            "offline",
            "disabled",
        ):
            errs.append("wandb.mode must be one of: online | offline | disabled.")

    def _validate_modality() -> None:
        valid_input = {m.value for m in InputModality}
        valid_output = {m.value for m in OutputModality}

        if cfg.modality.input_modality not in valid_input:
            errs.append(f"modality.input_modality must be one of: {sorted(valid_input)}.")
        if cfg.modality.output_modality not in valid_output:
            errs.append(f"modality.output_modality must be one of: {sorted(valid_output)}.")

        # TransformersCausalText connector only supports text output
        if cfg.connector == ConnectorType.TRANSFORMERS_TEXT.value:
            if cfg.modality.output_modality == OutputModality.AUDIO.value:
                errs.append("TransformersCausalText connector does not support audio output.")
            if cfg.modality.input_modality != InputModality.TEXT.value:
                errs.append("TransformersCausalText connector only supports text input.")

    def _validate_shap() -> None:
        if cfg.shap.mode not in (ModeType.CONTEXTUAL.value, ModeType.STATIC.value):
            errs.append("shap.mode must be 'CONTEXTUAL'.")
        if cfg.shap.normalizer not in NORMALIZER_MAP:
            errs.append(f"Unknown shap.normalizer: {cfg.shap.normalizer}")
        if cfg.shap.reducer not in REDUCER_MAP:
            errs.append(f"Unknown shap.reducer: {cfg.shap.reducer}")
        if cfg.shap.similarity not in (
            SimilarityType.COSINE.value,
            SimilarityType.TFIDF_COSINE.value,
            SimilarityType.EUCLIDEAN.value,
        ):
            errs.append(
                "shap.similarity must be 'CosineSimilarity' or 'TfIdfCosineSimilarity'."
            )

    def _validate_variants() -> None:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        if not cfg.experiments:
            errs.append("experiments must contain at least one variant.")
            return
        if cfg.connector not in (mk.value for mk in ConnectorType):
            kinds = ", ".join(mk.value for mk in ConnectorType)
            errs.append(f"connector must be one of: {kinds}.")
        for i, exp in enumerate(cfg.experiments):
            t = exp.explainer_type.lower()
            allowed = {e.value for e in ExplainerType}
            if t not in allowed:
                errs.append(
                    f"experiments[{i}].explainer_type must be {sorted(allowed)}."
                )
                continue

            # MC-like knobs required for: limited_mc, standard_mc, complementary
            wants_mc_knobs = t in (
                ExplainerType.LIMITED_MC.value,
                ExplainerType.STANDARD_MC.value,
                ExplainerType.LIMITED_CC.value,
                ExplainerType.STANDARD_CC.value,
                ExplainerType.LIMITED_NEYMAN.value,
                ExplainerType.STANDARD_NEYMAN.value,
            )
            if wants_mc_knobs:
                if not exp.num_samples and not exp.fractions and not exp.linear:
                    errs.append(
                        f"experiments[{i}]: MC-like explainer requires num_samples or fractions."
                    )
                if exp.num_samples is not None:
                    if not isinstance(exp.num_samples, list) or not exp.num_samples:
                        errs.append(
                            f"experiments[{i}].num_samples must be a non-empty list of ints."
                        )
                    else:
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
            if t == ExplainerType.HIERARCHICAL.value:
                minimal_k: int = 2
                # k
                ks = exp.hierarchical_ks or []
                if not ks:
                    errs.append(
                        f"experiments[{i}] hierarchical requires 'hierarchical_k' or 'hierarchical_ks'."
                    )
                else:
                    badk = [k for k in ks if not isinstance(k, int) or k < minimal_k]
                    if badk:
                        errs.append(
                            f"experiments[{i}] hierarchical k values must be integers >= 2; bad: {badk}"
                        )

                # inner shap type
                inner = (exp.hierarchical_shap_type or "").lower()
                allowed_inner = {
                    "precise",
                    "limited_mc",
                    "limited_cc",
                    "standard_cc",
                    "limited_neyman",
                    "standard_neyman",
                }
                if inner and inner not in allowed_inner:
                    errs.append(
                        f"experiments[{i}] hierarchical_shap_type must be one of {sorted(allowed_inner)}."
                    )

                # importance sampling (fixed true as requested)
                if not exp.hierarchical_use_importance_sampling:
                    errs.append(
                        f"experiments[{i}] hierarchical_use_importance_sampling must be True."
                    )

                # min fraction sweep
                if exp.hierarchical_importance_min_fractions is not None:
                    bad_imp = [
                        f
                        for f in exp.hierarchical_importance_min_fractions
                        if not 0.0 < float(f) <= 1.0
                    ]
                    if bad_imp:
                        errs.append(
                            f"experiments[{i}] hierarchical_importance_min_fractions must be in (0,1]; bad: {bad_imp}"
                        )

                # first-layer type
                flt = (exp.hierarchical_first_layer_type or "none").lower()
                allowed_first = {
                    "none",
                    "precise",
                    "limited_mc",
                    "limited_cc",
                    "standard_cc",
                    "limited_neyman",
                    "standard_neyman",
                }
                if flt not in allowed_first:
                    errs.append(
                        f"experiments[{i}] hierarchical_first_layer_type must be one of "
                        f"{sorted(allowed_first)}."
                    )

                # Warning: minmax normalizer
                min_max_norm = "MinMaxNormalizer"
                if cfg.shap.normalizer != min_max_norm:
                    LOGGER.warning(
                        "HierarchicalExplainer should be used with MinMaxNormalizer; you set %s.",
                        cfg.shap.normalizer,
                    )

    _validate_dataset()
    _validate_selection()
    _validate_wandb()
    _validate_modality()
    _validate_shap()
    _validate_variants()
    return errs
