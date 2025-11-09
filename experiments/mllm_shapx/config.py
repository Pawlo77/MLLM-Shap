"""Configuration models, registries, parsing and validation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    PowerShiftNormalizer,
)

from .constants import (
    DEFAULT_SIMILARITY,
    DEFAULT_SPLIT,
    DEFAULT_SUBSET,
    ExplainerType,
    ModelKind
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


@dataclass
class GenerationConfig:
    """Model text generation knobs."""
    max_new_tokens: int = 32
    text_temperature: float = 0.2


@dataclass
class ShapConfig:
    """SHAP-wide knobs that are shared across explainers."""
    mode: str = "CONTEXTUAL"      # maps to mllm_shap.shap.enums.Mode
    normalizer: str = "AbsSumNormalizer"
    reducer: str = "MeanReducer"
    similarity: str = DEFAULT_SIMILARITY


@dataclass
class ExplainerVariant:
    """
    One experiment variant.

    - explainer_type: 'exact' or 'mc'
    - For MC you can provide:
        * num_samples: list[int] (each entry yields a run)
        * fractions:   list[float] in (0, 1] (each entry yields a run)
    """
    explainer_type: str = ExplainerType.MC.value
    num_samples: Optional[List[int]] = None
    fractions: Optional[List[float]] = None
    name: Optional[str] = None


@dataclass
class ExperimentSet:
    """Top-level config for a set of experiment variants."""
    # pylint: disable=too-many-instance-attributes
    experiment_set_id: str
    output_root: str = "experiments_output"
    device: Optional[str] = None  # "cuda"|"cpu"|None (auto)
    model_kind: str = ModelKind.LIQUID_AUDIO.value
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)
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
    "PowerShiftNormalizer": PowerShiftNormalizer,  # has argument 'power'
}

REDUCER_MAP = {
    "MeanReducer": MeanReducer,
    "MaxReducer": MaxReducer,
    "MinReducer": MinReducer,
    "SumReducer": SumReducer,
    "FirstReducer": FirstReducer,
    "ZeroReducer": ZeroReducer,
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
    shp = _subdict(raw, "shap")

    experiments_raw = raw.get("experiments", []) or []
    exps: List[ExplainerVariant] = []
    for e in experiments_raw:
        exps.append(
            ExplainerVariant(
                explainer_type=e.get("explainer_type", ExplainerType.MC.value),
                num_samples=e.get("num_samples"),
                fractions=e.get("fractions"),
                name=e.get("name"),
            )
        )

    return ExperimentSet(
        experiment_set_id=raw["experiment_set_id"],
        output_root=raw.get("output_root", "experiments_output"),
        device=raw.get("device"),
        model_kind=raw.get("model_kind", ModelKind.LIQUID_AUDIO.value),
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
        shap=ShapConfig(
            mode=shp.get("mode", "CONTEXTUAL"),
            normalizer=shp.get("normalizer", "AbsSumNormalizer"),
            reducer=shp.get("reducer", "MeanReducer"),
            similarity=shp.get("similarity", DEFAULT_SIMILARITY),
        ),
        experiments=exps,
    )


def validate_config(cfg: ExperimentSet) -> List[str]:
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
        if cfg.selection.max_prompt_tokens is not None and cfg.selection.max_prompt_tokens <= 0:
            errs.append("selection.max_prompt_tokens must be positive if provided.")

    def _validate_wandb() -> None:
        if cfg.wandb.mode is not None and cfg.wandb.mode not in ("online", "offline", "disabled"):
            errs.append("wandb.mode must be one of: online | offline | disabled.")

    def _validate_shap() -> None:
        if cfg.shap.mode not in ("CONTEXTUAL",):
            errs.append("shap.mode must be 'CONTEXTUAL'.")
        if cfg.shap.normalizer not in NORMALIZER_MAP:
            errs.append(f"Unknown shap.normalizer: {cfg.shap.normalizer}")
        if cfg.shap.reducer not in REDUCER_MAP:
            errs.append(f"Unknown shap.reducer: {cfg.shap.reducer}")
        if cfg.shap.similarity != DEFAULT_SIMILARITY:
            errs.append(f"Only {DEFAULT_SIMILARITY} is supported currently.")

    def _validate_variants() -> None:
        if not cfg.experiments:
            errs.append("experiments must contain at least one variant.")
            return
        if cfg.model_kind not in (mk.value for mk in ModelKind):
            kinds = ", ".join(mk.value for mk in ModelKind)
            errs.append(f"model_kind must be one of: {kinds}.")
        for i, exp in enumerate(cfg.experiments):
            t = exp.explainer_type.lower()
            if t not in (ExplainerType.EXACT.value, ExplainerType.MC.value):
                errs.append(f"experiments[{i}].explainer_type must be 'exact' or 'mc'.")
                continue
            if t == ExplainerType.MC.value:
                if not exp.num_samples and not exp.fractions:
                    errs.append(f"experiments[{i}]: MC requires num_samples or fractions.")
                if exp.num_samples is not None:
                    if not isinstance(exp.num_samples, list) or not exp.num_samples:
                        errs.append(f"experiments[{i}].num_samples must be a non-empty list of ints.")
                    else:
                        bad_ns = [
                            ns for ns in exp.num_samples
                            if not isinstance(ns, int) or (ns != -1 and ns <= 0)
                        ]
                        if bad_ns:
                            errs.append(
                                f"experiments[{i}].num_samples entries must be -1 or positive ints; bad: {bad_ns}"
                            )
                if exp.fractions:
                    bad = [f for f in exp.fractions if not 0.0 < float(f) <= 1.0]
                    if bad:
                        errs.append(f"experiments[{i}].fractions must be in (0,1]; bad: {bad}")

    _validate_dataset()
    _validate_selection()
    _validate_wandb()
    _validate_shap()
    _validate_variants()
    return errs
