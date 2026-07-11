"""High-level validation for experiment configuration objects."""

import logging
from typing import TYPE_CHECKING, List

from ..constants import InputModality, OutputModality
from ..discovery import (
    get_available_explainer_types,
    is_mc_like_explainer,
    is_supported_connector,
)
from .registry import ALLOWED_HIERARCHICAL_INNER_TYPES

if TYPE_CHECKING:
    from .models import ExperimentSet

LOGGER = logging.getLogger(__name__)

# Known dataset subset names used for warning-level typo detection.
KNOWN_DATASET_SUBSETS = frozenset({
    "single_sentence",
    "single_sentence_1k",
    "single_sentence_500",
    "multi_lingual",
    "multi_sentence",
})


def validate_config(cfg: "ExperimentSet") -> List[str]:
    """Return a list of human-readable problems (empty list means valid)."""
    errs: List[str] = []

    if not is_supported_connector(cfg.connector):
        errs.append(f"Unsupported connector discovered at runtime: {cfg.connector}")

    available_explainers = get_available_explainer_types()

    if cfg.dataset.subset not in KNOWN_DATASET_SUBSETS:
        LOGGER.warning(
            "Unknown dataset subset '%s': not in known subsets %s. "
            "Proceeding anyway; this may indicate a typo or custom subset. "
            "Known subsets: %s",
            cfg.dataset.subset,
            sorted(KNOWN_DATASET_SUBSETS),
            ", ".join(sorted(KNOWN_DATASET_SUBSETS)),
        )

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
                "selection.balanced_token_counts entries must be positive ints; "
                f"bad: {bad_counts}"
            )
        if (
            cfg.selection.samples_per_token_count is None
            or cfg.selection.samples_per_token_count <= 0
        ):
            errs.append(
                "selection.samples_per_token_count must be positive when "
                "balanced_token_counts is set."
            )

    if cfg.connector == "hf_text":
        if cfg.modality.output_modality == OutputModality.AUDIO:
            errs.append(
                "TransformersCausalText connector does not support audio output."
            )
        if cfg.modality.input_modality != InputModality.TEXT:
            errs.append("TransformersCausalText connector only supports text input.")

    if cfg.connector == "openai_compat_text":
        if cfg.modality.output_modality == OutputModality.AUDIO:
            errs.append(
                "OpenAICompatCausalText connector does not support audio output."
            )
        if cfg.modality.input_modality != InputModality.TEXT:
            errs.append("OpenAICompatCausalText connector only supports text input.")

    if cfg.connector == "lm_studio_text":
        if cfg.modality.output_modality == OutputModality.AUDIO:
            errs.append("lm_studio_text connector does not support audio output.")
        if cfg.modality.input_modality != InputModality.TEXT:
            errs.append("lm_studio_text connector only supports text input.")

    if cfg.lm_studio.enabled:
        if not cfg.lm_studio.model_key:
            errs.append("lm_studio.model_key is required when lm_studio.enabled=true.")
        if cfg.connector not in ("openai_compat_text", "lm_studio_text"):
            errs.append(
                f"lm_studio.enabled requires connector 'openai_compat_text' or "
                f"'lm_studio_text', got '{cfg.connector}'."
            )

    if (
        cfg.audio_segmentation.method == "sgpa"
        and cfg.modality.input_modality == InputModality.TEXT
    ):
        errs.append("audio_segmentation.method='sgpa' requires audio input.")

    if not cfg.experiments:
        errs.append("experiments must contain at least one variant.")
        return errs

    for i, exp in enumerate(cfg.experiments):
        t_name = exp.explainer_type

        if t_name not in available_explainers:
            errs.append(
                f"experiments[{i}]: unsupported explainer_type '{t_name}' "
                "(not discovered in mllm_shap)."
            )
            continue

        if is_mc_like_explainer(t_name):
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

        if t_name == "hierarchical":
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

            if h.shap_type.lower() not in ALLOWED_HIERARCHICAL_INNER_TYPES:
                errs.append(
                    f"experiments[{i}]: hierarchical.shap_type must be one of "
                    f"{sorted(ALLOWED_HIERARCHICAL_INNER_TYPES)}."
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
                if h.first_layer_type.lower() not in ALLOWED_HIERARCHICAL_INNER_TYPES:
                    errs.append(
                        f"experiments[{i}]: hierarchical.first_layer_type must be one of "
                        f"{sorted(ALLOWED_HIERARCHICAL_INNER_TYPES)}."
                    )

            if cfg.shap.normalizer != "MinMaxNormalizer":
                LOGGER.warning(
                    "HierarchicalExplainer typically uses MinMaxNormalizer for better hierarchical attribution; "
                    "you configured %s instead. Verify this is intentional.",
                    cfg.shap.normalizer,
                )

    return errs
