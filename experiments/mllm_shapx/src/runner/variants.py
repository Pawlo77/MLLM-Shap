"""Variant expansion and device selection for experiment runs."""

import itertools
import logging
from typing import List

import torch

from ..config import ExperimentSet, ExplainerVariant, HierarchicalConfig
from ..discovery import get_mc_like_explainer_types
from .types import ExpandedVariant

LOGGER = logging.getLogger(__name__)


def pick_device(device_override: str | None = None) -> torch.device:
    """Resolve runtime device from override string or hardware availability."""
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _expand_exact(v: ExplainerVariant) -> List[ExpandedVariant]:
    """Expand exact explainers into a single concrete run variant."""
    run_slug = v.name or "exact"
    return [
        ExpandedVariant(
            run_slug=run_slug,
            variant=v,
            fraction=None,
            num_samples=None,
            linear=None,
        )
    ]


def _expand_mc_like(v: ExplainerVariant) -> List[ExpandedVariant]:
    """Expand Monte-Carlo-like explainers across sample, fraction, and linear knobs."""
    out: List[ExpandedVariant] = []
    base = v.name or str(v.explainer_type)

    if v.num_samples:
        for n in v.num_samples:
            out.append(
                ExpandedVariant(
                    run_slug=f"{base}_{v.explainer_type}_ns{int(n)}"
                    if v.name
                    else f"{v.explainer_type}_ns{int(n)}",
                    variant=v,
                    fraction=None,
                    num_samples=int(n),
                    linear=None,
                )
            )

    if v.fractions:
        for fr in v.fractions:
            out.append(
                ExpandedVariant(
                    run_slug=(
                        f"{base}_{v.explainer_type}_frac{str(float(fr)).replace('.', '_')}"
                        if v.name
                        else f"{v.explainer_type}_frac{str(float(fr)).replace('.', '_')}"
                    ),
                    variant=v,
                    fraction=float(fr),
                    num_samples=None,
                    linear=None,
                )
            )

    if v.linear:
        for fac in v.linear:
            out.append(
                ExpandedVariant(
                    run_slug=(
                        f"{base}_{v.explainer_type}_lin{str(float(fac)).replace('.', '_')}"
                        if v.name
                        else f"{v.explainer_type}_lin{str(float(fac)).replace('.', '_')}"
                    ),
                    variant=v,
                    fraction=None,
                    num_samples=None,
                    linear=float(fac),
                )
            )

    return out


def _expand_hierarchical(v: ExplainerVariant) -> List[ExpandedVariant]:
    """Expand hierarchical explainers over all configured hierarchical dimensions."""
    h = v.hierarchical or HierarchicalConfig()

    ks = h.ks or [10]
    inner_type = h.shap_type.lower()
    inner_ns_list = h.shap_num_samples or [None]
    inner_frac_list = h.shap_fractions or [None]
    first_layer_type = h.first_layer_type
    first_layer_ns_list = h.first_layer_num_samples or [None]
    first_layer_frac_list = h.first_layer_fractions or [None]
    importance_min_fractions = h.importance_min_fractions or [0.1]
    hier_mode = str(h.mode)

    out: List[ExpandedVariant] = []
    for k, imp_frac, inner_ns, inner_frac in itertools.product(
        ks,
        importance_min_fractions,
        inner_ns_list,
        inner_frac_list,
    ):
        if first_layer_type is None:
            slug = f"hier_{inner_type}_k{k}_imp{str(imp_frac).replace('.', '_')}"
            if inner_ns is not None:
                slug += f"_ns{inner_ns}"
            if inner_frac is not None:
                slug += f"_frac{str(inner_frac).replace('.', '_')}"
            out.append(
                ExpandedVariant(
                    run_slug=f"{v.name}_{slug}" if v.name else slug,
                    variant=v,
                    fraction=None,
                    num_samples=None,
                    linear=None,
                    hier_k=int(k),
                    hier_shap_type=inner_type,
                    hier_shap_num_samples=int(inner_ns)
                    if inner_ns is not None
                    else None,
                    hier_shap_fraction=float(inner_frac)
                    if inner_frac is not None
                    else None,
                    hier_first_layer_type=None,
                    hier_importance_min_fraction=float(imp_frac),
                    hier_mode=hier_mode,
                )
            )
        else:
            for first_ns, first_frac in itertools.product(
                first_layer_ns_list,
                first_layer_frac_list,
            ):
                slug = (
                    f"hier_{inner_type}_k{k}_imp{str(imp_frac).replace('.', '_')}_"
                    f"fl{first_layer_type}"
                )
                if inner_ns is not None:
                    slug += f"_ns{inner_ns}"
                if inner_frac is not None:
                    slug += f"_frac{str(inner_frac).replace('.', '_')}"
                if first_ns is not None:
                    slug += f"_flns{first_ns}"
                if first_frac is not None:
                    slug += f"_flfrac{str(first_frac).replace('.', '_')}"
                out.append(
                    ExpandedVariant(
                        run_slug=f"{v.name}_{slug}" if v.name else slug,
                        variant=v,
                        fraction=None,
                        num_samples=None,
                        linear=None,
                        hier_k=int(k),
                        hier_shap_type=inner_type,
                        hier_shap_num_samples=int(inner_ns)
                        if inner_ns is not None
                        else None,
                        hier_shap_fraction=float(inner_frac)
                        if inner_frac is not None
                        else None,
                        hier_first_layer_type=first_layer_type.lower(),
                        hier_first_layer_num_samples=int(first_ns)
                        if first_ns is not None
                        else None,
                        hier_first_layer_fraction=float(first_frac)
                        if first_frac is not None
                        else None,
                        hier_importance_min_fraction=float(imp_frac),
                        hier_mode=hier_mode,
                    )
                )

    return out


def expand_variants(cfg: ExperimentSet) -> List[ExpandedVariant]:
    """Expand all configured experiment variants into concrete runnable variants."""
    out: List[ExpandedVariant] = []
    mc_like_types = get_mc_like_explainer_types()

    for v in cfg.experiments:
        explainer_type = v.explainer_type.lower()
        if explainer_type == "exact":
            out.extend(_expand_exact(v))
        elif explainer_type == "hierarchical":
            out.extend(_expand_hierarchical(v))
        elif explainer_type in mc_like_types:
            out.extend(_expand_mc_like(v))
        else:
            raise ValueError(f"Unsupported explainer_type '{v.explainer_type}'")

    LOGGER.info("Expanded %d variants", len(out))
    for item in out:
        LOGGER.info("  - %s", item.run_slug)
    return out
