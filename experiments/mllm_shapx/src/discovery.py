"""Runtime discovery of available explainers/connectors and capabilities."""

import importlib
import inspect
import pkgutil
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, Set, Type


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase identifiers to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _iter_modules(package_name: str) -> Iterable[Any]:
    """Yield importable modules under a package, skipping broken imports."""
    pkg = importlib.import_module(package_name)
    if not hasattr(pkg, "__path__"):
        return []
    modules = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package_name}."):
        try:
            modules.append(importlib.import_module(info.name))
        except Exception:
            continue
    return modules


def _canonical_explainer_name(cls_name: str) -> str:
    """Map explainer class names to canonical config ids."""
    base = re.sub(r"(ShapExplainer|Explainer)$", "", cls_name)
    name = _camel_to_snake(base)
    name = name.replace("precise", "exact")
    name = name.replace("limited_complementary_neyman", "limited_neyman")
    name = name.replace("standard_complementary_neyman", "standard_neyman")
    name = name.replace("limited_complementary", "limited_cc")
    name = name.replace("standard_complementary", "standard_cc")
    return name


def _canonical_connector_name(cls_name: str) -> str:
    """Map connector class names to canonical config ids."""
    base = re.sub(r"(CausalText|Chat)$", "", cls_name)
    return _camel_to_snake(base)


def _safe_signature_params(cls: Type[Any]) -> Set[str]:
    """Return constructor parameter names, guarding against introspection errors."""
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return set()
    return set(sig.parameters.keys())


@lru_cache(maxsize=1)
def discover_explainer_capabilities() -> Dict[str, Dict[str, bool]]:
    """Discover explainers from mllm_shap.shap and infer capabilities."""
    out: Dict[str, Dict[str, bool]] = {}
    try:
        base_mod = importlib.import_module("mllm_shap.shap.base.shap_explainer")
        base_cls = getattr(base_mod, "BaseShapExplainer")
    except Exception:
        return out

    for mod in _iter_modules("mllm_shap.shap"):
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__:
                continue
            if cls.__name__.startswith("Base"):
                continue
            is_hierarchical_cls = cls.__name__ == "HierarchicalExplainer"
            if not is_hierarchical_cls and not issubclass(cls, base_cls):
                continue

            name = _canonical_explainer_name(cls.__name__)
            params = _safe_signature_params(cls)
            is_hierarchical = ("shap_explainer" in params and "k" in params) or (
                "hierarchical" in name
            )

            out[name] = {
                "supports_num_samples": "num_samples" in params,
                "supports_fraction": "fraction" in params,
                "hierarchical": is_hierarchical,
            }

    # Include common aliases used by configs.
    if "limited_mc" in out:
        out.setdefault("mc", out["limited_mc"])
    if "limited_cc" in out:
        out.setdefault("cc", out["limited_cc"])
    if "limited_neyman" in out:
        out.setdefault("neyman", out["limited_neyman"])

    return out


@lru_cache(maxsize=1)
def get_available_explainer_types() -> Set[str]:
    """Return all discovered explainer ids available in mllm_shap."""
    return set(discover_explainer_capabilities().keys())


@lru_cache(maxsize=1)
def get_mc_like_explainer_types() -> Set[str]:
    """Return discovered explainer ids that support MC-like sampling knobs."""
    caps = discover_explainer_capabilities()
    return {
        name
        for name, meta in caps.items()
        if (
            meta.get("supports_num_samples")
            or meta.get("supports_fraction")
            or ("mc" in name or "cc" in name or "neyman" in name)
        )
        and not meta.get("hierarchical")
        and name != "exact"
    }


@lru_cache(maxsize=1)
def discover_connector_types() -> Set[str]:
    """Discover connector ids from mllm_shap.connectors base model classes."""
    out: Set[str] = set()
    try:
        base_mod = importlib.import_module("mllm_shap.connectors.base.model")
        base_cls = getattr(base_mod, "BaseMllmModel")
    except Exception:
        return out

    for mod in _iter_modules("mllm_shap.connectors"):
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__:
                continue
            if cls.__name__.startswith("Base"):
                continue
            if not issubclass(cls, base_cls):
                continue
            out.add(_canonical_connector_name(cls.__name__))

    # Aliases used in existing mllm_shapx configs.
    if "transformers" in out:
        out.add("hf_text")
    if "open_ai_compat" in out:
        out.add("openai_compat_text")
    if "open_ai_compat_causal_text" in out:
        out.add("openai_compat_text")

    # LM Studio connector is an alias for openai_compat_text with preset defaults.
    if "openai_compat_text" in out or "open_ai_compat" in out:
        out.add("lm_studio_text")

    return out


def is_mc_like_explainer(name: str) -> bool:
    """Return True if an explainer id behaves like MC-style explainers."""
    return name in get_mc_like_explainer_types()


def is_supported_explainer(name: str) -> bool:
    """Return True if the explainer id is discovered at runtime."""
    return name in get_available_explainer_types()


def is_supported_connector(name: str) -> bool:
    """Return True if the connector id is discovered at runtime."""
    return name in discover_connector_types()
