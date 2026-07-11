"""Runtime-discovered registries and allowed-value sets for config validation."""

import inspect
from typing import Any, Callable, Dict, Mapping

from mllm_shap.shap.base.embeddings import BaseEmbeddingReducer
from mllm_shap.shap.base.normalizers import BaseNormalizer
from mllm_shap.shap import embeddings as embeddings_module
from mllm_shap.shap import normalizers as normalizers_module
from mllm_shap.shap.enums import Mode as ShapMode
from mllm_shap.shap.hierarchical.enums import Mode as HierarchicalMode
from mllm_shap.shap import similarity as similarity_module

# Allowed SHAP mode names from package enum.
ALLOWED_SHAP_MODES = frozenset(mode.name for mode in ShapMode)

# Allowed hierarchical mode names from package enum.
ALLOWED_HIERARCHICAL_MODES = frozenset(mode.name for mode in HierarchicalMode)

# Allowed inner explainer ids for hierarchical composition.
ALLOWED_HIERARCHICAL_INNER_TYPES = frozenset(
    {
        "precise",
        "limited_mc",
        "limited_cc",
        "standard_cc",
        "limited_neyman",
        "standard_neyman",
    }
)


def _is_default_constructible(cls: type[Any]) -> bool:
    """Return True when ``cls()`` is valid without required constructor args."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return False

    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty:
            return False
    return True


def _discover_module_classes(
    module: Any,
    
    base_cls: type[Any],
    include_name: Callable[[str], bool] | None = None,
    exclude_names: set[str] | None = None,
) -> Dict[str, Any]:
    """Discover concrete, default-constructible classes from a module."""
    exclude_names = exclude_names or set()
    discovered: Dict[str, Any] = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if name in exclude_names:
            continue
        if include_name is not None and not include_name(name):
            continue
        if not issubclass(obj, base_cls):
            continue
        if obj is base_cls or inspect.isabstract(obj):
            continue
        if not _is_default_constructible(obj):
            continue
        discovered[name] = obj
    return discovered


def _discover_normalizer_map() -> Dict[str, type[BaseNormalizer]]:
    """Discover available SHAP normalizers from ``mllm_shap.shap.normalizers``."""
    return _discover_module_classes(normalizers_module, base_cls=BaseNormalizer)


def _discover_reducer_map() -> Mapping[str, Callable[[], BaseEmbeddingReducer]]:
    """Discover available embedding reducers from ``mllm_shap.shap.embeddings``."""
    return _discover_module_classes(embeddings_module, base_cls=BaseEmbeddingReducer)


def _discover_similarity_map() -> Dict[str, Any]:
    """Discover available similarity classes from ``mllm_shap.shap.similarity``."""
    return _discover_module_classes(
        similarity_module,
        base_cls=object,
        include_name=lambda name: name.endswith("Similarity"),
        exclude_names={"Similarity"},
    )


# Runtime-discovered SHAP normalizers keyed by class name.
NORMALIZER_MAP: Dict[str, type[BaseNormalizer]] = _discover_normalizer_map()

# Runtime-discovered embedding reducers keyed by class name.
REDUCER_MAP: Mapping[str, Callable[[], BaseEmbeddingReducer]] = _discover_reducer_map()

# Runtime-discovered similarity classes keyed by class name.
SIMILARITY_MAP: Dict[str, Any] = _discover_similarity_map()
