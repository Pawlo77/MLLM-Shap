"""Factory to build explainers and chats for given variants and model kinds."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch

from mllm_shap.connectors import LiquidAudio, TransformersCausalText
from mllm_shap.connectors.enums import (
    ModelHistoryTrackingMode,
    Role,
    SystemRolesSetup,
)
from mllm_shap.connectors.filters import ExcludePunctuationTokensFilter
from mllm_shap.shap import Explainer, ComplementaryNeymanShapExplainer
from mllm_shap.shap.base.shap_explainer import BaseShapExplainer
from mllm_shap.shap.monte_carlo import LimitedMcShapExplainer, StandardMcShapExplainer
from mllm_shap.shap.complementary import LimitedComplementaryShapExplainer, StandardComplementaryShapExplainer
from mllm_shap.shap.hierarchical import HierarchicalExplainer
from mllm_shap.shap.enums import Mode
from mllm_shap.shap.hierarchical.enums import Mode as HierMode
from mllm_shap.shap.precise import PreciseShapExplainer
from mllm_shap.shap.similarity import TfIdfCosineSimilarity
from mllm_shap.shap.embeddings import CustomEmbedding

from .config import NORMALIZER_MAP, REDUCER_MAP, ExplainerVariant, ShapConfig, SIMILARITY_MAP
from .constants import ExplainerType, ConnectorType


def _build_model(device: torch.device, connector: str) -> Any:
    """Construct the selected connector."""
    if connector == ConnectorType.TRANSFORMERS_TEXT.value:
        return TransformersCausalText(
            device=device,
            history_tracking_mode=ModelHistoryTrackingMode.TEXT,
        )
    # default: LiquidAudio
    return LiquidAudio(
        device=device,
        history_tracking_mode=ModelHistoryTrackingMode.TEXT_AUDIO,
    )


def _build_external_embedding(
    model: Any, embedding_cfg: Any, device: torch.device
) -> Optional[CustomEmbedding]:
    """Build CustomEmbedding if requested; returns None if not configured."""
    if not embedding_cfg or not getattr(embedding_cfg, "model_id", None):
        return None

    if not hasattr(model, "processor"):
        raise ValueError(
            "External embedding requested but the selected connector exposes no tokenizer 'processor'. "
            "Use the Transformers text connector (hf_text) or a connector with a tokenizer."
        )

    return CustomEmbedding(
        generation_tokenizer=model.processor,
        embed_model_id=embedding_cfg.model_id,
        embed_revision=embedding_cfg.revision,
        device=device,
        max_length=int(embedding_cfg.max_length),
        batch_size=int(embedding_cfg.batch_size),
        l2_normalize=bool(embedding_cfg.l2_normalize),
        local_files_only=bool(embedding_cfg.local_files_only),
    )


def _maybe_add_fraction(kwargs: Dict[str, Any], fraction: Optional[float]) -> None:
    if fraction is not None:
        kwargs["fraction"] = float(fraction)  # never pass None


# Explainer type names as constants
_PRECISE_TYPE = "precise"
_LIMITED_MC_TYPE = "limited_mc"
_LIMITED_CC_TYPE = "limited_cc"
_STANDARD_CC_TYPE = "standard_cc"
_NEYMAN_TYPE = "neyman"


def _build_inner_shap(  # pylint: disable=too-many-branches
    which: str,
    common: Dict[str, Any],
    num_samples: Optional[int],
    fraction: Optional[float],
) -> BaseShapExplainer:
    w = which.lower()
    if w == _PRECISE_TYPE:
        return PreciseShapExplainer(**common)
    if w in (_LIMITED_MC_TYPE,):
        kw = dict(common)
        if num_samples is not None:
            kw["num_samples"] = int(num_samples)
        else:
            _maybe_add_fraction(kw, fraction)
        return LimitedMcShapExplainer(**kw)
    if w in (_LIMITED_CC_TYPE,):
        kw = dict(common)
        if num_samples is not None:
            kw["num_samples"] = int(num_samples)
        else:
            _maybe_add_fraction(kw, fraction)
        return LimitedComplementaryShapExplainer(**kw)
    if w in (_STANDARD_CC_TYPE,):
        kw = dict(common)
        if num_samples is not None:
            kw["num_samples"] = int(num_samples)
        else:
            _maybe_add_fraction(kw, fraction)
        return StandardComplementaryShapExplainer(**kw)
    if w == _NEYMAN_TYPE:
        kw = dict(common)
        if num_samples is not None:
            kw["num_samples"] = int(num_samples)
        else:
            _maybe_add_fraction(kw, fraction)
        return ComplementaryNeymanShapExplainer(**kw)
    raise ValueError(f"Unknown inner SHAP type: {which}")


def build_explainer_for_variant(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches
    device: torch.device,
    shap_cfg: ShapConfig,
    variant: ExplainerVariant,
    connector: str,
    *,
    embedding_cfg: Any | None,
    concrete_num_samples: int | None = None,
    concrete_fraction: float | None = None,
    hier_k: Optional[int] = None,
    hier_shap_type: Optional[str] = None,
    hier_shap_num_samples: Optional[int] = None,
    hier_shap_fraction: Optional[float] = None,
    hier_first_layer_type: Optional[str] = None,
    hier_first_layer_num_samples: Optional[int] = None,
    hier_first_layer_fraction: Optional[float] = None,
    hier_importance_min_fraction: Optional[float] = None,
) -> Explainer:
    """
    Create (model + shap_explainer) wrapped in Explainer.
    - complementary: MC-like; pass through num_samples/fraction
    - neyman: AUTO mode; explicitly pass neither num_samples nor fraction
    """
    mode = Mode[shap_cfg.mode]
    normalizer_cls = NORMALIZER_MAP[shap_cfg.normalizer]
    similarity_cls = SIMILARITY_MAP.get(shap_cfg.similarity, TfIdfCosineSimilarity)

    normalizer = normalizer_cls()
    reducer = REDUCER_MAP[shap_cfg.reducer]()
    similarity = similarity_cls()

    model = _build_model(device=device, connector=connector)
    external_emb = _build_external_embedding(model, embedding_cfg, device)

    def _common_kwargs() -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "mode": mode,
            "embedding_reducer": reducer,
            "similarity_measure": similarity,
            "normalizer": normalizer,
        }
        if external_emb is not None:
            kw["embedding_model"] = external_emb
        return kw

    t = variant.explainer_type.lower()

    if t == ExplainerType.EXACT.value:
        return Explainer(model=model, shap_explainer=PreciseShapExplainer(**_common_kwargs()))

    if t in (ExplainerType.LIMITED_MC.value, ExplainerType.STANDARD_MC.value):
        mc_ctor: Type[Any] = (LimitedMcShapExplainer
                              if t == ExplainerType.LIMITED_MC.value
                              else StandardMcShapExplainer)
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        else:
            _maybe_add_fraction(kw, concrete_fraction)
        return Explainer(model=model, shap_explainer=mc_ctor(**kw))

    if t in (ExplainerType.LIMITED_CC.value, ExplainerType.STANDARD_CC.value):
        ctor: Type[Any] = (LimitedComplementaryShapExplainer
                           if t == ExplainerType.LIMITED_CC.value
                           else StandardComplementaryShapExplainer)
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        else:
            _maybe_add_fraction(kw, concrete_fraction)
        return Explainer(model=model, shap_explainer=ctor(**kw))

    if t == ExplainerType.NEYMAN.value:
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        else:
            _maybe_add_fraction(kw, concrete_fraction)
        return Explainer(model=model, shap_explainer=ComplementaryNeymanShapExplainer(**kw))

    if t == ExplainerType.HIERARCHICAL.value:
        none_str = "none"
        # inner shap
        inner_type = (hier_shap_type or _NEYMAN_TYPE).lower()
        inner = _build_inner_shap(
            which=inner_type,
            common=_common_kwargs(),
            num_samples=hier_shap_num_samples,
            fraction=hier_shap_fraction,
        )
        # first-layer (optional)
        fl_type = (hier_first_layer_type or none_str).lower()
        first_layer: Optional[BaseShapExplainer] = None
        if fl_type != none_str:
            first_layer = _build_inner_shap(
                which=fl_type,
                common=_common_kwargs(),
                num_samples=hier_first_layer_num_samples,
                fraction=hier_first_layer_fraction,
            )

        return HierarchicalExplainer(
            model=model,
            shap_explainer=inner,
            first_layer_explainer=first_layer,
            mode=HierMode.MULTI_MODAL_MULTI_USER,
            k=int(hier_k or 10),
            use_importance_sampling=True,
            importance_sampling_min_fraction=float(hier_importance_min_fraction or 0.1),
        )

    raise ValueError(f"Unsupported explainer_type: {variant.explainer_type}")


def build_chat(  # pylint: disable=too-many-arguments
    model: Any,
    user_text: str,
    audio_bytes: bytes | None,
    text_only: bool = False,
    *,
    token_filter: Any | None = None,
) -> Any:
    """Prepare a chat turn with the given text+audio for the LiquidAudio model."""
    tf = token_filter or ExcludePunctuationTokensFilter()

    chat = None
    chat = model.get_new_chat(
        system_roles_setup=SystemRolesSetup.SYSTEM_ASSISTANT,
        token_filter=tf,
    )
    chat.new_turn(Role.ASSISTANT)
    chat.add_text("You are a helpful assitant.")
    chat.end_turn()
    chat.new_turn(Role.USER)
    chat.add_text(user_text)

    if not text_only and audio_bytes is not None:
        chat.add_audio(audio_bytes)

    chat.end_turn()
    chat.refresh(full=True)
    return chat
