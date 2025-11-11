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
from mllm_shap.shap import Explainer, ComplementaryNeymanShapExplainer, ComplementaryShapExplainer
from mllm_shap.shap.monte_carlo import LimitedMcShapExplainer, StandardMcShapExplainer
from mllm_shap.shap.enums import Mode
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


def build_explainer_for_variant(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches
    device: torch.device,
    shap_cfg: ShapConfig,
    variant: ExplainerVariant,
    connector: str,
    *,
    embedding_cfg: Any | None,
    concrete_num_samples: int | None = None,
    concrete_fraction: float | None = None,
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

    t = variant.explainer_type.lower()

    if t == ExplainerType.EXACT.value:
        kwargs: Dict[str, Any] = {
            "mode": mode,
            "embedding_reducer": reducer,
            "similarity_measure": similarity,
            "normalizer": normalizer,
        }
        if external_emb is not None:
            kwargs["embedding_model"] = external_emb
        shap_explainer = PreciseShapExplainer(**kwargs)

    elif t in (ExplainerType.LIMITED_MC.value, ExplainerType.STANDARD_MC.value):
        ctor: Type[Any] = (
            LimitedMcShapExplainer
            if t == ExplainerType.LIMITED_MC.value
            else StandardMcShapExplainer
        )
        kwargs = {
            "mode": mode,
            "embedding_reducer": reducer,
            "similarity_measure": similarity,
            "normalizer": normalizer,
        }
        if external_emb is not None:
            kwargs["embedding_model"] = external_emb

        if concrete_num_samples is not None:
            shap_explainer = ctor(num_samples=int(concrete_num_samples), **kwargs)
        else:
            shap_explainer = ctor(
                num_samples=None,
                fraction=float(concrete_fraction) if concrete_fraction is not None else None,
                **kwargs,
            )

    elif t == ExplainerType.COMPLEMENTARY.value:
        # MC-like: honor num_samples/fraction
        kwargs = {
            "mode": mode,
            "embedding_reducer": reducer,
            "similarity_measure": similarity,
            "normalizer": normalizer,
        }
        if external_emb is not None:
            kwargs["embedding_model"] = external_emb
        if concrete_num_samples is not None:
            shap_explainer = ComplementaryShapExplainer(num_samples=int(concrete_num_samples), **kwargs)
        else:
            shap_explainer = ComplementaryShapExplainer(
                num_samples=None,
                fraction=float(concrete_fraction) if concrete_fraction is not None else None,
                **kwargs,
            )

    elif t == ExplainerType.NEYMAN.value:
        kwargs = {
            "mode": mode,
            "embedding_reducer": reducer,
            "similarity_measure": similarity,
            "normalizer": normalizer,
        }
        if external_emb is not None:
            kwargs["embedding_model"] = external_emb
        if concrete_num_samples is not None:
            shap_explainer = ComplementaryNeymanShapExplainer(num_samples=int(concrete_num_samples), **kwargs)
        else:
            shap_explainer = ComplementaryNeymanShapExplainer(
                num_samples=None,
                fraction=float(concrete_fraction) if concrete_fraction is not None else None,
                **kwargs,
            )

    else:
        raise ValueError(f"Unsupported explainer_type: {variant.explainer_type}")

    return Explainer(model=model, shap_explainer=shap_explainer)


def build_chat(  # pylint: disable=too-many-arguments
    model: Any,
    user_text: str,
    audio_bytes: bytes | None,
    text_only: bool = False,
    *,
    token_filter: Any | None = None,
    is_neyman: bool = False
) -> Any:
    """Prepare a chat turn with the given text+audio for the LiquidAudio model."""
    tf = token_filter or ExcludePunctuationTokensFilter()

    chat = None
    if is_neyman:
        chat = model.get_new_chat(
            system_roles_setup=SystemRolesSetup.SYSTEM_ASSISTANT,
            token_filter=tf,
        )
        chat.new_turn(Role.ASSISTANT)
        chat.add_text("You are a helpful assitant.")
        chat.end_turn()
        chat.new_turn(Role.USER)
        chat.add_text(user_text)
    else:
        chat = model.get_new_chat(
            system_roles_setup=SystemRolesSetup.NONE,
            token_filter=tf,
        )
        chat.new_turn(Role.USER)
        chat.add_text(user_text)

    if not text_only and audio_bytes is not None:
        chat.add_audio(audio_bytes)

    chat.end_turn()
    chat.refresh(full=True)
    return chat
