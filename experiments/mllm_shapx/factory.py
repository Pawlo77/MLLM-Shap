"""Factory to build explainers and chats for given variants and model kinds."""
from __future__ import annotations

from typing import Any, Dict, cast

import torch
from mllm_shap.connectors import LiquidAudio, TransformersCausalText
from mllm_shap.connectors.enums import ModelHistoryTrackingMode, Role, SystemRolesSetup
from mllm_shap.connectors.filters import ExcludePunctuationTokensFilter
from mllm_shap.shap import Explainer, McShapExplainer
from mllm_shap.shap.enums import Mode
from mllm_shap.shap.precise import PreciseShapExplainer
from mllm_shap.shap.similarity import CosineSimilarity

from .config import NORMALIZER_MAP, REDUCER_MAP, ExplainerVariant, ShapConfig
from .constants import ExplainerType, ModelKind


def build_explainer_for_variant(
    device: torch.device,
    shap_cfg: ShapConfig,
    variant: ExplainerVariant,
    model_kind: str,
) -> Explainer:
    """
    Create (model + shap_explainer) wrapped in Explainer, with placeholders
    for MC (first entry of num_samples/fractions). Concrete values can be
    re-instantiated later by the runner.
    """
    mode = Mode[shap_cfg.mode]
    normalizer_cls = NORMALIZER_MAP[shap_cfg.normalizer]
    reducer_cls = REDUCER_MAP[shap_cfg.reducer]

    normalizer = normalizer_cls()  # PowerShiftNormalizer() uses default power=1.0
    reducer = reducer_cls()
    similarity = CosineSimilarity()

    if model_kind == ModelKind.TRANSFORMERS_TEXT.value:
        model = TransformersCausalText(device=device, history_tracking_mode=ModelHistoryTrackingMode.TEXT)
    else:
        model = LiquidAudio(device=device, history_tracking_mode=ModelHistoryTrackingMode.TEXT_AUDIO)

    if variant.explainer_type.lower() == ExplainerType.EXACT.value:
        shap_explainer = PreciseShapExplainer(
            mode=mode,
            embedding_reducer=reducer,
            similarity_measure=similarity,
            normalizer=normalizer,
        )
    else:
        # MC variant: use first item as placeholder
        shap_kwargs: Dict[str, Any] = {
            "mode": mode,
            "embedding_reducer": reducer,
            "similarity_measure": similarity,
            "normalizer": normalizer,
        }
        if variant.num_samples:
            shap_explainer = McShapExplainer(num_samples=int(variant.num_samples[0]), **shap_kwargs)
        else:
            fracs = cast(list[float], variant.fractions)
            shap_explainer = McShapExplainer(num_samples=None, fraction=float(fracs[0]), **shap_kwargs)

    return Explainer(model=model, shap_explainer=shap_explainer)


def build_chat(model: Any,
               user_question_text: str,
               audio_bytes: bytes | None,
               text_only: bool = False) -> Any:
    """
    Prepare a chat turn with the given text+audio for the LiquidAudio model.
    """
    chat = model.get_new_chat(
        system_roles_setup=SystemRolesSetup.SYSTEM,
        token_filter=ExcludePunctuationTokensFilter(),
    )
    chat.new_turn(Role.USER)
    chat.add_text(user_question_text)
    if not text_only and audio_bytes is not None:
        chat.add_audio(audio_bytes)
    chat.end_turn()
    return chat
