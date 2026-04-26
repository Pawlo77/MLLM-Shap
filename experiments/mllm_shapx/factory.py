"""Factory to build explainers and chats for given variants and model kinds."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, cast

import torch
from mllm_shap.connectors import LiquidAudio, TransformersCausalText
from mllm_shap.connectors.enums import (
    ModelHistoryTrackingMode,
    Role,
    SystemRolesSetup,
)
from mllm_shap.connectors.filters import ExcludePunctuationTokensFilter
from mllm_shap.shap import Explainer
from mllm_shap.shap.base.shap_explainer import BaseShapExplainer
from mllm_shap.shap.complementary import (
    LimitedComplementaryShapExplainer,
    StandardComplementaryShapExplainer,
)
from mllm_shap.shap.embeddings import CustomEmbedding
from mllm_shap.shap.enums import Mode
from mllm_shap.shap.hierarchical import HierarchicalExplainer
from mllm_shap.shap.hierarchical.enums import Mode as HierMode
from mllm_shap.shap.monte_carlo import LimitedMcShapExplainer, StandardMcShapExplainer
from mllm_shap.shap.neyman import (
    LimitedComplementaryNeymanShapExplainer,
    StandardComplementaryNeymanShapExplainer,
)
from mllm_shap.shap.precise import PreciseShapExplainer
from mllm_shap.shap.similarity import TfIdfCosineSimilarity

from .config import (
    NORMALIZER_MAP,
    REDUCER_MAP,
    ExplainerVariant,
    ShapConfig,
    SIMILARITY_MAP,
)
from .constants import ExplainerType, ConnectorType, InputModality, OutputModality


def _build_model(
    device: torch.device,
    connector: str,
    output_modality: OutputModality = OutputModality.TEXT,
) -> Any:
    """
    Construct the selected connector with appropriate history tracking mode.

    Args:
        device: Torch device to use.
        connector: Connector type string.
        output_modality: Output modality (TEXT or AUDIO).

    Returns:
        Configured model connector.
    """
    # Determine history tracking mode based on output modality
    if output_modality == OutputModality.AUDIO:
        tracking_mode = ModelHistoryTrackingMode.AUDIO
    else:
        tracking_mode = ModelHistoryTrackingMode.TEXT

    if connector == ConnectorType.TRANSFORMERS_TEXT.value:
        if output_modality == OutputModality.AUDIO:
            raise ValueError(
                "TransformersCausalText connector does not support audio output."
            )
        return TransformersCausalText(
            device=device,
            history_tracking_mode=tracking_mode,
        )
    # default: LiquidAudio
    return LiquidAudio(
        device=device,
        history_tracking_mode=tracking_mode,
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
_LIMITED_NEYMAN_TYPE = "limited_neyman"
_STANDARD_NEYMAN_TYPE = "standard_neyman"


def _build_inner_shap(
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
    if w == _LIMITED_NEYMAN_TYPE:
        kw = dict(common)
        if num_samples is not None:
            kw["num_samples"] = int(num_samples)
        else:
            _maybe_add_fraction(kw, fraction)
        return LimitedComplementaryNeymanShapExplainer(**kw)
    if w == _STANDARD_NEYMAN_TYPE:
        kw = dict(common)
        if num_samples is not None:
            kw["num_samples"] = int(num_samples)
        else:
            _maybe_add_fraction(kw, fraction)
        return StandardComplementaryNeymanShapExplainer(**kw)
    raise ValueError(f"Unknown inner SHAP type: {which}")


def build_explainer_for_variant(
    device: torch.device,
    shap_cfg: ShapConfig,
    variant: ExplainerVariant,
    connector: str,
    *,
    embedding_cfg: Any | None,
    output_modality: OutputModality = OutputModality.TEXT,
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

    Args:
        device: Torch device to use.
        shap_cfg: SHAP configuration.
        variant: Explainer variant configuration.
        connector: Connector type string.
        embedding_cfg: External embedding configuration.
        output_modality: Output modality (TEXT or AUDIO) - controls history tracking mode.
        concrete_num_samples: Concrete number of samples to use (overrides fraction).
        concrete_fraction: Concrete fraction of samples to use (if num_samples is None).
        hier_k: For hierarchical explainer, number of top-K users to consider.
        hier_shap_type: For hierarchical explainer, inner SHAP type.
        hier_shap_num_samples: For hierarchical explainer, inner SHAP num_samples.
        hier_shap_fraction: For hierarchical explainer, inner SHAP fraction.
        hier_first_layer_type: For hierarchical explainer, first-layer SHAP type.
        hier_first_layer_num_samples: For hierarchical explainer, first-layer SHAP num_samples.
        hier_first_layer_fraction: For hierarchical explainer, first-layer SHAP fraction.
        hier_importance_min_fraction: For hierarchical explainer, min fraction for importance sampling.
    """
    mode = Mode[shap_cfg.mode]
    normalizer_cls = NORMALIZER_MAP[shap_cfg.normalizer]
    similarity_cls = SIMILARITY_MAP.get(shap_cfg.similarity, TfIdfCosineSimilarity)

    normalizer = normalizer_cls()
    reducer = REDUCER_MAP[shap_cfg.reducer]()
    similarity = similarity_cls()

    model = _build_model(
        device=device, connector=connector, output_modality=output_modality
    )
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
        return Explainer(
            model=model, shap_explainer=PreciseShapExplainer(**_common_kwargs())
        )

    if t in (ExplainerType.LIMITED_MC.value, ExplainerType.STANDARD_MC.value):
        mc_ctor: Type[Any] = (
            LimitedMcShapExplainer
            if t == ExplainerType.LIMITED_MC.value
            else StandardMcShapExplainer
        )
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        else:
            _maybe_add_fraction(kw, concrete_fraction)
        return Explainer(model=model, shap_explainer=mc_ctor(**kw))

    if t in (ExplainerType.LIMITED_CC.value, ExplainerType.STANDARD_CC.value):
        ctor: Type[Any] = (
            LimitedComplementaryShapExplainer
            if t == ExplainerType.LIMITED_CC.value
            else StandardComplementaryShapExplainer
        )
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        else:
            _maybe_add_fraction(kw, concrete_fraction)
        return Explainer(model=model, shap_explainer=ctor(**kw))

    if t == ExplainerType.LIMITED_NEYMAN.value:
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        else:
            _maybe_add_fraction(kw, concrete_fraction)
        return Explainer(
            model=model, shap_explainer=LimitedComplementaryNeymanShapExplainer(**kw)
        )

    if t == ExplainerType.STANDARD_NEYMAN.value:
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        else:
            _maybe_add_fraction(kw, concrete_fraction)
        return Explainer(
            model=model, shap_explainer=StandardComplementaryNeymanShapExplainer(**kw)
        )

    if t == ExplainerType.HIERARCHICAL.value:
        none_str = "none"
        # inner shap
        inner_type = (hier_shap_type or _LIMITED_NEYMAN_TYPE).lower()
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

        return cast(
            Explainer,
            HierarchicalExplainer(
                model=model,
                shap_explainer=inner,
                first_layer_explainer=first_layer,
                mode=HierMode.MULTI_MODAL_MULTI_USER,
                k=int(hier_k or 10),
                use_importance_sampling=True,
                importance_sampling_min_fraction=float(
                    hier_importance_min_fraction or 0.1
                ),
            ),
        )

    raise ValueError(f"Unsupported explainer_type: {variant.explainer_type}")


# Interleaved modality groupings for easier checking
INTERLEAVED_TEXT_FIRST_MODALITIES = (
    InputModality.INTERLEAVED_TEXT_FIRST_MALE,
    InputModality.INTERLEAVED_TEXT_FIRST_FEMALE,
)
INTERLEAVED_AUDIO_FIRST_MODALITIES = (
    InputModality.INTERLEAVED_AUDIO_FIRST_MALE,
    InputModality.INTERLEAVED_AUDIO_FIRST_FEMALE,
)
INTERLEAVED_MODALITIES = (
    INTERLEAVED_TEXT_FIRST_MODALITIES + INTERLEAVED_AUDIO_FIRST_MODALITIES
)

AUDIO_MODALITIES = (
    InputModality.AUDIO_MALE,
    InputModality.AUDIO_FEMALE,
)


def build_chat(
    model: Any,
    user_texts: str | list[str] | None,
    audio_bytes_list: bytes | list[bytes] | None,
    input_modality: InputModality = InputModality.TEXT,
    *,
    token_filter: Any | None = None,
) -> Any:
    """
    Prepare chat turns with the given input modality for the model.

    For TEXT modality: supports multi-turn conversations where each sentence becomes a separate user turn.
    For AUDIO modality: all audio clips are concatenated into a single audio and added to one user turn
                        (required for SHAP masking compatibility with the liquid_audio model).
    For INTERLEAVED modalities: alternates between text and audio turns, starting with the specified type.
                                Audio clips are concatenated per-turn to maintain SHAP compatibility.

    Args:
        model: The model instance.
        user_texts: Text input(s) - single string or list of strings for multi-turn.
        audio_bytes_list: Audio content - single bytes or list of bytes (will be concatenated for audio-only,
                          or used per-turn for interleaved).
        input_modality: The input modality to use.
        token_filter: Optional token filter.

    Returns:
        Configured chat instance.
    """
    tf = token_filter or ExcludePunctuationTokensFilter()

    chat = model.get_new_chat(
        system_roles_setup=SystemRolesSetup.SYSTEM_ASSISTANT,
        token_filter=tf,
    )
    chat.new_turn(Role.ASSISTANT)
    chat.add_text("You are a helpful assistant.")
    chat.end_turn()

    # Normalize inputs to lists for uniform handling
    if input_modality == InputModality.TEXT:
        # For text: create separate user turns for each sentence
        texts = [user_texts] if isinstance(user_texts, str) else (user_texts or [""])
        for text in texts:
            if text and text.strip():
                chat.new_turn(Role.USER)
                chat.add_text(text)
                chat.end_turn()

    elif input_modality in AUDIO_MODALITIES:
        # For audio: add all clips to a single user turn
        if audio_bytes_list is None:
            raise ValueError(
                f"Audio bytes required for input modality: {input_modality}"
            )

        # Normalize to list and filter None values
        audios = (
            [audio_bytes_list]
            if isinstance(audio_bytes_list, bytes)
            else list(audio_bytes_list)
        )
        audios = [a for a in audios if a is not None]

        if audios:
            chat.new_turn(Role.USER)
            for audio in audios:
                chat.add_audio(audio)
            chat.end_turn()

    elif input_modality in INTERLEAVED_MODALITIES:
        # Interleaved: alternate between text and audio in subsequent turns
        # Each sentence uses EITHER text OR audio, alternating by index
        if audio_bytes_list is None:
            raise ValueError(
                f"Audio bytes required for input modality: {input_modality}"
            )

        # Normalize inputs
        texts = [user_texts] if isinstance(user_texts, str) else list(user_texts or [])
        audios = (
            [audio_bytes_list]
            if isinstance(audio_bytes_list, bytes)
            else list(audio_bytes_list)
        )
        audios = [a for a in audios if a is not None]

        # Determine starting modality
        text_first = input_modality in INTERLEAVED_TEXT_FIRST_MODALITIES

        # Build interleaved turns - each sentence alternates between modalities
        # text_first=True:  sentence 0 → text, sentence 1 → audio, sentence 2 → text, ...
        # text_first=False: sentence 0 → audio, sentence 1 → text, sentence 2 → audio, ...
        max_sentences = max(len(texts), len(audios))
        chat.new_turn(Role.USER)
        for i in range(max_sentences):
            use_text = (i % 2 == 0) if text_first else (i % 2 == 1)

            if use_text:
                # This sentence uses text
                if i < len(texts) and texts[i] and texts[i].strip():
                    chat.add_text(texts[i])
            elif i < len(audios) and audios[i] is not None:
                chat.add_audio(audios[i])
        chat.end_turn()

    chat.refresh(full=True)
    return chat
