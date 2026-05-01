"""Factory: build models, explainers, and chats via a pluggable registry."""

from typing import Any, Callable, Dict, Optional, Type, cast

import torch
from mllm_shap.connectors import LiquidAudio, TransformersCausalText, ModelConfig
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.connectors.base.model import BaseMllmModel
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
    SIMILARITY_MAP,
    ChatConfig,
    EmbeddingConfig,
    ExplainerVariant,
    GenerationConfig,
    ShapConfig,
)
from .constants import (
    AUDIO_MODALITIES,
    INTERLEAVED_MODALITIES,
    INTERLEAVED_TEXT_FIRST_MODALITIES,
    ConnectorType,
    ExplainerType,
    InputModality,
    OutputModality,
    TokenFilterType,
)


# ---------------------------
# CONNECTOR REGISTRY
# ---------------------------


def _build_liquid_audio(
    device: torch.device, tracking_mode: ModelHistoryTrackingMode, **kwargs: Any
) -> Any:
    return LiquidAudio(device=device, history_tracking_mode=tracking_mode, **kwargs)


def _build_hf_text(
    device: torch.device, tracking_mode: ModelHistoryTrackingMode, **kwargs: Any
) -> Any:
    return TransformersCausalText(
        device=device, history_tracking_mode=tracking_mode, **kwargs
    )


ConnectorFactory = Callable[..., BaseMllmModel]

CONNECTOR_REGISTRY: Dict[str, ConnectorFactory] = {
    ConnectorType.LIQUID_AUDIO: _build_liquid_audio,
    ConnectorType.TRANSFORMERS_TEXT: _build_hf_text,
}


def register_connector(name: str, factory: ConnectorFactory) -> None:
    """Register a custom connector factory at runtime."""
    CONNECTOR_REGISTRY[name] = factory


def build_model(
    device: torch.device,
    connector: str,
    output_modality: OutputModality = OutputModality.TEXT,
    connector_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construct the selected connector using the registry."""
    if output_modality == OutputModality.AUDIO:
        tracking_mode = ModelHistoryTrackingMode.AUDIO
    else:
        tracking_mode = ModelHistoryTrackingMode.TEXT

    if (
        connector == ConnectorType.TRANSFORMERS_TEXT
        and output_modality == OutputModality.AUDIO
    ):
        raise ValueError(
            "TransformersCausalText connector does not support audio output."
        )

    factory = CONNECTOR_REGISTRY.get(connector)
    if factory is None:
        available = ", ".join(sorted(CONNECTOR_REGISTRY.keys()))
        raise ValueError(f"Unknown connector '{connector}'. Available: {available}")

    extra = connector_kwargs or {}
    return factory(device=device, tracking_mode=tracking_mode, **extra)


# ---------------------------
# TOKEN FILTER
# ---------------------------


def build_token_filter(filter_type: TokenFilterType) -> Any:
    """Build a token filter based on the configured type."""
    if filter_type == TokenFilterType.EXCLUDE_PUNCTUATION:
        return ExcludePunctuationTokensFilter()
    return None  # TokenFilterType.NONE


# ---------------------------
# EXTERNAL EMBEDDING
# ---------------------------


def build_external_embedding(
    model: Any,
    embedding_cfg: EmbeddingConfig,
    device: torch.device,
) -> Optional[CustomEmbedding]:
    """Build CustomEmbedding if configured; returns None otherwise."""
    if not embedding_cfg.model_id:
        return None

    if not hasattr(model, "processor"):
        raise ValueError(
            "External embedding requested but connector has no 'processor'. "
            "Use hf_text connector or one with a tokenizer."
        )

    emb_device = torch.device(embedding_cfg.device) if embedding_cfg.device else device
    return CustomEmbedding(
        generation_tokenizer=model.processor,
        embed_model_id=embedding_cfg.model_id,
        embed_revision=embedding_cfg.revision,
        device=emb_device,
        max_length=embedding_cfg.max_length,
        batch_size=embedding_cfg.batch_size,
        l2_normalize=embedding_cfg.l2_normalize,
        local_files_only=embedding_cfg.local_files_only,
    )


# ---------------------------
# INNER SHAP BUILDER
# ---------------------------

_INNER_SHAP_MAP: Dict[str, Type[BaseShapExplainer]] = {
    "precise": PreciseShapExplainer,
    "limited_mc": LimitedMcShapExplainer,
    "mc": LimitedMcShapExplainer,  # shorthand alias
    "limited_cc": LimitedComplementaryShapExplainer,
    "cc": LimitedComplementaryShapExplainer,  # shorthand alias
    "standard_cc": StandardComplementaryShapExplainer,
    "limited_neyman": LimitedComplementaryNeymanShapExplainer,
    "neyman": LimitedComplementaryNeymanShapExplainer,  # shorthand alias
    "standard_neyman": StandardComplementaryNeymanShapExplainer,
}


def _build_inner_shap(
    which: str,
    common: Dict[str, Any],
    num_samples: Optional[int],
    fraction: Optional[float],
) -> BaseShapExplainer:
    """Build an inner SHAP explainer by name."""
    w = which.lower()
    cls = _INNER_SHAP_MAP.get(w)
    if cls is None:
        raise ValueError(
            f"Unknown inner SHAP type: {which}. Available: {sorted(_INNER_SHAP_MAP)}"
        )

    if w == "precise":
        return cls(**common)

    kw = dict(common)
    if num_samples is not None:
        kw["num_samples"] = int(num_samples)
    elif fraction is not None:
        kw["fraction"] = float(fraction)
    return cls(**kw)


# ---------------------------
# EXPLAINER BUILDER
# ---------------------------


def build_explainer_for_variant(
    device: torch.device,
    shap_cfg: ShapConfig,
    variant: ExplainerVariant,
    connector: str,
    embedding_cfg: EmbeddingConfig,
    output_modality: OutputModality = OutputModality.TEXT,
    connector_kwargs: Optional[Dict[str, Any]] = None,
    concrete_num_samples: Optional[int] = None,
    concrete_fraction: Optional[float] = None,
    # Hierarchical params (from ExpandedVariant)
    hier_k: Optional[int] = None,
    hier_shap_type: Optional[str] = None,
    hier_shap_num_samples: Optional[int] = None,
    hier_shap_fraction: Optional[float] = None,
    hier_first_layer_type: Optional[str] = None,
    hier_first_layer_num_samples: Optional[int] = None,
    hier_first_layer_fraction: Optional[float] = None,
    hier_importance_min_fraction: Optional[float] = None,
    hier_mode: Optional[str] = None,
) -> Explainer:
    """Create (model + shap_explainer) wrapped in Explainer."""
    mode = Mode[shap_cfg.mode]
    normalizer_cls = NORMALIZER_MAP[shap_cfg.normalizer]
    similarity_cls = SIMILARITY_MAP.get(shap_cfg.similarity, TfIdfCosineSimilarity)

    normalizer = normalizer_cls()
    reducer = REDUCER_MAP[shap_cfg.reducer]()
    similarity = similarity_cls()

    model = build_model(
        device=device,
        connector=connector,
        output_modality=output_modality,
        connector_kwargs=connector_kwargs,
    )
    external_emb = build_external_embedding(model, embedding_cfg, device)

    def _common_kwargs() -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "mode": mode,
            "embedding_reducer": reducer,
            "similarity_measure": similarity,
            "normalizer": normalizer,
            "allow_mask_duplicates": shap_cfg.allow_mask_duplicates,
        }
        if external_emb is not None:
            kw["embedding_model"] = external_emb
        return kw

    t = variant.explainer_type

    if t == ExplainerType.EXACT:
        return Explainer(
            model=model, shap_explainer=PreciseShapExplainer(**_common_kwargs())
        )

    if t in (ExplainerType.LIMITED_MC, ExplainerType.STANDARD_MC):
        mc_ctor: Type[Any] = (
            LimitedMcShapExplainer
            if t == ExplainerType.LIMITED_MC
            else StandardMcShapExplainer
        )
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        elif concrete_fraction is not None:
            kw["fraction"] = float(concrete_fraction)
        return Explainer(model=model, shap_explainer=mc_ctor(**kw))

    if t in (ExplainerType.LIMITED_CC, ExplainerType.STANDARD_CC):
        ctor: Type[Any] = (
            LimitedComplementaryShapExplainer
            if t == ExplainerType.LIMITED_CC
            else StandardComplementaryShapExplainer
        )
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        elif concrete_fraction is not None:
            kw["fraction"] = float(concrete_fraction)
        return Explainer(model=model, shap_explainer=ctor(**kw))

    if t == ExplainerType.LIMITED_NEYMAN:
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        elif concrete_fraction is not None:
            kw["fraction"] = float(concrete_fraction)
        return Explainer(
            model=model, shap_explainer=LimitedComplementaryNeymanShapExplainer(**kw)
        )

    if t == ExplainerType.STANDARD_NEYMAN:
        kw = _common_kwargs()
        if concrete_num_samples is not None:
            kw["num_samples"] = int(concrete_num_samples)
        elif concrete_fraction is not None:
            kw["fraction"] = float(concrete_fraction)
        return Explainer(
            model=model, shap_explainer=StandardComplementaryNeymanShapExplainer(**kw)
        )

    if t == ExplainerType.HIERARCHICAL:
        inner_type = (hier_shap_type or "limited_neyman").lower()
        inner = _build_inner_shap(
            which=inner_type,
            common=_common_kwargs(),
            num_samples=hier_shap_num_samples,
            fraction=hier_shap_fraction,
        )

        first_layer: Optional[BaseShapExplainer] = None
        if hier_first_layer_type is not None:
            first_layer = _build_inner_shap(
                which=hier_first_layer_type.lower(),
                common=_common_kwargs(),
                num_samples=hier_first_layer_num_samples,
                fraction=hier_first_layer_fraction,
            )

        # Resolve hierarchical mode
        h_mode = HierMode.MULTI_MODAL_MULTI_USER
        if hier_mode:
            h_mode = HierMode[hier_mode]

        return cast(
            Explainer,
            HierarchicalExplainer(
                model=model,
                shap_explainer=inner,
                first_layer_explainer=first_layer,
                mode=h_mode,
                k=int(hier_k or 10),
                use_importance_sampling=True,
                importance_sampling_min_fraction=float(
                    hier_importance_min_fraction or 0.1
                ),
            ),
        )

    raise ValueError(f"Unsupported explainer_type: {variant.explainer_type}")


# ---------------------------
# GENERATION KWARGS BUILDER
# ---------------------------


def build_generation_kwargs(gen_cfg: GenerationConfig) -> Dict[str, Any]:
    """Build the generation_kwargs dict for model.generate(), passing all exposed params."""
    model_config_kwargs: Dict[str, Any] = {}
    if gen_cfg.text_temperature is not None:
        model_config_kwargs["text_temperature"] = gen_cfg.text_temperature
    if gen_cfg.text_top_k is not None:
        model_config_kwargs["text_top_k"] = gen_cfg.text_top_k
    if gen_cfg.audio_temperature is not None:
        model_config_kwargs["audio_temperature"] = gen_cfg.audio_temperature
    if gen_cfg.audio_top_k is not None:
        model_config_kwargs["audio_top_k"] = gen_cfg.audio_top_k

    return {
        "max_new_tokens": gen_cfg.max_new_tokens,
        "model_config": ModelConfig(**model_config_kwargs),
    }


# ---------------------------
# CHAT BUILDER
# ---------------------------


def infer_audio_format(audio: bytes) -> str:
    """Infer container format from common audio byte signatures."""
    if audio.startswith(b"RIFF"):
        return "wav"
    if audio.startswith(b"ID3") or audio.startswith(b"\xff"):
        return "mp3"
    return "wav"


def build_chat(
    model: Any,
    user_texts: str | list[str] | None,
    audio_bytes_list: bytes | list[bytes] | None,
    input_modality: InputModality = InputModality.TEXT,
    token_filter: Any | None = None,
    audio_segmentation_method: str = "raw",
    aligner: SpectrogramGuidedAligner | None = None,
    chat_cfg: Optional[ChatConfig] = None,
) -> Any:
    """
    Prepare chat turns with the given input modality.

    Supports configurable system_roles_setup and system_prompt via ChatConfig.
    """
    cfg = chat_cfg or ChatConfig()

    # Resolve system roles setup
    roles_setup = SystemRolesSetup[cfg.system_roles_setup]

    chat = model.get_new_chat(
        system_roles_setup=roles_setup,
        token_filter=token_filter,
    )
    chat.new_turn(Role.ASSISTANT)
    chat.add_text(cfg.system_prompt)
    chat.end_turn()

    # Normalize inputs to lists for uniform handling
    if input_modality == InputModality.TEXT:
        texts = [user_texts] if isinstance(user_texts, str) else (user_texts or [""])
        for text in texts:
            if text and text.strip():
                chat.new_turn(Role.USER)
                chat.add_text(text)
                chat.end_turn()

    elif input_modality in AUDIO_MODALITIES:
        if audio_bytes_list is None:
            raise ValueError(
                f"Audio bytes required for input modality: {input_modality}"
            )

        audios = (
            [audio_bytes_list]
            if isinstance(audio_bytes_list, bytes)
            else list(audio_bytes_list)
        )
        audios = [a for a in audios if a is not None]

        if audios:
            chat.new_turn(Role.USER)
            if audio_segmentation_method == "sgpa":
                if aligner is None:
                    raise ValueError("SGPA audio segmentation requires an aligner.")
                if len(audios) != 1:
                    raise ValueError("SGPA audio segmentation expects one audio clip.")
                chat.add_audio_with_transcript(
                    audios[0],
                    transcript=user_texts or "",
                    aligner=aligner,
                    audio_format=infer_audio_format(audios[0]),
                    attach_audio=False,
                )
            else:
                for audio in audios:
                    chat.add_audio(audio, audio_format=infer_audio_format(audio))
            chat.end_turn()

    elif input_modality in INTERLEAVED_MODALITIES:
        if audio_bytes_list is None:
            raise ValueError(
                f"Audio bytes required for input modality: {input_modality}"
            )

        texts = [user_texts] if isinstance(user_texts, str) else list(user_texts or [])
        audios = (
            [audio_bytes_list]
            if isinstance(audio_bytes_list, bytes)
            else list(audio_bytes_list)
        )
        audios = [a for a in audios if a is not None]

        text_first = input_modality in INTERLEAVED_TEXT_FIRST_MODALITIES
        max_sentences = max(len(texts), len(audios))
        chat.new_turn(Role.USER)
        for i in range(max_sentences):
            use_text = (i % 2 == 0) if text_first else (i % 2 == 1)
            if use_text:
                if i < len(texts) and texts[i] and texts[i].strip():
                    chat.add_text(texts[i])
            elif i < len(audios) and audios[i] is not None:
                chat.add_audio(audios[i])
        chat.end_turn()

    chat.refresh(full=True)
    return chat
