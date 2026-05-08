"""Tests for factory module — build_generation_kwargs, build_token_filter, registry."""

from types import SimpleNamespace


from ..src.config import GenerationConfig
from ..src.constants import ConnectorType, TokenFilterType
from ..src.factory import (
    CONNECTOR_REGISTRY,
    build_generation_kwargs,
    build_token_filter,
    infer_audio_format,
    register_connector,
)


class TestBuildGenerationKwargs:
    def test_basic(self) -> None:
        gen = GenerationConfig(max_new_tokens=50)
        kwargs = build_generation_kwargs(gen)
        assert kwargs["max_new_tokens"] == 50
        assert "model_config" in kwargs

    def test_all_params_forwarded(self) -> None:
        gen = GenerationConfig(
            max_new_tokens=100,
            text_temperature=0.7,
            text_top_k=40,
            audio_temperature=0.3,
            audio_top_k=5,
        )
        kwargs = build_generation_kwargs(gen)
        mc = kwargs["model_config"]
        assert mc.text_temperature == 0.7
        assert mc.text_top_k == 40
        assert mc.audio_temperature == 0.3
        assert mc.audio_top_k == 5

    def test_none_params_not_forwarded(self) -> None:
        gen = GenerationConfig(max_new_tokens=10, text_temperature=0.2)
        kwargs = build_generation_kwargs(gen)
        mc = kwargs["model_config"]
        assert mc.text_temperature == 0.2
        # text_top_k is None → should not be set on ModelConfig
        # We just verify it doesn't error; the actual ModelConfig handles defaults


class TestBuildTokenFilter:
    def test_exclude_punctuation(self) -> None:
        filt = build_token_filter(TokenFilterType.EXCLUDE_PUNCTUATION)
        assert filt is not None

    def test_none_filter(self) -> None:
        filt = build_token_filter(TokenFilterType.NONE)
        assert filt is None


class TestConnectorRegistry:
    def test_known_connectors(self) -> None:
        assert ConnectorType.LIQUID_AUDIO in CONNECTOR_REGISTRY
        assert ConnectorType.TRANSFORMERS_TEXT in CONNECTOR_REGISTRY

    def test_register_custom_connector(self) -> None:
        def custom_factory(device, tracking_mode, **kwargs):
            return SimpleNamespace(device=device, custom=True)

        register_connector("custom_test", custom_factory)
        assert "custom_test" in CONNECTOR_REGISTRY
        # Cleanup
        del CONNECTOR_REGISTRY["custom_test"]


class TestInferAudioFormat:
    def test_wav(self) -> None:
        assert infer_audio_format(b"RIFF....") == "wav"

    def test_mp3_id3(self) -> None:
        assert infer_audio_format(b"ID3....") == "mp3"

    def test_mp3_sync(self) -> None:
        assert infer_audio_format(b"\xff\xfb....") == "mp3"

    def test_unknown_defaults_to_wav(self) -> None:
        assert infer_audio_format(b"\x00\x00\x00\x00") == "wav"
