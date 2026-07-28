"""Model-agnostic audio->text backend interface + factory.

Every second-model harness talks to its model exclusively through the
``AudioTextBackend`` protocol, so the faithfulness runner (``mm_faith``) stays
model-independent. Adding a new end-to-end audio LM is one new backend class
plus one entry in ``build_backend`` -- the SGPA aligner, exact-Shapley SV,
utilities and analysis are all reused unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AudioTextBackend(Protocol):
    """Minimal deterministic audio->text decoding contract.

    Implementations must expose the target sample rate they expect and a single
    deterministic ``generate_text`` call taking a mono waveform at ``target_sr``.
    """

    target_sr: int

    def generate_text(
        self,
        waveform_16k: np.ndarray,
        instruction: str = ...,
        max_new_tokens: int = ...,
    ) -> str: ...


def build_backend(name: str, device: str = "cuda", **kwargs) -> AudioTextBackend:
    """Instantiate a backend by short name.

    Args:
        name: one of ``{"qwen", "voxtral"}``.
        device: torch device string.
        kwargs: forwarded to the concrete backend constructor.
    """
    key = name.lower()
    if key in {"qwen", "qwen2-audio", "qwen2_audio"}:
        from .qwen_audio_backend import QwenAudioBackend

        return QwenAudioBackend(device=device, **kwargs)
    if key in {"voxtral", "voxtral-mini", "voxtral_mini", "mistral-audio"}:
        from .voxtral_audio_backend import VoxtralAudioBackend

        return VoxtralAudioBackend(device=device, **kwargs)
    raise ValueError(f"Unknown backend '{name}'. Available: qwen, voxtral.")
