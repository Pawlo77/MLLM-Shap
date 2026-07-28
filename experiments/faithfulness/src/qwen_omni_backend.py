"""Minimal Qwen2.5-Omni audio->text backend for second-model faithfulness.

Loads only the Thinker path (``enable_audio_output=False``) so a 3B end-to-end
audio LM fits in 12 GB VRAM. Exposes a single ``generate_text`` call: given a
waveform + instruction, return the decoded text response. This is intentionally
tiny -- it does NOT implement the full ``mllm_shap`` connector interface, only
what the standalone exact-Shapley faithfulness runner needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class QwenOmniBackend:
    """Thin wrapper around Qwen2.5-Omni for deterministic audio->text decoding."""

    def __init__(
        self,
        repo: str = "Qwen/Qwen2.5-Omni-3B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from transformers import (  # local import to keep module light
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        self.device = device
        self.processor = Qwen2_5OmniProcessor.from_pretrained(repo)
        load_kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device}
        try:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                repo, enable_audio_output=False, **load_kwargs
            )
        except TypeError:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                repo, **load_kwargs
            )
        self.model.eval()
        try:
            self.target_sr = int(self.processor.feature_extractor.sampling_rate)
        except Exception:  # noqa: BLE001
            self.target_sr = 16000

    @torch.no_grad()
    def generate_text(
        self,
        waveform_16k: np.ndarray,
        instruction: str = "Repeat the exact words that the speaker said.",
        max_new_tokens: int = 48,
    ) -> str:
        """Decode a deterministic text response for a mono 16 kHz waveform."""
        audio = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text,
            audio=[audio],
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        gen = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_audio=False,
        )
        if isinstance(gen, (tuple, list)):
            gen = gen[0]
        in_len = int(inputs["input_ids"].shape[1])
        new_tokens = gen[:, in_len:]
        return self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def _smoke() -> None:
    """Load the model and decode a synthetic tone + report VRAM, to validate the API."""
    backend = QwenOmniBackend()
    print("target_sr:", backend.target_sr, flush=True)
    sr = backend.target_sr
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    out = backend.generate_text(
        tone, instruction="What do you hear?", max_new_tokens=16
    )
    print("SMOKE OUTPUT:", repr(out), flush=True)
    if torch.cuda.is_available():
        print(
            f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB "
            f"reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB",
            flush=True,
        )


if __name__ == "__main__":
    _smoke()
