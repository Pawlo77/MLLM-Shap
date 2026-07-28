"""Minimal Qwen2-Audio-7B-Instruct audio->text backend (4-bit) for 2nd-model faithfulness.

Qwen2-Audio's processor is audio+text only (no vision), sidestepping the
image-processor auto-load bug that blocks Qwen2.5-Omni on transformers 5.4.
4-bit NF4 quantization keeps the 7B model well within 12 GB VRAM. Exposes a
single deterministic ``generate_text`` call for the standalone exact-Shapley
faithfulness runner (no full ``mllm_shap`` connector needed).
"""

from __future__ import annotations

import numpy as np
import torch


class QwenAudioBackend:
    """Deterministic audio->text decoding with Qwen2-Audio-7B-Instruct in 4-bit."""

    def __init__(
        self,
        repo: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda",
    ) -> None:
        from transformers import (
            BitsAndBytesConfig,
            Qwen2AudioForConditionalGeneration,
            Qwen2AudioProcessor,
        )

        self.device = device
        self.processor = Qwen2AudioProcessor.from_pretrained(repo)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            repo, quantization_config=bnb, device_map=device
        )
        self.model.eval()
        try:
            self.target_sr = int(self.processor.feature_extractor.sampling_rate)
        except Exception:  # noqa: BLE001
            self.target_sr = 16000

    def _build_inputs(self, audio: np.ndarray, instruction: str):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "inline"},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        # transformers 5.4 Qwen2AudioProcessor uses `audio=` (singular); `audios=`
        # is silently ignored, which drops the audio entirely.
        inputs = self.processor(
            text=text,
            audio=[audio],
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        if "input_features" not in inputs:
            raise RuntimeError(
                "Audio features missing from processor output; audio not encoded."
            )
        return inputs

    @torch.no_grad()
    def generate_text(
        self,
        waveform_16k: np.ndarray,
        instruction: str = "Repeat the exact words that the speaker said.",
        max_new_tokens: int = 48,
    ) -> str:
        """Decode a deterministic text response for a mono 16 kHz waveform."""
        audio = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        inputs = self._build_inputs(audio, instruction)
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        gen = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        if isinstance(gen, (tuple, list)):
            gen = gen[0]
        in_len = int(inputs["input_ids"].shape[1])
        new_tokens = gen[:, in_len:]
        return self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    @torch.no_grad()
    def score_text_logprob(
        self,
        waveform_16k: np.ndarray,
        target_text: str,
        instruction: str = "Repeat the exact words that the speaker said.",
    ) -> float:
        """Mean per-token log-probability of ``target_text`` teacher-forced under the audio.

        The target token sequence is fixed across coalitions, so the *drop* in this
        score when a word is silenced is a model-internal faithfulness endpoint that
        shares nothing with the E5/TF-IDF response-similarity utilities.
        """
        audio = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        inputs = self._build_inputs(audio, instruction)
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        prompt_ids = inputs["input_ids"]
        tgt = self.processor.tokenizer(
            target_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        if tgt.shape[1] == 0:
            return float("nan")
        full_ids = torch.cat([prompt_ids, tgt], dim=1)
        attn = torch.ones_like(full_ids)
        fwd = {
            k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")
        }
        out = self.model(input_ids=full_ids, attention_mask=attn, **fwd)
        p, t = int(prompt_ids.shape[1]), int(tgt.shape[1])
        pred = out.logits[:, p - 1 : p + t - 1, :].float()
        logp = torch.log_softmax(pred, dim=-1)
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return float(tok_lp.mean().item())


def _smoke() -> None:
    """Load the model and decode a synthetic tone, reporting VRAM, to validate the API."""
    backend = QwenAudioBackend()
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
