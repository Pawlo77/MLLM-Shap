"""Voxtral-Mini-3B (Mistral family) audio->text backend for 2nd-model faithfulness.

Voxtral is integrated *natively* in transformers (``VoxtralForConditionalGeneration``,
``VoxtralProcessor``) -- no ``trust_remote_code`` and therefore no version conflict
with transformers 5.4 (unlike Phi-4-multimodal, whose remote code targets ~4.48).
It is a different model family than Qwen2-Audio, giving the paper a cross-family
generalization result for SGPA.

We drive it in *transcription* mode via ``apply_transcription_request``, which
accepts in-memory ``np.ndarray`` waveforms directly -- ideal for the many perturbed
coalitions produced during exact-Shapley faithfulness. This mirrors the Qwen
"repeat the exact words" transcription setup, so deletion-based faithfulness is
measured on comparable behaviour across both models.
"""

from __future__ import annotations

import numpy as np
import torch


class VoxtralAudioBackend:
    """Deterministic audio->text transcription with Voxtral-Mini-3B (native transformers)."""

    def __init__(
        self,
        repo: str = "mistralai/Voxtral-Mini-3B-2507",
        device: str = "cuda",
        load_in_4bit: bool = True,
        language: str = "en",
        torch_dtype: "torch.dtype | None" = None,
    ) -> None:
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self.device = device
        self.repo = repo
        self.language = language
        self.processor = AutoProcessor.from_pretrained(repo)

        model_kwargs: dict = {}
        if load_in_4bit and "cuda" in str(device):
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = device
        else:
            # bf16 on GPU; float32 on CPU (bf16 CPU matmul is slow/patchy).
            if torch_dtype is None:
                torch_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = VoxtralForConditionalGeneration.from_pretrained(
            repo, **model_kwargs
        )
        if "device_map" not in model_kwargs:
            self.model = self.model.to(device)
        self.model.eval()

        try:
            self.target_sr = int(self.processor.feature_extractor.sampling_rate)
        except Exception:  # noqa: BLE001
            self.target_sr = 16000

    def _build_inputs(self, audio: np.ndarray):
        # apply_transcription_request accepts raw arrays; `format` must be a list
        # matching the number of audios, and it serialises each array to a WAV
        # buffer via soundfile internally before mel-feature extraction.
        inputs = self.processor.apply_transcription_request(
            audio=audio,
            model_id=self.repo,
            language=self.language,
            sampling_rate=self.target_sr,
            format=["wav"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if "input_features" not in inputs:
            raise RuntimeError(
                "Audio features missing from Voxtral processor output; audio not encoded."
            )
        return inputs

    @torch.no_grad()
    def generate_text(
        self,
        waveform_16k: np.ndarray,
        instruction: str = "Repeat the exact words that the speaker said.",
        max_new_tokens: int = 48,
    ) -> str:
        """Decode a deterministic transcription for a mono 16 kHz waveform.

        ``instruction`` is accepted for protocol compatibility but unused: Voxtral
        runs in transcription mode, which is the faithfulness-relevant behaviour.
        """
        audio = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        inputs = self._build_inputs(audio)
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

        ``instruction`` is unused (Voxtral runs in transcription mode); kept for
        protocol compatibility with the Qwen backend.
        """
        audio = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        inputs = self._build_inputs(audio)
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        prompt_ids = inputs["input_ids"]
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tgt = tok(
                target_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self.device)
        else:
            tgt = self.processor(
                text=target_text, return_tensors="pt", add_special_tokens=False
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
    backend = VoxtralAudioBackend()
    print("target_sr:", backend.target_sr, flush=True)
    sr = backend.target_sr
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    out = backend.generate_text(tone, max_new_tokens=16)
    print("SMOKE OUTPUT:", repr(out), flush=True)
    if torch.cuda.is_available():
        print(
            f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB "
            f"reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB",
            flush=True,
        )


if __name__ == "__main__":
    _smoke()
