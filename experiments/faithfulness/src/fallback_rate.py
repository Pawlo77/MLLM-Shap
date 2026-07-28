"""Boundary-refinement fallback-rate table: TTS vs natural speech.

Stage-3 of SGPA searches for a spectrally quiet cut inside each inter-character
gap; if none is found it falls back to the gap midpoint and marks the segment
``boundary_refined=False``. This script quantifies how often that happens on
synthetic (TTS) vs natural speech, closing the paper's stated open item that
"fallback rates on continuous natural speech still need fuller quantification."

Alignment-only: uses the SGPA aligner on CPU, no model generation, no GPU.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler

from experiments.mllm_shapx.src.data import extract_texts_from_row

from .helpers import as_list
from .io import experiment_set_from_spec, load_selected_rows, load_spec

# (label, spec_path, audio_column_override or None)
DEFAULT_CONDITIONS: list[tuple[str, str, str | None]] = [
    (
        "Male TTS",
        "experiments/faithfulness/configs/hp1_faithfulness_audio_male_spec.json",
        None,
    ),
    (
        "Female TTS",
        "experiments/faithfulness/configs/hp1_faithfulness_audio_female_spec.json",
        None,
    ),
]


def _resolve_natural_spec() -> tuple[str, str, str | None] | None:
    """Locate the LibriSpeech-natural SV run spec (audio__original) if present."""
    root = Path("experiments/experiments_output/aaai27_fixed_500_original")
    for run_dir in sorted(root.glob("*/")):
        if (run_dir / "spec.json").exists():
            return ("LibriSpeech natural", str(run_dir / "spec.json"), None)
    return None


def run_condition(
    label: str,
    spec_path: str,
    audio_column_override: str | None,
    aligner: SpectrogramGuidedAligner,
    max_samples: int,
) -> dict[str, Any]:
    """Align every sample in a condition and tally boundary-refinement fallbacks."""
    spec = load_spec(Path(spec_path).parent, spec_path=Path(spec_path))
    cfg = experiment_set_from_spec(spec)
    audio_column = audio_column_override or cfg.modality.input_modality
    rows = load_selected_rows(cfg, max_samples=max_samples)

    n_samples = 0
    n_samples_with_fallback = 0
    total_segments = 0
    total_fallbacks = 0
    per_sample_fallback_frac: list[float] = []
    errors = 0

    for _, row in rows.items():
        try:
            transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
            audio_bytes = as_list(row[audio_column])[0]
            waveform, sr = TorchAudioHandler.from_bytes(audio_bytes, audio_format="wav")
            segments = aligner(
                transcript=transcript,
                waveform=waveform,
                original_sr=int(sr),
                audio_format="wav",
                attach_audio=False,
            )
            if not segments:
                continue
            flags = [bool(getattr(s, "boundary_refined", True)) for s in segments]
            n_fb = sum(1 for f in flags if not f)
            n_samples += 1
            total_segments += len(flags)
            total_fallbacks += n_fb
            per_sample_fallback_frac.append(n_fb / len(flags))
            if n_fb > 0:
                n_samples_with_fallback += 1
        except Exception:  # noqa: BLE001
            errors += 1
            continue

    return {
        "condition": label,
        "audio_column": str(audio_column),
        "n_samples": n_samples,
        "n_segments": total_segments,
        "segment_fallback_rate": (
            total_fallbacks / total_segments if total_segments else None
        ),
        "sample_fallback_rate": (
            n_samples_with_fallback / n_samples if n_samples else None
        ),
        "mean_per_sample_fallback_frac": (
            float(np.mean(per_sample_fallback_frac))
            if per_sample_fallback_frac
            else None
        ),
        "errors": errors,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--aligner-device", default="cpu")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/faithfulness/outputs/fallback_rate.json"),
    )
    args = p.parse_args()

    conditions = list(DEFAULT_CONDITIONS)
    natural = _resolve_natural_spec()
    if natural is not None:
        conditions.append(natural)

    aligner = SpectrogramGuidedAligner(device=torch.device(args.aligner_device))
    results = []
    for label, spec_path, override in conditions:
        print(f"[fallback] {label} ...", flush=True)
        res = run_condition(label, spec_path, override, aligner, args.max_samples)
        print(json.dumps(res, indent=2), flush=True)
        results.append(res)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"conditions": results}, indent=2))
    # Compact table
    print("\n=== Fallback-rate table ===", flush=True)
    print(
        f"{'Condition':22s} {'N':>5s} {'segs':>6s} "
        f"{'seg fallback%':>13s} {'samp fallback%':>14s}",
        flush=True,
    )
    for r in results:
        sfr = r["segment_fallback_rate"]
        pfr = r["sample_fallback_rate"]
        print(
            f"{r['condition']:22s} {r['n_samples']:5d} {r['n_segments']:6d} "
            f"{(100 * sfr if sfr is not None else float('nan')):12.2f}% "
            f"{(100 * pfr if pfr is not None else float('nan')):13.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
