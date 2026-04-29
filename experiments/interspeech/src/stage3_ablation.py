"""Stage 3 ablation for SGPA boundary refinement.

This script compares raw CTC word-level cut points against the same cut points
after SGPA Stage 3 spectral refinement. It intentionally does not run LFM2 or
Shapley estimation; the goal is to isolate whether Stage 3 moves boundaries to
acoustically more stable regions.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import librosa
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from mllm_shap.connectors.base.audio import AudioSegment, SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler
from scipy import stats
from tqdm.auto import tqdm


DEFAULT_DATASET_NAME = "Pawlo77/mllm-swap"
DEFAULT_DATASET_CONFIG = "single_sentence_1k"
DEFAULT_DATASET_REVISION = "25b57f3a5ec82573e8a68ac7ecc0d9bd4418b66e"
DEFAULT_OUTPUT_DIR = Path("outputs/stage3_ablation")
EPS = 1e-9


@dataclass(frozen=True)
class SampleResult:
    """Per-utterance Stage 3 ablation result."""

    sample_id: int
    dataset_config: str
    transcript: str
    audio_column: str
    duration_sec: float
    n_words: int
    n_raw_boundaries: int
    n_refined_boundaries: int
    raw_mean_flux: float
    refined_mean_flux: float
    raw_median_flux: float
    refined_median_flux: float
    percent_reduction: float
    refined_boundary_rate: float
    runtime_sec: float


@dataclass(frozen=True)
class BoundaryResult:
    """One raw/refined boundary-pair measurement."""

    sample_id: int
    dataset_config: str
    boundary_idx: int
    transcript: str
    audio_column: str
    raw_time_sec: float
    refined_time_sec: float
    raw_flux: float
    refined_flux: float
    delta_time_ms: float


@dataclass(frozen=True)
class FailureResult:
    """A sample that could not be aligned or measured."""

    sample_id: int
    dataset_config: str
    transcript: str
    audio_column: str
    error_type: str
    error_message: str


def _to_mono_float(waveform: torch.Tensor) -> np.ndarray:
    """Convert a waveform tensor to mono float32 in [-1, 1]."""
    arr = waveform.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    arr = arr.astype(np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / (peak + EPS)
    return arr


def _spectral_flux_at_times(
    waveform: torch.Tensor,
    sample_rate: int,
    times_sec: list[float],
    n_fft: int = 256,
    hop_length: int = 64,
) -> list[float]:
    """Measure spectral flux at the nearest STFT frame for each timestamp."""
    x = _to_mono_float(waveform)
    if x.size < n_fft or not times_sec:
        return [0.0 for _ in times_sec]

    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    if stft.shape[1] < 2:
        return [0.0 for _ in times_sec]

    flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
    flux = np.pad(flux, (1, 0), mode="edge")
    frame_times = librosa.frames_to_time(
        np.arange(len(flux)), sr=sample_rate, hop_length=hop_length
    )

    values: list[float] = []
    for t in times_sec:
        idx = int(np.argmin(np.abs(frame_times - float(t))))
        values.append(float(flux[idx]))
    return values


def _segments_to_internal_boundaries(
    segments: list[AudioSegment],
    duration_sec: float,
) -> list[float]:
    """
    Convert word-level segments to masking boundary timestamps.

    We include starts and ends of all word segments except utterance edges,
    because either edge can become a silence transition when the corresponding
    player is removed from a coalition.
    """
    boundaries: list[float] = []
    for seg in segments:
        boundaries.extend([float(seg.start_time), float(seg.end_time)])

    filtered = [
        b for b in boundaries if 0.0 < b < duration_sec and np.isfinite(float(b))
    ]
    return sorted(set(round(b, 6) for b in filtered))


def _aggregate_raw_ctc_segments(
    aligner: SpectrogramGuidedAligner,
    token_spans: list[tuple[int, int, int]],
    emissions_gpu: torch.Tensor,
    waveform: torch.Tensor,
    original_sr: int,
    target_segments: list[str],
) -> list[AudioSegment]:
    """Build word segments from raw CTC spans, skipping Stage 3 refinement."""
    ratio = waveform.size(1) / emissions_gpu.size(0)
    raw_chars: list[dict[str, str | float | bool]] = []

    for sp_token, sp_start, sp_end in token_spans:
        char = cast(str, aligner.tokenizer.convert_ids_to_tokens(sp_token))
        if sp_end > sp_start:
            confidence = float(
                torch.exp(emissions_gpu[sp_start:sp_end, sp_token]).mean()
            )
        else:
            confidence = 0.0
        raw_chars.append(
            {
                "char": char,
                "start": float((sp_start * ratio) / original_sr),
                "end": float((sp_end * ratio) / original_sr),
                "confidence": confidence,
                "boundary_refined": False,
            }
        )

    aggregate = getattr(
        aligner, "_SpectrogramGuidedAligner__aggregate_chars_to_segments"
    )
    set_indices = getattr(aligner, "_SpectrogramGuidedAligner__set_segment_indices")
    segments = aggregate(raw_chars, target_segments)
    set_indices(segments, waveform, original_sr)
    return segments


def _raw_and_refined_segments(
    aligner: SpectrogramGuidedAligner,
    transcript: str,
    waveform: torch.Tensor,
    sample_rate: int,
) -> tuple[list[AudioSegment], list[AudioSegment]]:
    """Return raw CTC word segments and Stage-3-refined word segments."""
    prepare = getattr(aligner, "_SpectrogramGuidedAligner__prepare_transcript")
    perform_alignment = getattr(
        aligner, "_SpectrogramGuidedAligner__perform_forced_alignment"
    )
    merge_tokens = getattr(aligner, "_SpectrogramGuidedAligner__merge_tokens")
    refine_spans = getattr(aligner, "_SpectrogramGuidedAligner__refine_token_spans")
    aggregate = getattr(
        aligner, "_SpectrogramGuidedAligner__aggregate_chars_to_segments"
    )
    set_indices = getattr(aligner, "_SpectrogramGuidedAligner__set_segment_indices")

    _full, target_segments, _clean_text, valid_tokens = prepare(transcript)
    alignment_path, emissions_gpu = perform_alignment(
        waveform, sample_rate, valid_tokens
    )
    token_spans = merge_tokens(alignment_path, aligner.blank_id)

    raw_segments = _aggregate_raw_ctc_segments(
        aligner=aligner,
        token_spans=token_spans,
        emissions_gpu=emissions_gpu,
        waveform=waveform,
        original_sr=sample_rate,
        target_segments=target_segments,
    )

    refined_chars = refine_spans(token_spans, emissions_gpu, waveform, sample_rate)
    refined_segments = aggregate(refined_chars, target_segments)
    set_indices(refined_segments, waveform, sample_rate)

    return raw_segments, refined_segments


def _load_voicebench_like_rows(
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    audio_column: str,
    max_samples: int,
) -> pd.DataFrame:
    """Load and deterministically select single-sentence HF rows."""

    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return value
        return []

    ds = load_dataset(
        dataset_name,
        dataset_config,
        revision=dataset_revision,  # nosec B615 - pinned by CLI default
    )["test"]
    df = ds.to_pandas()

    required_cols = {"sentences", audio_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    df = df[df["sentences"].apply(lambda x: len(_as_list(x)) == 1).astype(bool)].copy()
    df = df[df[audio_column].apply(lambda x: len(_as_list(x)) > 0).astype(bool)].copy()
    df = df.copy()
    df["transcript"] = df["sentences"].apply(lambda x: str(_as_list(x)[0]))
    df["audio_bytes"] = df[audio_column].apply(lambda x: _as_list(x)[0])
    df = df.drop_duplicates(subset=["transcript"]).reset_index(drop=True)
    df["sent_length"] = df["transcript"].str.len()
    df = df.sort_values(["sent_length", "transcript"]).head(max_samples)
    return df[["transcript", "audio_bytes"]].reset_index(drop=True)


def _slugify_dataset_config(dataset_config: str) -> str:
    """Filesystem-safe slug for dataset config names."""
    return (
        str(dataset_config)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def run_stage3_ablation(
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    audio_column: str,
    max_samples: int,
    output_dir: Path,
    device: str | None,
) -> dict[str, Any]:
    """Run the Stage 3 ablation and write CSV/JSON outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    rows = _load_voicebench_like_rows(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        audio_column=audio_column,
        max_samples=max_samples,
    )

    aligner = SpectrogramGuidedAligner(device=torch_device)
    sample_results: list[SampleResult] = []
    boundary_results: list[BoundaryResult] = []
    failures: list[FailureResult] = []

    for sample_id, row in tqdm(
        rows.iterrows(), total=len(rows), desc=f"stage3 ablation ({audio_column})"
    ):
        transcript = str(row["transcript"])
        t0 = time.perf_counter()
        try:
            waveform, sample_rate = TorchAudioHandler.from_bytes(
                row["audio_bytes"], audio_format="wav"
            )
            duration_sec = float(waveform.size(-1) / sample_rate)
            raw_segments, refined_segments = _raw_and_refined_segments(
                aligner=aligner,
                transcript=transcript,
                waveform=waveform,
                sample_rate=int(sample_rate),
            )

            raw_boundaries = _segments_to_internal_boundaries(
                raw_segments, duration_sec=duration_sec
            )
            refined_boundaries = _segments_to_internal_boundaries(
                refined_segments, duration_sec=duration_sec
            )
            pair_count = min(len(raw_boundaries), len(refined_boundaries))
            if pair_count == 0:
                raise ValueError("No comparable internal boundaries produced.")

            raw_boundaries = raw_boundaries[:pair_count]
            refined_boundaries = refined_boundaries[:pair_count]
            raw_flux = _spectral_flux_at_times(
                waveform, int(sample_rate), raw_boundaries
            )
            refined_flux = _spectral_flux_at_times(
                waveform, int(sample_rate), refined_boundaries
            )

            for boundary_idx, (raw_t, refined_t, raw_f, refined_f) in enumerate(
                zip(raw_boundaries, refined_boundaries, raw_flux, refined_flux)
            ):
                boundary_results.append(
                    BoundaryResult(
                        sample_id=int(sample_id),
                        dataset_config=dataset_config,
                        boundary_idx=int(boundary_idx),
                        transcript=transcript,
                        audio_column=audio_column,
                        raw_time_sec=float(raw_t),
                        refined_time_sec=float(refined_t),
                        raw_flux=float(raw_f),
                        refined_flux=float(refined_f),
                        delta_time_ms=float((refined_t - raw_t) * 1000.0),
                    )
                )

            raw_mean = float(np.mean(raw_flux))
            refined_mean = float(np.mean(refined_flux))
            percent_reduction = (
                100.0 * (raw_mean - refined_mean) / raw_mean if raw_mean > 0 else 0.0
            )
            refined_flags = [
                bool(getattr(seg, "boundary_refined", False))
                for seg in refined_segments
            ]
            sample_results.append(
                SampleResult(
                    sample_id=int(sample_id),
                    dataset_config=dataset_config,
                    transcript=transcript,
                    audio_column=audio_column,
                    duration_sec=duration_sec,
                    n_words=len(refined_segments),
                    n_raw_boundaries=len(raw_boundaries),
                    n_refined_boundaries=len(refined_boundaries),
                    raw_mean_flux=raw_mean,
                    refined_mean_flux=refined_mean,
                    raw_median_flux=float(np.median(raw_flux)),
                    refined_median_flux=float(np.median(refined_flux)),
                    percent_reduction=percent_reduction,
                    refined_boundary_rate=float(np.mean(refined_flags))
                    if refined_flags
                    else 0.0,
                    runtime_sec=float(time.perf_counter() - t0),
                )
            )
        except Exception as exc:  # noqa: BLE001 - record failures and continue.
            failures.append(
                FailureResult(
                    sample_id=int(sample_id),
                    dataset_config=dataset_config,
                    transcript=transcript,
                    audio_column=audio_column,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    samples_df = pd.DataFrame([asdict(r) for r in sample_results])
    boundaries_df = pd.DataFrame([asdict(r) for r in boundary_results])
    failures_df = pd.DataFrame([asdict(r) for r in failures])

    cfg_slug = _slugify_dataset_config(dataset_config)
    stem = f"{cfg_slug}__{audio_column}_n{max_samples}"
    samples_path = output_dir / f"{stem}_samples.csv"
    boundaries_path = output_dir / f"{stem}_boundaries.csv"
    failures_path = output_dir / f"{stem}_failures.csv"
    summary_path = output_dir / f"{stem}_summary.json"

    samples_df.to_csv(samples_path, index=False)
    boundaries_df.to_csv(boundaries_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    if samples_df.empty:
        summary: dict[str, Any] = {
            "audio_column": audio_column,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "requested_samples": max_samples,
            "completed_samples": 0,
            "failed_samples": len(failures),
        }
    else:
        raw = samples_df["raw_mean_flux"].to_numpy(dtype=float)
        refined = samples_df["refined_mean_flux"].to_numpy(dtype=float)
        diff = raw - refined
        t_stat, p_value = stats.ttest_rel(raw, refined)
        cohen_dz = float(np.mean(diff) / (np.std(diff, ddof=1) + EPS))
        summary = {
            "audio_column": audio_column,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "requested_samples": max_samples,
            "completed_samples": int(len(samples_df)),
            "failed_samples": int(len(failures_df)),
            "mean_raw_flux": float(np.mean(raw)),
            "std_raw_flux": float(np.std(raw, ddof=1)),
            "mean_refined_flux": float(np.mean(refined)),
            "std_refined_flux": float(np.std(refined, ddof=1)),
            "mean_percent_reduction": float(samples_df["percent_reduction"].mean()),
            "median_percent_reduction": float(samples_df["percent_reduction"].median()),
            "paired_t_stat": float(t_stat),
            "paired_p_value": float(p_value),
            "cohen_dz": cohen_dz,
            "mean_refined_boundary_rate": float(
                samples_df["refined_boundary_rate"].mean()
            ),
            "mean_runtime_sec": float(samples_df["runtime_sec"].mean()),
            "samples_csv": str(samples_path),
            "boundaries_csv": str(boundaries_path),
            "failures_csv": str(failures_path),
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--audio-column", default="audio__male")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    """Run CLI."""
    args = build_argparser().parse_args()
    summary = run_stage3_ablation(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        audio_column=args.audio_column,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
