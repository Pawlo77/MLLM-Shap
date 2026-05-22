"""Stage 3 ablation: raw CTC boundaries vs SGPA spectral refinement."""

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

DEFAULT_DATASET_NAME: str = "Pawlo77/mllm-swap"
DEFAULT_DATASET_CONFIG: str = "single_sentence__voice_bench"
DEFAULT_DATASET_REVISION: str = "25b57f3a5ec82573e8a68ac7ecc0d9bd4418b66e"
DEFAULT_OUTPUT_DIR: Path = Path("outputs/stage3_ablation")
EPS: float = 1e-9


@dataclass(frozen=True)
class SampleResult:
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
    sample_id: int
    dataset_config: str
    transcript: str
    audio_column: str
    error_type: str
    error_message: str


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _to_mono_float(waveform: torch.Tensor) -> np.ndarray:
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
    x = _to_mono_float(waveform)
    if x.size < n_fft or not times_sec:
        return [0.0] * len(times_sec)

    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    if stft.shape[1] < 2:
        return [0.0] * len(times_sec)

    flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
    flux = np.pad(flux, (1, 0), mode="edge")
    frame_times = librosa.frames_to_time(
        np.arange(len(flux)), sr=sample_rate, hop_length=hop_length
    )

    return [
        float(flux[int(np.argmin(np.abs(frame_times - float(t))))]) for t in times_sec
    ]


def _segments_to_internal_boundaries(
    segments: list[AudioSegment], duration_sec: float
) -> list[float]:
    boundaries: list[float] = []
    for seg in segments:
        boundaries.extend([float(seg.start_time), float(seg.end_time)])
    filtered = [b for b in boundaries if 0.0 < b < duration_sec and np.isfinite(b)]
    return sorted(set(round(b, 6) for b in filtered))


def _aggregate_raw_ctc_segments(
    aligner: SpectrogramGuidedAligner,
    token_spans: list[tuple[int, int, int]],
    emissions_gpu: torch.Tensor,
    waveform: torch.Tensor,
    original_sr: int,
    target_segments: list[str],
) -> list[AudioSegment]:
    ratio = waveform.size(1) / emissions_gpu.size(0)
    raw_chars: list[dict[str, str | float | bool]] = []
    for sp_token, sp_start, sp_end in token_spans:
        char = cast(str, aligner.tokenizer.convert_ids_to_tokens(sp_token))
        confidence = (
            float(torch.exp(emissions_gpu[sp_start:sp_end, sp_token]).mean())
            if sp_end > sp_start
            else 0.0
        )
        raw_chars.append({
            "char": char,
            "start": float((sp_start * ratio) / original_sr),
            "end": float((sp_end * ratio) / original_sr),
            "confidence": confidence,
            "boundary_refined": False,
        })

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
        aligner,
        token_spans,
        emissions_gpu,
        waveform,
        sample_rate,
        target_segments,
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
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return value
        return []

    ds = load_dataset(dataset_name, dataset_config, revision=dataset_revision)["test"]
    df = ds.to_pandas()

    required_cols = {"sentences", audio_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[df["sentences"].apply(lambda x: len(_as_list(x)) == 1).astype(bool)].copy()
    df = df[df[audio_column].apply(lambda x: len(_as_list(x)) > 0).astype(bool)].copy()
    df["transcript"] = df["sentences"].apply(lambda x: str(_as_list(x)[0]))
    df["audio_bytes"] = df[audio_column].apply(lambda x: _as_list(x)[0])
    df = df.drop_duplicates(subset=["transcript"]).reset_index(drop=True)
    df["sent_length"] = df["transcript"].str.len()
    df = df.sort_values(["sent_length", "transcript"]).head(max_samples)
    return df[["transcript", "audio_bytes"]].reset_index(drop=True)


# ─── Main runner ─────────────────────────────────────────────────────────────


def _run_single_column(
    aligner: SpectrogramGuidedAligner,
    rows: pd.DataFrame,
    dataset_config: str,
    audio_column: str,
) -> tuple[list[SampleResult], list[BoundaryResult], list[FailureResult]]:
    """Run ablation for a single audio column. Returns raw result lists."""
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
                aligner, transcript, waveform, int(sample_rate)
            )

            raw_boundaries = _segments_to_internal_boundaries(
                raw_segments, duration_sec
            )
            refined_boundaries = _segments_to_internal_boundaries(
                refined_segments, duration_sec
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

            for bi, (raw_t, ref_t, raw_f, ref_f) in enumerate(
                zip(raw_boundaries, refined_boundaries, raw_flux, refined_flux)
            ):
                boundary_results.append(
                    BoundaryResult(
                        sample_id=int(sample_id),
                        dataset_config=dataset_config,
                        boundary_idx=bi,
                        transcript=transcript,
                        audio_column=audio_column,
                        raw_time_sec=float(raw_t),
                        refined_time_sec=float(ref_t),
                        raw_flux=float(raw_f),
                        refined_flux=float(ref_f),
                        delta_time_ms=float((ref_t - raw_t) * 1000.0),
                    )
                )

            raw_mean = float(np.mean(raw_flux))
            refined_mean = float(np.mean(refined_flux))
            pct_red = (
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
                    percent_reduction=pct_red,
                    refined_boundary_rate=float(np.mean(refined_flags))
                    if refined_flags
                    else 0.0,
                    runtime_sec=float(time.perf_counter() - t0),
                )
            )
        except Exception as exc:  # noqa: BLE001
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

    return sample_results, boundary_results, failures


def _compute_column_summary(
    samples_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    audio_column: str,
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    max_samples: int,
) -> dict[str, Any]:
    """Compute summary statistics for a single audio column."""
    col_samples = (
        samples_df[samples_df["audio_column"] == audio_column]
        if not samples_df.empty
        else samples_df
    )
    col_failures = (
        failures_df[failures_df["audio_column"] == audio_column]
        if not failures_df.empty
        else failures_df
    )

    if col_samples.empty:
        return {
            "audio_column": audio_column,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "requested_samples": max_samples,
            "completed_samples": 0,
            "failed_samples": int(len(col_failures)),
        }

    raw = col_samples["raw_mean_flux"].to_numpy(dtype=float)
    refined = col_samples["refined_mean_flux"].to_numpy(dtype=float)
    diff = raw - refined
    t_stat, p_value = stats.ttest_rel(raw, refined)
    return {
        "audio_column": audio_column,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "requested_samples": max_samples,
        "completed_samples": int(len(col_samples)),
        "failed_samples": int(len(col_failures)),
        "mean_raw_flux": float(np.mean(raw)),
        "std_raw_flux": float(np.std(raw, ddof=1)),
        "mean_refined_flux": float(np.mean(refined)),
        "std_refined_flux": float(np.std(refined, ddof=1)),
        "mean_percent_reduction": float(col_samples["percent_reduction"].mean()),
        "median_percent_reduction": float(col_samples["percent_reduction"].median()),
        "paired_t_stat": float(t_stat),
        "paired_p_value": float(p_value),
        "cohen_dz": float(np.mean(diff) / (np.std(diff, ddof=1) + EPS)),
        "mean_refined_boundary_rate": float(
            col_samples["refined_boundary_rate"].mean()
        ),
        "mean_runtime_sec": float(col_samples["runtime_sec"].mean()),
    }


def run_stage3_ablation(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    audio_columns: list[str] | None = None,
    max_samples: int = 1000,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str | None = None,
) -> dict[str, Any]:
    """Run stage 3 ablation across all audio columns.

    Saves 3 consolidated files:
      - samples.csv   (per-sample metrics for all columns)
      - boundaries.csv (per-boundary detail for all columns)
      - summary.json  (per-column + combined statistics)

    Failures are included in summary.json rather than a separate file.
    """
    if audio_columns is None:
        audio_columns = ["audio__male", "audio__female"]

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

    aligner = SpectrogramGuidedAligner(device=torch_device)
    all_sample_results: list[SampleResult] = []
    all_boundary_results: list[BoundaryResult] = []
    all_failures: list[FailureResult] = []

    for audio_column in audio_columns:
        rows = _load_voicebench_like_rows(
            dataset_name, dataset_config, dataset_revision, audio_column, max_samples
        )
        sr, br, fr = _run_single_column(aligner, rows, dataset_config, audio_column)
        all_sample_results.extend(sr)
        all_boundary_results.extend(br)
        all_failures.extend(fr)

    samples_df = pd.DataFrame([asdict(r) for r in all_sample_results])
    boundaries_df = pd.DataFrame([asdict(r) for r in all_boundary_results])
    failures_df = pd.DataFrame([asdict(r) for r in all_failures])

    # Save consolidated files
    samples_df.to_csv(output_dir / "samples.csv", index=False)
    boundaries_df.to_csv(output_dir / "boundaries.csv", index=False)

    # Build combined summary
    per_column = {
        col: _compute_column_summary(
            samples_df,
            failures_df,
            col,
            dataset_name,
            dataset_config,
            dataset_revision,
            max_samples,
        )
        for col in audio_columns
    }

    combined_summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "max_samples": max_samples,
        "audio_columns": audio_columns,
        "total_completed": int(len(samples_df)),
        "total_failed": int(len(failures_df)),
        "per_column": per_column,
        "failures": [asdict(f) for f in all_failures],
    }

    if not samples_df.empty:
        raw_all = samples_df["raw_mean_flux"].to_numpy(dtype=float)
        ref_all = samples_df["refined_mean_flux"].to_numpy(dtype=float)
        diff_all = raw_all - ref_all
        t_stat, p_value = stats.ttest_rel(raw_all, ref_all)
        combined_summary["combined"] = {
            "mean_raw_flux": float(np.mean(raw_all)),
            "mean_refined_flux": float(np.mean(ref_all)),
            "mean_percent_reduction": float(samples_df["percent_reduction"].mean()),
            "paired_t_stat": float(t_stat),
            "paired_p_value": float(p_value),
            "cohen_dz": float(np.mean(diff_all) / (np.std(diff_all, ddof=1) + EPS)),
        }

    (output_dir / "summary.json").write_text(json.dumps(combined_summary, indent=2))
    return combined_summary
