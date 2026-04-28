"""Deletion-based faithfulness validation for SGPA Shapley values.

The experiment consumes existing mllm_shapx SGPA runs, identifies the highest
absolute-SV word segment per sample, and tests whether deleting that segment
changes the model response more than deleting a random equal-duration segment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors.base.audio import AudioSegment, SpectrogramGuidedAligner
from mllm_shap.connectors.config import ModelConfig
from mllm_shap.shap.similarity import TfIdfCosineSimilarity
from mllm_shap.utils.audio import TorchAudioHandler
from scipy import stats
from tqdm.auto import tqdm

from experiments.mllm_shapx.config import ExperimentSet, parse_experiment_set
from experiments.mllm_shapx.constants import InputModality, OutputModality
from experiments.mllm_shapx.data import (
    choose_prompt_text_column,
    extract_texts_from_row,
    iter_rows_for_selection,
    load_df,
)
from experiments.mllm_shapx.factory import _build_model, build_chat


DEFAULT_MALE_RUN_DIR = Path(
    "experiments_output/single_sentence_2026_01_03/"
    "audio_male_audio_limited_neyman_lin3_0"
)
DEFAULT_FEMALE_RUN_DIR = Path(
    "experiments_output/single_sentence_2026_01_03/"
    "audio_female_audio_limited_neyman_lin3_0"
)
DEFAULT_OUTPUT_DIR = Path("experiments/interspeech/outputs/faithfulness_deletion")
EPS = 1e-9


@dataclass(frozen=True)
class FaithfulnessResult:
    """Per-sample deletion faithfulness result."""

    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    n_segments: int
    top_segment_idx: int
    random_segment_idx: int
    top_segment_token: str
    random_segment_token: str
    top_abs_sv: float
    top_sv: float
    original_similarity: float
    top_similarity: float
    random_similarity: float
    top_drop: float
    random_drop: float
    drop_difference: float
    top_start_sec: float
    top_end_sec: float
    random_start_sec: float
    random_end_sec: float
    mask_duration_sec: float
    runtime_sec: float


@dataclass(frozen=True)
class FailureResult:
    """Failed sample metadata."""

    sample_id: int
    row_index: int
    audio_column: str
    transcript: str
    error_type: str
    error_message: str


def _as_list(value: Any) -> list[Any]:
    """Normalize parquet/datasets list-like cells."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _load_spec(run_dir: Path, spec_path: Path | None = None) -> dict[str, Any]:
    spec_path = spec_path or (run_dir / "spec.json")
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Missing mllm_shapx spec: {spec_path}. "
            "Pass --spec-path for committed HP-1 specs when experiments_output is incomplete."
        )
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _experiment_set_from_spec(spec: dict[str, Any]) -> ExperimentSet:
    """Rehydrate the subset of ExperimentSet needed for dataset/model setup."""
    raw = {
        "experiment_set_id": spec["experiment_set_id"],
        "output_root": "experiments_output",
        "device": spec.get("device"),
        "connector": spec["connector"],
        "dataset": spec["dataset"],
        "selection": spec["selection"],
        "generation": spec["generation"],
        "modality": spec["modality"],
        "shap": spec["shap"],
        "embedding": spec.get("embedding") or {},
        "experiments": [],
        "wandb": {"enabled": False},
    }
    return parse_experiment_set(raw)


def _load_selected_rows(
    cfg: ExperimentSet, max_samples: int | None
) -> dict[int, dict[str, Any]]:
    """Load rows using the same ordering policy as mllm_shapx."""
    df = load_df(
        cfg.dataset.repo_id,
        cfg.dataset.subset,
        cfg.dataset.split,
        cfg.dataset.revision,
        use_parquet=cfg.dataset.use_parquet,
        trust_remote_code=cfg.dataset.trust_remote_code,
    )
    text_col = choose_prompt_text_column(df)
    selected: dict[int, dict[str, Any]] = {}
    for row_idx, row in iter_rows_for_selection(
        df=df,
        start_index=cfg.selection.start_index,
        max_samples=max_samples,
        shuffle_seed=cfg.selection.shuffle_seed,
    ):
        row_dict = dict(row)
        row_dict["_text_col"] = text_col
        selected[int(row_idx)] = row_dict
    return selected


def _extract_audio_sv(sample_json: dict[str, Any]) -> list[float]:
    """Extract non-null audio SHAP values from a serialized mllm_shapx sample."""
    values: list[float] = []
    for turn in sample_json.get("conversation", []):
        for entry in turn:
            if entry.get("content_type") != 1:
                continue
            for value in entry.get("shap_values") or []:
                if value is None:
                    continue
                value_f = float(value)
                if math.isfinite(value_f):
                    values.append(value_f)
            if values:
                return values
    raise ValueError("No audio SHAP values found in sample JSON.")


def _sample_paths(run_dir: Path, max_samples: int | None) -> list[Path]:
    paths = sorted((run_dir / "samples").glob("sample_*_result.json"))
    if max_samples is not None:
        paths = paths[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No sample JSON files found in {run_dir / 'samples'}")
    return paths


def _parse_sample_id(sample_path: Path) -> int:
    return int(sample_path.name.split("_")[1])


def _mask_interval(
    waveform: torch.Tensor,
    start_sample: int,
    end_sample: int,
) -> torch.Tensor:
    """Silence one interval while preserving duration."""
    out = waveform.clone()
    if out.dim() == 1:
        out = out.unsqueeze(0)
    start_sample = max(0, min(int(start_sample), out.size(-1)))
    end_sample = max(start_sample, min(int(end_sample), out.size(-1)))
    out[:, start_sample:end_sample] = 0.0
    return out


def _equal_duration_interval(
    center_sample: int,
    duration_samples: int,
    total_samples: int,
) -> tuple[int, int]:
    """Build an equal-duration interval centered near a reference segment."""
    duration_samples = max(1, min(int(duration_samples), int(total_samples)))
    start = int(center_sample - duration_samples // 2)
    start = max(0, min(start, total_samples - duration_samples))
    return start, start + duration_samples


def _segment_interval(seg: AudioSegment) -> tuple[int, int]:
    if seg.start_sample is None or seg.end_sample is None:
        raise ValueError("Segment is missing sample indices.")
    return int(seg.start_sample), int(seg.end_sample)


def _response_similarity(
    base: Any, top: Any, random: Any
) -> tuple[float, float, float]:
    """Compute the same token-level TF-IDF cosine payoff used in the SGPA runs."""
    sims = TfIdfCosineSimilarity()(base=base, other=[base, top, random])
    return float(sims[0].item()), float(sims[1].item()), float(sims[2].item())


def _preflight_one_sample(
    *,
    sample_path: Path,
    row: dict[str, Any],
    aligner: SpectrogramGuidedAligner,
    audio_column: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Validate data, SV extraction, alignment, interval selection and audio encoding."""
    sample_id = _parse_sample_id(sample_path)
    sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
    row_index = int(sample_json.get("row_index", sample_id))
    transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
    audio_values = _as_list(row[audio_column])
    if not audio_values:
        raise ValueError(f"No audio bytes in column {audio_column}")
    audio_bytes = audio_values[0]

    sv_values = _extract_audio_sv(sample_json)
    top_segment_idx = int(np.argmax(np.abs(np.asarray(sv_values, dtype=float))))
    waveform, sample_rate = TorchAudioHandler.from_bytes(
        audio_bytes, audio_format="wav"
    )
    segments = aligner(
        transcript=transcript,
        waveform=waveform,
        original_sr=int(sample_rate),
        audio_format="wav",
        attach_audio=False,
    )
    if top_segment_idx >= len(segments):
        raise ValueError(
            f"Top-SV index {top_segment_idx} exceeds aligned segment count {len(segments)}."
        )

    candidate_indices = [i for i in range(len(segments)) if i != top_segment_idx]
    random_segment_idx = int(rng.choice(candidate_indices))
    top_start, top_end = _segment_interval(segments[top_segment_idx])
    random_start_raw, random_end_raw = _segment_interval(segments[random_segment_idx])
    random_start, random_end = _equal_duration_interval(
        center_sample=(random_start_raw + random_end_raw) // 2,
        duration_samples=max(1, top_end - top_start),
        total_samples=waveform.size(-1),
    )
    top_audio = TorchAudioHandler.to_bytes(
        _mask_interval(waveform, top_start, top_end),
        sample_rate=int(sample_rate),
        audio_format="wav",
    )
    random_audio = TorchAudioHandler.to_bytes(
        _mask_interval(waveform, random_start, random_end),
        sample_rate=int(sample_rate),
        audio_format="wav",
    )
    return {
        "sample_id": sample_id,
        "row_index": row_index,
        "audio_column": audio_column,
        "transcript": transcript,
        "sv_count": len(sv_values),
        "segment_count": len(segments),
        "top_segment_idx": top_segment_idx,
        "random_segment_idx": random_segment_idx,
        "top_token": segments[top_segment_idx].token,
        "random_token": segments[random_segment_idx].token,
        "top_audio_bytes": len(top_audio),
        "random_audio_bytes": len(random_audio),
    }


def _generate_response(
    model: Any,
    audio_bytes: bytes,
    input_modality: InputModality,
    max_new_tokens: int,
    text_temperature: float,
) -> Any:
    chat = build_chat(
        model,
        user_texts=None,
        audio_bytes_list=[audio_bytes],
        input_modality=input_modality,
    )
    return model.generate(
        chat=chat,
        max_new_tokens=max_new_tokens,
        model_config=ModelConfig(text_temperature=float(text_temperature)),
        keep_history=False,
    )


def _run_one_sample(
    *,
    sample_path: Path,
    row: dict[str, Any],
    model: Any,
    aligner: SpectrogramGuidedAligner,
    input_modality: InputModality,
    audio_column: str,
    max_new_tokens: int,
    text_temperature: float,
    rng: np.random.Generator,
) -> FaithfulnessResult:
    sample_id = _parse_sample_id(sample_path)
    sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
    row_index = int(sample_json.get("row_index", sample_id))
    transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
    audio_values = _as_list(row[audio_column])
    if not audio_values:
        raise ValueError(f"No audio bytes in column {audio_column}")
    audio_bytes = audio_values[0]

    sv_values = _extract_audio_sv(sample_json)
    top_segment_idx = int(np.argmax(np.abs(np.asarray(sv_values, dtype=float))))
    top_sv = float(sv_values[top_segment_idx])

    waveform, sample_rate = TorchAudioHandler.from_bytes(
        audio_bytes, audio_format="wav"
    )
    segments = aligner(
        transcript=transcript,
        waveform=waveform,
        original_sr=int(sample_rate),
        audio_format="wav",
        attach_audio=False,
    )
    if not segments:
        raise ValueError("Alignment produced no segments.")
    if top_segment_idx >= len(segments):
        raise ValueError(
            f"Top-SV index {top_segment_idx} exceeds aligned segment count {len(segments)}."
        )

    candidate_indices = [i for i in range(len(segments)) if i != top_segment_idx]
    if not candidate_indices:
        raise ValueError("Need at least two segments for random baseline.")
    random_segment_idx = int(rng.choice(candidate_indices))

    top_seg = segments[top_segment_idx]
    random_seg = segments[random_segment_idx]
    top_start, top_end = _segment_interval(top_seg)
    mask_duration = max(1, top_end - top_start)
    random_start_raw, random_end_raw = _segment_interval(random_seg)
    random_center = (random_start_raw + random_end_raw) // 2
    random_start, random_end = _equal_duration_interval(
        center_sample=random_center,
        duration_samples=mask_duration,
        total_samples=waveform.size(-1),
    )

    top_waveform = _mask_interval(waveform, top_start, top_end)
    random_waveform = _mask_interval(waveform, random_start, random_end)
    top_audio = TorchAudioHandler.to_bytes(
        top_waveform, sample_rate=int(sample_rate), audio_format="wav"
    )
    random_audio = TorchAudioHandler.to_bytes(
        random_waveform, sample_rate=int(sample_rate), audio_format="wav"
    )

    t0 = time.perf_counter()
    base_response = _generate_response(
        model, audio_bytes, input_modality, max_new_tokens, text_temperature
    )
    top_response = _generate_response(
        model, top_audio, input_modality, max_new_tokens, text_temperature
    )
    random_response = _generate_response(
        model, random_audio, input_modality, max_new_tokens, text_temperature
    )
    original_sim, top_sim, random_sim = _response_similarity(
        base_response, top_response, random_response
    )

    top_drop = original_sim - top_sim
    random_drop = original_sim - random_sim
    return FaithfulnessResult(
        sample_id=sample_id,
        row_index=row_index,
        audio_column=audio_column,
        transcript=transcript,
        n_segments=len(segments),
        top_segment_idx=top_segment_idx,
        random_segment_idx=random_segment_idx,
        top_segment_token=top_seg.token,
        random_segment_token=random_seg.token,
        top_abs_sv=abs(top_sv),
        top_sv=top_sv,
        original_similarity=original_sim,
        top_similarity=top_sim,
        random_similarity=random_sim,
        top_drop=top_drop,
        random_drop=random_drop,
        drop_difference=top_drop - random_drop,
        top_start_sec=float(top_start / sample_rate),
        top_end_sec=float(top_end / sample_rate),
        random_start_sec=float(random_start / sample_rate),
        random_end_sec=float(random_end / sample_rate),
        mask_duration_sec=float(mask_duration / sample_rate),
        runtime_sec=float(time.perf_counter() - t0),
    )


def _summarize(results_df: pd.DataFrame, failures_df: pd.DataFrame) -> dict[str, Any]:
    if results_df.empty:
        return {
            "completed_samples": 0,
            "failed_samples": int(len(failures_df)),
        }

    top = results_df["top_drop"].to_numpy(dtype=float)
    random = results_df["random_drop"].to_numpy(dtype=float)
    diff = top - random
    has_pairwise_stats = len(results_df) >= 2
    t_stat, p_value = (
        stats.ttest_rel(top, random) if has_pairwise_stats else (None, None)
    )
    return {
        "completed_samples": int(len(results_df)),
        "failed_samples": int(len(failures_df)),
        "mean_top_drop": float(np.mean(top)),
        "std_top_drop": float(np.std(top, ddof=1)) if has_pairwise_stats else None,
        "mean_random_drop": float(np.mean(random)),
        "std_random_drop": float(np.std(random, ddof=1))
        if has_pairwise_stats
        else None,
        "mean_drop_difference": float(np.mean(diff)),
        "median_drop_difference": float(np.median(diff)),
        "paired_t_stat": float(t_stat) if t_stat is not None else None,
        "paired_p_value": float(p_value) if p_value is not None else None,
        "cohen_dz": (
            float(np.mean(diff) / (np.std(diff, ddof=1) + EPS))
            if has_pairwise_stats
            else None
        ),
        "top_greater_than_random_rate": float(np.mean(diff > 0)),
        "mean_runtime_sec": float(results_df["runtime_sec"].mean()),
    }


def combine_partition_outputs(output_dir: Path) -> dict[str, Any]:
    """Combine partition CSVs and write voice-level plus combined summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    combined_frames: list[pd.DataFrame] = []

    for audio_column in ("audio__male", "audio__female"):
        paths = sorted(output_dir.glob(f"{audio_column}_part*-of*_results.csv"))
        if not paths:
            continue
        df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        df = df.drop_duplicates(subset=["sample_id"], keep="last")
        df = df.sort_values("sample_id")
        result_path = output_dir / f"{audio_column}_combined_results.csv"
        summary_path = output_dir / f"{audio_column}_combined_summary.json"
        df.to_csv(result_path, index=False)
        failures_paths = sorted(
            output_dir.glob(f"{audio_column}_part*-of*_failures.csv")
        )
        failures_df = (
            pd.concat([pd.read_csv(path) for path in failures_paths], ignore_index=True)
            if failures_paths
            else pd.DataFrame()
        )
        summary = _summarize(df, failures_df)
        summary.update({"audio_column": audio_column, "results_csv": str(result_path)})
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[audio_column] = summary
        combined_frames.append(df)

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_df = combined_df.sort_values(["audio_column", "sample_id"])
        combined_path = output_dir / "combined_results.csv"
        combined_summary_path = output_dir / "combined_summary.json"
        combined_df.to_csv(combined_path, index=False)
        combined_summary = _summarize(combined_df, pd.DataFrame())
        combined_summary.update({"results_csv": str(combined_path)})
        combined_summary_path.write_text(
            json.dumps(combined_summary, indent=2), encoding="utf-8"
        )
        summaries["combined"] = combined_summary

    return summaries


def run_faithfulness(
    *,
    run_dir: Path,
    spec_path: Path | None,
    output_dir: Path,
    max_samples: int | None,
    device: str,
    aligner_device: str,
    seed: int,
    partition_index: int | None,
    num_partitions: int | None,
    resume: bool,
    preflight_only: bool = False,
    max_new_tokens_override: int | None = None,
    text_temperature_override: float | None = None,
) -> dict[str, Any]:
    """Run deletion faithfulness for one mllm_shapx run directory."""
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = _load_spec(run_dir, spec_path=spec_path)
    cfg = _experiment_set_from_spec(spec)
    input_modality = cfg.modality.get_input_modality()
    output_modality = cfg.modality.get_output_modality()
    if output_modality != OutputModality.AUDIO:
        raise ValueError("HP-1 faithfulness currently expects audio-output runs.")
    audio_column = cfg.modality.input_modality

    rows = _load_selected_rows(cfg, max_samples=max_samples)
    sample_paths = _sample_paths(run_dir, max_samples=max_samples)
    if partition_index is not None or num_partitions is not None:
        if partition_index is None or num_partitions is None:
            raise ValueError("Provide both partition_index and num_partitions.")
        sample_paths = [
            path
            for i, path in enumerate(sample_paths)
            if i % int(num_partitions) == int(partition_index)
        ]

    suffix = audio_column
    if partition_index is not None:
        suffix += f"_part{partition_index}-of-{num_partitions}"
    results_path = output_dir / f"{suffix}_results.csv"
    failures_path = output_dir / f"{suffix}_failures.csv"
    summary_path = output_dir / f"{suffix}_summary.json"

    existing_ids: set[int] = set()
    results: list[FaithfulnessResult] = []
    failures: list[FailureResult] = []
    if resume and results_path.exists():
        existing = pd.read_csv(results_path)
        existing_ids = set(existing["sample_id"].astype(int).tolist())

    aligner = SpectrogramGuidedAligner(device=torch.device(aligner_device))
    rng = np.random.default_rng(seed)

    if preflight_only:
        checks: list[dict[str, Any]] = []
        for sample_path in sample_paths:
            sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
            row_index = int(sample_json.get("row_index", _parse_sample_id(sample_path)))
            checks.append(
                _preflight_one_sample(
                    sample_path=sample_path,
                    row=rows[row_index],
                    aligner=aligner,
                    audio_column=audio_column,
                    rng=rng,
                )
            )
        preflight_path = output_dir / f"{suffix}_preflight.json"
        summary = {
            "preflight_only": True,
            "checked_samples": len(checks),
            "run_dir": str(run_dir),
            "audio_column": audio_column,
            "preflight_json": str(preflight_path),
            "checks": checks,
        }
        preflight_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    torch_device = torch.device(device)
    model = _build_model(
        device=torch_device,
        connector=cfg.connector,
        output_modality=output_modality,
    )
    max_new_tokens = (
        int(max_new_tokens_override)
        if max_new_tokens_override is not None
        else int(cfg.generation.max_new_tokens)
    )
    text_temperature = (
        float(text_temperature_override)
        if text_temperature_override is not None
        else float(cfg.generation.text_temperature)
    )

    for sample_path in tqdm(sample_paths, desc=f"faithfulness ({audio_column})"):
        sample_id = _parse_sample_id(sample_path)
        if sample_id in existing_ids:
            continue
        row_index = sample_id
        transcript = ""
        try:
            sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
            row_index = int(sample_json.get("row_index", sample_id))
            row = rows[row_index]
            transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
            result = _run_one_sample(
                sample_path=sample_path,
                row=row,
                model=model,
                aligner=aligner,
                input_modality=input_modality,
                audio_column=audio_column,
                max_new_tokens=max_new_tokens,
                text_temperature=text_temperature,
                rng=rng,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - keep long cluster runs alive.
            failures.append(
                FailureResult(
                    sample_id=sample_id,
                    row_index=row_index,
                    audio_column=audio_column,
                    transcript=transcript,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

        if results:
            new_df = pd.DataFrame([asdict(r) for r in results])
            if resume and results_path.exists():
                old_df = pd.read_csv(results_path)
                new_df = pd.concat([old_df, new_df], ignore_index=True)
                new_df = new_df.drop_duplicates(subset=["sample_id"], keep="last")
            new_df.sort_values("sample_id").to_csv(results_path, index=False)
        if failures:
            pd.DataFrame([asdict(f) for f in failures]).to_csv(
                failures_path, index=False
            )

    final_results = (
        pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    )
    final_failures = (
        pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    )
    summary = _summarize(final_results, final_failures)
    summary.update(
        {
            "run_dir": str(run_dir),
            "audio_column": audio_column,
            "results_csv": str(results_path),
            "failures_csv": str(failures_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=None,
        help="Path to committed mllm_shapx spec JSON; falls back to RUN_DIR/spec.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aligner-device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--partition-index", type=int, default=None)
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override spec generation.max_new_tokens, useful for one-sample smoke tests.",
    )
    parser.add_argument(
        "--text-temperature",
        type=float,
        default=None,
        help="Override spec generation.text_temperature.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate data/SV/alignment/masking/audio encoding without loading LFM2.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Combine partition CSVs in --output-dir and write aggregate summaries.",
    )
    return parser


def main() -> None:
    """Run CLI."""
    args = build_argparser().parse_args()
    if args.combine_only:
        summary = combine_partition_outputs(args.output_dir)
        print(json.dumps(summary, indent=2))
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --combine-only is used.")

    summary = run_faithfulness(
        run_dir=args.run_dir,
        spec_path=args.spec_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
        aligner_device=args.aligner_device,
        seed=args.seed,
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
        resume=args.resume,
        preflight_only=args.preflight_only,
        max_new_tokens_override=args.max_new_tokens,
        text_temperature_override=args.text_temperature,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
