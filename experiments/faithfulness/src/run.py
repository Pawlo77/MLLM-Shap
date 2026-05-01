"""Main faithfulness runner and CLI.

Deletion-based faithfulness validation for SGPA Shapley values.

Consumes existing mllm_shapx SGPA runs, identifies the highest absolute-SV
word segment per sample, and tests whether deleting that segment changes the
model response more than deleting the mean of all other segments.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler
from tqdm.auto import tqdm

from experiments.mllm_shapx.src.constants import OutputModality
from experiments.mllm_shapx.src.data import extract_texts_from_row
from experiments.mllm_shapx.src.factory import build_model

from .audio import (
    aggregate_sv_to_segments,
    extract_audio_sv,
    remove_interval,
    segment_interval,
)
from .helpers import as_list
from .io import (
    experiment_set_from_spec,
    load_selected_rows,
    load_spec,
    parse_sample_id,
    sample_paths,
)
from .models import FailureResult
from .runners import run_one_sample, run_one_sample_rankwise
from .summarize import combine_partition_outputs, summarize, summarize_rankwise

DEFAULT_OUTPUT_DIR = Path("experiments/faithfulness/outputs")


def run_faithfulness(
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
    all_rank_deletions: bool = False,
    max_new_tokens_override: int | None = None,
    text_temperature_override: float | None = None,
    random_draws: int = 20,
    strat_duration_bins: int = 4,
    strat_position_bins: int = 4,
    comprehensiveness_k: int = 3,
    sufficiency_k: int = 3,
    target_effect_size_dz: float = 0.5,
) -> dict[str, Any]:
    """Run the faithfulness evaluation for a given run directory
    and return a summary dictionary of results and metadata."""
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(run_dir, spec_path=spec_path)
    cfg = experiment_set_from_spec(spec)
    input_modality = cfg.modality.get_input_modality()
    if cfg.modality.get_output_modality() != OutputModality.AUDIO:
        raise ValueError("HP-1 faithfulness expects audio-output runs.")
    audio_column = cfg.modality.input_modality

    paths = sample_paths(run_dir, max_samples=max_samples)
    rows = load_selected_rows(cfg, max_samples=None)

    if partition_index is not None and num_partitions is not None:
        paths = [
            p
            for i, p in enumerate(paths)
            if i % int(num_partitions) == int(partition_index)
        ]
    elif (partition_index is None) != (num_partitions is None):
        raise ValueError("Provide both partition_index and num_partitions.")

    suffix = audio_column
    if partition_index is not None:
        suffix += f"_part{partition_index}-of-{num_partitions}"
    results_tag = "rankwise_results" if all_rank_deletions else "results"
    summary_tag = "rankwise_summary" if all_rank_deletions else "summary"
    results_path = output_dir / f"{suffix}_{results_tag}.csv"
    failures_path = output_dir / f"{suffix}_failures.csv"
    summary_path = output_dir / f"{suffix}_{summary_tag}.json"

    existing_ids: set[int] = set()
    if resume and results_path.exists():
        existing_ids = set(pd.read_csv(results_path)["sample_id"].astype(int).tolist())

    aligner = SpectrogramGuidedAligner(device=torch.device(aligner_device))
    rng = np.random.default_rng(seed)

    if preflight_only:
        checks = []
        for sp in paths:
            sample_json = json.loads(sp.read_text(encoding="utf-8"))
            row_index = int(sample_json.get("row_index", parse_sample_id(sp)))
            checks.append(
                _preflight_one_sample(
                    sample_path=sp,
                    row=rows[row_index],
                    aligner=aligner,
                    audio_column=audio_column,
                    rng=rng,
                )
            )
        preflight_path = output_dir / f"{suffix}_preflight.json"
        summary_data = {
            "preflight_only": True,
            "checked_samples": len(checks),
            "checks": checks,
        }
        preflight_path.write_text(json.dumps(summary_data, indent=2))
        return summary_data

    torch_device = torch.device(device)
    model = build_model(
        device=torch_device,
        connector=cfg.connector,
        output_modality=cfg.modality.get_output_modality(),
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

    results: list[Any] = []
    failures: list[FailureResult] = []

    for sample_path in tqdm(paths, desc=f"faithfulness ({audio_column})"):
        sample_id = parse_sample_id(sample_path)
        if sample_id in existing_ids:
            continue
        row_index = sample_id
        transcript = ""
        try:
            sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
            row_index = int(sample_json.get("row_index", sample_id))
            row = rows[row_index]
            transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
            if all_rank_deletions:
                results.extend(
                    run_one_sample_rankwise(
                        sample_path=sample_path,
                        row=row,
                        model=model,
                        aligner=aligner,
                        input_modality=input_modality,
                        audio_column=audio_column,
                        max_new_tokens=max_new_tokens,
                        text_temperature=text_temperature,
                    )
                )
            else:
                results.append(
                    run_one_sample(
                        sample_path=sample_path,
                        row=row,
                        model=model,
                        aligner=aligner,
                        input_modality=input_modality,
                        audio_column=audio_column,
                        max_new_tokens=max_new_tokens,
                        text_temperature=text_temperature,
                        rng=rng,
                        random_draws=random_draws,
                        strat_duration_bins=strat_duration_bins,
                        strat_position_bins=strat_position_bins,
                        comprehensiveness_k=comprehensiveness_k,
                        sufficiency_k=sufficiency_k,
                    )
                )
        except Exception as exc:  # noqa: BLE001
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
                new_df = pd.concat(
                    [pd.read_csv(results_path), new_df], ignore_index=True
                )
                dedup = (
                    ["sample_id", "segment_idx"]
                    if all_rank_deletions
                    else ["sample_id"]
                )
                new_df = new_df.drop_duplicates(subset=dedup, keep="last")
            sort_cols = (
                ["sample_id", "segment_rank_abs_sv", "segment_idx"]
                if all_rank_deletions
                else ["sample_id"]
            )
            new_df.sort_values(sort_cols).to_csv(results_path, index=False)
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
    summary_data = (
        summarize_rankwise(final_results, final_failures)
        if all_rank_deletions
        else summarize(
            final_results, final_failures, target_effect_size_dz=target_effect_size_dz
        )
    )
    summary_data.update(
        {
            "run_dir": str(run_dir),
            "audio_column": audio_column,
            "all_rank_deletions": all_rank_deletions,
            "results_csv": str(results_path),
            "failures_csv": str(failures_path),
            "random_draws": int(random_draws),
            "strat_duration_bins": int(strat_duration_bins),
            "strat_position_bins": int(strat_position_bins),
            "comprehensiveness_k": int(comprehensiveness_k),
            "sufficiency_k": int(sufficiency_k),
            "target_effect_size_dz": float(target_effect_size_dz),
        }
    )
    summary_path.write_text(json.dumps(summary_data, indent=2))
    return summary_data


def _preflight_one_sample(
    sample_path: Path,
    row: dict[str, Any],
    aligner: SpectrogramGuidedAligner,
    audio_column: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Perform preflight checks for a single sample to ensure it can be processed without
    errors during the main run. This includes loading the audio, performing alignment,
    and identifying segments and SV values. Returns a dictionary summarizing the checks
    performed and any issues encountered."""
    sample_id = parse_sample_id(sample_path)
    sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
    row_index = int(sample_json.get("row_index", sample_id))
    transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
    audio_bytes = as_list(row[audio_column])[0]

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
    sv_values = extract_audio_sv(sample_json)
    segment_sv_values, sv_bins = aggregate_sv_to_segments(
        sv_values, segments, total_samples=waveform.size(-1)
    )
    top_idx = int(np.argmax(np.asarray(segment_sv_values, dtype=float)))
    candidate_indices = [i for i in range(len(segments)) if i != top_idx]
    random_idx = int(rng.choice(candidate_indices))
    top_start, top_end = segment_interval(segments[top_idx])
    random_start, random_end = segment_interval(segments[random_idx])
    top_audio = TorchAudioHandler.to_bytes(
        remove_interval(waveform, top_start, top_end),
        sample_rate=int(sample_rate),
        audio_format="wav",
    )
    random_audio = TorchAudioHandler.to_bytes(
        remove_interval(waveform, random_start, random_end),
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
        "top_segment_idx": top_idx,
        "top_sv_bin": sv_bins[top_idx],
        "random_segment_idx": random_idx,
        "top_token": segments[top_idx].token,
        "random_token": segments[random_idx].token,
        "top_audio_bytes": len(top_audio),
        "random_audio_bytes": len(random_audio),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--spec-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aligner-device", default="cpu")
    p.add_argument("--seed", type=int, default=20260428)
    p.add_argument("--partition-index", type=int, default=None)
    p.add_argument("--num-partitions", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--text-temperature", type=float, default=None)
    p.add_argument("--random-draws", type=int, default=20)
    p.add_argument("--strat-duration-bins", type=int, default=4)
    p.add_argument("--strat-position-bins", type=int, default=4)
    p.add_argument("--comprehensiveness-k", type=int, default=3)
    p.add_argument("--sufficiency-k", type=int, default=3)
    p.add_argument("--target-effect-size-dz", type=float, default=0.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--all-rank-deletions", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--combine-only", action="store_true")
    return p


def main() -> None:
    """Parse command-line arguments and run the faithfulness evaluation, or combine existing
    partition outputs if --combine-only is specified. The results and summary will be saved
    to the specified output directory."""
    args = build_argparser().parse_args()
    if args.combine_only:
        print(
            json.dumps(
                combine_partition_outputs(
                    args.output_dir, all_rank_deletions=args.all_rank_deletions
                ),
                indent=2,
            )
        )
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
        all_rank_deletions=args.all_rank_deletions,
        max_new_tokens_override=args.max_new_tokens,
        text_temperature_override=args.text_temperature,
        random_draws=args.random_draws,
        strat_duration_bins=args.strat_duration_bins,
        strat_position_bins=args.strat_position_bins,
        comprehensiveness_k=args.comprehensiveness_k,
        sufficiency_k=args.sufficiency_k,
        target_effect_size_dz=args.target_effect_size_dz,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
