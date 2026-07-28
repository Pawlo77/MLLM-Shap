"""Log-prob (graded) faithfulness runner for SGPA Shapley values.

The similarity-based deletion metric saturates on this generative audio model:
deleting *any* word changes the response drastically, so single-word deletion
cannot discriminate high- from low-SV words. This runner replaces the
similarity utility with a *graded* one: the mean per-token log-probability of
the model's own reference text response, teacher-forced under the (perturbed)
audio prompt. Removing a salient word should lower ``logP(reference | audio)``
proportionally to its importance, giving a non-saturated faithfulness signal.

For each sample it records, per word segment:
  * single-deletion log-prob drop (delete that one segment),
  * cumulative log-prob drop (delete top-k segments by |SV|, in rank order).

Consumes the same mllm_shapx SGPA run directories as ``run.py``.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler
from scipy import stats
from tqdm.auto import tqdm

from experiments.mllm_shapx.src.constants import OutputModality
from experiments.mllm_shapx.src.factory import build_model

from .helpers import rank_abs_sv
from .io import (
    experiment_set_from_spec,
    load_selected_rows,
    load_spec,
    parse_sample_id,
    sample_paths,
)
from .runners import _prepare_sample
from .similarity import generate_response, text_logprob_score

DEFAULT_OUTPUT_DIR = Path("experiments/faithfulness/outputs")


def _audio_from_kept(
    waveform: torch.Tensor,
    intervals: list[tuple[int, int]],
    removed: set[int],
    sample_rate: int,
) -> bytes:
    """Return WAV bytes with the given segment indices removed (rest concatenated)."""
    pieces = [
        waveform[:, s:e] for idx, (s, e) in enumerate(intervals) if idx not in removed
    ]
    cat = torch.cat(pieces, dim=1) if pieces else torch.zeros(1, 1)
    return TorchAudioHandler.to_bytes(
        cat, sample_rate=int(sample_rate), audio_format="wav"
    )


def run_logprob_faithfulness(
    run_dir: Path,
    output_dir: Path,
    max_samples: int | None,
    device: str,
    aligner_device: str,
    max_new_tokens: int | None,
    text_temperature: float,
    resume: bool,
) -> dict[str, Any]:
    """Run the log-prob deletion faithfulness evaluation for one SGPA run."""
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(run_dir, spec_path=None)
    cfg = experiment_set_from_spec(spec)
    input_modality = cfg.modality.input_modality
    if cfg.modality.output_modality != OutputModality.AUDIO:
        raise ValueError("Log-prob faithfulness expects audio-output runs.")
    audio_column = cfg.modality.input_modality
    gen_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else int(cfg.generation.max_new_tokens)
    )

    results_path = output_dir / f"{audio_column}_logprob_results.csv"
    summary_path = output_dir / f"{audio_column}_logprob_summary.json"
    failures_path = output_dir / f"{audio_column}_logprob_failures.csv"

    existing_ids: set[int] = set()
    if resume and results_path.exists():
        existing_ids = set(pd.read_csv(results_path)["sample_id"].astype(int).tolist())

    paths = sample_paths(run_dir, max_samples=max_samples)
    rows = load_selected_rows(cfg, max_samples=None)

    torch_device = torch.device(device)
    model = build_model(
        device=torch_device,
        connector=cfg.connector,
        output_modality=cfg.modality.output_modality,
    )
    aligner = SpectrogramGuidedAligner(device=torch.device(aligner_device))

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for sample_path in tqdm(paths, desc=f"logprob-faith ({audio_column})"):
        sample_id = parse_sample_id(sample_path)
        if sample_id in existing_ids:
            continue
        try:
            sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
            row_index = int(sample_json.get("row_index", sample_id))
            row = rows[row_index]
            ctx = _prepare_sample(sample_path, row, aligner, audio_column)
            if len(ctx.segments) < 2:
                raise ValueError("Need >= 2 segments.")

            t0 = time.perf_counter()
            base_resp = generate_response(
                model,
                ctx.audio_bytes,
                input_modality,
                gen_tokens,
                text_temperature,
                user_texts=ctx.user_texts,
            )
            target = base_resp.generated_text_tokens
            if target.reshape(-1).numel() == 0:
                raise ValueError("Reference response produced no text tokens.")

            base_score = text_logprob_score(
                model, ctx.audio_bytes, input_modality, target, ctx.user_texts
            )

            rank_info = rank_abs_sv(ctx.segment_sv_values)
            rank_order = rank_info["order"]

            # single-segment deletions
            single_scores: list[float] = []
            for idx in range(len(ctx.segments)):
                a = _audio_from_kept(
                    ctx.waveform, ctx.intervals, {idx}, ctx.sample_rate
                )
                single_scores.append(
                    text_logprob_score(model, a, input_modality, target, ctx.user_texts)
                )

            # cumulative deletions in descending |SV| order
            cum_scores: list[float] = []
            for k in range(1, len(ctx.segments) + 1):
                removed = {int(rank_order[i]) for i in range(k)}
                a = _audio_from_kept(
                    ctx.waveform, ctx.intervals, removed, ctx.sample_rate
                )
                cum_scores.append(
                    text_logprob_score(model, a, input_modality, target, ctx.user_texts)
                )
            runtime = float(time.perf_counter() - t0)

            for idx, seg in enumerate(ctx.segments):
                rank = int(rank_info["ranks"][idx])
                start, end = ctx.intervals[idx]
                all_rows.append(
                    {
                        "sample_id": sample_id,
                        "row_index": row_index,
                        "audio_column": audio_column,
                        "transcript": ctx.transcript,
                        "n_segments": len(ctx.segments),
                        "n_target_text_tokens": int(target.reshape(-1).numel()),
                        "segment_idx": idx,
                        "segment_rank_abs_sv": rank,
                        "segment_token": seg.token,
                        "segment_sv": float(ctx.segment_sv_values[idx]),
                        "segment_abs_sv": float(rank_info["abs_values"][idx]),
                        "segment_abs_sv_share": float(rank_info["shares"][idx]),
                        "base_logprob": base_score,
                        "deleted_logprob": single_scores[idx],
                        "logprob_drop": base_score - single_scores[idx],
                        "cumulative_logprob": cum_scores[rank - 1],
                        "cumulative_logprob_drop": base_score - cum_scores[rank - 1],
                        "cumulative_n_deleted": rank,
                        "segment_start_sec": float(start / ctx.sample_rate),
                        "segment_end_sec": float(end / ctx.sample_rate),
                        "runtime_sec": runtime / max(1, len(ctx.segments)),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "sample_id": sample_id,
                    "audio_column": audio_column,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue

        # incremental save
        if all_rows:
            new_df = pd.DataFrame(all_rows)
            if resume and results_path.exists():
                new_df = pd.concat(
                    [pd.read_csv(results_path), new_df], ignore_index=True
                ).drop_duplicates(subset=["sample_id", "segment_idx"], keep="last")
            new_df.sort_values(["sample_id", "segment_rank_abs_sv"]).to_csv(
                results_path, index=False
            )
        if failures:
            pd.DataFrame(failures).to_csv(failures_path, index=False)

    final = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    summary = _summarize(final)
    summary.update(
        {
            "run_dir": str(run_dir),
            "audio_column": audio_column,
            "results_csv": str(results_path),
            "n_failures": len(failures),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Compute discrimination stats for the log-prob deletion utility."""
    if df.empty:
        return {"completed_samples": 0}
    out: dict[str, Any] = {
        "completed_samples": int(df.sample_id.nunique()),
        "completed_deletions": int(len(df)),
    }
    top = df[df.segment_rank_abs_sv == 1]["logprob_drop"]
    nontop = df[df.segment_rank_abs_sv > 1]["logprob_drop"]
    out["mean_top_rank_logprob_drop"] = float(top.mean())
    out["mean_non_top_logprob_drop"] = float(nontop.mean())
    out["mean_top_minus_non_top_drop"] = float(top.mean() - nontop.mean())
    rho, _ = stats.spearmanr(df.segment_abs_sv, df.logprob_drop)
    out["pooled_spearman_abs_sv_vs_drop"] = float(rho)
    ws = []
    for _, g in df.groupby("sample_id"):
        if g.segment_abs_sv.nunique() > 1 and len(g) >= 3:
            r, _ = stats.spearmanr(g.segment_abs_sv, g.logprob_drop)
            if np.isfinite(r):
                ws.append(r)
    out["mean_within_sample_spearman"] = float(np.mean(ws)) if ws else None
    out["median_within_sample_spearman"] = float(np.median(ws)) if ws else None
    out["within_sample_spearman_n"] = len(ws)
    per_rank = df.groupby("segment_rank_abs_sv")["logprob_drop"].agg(["mean", "count"])
    out["per_rank_mean_logprob_drop"] = {
        int(k): {"mean_drop": float(v["mean"]), "n": int(v["count"])}
        for k, v in per_rank.iterrows()
    }
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aligner-device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--text-temperature", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    summary = run_logprob_faithfulness(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
        aligner_device=args.aligner_device,
        max_new_tokens=args.max_new_tokens,
        text_temperature=args.text_temperature,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
