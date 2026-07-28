"""Model-internal log-probability faithfulness endpoint for SGPA (Qwen/Voxtral).

Every endpoint in the paper (E5, TF-IDF) scores *response similarity*. This runner
adds a model-internal endpoint: the mean per-token log-probability of the model's
own full-input response, teacher-forced under each (silence-)masked coalition. The
target token sequence is fixed across coalitions, so the drop when the top-|SV|
word is silenced---relative to a length-matched random word---is a faithfulness
signal that shares nothing with the similarity utilities.

It reuses the cached exact-SV runs: SGPA segments are re-derived by re-running the
(deterministic) aligner for spans, while the |SV| ranking and the target response
(``base_text``) are read from the cached ``*_exact_shapley_results.csv``. Only the
scoring is new -- no re-attribution.

    python -m experiments.faithfulness.src.logprob_endpoint \
        --model qwen --condition librispeech_original \
        --run-dir <SGPA run dir> \
        --cached-results experiments/faithfulness/outputs/qwen_exact_shapley_original/qwen_exact_shapley_results.csv \
        --output-dir experiments/faithfulness/outputs/logprob_qwen_voxtral
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler
from scipy import stats
from tqdm.auto import tqdm

from experiments.mllm_shapx.src.data import extract_texts_from_row

from .helpers import as_list
from .io import experiment_set_from_spec, load_selected_rows, load_spec
from .qwen_faith import _mask_waveform, select_word_banded_ids

TARGET_SR = 16000


def _load_backend(model: str, device: str):
    if model == "qwen":
        from .qwen_audio_backend import QwenAudioBackend

        return QwenAudioBackend(device=device)
    if model == "voxtral":
        from .voxtral_audio_backend import VoxtralAudioBackend

        return VoxtralAudioBackend(device=device)
    raise ValueError(f"unknown model {model!r}")


def run(
    model: str,
    condition: str,
    run_dir: Path,
    cached_results: Path,
    output_dir: Path,
    max_samples: int,
    max_players: int,
    device: str,
    aligner_device: str,
    instruction: str,
    full_pool: bool,
    min_words: int,
    max_words: int,
    resume: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(run_dir, spec_path=None)
    cfg = experiment_set_from_spec(spec)
    audio_column = cfg.modality.input_modality

    sel_update: dict[str, Any] = {"start_index": 0, "max_samples": None}
    if full_pool:
        sel_update["balanced_token_counts"] = None
        sel_update["samples_per_token_count"] = None
    cfg = cfg.model_copy(
        update={"selection": cfg.selection.model_copy(update=sel_update)}
    )
    rows = load_selected_rows(cfg, max_samples=None)

    cached = pd.read_csv(cached_results)
    # cached ranking + target text, keyed by sample and segment
    rank_by = {
        (int(r.sample_id), int(r.segment_idx)): (
            int(r.segment_rank_abs_sv),
            float(r.segment_abs_sv),
        )
        for r in cached.itertuples()
    }
    base_text_by = {int(r.sample_id): str(r.base_text) for r in cached.itertuples()}

    lo = max(2, min_words)
    hi = min(max_words, max_players)
    sample_ids, _ = select_word_banded_ids(rows, lo, hi, limit=max_samples)
    sample_ids = [s for s in sample_ids if s in base_text_by]

    results_path = output_dir / f"{model}_{condition}_logprob.csv"
    existing: set[int] = set()
    if resume and results_path.exists():
        existing = set(pd.read_csv(results_path)["sample_id"].astype(int))

    aligner = SpectrogramGuidedAligner(
        device=torch.device(aligner_device), refine_boundaries=True
    )
    backend = _load_backend(model, device)

    all_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for sid in tqdm(sample_ids, desc=f"logprob {model}/{condition}"):
        if sid in existing:
            continue
        try:
            row = rows[sid]
            transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
            audio_bytes = as_list(row[audio_column])[0]
            wav_t, sr = TorchAudioHandler.from_bytes(audio_bytes, audio_format="wav")
            wav_mono = wav_t[0] if wav_t.dim() > 1 else wav_t
            wav16 = torchaudio.functional.resample(wav_mono, int(sr), TARGET_SR)
            full_16k = wav16.detach().cpu().numpy().astype(np.float32)

            segments = aligner(
                transcript=transcript,
                waveform=wav_t,
                original_sr=int(sr),
                audio_format="wav",
                attach_audio=False,
            )
            n = len(segments)
            # must match the cached attribution segmentation
            if not all((sid, i) in rank_by for i in range(n)) or n < 2:
                skipped.append({"sample_id": sid, "reason": f"n={n} rank mismatch"})
                continue

            spans: list[tuple[int, int]] = []
            for seg in segments:
                s = max(0, int(seg.start_time * TARGET_SR))
                e = min(len(full_16k), int(seg.end_time * TARGET_SR))
                spans.append((s, min(max(s + 1, e), len(full_16k))))

            target = base_text_by[sid]
            full = frozenset(range(n))
            t0 = time.perf_counter()
            base_lp = backend.score_text_logprob(
                full_16k, target, instruction=instruction
            )
            for idx, seg in enumerate(segments):
                present = full - {idx}
                masked = _mask_waveform(full_16k, spans, present, mode="silence")
                del_lp = backend.score_text_logprob(
                    masked, target, instruction=instruction
                )
                rank, abs_sv = rank_by[(sid, idx)]
                all_rows.append(
                    {
                        "sample_id": sid,
                        "condition": condition,
                        "transcript": transcript,
                        "n_segments": n,
                        "segment_idx": idx,
                        "segment_rank_abs_sv": rank,
                        "segment_token": seg.token,
                        "segment_abs_sv": abs_sv,
                        "segment_dur_sec": float(seg.end_time - seg.start_time),
                        "base_logprob": base_lp,
                        "deleted_logprob": del_lp,
                        "logprob_drop": base_lp - del_lp,
                        "runtime_sec": (time.perf_counter() - t0) / max(1, n),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"sample_id": sid, "reason": f"{type(exc).__name__}: {exc}"})
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            continue

        if all_rows:
            df = pd.DataFrame(all_rows)
            if resume and results_path.exists():
                df = pd.concat(
                    [pd.read_csv(results_path), df], ignore_index=True
                ).drop_duplicates(subset=["sample_id", "segment_idx"], keep="last")
            df.sort_values(["sample_id", "segment_rank_abs_sv"]).to_csv(
                results_path, index=False
            )

    final = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    summary = summarize(final)
    summary.update(
        {
            "model": model,
            "condition": condition,
            "endpoint": "mean per-token logprob",
            "run_dir": str(run_dir),
            "audio_column": audio_column,
            "n_selected": len(sample_ids),
            "n_skipped": len(skipped),
            "results_csv": str(results_path),
        }
    )
    (output_dir / f"{model}_{condition}_logprob_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    if skipped:
        (output_dir / f"{model}_{condition}_logprob_skipped.json").write_text(
            json.dumps(skipped, indent=2)
        )
    return summary


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"completed_samples": 0}
    col = "logprob_drop"
    out: dict[str, Any] = {
        "completed_samples": int(df.sample_id.nunique()),
        "mean_top_drop": float(df[df.segment_rank_abs_sv == 1][col].mean()),
        "mean_nontop_drop": float(df[df.segment_rank_abs_sv > 1][col].mean()),
    }
    rho, _ = stats.spearmanr(df.segment_abs_sv, df[col])
    out["pooled_spearman"] = float(rho)
    ws = []
    for _, g in df.groupby("sample_id"):
        if g.segment_abs_sv.nunique() > 1 and len(g) >= 3:
            r, _ = stats.spearmanr(g.segment_abs_sv, g[col])
            if np.isfinite(r):
                ws.append(r)
    out["within_sample_spearman_median"] = float(np.median(ws)) if ws else None
    # paired top vs duration-matched random non-top
    deltas: list[float] = []
    for _, g in df.groupby("sample_id"):
        top = g[g.segment_rank_abs_sv == 1]
        nontop = g[g.segment_rank_abs_sv > 1]
        if top.empty or nontop.empty:
            continue
        td = float(top[col].iloc[0])
        tdur = float(top["segment_dur_sec"].iloc[0])
        cand = nontop.dropna(subset=["segment_dur_sec"])
        j = (cand["segment_dur_sec"] - tdur).abs().idxmin()
        deltas.append(td - float(cand.loc[j, col]))
    d = np.asarray(deltas, dtype=float)
    if d.size:
        out["n_pairs"] = int(d.size)
        out["mean_delta"] = float(d.mean())
        out["dz"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None
        out["pos_rate"] = float((d > 0).mean())
        try:
            w = stats.wilcoxon(d, alternative="greater")
            out["wilcoxon_p_greater"] = float(w.pvalue)
        except ValueError:
            out["wilcoxon_p_greater"] = None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["qwen", "voxtral"], required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--cached-results", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/faithfulness/outputs/logprob_qwen_voxtral"),
    )
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--max-players", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aligner-device", default="cpu")
    p.add_argument(
        "--instruction", default="Repeat the exact words that the speaker said."
    )
    p.add_argument("--full-pool", action="store_true")
    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--max-words", type=int, default=7)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    summary = run(
        model=args.model,
        condition=args.condition,
        run_dir=args.run_dir,
        cached_results=args.cached_results,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_players=args.max_players,
        device=args.device,
        aligner_device=args.aligner_device,
        instruction=args.instruction,
        full_pool=args.full_pool,
        min_words=args.min_words,
        max_words=args.max_words,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
