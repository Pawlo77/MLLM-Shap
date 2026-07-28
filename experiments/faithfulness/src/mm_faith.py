"""Model-agnostic SGPA exact-Shapley faithfulness runner.

Identical methodology to ``qwen_faith`` (SGPA word players, exact Shapley over
all 2^n coalitions via the package ``PreciseShapExplainer``, held-out E5 +
TF-IDF utilities, length-matched random-deletion baseline), but the model is
swapped in via the ``AudioTextBackend`` factory (``--backend qwen|voxtral``).

The stable helpers are imported from ``qwen_faith`` so this file adds no
duplicated logic and, crucially, does not modify ``qwen_faith`` (whose code the
overnight Qwen queue depends on).
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler
from tqdm.auto import tqdm

from experiments.mllm_shapx.src.data import extract_texts_from_row

from .backends import build_backend
from .helpers import as_list, rank_abs_sv
from .io import experiment_set_from_spec, load_selected_rows, load_spec
from .qwen_faith import (
    TARGET_SR,
    E5Embedder,
    _append_csv,
    _exact_shapley_pkg,
    _mask_waveform,
    _summarize,
    _tfidf_cos_to_base,
    select_word_banded_ids,
)


def run_faithfulness(
    run_dir: Path,
    output_dir: Path,
    backend_name: str,
    max_samples: int,
    max_players: int,
    device: str,
    aligner_device: str,
    instruction: str,
    max_new_tokens: int,
    resume: bool,
    mask_mode: str = "silence",
    stage3_off: bool = False,
    embedder_device: str | None = None,
    file_prefix: str = "exact_shapley",
    min_words: int = 4,
    max_words: int = 7,
    full_pool: bool = False,
) -> dict[str, Any]:
    """SGPA exact-Shapley faithfulness for an arbitrary audio->text backend."""
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(run_dir, spec_path=None)
    cfg = experiment_set_from_spec(spec)
    audio_column = cfg.modality.input_modality
    # Scan from index 0; word-band slice below fixes a comparable-length set.
    # ``full_pool`` bypasses token-count balancing to draw the word-banded 100
    # from the whole dataset (needed for the TTS voice sets).
    sel_update: dict[str, Any] = {"start_index": 0, "max_samples": None}
    if full_pool:
        sel_update["balanced_token_counts"] = None
        sel_update["samples_per_token_count"] = None
    sel = cfg.selection.model_copy(update=sel_update)
    cfg = cfg.model_copy(update={"selection": sel})
    rows = load_selected_rows(cfg, max_samples=None)

    results_path = output_dir / f"{file_prefix}_results.csv"
    summary_path = output_dir / f"{file_prefix}_summary.json"
    coalitions_path = output_dir / f"{file_prefix}_coalitions.csv"
    existing_ids: set[int] = set()
    if resume and results_path.exists():
        existing_ids = set(pd.read_csv(results_path)["sample_id"].astype(int).tolist())

    aligner = SpectrogramGuidedAligner(
        device=torch.device(aligner_device), refine_boundaries=not stage3_off
    )
    # E5 is tiny; keep it off the main GPU when the LM is memory-hungry (e.g. bf16).
    embedder = E5Embedder(device=embedder_device or device)
    backend = build_backend(backend_name, device=device)
    mask_rng = np.random.default_rng(1234)

    lo = max(2, min_words)
    hi = min(max_words, max_players)
    sample_ids, wc_hist = select_word_banded_ids(rows, lo, hi, limit=max_samples)
    print(
        f"[selection] word band [{lo},{hi}] -> {len(sample_ids)}/{max_samples} "
        f"samples from pool of {len(rows)}; word-count histogram: {wc_hist}",
        flush=True,
    )
    all_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    completed = len(existing_ids & set(sample_ids))

    pbar = tqdm(sample_ids, desc=f"{backend_name}-exact-shapley")
    for sample_id in pbar:
        pbar.set_postfix(done=completed, skipped=len(skipped))
        if sample_id in existing_ids:
            continue
        try:
            row = rows[sample_id]
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
            if n < 2:
                skipped.append({"sample_id": sample_id, "reason": f"n={n}"})
                continue
            if n > max_players:
                skipped.append({"sample_id": sample_id, "reason": f"n={n}>max"})
                continue

            spans: list[tuple[int, int]] = []
            for seg in segments:
                s = max(0, int(seg.start_time * TARGET_SR))
                e = min(len(full_16k), int(seg.end_time * TARGET_SR))
                spans.append((s, min(max(s + 1, e), len(full_16k))))

            t0 = time.perf_counter()
            present_sets: list[frozenset[int]] = [frozenset(range(n))]
            for k in range(n + 1):
                for combo in combinations(range(n), k):
                    fs = frozenset(combo)
                    if fs != present_sets[0]:
                        present_sets.append(fs)
            seen: set[frozenset[int]] = set()
            uniq: list[frozenset[int]] = []
            for fs in present_sets:
                if fs not in seen:
                    seen.add(fs)
                    uniq.append(fs)

            texts: list[str] = []
            for fs in uniq:
                masked = _mask_waveform(
                    full_16k, spans, fs, mode=mask_mode, rng=mask_rng
                )
                texts.append(
                    backend.generate_text(
                        masked, instruction=instruction, max_new_tokens=max_new_tokens
                    )
                )
            runtime = float(time.perf_counter() - t0)

            emb_sims = embedder.cos_to_base(texts)
            tfidf_sims = _tfidf_cos_to_base(texts)
            util_emb = {fs: float(emb_sims[i]) for i, fs in enumerate(uniq)}
            util_tfidf = {fs: float(tfidf_sims[i]) for i, fs in enumerate(uniq)}

            coalition_rows = [
                {
                    "sample_id": sample_id,
                    "n_segments": n,
                    "present_mask": int(sum(1 << j for j in fs)),
                    "coalition_size": len(fs),
                    "util_emb": float(emb_sims[i]),
                    "util_tfidf": float(tfidf_sims[i]),
                }
                for i, fs in enumerate(uniq)
            ]
            _append_csv(coalitions_path, coalition_rows)

            sv_emb = _exact_shapley_pkg(n, uniq, emb_sims)
            rank_info = rank_abs_sv(sv_emb)
            rank_order = rank_info["order"]
            full = frozenset(range(n))

            for idx, seg in enumerate(segments):
                present_wo_i = full - {idx}
                del_drop_emb = 1.0 - util_emb[present_wo_i]
                del_drop_tfidf = 1.0 - util_tfidf[present_wo_i]
                rank = int(rank_info["ranks"][idx])
                topk = frozenset(int(rank_order[j]) for j in range(rank))
                present_wo_topk = full - topk
                cum_drop_emb = 1.0 - util_emb[present_wo_topk]
                all_rows.append(
                    {
                        "sample_id": sample_id,
                        "transcript": transcript,
                        "n_segments": n,
                        "segment_idx": idx,
                        "segment_rank_abs_sv": rank,
                        "segment_token": seg.token,
                        "segment_sv": float(sv_emb[idx]),
                        "segment_abs_sv": float(rank_info["abs_values"][idx]),
                        "segment_abs_sv_share": float(rank_info["shares"][idx]),
                        "deletion_drop_emb": del_drop_emb,
                        "deletion_drop_tfidf": del_drop_tfidf,
                        "cumulative_drop_emb": cum_drop_emb,
                        "cumulative_n_deleted": rank,
                        "segment_dur_sec": float(seg.end_time - seg.start_time),
                        "util_empty_emb": float(util_emb[frozenset()]),
                        "base_text": texts[0],
                        "n_coalitions": len(uniq),
                        "runtime_sec": runtime / max(1, n),
                    }
                )

            if all_rows:
                new_df = pd.DataFrame(all_rows)
                if resume and results_path.exists():
                    new_df = pd.concat(
                        [pd.read_csv(results_path), new_df], ignore_index=True
                    ).drop_duplicates(subset=["sample_id", "segment_idx"], keep="last")
                new_df.sort_values(["sample_id", "segment_rank_abs_sv"]).to_csv(
                    results_path, index=False
                )
            completed += 1
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {"sample_id": sample_id, "reason": f"{type(exc).__name__}: {exc}"}
            )
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            continue

    final = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    summary = _summarize(final)
    summary.update(
        {
            "backend": backend_name,
            "instruction": instruction,
            "utility": "E5 embedding cosine (exact Shapley)",
            "mask_mode": mask_mode,
            "stage3_off": bool(stage3_off),
            "audio_column": audio_column,
            "run_dir": str(run_dir),
            "word_band": [lo, hi],
            "full_pool": bool(full_pool),
            "n_selected": len(sample_ids),
            "word_count_histogram": {str(k): v for k, v in wc_hist.items()},
            "n_skipped": len(skipped),
            "results_csv": str(results_path),
            "coalitions_csv": str(coalitions_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    (output_dir / f"{file_prefix}_skipped.json").write_text(
        json.dumps(skipped, indent=2)
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backend", default="voxtral", help="qwen | voxtral")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--max-players", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aligner-device", default="cpu")
    p.add_argument("--embedder-device", default=None)
    p.add_argument(
        "--instruction", default="Repeat the exact words that the speaker said."
    )
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--mask-mode", choices=["silence", "noise", "concat"], default="silence"
    )
    p.add_argument("--stage3-off", action="store_true")
    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--max-words", type=int, default=7)
    p.add_argument("--full-pool", action="store_true")
    args = p.parse_args()
    summary = run_faithfulness(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        backend_name=args.backend,
        max_samples=args.max_samples,
        max_players=args.max_players,
        device=args.device,
        aligner_device=args.aligner_device,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
        mask_mode=args.mask_mode,
        stage3_off=args.stage3_off,
        embedder_device=args.embedder_device,
        min_words=args.min_words,
        max_words=args.max_words,
        full_pool=args.full_pool,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
