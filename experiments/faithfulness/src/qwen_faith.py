"""Second-model SGPA faithfulness via EXACT word-level Shapley.

Because SGPA yields only ~4-8 word players, we can compute *exact* Shapley
values (all 2^n coalitions) instead of a budgeted estimator -- removing the
estimator as a confound. The same 2^n coalition responses also give every
single- and cumulative-deletion metric for free, so one pass produces both the
attributions and the faithfulness diagnostics.

Utility: cosine similarity (E5 sentence embeddings) between the model's text
response on a silenced coalition and its response on the full utterance. Absent
word players are replaced with silence (temporal layout preserved), matching
the SGPA baseline. Uses a lightweight audio->text backend (Qwen2-Audio), not the
full mllm_shap connector.
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
import torch.nn.functional as F
import torchaudio
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _sk_cos
from tqdm.auto import tqdm

from mllm_shap.shap.precise import PreciseShapExplainer

from experiments.mllm_shapx.src.data import extract_texts_from_row

from .helpers import as_list, rank_abs_sv
from .io import experiment_set_from_spec, load_selected_rows, load_spec
from .summarize import _summarize_delta

TARGET_SR = 16000


class E5Embedder:
    """Tiny E5 sentence embedder for a smooth text-similarity utility."""

    def __init__(self, device: str, repo: str = "intfloat/e5-small-v2") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).to(device).eval()
        self.device = device

    @torch.no_grad()
    def cos_to_base(self, texts: list[str]) -> np.ndarray:
        """Return cosine similarity of each text to texts[0] (the full-input response)."""
        safe = [t if t.strip() else "[empty]" for t in texts]
        batch = self.tok(
            ["query: " + t for t in safe],
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = F.normalize(emb, dim=-1)
        sims = (emb @ emb[0:1].T).squeeze(-1)
        return sims.float().cpu().numpy()


def _tfidf_cos_to_base(texts: list[str]) -> np.ndarray:
    """TF cosine similarity of each text to texts[0] (independent secondary utility)."""
    safe = [t if t.strip() else "[empty]" for t in texts]
    vec = TfidfVectorizer(use_idf=False, token_pattern=r"(?u)\b\w+\b")
    try:
        m = vec.fit_transform(safe)
    except ValueError:
        return np.ones(len(texts), dtype=float)
    return _sk_cos(m[0:1], m)[0]


def _exact_shapley_pkg(
    n: int, uniq: list[frozenset[int]], sims: np.ndarray
) -> list[float]:
    """Exact Shapley values via the package's ``PreciseShapExplainer`` math.

    Builds the boolean coalition-mask tensor (True = player present) for every
    subset and delegates the exact SV computation to the package, so the SGPA
    attributions are computed by ``mllm_shap`` rather than a bespoke formula.
    """
    device = torch.device("cpu")
    masks = torch.zeros((len(uniq), n), dtype=torch.bool)
    for r, fs in enumerate(uniq):
        for j in fs:
            masks[r, j] = True
    sims_t = torch.tensor(np.asarray(sims, dtype=np.float32), device=device)
    # _calculate_shap_values uses no instance state; bypass heavy __init__.
    explainer = object.__new__(PreciseShapExplainer)
    sv = explainer._calculate_shap_values(masks.to(device), sims_t, device)
    return sv.detach().cpu().float().numpy().tolist()


def _mask_waveform(
    full_16k: np.ndarray,
    spans: list[tuple[int, int]],
    present: frozenset[int],
    mode: str = "silence",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Remove absent word players from the waveform under one of three schemes.

    - ``silence``: zero the absent spans (default; preserves temporal layout).
    - ``noise``:   replace absent spans with white noise matched to the span's
                   original RMS (energy preserved, content destroyed).
    - ``concat``:  delete the absent spans' samples entirely and concatenate the
                   remainder (shortens audio; changes downstream timing).
    """
    if mode == "concat":
        keep = np.ones(len(full_16k), dtype=bool)
        for idx, (s, e) in enumerate(spans):
            if idx not in present:
                keep[s:e] = False
        out = full_16k[keep]
        # guard against an all-removed clip
        return out if out.size > 0 else full_16k[:1].copy()

    wav = full_16k.copy()
    for idx, (s, e) in enumerate(spans):
        if idx in present:
            continue
        if mode == "noise":
            seg = full_16k[s:e]
            rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
            gen = rng if rng is not None else np.random.default_rng(0)
            wav[s:e] = gen.normal(0.0, max(rms, 1e-4), size=(e - s)).astype(np.float32)
        else:  # silence
            wav[s:e] = 0.0
    return wav


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a CSV, writing a header only when the file is new."""
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def row_word_count(row: dict[str, Any]) -> int:
    """Number of whitespace-delimited words in a row's reference transcript.

    This is the lightweight analogue of the old ``mllm_shap`` token counter: we
    only need word count here because SGPA creates one word-level player per
    whitespace token, so word count upper-bounds the exact-Shapley player budget.
    """
    transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
    return len(transcript.split())


def select_word_banded_ids(
    rows: dict[int, dict[str, Any]],
    min_words: int,
    max_words: int,
    limit: int | None,
) -> tuple[list[int], dict[int, int]]:
    """Deterministically pick sample ids of comparable length by word count.

    Keeps only utterances with ``min_words <= words <= max_words`` (dropping the
    too-short and too-long), sorts by id for reproducibility, and takes the first
    ``limit`` of them. Because the same text dataset yields identical word counts
    regardless of speaker, every voice condition selects the *same* utterances,
    giving a fair paired comparison. Returns the ids plus a word-count histogram.
    """
    wc = {sid: row_word_count(r) for sid, r in rows.items()}
    cands = sorted(sid for sid, c in wc.items() if min_words <= c <= max_words)
    if limit is not None:
        cands = cands[:limit]
    hist: dict[int, int] = {}
    for sid in cands:
        hist[wc[sid]] = hist.get(wc[sid], 0) + 1
    return cands, dict(sorted(hist.items()))


def run_qwen_faithfulness(
    run_dir: Path,
    output_dir: Path,
    max_samples: int,
    max_players: int,
    device: str,
    aligner_device: str,
    instruction: str,
    max_new_tokens: int,
    resume: bool,
    mask_mode: str = "silence",
    stage3_off: bool = False,
    min_words: int = 4,
    max_words: int = 7,
    full_pool: bool = False,
) -> dict[str, Any]:
    """Exact-Shapley SGPA faithfulness for the second (Qwen2-Audio) model."""
    from .qwen_audio_backend import QwenAudioBackend

    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(run_dir, spec_path=None)
    cfg = experiment_set_from_spec(spec)
    audio_column = cfg.modality.input_modality
    # Scan from index 0, then pick a fixed word-count-banded slice below.
    # ``full_pool`` bypasses the spec's token-count balancing so we can draw a
    # word-banded 100 from the entire dataset (needed for the TTS voice sets,
    # whose token-balanced shard has too few in-band utterances); when off, the
    # original token-balanced subset is preserved.
    sel_update: dict[str, Any] = {"start_index": 0, "max_samples": None}
    if full_pool:
        sel_update["balanced_token_counts"] = None
        sel_update["samples_per_token_count"] = None
    sel = cfg.selection.model_copy(update=sel_update)
    cfg = cfg.model_copy(update={"selection": sel})
    rows = load_selected_rows(cfg, max_samples=None)

    results_path = output_dir / "qwen_exact_shapley_results.csv"
    summary_path = output_dir / "qwen_exact_shapley_summary.json"
    coalitions_path = output_dir / "qwen_coalitions.csv"
    existing_ids: set[int] = set()
    if resume and results_path.exists():
        existing_ids = set(pd.read_csv(results_path)["sample_id"].astype(int).tolist())

    aligner = SpectrogramGuidedAligner(
        device=torch.device(aligner_device), refine_boundaries=not stage3_off
    )
    embedder = E5Embedder(device=device)
    backend = QwenAudioBackend(device=device)
    mask_rng = np.random.default_rng(1234)

    # Fixed, comparable-length sample set: word count in [min_words, max_words],
    # clamped to the exact-Shapley player budget so every selected sample is
    # tractable and no bulk runtime skipping occurs.
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

    pbar = tqdm(sample_ids, desc="qwen-exact-shapley")
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
            # Enumerate all present-sets; index 0 reserved for full utterance.
            present_sets: list[frozenset[int]] = [frozenset(range(n))]
            for k in range(n + 1):
                for combo in combinations(range(n), k):
                    fs = frozenset(combo)
                    if fs != present_sets[0]:
                        present_sets.append(fs)
            # dedup while preserving order (full first)
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

            # Persist the full coalition->utility table so any-order AOPC
            # (SGPA order / random / LOO) is computable in post-processing.
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
            "model": "Qwen2-Audio-7B-Instruct (4bit)",
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
    (output_dir / "qwen_skipped.json").write_text(json.dumps(skipped, indent=2))
    return summary


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"completed_samples": 0}
    out: dict[str, Any] = {
        "completed_samples": int(df.sample_id.nunique()),
        "completed_deletions": int(len(df)),
    }
    for label, col in [("emb", "deletion_drop_emb"), ("tfidf", "deletion_drop_tfidf")]:
        top = df[df.segment_rank_abs_sv == 1][col]
        nontop = df[df.segment_rank_abs_sv > 1][col]
        rho, _ = stats.spearmanr(df.segment_abs_sv, df[col])
        ws = []
        for _, g in df.groupby("sample_id"):
            if g.segment_abs_sv.nunique() > 1 and len(g) >= 3:
                r, _ = stats.spearmanr(g.segment_abs_sv, g[col])
                if np.isfinite(r):
                    ws.append(r)
        out[f"mean_top_drop_{label}"] = float(top.mean())
        out[f"mean_non_top_drop_{label}"] = float(nontop.mean())
        out[f"top_minus_nontop_{label}"] = float(top.mean() - nontop.mean())
        out[f"pooled_spearman_{label}"] = float(rho)
        out[f"within_sample_spearman_mean_{label}"] = float(np.mean(ws)) if ws else None
        out[f"within_sample_spearman_median_{label}"] = (
            float(np.median(ws)) if ws else None
        )
        out[f"within_sample_n_{label}"] = len(ws)
        pr = df.groupby("segment_rank_abs_sv")[col].mean().round(4).to_dict()
        out[f"per_rank_mean_drop_{label}"] = {int(k): float(v) for k, v in pr.items()}

        # Length-matched random-deletion baseline (paper Table-4 style test):
        # per sample, compare the top-|SV| word's deletion drop against the drop
        # of a duration-matched *non-top* word. Positive delta => SGPA-selected
        # word matters more than a length-matched random word.
        deltas: list[float] = []
        has_dur = "segment_dur_sec" in df.columns
        for _, g in df.groupby("sample_id"):
            if len(g) < 2:
                continue
            top_row = g[g.segment_rank_abs_sv == 1]
            nontop = g[g.segment_rank_abs_sv > 1]
            if top_row.empty or nontop.empty:
                continue
            top_drop = float(top_row[col].iloc[0])
            matched_drop = float(nontop[col].mean())
            if has_dur:
                top_dur = float(top_row["segment_dur_sec"].iloc[0])
                cand = nontop.dropna(subset=["segment_dur_sec"])
                if np.isfinite(top_dur) and not cand.empty:
                    j = (cand["segment_dur_sec"] - top_dur).abs().idxmin()
                    matched_drop = float(cand.loc[j, col])
            deltas.append(top_drop - matched_drop)
        if deltas:
            out[f"top_vs_matched_random_{label}"] = _summarize_delta(
                np.asarray(deltas, dtype=float)
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/faithfulness/outputs/qwen_exact_shapley"),
    )
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--max-players", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aligner-device", default="cpu")
    p.add_argument(
        "--instruction", default="Repeat the exact words that the speaker said."
    )
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--mask-mode",
        choices=["silence", "noise", "concat"],
        default="silence",
        help="How absent word players are removed from the waveform.",
    )
    p.add_argument(
        "--stage3-off",
        action="store_true",
        help="Disable SGPA Stage-3 boundary refinement (raw CTC boundaries).",
    )
    p.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Minimum reference-transcript word count (drop too-short samples).",
    )
    p.add_argument(
        "--max-words",
        type=int,
        default=7,
        help="Maximum word count (drop too-long; clamped to --max-players).",
    )
    p.add_argument(
        "--full-pool",
        action="store_true",
        help="Ignore the spec's token-count balancing and draw the word-banded "
        "sample from the entire dataset (use for the TTS voice sets).",
    )
    args = p.parse_args()
    summary = run_qwen_faithfulness(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_players=args.max_players,
        device=args.device,
        aligner_device=args.aligner_device,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
        mask_mode=args.mask_mode,
        stage3_off=args.stage3_off,
        min_words=args.min_words,
        max_words=args.max_words,
        full_pool=args.full_pool,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
