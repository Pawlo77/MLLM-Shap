"""Three-way Stage-3 boundary decomposition (revision guide 7.2 + 7.3).

For every internal word boundary we record spectral flux at three cut positions:
  * raw_ctc   -- the raw CTC word-end edge,
  * midpoint  -- the gap midpoint (tL+tR)/2 between adjacent raw word spans
                 (the silence-aware fallback position), and
  * refined   -- the Stage-3 Eq.-2 argmin (with midpoint fallback).

This isolates how much of the headline flux reduction comes from simply moving to
the gap midpoint versus from the spectral search. Also records the per-boundary
``boundary_refined`` flag, which yields the segment fallback rate for every voice
condition (including Original TTS, missing from fallback_rate.json -- guide 7.3).

Runs on the same pinned 3010-utterance corpus as stage3_regen.py.

    .venv/bin/python -m experiments.faithfulness.src.stage3_threeway [max_samples_per_cond]
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler

_run_path = Path("experiments/interspeech/src/stage3/run.py")
_spec = importlib.util.spec_from_file_location("stage3_run", _run_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
_raw_and_refined_segments = _mod._raw_and_refined_segments
_spectral_flux_at_times = _mod._spectral_flux_at_times

CACHE = Path(
    "/home/mvishiu11/.cache/huggingface/hub/datasets--Pawlo77--mllm-shap/"
    "snapshots/6c046d6c94a76ddb2bb9e5577fd51e7fb77bb691"
)
SOURCES = [
    ("single_sentence_1k", "audio__male", "Male TTS"),
    ("single_sentence_1k", "audio__female", "Female TTS"),
    ("single_sentence_1k", "audio__original", "Original TTS"),
    ("single_sentence_500", "audio__original", "Natural (LibriSpeech)"),
]
OUT = Path("experiments/faithfulness/outputs/stage3_threeway")


def _as_bytes(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, list):
        v = v[0]
    return v


def _boundaries(raw_seg, ref_seg, dur):
    """Per internal word boundary k: (raw_edge, midpoint, refined, refined_flag)."""
    raw = sorted(raw_seg, key=lambda s: float(s.start_time))
    ref = sorted(ref_seg, key=lambda s: float(s.start_time))
    n = min(len(raw), len(ref))
    out = []
    for k in range(n - 1):
        tL = float(raw[k].end_time)
        tR = float(raw[k + 1].start_time)
        raw_pos = tL
        mid_pos = 0.5 * (tL + tR) if tR > tL else tL
        ref_pos = float(ref[k].end_time)
        flag = bool(getattr(ref[k], "boundary_refined", False))
        for p in (raw_pos, mid_pos, ref_pos):
            if not (0.0 < p < dur and np.isfinite(p)):
                break
        else:
            out.append((raw_pos, mid_pos, ref_pos, flag))
    return out


def _process(aligner, df, cfg, col, label, max_samples):
    recs = []
    n = min(len(df), max_samples)
    for i in tqdm(range(n), desc=f"{cfg}/{col}"):
        row = df.iloc[i]
        sents = row["sentences"]
        transcript = str(sents[0] if hasattr(sents, "__len__") else sents)
        try:
            waveform, sr = TorchAudioHandler.from_bytes(
                _as_bytes(row[col]), audio_format="wav"
            )
            dur = float(waveform.size(-1) / sr)
            raw_seg, ref_seg = _raw_and_refined_segments(
                aligner, transcript, waveform, int(sr)
            )
            bnds = _boundaries(raw_seg, ref_seg, dur)
            if not bnds:
                continue
            raw_t = [b[0] for b in bnds]
            mid_t = [b[1] for b in bnds]
            ref_t = [b[2] for b in bnds]
            fr = _spectral_flux_at_times(waveform, int(sr), raw_t)
            fm = _spectral_flux_at_times(waveform, int(sr), mid_t)
            fref = _spectral_flux_at_times(waveform, int(sr), ref_t)
            for bi, (b, r, m, x) in enumerate(zip(bnds, fr, fm, fref)):
                recs.append(
                    {
                        "dataset_config": cfg,
                        "audio_column": col,
                        "label": label,
                        "sample_idx": i,
                        "boundary_idx": bi,
                        "flux_raw": float(r),
                        "flux_mid": float(m),
                        "flux_refined": float(x),
                        "boundary_refined": bool(b[3]),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip {cfg}/{col} #{i}] {type(exc).__name__}: {exc}")
    return recs


def _red(a, b):
    a, b = float(a), float(b)
    return 100.0 * (a - b) / a if a > 0 else 0.0


def main(max_samples: int = 100000) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligner = SpectrogramGuidedAligner(device=device)

    recs = []
    for cfg, col, label in SOURCES:
        pq = CACHE / cfg / "test" / "0000.parquet"
        recs.extend(
            _process(aligner, pd.read_parquet(pq), cfg, col, label, max_samples)
        )
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "boundaries.csv", index=False)

    def block(g):
        raw, mid, ref = g.flux_raw.mean(), g.flux_mid.mean(), g.flux_refined.mean()
        gref = g[
            g.boundary_refined
        ]  # boundaries where the spectral search actually fired
        out = {
            "n_boundaries": int(len(g)),
            "fallback_rate": float(1.0 - g.boundary_refined.mean()),
            "flux_raw": float(raw),
            "flux_mid": float(mid),
            "flux_refined": float(ref),
            "reduction_total_pct": _red(raw, ref),
            "reduction_midpoint_pct": _red(raw, mid),
            "reduction_search_over_mid_pct": _red(mid, ref),
        }
        if len(gref):
            out["refined_only_flux_mid"] = float(gref.flux_mid.mean())
            out["refined_only_flux_refined"] = float(gref.flux_refined.mean())
            out["refined_only_search_gain_pct"] = _red(
                gref.flux_mid.mean(), gref.flux_refined.mean()
            )
        # decomposition share of the total raw->refined reduction
        tot = raw - ref
        out["share_from_midpoint"] = float((raw - mid) / tot) if tot != 0 else None
        out["share_from_search"] = float((mid - ref) / tot) if tot != 0 else None
        return out

    per = {lab: block(df[df.label == lab]) for lab in df.label.unique()}
    combined = block(df)
    summary = {"per_condition": per, "combined": combined}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Three-way Stage-3 decomposition ===")
    print(
        f"{'condition':24s} {'nB':>6} {'fallb%':>7} {'raw':>7} {'mid':>7} {'ref':>7} "
        f"{'tot%':>6} {'mid%':>6} {'srch%':>6}"
    )
    for lab, b in list(per.items()) + [("COMBINED", combined)]:
        print(
            f"{lab:24s} {b['n_boundaries']:6d} {100 * b['fallback_rate']:6.1f} "
            f"{b['flux_raw']:7.2f} {b['flux_mid']:7.2f} {b['flux_refined']:7.2f} "
            f"{b['reduction_total_pct']:6.1f} {b['reduction_midpoint_pct']:6.1f} "
            f"{b['reduction_search_over_mid_pct']:6.1f}"
        )
    print(f"\nwrote {OUT / 'summary.json'} and {OUT / 'boundaries.csv'}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    main(n)
