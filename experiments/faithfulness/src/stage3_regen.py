"""Regenerate the Stage-3 boundary-flux ablation on the exact paper corpus.

Reads the locally-cached parquet shards for the pinned dataset revision and runs
the aligner-only raw-CTC-vs-refined flux comparison on:
  * single_sentence_1k : audio__male, audio__female, audio__original (TTS; 854 each)
  * single_sentence_500: audio__original                            (natural; 448)

854*3 + 448 = 3010 utterances, matching the paper's n. Writes a merged summary
and a paper-styled figure that replaces the old Stage-3 figure/table.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner  # noqa: E402
from mllm_shap.utils.audio import TorchAudioHandler  # noqa: E402

# Import the aligner-flux helpers from the stage3 runner without triggering its
# package __init__ (which pulls in seaborn).
_run_path = Path("experiments/interspeech/src/stage3/run.py")
_spec = importlib.util.spec_from_file_location("stage3_run", _run_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
_raw_and_refined_segments = _mod._raw_and_refined_segments
_segments_to_internal_boundaries = _mod._segments_to_internal_boundaries
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
OUT = Path("experiments/faithfulness/outputs/stage3_regen")
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 11,
}


def _as_bytes(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, list):
        v = v[0]
    return v


def _process_column(aligner, df, cfg, col, label, max_samples):
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
            rb = _segments_to_internal_boundaries(raw_seg, dur)
            fb = _segments_to_internal_boundaries(ref_seg, dur)
            k = min(len(rb), len(fb))
            if k == 0:
                continue
            raw_flux = _spectral_flux_at_times(waveform, int(sr), rb[:k])
            ref_flux = _spectral_flux_at_times(waveform, int(sr), fb[:k])
            raw_m, ref_m = float(np.mean(raw_flux)), float(np.mean(ref_flux))
            recs.append(
                {
                    "dataset_config": cfg,
                    "audio_column": col,
                    "label": label,
                    "n_words": len(ref_seg),
                    "raw_mean_flux": raw_m,
                    "refined_mean_flux": ref_m,
                    "percent_reduction": 100 * (raw_m - ref_m) / raw_m
                    if raw_m > 0
                    else 0.0,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip {cfg}/{col} #{i}] {type(exc).__name__}: {exc}")
    return recs


def main(max_samples: int = 100000) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligner = SpectrogramGuidedAligner(device=device)

    all_recs = []
    for cfg, col, label in SOURCES:
        pq = CACHE / cfg / "test" / "0000.parquet"
        df = pd.read_parquet(pq)
        all_recs.extend(_process_column(aligner, df, cfg, col, label, max_samples))

    df = pd.DataFrame(all_recs)
    df.to_csv(OUT / "samples.csv", index=False)

    rows = []
    for (cfg, col), g in df.groupby(["dataset_config", "audio_column"], sort=False):
        raw = g["raw_mean_flux"].to_numpy(float)
        ref = g["refined_mean_flux"].to_numpy(float)
        rows.append(
            {
                "label": g["label"].iloc[0],
                "dataset_config": cfg,
                "audio_column": col,
                "n": int(len(g)),
                "raw": float(raw.mean()),
                "refined": float(ref.mean()),
                "reduction_pct": float(100 * (raw.mean() - ref.mean()) / raw.mean()),
            }
        )
    raw_all = df["raw_mean_flux"].to_numpy(float)
    ref_all = df["refined_mean_flux"].to_numpy(float)
    diff = raw_all - ref_all
    t_stat, p_val = stats.ttest_rel(raw_all, ref_all)
    combined = {
        "n": int(len(df)),
        "raw": float(raw_all.mean()),
        "refined": float(ref_all.mean()),
        "reduction_pct": float(
            100 * (raw_all.mean() - ref_all.mean()) / raw_all.mean()
        ),
        "paired_t": float(t_stat),
        "paired_p": float(p_val),
        "cohen_dz": float(diff.mean() / (diff.std(ddof=1) + 1e-9)),
    }
    (OUT / "summary.json").write_text(
        json.dumps({"per_condition": rows, "combined": combined}, indent=2)
    )

    print("\n=== Stage-3 flux ablation (regenerated) ===")
    for r in rows:
        print(
            f"  {r['label']:24s} n={r['n']:4d}  raw={r['raw']:6.2f}  "
            f"refined={r['refined']:6.2f}  -{r['reduction_pct']:.2f}%"
        )
    print(
        f"  {'COMBINED':24s} n={combined['n']:4d}  raw={combined['raw']:6.2f}  "
        f"refined={combined['refined']:6.2f}  -{combined['reduction_pct']:.2f}%"
    )
    print(
        f"  paired t={combined['paired_t']:.2f} p={combined['paired_p']:.2e} dz={combined['cohen_dz']:.2f}"
    )
    _figure(df, rows, combined)


def _figure(df, rows, combined) -> None:
    plt.rcParams.update(PAPER_RC)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    axL.bar(
        x - w / 2,
        [r["raw"] for r in rows],
        w,
        label="Raw CTC",
        color="#B0B0B0",
        edgecolor="black",
        linewidth=0.6,
    )
    axL.bar(
        x + w / 2,
        [r["refined"] for r in rows],
        w,
        label="SGPA refined",
        color="#0055A4",
        edgecolor="black",
        linewidth=0.6,
    )
    for i, r in enumerate(rows):
        axL.text(
            i,
            max(r["raw"], r["refined"]) + 0.4,
            f"-{r['reduction_pct']:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=18, ha="right")
    axL.set_ylabel("mean boundary spectral flux")
    axL.set_title("Raw CTC vs SGPA-refined")
    axL.legend(frameon=False, loc="upper right")
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)

    axR.hist(
        df["percent_reduction"].to_numpy(float),
        bins=30,
        color="#0055A4",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
    )
    axR.axvline(
        combined["reduction_pct"],
        color="#D55E00",
        ls="--",
        lw=1.6,
        label=f"mean {combined['reduction_pct']:.1f}%",
    )
    axR.set_xlabel("per-utterance flux reduction (%)")
    axR.set_ylabel("utterances")
    axR.set_title(f"Reduction distribution ($n={combined['n']}$)")
    axR.legend(frameon=False, loc="upper left")
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "stage3_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    main(n)
