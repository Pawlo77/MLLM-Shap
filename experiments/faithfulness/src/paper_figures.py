"""Generate paper-grade main-result figures for the AAAI-27 SGPA submission.

Reads the consolidated per-model analyses and writes figures directly into the
paper's figures directory, styled to match the existing SGPA plots (serif,
Okabe-Ito colours, 300 dpi):

  * faithfulness_main.png -- (L) AOPC curves on LibriSpeech (SGPA order vs
    random, both models); (R) top-vs-length-matched-random effect size (dz) by
    condition for both model families.

Appendix panels (monotonicity, fallback strata) are produced by
appendix_analysis.py and copied into the figures dir by the shell wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
}
MODEL_COLOR = {"Qwen2-Audio-7B": "#0055A4", "Voxtral-Mini-3B": "#D55E00"}
QROOT = Path("experiments/faithfulness/outputs")
COND_LABELS = {
    "librispeech_original": "LibriSpeech",
    "tts_male": "TTS male",
    "tts_female": "TTS female",
    "tts_original": "TTS orig.",
    "stage3_off": "Stage-3 off",
    "mask_noise": "noise mask",
    "mask_concat": "concat mask",
}


def _clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def faithfulness_main(out_dir: Path) -> None:
    plt.rcParams.update(PAPER_RC)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 3.9))

    # ── Left: AOPC on LibriSpeech original ──────────────────────────────────
    aopc = {
        "Qwen2-Audio-7B": QROOT / "qwen_analysis/qwen_aopc_librispeech_original.csv",
        "Voxtral-Mini-3B": QROOT
        / "voxtral_analysis/voxtral_aopc_librispeech_original.csv",
    }
    for model, f in aopc.items():
        if not f.exists():
            continue
        a = pd.read_csv(f)
        c = MODEL_COLOR[model]
        axL.plot(
            a["frac_removed"],
            a["sgpa_order_drop"],
            "-",
            marker="o",
            markersize=4,
            color=c,
            linewidth=2.0,
            label=f"{model} — SGPA order",
        )
        axL.plot(
            a["frac_removed"],
            a["random_drop"],
            "--",
            color=c,
            linewidth=1.6,
            alpha=0.65,
            label=f"{model} — random",
        )
        axL.fill_between(
            a["frac_removed"],
            a["random_drop"],
            a["sgpa_order_drop"],
            color=c,
            alpha=0.08,
        )
    axL.set_title("AOPC — LibriSpeech (natural)", loc="left", fontweight="bold")
    axL.set_xlabel("fraction of words removed")
    axL.set_ylabel("response change ($1-\\cos$)")
    axL.grid(axis="both", linestyle="--", alpha=0.3)
    axL.legend(frameon=False, fontsize=8.5, loc="lower right")
    _clean(axL)

    # ── Right: effect size (dz) by condition, both models ───────────────────
    cm = pd.read_csv(QROOT / "cross_model_analysis/cross_model_main.csv")
    conds = list(COND_LABELS)
    models = ["Qwen2-Audio-7B", "Voxtral-Mini-3B"]
    x = np.arange(len(conds))
    w = 0.38
    for i, model in enumerate(models):
        sub = cm[cm["model"] == model].set_index("condition").reindex(conds)
        vals = sub["dz"].to_numpy(dtype=float)
        axR.bar(
            x + (i - (len(models) - 1) / 2) * w,
            np.nan_to_num(vals),
            w,
            color=MODEL_COLOR[model],
            alpha=0.9,
            label=model,
        )
    axR.axhline(0.8, color="#009E73", linestyle="--", linewidth=1.2)
    axR.text(
        len(conds) - 0.5,
        0.82,
        "large effect ($d_z{=}0.8$)",
        color="#009E73",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    axR.set_xticks(x)
    axR.set_xticklabels([COND_LABELS[c] for c in conds], rotation=30, ha="right")
    axR.set_ylabel("effect size $d_z$ (top vs matched)")
    axR.set_title(
        "Deletion faithfulness across conditions", loc="left", fontweight="bold"
    )
    axR.legend(frameon=False, fontsize=9, loc="upper right")
    axR.grid(axis="y", linestyle="--", alpha=0.3)
    _clean(axR)

    fig.tight_layout()
    dest = out_dir / "faithfulness_main.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {dest}")


def worked_example(out_dir: Path, sample_id: int = 377) -> None:
    """Per-word |SV| share and deletion drop for one utterance, both models."""
    plt.rcParams.update(PAPER_RC)
    src = {
        "Qwen2-Audio-7B": QROOT
        / "qwen_exact_shapley_original/qwen_exact_shapley_results.csv",
        "Voxtral-Mini-3B": QROOT
        / "voxtral/exact_shapley_original/exact_shapley_results.csv",
    }
    frames = {}
    words = None
    for model, f in src.items():
        df = pd.read_csv(f)
        g = df[df["sample_id"] == sample_id].sort_values("segment_idx")
        frames[model] = g
        words = list(g["segment_token"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 3.4), sharey=True)
    y = np.arange(len(words))[::-1]
    h = 0.38
    models = list(src)
    for i, model in enumerate(models):
        g = frames[model]
        off = (i - (len(models) - 1) / 2) * h
        axL.barh(
            y + off,
            g["segment_abs_sv_share"],
            h,
            color=MODEL_COLOR[model],
            alpha=0.9,
            label=model,
        )
        axR.barh(
            y + off, g["deletion_drop_emb"], h, color=MODEL_COLOR[model], alpha=0.9
        )
    axL.set_yticks(y)
    axL.set_yticklabels(words)
    axL.set_xlabel("$|\\mathrm{SV}|$ share")
    axL.set_title(
        "Attribution concentrates on the content word",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    axL.legend(frameon=False, fontsize=9, loc="lower right")
    axL.grid(axis="x", linestyle="--", alpha=0.3)
    axR.set_xlabel("response-similarity drop when silenced")
    axR.set_title(
        "Deletion validates the ranking", loc="left", fontweight="bold", fontsize=11
    )
    axR.grid(axis="x", linestyle="--", alpha=0.3)
    for ax in (axL, axR):
        _clean(ax)
    fig.suptitle(
        "SGPA worked example: \u201cthe whole of this itinerary\u201d",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    dest = out_dir / "worked_example.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("../nlp-shap-research/papers/aaai27/sgpa/figures"),
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    faithfulness_main(args.out)
    worked_example(args.out)
    print(f"Paper figures written to {args.out}")


if __name__ == "__main__":
    main()
