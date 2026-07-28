"""Appendix analyses for the SGPA exact-Shapley faithfulness study.

Two panels, both models (Qwen2-Audio, Voxtral):

  Q1.3  Rank-wise monotonicity (full n): per-sample Spearman(|SV|, deletion
        drop) distribution + mean deletion drop by |SV| rank. Shows that under
        exact Shapley the drop decreases monotonically with rank -- the positive
        counterpart to the old LFM2 "flat/saturated" appendix diagnostic.

  Q2.3  Faithfulness stratified by Stage-3 boundary-refinement fallback bins
        (0% / 1-25% / >25%). Fallback fraction is a property of the audio +
        aligner, so it is computed once (CPU alignment of the LibriSpeech
        original set) and joined to each model's per-sample faithfulness.

Outputs (CSV/JSON + paper-style PNGs) go to outputs/appendix_analysis/.
Alignment for Q2.3 is cached to fallback_per_sample.json so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ── Paper styling (matches experiments.interspeech.src.sgpa_plot.STYLES) ──────
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
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}
# Okabe-Ito colour-blind-safe: Qwen blue, Voxtral vermillion.
MODEL_COLOR = {"Qwen2-Audio-7B": "#0055A4", "Voxtral-Mini-3B": "#D55E00"}

MODELS = {
    "Qwen2-Audio-7B": (
        Path("experiments/faithfulness/outputs"),
        "qwen_exact_shapley_{cond}",
        "qwen_exact_shapley_results.csv",
    ),
    "Voxtral-Mini-3B": (
        Path("experiments/faithfulness/outputs/voxtral"),
        "exact_shapley_{cond}",
        "exact_shapley_results.csv",
    ),
}
CONDS = [
    "original",
    "male",
    "female",
    "ttsorig",
    "stage3off",
    "mask_noise",
    "mask_concat",
]

ORIG_RUN_DIR = Path(
    "experiments/experiments_output/aaai27_fixed_500_original/"
    "audio_original_audio_sgpa_limited_neyman_lin3_0"
)


def _apply_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PAPER_RC)


def _clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_all(model: str) -> pd.DataFrame:
    base, dirtmpl, resname = MODELS[model]
    frames = []
    for cond in CONDS:
        f = base / dirtmpl.format(cond=cond) / resname
        if f.exists():
            df = pd.read_csv(f)
            df["condition"] = cond
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_original(model: str) -> pd.DataFrame:
    base, dirtmpl, resname = MODELS[model]
    f = base / dirtmpl.format(cond="original") / resname
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


# ── Q1.3: rank-wise monotonicity ──────────────────────────────────────────────
def _within_spearman(df: pd.DataFrame) -> list[float]:
    out = []
    for _, g in df.groupby(["condition", "sample_id"]):
        if g["segment_abs_sv"].nunique() < 2 or g["deletion_drop_emb"].nunique() < 2:
            continue
        r = stats.spearmanr(g["segment_abs_sv"], g["deletion_drop_emb"]).statistic
        if np.isfinite(r):
            out.append(float(r))
    return out


def q1_3_monotonicity(out: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    summary: dict[str, Any] = {}
    max_rank = 7

    for model in MODELS:
        df = _load_all(model)
        if df.empty:
            continue
        color = MODEL_COLOR[model]
        # Per-rank mean deletion drop + 95% CI (ranks with >=5 obs).
        rank_stats = []
        for rk in range(1, max_rank + 1):
            v = df.loc[df["segment_rank_abs_sv"] == rk, "deletion_drop_emb"].dropna()
            if len(v) < 5:
                continue
            m = float(v.mean())
            ci = float(v.sem() * stats.t.ppf(0.975, len(v) - 1))
            rank_stats.append((rk, m, ci, len(v)))
        if rank_stats:
            rks = [r[0] for r in rank_stats]
            ms = [r[1] for r in rank_stats]
            cis = [r[2] for r in rank_stats]
            axL.errorbar(
                rks,
                ms,
                yerr=cis,
                marker="o",
                markersize=5,
                linewidth=2.0,
                capsize=3,
                color=color,
                label=model,
                zorder=3,
            )
        ws = _within_spearman(df)
        summary[model] = {
            "n_samples": int(df.groupby(["condition", "sample_id"]).ngroups),
            "within_spearman_mean": float(np.mean(ws)) if ws else None,
            "within_spearman_median": float(np.median(ws)) if ws else None,
            "within_spearman_iqr": [
                float(np.percentile(ws, 25)),
                float(np.percentile(ws, 75)),
            ]
            if ws
            else None,
            "frac_positive": float(np.mean(np.array(ws) > 0)) if ws else None,
            "per_rank_mean_drop": {r[0]: round(r[1], 4) for r in rank_stats},
            "_ws": ws,
        }

    axL.set_title(
        "Deletion drop by $|\\mathrm{SV}|$ rank", loc="left", fontweight="bold"
    )
    axL.set_xlabel("segment rank by $|\\mathrm{SV}|$  (1 = highest)")
    axL.set_ylabel("response-similarity drop")
    axL.set_xticks(range(1, max_rank + 1))
    axL.grid(axis="y", linestyle="--", alpha=0.3)
    axL.legend(frameon=False)
    _clean(axL)

    # Right: within-sample Spearman distributions (violin per model).
    data, labels, colors = [], [], []
    for model in MODELS:
        ws = summary.get(model, {}).get("_ws")
        if ws:
            data.append(ws)
            labels.append(model.split("-")[0])
            colors.append(MODEL_COLOR[model])
    if data:
        parts = axR.violinplot(data, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.45)
        for i, (ws, c) in enumerate(zip(data, colors), start=1):
            jitter = np.random.default_rng(0).normal(0, 0.04, len(ws))
            axR.scatter(
                np.full(len(ws), i) + jitter,
                ws,
                s=6,
                color="#333",
                alpha=0.35,
                zorder=3,
            )
            axR.hlines(
                np.median(ws), i - 0.25, i + 0.25, color=c, linewidth=2.2, zorder=4
            )
        axR.set_xticks(range(1, len(labels) + 1))
        axR.set_xticklabels(labels)
    axR.axhline(0, color="#888", linestyle=":", linewidth=1.0)
    axR.set_ylim(-1.05, 1.05)
    axR.set_title(
        "Within-sample Spearman$(|\\mathrm{SV}|,\\,\\mathrm{drop})$",
        loc="left",
        fontweight="bold",
    )
    axR.set_ylabel("Spearman $\\rho$ per sample")
    axR.grid(axis="y", linestyle="--", alpha=0.3)
    _clean(axR)

    fig.tight_layout()
    fig.savefig(out / "appendix_monotonicity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    for m in summary:
        summary[m].pop("_ws", None)
    (out / "appendix_monotonicity.json").write_text(json.dumps(summary, indent=2))
    print(f"[Q1.3] monotonicity -> appendix_monotonicity.png ({list(summary)})")
    return summary


# ── Q2.3: faithfulness stratified by fallback bin ─────────────────────────────
def _compute_fallback_per_sample(sample_ids: set[int], out: Path) -> dict[int, float]:
    """Align the LibriSpeech original set (CPU) and return per-sample fallback frac."""
    cache = out / "fallback_per_sample.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        cached = {int(k): float(v) for k, v in d.items()}
        if sample_ids <= set(cached):
            print(f"[Q2.3] using cached fallback for {len(cached)} samples")
            return cached

    import torch
    from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
    from mllm_shap.utils.audio import TorchAudioHandler

    from experiments.mllm_shapx.src.data import extract_texts_from_row

    from .helpers import as_list
    from .io import experiment_set_from_spec, load_selected_rows, load_spec

    spec = load_spec(ORIG_RUN_DIR, spec_path=None)
    cfg = experiment_set_from_spec(spec)
    audio_column = cfg.modality.input_modality
    sel = cfg.selection.model_copy(update={"start_index": 0, "max_samples": None})
    cfg = cfg.model_copy(update={"selection": sel})
    rows = load_selected_rows(cfg, max_samples=None)

    aligner = SpectrogramGuidedAligner(
        device=torch.device("cpu"), refine_boundaries=True
    )
    frac: dict[int, float] = {}
    for sid in sorted(sample_ids):
        if sid not in rows:
            continue
        try:
            row = rows[sid]
            transcript = " ".join(extract_texts_from_row(row[row["_text_col"]]))
            audio_bytes = as_list(row[audio_column])[0]
            wav_t, sr = TorchAudioHandler.from_bytes(audio_bytes, audio_format="wav")
            segs = aligner(
                transcript=transcript,
                waveform=wav_t,
                original_sr=int(sr),
                audio_format="wav",
                attach_audio=False,
            )
            if not segs:
                continue
            flags = [bool(getattr(s, "boundary_refined", True)) for s in segs]
            frac[sid] = sum(1 for f in flags if not f) / len(flags)
        except Exception as e:  # noqa: BLE001
            print(f"[Q2.3] align failed sid={sid}: {e}")
    cache.write_text(json.dumps({str(k): v for k, v in frac.items()}, indent=2))
    print(f"[Q2.3] aligned {len(frac)} samples -> fallback_per_sample.json")
    return frac


def _per_sample_faith(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for sid, g in df.groupby("sample_id"):
        top = g.loc[g["segment_rank_abs_sv"] == 1, "deletion_drop_emb"]
        nontop = g.loc[g["segment_rank_abs_sv"] > 1, "deletion_drop_emb"]
        if top.empty or nontop.empty:
            continue
        recs.append(
            {
                "sample_id": int(sid),
                "top_drop": float(top.iloc[0]),
                "nontop_drop": float(nontop.mean()),
                "faith": float(top.iloc[0] - nontop.mean()),
            }
        )
    return pd.DataFrame(recs)


def _bin(frac: float) -> str:
    if frac <= 0:
        return "0%"
    if frac <= 0.25:
        return "1-25%"
    return ">25%"


BIN_ORDER = ["0%", "1-25%", ">25%"]


def q2_3_fallback(out: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    _apply_style()
    faith = {m: _per_sample_faith(_load_original(m)) for m in MODELS}
    all_ids: set[int] = set()
    for f in faith.values():
        all_ids |= set(f["sample_id"].tolist())
    frac = _compute_fallback_per_sample(all_ids, out)

    rows = []
    for model, f in faith.items():
        if f.empty:
            continue
        f = f.copy()
        f["fallback_frac"] = f["sample_id"].map(frac)
        f = f.dropna(subset=["fallback_frac"])
        f["bin"] = f["fallback_frac"].map(_bin)
        for b in BIN_ORDER:
            sub = f[f["bin"] == b]
            if sub.empty:
                continue
            faith_vals = sub["faith"].to_numpy()
            dz = (
                float(faith_vals.mean() / (faith_vals.std(ddof=1) + 1e-9))
                if len(sub) > 1
                else float("nan")
            )
            rows.append(
                {
                    "model": model,
                    "fallback_bin": b,
                    "n": int(len(sub)),
                    "mean_top_drop": round(float(sub["top_drop"].mean()), 4),
                    "mean_nontop_drop": round(float(sub["nontop_drop"].mean()), 4),
                    "mean_faith": round(float(sub["faith"].mean()), 4),
                    "cohen_dz": round(dz, 3),
                }
            )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "appendix_fallback_strata.csv", index=False)

    # Grouped bar chart of mean faithfulness (top - nontop) by bin, per model.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    models = list(MODELS)
    x = np.arange(len(BIN_ORDER))
    w = 0.38
    for i, model in enumerate(models):
        sub = tbl[tbl["model"] == model].set_index("fallback_bin").reindex(BIN_ORDER)
        vals = sub["mean_faith"].to_numpy(dtype=float)
        ns = sub["n"].to_numpy()
        bars = ax.bar(
            x + (i - (len(models) - 1) / 2) * w,
            np.nan_to_num(vals),
            w,
            label=model,
            color=MODEL_COLOR[model],
            alpha=0.9,
        )
        for b, n in zip(bars, ns):
            if np.isfinite(n):
                ax.annotate(
                    f"n={int(n)}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fallback\n{b}" for b in BIN_ORDER])
    ax.set_ylabel("top$-$nontop deletion drop")
    ax.set_title(
        "Faithfulness by boundary-refinement fallback", loc="left", fontweight="bold"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out / "appendix_fallback_strata.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Q2.3] fallback strata -> appendix_fallback_strata.csv ({len(tbl)} rows)")
    return tbl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/faithfulness/outputs/appendix_analysis"),
    )
    p.add_argument(
        "--skip-fallback", action="store_true", help="Q1.3 only (no alignment)"
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    q1_3_monotonicity(args.out)
    if not args.skip_fallback:
        tbl = q2_3_fallback(args.out)
        print(tbl.to_string(index=False))
    print(f"Appendix analysis written to {args.out}")


if __name__ == "__main__":
    main()
