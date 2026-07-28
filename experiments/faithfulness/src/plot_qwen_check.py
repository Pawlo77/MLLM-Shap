"""Quick comparison plot: Qwen2-Audio (exact Shapley) vs LFM2 (Neyman) faithfulness."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

QWEN = "experiments/faithfulness/outputs/qwen_exact_shapley_original/qwen_exact_shapley_results.csv"
LFM2 = "experiments/faithfulness/outputs/aaai27_fixed_500_original/audio__original_rankwise_results.csv"
OUT = "experiments/faithfulness/outputs/qwen_faithfulness_check.png"


def _within_spearman(df, abs_col, drop_col):
    vals = []
    for _, g in df.groupby("sample_id"):
        if g[abs_col].nunique() > 1 and len(g) >= 3:
            r, _ = stats.spearmanr(g[abs_col], g[drop_col])
            if np.isfinite(r):
                vals.append(r)
    return np.asarray(vals)


def _per_rank(df, drop_col, max_rank=6):
    d = df[df.segment_rank_abs_sv <= max_rank]
    m = d.groupby("segment_rank_abs_sv")[drop_col].mean()
    return m.index.to_numpy(), m.to_numpy()


def main():
    qwen = pd.read_csv(QWEN)
    lfm2 = pd.read_csv(LFM2)

    q_ws = _within_spearman(qwen, "segment_abs_sv", "deletion_drop_emb")
    l_ws = _within_spearman(lfm2, "segment_abs_sv", "deletion_drop")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # Panel 1: per-rank mean deletion drop (normalized to rank-1 for shape comparison)
    ax = axes[0]
    qr, qd = _per_rank(qwen, "deletion_drop_emb")
    lr, ld = _per_rank(lfm2, "deletion_drop")
    ax.plot(
        qr, qd / qd[0], "o-", color="#2a7", lw=2.5, ms=8, label="Qwen2-Audio (exact SV)"
    )
    ax.plot(lr, ld / ld[0], "s--", color="#c44", lw=2.5, ms=8, label="LFM2 (Neyman SV)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Word rank by |SV|  (1 = most important)")
    ax.set_ylabel("Mean deletion drop\n(normalized to rank 1)")
    ax.set_title("Faithfulness: does deleting the\ntop-|SV| word matter most?")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 2: within-sample Spearman distributions
    ax = axes[1]
    bins = np.linspace(-1, 1, 21)
    ax.hist(
        l_ws,
        bins=bins,
        alpha=0.6,
        color="#c44",
        label=f"LFM2 (med={np.median(l_ws):.2f})",
    )
    ax.hist(
        q_ws,
        bins=bins,
        alpha=0.6,
        color="#2a7",
        label=f"Qwen2-Audio (med={np.median(q_ws):.2f})",
    )
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.axvline(np.median(q_ws), color="#2a7", ls="-", lw=2)
    ax.axvline(np.median(l_ws), color="#c44", ls="--", lw=2)
    ax.set_xlabel("Within-sample Spearman(|SV|, deletion drop)")
    ax.set_ylabel("# samples")
    ax.set_title("Per-utterance rank↔impact correlation")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 3: Qwen scatter |SV| share vs deletion drop
    ax = axes[2]
    x = qwen["segment_abs_sv_share"].to_numpy()
    y = qwen["deletion_drop_emb"].to_numpy()
    ax.scatter(x, y, s=14, alpha=0.4, color="#2a7", edgecolor="none")
    if len(x) > 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color="#145", lw=2)
        rho, _ = stats.spearmanr(x, y)
        ax.text(
            0.05,
            0.92,
            f"pooled ρ = {rho:.2f}",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="#2a7"),
        )
    ax.set_xlabel("Word |SV| share")
    ax.set_ylabel("Deletion drop (embedding)")
    ax.set_title("Qwen2-Audio: attribution vs\nmeasured deletion impact")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "SGPA word-level Shapley faithfulness — same pipeline, two models "
        f"(Qwen n={qwen.sample_id.nunique()}, LFM2 n={lfm2.sample_id.nunique()})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved", OUT)
    print(
        f"Qwen within-spearman: mean={q_ws.mean():.3f} median={np.median(q_ws):.3f} n={len(q_ws)} frac>0={np.mean(q_ws > 0):.2f}"
    )
    print(
        f"LFM2 within-spearman: mean={l_ws.mean():.3f} median={np.median(l_ws):.3f} n={len(l_ws)} frac>0={np.mean(l_ws > 0):.2f}"
    )


if __name__ == "__main__":
    main()
