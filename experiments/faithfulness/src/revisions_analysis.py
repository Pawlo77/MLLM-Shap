"""V3 revision analyses (no model downloads; operates on cached CSV outputs).

Produces two things the AAAI-27 rebuttal needs:

1.  Stage-3 significance: is refined-vs-off faithfulness actually different?
    Compares per-sample (top drop - mean non-top drop) between the refined
    LibriSpeech run and the Stage-3-off run, paired by sample_id, for both
    models (Wilcoxon signed-rank). Prints dz for each and the paired p-value.

2.  Estimability of word players: on the exactly-solvable utterances, how well
    does a budgeted permutation (Monte-Carlo) Shapley estimator recover the
    exact ranking, as a function of sampling fraction? This is the scaling
    story -- exact when short, cheaply estimable when long -- computed directly
    from the cached full coalition tables (no new model calls).
    Writes: appendix_estimability.png (paper style).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QROOT = Path("experiments/faithfulness/outputs")
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
}
MODEL_COLOR = {"Qwen2-Audio-7B": "#0055A4", "Voxtral-Mini-3B": "#D55E00"}

MODELS = {
    "Qwen2-Audio-7B": {
        "orig": QROOT / "qwen_exact_shapley_original",
        "off": QROOT / "qwen_exact_shapley_stage3off",
        "prefix": "qwen",
    },
    "Voxtral-Mini-3B": {
        "orig": QROOT / "voxtral/exact_shapley_original",
        "off": QROOT / "voxtral/exact_shapley_stage3off",
        "prefix": "",
    },
}

# All conditions used to pool the estimability demonstration.
QWEN_CONDS = [
    "qwen_exact_shapley_original",
    "qwen_exact_shapley_male",
    "qwen_exact_shapley_female",
    "qwen_exact_shapley_ttsorig",
]
VOX_CONDS = [
    "voxtral/exact_shapley_original",
    "voxtral/exact_shapley_male",
    "voxtral/exact_shapley_female",
    "voxtral/exact_shapley_ttsorig",
]


def _results_path(d: Path, prefix: str) -> Path:
    name = (
        f"{prefix}_exact_shapley_results.csv" if prefix else "exact_shapley_results.csv"
    )
    return d / name


def _coalitions_path(d: Path, prefix: str) -> Path:
    # qwen runs store "qwen_coalitions.csv"; voxtral runs store
    # "exact_shapley_coalitions.csv".
    name = f"{prefix}_coalitions.csv" if prefix else "exact_shapley_coalitions.csv"
    return d / name


def _per_sample_faith(df: pd.DataFrame) -> pd.Series:
    """top drop minus mean non-top drop, per sample_id (embedding endpoint)."""
    out = {}
    for sid, g in df.groupby("sample_id"):
        g = g.sort_values("segment_rank_abs_sv")
        top = g.loc[g["segment_rank_abs_sv"] == 1, "deletion_drop_emb"]
        non = g.loc[g["segment_rank_abs_sv"] > 1, "deletion_drop_emb"]
        if len(top) and len(non):
            out[int(sid)] = float(top.iloc[0]) - float(non.mean())
    return pd.Series(out)


def stage3_significance() -> None:
    print("=" * 70)
    print("STAGE-3 SIGNIFICANCE: refined vs off (paired per-sample faithfulness)")
    print("=" * 70)
    for model, cfg in MODELS.items():
        rp_o = _results_path(cfg["orig"], cfg["prefix"])
        rp_f = _results_path(cfg["off"], cfg["prefix"])
        if not rp_o.exists() or not rp_f.exists():
            print(f"[skip] {model}: missing {rp_o} or {rp_f}")
            continue
        s_ref = _per_sample_faith(pd.read_csv(rp_o))
        s_off = _per_sample_faith(pd.read_csv(rp_f))
        common = sorted(set(s_ref.index) & set(s_off.index))
        a = s_ref.loc[common].to_numpy()
        b = s_off.loc[common].to_numpy()
        dz_ref = a.mean() / (a.std(ddof=1) + 1e-12)
        dz_off = b.mean() / (b.std(ddof=1) + 1e-12)
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        t_stat, t_p = stats.ttest_rel(a, b)
        print(f"\n{model} (n={len(common)} paired samples)")
        print(
            f"  refined dz={dz_ref:.3f}   off dz={dz_off:.3f}   Δ(off-ref)={(b.mean() - a.mean()):+.4f}"
        )
        print(f"  Wilcoxon refined-vs-off: W={w_stat:.1f}, p={w_p:.4f}")
        print(f"  paired t: t={t_stat:.3f}, p={t_p:.4f}")


def _exact_sv(util: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley from a full 2^n utility table indexed by bitmask."""
    from math import factorial

    phi = np.zeros(n)
    fact_n = factorial(n)
    for i in range(n):
        bit = 1 << i
        for S in range(1 << n):
            if S & bit:
                continue
            s = bin(S).count("1")
            w = factorial(s) * factorial(n - s - 1) / fact_n
            phi[i] += w * (util[S | bit] - util[S])
    return phi


def _mc_permutation_sv(
    util: np.ndarray, n: int, m: int, rng: np.random.Generator
) -> np.ndarray:
    """Monte-Carlo permutation estimate using m random permutations."""
    phi = np.zeros(n)
    idx = np.arange(n)
    for _ in range(m):
        rng.shuffle(idx)
        S = 0
        prev = util[0]
        for i in idx:
            Snew = S | (1 << i)
            phi[i] += util[Snew] - prev
            prev = util[Snew]
            S = Snew
    return phi / m


def estimability(out_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("ESTIMABILITY: permutation-estimator recovery of exact ranking")
    print("=" * 70)
    rng = np.random.default_rng(0)
    m_grid = [1, 2, 3, 5, 8, 12, 20, 40]

    curves: dict[str, dict[str, np.ndarray]] = {}
    for model, conds in [
        ("Qwen2-Audio-7B", QWEN_CONDS),
        ("Voxtral-Mini-3B", VOX_CONDS),
    ]:
        spearman = {m: [] for m in m_grid}
        top1 = {m: [] for m in m_grid}
        frac = {m: [] for m in m_grid}
        for cond in conds:
            prefix = "qwen" if model.startswith("Qwen") else ""
            cp = _coalitions_path(QROOT / cond, prefix)
            if not cp.exists():
                continue
            df = pd.read_csv(cp)
            for sid, g in df.groupby("sample_id"):
                n = int(g["n_segments"].iloc[0])
                if n < 3:
                    continue
                util = np.zeros(1 << n)
                for _, r in g.iterrows():
                    util[int(r["present_mask"])] = float(r["util_emb"])
                exact = _exact_sv(util, n)
                exact_rank = np.argsort(-np.abs(exact))
                for m in m_grid:
                    est = _mc_permutation_sv(util, n, m, rng)
                    rho = stats.spearmanr(np.abs(exact), np.abs(est)).correlation
                    if np.isnan(rho):
                        rho = 1.0 if np.allclose(exact, est) else 0.0
                    spearman[m].append(rho)
                    top1[m].append(
                        1.0 if np.argmax(np.abs(est)) == exact_rank[0] else 0.0
                    )
                    # unique coalitions a cached permutation estimator would touch
                    touched = min((1 << n), m * n + 1)
                    frac[m].append(touched / (1 << n))
        curves[model] = {
            "m": np.array(m_grid, dtype=float),
            "spearman": np.array([np.mean(spearman[m]) for m in m_grid]),
            "top1": np.array([np.mean(top1[m]) for m in m_grid]),
            "frac": np.array([np.mean(frac[m]) for m in m_grid]),
        }
        c = curves[model]
        print(f"\n{model}")
        for j, m in enumerate(m_grid):
            print(
                f"  m={m:>2}  frac_calls={c['frac'][j]:.2f}  "
                f"spearman={c['spearman'][j]:.3f}  top1={c['top1'][j]:.3f}"
            )

    plt.rcParams.update(PAPER_RC)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for model, c in curves.items():
        col = MODEL_COLOR[model]
        axL.plot(c["m"], c["spearman"], "-o", color=col, label=model, markersize=4)
        axR.plot(c["m"], c["top1"], "-o", color=col, label=model, markersize=4)
    for ax in (axL, axR):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("permutations $m$  (estimator budget $=mn$ calls)")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    axL.axhline(1.0, ls=":", color="grey", lw=1)
    axL.set_ylabel(
        r"Spearman$(|\mathrm{SV}|_{\mathrm{exact}},\,|\mathrm{SV}|_{\mathrm{est}})$"
    )
    axL.set_title("Rank agreement with exact SV")
    axL.legend(frameon=False, loc="lower right")
    axR.axhline(1.0, ls=":", color="grey", lw=1)
    axR.set_ylabel("top-$|\\mathrm{SV}|$ word recovered")
    axR.set_title("Top-word recovery")
    axR.set_ylim(0, 1.03)
    fig.tight_layout()
    out = out_dir / "appendix_estimability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import sys

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else QROOT / "cross_model_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage3_significance()
    estimability(out_dir)
