"""Cross-model comparison for the SGPA exact-Shapley faithfulness study.

Reads the per-model consolidated analyses (``qwen_analysis/`` and
``voxtral_analysis/``) and emits a side-by-side view demonstrating that SGPA is
faithful across two distinct model families:

  * cross_model_main.{csv,md}  -- per-condition top-drop / dz / Wilcoxon p / AOPC gap
  * cross_model_aopc_gap.png   -- grouped bar chart of AOPC gaps (Qwen vs Voxtral)
  * cross_model_aopc_original.png -- AOPC curves overlay (LibriSpeech original)

Safe to re-run; guarded so a missing artifact for one model does not abort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "qwen": (
        "Qwen2-Audio-7B",
        Path("experiments/faithfulness/outputs/qwen_analysis"),
        "qwen",
    ),
    "voxtral": (
        "Voxtral-Mini-3B",
        Path("experiments/faithfulness/outputs/voxtral_analysis"),
        "voxtral",
    ),
}

CONDITIONS = [
    "librispeech_original",
    "tts_male",
    "tts_female",
    "tts_original",
    "stage3_off",
    "mask_noise",
    "mask_concat",
]


def _to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _aopc_gap(analysis_dir: Path, prefix: str, cond: str) -> float:
    f = analysis_dir / f"{prefix}_aopc_{cond}.csv"
    if not f.exists():
        return float("nan")
    a = pd.read_csv(f)
    return float(
        np.trapezoid(a["sgpa_order_drop"] - a["random_drop"], a["frac_removed"])
    )


def build_main_table(out: Path) -> pd.DataFrame:
    rows = []
    for key, (label, adir, prefix) in MODELS.items():
        mf = adir / f"{prefix}_main_faithfulness.csv"
        if not mf.exists():
            print(f"[skip] {label}: no main_faithfulness.csv")
            continue
        df = pd.read_csv(mf).set_index("condition")
        for cond in CONDITIONS:
            if cond not in df.index:
                continue
            r = df.loc[cond]
            rows.append(
                {
                    "model": label,
                    "condition": cond,
                    "n": int(r["n"]),
                    "top_drop": round(float(r["top_drop_emb"]), 4),
                    "nontop_drop": round(float(r["nontop_drop_emb"]), 4),
                    "dz": round(float(r["dz_emb"]), 3),
                    "wilcoxon_p": f"{float(r['wilcoxon_p_emb']):.1e}",
                    "pooled_spearman": round(float(r["pooled_spearman_emb"]), 3),
                    "aopc_gap": round(_aopc_gap(adir, prefix, cond), 4),
                }
            )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "cross_model_main.csv", index=False)
    (out / "cross_model_main.md").write_text(_to_md(tbl))
    print(f"[main] {len(tbl)} rows -> cross_model_main.csv")
    return tbl


def plot_aopc_gap(tbl: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    piv = tbl.pivot(index="condition", columns="model", values="aopc_gap").reindex(
        CONDITIONS
    )
    x = np.arange(len(piv))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    models = list(piv.columns)
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, piv[m].to_numpy(), w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in piv.index], fontsize=8)
    ax.set_ylabel("AOPC gap (SGPA − random)")
    ax.set_title("SGPA faithfulness across model families")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "cross_model_aopc_gap.png", dpi=150)
    plt.close(fig)
    print("[fig] cross_model_aopc_gap.png")


def plot_aopc_original(out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    styles = {"qwen": ("o-", "s--"), "voxtral": ("^-", "v--")}
    for key, (label, adir, prefix) in MODELS.items():
        f = adir / f"{prefix}_aopc_librispeech_original.csv"
        if not f.exists():
            continue
        a = pd.read_csv(f)
        s, r = styles.get(key, ("o-", "s--"))
        ax.plot(
            a["frac_removed"], a["sgpa_order_drop"], s, label=f"{label}: SGPA order"
        )
        ax.plot(
            a["frac_removed"], a["random_drop"], r, label=f"{label}: random", alpha=0.6
        )
    ax.set_xlabel("fraction of words removed")
    ax.set_ylabel("response change (1 − cosine)")
    ax.set_title("AOPC — LibriSpeech original")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "cross_model_aopc_original.png", dpi=150)
    plt.close(fig)
    print("[fig] cross_model_aopc_original.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/faithfulness/outputs/cross_model_analysis"),
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tbl = build_main_table(args.out)
    if not tbl.empty:
        plot_aopc_gap(tbl, args.out)
    plot_aopc_original(args.out)
    print(f"Cross-model analysis written to {args.out}")


if __name__ == "__main__":
    main()
