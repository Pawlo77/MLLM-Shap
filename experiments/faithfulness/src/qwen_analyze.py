"""Consolidated analysis for the SGPA exact-Shapley faithfulness experiments.

Reads the per-condition output dirs produced by ``qwen_faith`` / ``mm_faith``
and emits paper-ready tables + figures (prefixed by the model name):

  * <m>_main_faithfulness.{csv,md}  -- top-|SV| vs length-matched random, per condition
  * <m>_voice_table.csv             -- voice-dependence view
  * <m>_attribution_stats.csv       -- Gini / entropy / top-20% share of |SV|
  * <m>_aopc_<cond>.{csv,png}       -- AOPC: SGPA-|SV| order vs random removal
  * <m>_stage3_compare.csv          -- refined vs raw-CTC (Stage-3-off) faithfulness
  * <m>_masking_ablation.csv        -- silence vs noise vs delete+concat

``--model {qwen,voxtral}`` selects the run-dir layout and file naming; the
default reproduces the original Qwen analysis exactly. Every section is guarded
so a single failure does not abort the rest -- safe to re-run at any time.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Cfg:
    """Model-specific paths and file-naming for the analysis."""

    base: Path
    out: Path
    dirmap: dict[str, str]
    res: str
    summ: str
    coal: str
    prefix: str


def _make_cfg(model: str, base: Path | None, out: Path | None) -> Cfg:
    model = model.lower()
    if model == "voxtral":
        dirmap = {
            "librispeech_original": "exact_shapley_original",
            "tts_male": "exact_shapley_male",
            "tts_female": "exact_shapley_female",
            "tts_original": "exact_shapley_ttsorig",
            "stage3_off": "exact_shapley_stage3off",
            "mask_noise": "exact_shapley_mask_noise",
            "mask_concat": "exact_shapley_mask_concat",
        }
        return Cfg(
            base=base or Path("experiments/faithfulness/outputs/voxtral"),
            out=out or Path("experiments/faithfulness/outputs/voxtral_analysis"),
            dirmap=dirmap,
            res="exact_shapley_results.csv",
            summ="exact_shapley_summary.json",
            coal="exact_shapley_coalitions.csv",
            prefix="voxtral",
        )
    # Default: Qwen2-Audio (original layout).
    dirmap = {
        "librispeech_original": "qwen_exact_shapley_original",
        "tts_male": "qwen_exact_shapley_male",
        "tts_female": "qwen_exact_shapley_female",
        "tts_original": "qwen_exact_shapley_ttsorig",
        "stage3_off": "qwen_exact_shapley_stage3off",
        "mask_noise": "qwen_exact_shapley_mask_noise",
        "mask_concat": "qwen_exact_shapley_mask_concat",
    }
    return Cfg(
        base=base or Path("experiments/faithfulness/outputs"),
        out=out or Path("experiments/faithfulness/outputs/qwen_analysis"),
        dirmap=dirmap,
        res="qwen_exact_shapley_results.csv",
        summ="qwen_exact_shapley_summary.json",
        coal="qwen_coalitions.csv",
        prefix="qwen",
    )


def _load(cfg: Cfg, dirname: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    d = cfg.base / dirname
    res = d / cfg.res
    summ = d / cfg.summ
    coal = d / cfg.coal
    df = pd.read_csv(res) if res.exists() else pd.DataFrame()
    sm = json.loads(summ.read_text()) if summ.exists() else {}
    cdf = pd.read_csv(coal) if coal.exists() else pd.DataFrame()
    return df, sm, cdf


def _to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no deps)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n)


def _entropy(shares: np.ndarray) -> float:
    p = shares[shares > 0]
    if p.size == 0:
        return float("nan")
    return float(-np.sum(p * np.log(p)))


def main_faithfulness(cfg: Cfg) -> None:
    rows = []
    for label, dirname in cfg.dirmap.items():
        _, sm, _ = _load(cfg, dirname)
        if not sm or sm.get("completed_samples", 0) == 0:
            continue
        tv = sm.get("top_vs_matched_random_emb") or {}
        tvt = sm.get("top_vs_matched_random_tfidf") or {}
        rows.append(
            {
                "condition": label,
                "n": sm.get("completed_samples"),
                "mask_mode": sm.get("mask_mode", "silence"),
                "stage3_off": sm.get("stage3_off", False),
                "top_drop_emb": round(sm.get("mean_top_drop_emb", float("nan")), 4),
                "nontop_drop_emb": round(
                    sm.get("mean_non_top_drop_emb", float("nan")), 4
                ),
                "dz_emb": round(tv.get("cohen_dz", float("nan")), 3),
                "pos_rate_emb": tv.get("positive_rate"),
                "wilcoxon_p_emb": tv.get("wilcoxon_p_value"),
                "dz_tfidf": round(tvt.get("cohen_dz", float("nan")), 3),
                "wilcoxon_p_tfidf": tvt.get("wilcoxon_p_value"),
                "pooled_spearman_emb": round(
                    sm.get("pooled_spearman_emb", float("nan")), 3
                ),
                "within_spearman_median_emb": sm.get(
                    "within_sample_spearman_median_emb"
                ),
            }
        )
    if not rows:
        print("[main_faithfulness] no completed conditions yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out / f"{cfg.prefix}_main_faithfulness.csv", index=False)
    (cfg.out / f"{cfg.prefix}_main_faithfulness.md").write_text(_to_md(df))
    print(
        f"[main_faithfulness] {len(df)} conditions -> {cfg.prefix}_main_faithfulness.csv"
    )


def voice_table(cfg: Cfg) -> None:
    voices = ["librispeech_original", "tts_male", "tts_female", "tts_original"]
    rows = []
    for label in voices:
        _, sm, _ = _load(cfg, cfg.dirmap[label])
        if not sm or sm.get("completed_samples", 0) == 0:
            continue
        tv = sm.get("top_vs_matched_random_emb") or {}
        rows.append(
            {
                "voice": label,
                "n": sm.get("completed_samples"),
                "top_drop_emb": round(sm.get("mean_top_drop_emb", float("nan")), 4),
                "dz_emb": round(tv.get("cohen_dz", float("nan")), 3),
                "within_spearman_median_emb": sm.get(
                    "within_sample_spearman_median_emb"
                ),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(
            cfg.out / f"{cfg.prefix}_voice_table.csv", index=False
        )
        print(f"[voice_table] {len(rows)} voices -> {cfg.prefix}_voice_table.csv")


def attribution_stats(cfg: Cfg) -> None:
    rows = []
    for label, dirname in cfg.dirmap.items():
        df, sm, _ = _load(cfg, dirname)
        if df.empty:
            continue
        ginis, ents, top20s, ns = [], [], [], []
        for _, g in df.groupby("sample_id"):
            shares = g["segment_abs_sv_share"].to_numpy(dtype=float)
            shares = shares[np.isfinite(shares)]
            if shares.size == 0:
                continue
            shares = shares / (shares.sum() + 1e-12)
            n = shares.size
            k = max(1, int(np.ceil(0.2 * n)))
            top20s.append(float(np.sort(shares)[::-1][:k].sum()))
            ginis.append(_gini(shares))
            ents.append(_entropy(shares))
            ns.append(n)
        if not ns:
            continue
        rows.append(
            {
                "condition": label,
                "n_samples": len(ns),
                "mean_players": round(float(np.mean(ns)), 2),
                "gini_mean": round(float(np.nanmean(ginis)), 3),
                "entropy_mean": round(float(np.nanmean(ents)), 3),
                "top20pct_share_mean": round(float(np.mean(top20s)), 3),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(
            cfg.out / f"{cfg.prefix}_attribution_stats.csv", index=False
        )
        print(
            f"[attribution_stats] {len(rows)} conditions -> {cfg.prefix}_attribution_stats.csv"
        )


def _sample_aopc(cdf_s: pd.DataFrame, svs: dict[int, float], grid: np.ndarray):
    """Return (sgpa_curve, random_curve) over a fraction-removed grid for one sample."""
    n = int(cdf_s["n_segments"].iloc[0])
    if n < 2 or not svs:
        return None
    util = dict(zip(cdf_s["present_mask"].astype(int), cdf_s["util_emb"].astype(float)))
    full_mask = (1 << n) - 1
    order = sorted(range(n), key=lambda i: -abs(svs.get(i, 0.0)))  # top-|SV| first
    # size -> list of (1-util) over all coalitions of that size (for random removal)
    by_size: dict[int, list[float]] = {}
    for m, u in util.items():
        by_size.setdefault(int(m).bit_count(), []).append(1.0 - u)
    sgpa_k, rand_k = [], []
    for k in range(0, n + 1):
        remove = order[:k]
        mask = full_mask
        for i in remove:
            mask &= ~(1 << i)
        sgpa_k.append(1.0 - util.get(mask, 1.0 if k == 0 else float("nan")))
        keep_size = n - k
        rand_vals = by_size.get(keep_size, [])
        rand_k.append(float(np.mean(rand_vals)) if rand_vals else float("nan"))
    fracs = np.arange(n + 1) / n
    sgpa_i = np.interp(grid, fracs, np.nan_to_num(sgpa_k))
    rand_i = np.interp(grid, fracs, np.nan_to_num(rand_k))
    return sgpa_i, rand_i


def aopc(cfg: Cfg) -> None:
    grid = np.linspace(0, 1, 11)
    for label, dirname in cfg.dirmap.items():
        df, sm, cdf = _load(cfg, dirname)
        if cdf.empty or df.empty:
            continue
        sv_by_sample = {
            sid: dict(zip(g["segment_idx"].astype(int), g["segment_sv"].astype(float)))
            for sid, g in df.groupby("sample_id")
        }
        sgpa_all, rand_all = [], []
        for sid, cg in cdf.groupby("sample_id"):
            res = _sample_aopc(cg, sv_by_sample.get(int(sid), {}), grid)
            if res is None:
                continue
            sgpa_all.append(res[0])
            rand_all.append(res[1])
        if not sgpa_all:
            continue
        sgpa_m = np.mean(sgpa_all, axis=0)
        rand_m = np.mean(rand_all, axis=0)
        adf = pd.DataFrame(
            {"frac_removed": grid, "sgpa_order_drop": sgpa_m, "random_drop": rand_m}
        )
        adf.to_csv(cfg.out / f"{cfg.prefix}_aopc_{label}.csv", index=False)
        aopc_gap = float(np.trapezoid(sgpa_m - rand_m, grid))
        print(f"[aopc] {label}: AOPC gap (SGPA-random) = {aopc_gap:.4f}")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.plot(grid, sgpa_m, "o-", label="SGPA top-|SV| order")
            ax.plot(grid, rand_m, "s--", label="random removal")
            ax.set_xlabel("fraction of words removed")
            ax.set_ylabel("response change (1 - cosine)")
            ax.set_title(f"AOPC — {cfg.prefix}/{label}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(cfg.out / f"{cfg.prefix}_aopc_{label}.png", dpi=150)
            plt.close(fig)
        except Exception:
            traceback.print_exc()


def stage3_and_masking(cfg: Cfg) -> None:
    def _row(label: str) -> dict[str, Any] | None:
        _, sm, _ = _load(cfg, cfg.dirmap[label])
        if not sm or sm.get("completed_samples", 0) == 0:
            return None
        tv = sm.get("top_vs_matched_random_emb") or {}
        return {
            "condition": label,
            "n": sm.get("completed_samples"),
            "top_drop_emb": round(sm.get("mean_top_drop_emb", float("nan")), 4),
            "dz_emb": round(tv.get("cohen_dz", float("nan")), 3),
            "within_spearman_median_emb": sm.get("within_sample_spearman_median_emb"),
        }

    s3 = [r for r in (_row("librispeech_original"), _row("stage3_off")) if r]
    if len(s3) == 2:
        for r, tag in zip(s3, ["refined", "raw_ctc"]):
            r["boundaries"] = tag
        pd.DataFrame(s3).to_csv(
            cfg.out / f"{cfg.prefix}_stage3_compare.csv", index=False
        )
        print(f"[stage3] -> {cfg.prefix}_stage3_compare.csv")

    mask = [
        r
        for r in (_row("librispeech_original"), _row("mask_noise"), _row("mask_concat"))
        if r
    ]
    if len(mask) >= 2:
        for r in mask:
            r.setdefault("mask_mode", r["condition"])
        pd.DataFrame(mask).to_csv(
            cfg.out / f"{cfg.prefix}_masking_ablation.csv", index=False
        )
        print(f"[masking] -> {cfg.prefix}_masking_ablation.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=["qwen", "voxtral"],
        default="qwen",
        help="Which model's runs to analyze (selects dir layout + file naming).",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Directory containing the run folders (default depends on --model).",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = _make_cfg(args.model, args.base, args.out)
    cfg.out.mkdir(parents=True, exist_ok=True)
    for fn in (
        main_faithfulness,
        voice_table,
        attribution_stats,
        aopc,
        stage3_and_masking,
    ):
        try:
            fn(cfg)
        except Exception:
            print(f"[WARN] {fn.__name__} failed:")
            traceback.print_exc()
    print(f"Analysis written to {cfg.out}")


if __name__ == "__main__":
    main()
