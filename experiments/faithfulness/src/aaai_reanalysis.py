"""AAAI-27 re-analysis of existing rebuttal rankwise faithfulness outputs.

No new model inference: consumes the pre-computed
``combined_rankwise_results.csv`` files under
``experiments/faithfulness/outputs/rebuttal_new_data/rankwise/`` and produces:

A1. Held-out faithfulness utility -- top-vs-non-top deletion effect scored with
    an *embedding* utility (and sequence-match) side-by-side with the TF-IDF
    utility that was also used for the Shapley game (keeps the circularity
    caveat explicit by reporting TF-IDF next to the held-out metrics).

A2. AOPC / top-k removal curves -- area over the cumulative top-|SV| removal
    curve, with the mask fraction k/n on the x-axis, per condition and per
    utility. Also reports a saturation / attribution-flatness diagnostic that
    contextualises how informative the |SV| ordering is.

Outputs (tables as CSV + Markdown + JSON, plus figures) are written under
``experiments/faithfulness/outputs/aaai27_reanalysis/``.

Run from the repo root::

    PYTHONPATH="$PWD/mllm_shap/src" \
        .venv/bin/python -m experiments.faithfulness.src.aaai_reanalysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ─── Configuration ──────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
RANKWISE_ROOT = (
    REPO_ROOT / "experiments/faithfulness/outputs/rebuttal_new_data/rankwise"
)
OUTPUT_DIR = REPO_ROOT / "experiments/faithfulness/outputs/aaai27_reanalysis"

# label -> (condition dir, speech kind). The native `raw` run is excluded here
# (only 5 samples exist; it is the ~30 min/sample native-token baseline).
CONDITIONS: dict[str, tuple[str, str]] = {
    "SM2S-1k (male TTS)": ("stage1_1k_sgpa_male", "TTS"),
    "SM2F-1k (female TTS)": ("stage1_1k_sgpa_female", "TTS"),
    "SO2S-1k (orig TTS)": ("stage1_1k_sgpa_original", "TTS"),
    "SO2S-500 (LibriSpeech)": ("stage2_500_sgpa_original", "natural"),
}

# metric key -> (single-deletion column, cumulative column, pretty name)
METRICS: dict[str, tuple[str, str, str]] = {
    "embedding": ("deletion_drop", "cumulative_drop", "Embedding cosine (held-out)"),
    "tfidf": (
        "tfidf_deletion_drop",
        "tfidf_cumulative_drop",
        "TF-IDF cosine (SV utility)",
    ),
    "seqmatch": (
        "seqmatch_deletion_drop",
        "seqmatch_cumulative_drop",
        "Sequence match (held-out)",
    ),
}

EPS = 1e-9


# ─── Statistics helpers ─────────────────────────────────────────────────────


def _paired_t_greater(values: np.ndarray) -> tuple[float | None, float | None]:
    """One-sided paired t-test proxy against zero (direction of the mean)."""
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, 0.0):
        return None, None
    stat, p_two = stats.ttest_1samp(values, popmean=0.0)
    if not (np.isfinite(stat) and np.isfinite(p_two)):
        return None, None
    p_one = p_two / 2.0 if stat >= 0 else 1.0 - p_two / 2.0
    return float(stat), float(p_one)


def _cohen_dz(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    std = float(np.std(values, ddof=1))
    return float(np.mean(values) / (std + EPS))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    r = stats.spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else None


# ─── A1: held-out utility (top vs non-top deletion) ─────────────────────────


@dataclass
class UtilityRow:
    condition: str
    speech: str
    metric: str
    n_samples: int
    n_deletions: int
    mean_top_drop: float
    mean_non_top_drop: float
    mean_top_minus_non_top: float
    paired_t: float | None
    paired_p_one_sided: float | None
    cohen_dz: float | None
    spearman_abs_sv_vs_drop: float | None
    mean_within_sample_spearman: float | None


def _held_out_rows(df: pd.DataFrame, condition: str, speech: str) -> list[UtilityRow]:
    rows: list[UtilityRow] = []
    n_samples = int(df[["sample_id"]].drop_duplicates().shape[0])
    for metric, (single_col, _cum_col, _pretty) in METRICS.items():
        if single_col not in df.columns:
            continue
        top = df[df["segment_rank_abs_sv"] == 1]
        non_top = df[df["segment_rank_abs_sv"] > 1]

        # Per-sample paired difference: top-rank drop minus that sample's mean
        # non-top drop (matches the summarize_rankwise "top minus non-top" idea
        # but computed per sample so a paired test is valid).
        per_sample_diff: list[float] = []
        for sid, grp in df.groupby("sample_id"):
            t = grp.loc[grp["segment_rank_abs_sv"] == 1, single_col]
            nt = grp.loc[grp["segment_rank_abs_sv"] > 1, single_col]
            if t.empty or nt.empty:
                continue
            per_sample_diff.append(float(t.mean() - nt.mean()))
        diff_arr = np.asarray(per_sample_diff, dtype=float)

        t_stat, p_one = _paired_t_greater(diff_arr)

        within: list[float] = []
        for _sid, grp in df.groupby("sample_id"):
            r = _safe_spearman(
                grp["segment_abs_sv"].to_numpy(float),
                grp[single_col].to_numpy(float),
            )
            if r is not None:
                within.append(r)

        rows.append(
            UtilityRow(
                condition=condition,
                speech=speech,
                metric=metric,
                n_samples=n_samples,
                n_deletions=int(df[single_col].notna().sum()),
                mean_top_drop=float(top[single_col].mean()),
                mean_non_top_drop=float(non_top[single_col].mean()),
                mean_top_minus_non_top=float(np.mean(diff_arr))
                if diff_arr.size
                else float("nan"),
                paired_t=t_stat,
                paired_p_one_sided=p_one,
                cohen_dz=_cohen_dz(diff_arr),
                spearman_abs_sv_vs_drop=_safe_spearman(
                    df["segment_abs_sv"].to_numpy(float),
                    df[single_col].to_numpy(float),
                ),
                mean_within_sample_spearman=float(np.mean(within)) if within else None,
            )
        )
    return rows


# ─── A2: AOPC / top-k removal curves ────────────────────────────────────────


@dataclass
class AopcRow:
    condition: str
    speech: str
    metric: str
    n_samples: int
    aopc_sv_order: float  # area over cumulative top-|SV| removal curve (x=k/n)
    aopc_single_mean: float  # mean single-segment deletion drop (order-free ref)
    mean_final_cumulative_drop: float  # drop after removing all segments


def _sample_aopc(grp: pd.DataFrame, cum_col: str) -> float | None:
    """AOPC over the cumulative top-k curve with x = k/n (trapezoid on [0,1])."""
    g = grp.sort_values("cumulative_n_deleted")
    n = float(g["n_segments"].iloc[0]) if "n_segments" in g else float(len(g))
    if n <= 0:
        return None
    k = g["cumulative_n_deleted"].to_numpy(float)
    y = g[cum_col].to_numpy(float)
    mask = np.isfinite(k) & np.isfinite(y)
    k, y = k[mask], y[mask]
    if k.size == 0:
        return None
    # prepend the origin (0 fraction removed -> 0 drop) so the area is anchored
    xs = np.concatenate([[0.0], k / n])
    ys = np.concatenate([[0.0], y])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(ys, xs))


def _aopc_rows(
    df: pd.DataFrame, condition: str, speech: str
) -> tuple[list[AopcRow], dict]:
    rows: list[AopcRow] = []
    curves: dict[str, dict] = {}
    n_samples = int(df[["sample_id"]].drop_duplicates().shape[0])
    for metric, (single_col, cum_col, _pretty) in METRICS.items():
        if cum_col not in df.columns:
            continue
        per_sample_aopc = [
            a
            for _sid, grp in df.groupby("sample_id")
            if (a := _sample_aopc(grp, cum_col)) is not None
        ]
        rows.append(
            AopcRow(
                condition=condition,
                speech=speech,
                metric=metric,
                n_samples=n_samples,
                aopc_sv_order=float(np.mean(per_sample_aopc))
                if per_sample_aopc
                else float("nan"),
                aopc_single_mean=float(df[single_col].mean())
                if single_col in df
                else float("nan"),
                mean_final_cumulative_drop=float(
                    df.loc[
                        df["cumulative_n_deleted"] == df["n_segments"], cum_col
                    ].mean()
                ),
            )
        )
        # mean cumulative-drop vs fraction curve (binned) for plotting
        frac = df["cumulative_n_deleted"] / df["n_segments"]
        binned = (
            pd.DataFrame({"frac": frac, "y": df[cum_col]})
            .assign(bin=lambda d: (d["frac"] * 10).round() / 10)
            .groupby("bin")["y"]
            .agg(["mean", "sem", "size"])
        )
        curves[metric] = {
            "frac": binned.index.tolist(),
            "mean": binned["mean"].tolist(),
            "sem": binned["sem"].fillna(0.0).tolist(),
            "size": binned["size"].astype(int).tolist(),
        }
    return rows, curves


# ─── Attribution flatness diagnostic ────────────────────────────────────────


def _flatness_row(df: pd.DataFrame, condition: str, speech: str) -> dict:
    """How informative is the |SV| ranking? Reports flat/degenerate share."""
    per_sample_flat = []
    entropies = []
    for _sid, grp in df.groupby("sample_id"):
        sv = grp["segment_abs_sv"].to_numpy(float)
        per_sample_flat.append(int(np.unique(np.round(sv, 6)).size <= 1))
        if "abs_sv_entropy_norm" in grp:
            entropies.append(float(grp["abs_sv_entropy_norm"].iloc[0]))
    drop = df["deletion_drop"] if "deletion_drop" in df else pd.Series(dtype=float)
    return {
        "condition": condition,
        "speech": speech,
        "n_samples": int(df[["sample_id"]].drop_duplicates().shape[0]),
        "flat_abs_sv_sample_share": float(np.mean(per_sample_flat))
        if per_sample_flat
        else None,
        "mean_abs_sv_entropy_norm": float(np.mean(entropies)) if entropies else None,
        "saturation_frac_drop_ge_0_8": float((drop >= 0.8).mean())
        if not drop.empty
        else None,
    }


# ─── Orchestration ──────────────────────────────────────────────────────────


def _load_condition(dir_name: str) -> pd.DataFrame | None:
    path = RANKWISE_ROOT / dir_name / "combined_rankwise_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)
    )
    return "\n".join([header, sep, body])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    utility_rows: list[UtilityRow] = []
    aopc_rows: list[AopcRow] = []
    flatness_rows: list[dict] = []
    all_curves: dict[str, dict] = {}

    for label, (dir_name, speech) in CONDITIONS.items():
        df = _load_condition(dir_name)
        if df is None or df.empty:
            print(f"[skip] {label}: no data at {dir_name}")
            continue
        utility_rows.extend(_held_out_rows(df, label, speech))
        rows, curves = _aopc_rows(df, label, speech)
        aopc_rows.extend(rows)
        all_curves[label] = curves
        flatness_rows.append(_flatness_row(df, label, speech))
        print(f"[ok] {label}: {df['sample_id'].nunique()} samples, {len(df)} deletions")

    util_df = pd.DataFrame([asdict(r) for r in utility_rows])
    aopc_df = pd.DataFrame([asdict(r) for r in aopc_rows])
    flat_df = pd.DataFrame(flatness_rows)

    util_df.to_csv(OUTPUT_DIR / "a1_held_out_utility.csv", index=False)
    aopc_df.to_csv(OUTPUT_DIR / "a2_aopc.csv", index=False)
    flat_df.to_csv(OUTPUT_DIR / "a2_flatness_diagnostic.csv", index=False)
    (OUTPUT_DIR / "curves.json").write_text(json.dumps(all_curves, indent=2))

    md = [
        "# AAAI-27 re-analysis (no new inference)",
        "",
        "## A1. Held-out faithfulness utility (top-rank vs non-top deletion)",
        "",
        "Positive `mean_top_minus_non_top` means deleting the top-|SV| segment",
        "reduces response similarity more than the average non-top segment.",
        "TF-IDF is the utility used inside the Shapley game (circularity caveat);",
        "embedding and sequence-match are held-out utilities.",
        "",
        _df_to_markdown(util_df),
        "",
        "## A2. AOPC (area over cumulative top-|SV| removal curve, x = k/n)",
        "",
        _df_to_markdown(aopc_df),
        "",
        "## A2. Attribution-flatness / saturation diagnostic",
        "",
        _df_to_markdown(flat_df),
        "",
    ]
    (OUTPUT_DIR / "REANALYSIS.md").write_text("\n".join(md))

    print("\n=== A1 held-out utility ===")
    print(util_df.to_string(index=False))
    print("\n=== A2 AOPC ===")
    print(aopc_df.to_string(index=False))
    print("\n=== A2 flatness diagnostic ===")
    print(flat_df.to_string(index=False))
    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
