"""Summarization and partition-combining functions."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .helpers import EPS


def summarize(results_df: pd.DataFrame, failures_df: pd.DataFrame) -> dict[str, Any]:
    if results_df.empty:
        return {"completed_samples": 0, "failed_samples": int(len(failures_df))}
    top = results_df["top_drop"].to_numpy(dtype=float)
    rand = results_df["mean_random_drop"].to_numpy(dtype=float)
    diff = top - rand
    has_pairs = len(results_df) >= 2
    t_stat, p_val = stats.ttest_rel(top, rand) if has_pairs else (None, None)
    return {
        "completed_samples": int(len(results_df)),
        "failed_samples": int(len(failures_df)),
        "mean_top_drop": float(np.mean(top)),
        "std_top_drop": float(np.std(top, ddof=1)) if has_pairs else None,
        "mean_random_drop": float(np.mean(rand)),
        "std_random_drop": float(np.std(rand, ddof=1)) if has_pairs else None,
        "mean_drop_difference": float(np.mean(diff)),
        "median_drop_difference": float(np.median(diff)),
        "paired_t_stat": float(t_stat) if t_stat is not None else None,
        "paired_p_value": float(p_val) if p_val is not None else None,
        "cohen_dz": float(np.mean(diff) / (np.std(diff, ddof=1) + EPS))
        if has_pairs
        else None,
        "top_greater_than_random_rate": float(np.mean(diff > 0)),
        "mean_runtime_sec": float(results_df["runtime_sec"].mean()),
    }


def _safe_spearman(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return None
    corr = stats.spearmanr(x.to_numpy(dtype=float), y.to_numpy(dtype=float)).statistic
    return float(corr) if math.isfinite(float(corr)) else None


def summarize_rankwise(
    results_df: pd.DataFrame, failures_df: pd.DataFrame
) -> dict[str, Any]:
    if results_df.empty:
        return {
            "completed_deletions": 0,
            "completed_samples": 0,
            "failed_samples": int(len(failures_df)),
        }

    per_rank = (
        results_df.groupby("segment_rank_abs_sv", as_index=True)
        .agg(
            n=("deletion_drop", "size"),
            mean_drop=("deletion_drop", "mean"),
            median_drop=("deletion_drop", "median"),
            mean_abs_sv_share=("segment_abs_sv_share", "mean"),
            mean_abs_sv=("segment_abs_sv", "mean"),
        )
        .sort_index()
    )
    per_rank_summary = {
        str(int(rank)): {k: float(v) for k, v in row.items()}
        for rank, row in per_rank.iterrows()
    }

    top_rows = results_df[results_df["segment_rank_abs_sv"] == 1]
    non_top_rows = results_df[results_df["segment_rank_abs_sv"] > 1]
    sample_count = int(
        results_df[["audio_column", "sample_id"]].drop_duplicates().shape[0]
    )

    per_sample_corrs: list[float] = []
    for _, group in results_df.groupby(["audio_column", "sample_id"]):
        corr = _safe_spearman(group["segment_abs_sv"], group["deletion_drop"])
        if corr is not None:
            per_sample_corrs.append(corr)

    global_corr = _safe_spearman(
        results_df["segment_abs_sv"], results_df["deletion_drop"]
    )
    global_rank_corr = _safe_spearman(
        -results_df["segment_rank_abs_sv"], results_df["deletion_drop"]
    )
    top_minus = (
        float(top_rows["deletion_drop"].mean() - non_top_rows["deletion_drop"].mean())
        if not top_rows.empty and not non_top_rows.empty
        else None
    )

    return {
        "completed_deletions": int(len(results_df)),
        "completed_samples": sample_count,
        "failed_samples": int(len(failures_df)),
        "mean_deletion_drop": float(results_df["deletion_drop"].mean()),
        "mean_top_rank_drop": float(top_rows["deletion_drop"].mean())
        if not top_rows.empty
        else None,
        "mean_non_top_rank_drop": float(non_top_rows["deletion_drop"].mean())
        if not non_top_rows.empty
        else None,
        "mean_top_minus_non_top_drop": top_minus,
        "spearman_abs_sv_vs_drop": global_corr,
        "spearman_negative_rank_vs_drop": global_rank_corr,
        "mean_within_sample_spearman_abs_sv_vs_drop": float(np.mean(per_sample_corrs))
        if per_sample_corrs
        else None,
        "median_within_sample_spearman_abs_sv_vs_drop": float(
            np.median(per_sample_corrs)
        )
        if per_sample_corrs
        else None,
        "within_sample_spearman_n": len(per_sample_corrs),
        "mean_top1_share": float(
            results_df[["audio_column", "sample_id", "top1_share"]]
            .drop_duplicates()["top1_share"]
            .mean()
        ),
        "mean_top1_top2_gap": float(
            results_df[["audio_column", "sample_id", "top1_top2_gap"]]
            .drop_duplicates()["top1_top2_gap"]
            .dropna()
            .mean()
        ),
        "mean_abs_sv_entropy_norm": float(
            results_df[["audio_column", "sample_id", "abs_sv_entropy_norm"]]
            .drop_duplicates()["abs_sv_entropy_norm"]
            .dropna()
            .mean()
        ),
        "per_rank": per_rank_summary,
        "mean_runtime_sec_per_deletion": float(results_df["runtime_sec"].mean()),
    }


def combine_partition_outputs(
    output_dir: Path, all_rank_deletions: bool = False
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    combined_frames: list[pd.DataFrame] = []
    combined_failure_frames: list[pd.DataFrame] = []
    tag = "rankwise_results" if all_rank_deletions else "results"
    summary_tag = "rankwise_summary" if all_rank_deletions else "summary"

    for audio_column in ("audio__male", "audio__female"):
        paths = sorted(output_dir.glob(f"{audio_column}_part*-of*_{tag}.csv"))
        if not paths:
            continue
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        dedup_cols = (
            ["sample_id", "segment_idx"] if all_rank_deletions else ["sample_id"]
        )
        sort_cols = (
            ["sample_id", "segment_rank_abs_sv", "segment_idx"]
            if all_rank_deletions
            else ["sample_id"]
        )
        df = df.drop_duplicates(subset=dedup_cols, keep="last").sort_values(sort_cols)

        result_path = output_dir / f"{audio_column}_combined_{tag}.csv"
        df.to_csv(result_path, index=False)

        failures_paths = sorted(
            output_dir.glob(f"{audio_column}_part*-of*_failures.csv")
        )
        failures_df = (
            pd.concat([pd.read_csv(p) for p in failures_paths], ignore_index=True)
            if failures_paths
            else pd.DataFrame()
        )
        if not failures_df.empty:
            failures_df = failures_df.drop_duplicates(
                subset=["sample_id", "audio_column", "error_type", "error_message"],
                keep="last",
            )
            combined_failure_frames.append(failures_df)

        summary = (
            summarize_rankwise(df, failures_df)
            if all_rank_deletions
            else summarize(df, failures_df)
        )
        summary.update({"audio_column": audio_column, "results_csv": str(result_path)})
        (output_dir / f"{audio_column}_combined_{summary_tag}.json").write_text(
            json.dumps(summary, indent=2)
        )
        summaries[audio_column] = summary
        combined_frames.append(df)

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        sort_cols = (
            ["audio_column", "sample_id", "segment_rank_abs_sv", "segment_idx"]
            if all_rank_deletions
            else ["audio_column", "sample_id"]
        )
        combined_df = combined_df.sort_values(sort_cols)
        combined_name = (
            "combined_rankwise_results.csv"
            if all_rank_deletions
            else "combined_results.csv"
        )
        combined_path = output_dir / combined_name
        combined_df.to_csv(combined_path, index=False)

        combined_failures = (
            pd.concat(combined_failure_frames, ignore_index=True)
            if combined_failure_frames
            else pd.DataFrame()
        )
        combined_summary = (
            summarize_rankwise(combined_df, combined_failures)
            if all_rank_deletions
            else summarize(combined_df, combined_failures)
        )
        combined_summary["results_csv"] = str(combined_path)
        summary_name = (
            "combined_rankwise_summary.json"
            if all_rank_deletions
            else "combined_summary.json"
        )
        (output_dir / summary_name).write_text(json.dumps(combined_summary, indent=2))
        summaries["combined"] = combined_summary

    return summaries
