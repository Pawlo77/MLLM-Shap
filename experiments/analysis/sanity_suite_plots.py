"""Matplotlib-only diagnostic plots for the sanity/insight test suite."""

# pylint: disable=too-many-locals

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sanity_suite_core import bootstrap_ci


_MIN_POINTS_FOR_FIT = 3
_WORD_ENTROPY_COL = "shapley_entropy_word_current"
_WORD_ENTROPY_COL_SUFFIXED = "shapley_entropy_word_current_word"


def plot_entropy_vs_length(sample_tok: pd.DataFrame, outpath: str) -> None:
    """Scatter: entropy vs token length by language."""

    df = sample_tok.dropna(subset=["shapley_entropy_text_current", "token_count_text_current", "language"]).copy()
    if df.empty:
        print("No data for entropy vs length plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for lang in sorted(df["language"].dropna().unique().tolist()):
        group = df[df["language"].astype(str) == str(lang)]
        ax.scatter(
            group["token_count_text_current"],
            group["shapley_entropy_text_current"],
            s=12,
            alpha=0.6,
            label=str(lang),
        )
        x = group["token_count_text_current"].to_numpy(dtype=float)
        y = group["shapley_entropy_text_current"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(mask)) >= _MIN_POINTS_FOR_FIT:
            b1, b0 = np.polyfit(x[mask], y[mask], deg=1)
            xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 50)
            ax.plot(xs, b1 * xs + b0, linewidth=2)

    ax.set_title("Entropy vs token_count (current text)")
    ax.set_xlabel("token_count_text_current")
    ax.set_ylabel("shapley_entropy_text_current (base 2)")
    ax.legend(title="language")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.show()


def plot_retention_all_vs_textonly(sample_tok: pd.DataFrame, outpath: str) -> None:
    """Scatter: retention with audio-in-denominator vs text-only denominator."""

    df = sample_tok.dropna(subset=["history_retention_pct_all", "history_retention_pct_textonly", "case"]).copy()
    if df.empty:
        print("No data for retention plot")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    for case in sorted(df["case"].astype(str).unique().tolist()):
        group = df[df["case"].astype(str) == str(case)]
        ax.scatter(
            group["history_retention_pct_all"],
            group["history_retention_pct_textonly"],
            s=12,
            alpha=0.5,
            label=str(case),
        )

    ax.plot([0, 100], [0, 100], linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Retention: all-denominator vs text-only")
    ax.set_xlabel("history_retention_pct_all")
    ax.set_ylabel("history_retention_pct_textonly")
    ax.legend(title="case")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.show()


def plot_pos_shares_with_ci(
    sample_tok: pd.DataFrame,
    outpath: str,
    pos_list: Sequence[str] = ("Noun", "Verb", "Adj"),
) -> None:
    """Bar plot: POS shares by language with bootstrap CI."""

    df = sample_tok.dropna(subset=["language"]).copy()
    if df.empty:
        print("No data for POS shares plot")
        return

    langs = sorted(df["language"].dropna().unique().tolist())
    cmap = plt.cm.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(pos_list))
    x0 = np.arange(len(langs))

    for j, pos in enumerate(pos_list):
        col = f"pos_share_{pos}"
        means, yerr_lo, yerr_hi = [], [], []
        for lang in langs:
            v = df[df["language"].astype(str) == str(lang)][col].to_numpy(dtype=float)
            est, lo, hi = bootstrap_ci(v, np.nanmean)
            means.append(est)
            yerr_lo.append(est - lo)
            yerr_hi.append(hi - est)
        ax.bar(x0 + j * width, means, width=width, label=pos, yerr=[yerr_lo, yerr_hi], capsize=3, color=cmap(j))

    ax.set_xticks(x0 + width * (len(pos_list) - 1) / 2)
    ax.set_xticklabels(langs)
    ax.set_title("POS shares (current text) with bootstrap 95% CI")
    ax.set_ylabel("share of current-text abs SV")
    ax.legend(title="POS")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.show()


def plot_hubness_real_vs_null(null_df: pd.DataFrame, outpath: str) -> None:
    """Boxplot: observed hubness vs null mean by case."""

    if null_df is None or null_df.empty:
        print("No data for hubness null plot")
        return

    df = null_df.copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    cases = sorted(df["case"].astype(str).unique().tolist())
    obs = [df[df["case"].astype(str) == c]["hubness_observed"].to_numpy(dtype=float) for c in cases]
    nul = [df[df["case"].astype(str) == c]["hubness_null_mean"].to_numpy(dtype=float) for c in cases]
    pos_obs = np.arange(len(cases)) * 2.0
    pos_nul = pos_obs + 0.7

    ax.boxplot(obs, positions=pos_obs, widths=0.6, patch_artist=True, boxprops={"facecolor": "lightblue"})
    ax.boxplot(nul, positions=pos_nul, widths=0.6, patch_artist=True, boxprops={"facecolor": "lightgray"})

    ax.set_xticks(pos_obs + 0.35)
    ax.set_xticklabels(cases)
    ax.set_title("Hubness: observed vs null (Dirichlet baseline)")
    ax.set_ylabel("hubness")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.show()


def plot_sensitivity_entropy_token_vs_word(sample_tok: pd.DataFrame, sample_word: pd.DataFrame, outpath: str) -> None:
    """Scatter: token-level vs word-level entropy by language."""

    if sample_word is None or sample_word.empty or _WORD_ENTROPY_COL not in sample_word.columns:
        print("No data for sensitivity plot (missing word entropy)")
        return

    right = sample_word[["sample_id", "case", "language", _WORD_ENTROPY_COL]].copy()
    df = sample_tok.merge(right, on=["sample_id", "case", "language"], how="left", suffixes=("_tok", "_word"))

    word_col = _WORD_ENTROPY_COL
    if _WORD_ENTROPY_COL_SUFFIXED in df.columns:
        word_col = _WORD_ENTROPY_COL_SUFFIXED

    df = df.dropna(subset=["shapley_entropy_text_current", word_col, "language"])
    if df.empty:
        print("No data for sensitivity plot")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    for lang in sorted(df["language"].astype(str).unique().tolist()):
        group = df[df["language"].astype(str) == str(lang)]
        ax.scatter(group["shapley_entropy_text_current"], group[word_col], s=12, alpha=0.6, label=str(lang))

    lo = float(np.nanmin([df["shapley_entropy_text_current"].min(), df[word_col].min()]))
    hi = float(np.nanmax([df["shapley_entropy_text_current"].max(), df[word_col].max()]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set_title("Sensitivity: entropy token-level vs word-level")
    ax.set_xlabel("token-level entropy")
    ax.set_ylabel("word-level entropy")
    ax.legend(title="language")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.show()
