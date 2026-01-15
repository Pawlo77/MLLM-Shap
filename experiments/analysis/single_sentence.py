# pylint: disable=magic-value-comparison

"""Single sentence experiments analysis utilities."""

import json
import os
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from mllm_shap.connectors.enums import Role  # pylint: disable=wrong-import-order
from scipy import stats

nlp = spacy.load("en_core_web_sm")
sns.set_style("whitegrid")


EXPERIMENTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments_output")
)
"""Root directory for experiments output."""

SINGLE_SENTENCE_EXPERIMENTS_DIR: str = os.path.join(
    EXPERIMENTS_DIR, "single_sentence_2026_01_03"
)
"""Single sentence experiments outputs directory."""

CASES: dict[str, str] = {
    "T2T": "text_text_limited_neyman_lin3_0",
    "T2S": "text_audio_limited_neyman_lin3_0",
    "SM2T": "audio_male_text_limited_neyman_lin3_0",
    "SM2S": "audio_male_audio_limited_neyman_lin3_0",
    "SF2T": "audio_female_text_limited_neyman_lin3_0",
    "SF2S": "audio_female_audio_limited_neyman_lin3_0",
}
"""Mapping of experiment case codes to directory names."""


def load_experiments_results(case: str) -> pd.DataFrame:
    """Load experiment results for a given case."""
    case_dir = os.path.join(SINGLE_SENTENCE_EXPERIMENTS_DIR, CASES[case])
    samples_dir = os.path.join(case_dir, "samples")
    all_files = [
        os.path.join(samples_dir, f)
        for f in os.listdir(samples_dir)
        if f.endswith(".json")
    ]

    dts = []
    for f in all_files:
        with open(f, "r", encoding="utf-8") as file:
            dts.append(json.load(file))

    experiments_df = pd.DataFrame(dts)

    experiments_test_df = experiments_df.copy()

    experiments_test_df["n_turns"] = experiments_test_df["conversation"].apply(len)
    experiments_test_df["raw_system_prompt"] = experiments_test_df[
        "conversation"
    ].apply(lambda x: "".join(x[0][0]["content"]))
    experiments_test_df["system_prompt_roles"] = experiments_test_df[
        "conversation"
    ].apply(lambda x: x[0][0]["roles"])
    experiments_test_df["model_response_roles"] = experiments_test_df[
        "conversation"
    ].apply(lambda x: x[2][0]["roles"])

    if not (
        experiments_test_df["row_index"].nunique() == len(experiments_test_df)
        and experiments_test_df["raw_system_prompt"].nunique() == 1
        and experiments_test_df["n_turns"].value_counts().nunique() == 1
        and experiments_test_df["system_prompt_roles"]
        .apply(lambda roles: (np.array(roles) != Role.USER).all())
        .all()
        .item()
        and experiments_test_df["model_response_roles"]
        .apply(lambda roles: (np.array(roles) != Role.USER).all())
        .all()
        .item()
    ):
        raise RuntimeError("Experiment results integrity check failed.")

    return experiments_df


def check_roles(row: pd.Series, target_col: str = "prompt_text_roles") -> bool:
    """Any non-NaN shap values should correspond to USER roles."""
    roles = np.asarray(row[target_col])
    sv = np.asarray(row["sv"])

    if len(roles) != len(sv):
        print("Length mismatch between roles and shap values.")
        return False

    _mask = pd.isna(sv)
    return (roles[~_mask] == Role.USER).all()


def plot_token_count_distribution(
    plot_df: pd.DataFrame,
    column: str = "count_text_tokens",
    plot_ax: plt.Axes | None = None,
    hue: str | None = None,
) -> None:
    """Plot the distribution of text token counts."""
    if plot_ax is None:
        _, plot_ax = plt.subplots(1, 1, figsize=(8, 5))

    sns.histplot(
        data=plot_df,
        x=column,
        bins=plot_df[column].nunique(),
        # kde=True,
        palette="Set2",
        stat="percent",
        ax=plot_ax,
        hue=hue,
        multiple="dodge",
    )
    plot_ax.set_xlabel(f"Count of {column.replace('_', ' ').title()}", fontsize=12)
    plot_ax.set_ylabel("Percentage", fontsize=12)
    plot_ax.set_title(
        f"Distribution of {column.replace('_', ' ').title()} (%)",
        fontsize=14,
        fontweight="bold",
    )
    plot_ax.grid(axis="y", linestyle="--", alpha=0.7)
    sns.despine(ax=plot_ax)
    plt.tight_layout()


def get_linguistic_stats(prompt: str) -> dict[str, Any]:
    """Get linguistic stats for prompt using spacy."""
    nlp_prompt = nlp(prompt)

    return [
        {
            "text": t.text,
            "dep": t.dep_,
            "pos": t.pos_,
            "children": len(list(t.children)),
        }
        for t in nlp_prompt
    ]


def map_subwords_to_tokens(subwords: list[str], tokens_text: list[str]) -> list[int]:
    """Map subwords to token indices using cosine similarity of embeddings."""
    # Compute embeddings
    token_vecs = np.array([nlp(t.strip()).vector for t in tokens_text])
    subword_vecs = np.array([nlp(s.strip()).vector for s in subwords])

    # Normalize for cosine similarity
    token_vecs_norm = token_vecs / np.linalg.norm(token_vecs, axis=1, keepdims=True)
    subword_vecs_norm = subword_vecs / np.linalg.norm(
        subword_vecs, axis=1, keepdims=True
    )

    # Compute cosine similarity matrix (subwords x tokens)
    sim_matrix = np.dot(subword_vecs_norm, token_vecs_norm.T)

    # Map each subword to token with highest similarity
    mapping = np.argmax(sim_matrix, axis=1)

    return mapping.tolist()


def perform_ttest(comparison_data: pd.DataFrame) -> pd.DataFrame:
    """Perform paired t-tests between modes."""
    modes = comparison_data.columns.tolist()
    ttest_results = []
    for mode_a, mode_b in combinations(modes, 2):
        valid_pair = comparison_data[[mode_a, mode_b]].dropna()

        if len(valid_pair) > 1:
            t_stat, p_val = stats.ttest_rel(valid_pair[mode_a], valid_pair[mode_b])

            ttest_results.append(
                {
                    "Mode_1": mode_a,
                    "Mode_2": mode_b,
                    "T_Statistic": t_stat,
                    "P_Value": p_val,
                    "Degrees_of_Freedom": len(valid_pair) - 1,
                    "Sample_Size": len(valid_pair),
                    "Mean_Diff": (valid_pair[mode_a] - valid_pair[mode_b]).mean(),
                }
            )
        else:
            # Not enough overlapping data points for a t-test
            ttest_results.append(
                {
                    "Mode_1": mode_a,
                    "Mode_2": mode_b,
                    "T_Statistic": None,
                    "P_Value": None,
                    "Sample_Size": len(valid_pair),
                }
            )

    ttest_summary = pd.DataFrame(ttest_results)

    # Add a significance flag (Bonferroni correction)
    alpha = 0.05 / len(ttest_results) if len(ttest_results) > 0 else 0.05
    ttest_summary["Significant_Adj"] = ttest_summary["P_Value"] < alpha

    ttest_summary = ttest_summary[
        ~((ttest_summary["Mode_1"] == "S2S") & (ttest_summary["Mode_2"] == "SF2S"))
        & ~((ttest_summary["Mode_1"] == "S2S") & (ttest_summary["Mode_2"] == "SM2S"))
        & ~((ttest_summary["Mode_1"] == "S2T") & (ttest_summary["Mode_2"] == "SF2T"))
        & ~((ttest_summary["Mode_1"] == "S2T") & (ttest_summary["Mode_2"] == "SM2T"))
        & ~(
            (ttest_summary["Mode_1"] == "T2*")
            & (ttest_summary["Mode_2"].isin(["T2T", "T2S", "*2T", "*2S"]))
        )
        & ~(
            (ttest_summary["Mode_1"] == "S2*")
            & (
                ttest_summary["Mode_2"].isin(
                    ["S2T", "S2S", "SM2T", "SM2S", "SF2T", "SF2S", "*2T", "*2S"]
                )
            )
        )
        & ~(
            (ttest_summary["Mode_1"] == "*2T")
            & (
                ttest_summary["Mode_2"].isin(
                    ["T2T", "S2T", "SM2T", "SF2T", "T2*", "S2*"]
                )
            )
        )
        & ~(
            (ttest_summary["Mode_1"] == "*2S")
            & (
                ttest_summary["Mode_2"].isin(
                    ["T2S", "S2S", "SM2S", "SF2S", "T2*", "S2*"]
                )
            )
        )
    ]

    ttest_summary.set_index(["Mode_1", "Mode_2"], inplace=True)
    ttest_summary["P_Value"] = (
        ttest_summary["P_Value"]
        .round(2)
        .astype(str)
        .map(lambda x: "<0.01" if x == "0.0" else x)
    )
    ttest_summary = ttest_summary.drop("ALL", level=0)

    return ttest_summary


def interpolate_cumsum(values: list[float], target_len: int) -> list[float]:
    """Interpolate cumulative shap values to a target length."""
    values = np.asarray(values, dtype=float)
    n = len(values)

    # Only one point -> repeat it
    if n == 1:
        return values.repeat(target_len).tolist()

    # Original and target normalized positions
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)

    # Linear interpolation
    y = np.interp(x_new, x_old, values)

    # Ensure cumulative (monotonic increasing)
    y = np.maximum.accumulate(y)

    # Renormalize so final value matches the original cumsum
    if y[-1] > 0:
        y *= values[-1] / y[-1]

    return y.tolist()


def normalize_lengths(group: pd.DataFrame, col="sv_cumsum") -> pd.DataFrame:
    """Normalize lengths of cumulative shap values within a group."""
    target_len = group[col].apply(len).max()
    group[col] = group[col].apply(lambda v: interpolate_cumsum(v, target_len))
    return group


def plot_dist(
    df: pd.DataFrame, hue_col: str, modes: str, hspace: float = -0.4, sv_col: str = "sv"
) -> sns.displot:
    """Create a distribution plot of shapley values for specified modes."""

    g = sns.displot(
        data=df[df["mode"].isin(modes)],
        x=sv_col,
        hue=hue_col,
        row="mode",
        row_order=modes,
        kind="kde",
        palette="Set2",
        fill=True,
        alpha=0.2,
        aspect=4,
        height=1.5,
        common_norm=False,
        facet_kws={"margin_titles": True},
    )

    g.set_titles(row_template="{row_name}", size=12, fontweight="bold")
    g.set(yticks=[], ylabel="")
    g.despine(left=True)
    g.figure.subplots_adjust(hspace=hspace)
    g.set_axis_labels("Shapley Value", "")


def plot_attribution_trend(df: pd.DataFrame, n_bins: int = 100) -> None:
    """Plot attribution trend by sentence position."""
    df = df.copy()
    df["pos_bin"] = pd.cut(df["normalized_position"], bins=n_bins, labels=False)
    # Convert bin index back to 0-1 scale for plotting
    df["bin_center"] = (df["pos_bin"] + 0.5) / n_bins
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x="bin_center",
        y="sv",
        hue="mode",
        palette="Set2",
        errorbar=None,
        linewidth=2.5,
        alpha=0.85,
    )

    plt.title(
        "Attribution Trend by Sentence Position", fontsize=14, fontweight="bold", pad=20
    )
    plt.ylabel("Mean Shapley Value", fontsize=12)
    plt.xlabel("Relative Sentence Position", fontsize=12)

    plt.xticks([0, 0.5, 1.0], ["Start\n(0%)", "Middle\n(50%)", "End\n(100%)"])
    plt.xlim(0, 1)
    sns.despine(trim=True)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, title="Mode"
    )
    plt.tight_layout()
