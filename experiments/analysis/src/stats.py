"""Statistical analysis functions for experiments."""

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from .transformations import calculate_normalized_entropy

AGG_DICT: dict[str, Any] = {
    "mean": "mean",
    "gini": lambda x: calculate_gini(x),
    "top_20_mass": lambda x: calculate_top_k_mass(x, 0.2),
    "max": "max",
    "entropy": calculate_normalized_entropy,
}
"""Aggregation functions for statistical analysis."""


def perform_ttest(comparison_data: pd.DataFrame) -> pd.DataFrame:
    """Perform paired t-tests between modes."""
    modes = comparison_data.columns.tolist()
    ttest_results = []

    for mode_a, mode_b in combinations(modes, 2):
        valid_pair = comparison_data[[mode_a, mode_b]].dropna()

        if len(valid_pair) > 1 and not np.allclose(
            valid_pair[mode_a], valid_pair[mode_b]
        ):
            diff = valid_pair[mode_a] - valid_pair[mode_b]
            kurtosis = stats.kurtosis(diff)
            _, p_norm = stats.shapiro(diff)
            t_stat, p_val = stats.ttest_rel(valid_pair[mode_a], valid_pair[mode_b])

            diffs = valid_pair[mode_a] - valid_pair[mode_b]
            mean_diff = np.mean(diffs)
            sd_diff = np.std(diffs, ddof=1)
            cohens_dz = mean_diff / sd_diff

            ttest_results.append({
                "Mode_1": mode_a,
                "Mode_2": mode_b,
                "T_Statistic": t_stat,
                "P_Value": p_val,
                "Degrees_of_Freedom": len(valid_pair) - 1,
                "Sample_Size": len(valid_pair),
                "Mean_Diff": (valid_pair[mode_a] - valid_pair[mode_b]).mean(),
                "Kurtosis_Diff": kurtosis,
                "Shapiro_P": p_norm,
                "Cohens_dz": cohens_dz.round(2),
            })
        else:
            continue
            # print(
            #     f"Skipping t-test for {mode_a} vs {mode_b} due to insufficient or identical data."
            # )

    results_df = pd.DataFrame(ttest_results)
    if results_df.empty:
        return results_df

    valid_tests_count = results_df["P_Value"].notna().sum()

    # Add a significance flag (Bonferroni correction)
    if valid_tests_count > 0:
        results_df["Significant_Adj"] = results_df["P_Value"] < (
            0.05 / valid_tests_count
        )
    else:
        results_df["Significant_Adj"] = False
    results_df["Shapiro_Significant"] = results_df["Shapiro_P"] < 0.05

    # Helper function to get the first element if it's a tuple, otherwise return the value itself
    def get_base_mode(val):
        return val[0] if isinstance(val, tuple) else val

    m1 = results_df["Mode_1"].apply(get_base_mode)
    m2 = results_df["Mode_2"].apply(get_base_mode)

    results_df = results_df[
        ~((m1 == "S2S") & (m2 == "SF2S"))
        & ~((m1 == "S2S") & (m2 == "SM2S"))
        & ~((m1 == "S2T") & (m2 == "SF2T"))
        & ~((m1 == "S2T") & (m2 == "SM2T"))
        & ~((m1 == "T2*") & (m2.isin(["T2T", "T2S", "*2T", "*2S"])))
        & ~(
            (m1 == "S2*")
            & (m2.isin(["S2T", "S2S", "SM2T", "SM2S", "SF2T", "SF2S", "*2T", "*2S"]))
        )
        & ~((m1 == "*2T") & (m2.isin(["T2T", "S2T", "SM2T", "SF2T", "T2*", "S2*"])))
        & ~((m1 == "*2S") & (m2.isin(["T2S", "S2S", "SM2S", "SF2S", "T2*", "S2*"])))
    ]

    results_df.set_index(["Mode_1", "Mode_2"], inplace=True)
    results_df["P_Value"] = (
        results_df["P_Value"]
        .round(2)
        .astype(str)
        .map(lambda x: "<0.01" if x == "0.0" else x)
    )
    if isinstance(results_df.index, pd.MultiIndex):
        results_df = results_df[
            results_df.index.get_level_values(0).map(lambda x: x[0] != "ALL")
        ]

    return results_df


def calculate_gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # and: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.abs(np.array(array, dtype=float))
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def calculate_top_k_mass(array, k_fraction=0.2):
    """Calculate the fraction of total absolute mass held by the top k% of elements."""
    array = np.abs(np.array(array, dtype=float))
    if len(array) == 0:
        return 0.0

    total_mass = np.sum(array)
    if total_mass == 0:
        return 0.0

    sorted_array = np.sort(array)[::-1]  # descending
    n_top = max(1, int(np.ceil(len(array) * k_fraction)))
    top_mass = np.sum(sorted_array[:n_top])

    return top_mass / total_mass


def perform_advanced_ttest(
    source_df: pd.DataFrame,
    grp_by_cols: list[str],
    value_col: str = "sv",
    unstack: list[str] | None = None,
) -> pd.DataFrame:
    """Perform t-tests within groups defined by grp_by_cols."""
    results_list = []
    grouped = source_df.groupby(grp_by_cols)[value_col]

    for agg_name, agg_func in AGG_DICT.items():
        agg_df = (
            grouped.agg(agg_func)
            .rename(agg_name)
            .unstack(level=unstack or grp_by_cols[1:])
        )
        ttest_result = perform_ttest(agg_df)
        ttest_result["metric"] = agg_name
        results_list.append(ttest_result)

    combined_results = pd.concat(results_list, ignore_index=False)
    return combined_results.reset_index()
