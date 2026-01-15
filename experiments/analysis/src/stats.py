# pylint: disable=magic-value-comparison
"""Statistical analysis functions for experiments."""

from itertools import combinations
from typing import Any
from functools import lru_cache

import numpy as np
import pandas as pd
import spacy
from scipy import stats


@lru_cache(maxsize=3)
def get_nlp(language: str):
    """Load the spaCy language model based on the specified language."""
    if language == "en":
        return spacy.load("en_core_web_sm")
    if language == "fr":
        return spacy.load("fr_core_news_sm")
    if language == "es":
        return spacy.load("es_core_news_sm")
    raise ValueError(f"Unsupported language: {language}")


def get_linguistic_stats(prompt: str, language: str = "en") -> dict[str, Any]:
    """Get linguistic stats for prompt using spacy."""
    nlp_prompt = get_nlp(language)(prompt)

    return [
        {
            "text": t.text,
            "dep": t.dep_,
            "pos": t.pos_,
            "children": len(list(t.children)),
        }
        for t in nlp_prompt
    ]


def map_subwords_to_tokens(
    subwords: list[str], tokens_text: list[str], language: str = "en"
) -> list[int]:
    """Map subwords to token indices using cosine similarity of embeddings."""
    nlp = get_nlp(language=language)

    # Compute embeddings
    processed_tokens = [nlp(t.strip()) for t in tokens_text]
    token_vecs = np.array(
        [
            pt.vector if pt.vector.shape[0] > 0 else np.zeros(96) + 1e-9
            for pt in processed_tokens
        ]
    )
    processed_subwords = [nlp(s.strip()) for s in subwords]
    subword_vecs = np.array(
        [
            ps.vector if ps.vector.shape[0] > 0 else np.zeros(96) + 1e-9
            for ps in processed_subwords
        ]
    )

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

        if len(valid_pair) > 1 and not np.allclose(
            valid_pair[mode_a], valid_pair[mode_b]
        ):
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
            print(f"Skipping t-test for {mode_a} vs {mode_b} due to insufficient or identical data.")

    results_df = pd.DataFrame(ttest_results)
    valid_tests_count = results_df["P_Value"].notna().sum()

    # Add a significance flag (Bonferroni correction)
    if valid_tests_count > 0:
        results_df["Significant_Adj"] = results_df["P_Value"] < (
            0.05 / valid_tests_count
        )
    else:
        results_df["Significant_Adj"] = False

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
        & ~(
            (m1 == "T2*")
            & (m2.isin(["T2T", "T2S", "*2T", "*2S"]))
        )
        & ~(
            (m1 == "S2*")
            & (
                m2.isin(
                    ["S2T", "S2S", "SM2T", "SM2S", "SF2T", "SF2S", "*2T", "*2S"]
                )
            )
        )
        & ~(
            (m1 == "*2T")
            & (m2.isin(["T2T", "S2T", "SM2T", "SF2T", "T2*", "S2*"]))
        )
        & ~(
            (m1 == "*2S")
            & (m2.isin(["T2S", "S2S", "SM2S", "SF2S", "T2*", "S2*"]))
        )
    ]

    results_df.set_index(["Mode_1", "Mode_2"], inplace=True)
    results_df["P_Value"] = (
        results_df["P_Value"]
        .round(2)
        .astype(str)
        .map(lambda x: "<0.01" if x == "0.0" else x)
    )
    if isinstance(results_df.index, pd.MultiIndex):
        results_df = results_df[results_df.index.get_level_values(0).map(lambda x: x[0] != "ALL")]

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
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def calculate_top_k_mass(array, k_fraction=0.2):
    """Calculate the fraction of total absolute mass held by the top k% of elements."""
    array = np.abs(np.array(array, dtype=float))
    if len(array) == 0:
        return 0.0
    
    total_mass = np.sum(array)
    if total_mass == 0:
        return 0.0
        
    sorted_array = np.sort(array)[::-1] # descending
    n_top = max(1, int(np.ceil(len(array) * k_fraction)))
    top_mass = np.sum(sorted_array[:n_top])
    
    return top_mass / total_mass
