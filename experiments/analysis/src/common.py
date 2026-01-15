"""Common utilities for experiments analysis."""

import json
import os

import numpy as np
import pandas as pd
from mllm_shap.connectors.enums import Role  # pylint: disable=wrong-import-order

EXPERIMENTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "experiments_output")
)
"""Root directory for experiments output."""

FIGURES_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures")
)
"""Root directory for figures."""

STATS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stats"))
"""Root directory for stats."""


def load_experiments_results(
    case: str,
    cases: dict[str, str],
    experiments_dir: str,
    is_multi_lingual: bool = False,
) -> pd.DataFrame:
    """Load experiment results for a given case."""
    case_dir = os.path.join(experiments_dir, cases[case])
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

    # TODO: fix it in multi-lingual dataset
    if is_multi_lingual:
        mask = experiments_test_df["n_turns"] == 3  # pylint: disable=magic-value-comparison
        experiments_df = experiments_df[mask]
        experiments_test_df = experiments_test_df[mask]

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
