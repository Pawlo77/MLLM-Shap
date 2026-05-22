"""Tests for the summarize function that compiles results and computes power planning."""

import pandas as pd

from experiments.faithfulness.src.summarize import summarize


def _rows() -> pd.DataFrame:
    """Helper function to generate a DataFrame with test data for summarization tests.
    This creates 6 rows with incrementally increasing values
    for various metrics to simulate realistic results."""
    rows = []
    for idx in range(1, 7):
        rows.append({
            "sample_id": idx,
            "top_drop": 0.45 + idx * 0.01,
            "mean_random_drop": 0.30 + idx * 0.005,
            "drop_difference": 0.12 + idx * 0.01,
            "pos_stratified_drop_difference": 0.10 + idx * 0.009,
            "neg_drop_improvement": 0.08 + idx * 0.008,
            "neg_stratified_drop_improvement": 0.07 + idx * 0.007,
            "comprehensiveness_drop_difference": 0.14 + idx * 0.006,
            "comprehensiveness_stratified_drop_difference": 0.13 + idx * 0.005,
            "sufficiency_advantage": 0.09 + idx * 0.004,
            "sufficiency_stratified_advantage": 0.08 + idx * 0.004,
            "monotonicity_score": 0.20 + idx * 0.01,
            "tfidf_drop_difference": 0.04 + idx * 0.005,
            "tfidf_pos_stratified_drop_difference": 0.035 + idx * 0.005,
            "tfidf_neg_drop_improvement": 0.03 + idx * 0.004,
            "tfidf_neg_stratified_drop_improvement": 0.025 + idx * 0.004,
            "tfidf_comprehensiveness_drop_difference": 0.05 + idx * 0.004,
            "tfidf_comprehensiveness_stratified_drop_difference": 0.045 + idx * 0.004,
            "tfidf_sufficiency_advantage": 0.03 + idx * 0.003,
            "tfidf_sufficiency_stratified_advantage": 0.028 + idx * 0.003,
            "tfidf_monotonicity_score": 0.10 + idx * 0.01,
            "seqmatch_drop_difference": 0.07 + idx * 0.006,
            "seqmatch_neg_drop_improvement": 0.05 + idx * 0.005,
            "seqmatch_comprehensiveness_drop_difference": 0.08 + idx * 0.005,
            "seqmatch_sufficiency_advantage": 0.04 + idx * 0.004,
            "seqmatch_monotonicity_score": 0.15 + idx * 0.008,
            "runtime_sec": 1.0 + idx * 0.1,
        })
    return pd.DataFrame(rows)


def test_summarize_includes_new_tests_and_power_block() -> None:
    """Test that the summarize function correctly processes the input DataFrame, computes new test statistics,
    and includes a power planning block in the output summary."""
    results_df = _rows()
    summary = summarize(results_df, pd.DataFrame(), target_effect_size_dz=0.4)

    assert summary["completed_samples"] == 6
    assert "tests" in summary
    assert "pos_uniform" in summary["tests"]
    assert "neg_uniform" in summary["tests"]
    assert "comprehensiveness_uniform" in summary["tests"]
    assert "sufficiency_uniform" in summary["tests"]
    assert "monotonicity_embedding" in summary["tests"]
    assert "tfidf_pos_uniform" in summary["tests"]
    assert "seqmatch_pos_uniform" in summary["tests"]
    assert "seqmatch_neg_uniform" in summary["tests"]
    assert "seqmatch_comprehensiveness_uniform" in summary["tests"]
    assert "seqmatch_sufficiency_uniform" in summary["tests"]
    assert "seqmatch_monotonicity" in summary["tests"]

    pos_uniform = summary["tests"]["pos_uniform"]
    assert pos_uniform["n"] == 6
    assert pos_uniform["mean"] > 0.0
    assert pos_uniform["wilcoxon_p_value"] is not None
    assert "wilcoxon_p_value_bh_fdr" in pos_uniform
    assert "paired_t_p_value_bh_fdr" in pos_uniform

    power = summary["power_planning"]
    assert power["target_effect_size_dz"] == 0.4
    assert power["estimated_required_n"] is not None
    assert power["estimated_required_n"] > 0
    assert "n_tests" in power
    assert "family_corrected_alpha" in power
    assert "family_corrected_estimated_required_n" in power
    assert (
        power["family_corrected_estimated_required_n"] > power["estimated_required_n"]
    )
