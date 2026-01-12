"""End-to-end runner for the multilingual sanity + insight test suite."""

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=magic-value-comparison

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from sanity_suite_core import (
    ColumnMap,
    DEFAULT_WORD_MODE,
    FIG_DIR,
    adapt_token_level_raw,
    adapt_units_level_raw,
    add_word_features,
    build_findings,
    compute_bridge_metrics,
    compute_sample_metrics,
    load_data,
    null_bridge_hubness,
    permutation_test_pos_share,
    test_case_effect_controlling_length,
    test_denominator_artifact,
)
from sanity_suite_plots import (
    plot_entropy_vs_length,
    plot_hubness_real_vs_null,
    plot_pos_shares_with_ci,
    plot_retention_all_vs_textonly,
    plot_sensitivity_entropy_token_vs_word,
)
from sanity_suite_pos import ensure_pos_tags


def run_suite(*, save_figures: bool = True) -> Dict[str, Any]:
    """Run the suite top-to-bottom and return a dict of intermediate tables."""

    os.makedirs(FIG_DIR, exist_ok=True)

    raw = load_data()

    # Supported loader contracts:
    # - {'raw_units': units_df} (repo loaders)
    # - (df_raw, ColumnMap) (manual/custom)
    if isinstance(raw, dict):
        if "raw_units" not in raw:
            raise ValueError("load_data() dict must contain key 'raw_units'")
        df_raw = raw["raw_units"]
        colmap = ColumnMap()
    elif isinstance(raw, (tuple, list)) and len(raw) == 2:
        df_raw = raw[0]
        colmap = raw[1]
    else:
        raise ValueError("load_data() must return either {'raw_units': df} or (df_raw, ColumnMap)")

    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(f"load_data() returned non-DataFrame df_raw: {type(df_raw)}")

    # Heuristic: decide adapter by presence of token arrays in units schema
    if (
        colmap.tokens_list in df_raw.columns
        and len(df_raw) > 0
        and isinstance(df_raw[colmap.tokens_list].iloc[0], (list, tuple))
    ):
        df_tokens = adapt_units_level_raw(df_raw, colmap)
    else:
        df_tokens = adapt_token_level_raw(df_raw, colmap)

    df_tokens = add_word_features(df_tokens, merge_mode=DEFAULT_WORD_MODE)
    df_tokens = ensure_pos_tags(df_tokens)

    sample_tok = compute_sample_metrics(df_tokens, word_level=False)
    sample_word = compute_sample_metrics(df_tokens, word_level=True)

    print("token-level samples:", sample_tok.shape, "word-level samples:", sample_word.shape)

    print("\n[Sanity] Denominator artifact")
    denom = test_denominator_artifact(sample_tok, by="case")
    if denom is not None and not denom.empty:
        print(denom.to_string(index=False))

    print("\n[Sanity] Length control: case effects")
    candidates = [
        ("shapley_entropy_text_current", "token_count_text_current"),
        ("shapley_entropy_audio_current", "token_count_audio_current"),
        ("shapley_entropy_all_current", "token_count_all_current"),
    ]
    case_len_parts = []
    for met, lcol in candidates:
        if met not in sample_tok.columns or lcol not in sample_tok.columns:
            continue
        d0 = sample_tok.dropna(subset=[met, lcol, "case"]).copy()
        if d0["case"].nunique(dropna=True) < 2:
            print(f"  - Skipping {met} ~ {lcol}: only {d0['case'].nunique(dropna=True)} case with data")
            continue
        out = test_case_effect_controlling_length(sample_tok, metric=met, length_col=lcol)
        if out is not None and not out.empty:
            print(f"  - {met} ~ {lcol} (n={int(out['n'].iloc[0]) if 'n' in out.columns and len(out) else 'NA'})")
            if "term" in out.columns:
                print(out[out["term"].str.startswith("C(case)")].to_string(index=False))
            else:
                print(out.to_string(index=False))
            case_len_parts.append(out)

    case_len = pd.concat(case_len_parts, ignore_index=True) if case_len_parts else pd.DataFrame()

    print("\n[Sanity] POS permutation test (language-wise, current text)")
    pos_perm_rows = []
    langs = sorted([x for x in df_tokens["language"].dropna().astype(str).unique().tolist() if x])
    for lang in langs:
        for pos in ("Noun", "Verb", "Adj"):
            pos_perm_rows.append(permutation_test_pos_share(df_tokens, language=lang, pos=pos, n_perm=200))
    pos_perm = pd.DataFrame(pos_perm_rows)
    if not pos_perm.empty:
        print(pos_perm.sort_values(["language", "pos"]).to_string(index=False))
    else:
        print("No language/POS data available for permutation test.")

    print("\n[Sanity] Bridge hubness null model")
    bridge = compute_bridge_metrics(df_tokens)
    hub_null = null_bridge_hubness(df_tokens)
    if hub_null is not None and not hub_null.empty:
        print(hub_null.to_string(index=False))
    else:
        print("Bridge hubness skipped (missing fields or no bridge structure).")

    print("\n[Findings] Ranked, caveated top effects")
    findings = build_findings(sample_tok=sample_tok, sample_word=sample_word, bridge=bridge, denom=denom)
    if findings is not None and not findings.empty:
        print(findings.to_string(index=False))
    else:
        print("No findings (insufficient data for comparisons).")

    if save_figures:
        plot_entropy_vs_length(sample_tok, os.path.join(FIG_DIR, "entropy_vs_length.png"))
        plot_retention_all_vs_textonly(sample_tok, os.path.join(FIG_DIR, "retention_all_vs_textonly.png"))
        plot_pos_shares_with_ci(sample_tok, os.path.join(FIG_DIR, "pos_shares_ci.png"))
        if hub_null is not None and not hub_null.empty:
            plot_hubness_real_vs_null(hub_null, os.path.join(FIG_DIR, "hubness_real_vs_null.png"))
        if sample_word is not None and not sample_word.empty and "shapley_entropy_word_current" in sample_word.columns:
            plot_sensitivity_entropy_token_vs_word(
                sample_tok,
                sample_word,
                os.path.join(FIG_DIR, "entropy_token_vs_word.png"),
            )
        else:
            print("Skipping sensitivity plot: 'shapley_entropy_word_current' missing in sample_word")

    return {
        "df_tokens": df_tokens,
        "sample_tok": sample_tok,
        "sample_word": sample_word,
        "sanity_denominator": denom,
        "sanity_case_length": case_len,
        "sanity_pos_perm": pos_perm,
        "bridge": bridge,
        "hubness_null": hub_null,
        "findings": findings,
    }
