# HP-1 Rank-Wise Deletion Diagnostic

## What Was Tested

This diagnostic reruns only phase 2 using the existing phase-1 SHAP sample JSONs. For every sample, each SGPA segment is deleted once, then segments are ranked by aggregated absolute SV within that sample.

The goal was to test whether deletion impact decreases with SV rank and to check whether the previous top-vs-random baseline was weak because random segments often have similar SV mass.

## Main Result

The run completed 591 segment deletions over 99 samples with 0 failures.

| Split | Samples | Deletions | Rank-1 drop | Non-top drop | Rank-1 minus non-top | Spearman | Within-sample Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined | 99 | 591 | 0.8074 | 0.8231 | -0.0157 | 0.0422 | -0.0904 |
| Male TTS | 51 | 306 | 0.8029 | 0.8326 | -0.0297 | 0.0576 | -0.1118 |
| Female TTS | 48 | 285 | 0.8122 | 0.8128 | -0.0006 | 0.0328 | -0.0676 |

Additional paired sample-level check:

- Rank-1 minus mean non-top drop: mean `-0.0225`, median `0.0157`, paired `p=0.320`.
- Rank-1 deletion exceeded mean non-top deletion for `59.6%` of samples.
- The absolute-SV vs deletion-drop Spearman correlation is near zero globally and within samples.

## SV Concentration Diagnostic

The SV distributions are mostly flat:

- mean top-1 absolute-SV share: `0.2585`;
- median top-1 absolute-SV share: `0.2301`;
- mean top1-top2 absolute-SV gap: `0.0439`;
- median top1-top2 absolute-SV gap: `0.0203`;
- mean normalized entropy: `0.9709`.

This confirms that ranks often do not represent large attribution separations. Rank 1 and rank 2 are frequently close, so random non-top deletion can easily hit a segment with similar SV mass.

However, the all-rank deletion data does **not** show that higher absolute-SV rank reliably predicts larger response changes. Deletion drops are high for most segments, and the response-similarity metric is heavily saturated: `73.6%` of deletions produce drop `>= 0.8`, and `42.0%` produce drop `>= 0.9`.

## Rebuttal Interpretation

This result should not be used as a positive rank-faithfulness figure. It weakens a strong claim that SGPA-SV rank monotonically predicts behavioral importance.

It is still useful as an internal diagnostic:

- It explains why top-vs-random deletion gave only a modest aggregate effect.
- Random deletion is not necessarily a weak baseline in short utterances with flat attribution distributions.
- The current generated-response TF-IDF similarity endpoint may be too saturated to resolve fine-grained rank differences.

Rebuttal-safe wording:

> As an additional diagnostic, we reran deletion over all SGPA segment ranks using the existing SHAP outputs. This showed that the attribution mass is often broadly distributed across short utterances (mean top-1 |SV| share = 0.259; normalized entropy = 0.971), so random deletions often remove segments with similar attribution mass. The rank-wise deletion curve did not show a reliable monotonic relationship between |SV| rank and response-similarity drop, suggesting that the deletion endpoint is too coarse/saturated for fine-grained rank validation. We therefore report the top-vs-random deletion result only as modest aggregate faithfulness evidence, not as evidence of deterministic per-rank ordering.

## Artifacts

- Row-level all-rank results: `combined_rankwise_results.csv`
- Overall summary: `combined_rankwise_summary.json`
- Figure: `paper/interspeech/figures/faithfulness_rankwise.{png,pdf}`
