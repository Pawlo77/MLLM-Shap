# HP-1 Deletion Faithfulness Result

## What Was Tested

We evaluated whether the audio segment with the largest absolute SGPA-SV attribution is more behaviorally important than a random segment. For each sample, we removed:

- the SGPA segment with the largest absolute SV after aggregating serialized audio SVs onto aligned SGPA segments, and
- a length-matched random SGPA segment.

We then regenerated the model response and measured the response-similarity drop. A positive paired difference means the top-SV deletion changed the response more than the random deletion.

## Main Result

The experiment completed 98 deletion tests with 0 failures.

| Split | n | Top-SV drop | Random drop | Mean paired diff | Paired p | Cohen dz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined | 98 | 0.7722 | 0.7353 | 0.0369 | 0.0406 | 0.21 |
| Male TTS | 50 | 0.7722 | 0.7671 | 0.0051 | 0.7821 | 0.04 |
| Female TTS | 48 | 0.7722 | 0.7023 | 0.0699 | 0.0251 | 0.33 |

Additional validation on the row-level CSV confirmed:

- no duplicate `(audio_column, sample_id)` rows;
- paired t-tests recompute from the CSV exactly;
- the combined 95% CI for the paired difference is `[0.0016, 0.0721]`;
- Wilcoxon one-sided test also supports the aggregate effect (`p=0.0181`);
- the effect is modest overall and stronger for the female voice subset.

## Rebuttal-Safe Interpretation

These results provide deletion-based faithfulness evidence for the SGPA-SV attributions: across 98 audio-output explanations, deleting the top-SV SGPA segment caused a significantly larger response-similarity drop than deleting a length-matched random segment. The result should be described as an aggregate, modest effect, with a transparent voice split. It should not be overclaimed as uniformly strong across both voices, since the male subset is neutral while the female subset drives much of the aggregate effect.

Suggested wording:

> We added a deletion-based faithfulness check on 98 generated-audio explanations. For each sample, we removed the SGPA segment with the largest absolute SV and compared the response-similarity drop against a length-matched random segment deletion. Top-SV deletion produced a larger drop overall (mean paired difference = 0.0369, paired t(97)=2.07, p=0.041), with no failed samples after validation. The effect was strongest for female TTS (mean difference = 0.0699, p=0.025), while male TTS was neutral, so we present this as aggregate evidence that SGPA-SV attributions identify behaviorally relevant audio regions rather than as a universal per-voice guarantee.

## Artifacts

- Row-level combined results: `combined_results.csv`
- Overall summary: `combined_summary.json`
- Male summary/results: `audio__male_combined_summary.json`, `audio__male_combined_results.csv`
- Female summary/results: `audio__female_combined_summary.json`, `audio__female_combined_results.csv`
- Figure: `experiments/interspeech/figures/faithfulness_deletion.{png,pdf}`
