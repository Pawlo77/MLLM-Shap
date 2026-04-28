# Stage 3 Ablation Results

This experiment isolates the contribution of SGPA Stage 3, the spectral boundary refinement step. It compares word-level cut points obtained directly from raw CTC alignment against the same word-level segmentation after Stage 3 refinement. The segment count is unchanged; only the boundary locations differ.

## Rebuttal Claim

Stage 3 is not merely cosmetic. Across 200 TTS utterances (100 male, 100 female), SGPA refinement moves boundaries to substantially lower spectral-flux regions than raw CTC boundaries.

## Main Result

| Split | n | Raw mean flux | Refined mean flux | Mean reduction | t | p | Cohen's dz |
|---|---:|---:|---:|---:|---:|---:|---:|
| Male TTS | 100 | 34.25 | 16.12 | 53.12% | 11.27 | 1.92e-19 | 1.13 |
| Female TTS | 100 | 15.81 | 6.93 | 51.61% | 11.47 | 7.07e-20 | 1.15 |
| Combined | 200 | 25.03 | 11.53 | 52.36% | 14.23 | 3.74e-32 | 1.01 |

No samples failed alignment or measurement.

## Rebuttal-Ready Wording

To address the concern that Stage 3 may be unnecessary, we added an ablation comparing raw CTC word boundaries against SGPA-refined boundaries while holding the word-level player set fixed. Over 200 synthesized VoiceBench utterances (100 male, 100 female), SGPA reduced mean boundary spectral flux from 25.03 to 11.53, a 52.36% reduction (paired t-test: t=14.23, p=3.74e-32, Cohen's dz=1.01). This shows that Stage 3 contributes independently of dimensionality reduction: it systematically relocates cuts away from acoustically unstable regions where masking is more likely to introduce out-of-distribution artifacts.

## Outputs

- `combined_n200_summary.json`: combined statistical summary.
- `combined_n200_samples.csv`: per-sample male+female measurements.
- `audio__male_n100_samples.csv`: per-sample male measurements.
- `audio__female_n100_samples.csv`: per-sample female measurements.
- `audio__male_n100_boundaries.csv`: per-boundary male measurements.
- `audio__female_n100_boundaries.csv`: per-boundary female measurements.
- `../../../paper/interspeech/figures/stage3_ablation.png`: paper-style figure.
- `../../../paper/interspeech/figures/stage3_ablation.pdf`: vector figure.

## Reproduction

From the repository root:

```bash
PYTHONPATH="/home/mvishiu11/Desktop/thesis-code/AudioShap_WUT_Bachelor" \
UV_PROJECT_ENVIRONMENT="/home/mvishiu11/Desktop/thesis-code/AudioShap_WUT_Bachelor/.venv" \
uv run --project "/home/mvishiu11/Desktop/thesis-code/AudioShap_WUT_Bachelor" --group dev \
python -m experiments.interspeech.src.stage3_ablation \
  --max-samples 100 \
  --audio-column audio__male \
  --output-dir experiments/interspeech/outputs/stage3_ablation \
  --device cpu
```

Repeat with `--audio-column audio__female`, then generate the figure:

```bash
PYTHONPATH="/home/mvishiu11/Desktop/thesis-code/AudioShap_WUT_Bachelor" \
UV_PROJECT_ENVIRONMENT="/home/mvishiu11/Desktop/thesis-code/AudioShap_WUT_Bachelor/.venv" \
uv run --project "/home/mvishiu11/Desktop/thesis-code/AudioShap_WUT_Bachelor" --group dev \
python -m experiments.interspeech.src.stage3_ablation_plot
```
