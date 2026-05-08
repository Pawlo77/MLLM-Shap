# Stage 3 Ablation Results

This experiment isolates SGPA Stage 3 (spectral boundary refinement).
For each utterance, we compare boundary spectral flux at:
- raw CTC word boundaries
- SGPA-refined boundaries

Segment count stays fixed; only boundary positions change.

## Output files

All results are saved as consolidated files:

- `samples.csv` — per-sample metrics for all audio columns
- `boundaries.csv` — per-boundary detail for all audio columns
- `summary.json` — per-column statistics, combined statistics, and failure records

## Reproduction

Run the notebook `experiments/interspeech/stage3_ablation_plot.ipynb` end-to-end.
It executes the ablation and generates figures in a single pass.

Configuration is at the top of the notebook:

```python
DATASET_CONFIG = "single_sentence_1k"
AUDIO_COLUMNS = ["audio__male", "audio__female"]
MAX_SAMPLES = 1000
```

Figures are saved to `experiments/interspeech/figures/`.

## Plot metadata

When `save_meta=True` is passed to `plot_stage3_ablation`, metadata JSON is written beside the source CSVs:

- combined call (no dataset filter): `image_meta.json`
- single-config call: `image_meta__<dataset_config>.json`
