# Stage 3 Ablation Results

This experiment isolates SGPA Stage 3 (spectral boundary refinement).
For each utterance, we compare boundary spectral flux at:
- raw CTC word boundaries
- SGPA-refined boundaries

Segment count stays fixed; only boundary positions change.

## Multi-dataset / multi-config workflow

The pipeline now supports running ablations for multiple dataset configs in the same output directory (e.g. `single_sentence_1k` and `single_sentence_500`) without filename collisions.

### Output naming

All files are now prefixed by dataset config:

- `<dataset_config>__<audio_column>_n<max_samples>_samples.csv`
- `<dataset_config>__<audio_column>_n<max_samples>_boundaries.csv`
- `<dataset_config>__<audio_column>_n<max_samples>_failures.csv`
- `<dataset_config>__<audio_column>_n<max_samples>_summary.json`

Example:
- `single_sentence_500__audio__original_n1000_samples.csv`

## Reproduction

Output directory used below:

```bash
export OUT_DIR="experiments/interspeech/outputs/stage3_ablation"
```

### 1) Run ablation for `single_sentence_1k`

```bash
for col in audio__male audio__female audio__original; do
  uv run python -m experiments.interspeech.src.stage3_ablation \
    --dataset-config single_sentence_1k \
    --audio-column "$col" \
    --max-samples 1000 \
    --output-dir "$OUT_DIR"
done
```

### 2) Run ablation for `single_sentence_500` (including original audio)

```bash
uv run python -m experiments.interspeech.src.stage3_ablation \
  --dataset-config single_sentence_500 \
  --audio-column audio__original \
  --max-samples 500 \
  --output-dir "$OUT_DIR"
```

### 3) Generate figures

- Single combined figure (all dataset configs in one plot).
- Hue encodes `voice | dataset_config`.

```bash
uv run python -m experiments.interspeech.src.stage3_ablation_plot \
  --input-dir "$OUT_DIR" \
  --output-base paper/interspeech/figures/stage3_ablation
```

This emits:
- `paper/interspeech/figures/stage3_ablation.png`

- Optional: filter to single dataset config:

```bash
uv run python -m experiments.interspeech.src.stage3_ablation_plot \
  --input-dir "$OUT_DIR" \
  --dataset-config single_sentence_500 \
  --max-samples 500 \
  --output-base paper/interspeech/figures/stage3_ablation_single_sentence_500
```

## Plot metadata

Each rendered image also writes metadata JSON beside source CSVs:

- combined call (no dataset filter): `image_meta.json`
- single-config call: `image_meta__<dataset_config>.json`
