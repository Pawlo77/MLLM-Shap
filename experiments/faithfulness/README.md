# Faithfulness Deletion Experiment (HP-1)

Deletion-based faithfulness validation for SGPA Shapley values. Tests whether the segments assigned the highest Shapley values are **causally predictive** of model output.

## Overview

This experiment validates the SGPA pipeline's attributions by performing post-hoc deletion tests over existing `mllm_shapx` SGPA runs. It does **not** re-estimate Shapley values — it consumes pre-computed results and measures whether masking high-SV segments degrades model output more than masking random segments.

## Directory Layout

```
experiments/faithfulness/
├── configs/
│   ├── hp1_faithfulness_audio_male_spec.json
│   └── hp1_faithfulness_audio_female_spec.json
├── src/
│   ├── __init__.py
│   ├── run.py          # Main CLI runner
│   ├── runners.py      # Single-sample evaluation logic
│   ├── helpers.py      # Audio manipulation, SV extraction, similarity
│   ├── models.py       # Result dataclasses
│   ├── summarize.py    # Aggregate statistics & effect sizes
│   └── plot.py         # Figure generation
├── figures/
│   ├── faithfulness_deletion_embedding.png
│   ├── faithfulness_deletion_tfidf.png
│   ├── faithfulness_rankwise_embedding.png
│   └── faithfulness_rankwise_tfidf.png
├── outputs/
│   ├── rankwise/       # Per-rank deletion results
│   └── remap/          # Remapped results
├── faithfulness_plot.ipynb  # Interactive plotting notebook
└── README.md
```

## Method

### Variant 1 — Top-SV vs Random

For each sample from a completed `mllm_shapx` SGPA run:

1. Load the original audio and pre-computed Shapley values
2. Identify the **Top-1 segment** (`argmax(|SV_i|)`)
3. Silence the Top-1 segment; silence a duration-matched random segment
4. Run LFM2-Audio on original, top-deleted, and random-deleted audio
5. Compute embedding cosine similarity drops
6. Paired t-test + Cohen's d across all samples

### Variant 2 — Rankwise Deletion

Delete segments in descending SV rank order and track cumulative response degradation, confirming that high-SV segments contribute more to output than low-SV segments.

## Similarity Metric

Embedding cosine similarity (matches the SV utility function):
- Prefill-only forward pass to get hidden states
- Mean-pool to a single vector per response
- Cosine similarity between full-audio and masked-audio embeddings

## Running

```bash
# From repo root
python -m experiments.faithfulness.src.run \
    --spec experiments/faithfulness/configs/hp1_faithfulness_audio_male_spec.json \
    --output-dir experiments/faithfulness/outputs/rankwise
```

## SLURM

```bash
sbatch experiments/mllm_shapx/sbatchs/run_hp1_faithfulness.sbatch
```

## Key Results

The faithfulness test confirms that SGPA-identified top segments are causally predictive:
- Masking the Top-1 SV segment produces a significantly larger response change than masking a random segment of equivalent duration
- Effect is consistent across both male and female voice variants
- See `outputs/README.md` for detailed statistics
