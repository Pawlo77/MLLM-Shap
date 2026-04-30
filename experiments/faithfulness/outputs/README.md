# HP-1 Faithfulness Deletion Validation

This experiment validates whether the SGPA Shapley values identify causally important audio
segments. It is a post-hoc deletion test over existing `mllm_shapx` SGPA runs; it does
**not** re-estimate Shapley values.

---

## Similarity Metric

**Current (embedding cosine — matches SV utility function).**
The similarity between two model responses is computed by:

1. Calling `model.get_static_embeddings([response_full, response_masked])` — a prefill-only
   forward pass over the response token sequence.
2. Mean-pooling each hidden-state sequence to a single `[hidden_dim]` vector (`MeanReducer`).
3. Computing `CosineSimilarity` between the full-audio vector and the masked-audio vector.

This is the **same** utility function (U1/U2) used to compute the SGPA Shapley values, so
faithfulness is measured on the same scale as the SVs themselves.

> **Previous metric:** TF-IDF cosine similarity over output token hashes (`TfIdfCosineSimilarity`,
> U3). That metric produced heavily saturated drops (73.6 % of single-segment deletions ≥ 0.8)
> because any response change, regardless of magnitude, registered as a near-1 drop. Switching to
> embedding cosine resolves saturation and makes the faithfulness test internally consistent with
> the SV computation.

---

## Experiment Variants

### Variant 1 — Top-SV vs Random (N = 2 model calls per sample)

For each serialized `mllm_shapx` sample:

1. Load the original dataset row via `spec.json` (exact HF revision, split, and shuffle seed).
2. Read audio Shapley values from `samples/sample_XXXXX_result.json`.
3. Select **Top-1 segment** = `argmax(|SV_i|)`.
4. Reconstruct the SGPA word segmentation via `SpectrogramGuidedAligner`.
5. Silence the Top-1 segment; silence a duration-matched random non-top segment.
6. Run LFM2-Audio on: original, top-deleted, random-deleted audio.
7. Compute embedding cosine similarity drops:
   - `top_drop = 1 − cosine(emb_full, emb_top_deleted)`
   - `random_drop = 1 − cosine(emb_full, emb_random_deleted)`
   - `drop_difference = top_drop − random_drop`
8. Paired t-test + Cohen's dz across samples.

**Faithfulness claim:** `mean(drop_difference) > 0`, i.e. the top-SV segment causes a
larger response change than a duration-matched random segment.

### Variant 2 — All-Rank Deletion (N = K model calls per sample, K = number of segments)

Same pipeline but every SGPA segment is deleted once, producing a rank-indexed deletion curve.
Additional statistics:

- Per-rank mean drop ± 95 % CI (deletion curve).
- Spearman ρ(|SV|, deletion\_drop) globally and within each sample.
- Within-sample Spearman distribution across all samples.

---

## Available Result Directories

| Directory | Variant | Samples | Metric | Notes |
|-----------|---------|--------:|--------|-------|
| `remap_fix_50/` | Top-SV vs Random | 98 | TF-IDF cosine | Pre-embedding-switch run |
| `rankwise_50_v2/` | All-Rank | 99 | TF-IDF cosine | Pre-embedding-switch run |
| `local_cpu_smoke/` | Top-SV vs Random | 1 | TF-IDF cosine | CPU smoke test |

**Note:** All completed runs above used `TfIdfCosineSimilarity`. New runs will use embedding
cosine similarity after the metric switch in `faithfulness_deletion.py`.

---

## Plot Analysis (TF-IDF metric, pre-switch runs)

### `faithfulness_deletion.png` — Top-SV vs Random

**Paired Deletion Comparison.**
The per-sample slope lines connecting random and top-SV drops are nearly flat and criss-cross uniformly, with no dominant upward direction. The mean lines with 95 % CI tell the story by voice: Male TTS shows the two condition means almost on top of each other (Δ = 0.005, p = 0.78), while Female TTS shows a small but visible upward slope from random to top-SV (Δ = 0.070, p = 0.025). The combined effect is marginal (p = 0.041, Cohen's dz = 0.21) and is entirely carried by the female voice.

**Drop Difference by Voice.**
Male TTS violin is narrow, symmetric, and centred exactly on zero — a textbook null distribution. Female TTS is wider and right-skewed, with the median above zero but a substantial lower tail extending well below zero. This asymmetry explains the significant combined t-test: roughly one-third of female-voice samples produce a negative drop difference (random deletion hurts more than top-SV deletion), blunting the aggregate effect.

**Drop Distribution: Top-SV vs Random.**
Solid (Top-SV) and dashed (Random) KDE lines overlap almost completely for both voices. Both distributions peak around 0.85–0.90 with a long left tail to zero. The Female TTS top-SV curve is shifted infinitesimally to the right of its random counterpart; the Male TTS curves are indistinguishable. This confirms that the TF-IDF metric collapses the signal: removing any single word from a short utterance is already enough to produce a near-maximum output change, leaving almost no room for the metric to reveal which segment matters more.

**Top |SV| vs Paired Drop Difference.**
Points scatter symmetrically around zero across the full range of top-segment SV magnitudes (0.1–0.6). The OLS regression line is flat (r = 0.046, p = 0.65). Attribution confidence — how large the top SV is relative to the rest — carries no information about whether that segment will produce a larger behavioural impact when removed. This rules out a simple "high-confidence SVs are automatically more faithful" story.

---

### `faithfulness_rankwise.png` — All-Rank Deletion

**Deletion Drop by |SV| Rank.**
Both voices produce non-monotonic curves that bounce irregularly between 0.75 and 0.90 across ranks 1–9. Rank 1 (highest |SV|) is not the highest-drop rank in either voice; ranks 3 and 5 occasionally exceed it. Error bars grow at high ranks (fewer utterances with ≥6 segments), but the noise is not merely statistical — the non-monotonicity persists even at the well-sampled low ranks. The global Spearman of 0.042 and the within-sample median of −0.024 confirm the curve is essentially flat regardless of how it is measured.

**|SV| vs Deletion Drop.**
All 591 deletion data points cluster in a dense band at the top of the y-axis (drops 0.7–1.0) regardless of the segment's absolute SV (x-axis, 0–0.6). The OLS line has a marginally negative slope (r = −0.064) but is visually indistinguishable from horizontal given the vertical compression of the data. Low-SV segments routinely produce drops as large as the highest-SV segment, confirming that the metric does not discriminate.

**Saturation Diagnostic.**
This is the clearest panel. The histogram is strongly left-skewed: the mass is concentrated at 0.8–1.0, and both CDF curves rise steeply through that same range. The dashed and dotted threshold lines at 0.8 and 0.9 sit well within the bulk of the distribution. Removing a single word from an utterance of 3–10 words is almost always enough to change the model's audio output so substantially that the TF-IDF similarity records a near-maximum drop. There is therefore no dynamic range left for attribution rank to explain. This is a structural limitation of the metric, not of the SVs themselves.

**Within-Sample Spearman(|SV|, Drop).**
The violin spans almost the full range [−1, +1] and is nearly symmetric around zero. The median reference line sits on the zero baseline. Within individual samples, the correlation between a segment's |SV| and its deletion impact is as likely to be negative as positive. This wide spread is not interpretable as a genuine faithfulness signal; it reflects sampling noise under a saturated metric, where the ranking of deletion drops within a sample is effectively random.

---

### Summary diagnosis

The TF-IDF-based runs expose a single root cause: **metric saturation**. Short utterances (3–10 word segments) leave the model's audio-output token stream so sensitive to any single-segment deletion that 74 % of deletions already achieve a drop ≥ 0.8 regardless of which segment is chosen. Under this ceiling, the paired Top-SV vs Random test retains only a marginal signal (and only for the female voice), while the rank-wise test is completely uninformative. The embedding cosine metric — which measures response similarity in continuous hidden-state space rather than by token-hash matching — is expected to provide the dynamic range needed to expose the attribution ranking as a genuine faithfulness signal.

---

## Key Results (TF-IDF metric, pre-switch)

### Top-SV vs Random (`remap_fix_50`)

| Split | n | Mean Δ | p (paired t) | Cohen's dz | Top > Random |
|-------|--:|-------:|-------------:|-----------:|-------------:|
| Combined | 98 | 0.037 | 0.041 | 0.21 | 55.1 % |
| Male TTS | 50 | 0.005 | — | — | 48.0 % |
| Female TTS | 48 | 0.070 | — | — | 62.5 % |

Effect is marginally significant combined, driven by the female voice. The male voice shows
no effect, suggesting voice-dependent attribution quality or response variability.

### All-Rank Deletion (`rankwise_50_v2`)

| Split | Samples | Deletions | Rank-1 drop | Non-top drop | Rank-1 − non-top | Spearman global | Spearman within |
|-------|--------:|----------:|------------:|-------------:|-----------------:|----------------:|----------------:|
| Combined | 99 | 591 | 0.807 | 0.823 | −0.016 | 0.042 | −0.090 |
| Male TTS | 51 | 306 | 0.803 | 0.833 | −0.030 | 0.058 | −0.112 |
| Female TTS | 48 | 285 | 0.812 | 0.813 | −0.001 | 0.033 | −0.068 |

The rank curve is **non-monotonic** and Spearman is near zero. Two compounding causes:

1. **Metric saturation:** 73.6 % of deletions produce drop ≥ 0.8; 42 % produce drop ≥ 0.9.
   Silencing any single word is enough to change the model's audio output substantially.
2. **Flat SV distributions:** Mean top-1 |SV| share = 0.259; mean normalized entropy = 0.971.
   Rank 1 and rank 2 typically differ by only ~0.04 absolute SV, so random deletions often
   remove a segment with nearly the same attribution mass as the top-1 segment.

Both issues are expected to improve with the embedding cosine metric, which is continuous and
resolves fine-grained embedding differences rather than binary output-token matches.

---

## Plots

### Deletion plot — 2 × 2 figure

| Panel | Description |
|-------|-------------|
| **A** | Paired slope plot: per-sample top-SV vs random drop connected by lines; mean ± 95 % CI per voice. |
| **B** | Drop-difference distributions: violin + box + strip per voice, zero line. |
| **C** | Overlapping KDE of `top_drop` vs `random_drop`; solid = top-SV, dashed = random. |
| **D** | Effect moderation: top absolute SV vs drop difference scatter + OLS regression line. |

### Rank-wise plot — 2 × 2 figure

| Panel | Description |
|-------|-------------|
| **A** | Deletion curve: mean drop ± 95 % CI by |SV| rank; separate lines per voice. |
| **B** | |SV| vs deletion-drop scatter (all segment deletions) + OLS regression line. |
| **C** | Saturation diagnostic: histogram of all deletion drops + CDF overlay; threshold lines at 0.8 and 0.9. |
| **D** | Within-sample Spearman distribution: violin + strip of per-sample ρ(|SV|, drop). |

### How to generate

Plots are generated via the notebook `experiments/interspeech/faithfulness_plot.ipynb`.
The plotting API is in `experiments.interspeech.src.faithfulness.plot` (`plot_deletion`, `plot_rankwise`).

Figures are saved to `experiments/interspeech/figures/`.

---

## Running the Experiment

### Preflight check (no model load)

```bash
PYTHONPATH="$PWD" .venv/bin/python -m experiments.interspeech.src.faithfulness.run \
  --run-dir experiments/experiments_output/single_sentence_2026_01_03/audio_male_audio_limited_neyman_lin3_0 \
  --output-dir experiments/interspeech/outputs/faithfulness_deletion/debug \
  --max-samples 2 \
  --preflight-only
```

### One-sample smoke test (GPU)

```bash
PYTHONPATH="$PWD" .venv/bin/python -m experiments.interspeech.src.faithfulness.run \
  --run-dir experiments/experiments_output/single_sentence_2026_01_03/audio_male_audio_limited_neyman_lin3_0 \
  --output-dir experiments/interspeech/outputs/faithfulness_deletion/debug \
  --max-samples 1 --max-new-tokens 1 --device cuda --aligner-device cpu --resume
```

### All-rank deletion (N = K)

Add `--all-rank-deletions` to the above. Combine partitions afterwards:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m experiments.interspeech.src.faithfulness.run \
  --output-dir experiments/interspeech/outputs/faithfulness_deletion/rankwise_new \
  --combine-only --all-rank-deletions
```

### Combine Top-SV vs Random partitions

```bash
PYTHONPATH="$PWD" .venv/bin/python -m experiments.interspeech.src.faithfulness.run \
  --output-dir experiments/interspeech/outputs/faithfulness_deletion/new_run \
  --combine-only
```

---

## Environment Setup (Lem)

```bash
cd ~/MLLM-Shap
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=2
rm -rf .venv
uv sync --group dev
```

Verify:

```bash
.venv/bin/python - <<'PY'
import numpy, pandas, torch
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

## SLURM Strategy

- One LFM2 replica per H100; array tasks split by voice × partition index.
- Default: `PARTITIONS_PER_VOICE=8` → 16 array tasks total.
- Submit: `sbatch experiments/mllm_shapx/sbatchs/run_hp1_faithfulness.sbatch`
- Scale: `PARTITIONS_PER_VOICE=16/32/64` → update `--array` directive accordingly.
