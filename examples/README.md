<div align="center">
  <h1>💡 Examples Gallery</h1>
  <p><strong>Curated notebook workflows for text/audio SHAP explainability.</strong></p>
</div>

# 🎯 Purpose

Directory contains practical, reproducible notebook scenarios for `mllm-shap`.

# Setup

From the repository root:

```bash
make install
```

or install the package in editable mode:

```bash
pip install -e mllm_shap
```

# 📊 Collection Snapshot

- **10 notebooks** across core + advanced explainers
- **modalities covered**: `Text → Text`, `Text → Audio`, `Audio + Text → Audio`
- **best first notebook**: `text_only_hf_model.ipynb` — runs on CPU with any HuggingFace causal LM, no special hardware needed

# 📓 Notebooks

## Text → Text (HuggingFace `TransformersCausalText`)

- `text_only_hf_model.ipynb`
  - **recommended starting point** — works with any public HuggingFace causal LM on CPU
  - `McShapExplainer` with `SYSTEM_ASSISTANT` setup and `KeepAllTokens`

- `custom_embeddings.ipynb`
  - custom external embedding model (`intfloat/e5-small-v2`) for SHAP similarity scoring
  - `PreciseShapExplainer` for single-turn; falls back to `McShapExplainer` for multi-turn to avoid exponential mask space

## Text → Text / Audio (`LiquidAudio`)

- `text_multi_turn.ipynb`
  - compares exact (`PreciseShapExplainer`) and approximate (`McShapExplainer`) behavior side-by-side
  - demonstrates cache reuse across turns and the `approximate_budget()` utility

- `text_monte_carlo.ipynb`
  - focused Monte Carlo attribution run with `ExcludePunctuationTokensFilter`
  - shows full mask enumeration via `num_samples=-1`

## Text → Audio / Audio + Text → Audio (`LiquidAudio`)

- `audio_internal.ipynb`
  - explains pipelines where model output is audio (internal audio generation)

- `audio_external.ipynb`
  - end-to-end multimodal example with audio input and audio output

- `audio_external_spectogram.ipynb`
  - extends `audio_external.ipynb` with `SpectrogramGuidedAligner` for segment-level audio attribution

## Advanced Explainers (`LiquidAudio`)

- `complementary.ipynb` — `ComplementaryShapExplainer` with fractional sampling
- `neyman.ipynb` — `ComplementaryNeymanShapExplainer` for Neyman-optimal sample allocation
- `hierarchical.ipynb` — `HierarchicalExplainer` for multi-level token group decomposition

# 🧪 Quality Note

These notebooks double as practical smoke tests for connector and explainer integration.
