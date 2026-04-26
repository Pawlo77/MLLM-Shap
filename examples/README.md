<div align="center">
  <h1>💡 Examples Gallery</h1>
  <p><strong>Curated notebook workflows for text/audio SHAP explainability.</strong></p>
</div>

## 🎯 Purpose

Directory contains practical, reproducible notebook scenarios for `mllm-shap`.

## Setup

From the repository root:

```bash
make install
```

or install the package in editable mode:

```bash
pip install -e mllm_shap
```

## 📊 Collection Snapshot

- **7 primary notebooks** across core + advanced explainers
- **modalities covered**: `Text -> Text`, `Text -> Audio`, `Audio + Text -> Audio`
- **best first notebook**: `text_multi_turn.ipynb`

## 📓 Notebooks

### Core Scenarios

- `text_multi_turn.ipynb` (`Text -> Text`)
  - recommended starting point
  - compares exact and approximate SHAP behavior in a multi-turn setup

- `text_monte_carlo.ipynb` (`Text -> Text`)
  - focused on low-cost Monte Carlo attribution runs

- `audio_internal.ipynb` (`Text -> Audio`)
  - explains pipelines where model output is audio

- `audio_external.ipynb` (`Audio + Text -> Audio`)
  - end-to-end multimodal example with mixed input modalities

### Advanced Explainers

- `complementary.ipynb` - complementary SHAP flow
- `neyman.ipynb` - Neyman-based attribution variant
- `hierarchical.ipynb` - hierarchical decomposition approach

## 🧪 Quality Note

These notebooks double as practical smoke tests for connector and explainer integration.
