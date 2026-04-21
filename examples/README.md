<div align="center">
  <h1>💡 MLLM-SHAP Examples</h1>
  <p>Interactive Jupyter Notebooks demonstrating XAI for Multi-modal LLMs.</p>
</div>

---

This folder contains example use cases of the `mllm-shap` package. Most examples utilize the built-in `liquid_audio` model (see the [connector reference](./../mllm_shap/src/mllm_shap/connectors/liquid/)).

## ⚙️ Prerequisites

Before running the examples, ensure you have the required scientific mapping dependencies installed (e.g., `jupyter`, `torch`, `transformers`). You can install the package in editable mode from the root directory:

```bash
pip install -e mllm_shap
```

## 📓 Notebooks Overview

Each notebook can also serve as an extensive **manual test** to verify the package’s correct functionality—especially useful for researchers developing custom connectors or extending the package’s capabilities.

### 🎭 Basic Usage & Multi-modal Scenarios
| Notebook | Modality | Description |
|---|---|---|
| **[📄 Multi Turn Text](./text_multi_turn.ipynb)** | `Text -> Text` | Recommended starting point. Demonstrates exact Shapley computation and Monte Carlo approximations in a 2-turn conversation. |
| **[🎲 Monte Carlo Text](./text_monte_carlo.ipynb)** | `Text -> Text` | Continuation of *Multi Turn Text*, illustrating low-call Monte Carlo setups for extremely rapid, albeit noisy, SHAP approximations. |
| **[🎙️ Monte Carlo Internal Audio](./audio_internal.ipynb)** | `Text -> Audio` | Explains multi-turn models outputting audio. Guides on forcing audio representations into subsequent expandability operations. |
| **[🎧 Monte Carlo Audio](./audio_external.ipynb)** | `Audio+Text -> Audio` | Comprehensive example blending all supported modalities simultaneously safely within the API. |

### 🛠️ Advanced Explainers
| Notebook | Concept | Description |
|---|---|---|
| **[🧩 Complementary](./complementary.ipynb)** | `Complementary SHAP` | Demonstrates the *Complementary* SHAP explainer logic, rendering internal calculation matrices for validation. |
| **[⚖️ Neyman](./neyman.ipynb)** | `Neyman-Shapley` | Showcases the *Neyman-Shapley* allocation metric for targeted attribution logic and constraint distributions. |
| **[🌲 Hierarchical](./hierarchical.ipynb)** | `Hierarchical SHAP` | Illustrates structural inference breakdown via the *Hierarchical* SHAP module algorithms. |

---
<sub>Need more structure? Check out the [core library](../mllm_shap/) or review [research outputs](../experiments/).</sub>
