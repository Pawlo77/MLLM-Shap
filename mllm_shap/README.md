<div align="center">
  <h1>🌟 <code>mllm-shap</code> Package</h1>

  <p>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/v/mllm-shap.svg" alt="PyPI"></a>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/pyversions/mllm-shap.svg" alt="Python"></a>
    <a href="https://pawlo77.github.io/MLLM-Shap/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Docs"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  </p>

  <p>
    <b>A robust toolkit for interpreting Large Language Models across multiple modalities.</b>
  </p>
</div>

**mllm-shap** is a robust Python package designed to **interpret the predictions of large language models (LLMs)** using **SHAP (SHapley Additive exPlanations)** values.
It helps you understand the contribution of input features to model outputs, enabling **transparent and explainable AI workflows**.
This work also has companion GUI visualization tools for easier interpretation of results, which is available at the official [shap-mllm-explainer](https://github.com/mvishiu11/shap-mllm-explainer) repository.

---

## ✨ Key Features

- Integration with **audio and text models**, supporting multi-modal inputs and outputs.
- Flexible aggregation strategies: *mean*, *sum*, *max*, *min*, etc.
- Multiple similarity metrics (*cosine*, *euclidean*, etc.) for embedding analysis.
- Customizable SHAP calculation algorithms: *exact*, *Monte Carlo approximations*, and more.
- Examples showcasing common explainability pipelines in [`examples/`](https://github.com/Pawlo77/MLLM-Shap/tree/main/examples) on the official GitHub repository.

---

## 💾 Installation

You can install `mllm-shap` directly via pip (**v0.1.8+** recommended):

```bash
pip install mllm-shap
```

Or install it from the source repository:

```bash
git clone https://github.com/Pawlo77/MLLM-Shap.git
cd MLLM-Shap/mllm_shap
pip install .
```

*Note: Core machine learning dependencies like `torch` and `transformers` should be installed in your environment.*

---

## 🚀 Quick Start

Here is a minimal example demonstrating how to initialize an explainer and calculate early SHAP values:

```python
from mllm_shap import MllmShap
from mllm_shap.shap import ExactExplainer
# Import your specific model connector (e.g., for LiquidAudio or similar models)
from mllm_shap.connectors import LLMConnector

# 1. Initialize your model connector
connector = LLMConnector(...)

# 2. Set up the MLLM SHAP explainer with an exact calculation method
explainer = MllmShap(
    connector=connector,
    explainer=ExactExplainer(),
    similarity_metric="cosine"
)

# 3. Calculate SHAP values for an input string
results = explainer.explain("Analyze this text structure.")
print(results)
```

---

## 📊 Visualization & Examples

If you’re interested in GUI visualization of SHAP values, check out the section  **Extension - GUI Visualization** in the repository docs.

For more advanced CLI usages, refer to:

- The [official GitHub repository examples](https://github.com/Pawlo77/MLLM-Shap/tree/main/examples)
- Or explore more advanced pipelines from exemplary [research projects](https://github.com/Pawlo77/MLLM-Shap/tree/main/experiments)

---

<div align="center">
  <h2>🤖 Supported LLM Integrations</h2>
   <br>
   <a href="https://github.com/Liquid4All/liquid-audio/">
       <b>Liquid-Audio</b>
   </a>
   | More robust text, text-to-audio, and audio pipelines coming soon.
</div>
