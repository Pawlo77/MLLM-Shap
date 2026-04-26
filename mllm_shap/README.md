<div align="center">
  <h1>📦 mllm-shap Package</h1>
  <p><strong>Core engine for multimodal SHAP attribution in text/audio LLM pipelines.</strong></p>
  <p>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/v/mllm-shap.svg" alt="PyPI"></a>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/pyversions/mllm-shap.svg" alt="Python"></a>
    <a href="https://pawlo77.github.io/MLLM-Shap/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Docs"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  </p>
</div>

# ✨ Product Highlights

- exact and approximate SHAP explainers under single API surface
- text/audio-ready connector model with typed abstractions
- configurable normalizers, reducers, similarity backends
- package-level test and docs structure for stable iteration

# 📊 Package Snapshot

- **Main modules**: `connectors`, `shap`, `utils`
- **Explainability modes**: precise + Monte Carlo families + advanced variants
- **Target runtime**: Python 3.12

# 💾 Installation

Install from PyPI:

```bash
pip install mllm-shap
```

Install from source:

```bash
git clone https://github.com/Pawlo77/MLLM-Shap.git
cd MLLM-Shap/mllm_shap
pip install .
```

# 🧱 Package Layout

- `src/mllm_shap/connectors/` - model and chat connectors
- `src/mllm_shap/shap/` - explainers, attribution internals, result objects
- `src/mllm_shap/utils/` - utility helpers
- `tests/` - package verification suite
- `docs/` - documentation sources

# 🚀 Usage and Ecosystem

- notebook workflows: `../examples/README.md`
- experiment runner integration: `../experiments/mllm_shapx/README.md`
- GUI companion for visualization: [shap-mllm-explainer](https://github.com/mvishiu11/shap-mllm-explainer)

# 📄 License

MIT License.
