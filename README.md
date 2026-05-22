<div align="center">
  <h1>🌟 MLLM-SHAP</h1>
  <p><strong>Premium research platform for Shapley explanations in multimodal LLM systems.</strong></p>
  <p>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/v/mllm-shap.svg" alt="PyPI Version"></a>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/pyversions/mllm-shap.svg" alt="Python Version"></a>
    <a href="https://pawlo77.github.io/MLLM-Shap/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit"></a>
  </p>
</div>

# 🧭 Executive Overview

`MLLM-SHAP` combines production-grade package engineering with research-grade experiment tooling.
Repository designed for teams that need explainability across text/audio model pipelines without losing reproducibility.

# ⚡ Quick Start

```bash
pip install mllm-shap
```

For full monorepo environment:

```bash
make install
```

# 🏗️ Repository Architecture

- `mllm_shap/` - package source, tests, docs
- `examples/` - end-to-end notebooks for explainability workflows
- `experiments/` - dataset prep, config runner, analytics
- `paper/` - publication assets and figure pipeline

# 🧰 Tooling Standards

- primary developer interface: `make`
- dependency and env management backend: `uv`
- code quality gates: `pre-commit`, `black`, `isort`, `flake8`
- documentation: Sphinx with autodoc
- packaging and docs pipeline: `pyproject.toml` + Sphinx

# 🔗 Primary Entry Points

- package guide: `mllm_shap/README.md`
- examples guide: `examples/README.md`
- experiments guide: `experiments/README.md`
- contribution rules: `CONTRIBUTING.md`

# 🔬 Research References

- [Bridging Traditional Explainability Methods and Multimodal Multilingual Models](https://zenodo.org/records/19677572)
- [SGPA: Spectrogram-Guided Phonetic Alignment for Feasible Shapley Value Explanations](https://arxiv.org/abs/2603.02250)
- [mllm-shap Zenodo release](https://zenodo.org/records/19678283)

# 📄 License

Apache License 2.0. See `LICENSE`.
