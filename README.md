<div align="center">
  <h1>🌟 MLLM-SHAP</h1>

  <p>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/v/mllm-shap.svg" alt="PyPI Version"></a>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/pyversions/mllm-shap.svg" alt="Python Version"></a>
    <a href="https://pawlo77.github.io/MLLM-Shap/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
    <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="Code style: pre-commit"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  </p>

  <p>
    <b>A robust, comprehensive Python platform for interpreting Multimodal Large Language Models (text-audio) utilizing Shapley values.</b>
  </p>
</div>

---

## 📖 Overview

As Large Language Models rapidly expand into multimodal domains—processing combinations of text, audio, and visual inputs—interpreting their internal decision-making logic becomes exceedingly complex. **MLLM-SHAP** was specifically built to address this exact interpretability gap. By providing a scalable, easy-to-use framework for computing **SHAP (SHapley Additive exPlanations)** values, this library uniquely supports both single-modal and multi-modal inputs and outputs.

Whether you are a researcher analyzing phonetic alignments across complex audio datasets, or an engineer deploying interpretable components for production GenAI workflows, this monorepo houses the entire ecosystem you need: the core libraries, comprehensive experimentation scripts, and documented research results.

## ⚡ Quick Install

To use the core explanation package, simply install it via pip:

```bash
pip install mllm-shap
```
*(For advanced installation guides, see the [Package Documentation](./mllm_shap/README.md) or the [Contribution Guide](./CONTRIBUTING.md))*

## 📚 Repository Structure

*   📁 **[`mllm_shap/`](./mllm_shap/README.md)**: Intended for end-users. This is the main Python package containing the implementation of the algorithms, utilities, core functionality, documentation, and tests.
*   📁 **[`examples/`](./examples/README.md)**: Example scripts and Jupyter notebooks demonstrating how to use the MLLM SHAP package for various tasks.
*   📁 **[`experiments/`](./experiments/README.md)**: Code and resources for research experiments conducted in the project, including data preparation and runner utilities.
*   📁 **[`paper/`](./paper/README.md)**: Materials relating to the documented paper and research project figures.

## 🔬 Academic Research & Results

This project powers and supports several peer-reviewed explainability analyses in multimodal spaces. The following papers and data deployments showcase the library's results:

- 📄 **[Bridging Traditional Explainability Methods and Multimodal Multilingual Models: An XAI-Based Analysis](https://zenodo.org/records/19677572)**
- 📄 **[SGPA: Spectrogram-Guided Phonetic Alignment for Feasible Shapley Value Explanations in Multimodal Large Language Models](https://arxiv.org/abs/2603.02250)**
- 📦 **[mllm-shap: A Shapley Value Explainability Platform for Text-Audio Multimodal Large Language Models](https://zenodo.org/records/19678283)**

## 🔗 Quick Paths

*   📖 **[Root README](./README.md)**
*   🤝 **[Contributing Guidelines](./CONTRIBUTING.md)**
*   📚 **[Sphinx Documentation Sources](./mllm_shap/docs/)**
*   💻 **[Core Package Source Code](./mllm_shap/src/mllm_shap/)**
*   🧪 **[Tests Suite](./mllm_shap/tests/)**
*   📊 **[Interactive Examples](./examples/)**
*   🔬 **[Research & Experiments](./experiments/)**

---

<div align="center">
  <sub>Built with ❤️ by the MLLM-SHAP contributors. Distributed under the MIT License.</sub>
</div>
