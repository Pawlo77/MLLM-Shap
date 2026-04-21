# Data Preparation Scripts

This directory contains Jupyter notebooks for creating and processing datasets used in the experiments.

All datasets created here are uploaded to [HuggingFace Datasets](https://huggingface.co/datasets/Pawlo77/mllm-shap).

## Available Notebooks

### 🎙️ Google TTS Demo
[`google_tts_demo.ipynb`](./google_tts_demo.ipynb)
- Demonstrates voice samples across different languages
- Analyzes and compares available TTS models

### 🔄 Infinity Instruct Processing
[`infinity_instruct.ipynb`](./infinity_instruct.ipynb)
- Creates *Multi Turn* and *Multi Lingual* datasets
- Source: [Infinity Instruct Dataset](https://huggingface.co/datasets/BAAI/Infinity-Instruct)

### 🗣️ Voice Bench Processing
[`voice_bench.ipynb`](./voice_bench.ipynb)
- Generates *Multi Sentence* and *Single Sentence* datasets
- Source: [Voice Bench Dataset](https://huggingface.co/datasets/hlt-lab/voicebench)

### 🗣️ Sample Usage
[`overview.ipynb`](./overview.ipynb)
- Showcase how to use one of the created datasets with *HuggingFace*
- Dataset card is [here](https://huggingface.co/datasets/Pawlo77/mllm-shap)
