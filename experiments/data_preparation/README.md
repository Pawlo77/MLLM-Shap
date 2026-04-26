<div align="center">
  <h1>🗃️ Data Preparation</h1>
  <p><strong>Dataset engineering layer for experiment-ready SHAP benchmarks.</strong></p>
</div>

This directory contains notebooks used to build and transform datasets for experiment runs.

Published outputs are available on Hugging Face:
[Pawlo77/mllm-shap](https://huggingface.co/datasets/Pawlo77/mllm-shap)

# 📊 Dataset Pipeline Snapshot

- source families: **Infinity-Instruct** and **VoiceBench**
- output types: multilingual, multi-turn, multi-sentence, single-sentence
- publication target: Hugging Face dataset hub

# 📓 Notebooks

- `google_tts_demo.ipynb`
  - explores multilingual TTS samples
  - compares selected TTS model behavior

- `infinity_instruct.ipynb`
  - prepares multi-turn and multilingual subsets
  - source dataset: [BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)

- `voice_bench.ipynb`
  - prepares single-sentence and multi-sentence subsets
  - source dataset: [hlt-lab/voicebench](https://huggingface.co/datasets/hlt-lab/voicebench)

- `overview.ipynb`
  - quick walkthrough of loading and using prepared datasets
  - useful for sanity checks before large experiment runs
