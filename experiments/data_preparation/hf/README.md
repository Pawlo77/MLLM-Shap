---
license: apache-2.0
language:
- en
- fr
- es
pretty_name: MLLM-SHAP Benchmarks
size_categories:
- n<1K
task_categories:
- text-generation
- question-answering
tags:
- multimodal
- explainability
- shapley-values
- audio
- text-to-speech
- benchmark
- english
- multilingual
configs:
- config_name: single_sentence__voice_bench
  description: >-
    English single-sentence prompts from VoiceBench, filtered for NLP quality and
    a short explainability token budget (≤10). Includes source TTS audio plus
    Google Cloud TTS male/female resyntheses.
  features:
  - name: datasets
    list: string
  - name: sentences
    list: string
  - name: token_count
    dtype: int32
  - name: audio__original
    list: binary
  - name: audio__original__duration
    list: float32
  - name: audio__male
    list: binary
  - name: audio__male__duration
    list: float32
  - name: audio__female
    list: binary
  - name: audio__female__duration
    list: float32
  data_files:
  - split: test
    path: single_sentence__voice_bench/test/*
- config_name: multi_sentence__voice_bench
  description: >-
    English multi-sentence prompts (2–8 sentences) from VoiceBench, kept within a
    multi-turn explainability token budget (≤30). Includes source audio and
    male/female TTS.
  features:
  - name: datasets
    list: string
  - name: sentences
    list: string
  - name: sentences__num
    dtype: int32
  - name: token_count
    dtype: int32
  - name: audio__original
    list: binary
  - name: audio__original__duration
    list: float32
  - name: audio__male
    list: binary
  - name: audio__male__duration
    list: float32
  - name: audio__female
    list: binary
  - name: audio__female__duration
    list: float32
  data_files:
  - split: test
    path: multi_sentence__voice_bench/test/*
- config_name: single_sentence__librispeech_asr
  description: >-
    English single-sentence prompts aligned with LibriSpeech ASR transcripts
    (recorded speech, no synthetic TTS). Used to study explanations on natural
    audio rather than TTS-only stimuli.
  features:
  - name: datasets
    list: string
  - name: sentences
    list: string
  - name: token_count
    dtype: int32
  - name: audio__original
    list: binary
  - name: audio__original__duration
    list: float32
  data_files:
  - split: test
    path: single_sentence__librispeech_asr/test/*
- config_name: multi_lingual__infinity_instruct
  description: >-
    Multilingual one-turn conversations (English, French, Spanish) derived from
    Infinity-Instruct. Each base prompt is translated into all three languages;
    per-language Google TTS (male/female). No source recording column.
  features:
  - name: labels
    list: string
  - name: language
    dtype: string
  - name: n_messages
    dtype: int32
  - name: sentences__num
    dtype: int32
  - name: sentences
    list: string
  - name: original_language
    dtype: string
  - name: token_count
    dtype: int32
  - name: audio__male
    list: binary
  - name: audio__male__duration
    list: float32
  - name: audio__female
    list: binary
  - name: audio__female__duration
    list: float32
  data_files:
  - split: test
    path: multi_lingual__infinity_instruct/test/*
---

# MLLM-SHAP experiment datasets

Curated **test** splits for studying [Shapley-value explanations](https://github.com/Pawlo77/MLLM-Shap) in multimodal large language models (text and audio inputs). Each configuration is a filtered, size-controlled subset built for reproducible benchmarking—not a full copy of the upstream corpora.

Configs follow the naming pattern `{task}__{source}` (for example `single_sentence__voice_bench`).

## Quick load

Pin a dataset revision for reproducibility (replace `REVISION` with the commit hash printed after upload or from the Hub **History** tab):

```python
from datasets import load_dataset

REVISION = "main"  # or a 40-character commit hash

ds = load_dataset(
    "Pawlo77/mllm-shap",
    "single_sentence__voice_bench",
    split="test",
    revision=REVISION,
)
row = ds[0]
# row["sentences"]     — list of prompt strings (one or more sentences)
# row["token_count"]   — explainability token budget (LiquidAudio)
# row["audio__male"]   — list[bytes] WAV clips (when present)
```

The [`mllm-shap`](https://pypi.org/project/mllm-shap/) experiment runner loads the same layout via `{config}/test/0000.parquet`.

## Configurations

| Config | Source | Rows (last publish) | Languages | Audio columns |
|--------|--------|---------------------|-----------|---------------|
| `single_sentence__voice_bench` | [VoiceBench](https://huggingface.co/datasets/hlt-lab/voicebench) | **854** | `en` | `audio__original`, `audio__male`, `audio__female` |
| `multi_sentence__voice_bench` | [VoiceBench](https://huggingface.co/datasets/hlt-lab/voicebench) | **103** | `en` | `audio__original`, `audio__male`, `audio__female` |
| `single_sentence__librispeech_asr` | [LibriSpeech ASR](https://huggingface.co/datasets/openslr/librispeech_asr) | **609** (target 1k) | `en` | `audio__original` (recorded) |
| `multi_lingual__infinity_instruct` | [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) | **435** (145 × 3 languages) | `en`, `fr`, `es` | `audio__male`, `audio__female` |

Row counts are below the nominal **1,000**-sample target when quality or token filters shrink the eligible pool. Rebuilding from the [data preparation notebooks](https://github.com/Pawlo77/MLLM-Shap/tree/main/experiments/data_preparation) may change these numbers slightly.

### `single_sentence__voice_bench`

- **Task:** one sentence per prompt; English only.
- **Filters:** embedding-based interestingness, semantic deduplication, `token_count ≤ 10`, stratified sampling toward 1k.
- **Audio:** VoiceBench `audio__original` plus Google Cloud TTS (British English male/female).

### `multi_sentence__voice_bench`

- **Task:** 2–8 sentences per prompt; English only.
- **Filters:** `token_count ≤ 30`, stratified sampling (35% by dataset × sentence count, then token-balanced draw toward 1k).
- **Audio:** same three columns as the single-sentence VoiceBench split.

### `single_sentence__librispeech_asr`

- **Task:** one sentence per prompt; English read speech from LibriSpeech ASR (`clean`: `train.100`, `train.360`, `test`).
- **Filters:** embedding-based interestingness, semantic deduplication, `token_count ≤ 12`, stratified sampling toward 1k — **609** rows in the last publish (only 609 prompts pass the token budget).
- **Audio:** recorded `audio__original` only (no TTS columns).

### `multi_lingual__infinity_instruct`

- **Task:** one human turn per row; labels and metadata from Infinity-Instruct.
- **Languages:** each of 145 base prompts appears in **en**, **fr**, and **es** after translation augmentation (**435** rows total).
- **Filters:** strict multi-turn token budget (`token_count ≤ 20` on the base table).
- **Audio:** per-language Google TTS (`audio__male`, `audio__female`).

## Legacy configs

Older Hub config names (`single_sentence`, `single_sentence_1k`, `single_sentence_500`, `multi_sentence`, `multi_lingual`) are **deprecated** in favor of the `{task}__{source}` configs above. The `multi_turn` config may still exist on the Hub for historical runs but is not rebuilt by the current preparation pipeline.

## Citation

If you use these splits, please cite the [MLLM-SHAP](https://github.com/Pawlo77/MLLM-Shap) repository and the corresponding upstream datasets (VoiceBench, LibriSpeech ASR, Infinity-Instruct). A Zenodo archive is linked from the project README.

## License

Apache 2.0 for this dataset packaging. Upstream corpora remain subject to their original licenses and terms of use.
