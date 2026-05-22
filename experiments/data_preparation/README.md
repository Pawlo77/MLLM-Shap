# Data preparation

Notebooks and helpers that build and validate datasets for MLLM-SHAP experiment runs.

**Published outputs:** [Pawlo77/mllm-shap](https://huggingface.co/datasets/Pawlo77/mllm-shap) on the Hugging Face Hub.

## What this folder does

There are two kinds of notebooks:

1. **Build notebooks** — ingest raw sources, apply quality and token-budget filters, synthesize or attach audio, write parquet under `data/`.
2. **Validation notebooks** — use finished Hub data; no new rows are produced.

## Published Hub configs

Naming: `{task}__{source}` (e.g. `single_sentence__voice_bench`). Builders target **`HUB_TARGET_SAMPLES` = 1000** in `src/constants.py`; actual parquet sizes depend on filters (VoiceBench single-sentence **854**, multi-sentence **103**; LibriSpeech ASR **609**; Infinity-Instruct **435** in the last runs).

| Hub config | Origin notebook | Role |
|------------|-----------------|------|
| `single_sentence__voice_bench` | `voice_bench.ipynb` | **854** English single-sentence prompts (target 1k); VoiceBench `audio__original` + Google TTS |
| `multi_sentence__voice_bench` | `voice_bench.ipynb` | **103** multi-sentence prompts (2–8 sentences; target 1k); male/female TTS |
| `single_sentence__librispeech_asr` | `librispeech_asr.ipynb` | **609** single-sentence prompts with **recorded** LibriSpeech audio (target 1k) |
| `multi_lingual__infinity_instruct` | `infinity_instruct.ipynb` | **435** rows (145 prompts × en/fr/es after translation); per-language TTS |

Load any config from Python:

```python
from datasets import load_dataset

ds = load_dataset(
    "Pawlo77/mllm-shap",
    "single_sentence__voice_bench",
    split="test",
)
```

`overview.ipynb` demonstrates loading and playing `audio__male` / `audio__female`.

## Notebooks

### Dataset builders

#### `voice_bench.ipynb`

Source: [hlt-lab/voicebench](https://huggingface.co/datasets/hlt-lab/voicebench).

Produces two Hub splits in one run:

- **`single_sentence__voice_bench`** — NLP interestingness filter, semantic dedup, stratified sampling (target 1k); **854** rows in the last run (`token_count ≤ 10`); `audio__original`, `audio__male`, `audio__female`.
- **`multi_sentence__voice_bench`** — 2–8 sentences per prompt, multi-turn token budget (≤30 tokens); **103** rows in the last run (35% strat pool smaller than 1k target); male/female TTS.

The former ~100-row `single_sentence` split was removed.

#### `librispeech_asr.ipynb`

Source: [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) (`clean` splits).

Produces **`single_sentence__librispeech_asr`** — NLP interestingness filter, semantic dedup, `token_count ≤ 12`, stratified sampling (target 1k); **609** rows in the last run (token filter caps the pool below 1k). Recorded LibriSpeech `audio__original` only (no TTS). Text-first load from `clean` splits (`train.100`, `train.360`, `test`), then attach audio for the final prompt set only.

#### `infinity_instruct.ipynb`

Source: [BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct).

Builds **`multi_lingual__infinity_instruct`** — multilingual rows (en/fr/es) with per-language Google TTS after translation augmentation. With `MAX_TOKEN_COUNT_ML = 20`, the last full notebook run produced **435** saved rows: 145 prompts that pass the token filter, each expanded to three languages (145 per language). Sampling targets `HUB_SAMPLES_PER_LANGUAGE` (= 333) per language but the filtered pool is smaller than the cap.

### Validation / exploration (no ETL)

#### `audio.ipynb`

**SGPA alignment demo.** Loads Hub data, runs `SpectrogramGuidedAligner` on TTS audio, and displays per-token timings and clips.

#### `overview.ipynb`

Random Hub config loader and audio playback for quick checks before large experiments.

#### `google_tts_demo.ipynb`

Explores Google Cloud TTS voices and languages (`TTS_CONFIGS` in `constants.py`). Use before running large synthesis jobs in the builder notebooks.

## Directory layout

```text
experiments/data_preparation/
├── README.md
├── Makefile              # Hub upload shortcuts (make help)
├── upload_to_hub.py
├── hf/
│   └── README.md         # Dataset card published to the Hub repo root
├── data/
│   ├── voicebench/
│   ├── librispeech_asr/
│   └── infinity_instruct/
└── src/
    ├── constants.py          # HUB_TARGET_SAMPLES, Hub config name constants
    ├── save.py               # save_single_sentence, save_multi_sentence, …
    ├── hub_upload.py
    └── …
```

## Typical builder workflow

1. Set pinned dataset revision in `constants.py` (40-char commit hash).
2. Run the relevant builder notebook top to bottom.
3. Inspect `report_dataset_stats()` between sections.
4. Upload parquets and/or the Hub dataset card (see **Hub upload** below).
5. Run `overview.ipynb` and `audio.ipynb` on the new Hub revision.

## Hub upload (automated)

**Authentication:** set `HF_TOKEN` with write access to `Pawlo77/mllm-shap`, or run `hf auth login`.

### Makefile (recommended)

From the **repository root**:

```bash
make -C experiments/data_preparation help
make -C experiments/data_preparation test
make -C experiments/data_preparation list
make -C experiments/data_preparation dry-run
make -C experiments/data_preparation upload-all
make -C experiments/data_preparation upload-readme
make -C experiments/data_preparation upload-all-with-readme
```

Upload one config:

```bash
make -C experiments/data_preparation upload-single-sentence-voice-bench
make -C experiments/data_preparation upload-multi-sentence-voice-bench
make -C experiments/data_preparation upload-single-sentence-librispeech-asr
make -C experiments/data_preparation upload-multi-lingual-infinity-instruct
```

Short aliases: `upload-vb-single`, `upload-vb-multi`, `upload-ls`, `upload-ii`.

Optional:

```bash
make -C experiments/data_preparation upload-vb-single MESSAGE="Rebuild VoiceBench single-sentence"
make -C experiments/data_preparation upload-all CREATE_PR=1
make -C experiments/data_preparation revision
```

You can also `cd experiments/data_preparation` and run `make upload-vb-single` (same targets).

| Artifact | Make target | Local path |
|----------|-------------|------------|
| Dataset card | `upload-readme` | `hf/README.md` |

### CLI (equivalent)

```bash
uv run python experiments/data_preparation/upload_to_hub.py --list
uv run python experiments/data_preparation/upload_to_hub.py --all --dry-run
uv run python experiments/data_preparation/upload_to_hub.py --config single_sentence__voice_bench
uv run python experiments/data_preparation/upload_to_hub.py --readme
uv run python experiments/data_preparation/upload_to_hub.py --all --readme
```

Edit `hf/README.md` (YAML frontmatter + markdown), then publish with `make upload-readme` or bundle with `make upload-all-with-readme`.

| Hub config | Make target | Local parquet |
|------------|-------------|----------------|
| `single_sentence__voice_bench` | `upload-single-sentence-voice-bench` | `data/voicebench/single_sentence__voice_bench.parquet` |
| `multi_sentence__voice_bench` | `upload-multi-sentence-voice-bench` | `data/voicebench/multi_sentence__voice_bench.parquet` |
| `single_sentence__librispeech_asr` | `upload-single-sentence-librispeech-asr` | `data/librispeech_asr/single_sentence__librispeech_asr.parquet` |
| `multi_lingual__infinity_instruct` | `upload-multi-lingual-infinity-instruct` | `data/infinity_instruct/multi_lingual__infinity_instruct.parquet` |

Pin the printed commit hash in `overview.ipynb` (`REVISION`) and `mllm_shapx` configs (`dataset.subset` / `dataset.revision`).

## Prerequisites

- Project venv with `mllm_shap`, `torch`, `datasets`, `transformers`, `pydub`.
- **Google Cloud TTS** credentials for builder notebooks that call `nlp.TTS`.
- **GPU/MPS** optional for token counting and alignment.

## Tests

From the repository root:

```bash
make -C experiments/data_preparation test
```

Covers Hub upload planning, sampling, preprocessing, filters, Infinity-Instruct helpers, save/I/O, NLP/statistics, and constants—without network or GPU.

## Related code

- **SGPA:** `mllm_shap/connectors/base/audio.py`
- **Experiment consumers:** `experiments/mllm_shapx/`, `experiments/faithfulness/`, `experiments/interspeech/`
