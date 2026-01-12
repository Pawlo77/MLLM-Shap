# Experiments Analysis (single / multilingual / multi-sentence)

This folder contains lightweight Python utilities and notebooks used to analyze experiment outputs saved under `experiments/experiments_output/`.

## Where results must live

All analyses expect results in this layout:

- `experiments/experiments_output/<RUN_NAME>/<CASE_DIR>/samples/sample_XXXXX_result.json`

Examples (already in this repo):

- `experiments/experiments_output/multi_lingual_2026_01_03/text_text_limited_neyman_lin3_0/samples/…`
- `experiments/experiments_output/multi_sentence_2026_01_03/text_text_limited_neyman_lin3_0/samples/…`

`<RUN_NAME>` is a dated folder (e.g. `multi_lingual_2026_01_03`).
`<CASE_DIR>` encodes modality + method configuration (e.g. `text_text_limited_neyman_lin3_0`).

## Minimal JSON schema (what the analysis needs)

Each `sample_*_result.json` must contain:

Top-level fields:

- `row_index`: int (unique within the run/case)
- `runtime_sec`: float
- `n_calls`: int
- `neyman_steps`: int
- `prompt_texts`: list[str] (1 item for single-sentence, N items for multi-sentence)
- `input_modality`: str (e.g. `text`, `audio__male`, `audio__female`, `audio`)
- `output_modality`: str (e.g. `text`)
- `attr_summary`: object with at least:
  - `count_text_tokens`: int
  - `count_audio_segments`: int
- `conversation`: list[turn]

`conversation` format:

- `conversation` is a list of turns.
- Each turn is a list of message objects.
- Each message object must have:
  - `content_type`: int (`0` for text tokens, `1` for audio payload)
  - `content`: list (strings for text tokens; audio payload objects for audio)
  - `roles`: list[int] (same length as `content`)
  - `shap_values`: list[float | null] (same length as `content`)

Important invariant used by the analysis:

- Any non-null `shap_values` should correspond to USER-role tokens (same check as in `single_sentence.py`).

Audio-attribution format (only for audio inputs):

- `audio_segments`: object mapping a segment-group key (commonly `"2"`) to a list of segments.
- Each segment must have:
  - `token`: str
  - `shap_value`: float

## What you prepare for analysis

1. Run experiments so that they write `sample_*_result.json` files under a run folder.
2. Ensure `row_index` is consistent across cases so you can compare modalities for the same sample.
3. For multilingual runs, include language metadata per sample:
   - `language` (the language of the presented prompt)
   - `original_language` (if prompts are translated / derived)

The loaders in this folder validate the above structure and will raise a clear error if something is missing or inconsistent.
