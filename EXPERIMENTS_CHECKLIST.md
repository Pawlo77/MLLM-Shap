# AAAI-27 SGPA — Experiment Checklist

Living checklist for the SGPA (exact Shapley) experiment plan. Boxes are ticked
as each run/artifact completes. Monitor live with `./monitor_jobs.sh`.

**Models (both complete):**

- **Primary:** Qwen2-Audio-7B-Instruct (4-bit) — the FIRST model.
- **Cross-family:** Voxtral-Mini-3B-2507 (Mistral family, 4-bit) — replaces the
scrapped Phi-4; demonstrates SGPA generalizes across model families.

**LFM2-Audio is scrapped** (unreliable results, removed from cache).
**SV:** exact Shapley via package `PreciseShapExplainer` · **Utility:** held-out
E5 embedding cosine (+ TF-IDF) · **Segmentation:** SGPA (`SpectrogramGuidedAligner`).

Pipelines: `./run_queue.sh` (Qwen, 7 jobs + auto-analysis) and
`./run_second_model.sh` (Voxtral, 7 jobs). Consolidated per-model tables/figures
→ `.../qwen_analysis/` and `.../voxtral_analysis/`; cross-model comparison →
`.../cross_model_analysis/`.

**STATUS: all 14 GPU runs finished `rc=0`; per-model + cross-model analyses done.**

*Last updated: 2026-07-28 07:33 — all runs complete; analyses regenerated.*

---

## Q0 — Infrastructure (Path B)

- [x] Refactor SV to package `PreciseShapExplainer` (validated vs brute-force, max |diff| ~1e-8)
- [x] Length-matched random-deletion baseline + paired stats (Cohen's dz, Wilcoxon) via `summarize._summarize_delta`
- [x] Per-segment `segment_dur_sec` column for duration matching
- [x] Word-banded exactly-100 sample selection (`select_word_banded_ids`, 4–7 words); full-pool for TTS voices, token-balanced 100 preserved for the original run
- [x] Tooling: `monitor_jobs.sh` (dashboard), `run_queue.sh` + `run_second_model.sh` (sequential GPU queues)
- [x] Model-agnostic runner `mm_faith.py` + `AudioTextBackend` protocol (`backends.py`)

## Q1 — Faithfulness (primary)  [Qwen + Voxtral]

- [x] **Q1.1** MAIN faithfulness table (top-|SV| vs length-matched random), LibriSpeech, 100 — Qwen dz=0.94, p≈6e-13; Voxtral dz=1.13, p≈4e-15
- [x] **Q1.4a** TTS male, 100 (both models)
- [x] **Q1.4b** TTS female, 100 (both models)
- [x] **Q1.4c** TTS original, 100 (both models)
- [x] **Q1.4** Voice-dependence table (`*_voice_table.csv`, both models)
- [x] **Q1.2** AOPC / top-k curves (SGPA order vs random) → `*_aopc_*.png`; all gaps positive (Qwen 0.030–0.043, Voxtral 0.041–0.059)
- [x] **Q1.3** Rank-wise Spearman monotonicity, full n → appendix panel (both models: median within-ρ 0.80, ~97% positive; `appendix_analysis.py` → `appendix_monotonicity.png`)
- [x] **Q1.5** Stage-3-off faithfulness (raw CTC vs refined, fixed word players), 100 (both models)

## Q2 — Supporting results

- [x] **Q2.1** Feasibility framing (exact SV tractable via SGPA) + attribution stats (Gini/entropy/top-20%) → `*_attribution_stats.csv`
- [x] ~~**Q2.2** Estimator stability on LFM2~~ — **CANCELLED**: LFM2 scrapped; exact Shapley is deterministic, no estimator to stabilize (folds into feasibility framing).
- [x] **Q2.3** Faithfulness stratified by `boundary_refined` fallback bins (0% / 1–25% / >25%) — positive in every bin, both models (>25% bin: Qwen dz=1.16, Voxtral dz=1.56); `appendix_fallback_strata.png`

## Q3 — Masking ablation + second model

- [x] **Q3.1** Masking ablation: silence vs ambient-noise vs delete+concat (60 each, both models) — faithfulness holds under all three masking strategies
- [x] **Q3.2** SECOND model = **Voxtral-Mini-3B-2507** (Mistral family; native `transformers 5.4`, replaced incompatible Phi-4). Full 7-condition suite in `outputs/voxtral/`; cross-family faithfulness confirmed (matches/exceeds Qwen).

## Cross-model

- [x] Side-by-side main table + AOPC-gap bar chart + AOPC overlay → `outputs/cross_model_analysis/` (`compare_models.py`)

## Already done (feeds TODO items)

- [x] `boundary_refined` fallback-rate table (Male 61% / Female 67% / LibriSpeech 61% segment fallback)
- [x] SGPA aligner utterance-collapse bug fixed; exact-SV path validated vs brute-force (~1e-8)

## Write-up

- [x] Reframe paper around Qwen2-Audio (primary) + Voxtral (cross-family generalization); **LFM2-Audio removed entirely**
- [x] Pull tables/figures into `paper.tex` (feasibility, faithfulness, masking, Stage-3, appendix monotonicity + fallback); new figs `faithfulness_main.png`, `appendix_monotonicity.png`, `appendix_fallback_strata.png`; bib entries added (Qwen2-Audio, Voxtral, E5)
- [x] Compiles clean: 7 pages total (content+appendix+refs), no undefined refs, overfull <15pt

---

### Output locations


| Condition                       | Qwen dir                                  | Voxtral dir                                  |
| ------------------------------- | ----------------------------------------- | -------------------------------------------- |
| LibriSpeech-original (MAIN)     | `outputs/qwen_exact_shapley_original/`    | `outputs/voxtral/exact_shapley_original/`    |
| TTS male                        | `outputs/qwen_exact_shapley_male/`        | `outputs/voxtral/exact_shapley_male/`        |
| TTS female                      | `outputs/qwen_exact_shapley_female/`      | `outputs/voxtral/exact_shapley_female/`      |
| TTS original                    | `outputs/qwen_exact_shapley_ttsorig/`     | `outputs/voxtral/exact_shapley_ttsorig/`     |
| Stage-3-off (raw CTC)           | `outputs/qwen_exact_shapley_stage3off/`   | `outputs/voxtral/exact_shapley_stage3off/`   |
| Masking: noise                  | `outputs/qwen_exact_shapley_mask_noise/`  | `outputs/voxtral/exact_shapley_mask_noise/`  |
| Masking: delete+concat          | `outputs/qwen_exact_shapley_mask_concat/` | `outputs/voxtral/exact_shapley_mask_concat/` |
| **Consolidated tables/figures** | `outputs/qwen_analysis/`                  | `outputs/voxtral_analysis/`                  |
| **Cross-model comparison**      | `outputs/cross_model_analysis/`           | (same)                                       |


(paths relative to `experiments/faithfulness/`). Each run dir holds a
`*_results.csv` (per-segment), `*_coalitions.csv` (full coalition→utility table,
enables any-order AOPC), and `*_summary.json` (aggregate stats + paired test).
Analysis is regenerated with `qwen_analyze.py --model {qwen,voxtral}` and
`compare_models.py`.

### Third-model note

Adding a further audio LM = one new backend class exposing
`generate_text(audio_np, instruction)` + a 4-bit loader; the rest (SGPA aligner,
exact SV via `PreciseShapExplainer`, utilities, `summarize`, queue, analysis) is
reused unchanged.
