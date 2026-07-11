# T2T Experiment Execution Plan

This document maps the research scope (see [scope.md](scope.md)) to concrete `mllm_shapx` experiment runs.

## Design Principles

- Each experiment run can answer **multiple** scope bullets via different post-hoc analyses on the same outputs.
- All experiments use **LM Studio on Mac (MPS)** via `lm_studio_text` connector.
- Minimum 3 models × 3 seeds for generalizability.
- Sample sizes guided by Phase 5 power analysis: ≥180 for main claims (d≈0.3, α=0.05, 80% power).

## Models

| Alias | LM Studio Model Key | Rationale |
|-------|-------------------|-----------|
| `gemma3` | `mlx-community/gemma-3-4b-it-4bit` | Google architecture, 4B instruct, primary research model |
| `qwen2.5` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Qwen architecture, 3B instruct, different tokenizer, multilingual |
| `phi-4-mini` | `mlx-community/phi-4-mini-instruct-4bit` | Microsoft Phi-4 Mini, 3.8B, different training methodology |

## Seeds

All experiments use seeds: `42`, `123`, `7`

---

## Experiment Table

| Exp ID | Name | Scope Bullets | Dataset | Samples | Models | Seeds | Prerequisites |
|--------|------|---------------|---------|---------|--------|-------|---------------|
| **T2T-01** | Toy Game SV Validation | P0 (Banzhaf vs Shapley audit), P3 (synthetic games, Precise as truth, Banzhaf/Shapley decision) | `single_sentence_1k` subset (≤5 token prompts) | 20 | gemma3, qwen2.5, phi-4-mini | 42, 123, 7 | None — ready to run. Uses tiny subset where `precise` explainer is feasible. |
| **T2T-02** | Precise vs Approximate Convergence | P3 (PreciseShap truth source, Neyman consistency, MC convergence) | `single_sentence_1k` (≤5 token prompts) | 50 | gemma3, qwen2.5, phi-4-mini | 42, 123, 7 | None — ready to run. Post-hoc: MAE, rank-τ, calibration by coalition size. |
| **T2T-03** | Estimator Budget Sweep | P3 (stopping/uncertainty criteria, Neyman validation), P6 (cost vs fidelity) | `single_sentence_1k` | 200 | gemma3, qwen2.5, phi-4-mini | 42, 123, 7 | None — ready to run. Sweep `num_samples` for MC/CC and `linear` for Neyman. Post-hoc: convergence curves, CI width, wall-clock, rank-stability. |
| **T2T-04** | Utility Function Ablation | P4 (greedy-sim baseline, embedding/reducer ablation, TF-IDF fix), P1 (v(∅)/v(N)) | `single_sentence_1k` | 300 | gemma3, qwen2.5, phi-4-mini | 42, 123, 7 | None — ready to run with existing backends. **Log-probability scorer NOT yet implemented** — that sub-ablation blocked until package extension. |
| **T2T-05** | Faithfulness (single-turn) | P5 (comprehensiveness, sufficiency, negative-SV, monotonicity, AOPC, rank-stability) | `single_sentence_1k` | 500 | gemma3, qwen2.5, phi-4-mini | 42, 123, 7 | T2T-04 results (to pick best utility function). **LIME baseline not yet integrated** — LOO computable from same run. |
| **T2T-06** | Multi-Turn Attribution | P2 (chat-state masking invariants), P5 (multi-turn/system-prompt behavior) | `multi_turn` | 100 | gemma3, qwen2.5, phi-4-mini | 42, 123, 7 | **DATASET INSUFFICIENT** — need ≥180 samples. Expand `multi_turn` dataset before running for main claims. Can run as preliminary (100 samples). |
| **T2T-07** | Prompt-Format Sensitivity | P5 (format sensitivity), P1 (absence semantics) | `single_sentence_1k` (3 format variants per prompt) | 200 | gemma3, qwen2.5, phi-4-mini | 42, 123, 7 | **Reformatted prompt variants not yet created.** Need to create 3 format variants (original, rephrased, delimiter-changed) for ≥200 prompts from single_sentence_1k. |
| **T2T-08** | Determinism & Cache Guard | P2 (cache determinism, connector contract, raw-vs-display) | `single_sentence_1k` | 30 | gemma3 | 42, 123, 7, 999, 2024 | None — ready to run. 5 seeds × 2 temperature settings (0.0, 0.2). Post-hoc: exact reproducibility at T=0, variance at T>0, cache correctness. |

---

## Dataset Sufficiency Assessment

| Dataset | Available | Required | Status |
|---------|-----------|----------|--------|
| `single_sentence_1k` | ~854 | 500 (largest experiment) | **SUFFICIENT** |
| `multi_turn` | ~100 | 180 | **INSUFFICIENT — expand to ≥200** |
| Reformatted prompts (T2T-07) | 0 | 200×3 variants | **NOT CREATED** |
| Constrained generation prompts | 0 | 180 | **NOT CREATED** (Phase 1 prompt suite) |
| Long-context retrieval prompts | 0 | 180 | **NOT CREATED** (Phase 1 prompt suite) |
| Extraction task prompts | 0 | 180 | **NOT CREATED** (Phase 1 prompt suite) |

## Prerequisites Summary

| Blocker | Affects | Action Required |
|---------|---------|-----------------|
| Log-probability value function | T2T-04 (partial) | Implement scorer in `mllm_shap.shap.similarity` |
| LIME baseline | T2T-05 (partial) | Integrate LIME as comparison method |
| FastSHAP surrogate | Phase 3 (not in table) | Implement/adapt FastSHAP for T2T |
| `multi_turn` dataset expansion | T2T-06 | Expand to ≥200 samples in data_preparation |
| Reformatted prompt variants | T2T-07 | Script to create format variants from single_sentence_1k |
| Phase 1 prompt suite | Full Phase 5 coverage | Create QA/extraction/generation/retrieval buckets (50-100 each) |

## Config Structure

All configs live in `experiments/text/configs/`. Each experiment has one base config per model:
```
configs/
├── _base.json              # Shared defaults (device, embedding, shap, generation)
├── _model_gemma3.json      # gemma-3-4b MLX model config
├── _model_qwen25.json      # Qwen2.5-3B MLX model config
├── _model_phi4mini.json    # phi-4-mini MLX model config
├── t2t_01_gemma3.json      # T2T-01 with gemma3
├── t2t_01_qwen25.json      # T2T-01 with qwen2.5
├── t2t_01_phi4mini.json    # T2T-01 with phi-4-mini
├── t2t_02_gemma3.json
├── ...
└── t2t_08_gemma3.json      # T2T-08 only uses gemma3
```

Seeds are encoded as experiment variants via `selection.shuffle_seed` overrides in the `experiments` array.
