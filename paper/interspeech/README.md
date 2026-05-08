# Interspeech 2025 — SGPA Paper

**SGPA: Spectrogram-Guided Phonetic Alignment for Feasible Shapley Value Explanations in Multimodal Large Language Models**

Authors: Paweł D. Pozorski\*, Jakub M. Muszyński\*, Maria Ganzha
(*equal contribution)

Affiliation: Warsaw University of Technology

## Abstract

End-to-end audio language models process speech without explicit linguistic structure, making it difficult to identify which parts of an utterance drive a given response. We introduce **Spectrogram-Guided Phonetic Alignment (SGPA)**, a four-stage pipeline that combines CTC forced alignment with spectral boundary refinement to produce acoustically stable, word-aligned audio segments. SGPA yields a **43× reduction** in model evaluations and is validated on LFM2-Audio-1.5B with VoiceBench.

## Directory Layout

```
paper/interspeech/
├── paper/
│   ├── paper.tex              # Main manuscript (camera-ready)
│   ├── template.tex           # Interspeech class template
│   ├── mybib.bib              # Bibliography
│   ├── Interspeech.cls        # Conference style class
│   ├── IEEEtran.bst           # IEEE bibliography style
│   └── paper.pdf              # Compiled PDF
├── figures/
│   ├── sgpa_pipeline.png      # SGPA 4-stage pipeline diagram
│   ├── cumulative_sv.png      # Cumulative Shapley value profiles
│   ├── sv_entropy_by_mode.png # SV entropy comparison
│   ├── stage3_ablation.png    # Stage 3 ablation results
│   └── token_count_distribution.png
├── rebuttal_plan.md           # Reviewer response strategy
├── rebuttal_progress.md       # Implementation tracking
└── review.pdf                 # Reviewer feedback
```

## Building the Paper

Requires a LaTeX distribution (TeX Live / MacTeX) with `latexmk`:

```bash
cd paper/interspeech/paper
latexmk -pdf -outdir=build_latex paper.tex
```

The compiled PDF is written to `build_latex/paper.pdf`.

To clean build artifacts:

```bash
latexmk -C -outdir=build_latex paper.tex
```

## Key Results

| Metric | Raw Tokenization | SGPA |
|--------|-----------------|------|
| Mean players per sample | ~150 frames | ~7 words |
| Model evaluations per sample | ~2,552 | ~59 |
| Speedup factor | — | **43×** |
| Stage 3 spectral flux reduction | — | 42.7% (p < 1e-300, d = 0.92) |

## Related Experiment Code

- SGPA pipeline implementation: [`mllm_shap/src/`](../../mllm_shap/src/)
- Experiment configs & runner: [`experiments/mllm_shapx/`](../../experiments/mllm_shapx/)
- Stage 3 ablation: [`experiments/interspeech/`](../../experiments/interspeech/)
- Published package: [mllm-shap on PyPI](https://pypi.org/project/mllm-shap/)

## Rebuttal Status

See [`rebuttal_progress.md`](rebuttal_progress.md) for detailed tracking. Summary:

- ✅ Stage 3 ablation (HP-2)
- ✅ Novelty framing (HP-4)
- ✅ Entropy normalization (MP-1)
- ✅ SV objective definition (MP-2)
- ✅ Shapley basics paragraph (MP-3)
- ✅ CTC blank handling (MP-4)
- ✅ Hyperparameter protocol (MP-6)
- ❌ Faithfulness test (HP-1) — Experiment A pending
- ⚠️ Natural speech pilot (HP-3) — text done, experiment pending
