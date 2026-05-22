# Interspeech 2025

**SGPA: Spectrogram-Guided Phonetic Alignment for Feasible Shapley Value Explanations in Multimodal Large Language Models**

Authors: Paweł D. Pozorski\*, Jakub M. Muszyński\*, Maria Ganzha (*equal contribution)

Affiliation: Warsaw University of Technology

## Contents

```
papers/interspeech/
├── paper/
│   ├── paper.tex              # Main manuscript (camera-ready)
│   ├── template.tex           # Interspeech class template
│   ├── mybib.bib              # Bibliography
│   ├── Interspeech.cls        # Conference style class
│   ├── IEEEtran.bst           # IEEE bibliography style
│   └── paper.pdf              # Compiled PDF
├── figures/
│   ├── sgpa_pipeline.png
│   ├── cumulative_sv.png
│   ├── sv_entropy_by_mode.png
│   ├── stage3_ablation.png
│   └── token_count_distribution.png
├── rebuttal_plan.md           # Reviewer response strategy
├── rebuttal_progress.md       # Implementation tracking
└── review.pdf                 # Reviewer feedback
```

## Build

```bash
cd papers/interspeech/paper
latexmk -pdf -outdir=build_latex paper.tex
```

## Related Code

- SGPA implementation: [`mllm_shap/src/`](../../mllm_shap/src/)
- Experiment runner: [`experiments/mllm_shapx/`](../../experiments/mllm_shapx/)
- Stage 3 ablation: [`experiments/interspeech/`](../../experiments/interspeech/)
- ⚠️ Natural speech pilot (HP-3) — text done, experiment pending
