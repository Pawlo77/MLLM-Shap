# ACL 2026 System Demonstration

**mllm-shap: A Shapley Value Explainability Platform for Multimodal Large Language Models**

Authors: Paweł D. Pozorski\*, Jakub M. Muszyński\*, Maria Ganzha (*equal contribution)

Affiliation: Warsaw University of Technology

## Contents

```
papers/acl_demo/
├── paper/
│   ├── paper.tex          # Main manuscript
│   ├── references.bib     # Bibliography
│   ├── acl.sty            # ACL style
│   └── paper.pdf          # Compiled PDF
├── figures/
│   ├── arch_simple.png
│   ├── artifact_layout.png
│   ├── gui_screenshot.png
│   ├── gui_screenshot_audio.png
│   ├── methods_comparison_by_param.png
│   ├── multi-modal-example.png
│   ├── package_arch.png
│   └── sgpa.png
└── reviews/
    ├── meta_review.md
    ├── official_review_1.md
    ├── official_review_2.md
    └── official_review_3.md
```

## Build

```bash
cd papers/acl_demo/paper
latexmk -pdf paper.tex
```
