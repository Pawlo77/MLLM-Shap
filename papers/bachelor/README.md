# Bachelor Thesis (WUT)

**Shapley Value-Based Explainability for Multimodal Large Language Models**

Author: Paweł D. Pozorski

Affiliation: Warsaw University of Technology

## Contents

```
papers/bachelor/
├── paper/
│   ├── main.tex               # Thesis source
│   ├── thesis-content.tex     # Content body
│   ├── references.bib         # Bibliography
│   └── main.pdf               # Compiled PDF
├── figures/                   # All thesis figures
├── helpers/
│   ├── cloud.py               # Word cloud rendering
│   ├── words.txt              # Word cloud input
│   ├── Hierarchical_Example.drawio
│   └── package_structure.ipynb
└── README.md
```

## Build

```bash
cd papers/bachelor/paper
latexmk -pdf main.tex
```
