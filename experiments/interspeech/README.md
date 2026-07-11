# Interspeech Experiments

> **Legacy**: This experiment used the legacy `mllm-shapx` platform and can no longer be executed.

Experiments and analysis for the Interspeech paper submission, covering audio Shapley value attribution with SGPA (Spectrogram-Guided Phoneme Aggregation).

## Contents

- `alpha_search.ipynb` / `src/alpha_search.py` — Hyperparameter search for the alpha weighting parameter
- `sgpa_plot.ipynb` / `src/sgpa_plot.py` — SGPA result visualizations
- `stage3_ablation_plot.ipynb` / `src/stage3/` — Stage 3 ablation study plots
- `rebuttal_dashboard.py` — Streamlit dashboard for rebuttal analysis
- `configs/` — Run configurations for audio attribution experiments (male/female/original voices, raw and SGPA variants)
- `figures/` — Generated figures
- `outputs/` — Experiment outputs
