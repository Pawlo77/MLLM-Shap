<div align="center">
  <h1>📈 Analysis Suite</h1>
  <p><strong>Post-run metrics, visualizations, and statistical interpretation.</strong></p>
</div>

This directory contains analysis notebooks and scripts for outputs produced by `mllm_shapx`.

## Inputs

Most notebooks expect run artifacts under `../experiments_output/`, especially per-sample JSON outputs and aggregated summaries.

## 📊 Notebook Portfolio

- `multi_lingual.ipynb` - multilingual behavior and attribution consistency
- `multi_sentence.ipynb` - long-context/multi-sentence scenario analysis
- `single_sentence.ipynb` - baseline single-sentence attribution metrics
- `sgpa.ipynb` - analysis and figures for SGPA-related experiments

## 🧠 Best Practices

- Keep notebook outputs versioned only when needed for paper artifacts.
- Store large generated assets outside git when possible.
