<div align="center">
  <h1>🔬 Research Experiments</h1>
  <p>Reproducible experiment configurations, datasets, and execution runners for MLLM-SHAP research.</p>
</div>

---

This directory contains the computational infrastructure and resource code for the various empirical experiments conducted during the development and validation of the package.

### 📁 Directory Layout

- 📊 **[Analysis](analysis/)**: Post-run notebooks and computational scripts generating final plots, statistical significance tests, and results analysis matrices.
- 💾 **[Data Preparation](data_preparation/)**: Scaffolding scripts and notebooks assembling the specialized raw input datasets directly hooked into the evaluation.
- 👻 **[Ghost Busters](ghost_busters/)**: Technical utilities, local server mitigations, and resource management scripts to keep long-running compute jobs clean and localized.
- 🚀 **[MLLM SHAPX](mllm_shapx/)**: The unified experiment orchestration CLI runner, designed for distributed environment deployments and batch inference configuration parsing.
- 📂 **[Experiments Output](experiments_output/)**: The raw `.json` output artifacts, metric traces, and cache checkpoints flushed by active pipeline runs.

<br>
<sub><i>Note: Other `.sh` and `.py` loose files in this directory are helper stubs specific to our institutional SLURM cluster setups and ad-hoc job allocations.</i></sub>
