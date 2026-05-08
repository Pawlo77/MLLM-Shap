# Other — Infrastructure and Tooling Improvements

This file tracks engineering, tooling, and infrastructure tasks that support the research workflows but do not belong to any specific research plan.

All items here are improvements to the development environment, experimentation framework, or package internals. They should be picked up opportunistically alongside active research phases or during dedicated cleanup sprints.

---

## Phase 1: Data and Artifact Management
*Objective: clean up the repository and establish reliable artifact storage so experiment outputs are reproducible and traceable.*

*   [ ] **Remove results from the repo and store in proper location**
    *   **Context:** Large result files committed to the repository inflate clone size and make history noisy.
    *   **Action:** Move all result artifacts out of the repository and into a designated external storage location.
    *   **Implementation:** Agree on a storage convention (e.g. DVC, cloud bucket, or shared network path) and update experiment scripts to read/write there instead of committing to git.
*   [ ] **Add pipeline versioning and artifact metadata**
    *   **Context:** Pre-fix and post-fix experiment outputs can be mixed when the pipeline changes without a version bump.
    *   **Action:** Tag every artifact with the pipeline version, git commit, and config hash that produced it.
    *   **Implementation:** Embed a metadata sidecar alongside every result file so outputs from different pipeline versions are never silently compared.

## Phase 2: Experimentation Framework
*Objective: make the experimentation framework produce richer, more structured outputs that support deeper analysis.*

*   [ ] **Upgrade experimentation framework output format**
    *   **Context:** Current outputs lack per-sample hardware traces and intermediate artifacts needed for debugging and auditing.
    *   **Action:** Extend the framework to record hardware usage per sample and to save output masks and their model responses as explicit artifacts.
    *   **Implementation:** Emit one structured record per sample containing predictions, masks, hardware metrics, and the full artifact set.
*   [ ] **Improve benchmark reporting**
    *   **Context:** Current run reports do not capture enough metadata to reproduce or compare runs reliably.
    *   **Action:** Record wall-clock time, memory usage, model config, explainer config, and seed in a single machine-readable bundle for every run.
    *   **Implementation:** Write the bundle as a JSON or YAML sidecar next to each result file and include a schema definition.

## Phase 3: Package Performance and Footprint
*Objective: reduce the computational and memory cost of the package so larger experiments are feasible on consumer hardware.*

*   [ ] **Optimise package for better performance**
    *   **Context:** Mask generation and sampling stages are the current bottlenecks for large-scale experiments.
    *   **Action:** Profile both stages and apply targeted optimizations (vectorization, batching, caching) where the most time is spent.
    *   **Implementation:** Establish a micro-benchmark suite so regressions are caught before merging.
*   [ ] **Minimise RAM footprint for 10k+ call explanations**
    *   **Context:** High memory usage prevents running large explanations on consumer hardware.
    *   **Action:** Identify and eliminate unnecessary tensor copies and intermediate buffers during explanation computation.
    *   **Implementation:** Target a memory profile that allows 10 000+ coalition evaluations without exceeding typical consumer DRAM limits.

## Phase 4: Visualization and GUI
*Objective: provide richer visual summaries so results are easier to inspect and communicate.*

*   [ ] **Add notebook-friendly visualization for faithfulness and attribution results**
    *   **Context:** Current outputs require custom plotting code; mature SHAP libraries ship visual summaries out of the box.
    *   **Action:** Implement inline notebook visualizations for faithfulness curves and attribution heatmaps comparable in quality to SHAP's built-in plots.
    *   **Implementation:** Cover at least bar plots, waterfall plots, and faithfulness-curve overlays with sensible defaults.
*   [ ] **Add richer GUI views for SHAP value comparison**
    *   **Context:** The GUI does not yet expose estimator variance or faithfulness curves alongside raw and normalized SHAP values.
    *   **Action:** Extend the GUI with views for comparing raw vs. normalized SHAP values, estimator variance bands, and faithfulness curves.
    *   **Implementation:** Reuse existing GUI infrastructure; add the new views as additional tabs or panels without breaking current workflows.
