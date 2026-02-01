Release Notes
=============

Version 0.1.7
-------------
* Introduced `StandardComplementaryNeymanShapExplainer`, a new explainer that utilizes standard Neyman allocation for SHAP value estimation. Existing `ComplementaryNeymanShapExplainer` has been renamed to `LimitedComplementaryNeymanShapExplainer` to reflect its limited sampling strategy.
* Internal refactoring and bug fixes to improve code maintainability and performance.

Version 0.1.6
-------------
* Includes fixes for complementary-based explainers to ensure correct SHAP value computations.
* Improved documentation and added more comprehensive tests for all modules.
* Small fixes for bugs discovered in previous versions during production use.
* Ensures minimal fraction is respected in `HierarchicalExplainer` when using importance sampling.

Version 0.1.5
-------------
* Added `ComplementaryNeymanShapExplainer`, a new explainer leveraging Neyman allocation for more efficient sampling.
* Refactored base classes to improve code clarity, maintainability, and facilitate future extensions.
* Fixed minor bugs in `ComplementaryShapApproximation` to ensure accurate SHAP value calculations.
* Fixes bug in `PreciseShapExplainer` related to generator dead-lock.
* Introduced new modes in `HierarchicalExplainer` for more flexible explanation strategies, including:
  - Multi-modal explanations.
  - Business-aware first-level splits (at the cost of performance).
  - Importance-aware approximation budgets (for each group) for better resource allocation.

Version 0.1.4
-------------
* Added `ComplementaryShapApproximation`, an explainer using complementary contribution sampling for faster SHAP value estimation.
* Enhanced global tests to cover new functionalities and improve robustness.

Version 0.1.3
-------------
* Added `HierarchicalExplainer`, an explainer using hierarchical sampling to accelerate SHAP value estimation for long text inputs (text modality only).
* Refactored base classes to improve code organization and maintainability.

Version 0.1.2
-------------
* Added Monte Carlo-inspired explainers:
  - `LimitedMcShapExplainer`: uses limited sampling strategies to efficiently estimate SHAP values.
  - `LimitedMcShapExplainer` (restricted to unique masks): improves sampling efficiency by avoiding duplicates.
* Fixed issues in the `connectors` module to enhance stability and performance.

Version 0.1.1
-------------
* Initial release of **MLLM-SHAP**, a library for SHAP value estimation in Multi-Modal Large Language Models (MLLMs).
* Added `PreciseShapExplainer` for exact SHAP value computation.
* Introduced the `connectors` module with support for the `Liquid Audio LMF-2` multi-modal model.
