# Summary
The paper presents mllm-shap, a system that extends Shapley-value-based explainability from text LLMs to text–audio multimodal LLMs, since no existing tool supports Shapley explanations for multimodal LLMs. They provide multiple estimator and utility functions.

# Review
This package addresses a genuine gap in XAI tooling for multimodal LLMs. The system is well-engineered, the paper is clearly written, and the contribution is practical and useful. The platform provides multiple SV estimators (including a Neyman-optimal variant) and utility functions, along with an interactive web GUI.

# Reasons To Accept
* The paper identifies three challenges (granularity mismatch, modality-aware masking, conversational structure) and addresses each one with specific technical solutions.
* The modular architecture separating connectors, masking, utility functions, and estimators is well-designed. The SGPA integration allows 43× coalition-space reduction is impressive.
* 593 analyses across multiple configurations (VoiceBench, Infinity-Instruct, three languages) demonstrate thorough evaluation.

**Rating:** 9: Top 15% of accepted papers, strong accept

# Reasons To Reject
* While the architecture appears modular, the paper lacks discussion of how complicated it is to add a custom estimator or utility function

# Questions And Additional Feedback
See Reasons to Reject

# Meta-Information
* **Needs Ethical Review:** No
* **Reproducibility:** 5: They could easily reproduce the results.
* **Software Or Live Demo:** 5: Enabling: The newly released software / live demo should affect other people's choice of research or development projects to undertake.
* **Datasets:** 1: No usable datasets submitted.
* **Overall Assessment:** 9: Top 15% of accepted papers, strong accept
