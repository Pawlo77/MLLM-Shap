# Summary
This paper presents mllm-shap, a platform for Shapley value (SV)-based explainability of multimodal large language models (MLLMs) that process both text and audio inputs. The work extends prior token-level attribution methods (e.g., TokenSHAP) to multimodal and conversational settings, where inputs may consist of heterogeneous modalities and multiple dialogue turns.

The paper identifies three key challenges in this setting: (1) granularity mismatch between text tokens and dense audio frames, (2) modality-aware masking, where text and audio require fundamentally different removal strategies, and (3) multi-turn conversational structure, where attribution must track roles (user/system) and turns.

To address these, the system introduces a feature-unit abstraction that augments tokens with modality, role, and turn metadata, enabling unified attribution across modalities and dialogue structure. It further proposes modality-aware masking strategies (token removal for text, waveform masking for audio) and a phonetic alignment-based grouping method (SGPA) to reduce the coalition space for audio inputs. The platform implements multiple SV estimators, including a Neyman-optimized allocation strategy for more efficient approximation, and provides an interactive GUI for visualization and analysis.

The system is released as an open-source Python package with a full reproducibility pipeline and is evaluated on 593 attribution runs across multiple multimodal settings on a single GPU.

# Review

**Pros:**
* The paper addresses an important and underexplored problem: explainability for multimodal LLMs. The challenges of extending token-based attribution methods to multimodal and conversational settings are clearly articulated and well-motivated.
* The feature-unit abstraction (tracking modality, role, and turn) is a well-designed conceptual contribution that enables consistent attribution across heterogeneous inputs. The modular pipeline (connectors, masking, utility, estimation) is also well-structured and extensible.
* The proposed modality-aware masking and SGPA-based audio grouping provide reasonable and practical solutions to otherwise intractable issues such as coalition explosion and invalid input distributions.
* The system is highly complete: it includes a pip-installable package, an interactive GUI, asynchronous execution, session persistence, and reproducible infrastructure. This level of engineering maturity is particularly well-suited for the demo track.

**Cons:**
* The package requires a local GPU to run, and requires a bit of backend setup (e.g. requiring that the user deploy with docker compose). This limits the usability and accessibility of the package, although very minor.
* In the demo itself, the runtime telemetry (e.g. GPU utilization, resource usage) looks interesting but I do not think it contributes to the explainability aspect of the demo. I would consider removing it.
* Evaluation focuses on engineering rather than explanation quality. The evaluation primarily measures estimator accuracy (vs. exact SV) and system feasibility (runtime, scalability). However, it does not assess whether the generated explanations are actually useful or meaningful, e.g human evaluation, debugging tasks, or comparison to alternative explainability methods.

This paper presents a well-designed and practically useful system for multimodal explainability, addressing a clear gap in current tooling. While the methodological novelty is limited and the evaluation could be strengthened, the system is comprehensive, reproducible, and likely to be valuable to researchers working with multimodal LLMs.

# Reasons To Accept
* **Clear problem and strong motivation:** The paper addresses an important and underexplored problem: explainability for multimodal LLMs. The challenges of extending token-based attribution methods to multimodal and conversational settings are clearly articulated and well-motivated.
* **Clean and principled system design:** The feature-unit abstraction (tracking modality, role, and turn) is a well-designed conceptual contribution that enables consistent attribution across heterogeneous inputs. The modular pipeline (connectors, masking, utility, estimation) is also well-structured and extensible.
* **Practical handling of multimodal challenges:** The proposed modality-aware masking and SGPA-based audio grouping provide reasonable and practical solutions to otherwise intractable issues such as coalition explosion and invalid input distributions.
* **Strong engineering contribution and completeness:** The system is highly complete: it includes a pip-installable package, an interactive GUI, asynchronous execution, session persistence, and reproducible infrastructure. This level of engineering maturity is particularly well-suited for the demo track.

**Rating:** 7: Good paper, accept

# Reasons To Reject
* **Evaluation focuses on engineering rather than explanation quality:** evaluation primarily measures estimator accuracy (vs. exact SV) and system feasibility (runtime, scalability). However, it does not assess whether the generated explanations are actually useful or meaningful, e.g., via: human evaluation, debugging tasks, or comparison to alternative explainability methods.
* **Limited model diversity:** experiments are conducted on a single multimodal model, which limits the strength of claims about general applicability.
* **Lack of comparison with alternative XAI methods:** the paper does not compare against other explanation approaches (e.g., LIME, Integrated Gradients, attention-based methods), making it difficult to assess relative advantages.

# Questions And Additional Feedback
None

# Meta-Information
* **Needs Ethical Review:** No
* **Reproducibility:** 3: They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
* **Software Or Live Demo:** 4: Useful: I would recommend the new software / live demo to other researchers or developers for their ongoing work.
* **Datasets:** 1: No usable datasets submitted.
* **Overall Assessment:** 7: Good paper, accept
