# Summary
This paper introduces mllm-shap, an open-source Python platform and interactive Web GUI designed to compute Shapley Value (SV) attributions for text-audio Multimodal Large Language Models (MLLMs). It directly addresses the severe computational and architectural bottlenecks of applying standard SV to multimodal inputs. The authors propose three key solutions: (1) modality-aware masking (e.g., physically dropping text tokens while zeroing audio amplitude), (2) multi-turn conversational metadata tracking, and (3) a Spectrogram-Guided Phonetic Alignment (SGPA) method that aligns dense audio frames into phonetic segments, effectively reducing the SV coalition space by 10-50x. The system provides multiple SV estimators (including a Neyman-optimal allocator) and is released under an MIT license.

# Review
The engineering quality of this demo is excellent, with a well-designed system that includes an asynchronous backend (FastAPI), database support (PostgreSQL), and a responsive React frontend, effectively avoiding timeout issues for heavy computations. The paper is also very clear and well organized, with helpful diagrams and a clear separation between API usage and GUI workflow. In terms of originality, extending Shapley-value-based methods to multimodal text-audio inputs—especially using SGPA to group audio tokens—is a practical and novel solution to scalability challenges. This is also highly significant, as multimodal explainability tools are increasingly needed for audio-based models. The system’s strengths include solving a key computational bottleneck, strong demo-oriented system design (async execution, session persistence, Docker), and a rich, open-source feature set. However, it is only evaluated on a single model, and performance remains slow for longer inputs (e.g., ~400 seconds), limiting real-time interactivity.

# Reasons To Accept
* This is a textbook example of a strong ACL Demo paper. It tackles a timely problem (multimodal interpretability), provides a scientifically sound workaround for the computational complexity (Neyman allocation + SGPA), and wraps it in a robust, production-ready software architecture. The NLP/Speech community will immediately benefit from having a reproducible tool to debug text-audio models.

**Rating:** 8: Top 50% of accepted papers, clear accept

# Reasons To Reject
* The main risk is the unproven generalizability of the connector and masking strategy. The authors demonstrate this on a continuous audio encoder, but it is unclear how smoothly this framework (and specifically the amplitude-zeroing masking strategy) adapts to MLLMs that utilize discrete audio codecs (e.g., models using EnCodec or semantic tokens).

# Questions And Additional Feedback
How difficult would it be for a user to write a BaseMllmModel connector for an MLLM that operates purely on discrete audio codec tokens rather than continuous audio frames? Would the amplitude-zeroing masking strategy still apply, or would it require a token-dropping strategy similar to text?

Have you observed any instances where zeroing the audio amplitude within specific SGPA boundaries creates acoustic "glitches" or artifacts that push the audio encoder severely out-of-distribution, leading to pathological model outputs?

For the ~400-second multi-sentence tasks, does the FastAPI backend effectively hold the WebSocket/polling connection alive, or is the user forced to rely heavily on the PostgreSQL session resumption?

# Meta-Information
* **Needs Ethical Review:** No
* **Reproducibility:** 5: They could easily reproduce the results.
* **Software Or Live Demo:** 5: Enabling: The newly released software / live demo should affect other people's choice of research or development projects to undertake.
* **Datasets:** 4: Useful: I would recommend the new datasets to other researchers or developers for their ongoing work.
* **Overall Assessment:** 8: Top 50% of accepted papers, clear accept
