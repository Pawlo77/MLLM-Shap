# Future Works — Extended Research Directions

This file keeps broader research directions that sit outside the immediate execution plans in `sgpa.md` and `sv_analysis.md`.

All those should be treated as second-wave or later-stage work. Some of them depend on stabilizing the current T2T and audio pipelines first, while others are intentionally broader questions aimed at future papers rather than the next implementation cycle.

We should keep this file for work that is one of the following:
- outside the current package-validation scope
- cross-modal or cross-lingual rather than connector-specific
- exploratory enough that it should not block current research claims

Items already fully scoped in active plans are not repeated here: masking ablations and segmentation benchmarks are in `sgpa.md` Phase 1–2; cross-lingual evaluation and multi-architecture benchmarks are in `sgpa.md` Phase 3; prompt-format sensitivity is in `sv_analysis.md` Phase 5.

---

## Phase 1: Audio and Multimodal Foundations
*Objective: extend the current explainability framework beyond text-only analysis and make multimodal claims defensible.*
*Depends on: `sgpa.md` Phase 1–2 (pipeline stabilisation, masking ablations, independence check) being complete.*

*   [ ] **Study multimodal interaction values (pairwise, tractable scope)**
    *   **Context:** Standard marginal attributions miss effects that appear only when text and audio are present together. Shapley Interaction Indices (SII) exist but are expensive; scaling to full utterances is not yet tractable.
    *   **Action:** Explore pairwise SII between individual audio segments and individual text tokens on a controlled task where the cross-modal interaction signal is deliberately injected (e.g., a number spoken in audio whose written form also appears in the prompt text).
    *   **Scope constraint:** Limit to n ≤ 15 total players to stay tractable. Do not scale to full-utterance explanations until a cheaper approximation is validated. Specify which interaction index (SII, Banzhaf Interaction, FSII) before running.
    *   **Success criterion:** The cross-modal pairwise SII detects the planted interaction and marginal SVs alone do not; demonstrate on at least 10 controlled examples.
    *   **Dependency:** Requires stable marginal SV estimates from `sgpa.md` Phase 4.

*   [ ] **SHAP estimator K-sensitivity for audio (broader validation)**
    *   **Context:** `sgpa.md` Phase 4 validates K for the current model and dataset. This item extends that to understand whether K requirements shift across encoder types, languages, and task families — providing a generalizable rule rather than a one-off calibration.
    *   **Action:** Repeat the K ∈ {50, 100, 200, 500, 1000} sweep across the Phase 2 encoder variants and cross-lingual samples from `sgpa.md` Phase 3.
    *   **Success criterion:** Identify whether a single universal K threshold is sufficient across settings or whether K must be tuned per-configuration.
    *   **Dependency:** Requires `sgpa.md` Phase 3 and Phase 4 K-sensitivity item to be complete.

*   [ ] **Evaluate explanation quality beyond faithfulness**
    *   **Context:** Faithfulness (comprehensiveness/sufficiency AUC and AOPC) is necessary but not sufficient. The Jacovi & Goldberg (2020) taxonomy distinguishes faithfulness (does the explanation reflect the model's actual computation?) from plausibility (does it match human intuition?). These are orthogonal: an explanation can be faithful but implausible (the model uses features humans find unintuitive) or plausible but unfaithful (the explanation is post-hoc rationalisation). The plans must not conflate them.
    *   **Action:** Add three additional metrics beyond faithfulness: (1) sparsity as Gini coefficient of absolute SV distribution; (2) plausibility as overlap with the segment-importance annotations collected in `sgpa.md` Phase 4 human study — the human study is explicitly designed to collect these annotations so this metric requires no additional data collection; (3) OOD stability as AOPC on AudioSet-corrupted samples.
    *   **Success criterion:** All three metrics reported alongside faithfulness curves. The paper explicitly labels which metrics measure faithfulness and which measure plausibility, following Jacovi & Goldberg (2020) terminology.
    *   **Dependency:** Plausibility metric requires `sgpa.md` Phase 4 human study (which collects the needed annotations). OOD stability and sparsity require only `sgpa.md` Phase 3 full-dataset runs.

## Phase 2: Generalization Across Models, Tasks, and Modalities
*Objective: understand whether observed attribution behavior is model-specific or general across broader settings not covered by `sgpa.md` Phase 3.*

*   [ ] **Generalize across audio encoder backbones**
    *   **Context:** SGPA uses Wav2Vec2-XLSR-53 for CTC alignment. Whether explanation quality transfers to Whisper-encoder-aligned segments or EnCodec tokenized audio is unknown and directly affects adoption by practitioners who use different encoders.
    *   **Action:** Re-run the faithfulness and boundary-quality benchmark from `sgpa.md` with at least one alternative encoder (Whisper-tiny or EnCodec) as the segmentation backbone.
    *   **Success criterion:** Either faithfulness AUC difference is within noise across encoder choice, or identify which encoder type yields systematically better attributions and document the trade-off.
    *   **Dependency:** Requires `sgpa.md` Phase 2 benchmarks as baseline. Independent of language generalization.

*   [ ] **Generalize across audio tasks beyond transcription and QA**
    *   **Context:** All current experiments target transcription-adjacent or question-answering tasks. Attribution behavior on sentiment classification, speaker identification, and emotion recognition may differ fundamentally because the relevant signal is prosodic (pitch, energy, rhythm) rather than lexical. If SGPA only works on lexical tasks, its practical scope is much narrower than claimed.
    *   **Action:** Run SGPA on at least one prosody-heavy task (sentiment or emotion recognition) and compare which segment types (phoneme-boundary vs. syllable-stress regions) receive high attribution.
    *   **Success criterion:** Demonstrate whether SGPA faithfulness holds on prosodic tasks or explicitly document where the word-level reformulation breaks down.
    *   **Dependency:** Independent of `sgpa.md`; requires a model that handles the target task.

*   [ ] **Benchmark against gradient-based and attention-based attribution baselines**
    *   **Context:** The paper currently compares SGPA against native-tokenization SHAP and LOO. Reviewers at INTERSPEECH, ACL, or NeurIPS will additionally require Integrated Gradients, SmoothGrad, and attention-based attribution, which are the standard baselines in audio and multimodal interpretability literature. For attention baselines, specify which variant: raw attention weights (shown unreliable by Jain & Wallace, 2019), attention rollout (Abnar & Zuidema, 2020), or attention-gradient products (Hao et al., 2021). These produce substantially different attributions and "attention-based attribution" is not a single method. LIME for T2T is handled directly in `sv_analysis.md` Phase 5.
    *   **Action:** Implement or adapt IG and one named attention variant for the same audio-text model. Evaluate on the shared faithfulness benchmark from `sgpa.md` Phase 4 using both AOPC and AUC.
    *   **Success criterion:** Either SGPA Pareto-dominates on faithfulness vs. compute, or the paper explicitly characterises conditions under which each method is preferable.
    *   **Dependency:** Requires `sgpa.md` Phase 3 evaluation set to be locked before adding baselines.

*   [ ] **Study model-size scaling effects on explanation quality**
    *   **Context:** All primary experiments use one model size. Whether explanation faithfulness degrades with larger models (due to longer effective context, stronger cross-layer entanglement, or increased instruction-following capability overriding feature importance) is unknown and is a likely reviewer question.
    *   **Action:** Compare faithfulness AUC and SV rank stability across at least two model scales (e.g., 1B vs. 7B parameters in the same model family).
    *   **Success criterion:** Determine whether model scale is a confound in faithfulness claims or orthogonal to them. If confounded, note it as a limitation.
    *   **Dependency:** Requires `sgpa.md` Phase 3 evaluation protocol as baseline.

*   [ ] **Analyze how models trade off modalities under attribution**
    *   **Context:** The current research measures per-modality attribution magnitude but does not characterise when the model systematically ignores one modality. Knowing which conditions cause audio vs. text dominance has practical interpretability value.
    *   **Action:** Study how SV magnitude ratios (audio total vs. text total) shift across controlled inputs where the dominant modality is known in advance (e.g., contradictory audio and text content, or audio-only information tasks).
    *   **Success criterion:** Identify at least two distinct attribution regimes (audio-dominant, text-dominant) and characterise the input features that determine which regime applies.
    *   **Dependency:** Independent of `sgpa.md`; can run in parallel with Phase 1.

*   [ ] **Study fine-tuning effects on explanations**
    *   **Context:** Two models with the same base architecture may explain very differently after task-specific fine-tuning. Practitioners need to know whether deploying a fine-tuned checkpoint changes explanation structure, not only prediction quality.
    *   **Action:** Compare attribution patterns before and after fine-tuning, operationalised as Kendall rank correlation of per-segment SVs on the same held-out inputs. Use a base checkpoint and a task-specific fine-tuned checkpoint from the same model family.
    *   **Success criterion:** Determine whether fine-tuning shifts the SV distribution in a predictable direction (e.g., increased concentration on task-relevant tokens) or unpredictably, and report the magnitude of the shift.
    *   **Dependency:** Independent; requires access to a base and fine-tuned checkpoint pair.

## Phase 3: Longer-Horizon Method Development
*Objective: explore more ambitious algorithmic directions once the current framework is stable enough to support them.*

*   [ ] **Learn coalition-sampling policies beyond fixed Neyman heuristics**
    *   **Context:** Static sampling rules may not be sample-efficient enough for harder tasks or larger player counts.
    *   **Action:** Explore learned sampling policies that predict which coalitions are most informative, starting with a lightweight regression on variance predictors (coalition size, player entropy, prior SV magnitude) before moving to bandit-style approaches.
    *   **Key design questions to resolve before implementation:** (1) whether the policy generalises across inputs (global policy, trained once) or must be learned per-input (per-prompt policy — expensive and potentially circular); (2) how to evaluate policy quality independently of the SHAP estimate to avoid circular validation; (3) whether the learned policy preserves Shapley estimand consistency or introduces systematic bias.
    *   **Success criterion:** The learned policy reaches the same MAD target as Neyman at ≤ 60% of coalition evaluations on a held-out prompt set.
    *   **Dependency:** Requires `sv_analysis.md` Phase 3 Neyman consistency validation as baseline before any policy comparison is meaningful.

*   [ ] **Explore hierarchical grouping strategies**
    *   **Context:** Flat token-level explanations are costly and hard to interpret in longer contexts. The phrase-level granularity experiments in `sgpa.md` Phase 3 are the direct predecessor; this item extends that to dynamic, learned, or linguistically-motivated groupings.
    *   **Action:** Study hierarchical grouping over linguistic structures (constituency parse, discourse units) or multimodal structures (prosodic phrases aligned to syntactic phrases).
    *   **Implementation:** Compare computational savings and faithfulness tradeoffs against the flat SHAP baseline from `sgpa.md` Phase 3.
    *   **Success criterion:** Demonstrate at least 30% compute reduction without measurable faithfulness loss on the shared evaluation set.
    *   **Dependency:** `sgpa.md` Phase 3 phrase-level granularity experiments are the direct prerequisite.

*   [ ] **Extend utility functions to next-token distribution scoring**
    *   **Context:** Current text scoring uses greedy-sequence similarity or log-probability of a fixed base response. Both are blind to cases where the model's output distribution shifts materially while the top-1 response stays unchanged. A distribution-aware metric captures this but requires tractable approximation.
    *   **Action:** Implement a prefix-conditional KL divergence scorer that compares next-token distributions under coalition S vs. full prompt, averaged over the first L tokens of the base response (cap L at 20 to bound extra calls).
    *   **Implementation:** Validate that the KL score correlates with sequence-level faithfulness before promoting it to a default. Demonstrate on at least 20 examples where greedy similarity scores 1.0 but token probability mass has shifted significantly.
    *   **Success criterion:** KL scorer detects cases that greedy similarity misses; correlation between KL score and sufficiency/comprehensiveness AOPC is statistically significant.
    *   **Dependency:** Requires `sv_analysis.md` Phase 4 log-probability value function to be implemented and validated first.

*   [ ] **Contrastive/counterfactual explanations**
    *   **Context:** Standard SHAP answers "why does the model predict X?" Contrastive SHAP answers "why X and not Y?" — a substantially different quantity that is more aligned with how humans reason about model decisions (e.g., "why did the model transcribe this word as A rather than B?"). This direction is increasingly expected in XAI papers and is absent from all current plans.
    *   **Action:** Define a contrastive characteristic function `v_contrast(S) = v(S | target=A) − v(S | target=B)` for a pair of competing outputs. Compute contrastive SVs on controlled examples where the competing outputs are known. Compare whether contrastive SVs identify different features than absolute SVs.
    *   **Success criterion:** Demonstrate at least one setting where absolute SVs are uniform across segments but contrastive SVs are concentrated — showing that contrast reveals structure that absolute attribution misses.
    *   **Dependency:** Requires `sgpa.md` Phase 4 faithfulness evaluation to be complete as a reference point.

*   [ ] **Clarify explanation layer: input space vs. latent space**
    *   **Context:** All current plans apply SHAP at the input level (audio segments or text tokens). But the MLLM processes inputs through many stages: audio encoder, cross-modal projection, text decoder. SHAP applied at the input measures input-level importance; it does not localise which processing stage is responsible for the attribution. Two methods can assign identical input-level SVs while explaining completely different internal mechanisms — one because the audio encoder ignores those segments, the other because the decoder ignores them.
    *   **Action:** Design one experiment that compares input-level SGPA SVs against encoder-output-level SVs (by treating the encoder's segment representations as the players rather than the raw audio segments). Determine whether input-level and encoder-level attributions agree or diverge, and what the divergence implies about where the MLLM processes the audio information.
    *   **Success criterion:** Either input-level and encoder-level attributions agree (validating that input-level explanations reflect encoder processing), or the divergence is characterised and reported as a scope limitation of input-level SHAP.
    *   **Dependency:** Independent; requires access to encoder intermediate representations (API-only models cannot support this).
