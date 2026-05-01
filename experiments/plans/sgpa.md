# SGPA — Enhanced Research Execution Plan

This plan extends on work for the InterSpeech conference.

All those should be ran on the tuned hyperparameters of the mllm-shap package. We have currently suspicions that current utility function significantly affects the final faithfulness results. We do not have precise estimations on SV approximation accuracy as well.

---

## Phase 1: Core Pipeline & Parameter Refinement
*Objective: Solidify the SGPA pipeline, lock in hyperparameters, and justify low-level design choices before scaling up.*

*   [ ] **Retune hyperparameters on real speech with 3 blind annotators**
    *   **Context:** Current α/β tuning used only synthetic TTS and a single annotator.
    *   **Action:** Redo the grid search on natural LibriSpeech audio with 3 independent annotators (blind to sample metadata), computing inter-annotator agreement (Krippendorff's α or Fleiss' κ). Extend the grid to tune the 40 ms VOT padding.
    *   **Implementation:** Redesign annotation interface — present audio only, no text labels, randomized order. Extended grid: α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, padding ∈ {20, 30, 40, 50, 60} ms. Run on both synthetic and real speech. Report IAA. Directly resolves Limitation 1.
*   [ ] **Add audio processing step**
    *   **Context:** Future step mentioned in paper.
    *   **Action:** Implement a pre-alignment audio processing pipeline: (1) VAD-based trimming of leading/trailing silence, (2) amplitude normalization, (3) optional noise reduction.
    *   **Implementation:** Test whether this reduces boundary flux and improves alignment quality. Report results in a new ablation table.
*   [ ] **Ablation: silence vs. noise masking baseline**
    *   **Context:** SGPA uses zero-amplitude silence as the coalition absence baseline.
    *   **Action:** Test Gaussian noise masking at the ambient noise level of the recording.
    *   **Implementation:** Compare faithfulness scores and click-artifact MOS to empirically justify the masking design choice.
*   [ ] **Quantify Stage 3 fallback rate and its downstream effect**
    *   **Context:** The `boundary_refined` flag exists but its rate and impact are unreported.
    *   **Action:** Compute what % of boundaries fall back to the raw gap midpoint across all corpora.
    *   **Implementation:** Group samples by fallback rate (0%, 1–25%, >25%). Plot faithfulness AUC by group to address Limitation 5 at near-zero additional cost.

## Phase 2: Segmentation Validation & Baselines
*Objective: Rigorously prove that the Stage 2 alignment is high-quality and better than naive alternatives before using it to calculate Shapley Values.*

*   [ ] **Formal CTC alignment quality benchmark**
    *   **Context:** Need to demonstrate the Stage 2 segmentation model is robust on standard benchmarks.
    *   **Action:** Benchmark Stage 2 alignment quality against ground-truth phoneme timestamps using TIMIT or the LibriSpeech forced-alignment release.
    *   **Implementation:** Use torchaudio forced-aligner or Kaldi reference alignments as ground truth. Report boundary error in ms (mean, median, 90th percentile). Report error separately for word-initial vs. word-final boundaries, and for stop consonants vs. continuants.
*   [ ] **Stage 2 ablation: prove CTC is necessary**
    *   **Context:** Reviewers will ask "why CTC specifically?".
    *   **Action:** Demonstrate CTC provides alignment quality beyond naive alternatives.
    *   **Implementation:** Compare against (1) uniform word-duration split, (2) WebRTC VAD-based split, (3) Montreal Forced Aligner. Report boundary spectral flux reduction for each.
*   [ ] **Error analysis by phoneme category**
    *   **Context:** Validates the 40ms VOT padding motivation.
    *   **Action:** Categorize each word boundary by its flanking phoneme type using the CMU Pronouncing Dictionary.
    *   **Implementation:** Report Stage 3 flux reduction and fallback rate by phoneme category. (Re-analysis of existing boundary data).

## Phase 3: Large-Scale Evaluation & Generalizability
*Objective: Run the heavy-compute experiments using the locked-in pipeline to prove feasibility and generalizability.*

*   [ ] **Run all experiments on full datasets**
    *   **Context:** Current numbers are preliminary (n=100).
    *   **Action:** Run all reported experiments on the complete `voice_bench_single_sentence_1k` (854 entries) and `single_sentence_500` (448 entries) corpora.
*   [ ] **Cross-lingual evaluation**
    *   **Context:** SGPA uses multilingual Wav2Vec2-XLSR-53.
    *   **Action:** Test alignment quality and SV attribution on at least 2 additional languages (e.g., German, Spanish) using Common Voice or MLS.
    *   **Implementation:** Report boundary flux reduction and fallback rate by language on a 100-sample dataset per language. Note tonal language limitations. Resolves Limitation 1.
*   [ ] **Benchmark on 2 additional models**
    *   **Context:** Replicate key findings on larger architectures.
    *   **Action:** Benchmark on Kimi Audio, Qwen-Audio, or Gemini 1.5 Flash (API-based).
    *   **Implementation:** API-based model with n=50 samples is fine if compute is limited. Report feasibility table and faithfulness curve. Label as a generalizability probe. Resolves Limitation 2.
*   [ ] **Phrase-level and sub-word granularity experiments**
    *   **Context:** Mentioned as future work.
    *   **Action:** Run a controlled comparison of word-level, phrase-level (noun-phrase chunking via spaCy), and character-level players.
    *   **Implementation:** Run on a 100-sample subset. Report faithfulness AUC and computational cost for each.

## Phase 4: Advanced Analysis & Human Validation
*Objective: Prove that the attributions generated by SGPA are mathematically stable, faithful to the model, and align with human intuition.*

*   [ ] **Faithfulness evaluation on all datasets and modes**
    *   **Context:** The core claim of the paper is better attributions.
    *   **Action:** Measure whether top-attributed word segments, when removed, cause a larger utility drop than bottom-attributed segments.
    *   **Implementation:** Sufficiency + necessity protocol, AUC over masking fraction k. Compare faithfulness curves for SGPA vs. native tokenization.
*   [ ] **Stability analysis: SV estimator variance under SGPA**
    *   **Context:** Prove SGPA yields more reliable attributions.
    *   **Action:** Report SV estimate variance across multiple runs of the Neyman estimator for the same sample.
    *   **Implementation:** Run 20 samples 5 times with different random seeds and temperatures. Report mean absolute deviation of per-word SV across runs.
*   [ ] **Human evaluation of SV explanations (end-user study)**
    *   **Context:** Upgrades the paper to a "user-validated explainability tool".
    *   **Action:** Run a user study (n ≥ 15) asking participants to predict model output from highlighted word segments.
    *   **Implementation:** Use standard ROAR protocol or human simulation task. Compare SGPA vs. native tokenization highlights.

## Phase 5: Theory, Paper Revisions & Reproducibility
*Objective: Final paper polish, theoretical backing, and ensuring open-science standards.*

*   [ ] **Theoretical proposition: SGPA preserves SV efficiency axiom**
    *   **Context:** Elevates paper from empirical to theory-backed.
    *   **Action:** Add a formal proposition showing SGPA's word-level reformulation preserves the SV efficiency axiom.
    *   **Implementation:** Include a short Proposition + Proof in an appendix bounding how much the partition change shifts individual SVs.
*   [ ] **Reframe Section 3 as a reusable benchmark protocol**
    *   **Context:** Increases citation potential.
    *   **Action:** Rewrite Section 3 to explicitly frame the 4 subsections as a benchmark protocol (feasibility, non-neutrality, faithfulness, boundary quality).
    *   **Implementation:** Add an "Evaluation Protocol" paragraph and propose it as a community standard in the Discussion.
*   [ ] **Report both √n and log(n) entropy normalizations**
    *   **Context:** Closes a potential reviewer quibble.
    *   **Action:** Report both normalizations and confirm that the core conclusion holds under both.
    *   **Implementation:** Add one paragraph to Section 3.2 and a supplementary table.
*   [ ] **Figure 1 (Pick better sample)**
    *   **Context:** The current sample is artificial.
    *   **Action:** Record a 2-word voice sample and show the SV profile for that to make it highly illustrative.
*   [ ] **Docker image and pinned environment for reproducibility**
    *   **Context:** Crucial for reviewer validation.
    *   **Action:** Provide a fully pinned Docker image or conda lockfile.
    *   **Implementation:** Pin the exact `mllm-shap` version. Include GPU memory profiling output to empirically confirm the "16 GB VRAM" claim.
