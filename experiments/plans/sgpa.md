# SGPA — Enhanced Research Execution Plan

This plan extends on work for the InterSpeech conference.

All those should be ran on the tuned hyperparameters of the mllm-shap package. We have currently suspicions that current utility function significantly affects the final faithfulness results. We do not have precise estimations on SV approximation accuracy as well.

We should use 4 datasets there:
- voice_bench single_sentence of 500 entries 6-10 tokens length, 100 per length, english only, original audio + male tts + female tts
- librispeech single_sentence of 500 entries 6-10 tokens length, 100 per length, english only, original audio
- 2 additional languages that differs significantly in phonetic structure and have proper models and datasets available. LMF2 should have had those languages in the training data, so we can expect good performance. *According to gemini, it was trained on English, Arabic, Chinese, French, German, Japanese, Korean, Spanish - suggestion is to use Japanese and Arabic. AP1 for details*.

> For now either connector to LFM2-Audio doesn't work correctly, or that model doesn't work correctly. It should be replaced before any of the below experiments. This might require choosing different languages as well, depending on which ones are supported by the working model and its training corpus.

We should use a primary + replication model matrix. Any claim in the paper must hold on more than one model or be explicitly scoped as model-specific:

- **Primary model:** the working replacement for LFM2-Audio (see Phase 0). All pipeline design, hyperparameter tuning, masking ablations, and segmentation validation work runs here first.
- **Secondary model A:** a different architecture family from the primary (e.g., if primary is encoder-decoder, secondary A should be decoder-only or vice versa). Used to replicate core faithfulness and stability results.
- **Secondary model B:** API-based (Kimi Audio, Qwen-Audio, or Gemini 1.5 Flash). Used as a compute-constrained replication probe. API-only status means gradient attribution baselines are not available for this model — note this explicitly.

Division of work across phases:
- **Phases 1–2** (pipeline design, ablations, segmentation): primary model only. Ablation choices (masking strategy, CTC vs. naive) are design decisions that should generalise, but running all ablations on every model is not cost-effective at this stage.
- **Phase 3** (large-scale evaluation): all three models. Main faithfulness and boundary-quality claims must be run on primary + secondary A with power-analysis justified sample sizes. Secondary B at reduced n is acceptable as a supplementary probe.
- **Phase 4** (faithfulness, stability): primary model for the full evaluation; core results (faithfulness AUC, AOPC, SV rank stability) must be replicated on secondary A before any claim is presented as general. A claim that holds only on the primary model is a claim about that model, not about SGPA.

---

## Phase 0: Unresolved Blockers
*These must be resolved before any experiment is run. Every phase below depends on them.*

*   [ ] **Lock the full model matrix: primary + secondary A + secondary B (hard blocker)**
    *   **Context:** The LFM2-Audio connector does not work correctly, and the entire plan currently has no confirmed model. Beyond replacing LFM2, the plan requires a declared secondary model (different architecture family) at this stage because the choice of secondary model affects: dataset language coverage, whether gradient attribution baselines are feasible (API-only models block this), and whether the cross-lingual evaluation plan needs redesign. Choosing the secondary model at Phase 3 is too late — the datasets and evaluation design must account for it from the start.
    *   **Action:** (1) Identify a working primary audio-text model with documented training languages and a functioning local connector. (2) Choose secondary model A from a different architecture family; confirm it supports local inference so gradient baselines are possible. (3) Choose secondary model B as an API-accessible model (Kimi Audio, Qwen-Audio, or Gemini 1.5 Flash) and document that gradient baselines are unavailable for it. (4) Verify all three produce coherent outputs on at least 10 manual test samples across all three audio types before any timed experiment.
    *   **Success criterion:** All three models are named, connectors verified, training language lists documented, and API-only limitations noted. Update Appendix A language choices to reflect what the primary model actually supports.

*   [ ] **Resolve Banzhaf vs. Shapley identity of existing results (data integrity)**
    *   **Context:** `sv_analysis.md` Phase 3 correctly flags that unweighted coalition averaging is not automatically Shapley. If the current MC path estimates Banzhaf values and existing results are labelled as Shapley Values, those results are wrong. This must be resolved before any further experiments are run under the existing pipeline.
    *   **Action:** Audit the coalition weighting in the current MC explainer. Confirm whether it implements Shapley weighting (`|S|!(n-|S|-1)!/n!`) or uniform weighting (Banzhaf). If uniform, relabel all existing results as Banzhaf or recompute with correct Shapley weights.
    *   **Success criterion:** Every result in the paper is correctly labelled with the estimand it approximates. No result labelled "Shapley" uses uniform coalition weighting.

*   [ ] **Define the SGPA coalition game precisely for the multimodal setting**
    *   **Context:** SHAP axioms apply to a well-defined characteristic function v: 2^N → ℝ. In a multimodal model with both audio segments and text tokens, it is not defined what a "coalition" is. Are text tokens always fully present (pure audio game)? Are both modalities players simultaneously (joint game)? Different answers yield different games with different SVs, and none of the current plans specify which one is intended.
    *   **Action:** State explicitly: (1) the player set N (audio segments only, text tokens only, or both); (2) what "absent player" means for each modality (masked audio segment, deleted text token, or something else); (3) the exact formula for v(∅) and v(N). Write this as a one-page formal definition that every experiment references.
    *   **Success criterion:** Every reported SV can be mapped to a named, formally stated game. Reviewers can check whether the game is well-formed before evaluating the estimates.

---

## Phase 1: Core Pipeline & Parameter Refinement
*Objective: Solidify the SGPA pipeline, lock in hyperparameters, and justify low-level design choices before scaling up.*
*Depends on: Phase 0 blockers fully resolved.*

*   [ ] **Verify approximate additivity of the value function over adjacent audio segments (prerequisite)**
    *   **Context:** SHAP's independence requirement is that the marginal contribution of player i — `v(S∪{i}) − v(S)` — does not depend systematically on whether i's neighbours are already in S. This is a property of the characteristic function v, not a property of the audio signal. Testing temporal autocorrelation between waveforms or mutual information between segment features is the wrong test — it measures input similarity, not value-function structure.
    *   **Action:** For a 30-sample diagnostic set, compute `v(S∪{i}) − v(S)` for segment i across coalitions S that include vs. exclude i's left and right neighbours. Measure whether the marginal contribution of i is systematically higher or lower when its neighbours are absent (superadditivity) vs. present (subadditivity). If strong (super/sub)additivity is detected, this indicates the independence assumption fails at word granularity.
    *   **Implementation:** Compare the word-level and phrase-level Shapley estimates; if strong additivity violations exist at word level, phrase-level grouping should reduce them. Gate the rest of Phase 1 on the outcome.
    *   **Success criterion:** Mean absolute difference in marginal contributions between neighbour-in and neighbour-out coalitions is below 10% of the SV range, or default granularity is switched to phrase level.
*   [ ] **Retune hyperparameters on real speech with 3 blind annotators**
    *   **Context:** Current α/β tuning used only synthetic TTS and a single annotator.
    *   **Action:** Redo the grid search on natural LibriSpeech audio with 3 independent annotators (blind to sample metadata), computing inter-annotator agreement (Krippendorff's α or Fleiss' κ). Extend the grid to tune the 40 ms VOT padding.
    *   **Implementation:** Redesign annotation interface — present audio only, no text labels, randomized order. Extended grid: α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, padding ∈ {20, 30, 40, 50, 60} ms. Run on both synthetic and real speech. Report IAA. Directly resolves Limitation 1.
*   [ ] **Add audio processing step**
    *   **Context:** Future step mentioned in paper.
    *   **Action:** Implement a pre-alignment audio processing pipeline: (1) VAD-based trimming of leading/trailing silence, (2) amplitude normalization, (3) optional noise reduction.
    *   **Implementation:** Test whether this reduces boundary flux and improves alignment quality. Report results in a new ablation table.
*   [ ] **Ablation: audio processing pipeline on boundary quality**
    *   **Context:** VAD trimming and amplitude normalisation may reduce the boundary flux that motivates Stage 3. Whether this preprocessing interacts with the masking strategy choice is unknown.
    *   **Implementation:** Run the audio processing step ablation jointly with the masking strategy ablation below so both ablations share the same audio samples and reduce total experiment cost.
*   [ ] **Quantify Stage 3 fallback rate and its downstream effect**
    *   **Context:** The `boundary_refined` flag exists but its rate and impact are unreported.
    *   **Action:** Compute what % of boundaries fall back to the raw gap midpoint across all corpora.
    *   **Implementation:** Group samples by fallback rate (0%, 1–25%, >25%). Plot faithfulness AUC by group to address Limitation 5 at near-zero additional cost.

## Phase 2: Segmentation Validation & Baselines
*Objective: Rigorously prove that the Stage 2 alignment is high-quality and better than naive alternatives before using it to calculate Shapley Values.*

> **Invalidation risk:** If the Masking Strategy ablation below shows that ambient-noise replacement yields materially better faithfulness than deletion+concatenation (the current baseline), all existing faithfulness numbers are biased and must be rerun with the better masking before submission. Treat the ablation result as a go/no-go gate on using current numbers.

*   [ ] **CTC alignment quality benchmark and necessity ablation (joint experiment)**
    *   **Context:** Two related questions require the same ground-truth phoneme timestamps and the same audio samples: (1) is CTC alignment accurate in absolute terms? (2) is CTC better than naive alternatives? Running them as separate experiments doubles annotation and compute cost for no benefit — they are two output columns of the same table.
    *   **Action:** On the same set of samples from TIMIT or the LibriSpeech forced-alignment release, run CTC alongside all naive baselines: (1) uniform word-duration split, (2) WebRTC VAD-based split, (3) Montreal Forced Aligner. Report in a single table: boundary error in ms (mean, median, 90th percentile) for each method, spectral flux reduction for each method, and error separately for word-initial vs. word-final boundaries and for stop consonants vs. continuants.
    *   **Success criterion:** CTC achieves strictly lower boundary error than all three baselines AND lower spectral flux. If MFA matches CTC quality, justify why CTC is still preferred (e.g., computational cost, no lexicon requirement).
*   [ ] **Error analysis by phoneme category**
    *   **Context:** Validates the 40ms VOT padding motivation.
    *   **Action:** Categorize each word boundary by its flanking phoneme type using the CMU Pronouncing Dictionary.
    *   **Implementation:** Report Stage 3 flux reduction and fallback rate by phoneme category. (Re-analysis of existing boundary data).
*   [ ] **Ablation: Masking Strategy and Temporal Confound**
    *   **Context:** Standard segment removal (deletion + concatenation) destroys natural prosody and introduces a systematic "position shift confound" for early segments, while silence padding may be out-of-distribution (OOD) for the encoder.
    *   **Action:** Conduct a formal comparison of three masking baselines:
    *   1. **Deletion + Concatenation:** (Current baseline)
    *   2. **Silence Padding:** (Preserves temporal structure)
    *   3. **Ambient Noise Replacement:** (Preserves structure and reduces OOD risk)
    *   **Implementation:** Compare all three on faithfulness AUC and AOPC (Area Over Perturbation Curve) — both metrics must be reported since the community uses AOPC as the primary faithfulness measure and AUC alone prevents direct comparison to prior work. Additionally: (a) measure click-artifact MOS for silence and noise conditions to empirically justify the acoustic quality of each masking strategy; (b) probe encoder activation shifts on silence-padded vs. natural vs. noise-padded audio to test the OOD risk. Run jointly with the audio processing ablation above on the same sample set.

## Phase 3: Large-Scale Evaluation & Generalizability
*Objective: Run the heavy-compute experiments using the locked-in pipeline to prove feasibility and generalizability.*

*   [ ] **Run power analysis before committing to sample sizes**
    *   **Context:** The current sample sizes (n=100 per language, n=50 for model benchmarks, n=20 for K-sensitivity) were not derived from power calculations. For a paired faithfulness test with a realistic effect size of d≈0.3 at 80% power and α=0.05 you need approximately 180 samples. n=100 is likely underpowered for moderate effects.
    *   **Action:** Run a prospective power analysis for each main comparison: SGPA vs. baseline faithfulness AUC (paired), cross-lingual boundary quality (multi-group), and model benchmark feasibility. Use an assumed effect size from a pilot on 20 samples.
    *   **Success criterion:** Every reported sample size is explicitly justified by a power calculation. If current counts are insufficient, adjust before running full experiments.
*   [ ] **Run all experiments on full datasets**
    *   **Context:** Current numbers are preliminary (n=100).
    *   **Action:** Run all reported experiments on the complete `voice_bench_single_sentence_1k` (854 entries) and `single_sentence_500` (448 entries) corpora.
*   [ ] **Cross-lingual evaluation**
    *   **Context:** SGPA uses multilingual Wav2Vec2-XLSR-53.
    *   **Action:** Test alignment quality and SV attribution on at least 2 additional languages (e.g., German, Spanish) using Common Voice or MLS.
    *   **Implementation:** Report boundary flux reduction and fallback rate by language on a 100-sample dataset per language. Note tonal language limitations. Resolves Limitation 1.
    *   **Multiple comparisons note:** Results across languages and models form a large comparison matrix. Apply FDR correction (Benjamini-Hochberg) across all pairwise tests before reporting significance.
*   [ ] **Replicate core claims on secondary model A and secondary model B**
    *   **Context:** All Phase 1–2 design choices were made on the primary model. Before claiming SGPA generalises, the core results must be reproduced on secondary A (different architecture family) with adequate sample sizes. A result that holds only on the primary model is a finding about that model, not about the method.
    *   **Action:** Run the following on secondary A at power-analysis-justified sample sizes: (1) faithfulness AOPC and AUC — the primary result of the paper; (2) SV rank stability — the reliability claim; (3) the winning masking strategy from Phase 2, to verify it is not primary-model-specific. Run the same on secondary B (API model) at reduced n with explicit power caveats. Report results in a single stratified table: primary / secondary A / secondary B side by side.
    *   **Acceptance criterion:** The ranking of SGPA vs. baselines (native tokenization, LOO) must hold on secondary A. If the ranking reverses or disappears on secondary A, the paper's generalisation claim must be revised or removed before submission. Secondary B results are supplementary evidence only.
*   [ ] **Stratify all results by TTS vs. natural speech**
    *   **Context:** VoiceBench uses original audio, male TTS, and female TTS for the same sentences. TTS has cleaner boundaries, more regular prosody, and a lower noise floor. If SGPA works better on TTS than natural speech and results are pooled, the method appears more general than it actually is.
    *   **Action:** Report faithfulness AUC, boundary quality, and SV rank stability separately for TTS and natural speech subsets in every experiment table.
    *   **Success criterion:** Either performance is statistically equivalent across audio type (show the test), or the paper explicitly acknowledges and quantifies the TTS advantage as a limitation.
*   [ ] **Phrase-level and sub-word granularity experiments**
    *   **Context:** Mentioned as future work.
    *   **Action:** Run a controlled comparison of word-level, phrase-level (noun-phrase chunking via spaCy), and character-level players.
    *   **Implementation:** Run on a 100-sample subset. Report faithfulness AUC and computational cost for each.

## Phase 4: Advanced Analysis & Human Validation
*Objective: Prove that the attributions generated by SGPA are mathematically stable, faithful to the model, and align with human intuition.*
*Model coverage: full evaluation runs on the primary model. Faithfulness and stability results must additionally be replicated on secondary A (as required by the Phase 3 replication item) before any cross-cutting claim is submitted. Results that differ across models must be reported, not averaged away.*

*   [ ] **Faithfulness evaluation on all datasets and modes**
    *   **Context:** The core claim of the paper is better attributions. Faithfulness here means the Jacovi & Goldberg (2020) sense — does the explanation reflect what the model actually computed, not merely what seems plausible to humans? This is distinct from plausibility (human agreement), which is measured separately in the human study below.
    *   **Action:** Measure whether top-attributed word segments, when removed, cause a larger utility drop than bottom-attributed segments. Report three baselines: (1) SGPA, (2) native tokenization SHAP, (3) LOO attribution (v(N) − v(N\{i})) — the simplest possible attribution, which is the universal minimum baseline any method must beat.
    *   **Metrics:** Report both AOPC (Area Over Perturbation Curve) and AUC over masking fraction k. AOPC is the primary community metric; AUC enables fine-grained comparison. Both must be reported for the paper to be directly comparable to prior work.
    *   **Implementation:** Removal must be simultaneous-at-each-k (all top-k segments removed at once per evaluation point), not sequential greedy — sequential removal creates path-dependencies that make AUC values incomparable across methods and papers.
    *   **Circularity check (mandatory):** Confirm that the utility function used to evaluate faithfulness is NOT the same as the utility function used to compute SHAP values. If they are the same, the faithfulness test is circular — SHAP values were optimised to minimise exactly that residual. If they must share a common base, evaluate faithfulness under a second independent utility function as a cross-check.
*   [ ] **Stability analysis: SV estimator variance under SGPA**
    *   **Context:** Prove SGPA yields more reliable attributions. Two independent sources of variance exist and must be measured separately: (1) SHAP estimator variance (due to random coalition sampling at fixed K) and (2) model output variance (due to non-zero generation temperature). Mixing them into a single run produces a number that cannot be attributed to either source.
    *   **Action:** Run two separate protocols: (a) temperature=0, 5 runs with different SHAP estimator seeds — measures pure estimator variance; (b) fixed SHAP seed, 5 runs with temperature>0 — measures pure model output variance. Report MAD separately for each source.
    *   **Implementation:** Run on 20 samples per protocol. This directly informs whether variance reduction should target the estimator (more samples K) or the model (temperature=0 for research runs).
    *   **Shared protocol note:** This stability protocol is the audio equivalent of the T2T monotonicity and rank-stability checks in `sv_analysis.md` Phase 5. Both must use identical metrics (MAD, top-k Kendall τ, sign consistency) so that audio and text stability numbers are directly comparable in a joint results table.
*   [ ] **K-sensitivity: validate coalition sample count for audio**
    *   **Context:** All faithfulness and stability claims depend on K (coalition samples) being sufficient for audio inputs, but the current K is ported from text settings without audio-specific validation. Coalition value variance can be materially higher for audio than for text, making the text-derived K insufficient.
    *   **Action:** Run K ∈ {50, 100, 200, 500, 1000} on 20 fixed audio samples and plot mean absolute deviation of per-segment SVs vs. K.
    *   **Implementation:** Report the K at which MAD falls below 5% of the SV range. Confirm that all reported experiments use at least this K, or rerun with the validated K if not.
    *   **Success criterion:** Published experiment K is validated against the MAD threshold. Any result section that relies on an under-K setting must be rerun before submission.
*   [ ] **Human evaluation of SV explanations (end-user study)**
    *   **Context:** Upgrades the paper to a "user-validated explainability tool." The output of this study also directly feeds the plausibility metric in `future_works.md` Phase 1 — participant annotations of which segments they consider important become the ground truth for overlap-based plausibility scoring. Running these separately would mean collecting human annotations twice on the same data.
    *   **Action:** Run a user study (n ≥ 15) asking participants to predict model output from highlighted word segments. Collect both a forward-task score (predict output from highlights) and a highlight-agreement annotation (which segments would you highlight yourself?) so the same study serves both the current-paper human evaluation and the future plausibility metric.
    *   **Implementation:** Use standard ROAR protocol or human simulation task. Compare SGPA vs. native tokenization highlights vs. LOO attribution highlights.

## Phase 5: Theory, Paper Revisions & Reproducibility
*Objective: Final paper polish, theoretical backing, and ensuring open-science standards.*

*   [ ] **Theoretical proposition: SGPA defines a valid word-level coalition game**
    *   **Context:** The SV efficiency axiom (∑φᵢ = v(N) − v(∅)) holds trivially for any well-formed coalition game — including the word-level SGPA game. Claiming to "preserve" it across a game reformulation proves nothing nontrivial, because efficiency is a property of every valid SV computation, not something that can be lost. The interesting and non-trivial theoretical question is: what is the relationship between the token-level SVs of the original game and the word-level SVs of the SGPA game? These are different games with different player sets and there is no a priori reason their solutions should agree.
    *   **Action:** Reframe the proposition as: (1) prove that the SGPA word-level game is a valid characteristic function (superadditivity or at least boundedness conditions); (2) derive a bound on how much the word-level SV of word w can deviate from the sum of token-level SVs for the constituent tokens of w, as a function of the within-word interaction structure. The bound is the nontrivial contribution.
    *   **Implementation:** Include Proposition + Proof in an appendix. If the bound is loose, acknowledge it as a limitation rather than claiming "preservation."
*   [ ] **Define "non-neutrality" and reframe Section 3 as a reusable benchmark protocol**
    *   **Context:** "Non-neutrality" is listed as one of the four benchmark protocol components but is never defined anywhere in the plans. Proposing a community standard benchmark with an undefined component is not viable; reviewers will block on this immediately.
    *   **Action:** (1) Define non-neutrality operationally: a method exhibits non-neutrality if it assigns significantly different SVs to segments with different linguistic roles (e.g., content words vs. function words, keyword vs. filler) on a set of inputs where these roles are unambiguous. Report a test statistic (e.g., Mann-Whitney U) comparing SV distributions across role groups. (2) Once defined, rewrite Section 3 to frame all four subsections (feasibility, non-neutrality, faithfulness, boundary quality) as a named benchmark protocol with clear input/output specifications for each component.
    *   **Implementation:** Add an "Evaluation Protocol" paragraph and propose it as a community standard. The non-neutrality definition must appear before Section 3 is submitted for any peer review.
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

## Phase 6: Additional Ablations & Explorations

*   [ ] **Characterize SGPA failure modes**
    *   **Context:** Every plan describes success conditions. No plan answers: under what conditions does SGPA produce systematically misleading attributions? A senior reviewer will always ask this. Presenting failure conditions proactively is stronger than leaving reviewers to find them.
    *   **Action:** Identify and empirically test at least four failure condition candidates: (1) co-articulation — phoneme identity changes at word boundaries, making the "removed segment" acoustically ambiguous; (2) fast speech / boundary collapse — words run together so CTC boundaries are unreliable; (3) formulaic speech — numbers, proper nouns, idioms, where word identity matters more than phonetic content; (4) prosody-dominant tasks — models that respond to pitch and energy patterns that span multiple words, making per-word attribution meaningless.
    *   **Implementation:** For each condition, construct 10 targeted examples, run SGPA, and test whether faithfulness AUC drops relative to the baseline condition. Report this as a limitations section with quantitative support.
    *   **Success criterion:** The paper proactively defines and quantifies the conditions under which SGPA should not be used.

*   [ ] **Extension: non-token-based audio models**
    * **Context:** SGPA is designed for token-based models, but the core idea of "aligning to meaningful segments" could apply to non-token-based models (e.g., end-to-end ASR or audio classification).
    * **Action:** Test whether a similar segmentation and masking approach can yield meaningful attributions for a non-token-based model.
    * **Implementation:** Choose a non-token-based model (e.g., an end-to-end ASR model without explicit tokenization). Use the same CTC-based segmentation to define "coalitions" of audio segments. Compute Shapley Values based on masking these segments. Report whether the resulting attributions align with human intuition and whether they are more faithful than random segment masking.

# Appendix

## A: Languages

Out of the languages supported by the LFM2-Audio model, here are the two best choices that maximize linguistic variance from English and from each other, while still having excellent dataset availability:

### 1. Japanese (Japonic family)
*   **Why it differs:** As mentioned earlier, Japanese is an **agglutinative** language (stacking suffixes for grammatical meaning) and is **mora-timed** (rhythm is based on equal-length beats rather than stress). It also uses subject-object-verb (SOV) word order, unlike English's SVO.
*   **Why it tests your pipeline:** Its moraic rhythm makes it the perfect stress test for your masking ablation (deletion vs. silence padding). Deleting a segment will abruptly break the mathematical regularity of the spoken rhythm, which might severely confuse the model.
*   **Datasets:** Mozilla Common Voice, Corpus of Spontaneous Japanese (CSJ), FLEURS.

### 2. Arabic (Afroasiatic family)
*   **Why it differs:** Arabic uses **non-concatenative (root-and-pattern) morphology**. Unlike English (which adds separate words) or Japanese (which stacks suffixes onto the end of a word), Arabic often changes the internal vowels of a root word to alter its meaning. For example, the root *k-t-b* (writing) becomes *kitāb* (book), *kātib* (writer), or *maktab* (desk/office).
*   **Why it tests your pipeline:** Because semantic concepts are interwoven directly into the phonetic structure of a single word, it fundamentally challenges how Shapley Values assign "importance." It will test if your word-level CTC aligner can handle attributions when the core semantic meaning is distributed across vowels *inside* the consonants.
*   **Datasets:** Mozilla Common Voice, FLEURS, MGB (Multi-Genre Broadcast) Challenge datasets.
