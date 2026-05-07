# SV for T2T — Research Execution Plan

This plan focuses only on text-to-text work in the `mllm_shap` package.

All those should be run on the text-only connector path and on the SHAP configurations that we currently expose in the package. We already have decent unit coverage for parts of the connector and explainer stack, but we still do not have a strong end-to-end validation story for what quantity is being estimated, whether the utility function is the right one, and whether the resulting text-token attributions are stable and faithful enough for further research.

We should use at least 3 model groups there:
- one tiny deterministic causal LM for exact checks, CI, and regression tests
- one instruct/chat causal LM for realistic multi-turn prompting
- one longer-context text model, if feasible, to stress prompt length and retrieval behavior

We should also build one held-out prompt suite that covers:
- single-turn QA and extraction
- rewrite / paraphrase / constrained generation
- multi-turn state tracking
- longer-context retrieval and instruction following

> Scope here is only T2T. Audio, segmentation, and multimodal follow-up should stay in `other.md` or `future_works.md`.

---

## Phase 0: Data Integrity Check
*This must be resolved before any result from the current pipeline is cited in a paper.*

*   [ ] **Audit Banzhaf vs. Shapley identity of all existing results**
    *   **Context:** Unweighted coalition averaging (uniform sampling) estimates Banzhaf values, not Shapley values. Shapley requires weighting each coalition of size k by `k!(n-k-1)!/n!`. If the current MC path does not apply this weighting, every result currently labelled "Shapley Values" is mislabelled. This is not a future research direction — it is a current data integrity issue that affects all published or submitted results.
    *   **Action:** Inspect the coalition weighting in the MC explainer. Determine definitively whether it implements Shapley or Banzhaf weighting. If Banzhaf: relabel all existing outputs, update all paper sections that claim Shapley, and decide whether to recompute with Shapley weights or to reframe the paper around Banzhaf values (which have their own valid axiomatic justification).
    *   **Success criterion:** Every result in the package and any submitted paper is correctly labelled with the exact estimand it approximates. The decision is documented in a code comment and the paper explicitly names the value function.

---

## Phase 1: Scope Lock and Evaluation Ground
*Objective: define exactly what the T2T pipeline explains, how value is measured, and what datasets/models are authoritative for validation.*

*   [ ] **Freeze the T2T model matrix and enforce it across all phases**
	*   **Context:** T2T conclusions are only trustworthy if they survive more than one model style. Defining a 3-model matrix here but then running subsequent phases on a single model makes the matrix declaration meaningless.
	*   **Action:** Lock a minimum 3-model matrix: (1) one tiny deterministic causal LM for exact checks, CI, and regression tests — not used for published research claims; (2) one instruct/chat causal LM as the primary research model; (3) one longer-context text model as the secondary research model, used to replicate claims from the primary.
	*   **Enforcement across phases:** Phase 2 (package correctness) runs on the tiny model only. Phase 3 (estimator validation) runs real-prompt benchmarks on both research models — estimator consistency must hold on both. Phase 4 (utility function ablations) runs the main scoring backend comparison on both research models — a utility function that is optimal on one model but not the other is a model-specific finding that must be reported. Phase 5 (behavioral faithfulness) reports faithfulness curves stratified by model; if the ranking of methods changes across models, this is a result, not noise to be averaged away.
	*   **Implementation:** Record tokenizer type, chat template behaviour, context length, and sampling defaults for each model. Any experiment that is run on only one research model must explicitly label itself as preliminary and must not appear in the paper's main results table.
*   [ ] **Build a held-out prompt suite covering all T2T behaviors**
	*   **Context:** Current package tests cover mechanics, not full text-task behavior.
	*   **Action:** Create prompt buckets for single-turn QA, constrained generation, rewrite/paraphrase, extraction, multi-turn state tracking, and long-context retrieval.
	*   **Implementation:** Target 50-100 prompts per bucket. Store expected metadata: prompt length, output length, turn count, and task family.
*   [ ] **Define player granularity for text explanations**
	*   **Context:** T2T explanations currently operate on tokenizer-level units, but research claims may want word or phrase semantics.
	*   **Action:** Make token, word-group, and phrase-group modes explicit and document which one is the default research unit.
	*   **Implementation:** Treat raw tokenizer units as implementation baseline; add grouped-text experiments only after token-level math is validated.
*   [ ] **Define text-token absence semantics**
	*   **Context:** In text-only chat state, masking removes tokens from the prompt, which changes later token positions.
	*   **Action:** Compare at least three absence policies: deletion, special mask placeholder, and neutral filler/token-preserving baseline where supported by model/tokenizer.
	*   **Implementation:** Document which connectors can support each policy without undefined behavior. Measure prompt validity failure rate per policy.
*   [ ] **Define `v(∅)` and `v(N)` for text-only runs, aligned with audio definitions**
	*   **Context:** Current research notes correctly flag that the reference point is underdefined. Additionally, if v(∅) is defined differently for T2T and for audio (in `sgpa.md` Phase 0), multimodal analysis that combines text and audio SVs in a single explanation becomes impossible — the two games are incommensurable. Both definitions must be developed jointly or explicitly aligned.
	*   **Action:** For each value-function candidate, make `v(∅)` and `v(N)` explicit and logged. Coordinate with `sgpa.md` Phase 0 "Define the SGPA coalition game" to ensure the T2T and audio definitions use the same normalisation and reference semantics.
	*   **Implementation:** If using sequence similarity, compute both endpoints from real generated responses. If using log-probability, compute `log P(base_seq | prompt_S)` for both empty and full coalitions.
	*   **Circularity warning:** The utility function v used to compute SHAP values must not be the same function used to evaluate faithfulness. If they are the same, the test is trivially passed and proves nothing. Document explicitly which function is used for computation and which for evaluation; if they share a common base, add a cross-check using a second independent utility function.

## Phase 2: Core Package Correctness
*Objective: lock the mechanical behavior of the T2T connector and SHAP pipeline before debating research metrics.*

*   [ ] **Connector contract tests for `TransformersCausalText`**
	*   **Context:** Existing tests cover warnings and basic generation, but not the full research contract.
	*   **Action:** Expand tests for generation config inheritance, deterministic vs sampled generation flags, EOS/pad behavior, history updates, and contextual/static embedding path parity.
	*   **Implementation:** Keep all of these stub-based and fast. Every future connector change must preserve this contract.
*   [ ] **Chat-state masking invariants for `TransformersTextChat`**
	*   **Context:** T2T explanations depend on exact correspondence between `_text_ids`, `input_tokens`, `tokens_modality_flag`, and `text_tokens_no_system_mask`.
	*   **Action:** Add invariant tests for masking across single-turn, multi-turn, system-role, and empty-turn edge cases.
	*   **Implementation:** Assert no stale cached properties after append/mask/clone, and assert no accidental AUDIO path behavior leaks into text-only runs.
*   [ ] **Cache determinism guard**
	*   **Context:** Reusing mask results is correct only when the value function is deterministic for repeated masks.
	*   **Action:** Make deterministic and stochastic modes explicit in tests and runtime policy.
	*   **Implementation:** If `text_temperature > 0`, either disable dedup cache or include stochastic-run metadata that prevents semantic cache reuse.
*   [ ] **Raw-vs-display SHAP output contract**
	*   **Context:** `BaseShapExplainer` currently stores both raw and normalized values, while the default normalizer is unsafe for signed interpretation.
	*   **Action:** Make raw signed SHAP values the research source of truth and treat any normalization as display-only.
	*   **Implementation:** Add regression tests that fail if negative values become silently positive in research outputs.
*   [ ] **Notebook smoke coverage for text examples**
	*   **Context:** `examples/text_multi_turn.ipynb` and `examples/text_monte_carlo.ipynb` are user-facing integration surfaces.
	*   **Action:** Add a lightweight smoke run path that verifies both notebooks still match the connector and explainer APIs.
	*   **Implementation:** One small prompt per notebook is enough for CI; full notebook reruns can remain manual or scheduled.

## Phase 3: Mathematical Validation of SV Estimators
*Objective: prove what quantity each explainer estimates and under what sampling regime it remains valid.*
*Model coverage: toy-game fixtures require no LM calls. All real-prompt comparisons (PreciseShapExplainer, Neyman consistency, FastSHAP) must run on both research models from Phase 1. An estimator that converges correctly on the instruct model but not the longer-context model is a model-dependent result.*

*   [ ] **Create synthetic toy games with exact ground truth**
	*   **Context:** This is the cheapest high-signal validation surface for SHAP math.
	*   **Action:** Add unanimity, additive, interaction, and dummy-player games with known exact Shapley and Banzhaf values.
	*   **Implementation:** These fixtures must run without any LM call. They become permanent regression anchors for the package.
*   [ ] **Use `PreciseShapExplainer` as package truth source**
	*   **Context:** Approximate T2T explainers must converge to a known internal target before any behavioral analysis matters.
	*   **Action:** Compare Monte Carlo, Complementary, and Neyman explainers against `PreciseShapExplainer` on the same toy games and small real prompts.
	*   **Implementation:** Track mean absolute error, rank correlation, and calibration by coalition size.
*   [ ] **Benchmark FastSHAP as a surrogate-model estimator baseline**
	*   **Context:** FastSHAP (Jethani et al., 2021) trains a surrogate model to predict SHAP values in a single forward pass, orders of magnitude faster than any sampling-based method. It is a direct competitor to the Neyman and MC estimators and must be included in any paper that claims to offer an efficient SHAP estimation strategy. Omitting it will be flagged by reviewers familiar with the SHAP estimation literature.
	*   **Action:** Implement or adapt FastSHAP for the T2T setting (the surrogate maps prompt embeddings to SV estimates). Compare against Neyman and MC on the shared toy games and real-prompt benchmark: report MAE vs. PreciseShapExplainer, tokens explained per second, and memory cost.
	*   **Success criterion:** Either Neyman Pareto-dominates FastSHAP on accuracy-vs-cost, or FastSHAP is incorporated as the recommended default for low-compute settings.
*   [ ] **Make explicit Banzhaf vs Shapley decision for Monte Carlo path**
	*   **Context:** Unweighted coalition averaging is not automatically Shapley.
	*   **Action:** Decide whether the MC path is meant to estimate Banzhaf or Shapley, then align naming, docs, and weighting accordingly.
	*   **Implementation:** Add dedicated tests that prove convergence to the declared target, not merely to a plausible attribution vector.
*   [ ] **Validate Neyman estimator consistency before using it for claims**
	*   **Context:** Variance reduction is useful only if it does not change the estimand.
	*   **Action:** Run Neyman vs precise comparisons stratified by coalition size and prompt family.
	*   **Implementation:** If consistency is not established, keep Neyman flagged as exploratory and exclude it from main research claims.
*   [ ] **Add stopping and uncertainty criteria**
	*   **Context:** Sample budgets are not meaningful without convergence diagnostics.
	*   **Action:** Export per-token variance / confidence intervals and define a stop rule based on CI width or rank stability.
	*   **Implementation:** Add tests that the stop rule actually tightens with more samples and does not stop prematurely on high-variance prompts.

## Phase 4: Utility Function and Response-Scoring Ablations
*Objective: determine which text-output scoring function gives the most faithful and stable T2T attributions.*
*Model coverage: scoring backend comparisons must run on both research models. A utility function that ranks best on an instruct model (which follows instructions tightly) may rank differently on a longer-context model (which must retrieve from a larger context). The recommended default utility function must be justified across both.*

*   [ ] **Baseline current greedy-sequence similarity path**
	*   **Context:** Current package behavior is easy to run but may be blind to distribution changes.
	*   **Action:** Quantify how often top-1 text stays unchanged while token probabilities shift materially.
	*   **Implementation:** Use a teacher-forced scorer on the same outputs to estimate hidden sensitivity missed by greedy sequence comparison.
*   [ ] **Implement and benchmark log-probability value function**
	*   **Context:** `log P(base_seq | prompt_S)` is strongest immediate alternative for T2T.
	*   **Action:** Add a text-only scoring path that computes token log-probabilities of the base response under each coalition.
	*   **Implementation:** Compare compute cost, stability, and faithfulness against current embedding/cosine scoring.
*   [ ] **Compare output-distribution distance metrics**
	*   **Context:** Future research should go beyond single-sequence scoring when feasible.
	*   **Action:** Evaluate KL-style prefix-aligned comparisons, Jensen-Shannon variants, or Wasserstein-like approximations over next-token distributions.
	*   **Implementation:** Keep this as second-wave work after log-prob is stable. Any distribution metric must define length alignment explicitly.
*   [ ] **Ablate embedding choices and reducers**
	*   **Context:** Current defaults combine static/contextual embeddings with mean reduction, which may wash out token order.
	*   **Action:** Compare static vs contextual vs external embedding model, and mean reduction vs token-level or position-aware reducers.
	*   **Implementation:** Record whether each method preserves single-token edits, reordering sensitivity, and output-length sensitivity.
*   [ ] **Fix and validate text-only lexical similarity backends**
	*   **Context:** Current TF-IDF path is batch-dependent and therefore not a fixed characteristic function.
	*   **Action:** Compare raw TF, frozen-IDF, and no-bag-of-words alternatives for text-only outputs.
	*   **Implementation:** Add invariants: identical outputs score 1.0, any one-token change scores < 1.0, and the same coalition gets the same score regardless of batchmates.
*   [ ] **Track output-length change as first-class signal**
	*   **Context:** T2T coalitions often change completion length, and that itself is explanatory.
	*   **Action:** Log generated length, EOS position, and early-stop behavior for every coalition evaluation.
	*   **Implementation:** Analyze whether a metric is faithful only because it ignores harmful length drift.

## Phase 5: Behavioral Faithfulness and Robustness
*Objective: show that T2T attributions identify prompt tokens that genuinely control model behavior.*
*Model coverage: every faithfulness result must be reported stratified by research model (instruct vs. longer-context). This is not optional: the ablation matrix already spans multiple scoring backends × masking policies × task families, and adding model as a dimension is the only way to know whether conclusions are model-specific. If a result holds only on the instruct model, it must be scoped accordingly in the paper.*

*   [ ] **Run comprehensiveness and sufficiency tests on held-out prompts**
	*   **Context:** Removing top-ranked tokens should hurt the model more than removing random or low-ranked ones. Faithfulness here means the Jacovi & Goldberg (2020) definition — the explanation reflects what the model actually computed. This is distinct from plausibility (human agreement with the explanation), which is a separate and orthogonal property. Papers that conflate the two will be rejected.
	*   **Action:** Measure utility drop after removing top-k, bottom-k, and random-matched tokens. Report three baselines: (1) the primary explainer (whichever estimand is confirmed in Phase 0), (2) native tokenization SHAP, (3) LOO attribution (v(N) − v(N\{i})) — the universal minimum baseline that any method must beat. Also situate results against the ERASER benchmark (DeYoung et al., 2020), which is the standard community reference for faithfulness evaluation in NLP; results should be reported in ERASER-compatible metrics (comprehensiveness, sufficiency, token-F1 against human rationales where available) to allow direct comparison with prior work.
	*   **Metrics:** Report both AOPC (Area Over Perturbation Curve) and AUC over masking fraction. AOPC is the primary community metric; reporting only AUC makes the paper incomparable to most prior work.
	*   **Implementation:** Removal must be simultaneous-at-each-k (all top-k tokens removed at once per evaluation point), not sequential greedy. Sequential removal creates path-dependencies where each step changes the context for the next, making AUC values incomparable across methods. Add LIME as a text-side attribution baseline alongside LOO and native SHAP — LIME is what many practitioners use and the paper must show whether SHAP adds value over it.
*   [ ] **Test negative-SV behavior explicitly**
	*   **Context:** Some prompt tokens should hurt or distract the model.
	*   **Action:** Add evaluations where removing negative-SV tokens improves utility.
	*   **Implementation:** This is required to verify that signed raw SHAP values remain meaningful after pipeline changes.
*   [ ] **Add monotonicity and rank-stability checks**
	*   **Context:** Small sampling noise should not randomly reorder the most important tokens.
	*   **Action:** Measure top-k overlap, Kendall/Spearman rank correlation, and sign consistency across repeated runs.
	*   **Implementation:** Run repeated seeds for exact same prompt/model/config and stratify by prompt family. Use the same metrics (MAD, top-k Kendall τ, sign consistency) as the audio stability protocol in `sgpa.md` Phase 4 so that T2T and audio stability results are directly comparable in a joint table.
*   [ ] **Validate multi-turn and system-prompt behavior**
	*   **Context:** T2T package supports multi-turn chats; explanation semantics change when earlier turns and system text are present.
	*   **Action:** Test that user-controlled tokens can be explained while protected/system-only tokens remain excluded unless explicitly enabled.
	*   **Implementation:** Add prompt suites where the decisive fact sits in turn 1, turn N, and system text.
*   [ ] **Test prompt-format sensitivity**
	*   **Context:** Text-only models can change behavior strongly under formatting changes that are semantically minor to a human.
	*   **Action:** Compare attributions across equivalent prompt rewrites, delimiter changes, and role-template variants.
	*   **Implementation:** Separate genuine semantic sensitivity from template/tokenization artifacts.
*   [ ] **Add significance testing, power analysis, and multiple comparisons correction**
	*   **Context:** Faithfulness claims must survive statistical scrutiny. The T2T ablation matrix is wide: multiple scoring backends × multiple masking policies × multiple model families × multiple task families. Without correction, the expected number of spurious significant results is high.
	*   **Action:** (1) Run a prospective power analysis before committing to prompt suite sizes (target 80% power at α=0.05 for the primary comparison; realistic effect size d≈0.3 requires ~180 samples). (2) Apply FDR correction (Benjamini-Hochberg) across all pairwise comparisons in the ablation tables. (3) No main result should rely on fewer than 50 examples or on anecdotal prompt examples only.
	*   **Implementation:** Power analysis should be run on a 20-sample pilot before the full prompt suite is built, so sample sizes are justified before the compute cost is committed.

## Phase 6: Reproducibility, Documentation & Package Hardening
*Objective: finish the T2T track in a form that is easy to rerun, compare, and extend in later research.*

*   [ ] **Version outputs and invalidate pre-fix artifacts cleanly**
	*   **Context:** Value-function or normalization changes break comparability.
	*   **Action:** Stamp T2T outputs with pipeline version, scoring backend, mask policy, explainer type, and seed.
	*   **Implementation:** Maintain clear separation between pre-fix and post-fix research artifacts.
*   [ ] **Add benchmark reporting for cost vs fidelity**
	*   **Context:** T2T research decisions need compute-aware tradeoff visibility.
	*   **Action:** Track tokens explained, calls per explanation, wall-clock time, memory, and faithfulness/error metrics together.
	*   **Implementation:** This should make it easier to choose between precise, Monte Carlo, complementary, and Neyman paths in later studies.
*   [ ] **Document accepted T2T operating modes**
	*   **Context:** After the ablations, not every backend should remain equally recommended.
	*   **Action:** Produce a short decision table for recommended default, safe-but-slow option, exploratory options, and deprecated options.
	*   **Implementation:** Sync README, docs, and notebooks with the outcome so examples do not teach unsupported research practice.
*   [ ] **Turn text notebooks into stable smoke-test examples**
	*   **Context:** `text_multi_turn.ipynb` and `text_monte_carlo.ipynb` are practical entry points for users.
	*   **Action:** Keep them aligned with the validated T2T defaults and rerun them after any major package change.
	*   **Implementation:** Treat them as lightweight integration checks and package demos at the same time.
