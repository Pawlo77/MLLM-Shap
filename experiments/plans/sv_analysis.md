# Shapley Value Calculation — Methodology Analysis

## 1. Value Function / Characteristic Function

### 1.1 Greedy Decoding Collapses the Output Distribution

**Configuration** ([config.py:19-29](mllm_shap/src/mllm_shap/connectors/config.py#L19-L29)): `text_top_k=1`, `text_temperature=0.0`, `audio_top_k=1`, `audio_temperature=0.0` — fully deterministic greedy decoding.

**Problem**: The value function `v(S)` is computed by comparing the output *sequence* produced under coalition `S` to the base sequence. This means we observe only a single sample from the model's output distribution — the argmax. Two very different probability distributions that happen to share the same top-1 token are completely indistinguishable.

**Concrete failure mode**: Removing word X shifts the model's confidence from 99% → 51% on the same output token. The similarity score is ≈1.0, SV for X ≈ 0. The explanation is wrong: X clearly matters, but we can't see it through greedy decoding.

**Alternatives to explore**:
- **Log-probability of the base sequence under the perturbed input**: `v(S) = log P(base_sequence | input_S)`. This is computable without re-running full generation — just a forward pass. Directly measures how much the coalition supports the observed output. **Preferred alternative** — see KL caveat below.
- **KL divergence** between the output distributions at each generation step: `v(S) = -E_t[KL(P(y_t | full_input) || P(y_t | input_S))]`. Captures distribution shifts invisible to greedy decoding. **Caveat**: this requires aligning step `t` between the full-input and coalition responses. If the two responses have different lengths, the step-by-step sum is undefined — there is no canonical alignment. The KL formulation is incomplete until a length-alignment strategy (e.g., truncate to shorter, or prefix-only scoring) is specified. Log P does not have this defect and should be the primary recommendation.
- **Sequence-level expected similarity**: Sample multiple outputs per coalition and average, removing the greedy approximation bias at the cost of extra inference calls.

**Computational note on log P**: The forward-pass framing may understate cost. A teacher-forced forward pass over a long output sequence is O(output_length × model_size) per coalition evaluation — the same computational structure as generation, minus the sampling step. For K coalition samples this is K full forward passes and should be budgeted before selecting it as the default fix.

**Coupling with v(∅) and v(N)**: Under the log P framing, both endpoints of the efficiency equation change. `v(N) = log P(base_seq | full_input)` is no longer implicitly 1.0 — it must be explicitly computed alongside v(∅). See §1.4.

**Coupling constraint**: Any fix that introduces stochasticity (the sampling alternative above) must simultaneously address §6.4. The current cache is correct only when the model is deterministic; enabling temperature > 0 silently corrupts cached evaluations for all repeated masks.

### 1.2 Cosine Similarity on Mean-Pooled Static Embeddings

**Default path** ([embeddings.py:35-43](mllm_shap/src/mllm_shap/shap/embeddings.py#L35-L43), [similarity.py:35-57](mllm_shap/src/mllm_shap/shap/similarity.py#L35-L57)): Token IDs from the output layer → static embedding lookup → mean pool → cosine similarity.

**Problems**:
- **Static embeddings ignore context**: The embedding lookup uses the generation model's input embedding matrix, not a contextual representation. Two synonyms map to different embeddings regardless of how similarly the model used them.
- **Mean pooling destroys order and locality**: The operation is permutation-invariant. "The cat sat" and "sat cat the" produce identical mean-pooled embeddings if all tokens are the same. A single changed token is diluted across the full sequence.
- **Length insensitivity**: A 5-token response and a 50-token response that share content get similar cosine scores, even though their informational content differs dramatically.
- **Output length change as confound**: When a coalition causes the model to generate a response of a *different length* than the base response, mean-pooled cosine similarity is comparing vectors computed over differently-sized sequences. This is geometrically inconsistent beyond same-length insensitivity: the denominator shifts with sequence length, and a substantially shorter response covering half the content scores similarly to a full-length response. Output length change is also itself a signal of coalition importance that the current value function discards entirely.

**Contextual embeddings** (`Mode.CONTEXTUAL`) address the first problem but not the second. Mean pooling remains the default reducer even there.

### 1.3 TF-IDF Cosine Similarity

**Implementation** ([similarity.py:60-145](mllm_shap/src/mllm_shap/shap/similarity.py#L60-L145)): Operates on output token sequences, vectorizes via TF-IDF, then computes cosine.

**Problems**:
- **IDF fitted per batch**: The vectorizer is `fit_transform`-ed on `[base_response, *other_responses]`. IDF weights change depending on which coalitions are being scored simultaneously. The characteristic function is therefore not fixed — `v(S)` depends on which other subsets `v(T)` are being evaluated at the same time. This violates the fundamental requirement of a game-theoretic value function. Beyond single-run violations, this invalidates cross-run comparisons: two runs with different batch sizes or coalition orderings produce different IDF vocabularies, making their SV outputs non-comparable. Any analysis that aggregates or compares TF-IDF SVs across multiple runs or audio samples is invalid.
- **Bag-of-words**: "The model is correct" and "Is the model correct?" get identical scores.
- **Mixed modality vocabularies**: Text and audio tokens live in completely different vocabulary spaces. They are concatenated and treated as the same domain ([similarity.py:105-118](mllm_shap/src/mllm_shap/shap/similarity.py#L105-L118)), which creates meaningless cross-modal IDF weights.

**Sharpened fix**: "Pre-fit on held-out corpus" is underspecified. LLM output vocabulary is narrow and domain-specific; a static IDF from a general corpus will mismatch at inference time. A more principled alternative is **raw term frequency (TF) without IDF weighting**: TF is fixed by construction, eliminates per-batch contamination entirely, and requires no corpus. It down-weights common tokens less aggressively than IDF, but it is a valid fixed characteristic function `v(S)`. If IDF weighting is required, it must be fitted once on a representative held-out set of model outputs and frozen before any SV computation begins.

### 1.4 v(∅) Is Not Defined

The efficiency axiom states `Σ φᵢ = v(N) − v(∅)`. The full-coalition value `v(N)` is implicit (the base response). The empty-coalition value `v(∅)` — the model's output given no audio input at all — is never specified or computed.

**Why it matters**: Every Shapley value is measured relative to `v(∅)`. Different choices yield fundamentally different attributions:
- **Silence**: the model receives a zero-amplitude waveform. Behaviour depends entirely on how the audio encoder handles silence.
- **Random noise**: SVs measure deviation from a noise reference, not from absence.
- **Mean audio**: a statistical average of the corpus. Shifts interpretation from "presence vs. absence" to "this segment vs. average speech."

The choice is not an implementation detail — it defines what the explanation means. It should be stated, computed, and justified.

**Under log P framing (§1.1)**: If the log P value function is adopted, `v(N)` is also no longer implicit — it becomes `log P(base_seq | full_input)`, which must be explicitly computed. Both v(∅) and v(N) need to be defined, computed, and logged before any SV can be interpreted under the log P framing. The coupling between §1.1 and §1.4 means this section cannot be resolved independently.

---

## 3. Approximation Methods

### 3.1 Monte Carlo: Unweighted Averaging is Banzhaf, Not Shapley

**Formula** ([monte_carlo/_base.py:57-64](mllm_shap/src/mllm_shap/shap/monte_carlo/_base.py#L57-L64)):
```python
included_mean - excluded_mean
```
This is the **Banzhaf value** (unweighted average of marginal contributions over all coalitions). The true **Shapley value** weights each coalition by `|S|!(n-|S|-1)!/n!`, giving more weight to small and large coalitions. When sampling is uniform over coalitions, the estimator is consistent for the Banzhaf value, not the Shapley value — unless sampling is stratified by coalition size with Shapley weights applied.

**Stronger failure mode**: The estimator is only consistent for the Banzhaf value under **uniform** coalition sampling. With `include_minimal_masks=True` (which adds LOO masks at above-random frequency) and the Neyman allocator (which further skews sampling toward high-variance features), the effective sampling distribution is neither uniform nor Shapley-weighted. The resulting estimator may not converge to any well-defined game-theoretic quantity — not just mislabeled as Shapley, but potentially inconsistent for both targets.

**Implication**: The Shapley axioms (efficiency, symmetry, dummy, additivity) hold for the Shapley value, not the Banzhaf value. Claims about Shapley-value properties should be checked against which game-theoretic value is actually being computed.

**The precise explainer** ([precise.py](mllm_shap/src/mllm_shap/shap/precise.py)) applies the correct weights and computes the true Shapley value. MC and complementary approximations should ideally converge to the same quantity with correct weighting.

**Justification for Shapley over Banzhaf required**: The Banzhaf value has its own axiomatic characterization (equal treatment, total power) and is easier to estimate consistently without stratified sampling. The document asserts Shapley is the correct target without a domain-specific reason. If the efficiency axiom is required for a specific downstream analysis, this should be stated explicitly. Without that justification, fixing the MC estimator to target Shapley adds implementation complexity without a clear benefit over the Banzhaf value already being computed.

**Decision gate — §3.1 → §4.2 chain**: The efficiency axiom (§4.2) belongs to the Shapley value only. If the decision here is to accept the Banzhaf value, §4.2's efficiency concern is **irrelevant** for MC results. This choice must be made explicitly: if Shapley → fix MC weights → normalization must preserve efficiency (§4.2 is load-bearing); if Banzhaf → §4.2 is moot for MC. Do not treat §3.1 and §4.2 as independent issues to be fixed in parallel.

### 3.2 Minimal Masks: LOO Only is Not Enough

`include_minimal_masks=True` always adds leave-one-out masks as a baseline. LOO on its own gives the marginal contribution of each feature when all others are present — one data point on the SV polytope. At low sample budgets, the approximation is dominated by LOO, which systematically overweights the full-coalition marginal contribution. Features that only matter in small coalitions will be underestimated.

**No convergence criterion**: None of the sampling approximators specify when enough samples have been taken. A standard approach is to monitor per-feature CI width and halt when it drops below a threshold (e.g., 5% of the current SV range). Without a stopping criterion, sample budgets are arbitrary and results are not comparable across runs or experiments.

### 3.3 Neyman Allocation: Good Principle, Unclear Consistency

The Neyman strategy allocates more budget to high-variance features. This is statistically sound for variance reduction. The open question is whether the estimator remains consistent (converges to the right target quantity) when budget is allocated non-uniformly. Standard Neyman allocation in surveys assumes a fixed target statistic; here the target is the Shapley value, which is a weighted expectation over coalition sizes. The weighting introduces a dependency between allocation and estimand that should be verified.

**Present risk, not open question**: The Neyman allocator is already deployed and producing results reported as Shapley values. If the estimator is inconsistent, all Neyman-produced SV estimates in the codebase are systematically biased now. This should be formally verified before publication, or Neyman results should be excluded from any claims until verification is complete.

---

## 4. Normalization

### 4.1 PowerShiftNormalizer Destroys Sign

**Default normalizer** ([normalizers.py](mllm_shap/src/mllm_shap/shap/normalizers.py)):
```python
shifted = shap_values - shap_values.min()   # min is now 0
powered = shifted.pow(power)
normalized = powered / powered.sum()
```

After shifting, all values are ≥ 0. A feature with a strongly negative SV (one that actively hurts the output when present) is mapped to a small positive value and treated as "slightly helpful." The sign information — which distinguishes helpful from harmful features — is discarded.

**Consequence**: The normalized SVs cannot be used to identify features that should be removed (negative SV), only features to keep (positive SV). For an audio explanation where the goal might be "which parts of the speech confuse the model?", this is a critical loss.

**Better options**: `AbsSumNormalizer` preserves relative magnitude and sign. **However**: `AbsSumNormalizer` (dividing by `Σ|φ_i|`) still destroys the efficiency axiom — any normalizer that rescales destroys `Σφ_i = v(N) - v(∅)` (see §4.2). The sign-destruction bug and the efficiency-axiom loss share the same root cause: normalization is applied to the game-theoretic values rather than reserved for display. Both are fixed simultaneously by storing raw signed SVs and normalizing only at visualization time. Do not treat §4.1 and §4.2 as separately fixable.

**Second distortion from `pow(power)`**: Beyond sign loss, the power transform independently distorts relative magnitudes. If `power > 1`, large SVs are amplified super-linearly relative to small ones; if `power < 1`, differences are compressed. The ranking of features with similar SVs changes depending on the `power` value. This is a distinct problem from the sign issue: even with the shift removed, the power transform would still corrupt the attribution landscape.

### 4.2 Efficiency Axiom Lost

Shapley values sum to the total "game value" (efficiency): `Σ φ_i = v(N) - v(∅)`. After any sum-to-1 normalization — including `AbsSumNormalizer` — this property is destroyed. Any downstream analysis relying on the additivity of SVs (e.g., explaining a difference in model output) will be incorrect. The only fix that preserves efficiency is to not normalize at all: store raw SVs and apply normalization only at display time, clearly labeled as non-additive.

---

## 5. Structural / Design Issues

### 5.1 No Uncertainty Quantification

SVs are point estimates. The sampling-based approximators (MC, Complementary, Neyman) accumulate enough information to estimate variance per feature but never expose it. Without confidence intervals, two features with SVs of 0.31 and 0.29 look meaningfully different even if their 95% CIs heavily overlap.

Exposing `shap_values_variance` (estimable from MC samples) would let downstream users threshold by significance rather than raw value.

**Inter-run stability is also unaddressed**: Beyond within-run variance, it is unknown whether two runs of the full pipeline on identical input with the same seed produce the same segment rankings. High cross-run instability means that even features with narrow within-run CIs are not reproducible in practice — a distinct failure mode from high within-run variance.

### 5.2 No Multimodal Interaction Capture

Standard Shapley values capture *individual* contributions. When both text and audio tokens are present in the input, the game treats them as independent features. Interactions between modalities (e.g., "this word only matters when the tone is also uncertain") are invisible. **Shapley interaction indices** (also called SHAP-interaction values) extend the framework to pairwise interactions, at the cost of `O(n^2)` additional evaluations.

### 5.3 Alignment Confidence Unused

Each `AudioSegment` carries a `confidence` score from the forced aligner ([audio.py:35-91](mllm_shap/src/mllm_shap/connectors/base/audio.py#L35-L91)). Low-confidence alignments mean the segment boundary is uncertain — the token may be attributed to the wrong audio region. This confidence is never used to weight or flag SV estimates. A low-confidence alignment producing a high SV is suspicious and should be flagged.

**Potential feedback loop**: If the forced aligner uses a speech or language model that shares weights or training data with the model being explained, segment boundaries are not chosen independently of the model's behavior. The explainer and the explanation target become entangled. Whether the current aligner introduces this dependency should be confirmed explicitly — this is checkable in under an hour by reviewing the aligner's model card and training data description.

### 5.4 Caching Assumes Determinism

The mask cache ([_masks_manager.py:66-104](mllm_shap/src/mllm_shap/shap/base/_masks_manager.py#L66-L104)) deduplicates by hash. This is only correct if the model is deterministic. With temperature > 0, two evaluations of the same mask produce different outputs, but only the first is stored. Any non-zero-temperature runs silently use cached (wrong) values for repeated masks.

### 5.5 Adjacency Between Segments Violates Feature Independence

The Shapley framework treats players as interchangeable and independent. Adjacent audio segments are not: neighboring phonemes and words are sequentially correlated by the structure of speech, and their combined presence is qualitatively different from either in isolation.

This is related to but distinct from §5.2:
- §5.2: *given* features are treated independently, can we compute pairwise interactions?
- §5.5: is the independence assumption valid enough for the game model to apply at all?

For coarse-grained segments (full words, phrases), independence is a reasonable approximation. For fine-grained segmentation (phonemes, sub-word units), sequential correlation is severe and the game model may be structurally inappropriate. **Granularity floor**: word-level segments should be treated as the minimum granularity at which the independence assumption is defensible; any sub-word or phoneme-level experiments must be explicitly labeled as violating the game-theoretic foundation and interpreted accordingly.

### 5.6 Text Token Masking Is Unaddressed

§5.2 refers to "both text and audio tokens" as features in the game, but the masking strategy for **text tokens** is never described or critiqued. All issues in §2 have symmetric counterparts on the text side:
- What constitutes "absence" for a text token — zero embedding, pad token, removal (shifting positional indices), or a mask token?
- Does removing a text token shift positional encodings of subsequent tokens in the same way §2.3 describes for audio?
- What granularity is used — character, subword, word, phrase — and is there a sensitivity analysis?

The analysis treats text masking as given when it is subject to the same category of concerns as audio masking.

---

## 6. Process and Repair-Readiness

### 6.1 No Dependency Graph Between Fixes

Multiple issues are coupled. Implementing fixes in the wrong order wastes effort or requires rework. The load-bearing dependencies are:

1. **§1.4 → everything**: v(∅) must be defined before any SV can be interpreted. Resolve this first.
2. **§4.1/§4.2 → §5.2**: Normalization must be fixed before rerunning faithfulness. Current faithfulness numbers are built on corrupted selection criteria.
3. **§3.1 decision → §4.2**: The Banzhaf vs. Shapley decision must be made before treating §4.2 as load-bearing. If Banzhaf is accepted, §4.2's efficiency concern is moot for MC results and fixing MC weights for Shapley becomes unnecessary.
4. **§1.1 (log P) → §1.4**: Under log P, v(N) also requires explicit computation alongside v(∅).
5. **§1.1 (log P) → §5.4**: The cache key must include the base sequence under log P framing.

**Sequenced fix plan**:
1. Define v(∅) and v(N) under target value function (hours)
2. Fix normalization — expose raw signed SVs, normalize only for display (hours)
3. Rerun faithfulness evaluation with corrected SVs and significance test
4. Make Banzhaf vs. Shapley decision; fix MC weights accordingly (days)
5. Adopt log P value function; update cache key (days)
6. Neyman consistency: formally verify or exclude results from publication claims (days–weeks)

### 6.2 No Acceptance Criteria or Regression Tests

The analysis identifies what to fix but not what "fixed" looks like. Without acceptance criteria, fixes cannot be verified and may introduce new errors silently.

**Minimum required regression test**: A synthetic toy game with known exact SVs (e.g., a 4-player unanimity game where Player 1 is a known dummy) should be implemented as a unit test fixture. Any change to the approximation logic must pass this fixture. Specifically:
- MC estimator must converge to `precise.py` results on the toy game within expected variance.
- Normalization must preserve sign in a signed test case.
- TF-only similarity must return 1.0 for identical sequences and < 1.0 for any single-token change.

### 6.3 Rollback Strategy for Stored Results

Existing experiment outputs (under `experiments/faithfulness/outputs/`, `experiments/interspeech/outputs/`) were computed with the current pipeline. Fixes to the value function, normalization, or masking invalidate stored results. Before applying fixes:
- Tag all existing outputs with a `pipeline_version=pre-fix` label to distinguish them from post-fix runs.
- Identify which stored results are invalidated by which fixes (normalization fix invalidates all stored normalized SVs; value function fix invalidates all raw SVs; TF fix invalidates all TF-IDF similarity scores).
- Decide whether to rerun existing experiments or document them as "baseline under pre-fix pipeline" only.

Failing to do this makes it impossible to distinguish pre-fix and post-fix results in comparison tables or figures.

### 6.4 No Ablation Design for Competing Alternatives

Multiple alternatives are proposed across sections (log P vs. KL, silence vs. deletion, static vs. contextual embeddings, TF vs. TF-IDF). No controlled ablation design is specified. Changing two things simultaneously makes it impossible to attribute observed differences to a specific fix.

**Minimum design**: For each axis of variation (value function, masking strategy, similarity metric), run a one-factor-at-a-time ablation on a common held-out evaluation set with all other factors held at their current values. This isolates the contribution of each fix and allows prioritization if not all can be completed before a deadline.

---

## 7. Summary Table

| Area | Current Approach | Issue | Suggested Direction | Severity | Est. Cost |
|---|---|---|---|---|---|
| v(∅) and v(N) definition | Both unspecified | Game reference point undefined; under log P, v(N) is also unspecified; all SVs uninterpretable | Define, compute, and justify both endpoints; required before any other fix | Critical | Hours |
| Output representation | Greedy decoded sequence (top_k=1) | Ignores output distribution; confidence shifts invisible | Log P(base_seq \| coalition) preferred; KL requires length-alignment strategy | Critical | Days |
| Faithfulness on corrupted SVs | Top-SV selected from PowerShift output | Evaluation built on sign-destroyed values; faithfulness numbers unreliable | Fix normalization before running evaluation | Critical | Hours |
| MC approximation target | Unweighted average (`included - excluded` mean) | Converges to Banzhaf under uniform sampling; with LOO + Neyman, inconsistent for both targets | Decide Banzhaf vs. Shapley first; add coalition-size weights if Shapley required | Critical | Days |
| Neyman consistency | Non-uniform allocation deployed | Estimator consistency unverified; all Neyman SVs may be systematically biased now | Verify formally or exclude Neyman results from claims until resolved | Critical | Days–Weeks |
| Normalization (sign + efficiency) | PowerShift (shift + sum to 1) | Destroys sign; AbsSumNormalizer also destroys efficiency; root cause shared | Expose raw signed SVs; normalize only at display layer; fixes §4.1 and §4.2 together | Critical | Hours |
| Audio masking baseline | Delete + concatenate | Prosody destruction; position shift confound; choice unjustified | Silence padding (verify OOD risk empirically) or justify deletion | High | Days |
| Similarity (embedding) | Cosine on mean-pooled static embeddings | Permutation-invariant; output-length-change confound; single token diluted | Contextual embeddings + token-level comparison; log output length per coalition | High | Days |
| Similarity (TF-IDF) | IDF fitted per batch | Non-fixed value function; cross-run SVs non-comparable | Raw TF (no IDF) eliminates contamination by construction; or frozen IDF from held-out model outputs | High | Days |
| Normalization (power distortion) | `pow(power)` after shift | Super-linear/sub-linear distortion of relative magnitudes | Remove power transform; keep raw SVs | High | Hours |
| Faithfulness one-sided | Positive SV removal only | Negative SVs, comprehensiveness, sufficiency, and monotonicity untested | Add negative-SV, comprehensiveness, and monotonicity tests | High | Days |
| Faithfulness statistical significance | None | Differences not tested for significance; evaluation set size unvalidated | Add paired significance test (Wilcoxon or t-test); run power analysis before data collection | High | Days |
| Uncertainty | None | No way to assess SV reliability | Expose variance from MC samples | High | Days |
| Segment granularity | Fixed by aligner defaults | Unanalyzed hyperparameter; may dominate SVs | Sensitivity analysis; word-level minimum for independence assumption | High | Days |
| Text token masking | Unaddressed | Symmetric to audio masking issues: absence semantics, position shift, granularity undefined | Define and critique text masking strategy | High | Days |
| Output length change | Ignored in similarity | Differently-sized responses make cosine comparisons inconsistent; length change is itself an importance signal | Log output length per coalition; include in value function design | High | Days |
| Dependency graph | Absent | Fixes implemented in wrong order require rework | Explicit sequenced fix plan (§7.1) | Medium | Hours |
| Acceptance criteria / regression tests | None | Fixes cannot be verified; silent regressions possible | Synthetic toy game unit test; per-fix acceptance criteria (§7.2) | Medium | Hours |
| Rollback strategy for stored results | None | Pre-fix and post-fix results indistinguishable | Tag existing outputs with pipeline version before applying fixes (§7.3) | Medium | Hours |
| Ablation design | None | Multi-factor changes confound attribution of improvements | One-factor-at-a-time ablation on common held-out set (§7.4) | Medium | Days |
| Faithfulness baseline | Duration-matched random segment | Biased sampling over position/content | Uniform sampling or stratified by segment type | Medium | Hours |
| Caching | Hash deduplication | Wrong for stochastic models; under log P, key must include base sequence | Guard with determinism check; update key for log P framing | Medium | Hours |
| Alignment confidence | Stored but unused | Low-confidence alignment pollutes SVs | Weight or flag SVs by alignment confidence | Medium | Hours |
| Inter-run stability | Not measured | SVs may not be reproducible across runs even with fixed seed | Track rank correlation across repeated runs on held-out set | Medium | Days |
| Convergence criterion | None | Sample budgets arbitrary; runs not comparable across papers | Stop on per-feature CI width threshold | Medium | Hours |
| Shapley vs. Banzhaf justification | Asserted without motivation | No domain-specific reason Shapley is required over Banzhaf; decision gates §4.2 | State why efficiency axiom is needed downstream, or accept Banzhaf | Medium | Hours |
| KL divergence length alignment | Unaddressed in proposal | Step-wise KL undefined when response lengths differ | Specify alignment strategy or use log P only | Medium | Hours |
| Multimodal interactions | Not captured | Interaction effects invisible | Shapley interaction indices for audio×text pairs | Low | Weeks |
| Adjacency / independence | Assumed independent | Sequential correlation violates game model at fine granularity | Enforce word-level granularity floor | Low | Days |
| Aligner feedback loop | Unverified | May entangle explainer with model being explained | Check aligner model card (< 1 hour) | Low | Hours |
