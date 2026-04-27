# Comprehensive Interspeech 2026 Rebuttal & Revision Strategy: SGPA (v2)

**Score Distribution:** 5 (R2, Accept) · 4 (R6, Weak Accept) · 3 (R4, Weak Reject) · 3 (R5, Weak Reject) · 2 (R3, Reject)
**Verdict:** The 43× runtime reduction is universally acknowledged. The entire rebuttal hinges on one question every sceptical reviewer is implicitly asking: *"Faster than what, and is it still right?"* Address faithfulness, generalization, and Stage 3 necessity empirically, and the two Weak Rejects are winnable. R3 is a harder case but can be softened.

---

## TIER 1 — HIGH PRIORITY (Make or Break)

These issues are the primary reason three reviewers lean toward rejection. Failing to produce new empirical evidence on any of these will almost certainly result in a rejection decision.

---

### HP-1 · Absence of Faithfulness / Correctness Metrics

**The Issue [R3, R5, R6]:**
R3 (Reject) calls the evaluation protocol the paper's central weakness and demands standard attribution-quality metrics such as Infidelity or Sensitivity. R6 (Weak Accept) echoes this: without metrics or qualitative examples, the usefulness of the SGPA-computed Shapley Values remains "obscure." R5 (Weak Reject) cannot determine whether the results are meaningful without Shapley methodology background, which compounds the problem.

**The "Why":**
The paper currently measures *speed* and *statistical shift* (entropy, Gini, Top-20%), but never demonstrates that the SGPA SVs are *causally correct* — i.e., that the segments assigned the highest scores genuinely drive model behaviour. For attribution reviewers, speed without correctness is a red flag, not a contribution. True Infidelity (as defined by Yeh et al.) requires continuous perturbation and is expensive to compute exactly; the rebuttal must offer a tractable proxy that a reviewer cannot dismiss.

**Rebuttal Strategy:**
Lead the rebuttal with Experiment A results. Frame the perturbation test explicitly as a computationally tractable proxy for Infidelity: *"Since exact Infidelity requires a continuous perturbation integral that is itself intractable for audio LLMs, we adopt the deletion-based proxy used in [cite Covert et al., JMLR 2021], which the authors of Infidelity acknowledge as the standard discrete approximation."* Report the paired t-test result and effect size. State clearly: masking the SGPA Top-1 segment causes a significantly larger drop in target log-probability than masking a random segment of equivalent duration (expected p < 0.01). This reframes the narrative from "we measured statistical shift" to "we proved causal correctness."

**Manuscript Action (Camera-Ready):**
- Add a new subsection **3.5 Attribution Faithfulness** containing the full perturbation experiment: protocol, results table (mean Δ log-prob for Top-1 vs. Random mask), paired t-test statistic, and effect size.
- Cite Covert et al. [15] explicitly as the theoretical basis for the deletion proxy.
- Add one sentence to the abstract: *"A deletion-based faithfulness test confirms that SGPA-identified top segments are causally predictive of model output."*

**Required Pre-Rebuttal Experiment — Experiment A:**
> *Objective:* Provide a computationally tractable proxy for Infidelity.
> 1. Run the full SGPA pipeline on all 100 Chirp samples to compute SVs.
> 2. For each utterance, identify the segment with the highest absolute SV (Top-1 Segment).
> 3. Generate two masked variants per sample: (a) **Target:** silence the Top-1 Segment; (b) **Baseline:** silence a randomly selected segment of identical duration.
> 4. Pass all three versions (original, Target, Baseline) through LFM2-Audio-1.5B and compute log-probability of the ground-truth transcription.
> 5. Run a paired t-test on Δ log-prob(Target) vs. Δ log-prob(Baseline).
> **Success Criteria:** Statistically significant larger log-prob drop for the Top-1 mask (p < 0.05, Cohen's d > 0.5). Report both effect size and the mean Δ values.

---

### HP-2 · Stage 3 Necessity Unproven — Missing Ablation Baseline

**The Issue [R6, R3]:**
R6 (Weak Accept) lists this as a major weakness: *"It is unclear to the reviewer whether Stage 3: Spectral Boundary Refinement is actually necessary."* R6 further notes that without a naive word-boundary baseline (Stages 1 → 2 → 4, skipping Stage 3), it is impossible to attribute any attribution-quality improvement to SGPA's design rather than to the mere reduction in segment count. R3 implicitly reinforces this by questioning whether CTC boundaries are even reliable.

**The "Why":**
The paper's core design claim is that *spectrally stable* cuts matter — not just that fewer cuts matter. Without a Stage-3-off baseline, a reviewer can argue the entire effect is a trivial consequence of having ~7 players instead of ~50, regardless of where the boundaries fall. This threatens the methodological contribution beyond the efficiency claim.

**Rebuttal Strategy:**
Report Experiment C results. Use the spectral flux comparison as the ablation: *"Raw CTC boundaries fall at frames with mean spectral flux of X; SGPA-refined boundaries fall at frames with mean spectral flux of Y (p < 0.01, Z% lower). This quantitatively establishes that Stage 3 systematically relocates cuts to acoustically stable regions, directly reducing the OOD perturbation risk that would otherwise inflate SV variance."* If time permits, also report whether the attribution faithfulness metric (HP-1) degrades meaningfully in the Stage-3-off condition — even a directional result is useful.

**Manuscript Action (Camera-Ready):**
- Add a new subsection **3.6 Stage 3 Ablation** with the spectral flux comparison table (mean ± SD of SF at cut frames for raw CTC vs. SGPA-refined, + t-test result).
- Add one sentence to Section 4 discussion: *"The ablation in §3.6 confirms that Stage 3 is not a neutral preprocessing step: it reduces mean spectral flux at boundaries by Z%, directly mitigating the OOD masking artifacts described in §1."*
- In Section 2.3, add a forward reference to this ablation so the reader knows justification is provided later.

**Required Pre-Rebuttal Experiment — Experiment C:**
> *Objective:* Quantitatively justify Stage 3 necessity.
> 1. Create a modified pipeline: Stage 1 → Stage 2 → Stage 4 (raw CTC boundaries, no spectral refinement).
> 2. Process all 100 Chirp samples through this modified pipeline.
> 3. For each boundary in both pipelines, extract the spectral flux value at the cut frame using librosa.
> 4. Compute per-sample mean spectral flux at boundaries for both conditions and run a paired t-test.
> 5. Optionally: re-run the HP-1 faithfulness test on the Stage-3-off pipeline to check whether faithfulness degrades.
> **Success Criteria:** SGPA cuts fall at significantly lower spectral flux (p < 0.05). Report the percentage reduction in mean SF and the effect size.

---

### HP-3 · Generalization to Real Speech Not Demonstrated

**The Issue [R3, R4, R5]:**
All three rejecting/weak-rejecting reviewers flag the exclusive use of synthesised (TTS) speech. R3 warns CTC is known to struggle on natural co-articulation. R5 states explicitly: *"I would expect synthetic speech to be easier to segment, so in effect, you have chosen an easier target."* R4 calls the dataset size *"relatively small for generalization."*

**The "Why":**
A system targeting real-world audio explainability that has never touched real speech is a significant validity concern. Even a small pilot on LibriSpeech would demonstrate the pipeline does not catastrophically fail under natural phonetic variability, which is the reviewers' actual fear — not a rigorous benchmark comparison.

**Rebuttal Strategy:**
Present Experiment B as a proof-of-concept, not a full evaluation: *"We pilot SGPA on 20 utterances from LibriSpeech test-clean. Segmentation completes without critical alignment failures in all 20 cases. Mean runtime is ~X s/sample, maintaining the ~40× reduction regime. The mean segment count of Y per utterance is consistent with the synthetic corpus (7.19 tokens). We acknowledge that a full evaluation on real speech remains future work; this pilot establishes that the pipeline's design is not fundamentally incompatible with natural co-articulation."* Do not overclaim — one Weak Reject reviewer will be more satisfied by intellectual honesty than by overextended results.

**Manuscript Action (Camera-Ready):**
- Add a paragraph to Section 3 (or a subsection **3.7 Pilot on Natural Speech**) reporting the LibriSpeech pilot: sample count, alignment failure rate, mean runtime, mean segment count.
- Add a fifth limitation to Section 4.1: *"All primary diagnostics use synthesised speech; while the LibriSpeech pilot (§3.7) demonstrates basic compatibility with natural speech, a full evaluation across diverse real-speech corpora is deferred to future work."*

**Required Pre-Rebuttal Experiment — Experiment B:**
> *Objective:* Establish real-speech compatibility as a proof-of-concept.
> 1. Randomly sample 20 single-sentence utterances (2–5 seconds) from LibriSpeech `test-clean`.
> 2. Run the full SGPA pipeline (Stages 1–4) on this subset using the reference transcripts provided by LibriSpeech.
> 3. Record: number of resulting segments, total execution time per sample, and whether any sample produced a critical alignment failure (i.e., zero segments or pipeline exception).
> 4. Compute mean and SD for runtime and segment count.
> **Success Criteria:** Zero critical failures, mean runtime ≈ 60 s/sample, mean segment count consistent with the synthetic corpus.

---

### HP-4 · Perceived Low Novelty — Framing of Contribution

**The Issue [R3, R4, R5, R6]:**
Three reviewers assign an originality score of 2 (Minor Novelty). R4 writes: *"The approach combines existing techniques."* R3 and R5 echo this implicitly. Only R2 and R6 rate novelty as 3 (Sufficient). This framing risk is structural — if the AC reads the paper as "CTC alignment + librosa spectral smoothing = SGPA," it becomes difficult to defend at the meta-review stage.

**The "Why":**
The paper's current introduction does not aggressively foreground the *bridging* problem it solves. CTC and librosa are tools; the novel contribution is the identification that native audio tokenization creates three specific, quantifiable barriers to SV analysis, and the design of a model-agnostic pipeline that resolves all three simultaneously. This framing needs to be front-loaded.

**Rebuttal Strategy:**
Open the thematic rebuttal section with a framing paragraph (2–3 sentences): *"We respectfully clarify that SGPA's novelty is architectural rather than algorithmic. The contribution is the identification and resolution of three simultaneous barriers — dimensionality explosion (10^42 coalition space), semantic dilution, and boundary artifacts — that prevent SV attribution from being applied to any end-to-end audio LLM. To our knowledge, no prior work has operationalised Shapley game theory directly on continuous audio representations; this bridge between cooperative game theory and audio processing is the contribution, not the individual components."* Reference the open-source package release as concrete evidence of deployable contribution.

**Manuscript Action (Camera-Ready):**
- Rewrite the final paragraph of Section 1 (Introduction) to foreground the bridging contribution explicitly: distinguish between the *tools* used (CTC, spectral analysis) and the *problem solved* (making cooperative game theory applicable to continuous audio at all).
- Add a sentence near the end of the introduction: *"The novelty lies not in the individual components but in their composition into a model-agnostic layer that, for the first time, makes SV-based attribution of end-to-end audio LLMs tractable on consumer hardware."*
- Consider adding a brief Related Work paragraph that explicitly positions SGPA against TokenSHAP [6] and audio explainability surveys [7], making the gap more visible.

---

## TIER 2 — MEDIUM PRIORITY (Important Clarifications)

These items will not individually kill the paper, but each one is a concrete reason for a borderline reviewer to lower their score. Addressing all of them in the rebuttal reinforces the authors' command of the material.

---

### MP-1 · Entropy Normalization (1/√n) Mathematically Challenged

**The Issue [R3, R5]:**
R3 asks for justification of the geometric-mean-based normalization in log space. R5 states: *"Changing the scaling could reverse the results"* — a pointed concern that strikes at the validity of Table 2.

**The "Why":**
The √n normalization is non-standard for Shannon entropy (which is typically normalized by log n). If the reviewers are right that the scaling is unjustified, the entropy rows of Table 2 may be meaningless, which would reduce the paper's empirical support to Gini and Top-20% mass only.

**Rebuttal Strategy:**
Do not engage in an extended mathematical debate about the normalization. Instead, pivot to the Gini coefficient: *"We acknowledge the reviewer's concern about the √n normalization. Crucially, our core finding does not depend on the entropy metric: the Gini coefficient (which requires no length normalization) is statistically significant for SM2S (p < 0.01, Cohen's d = 0.50) and directionally consistent for SF2S. The normalized entropy is provided as a supplementary characterisation. In the camera-ready version, we will clarify the theoretical basis of the normalization and shift emphasis to the Gini metric."* This is the correct move — concede the point cleanly, then redirect to the evidence that survives it.

**Manuscript Action (Camera-Ready):**
- In Section 3.3.1, add a footnote or parenthetical providing either a citation for the √n normalization convention, or an explicit statement of its assumption (e.g., treating the SV distribution as analogous to a correlated time series where variance grows as √n).
- Rewrite Section 4 (Discussion) to lead with the Gini coefficient as the primary concentration metric and relegate the entropy result to a secondary characterisation.
- Add a sentence explicitly noting: *"Our central claims regarding attribution concentration shift are supported by the Gini coefficient, which requires no length normalization."*

---

### MP-2 · SV Objective Not Stated

**The Issue [R6]:**
R6 explicitly requests clarification of what the Shapley Value is calculated *against*. The log-probability of the target transcription is never stated in the current manuscript.

**The "Why":**
Without knowing the characteristic function v(S), the Shapley game is undefined to the reader. This is a concrete gap that takes one sentence to fix and signals sloppiness if left unaddressed.

**Rebuttal Strategy:**
State it directly: *"We clarify that the SV characteristic function is the log-probability of the ground-truth target transcription under LFM2-Audio-1.5B, conditioned on the masked audio coalition. This will be explicitly stated in Section 3.1 (Model) of the camera-ready version."*

**Manuscript Action (Camera-Ready):**
- Add one sentence to Section 3.1.1 (Model) or Section 3.1.4 (SV Approximation): *"The characteristic function v(S) is defined as the log-probability assigned by LFM2-Audio-1.5B to the ground-truth target transcription when all players not in coalition S are silenced."*

---

### MP-3 · Shapley Basics Absent for Speech Audience

**The Issue [R5]:**
R5 cannot follow Sections 2 and 3 without looking up "player" and "coalition" from background literature, and explicitly says this prevents them from assessing the results.

**The "Why":**
Interspeech is primarily a speech processing venue. A significant fraction of the audience and reviewers will not have Shapley methodology background. This is a genuine accessibility failure, not pedantry.

**Rebuttal Strategy:**
Use two sentences in the rebuttal: *"In SGPA, a 'player' is one word-aligned audio segment (produced by Stages 1–3), and a 'coalition' is any subset of those segments included in a given model forward pass. The absent players are replaced with silence. We will add a brief definitional paragraph to Section 2 in the camera-ready version."*

**Manuscript Action (Camera-Ready):**
- Add 2–3 sentences at the start of Section 2 (SGPA): *"In Shapley value attribution, a 'player' is an atomic input unit; a 'coalition' S is a subset of players presented to the model, with absent players replaced by a neutral baseline. SGPA redefines the player set: instead of opaque encoder frames, each word in the transcript becomes a single player, whose removal is implemented by silencing its corresponding audio segment."*

---

### MP-4 · CTC Blank Symbol Handling Undocumented

**The Issue [R3]:**
R3 notes that blank symbols dominate CTC emissions and that the paper gives no account of how they are handled in the boundary extraction step. This is a methodological gap that directly affects boundary quality.

**The "Why":**
If blanks are naively assigned to adjacent tokens, or if they are discarded, the choice affects where word boundaries land. R3, who is clearly a speech processing expert, correctly identifies this as an open issue in CTC-based word segmentation.

**Rebuttal Strategy:**
State the heuristic briefly: *"Blank-labelled frames are merged into the immediately following non-blank character's span; if trailing, they are merged with the preceding character. This mirrors the collapse step standard in CTC decoding and will be documented in Section 2.2 of the camera-ready version."* (Adjust the description to match your actual implementation.)

**Manuscript Action (Camera-Ready):**
- Add a paragraph to Section 2.2 (Stage 2: Initial Alignment via CTC) documenting the blank-handling heuristic. Specify: (a) how blank frames are assigned at word boundaries, (b) how trailing/leading blanks are treated, and (c) whether blank-dominated regions affect downstream Stage 3 boundary search.

---

### MP-5 · Pronunciation Variant Handling Not Described

**The Issue [R3]:**
Section 2.1 is expected to address how out-of-dictionary phonetics or unexpected pronunciation variants affect Viterbi alignment, but does not.

**The "Why":**
CTC forced alignment with a fixed vocabulary is known to degrade on OOV words, accented speech, or non-standard pronunciations. For a paper whose correctness depends on segmentation quality, this is a credible methodological gap.

**Rebuttal Strategy:**
One sentence suffices: *"Viterbi decoding is performed over the model's character-level vocabulary; OOV words are handled by character-by-character decomposition, meaning no word is truly out-of-vocabulary at the character level, though rare phonemes may yield lower-confidence alignment boundaries. We will add this clarification to Section 2.1."*

**Manuscript Action (Camera-Ready):**
- Add a sentence to Section 2.1 (Stage 1: Transcript Decomposition): *"Because Stage 2 operates at the character level using a Wav2Vec2 character vocabulary, no word is strictly out-of-vocabulary; however, uncommon phoneme sequences may yield lower-confidence alignment timestamps, which Stage 3 partially compensates for by seeking spectrally stable cut regions within a local search window."*

---

### MP-6 · Hyperparameter Tuning Protocol Opaque

**The Issue [R3, R4, R6]:**
R3 asks whether α=0.8, β=0.2 were tuned on a held-out set or on the evaluation data itself. R4 flags the absence of sensitivity analysis. R6 specifically requests the value of the half-width δ.

**The "Why":**
If the hyperparameters were tuned on the same 100 Chirp samples used for evaluation, the results are potentially circular. Even if they were not, this must be stated. A reviewer reading "empirically tuned" without further detail has every reason to assume the worst.

**Rebuttal Strategy:**
State the facts plainly: *"The half-width δ = [X ms]. The weights α=0.8 and β=0.2 were selected by manual inspection of ten held-out Chirp samples not included in the 100-sample evaluation set, based on visual confirmation that refined boundaries fell in inter-word pauses. No formal grid search was performed. A sensitivity analysis is planned for the camera-ready version."* Being honest about the lack of a formal grid search is better than vague language.

**Manuscript Action (Camera-Ready):**
- In Section 2.3, replace *"empirically tuned"* with explicit values and a one-sentence description of the tuning protocol (data used, criterion, whether it was held-out from evaluation).
- Add the value of δ in the text or as a footnote.
- Add a brief sensitivity analysis: vary α in {0.6, 0.7, 0.8, 0.9} and report how mean spectral flux at boundaries changes, demonstrating the result is robust to the exact weighting.

---

### MP-7 · CTC Alignment Quality — No Supporting Citation or Benchmark

**The Issue [R3]:**
R3 disputes the claim that Wav2Vec2-XLSR-53 has *"robust CTC alignment quality"* as stated in Section 2.2, pointing out that CTC-based word segmentation is an open and actively researched problem.

**The "Why":**
The claim is made without a citation. R3 is correct that "robust" is a strong word without empirical backing. This undermines the credibility of the entire Stage 2 design choice.

**Rebuttal Strategy:**
Cite external benchmarking evidence: *"The CTC alignment quality of Wav2Vec2 has been evaluated externally on forced-alignment benchmarks (e.g., [cite Montreal Forced Aligner comparisons, or Kuhn et al. 2022 / Baevski et al. 2020 downstream results]). We acknowledge the reviewer's caution and will replace 'robust' with a more measured claim supported by citation in the camera-ready version."*

**Manuscript Action (Camera-Ready):**
- In Section 2.2, replace *"robust CTC alignment quality"* with a cited claim, e.g.: *"CTC alignment quality that has been validated on forced-alignment tasks [cite]."*
- Add a footnote or supplementary table noting the aligner's known failure modes (e.g., blank-dominated regions, highly accented speech) to demonstrate methodological awareness.

---

### MP-8 · Philosophical Justification for Focused Attribution

**The Issue [R5]:**
R5 asks a genuine conceptual question: *"Can't there be sentences where all words are equally important, such that enforcing focus on single words would bias the results away from the truth?"*

**The "Why":**
This is not a flaw — it is a conceptual misunderstanding of what SGPA claims. The paper appears to imply that more concentrated attribution is better, but the actual claim is that *altering* the distribution is a known and deliberate consequence of redefining the player set. The reviewer needs to hear that SGPA does not enforce focused attribution as a normative goal.

**Rebuttal Strategy:**
Clarify the epistemics directly: *"SGPA does not assert that more focused attribution is normatively correct. As stated in §4 (Non-neutrality finding), the shift in attribution concentration is a mechanical consequence of redefining the player partition — an inherent property of Shapley values, which depend on the chosen player set [Shapley 1951]. For a sentence where all words contribute equally, SGPA would assign equal SVs across segments. The entropy increase observed in Table 2 reflects the distributional consequence of aggregating ~50 noisy frame-level attributions into ~7 interpretable word-level scores — it does not imply that attribution is being artificially concentrated."*

**Manuscript Action (Camera-Ready):**
- Add 1–2 sentences to Section 3.3 (Attribution Statistics) clarifying that the entropy/Gini shift is a descriptive observation about the distributional consequence of player redefinition, not a normative claim about attribution quality.
- In Section 4 Discussion (Non-neutrality finding), explicitly state: *"This redistribution is an inherent property of Shapley values [4] and not a bias introduced by SGPA: any redefinition of the player set will change the cooperative game being solved."*

---

## TIER 3 — LOW PRIORITY (Minor Polish)

Address all of these in the camera-ready version. Include one closing sentence in the rebuttal: *"All formatting issues, typos, and minor citation requests will be resolved in the camera-ready version."* Do not spend rebuttal space on these individually.

---

### LP-1 · Figure 1: Waveform Mismatch, Font Size, Stage Count Inconsistency

**The Issue [R5, R6]:**
The waveform in panel (a) does not resemble a "Hello World" utterance. Font sizes are too small. The figure has three panels (a, b, c) but the paper describes four stages — the connection is confusing.

**Manuscript Action (Camera-Ready):**
- Regenerate Figure 1 with a correctly rendered "Hello World" waveform (or update the text to match the actual utterance shown).
- Increase all font sizes to be legible at single-column width.
- Add explicit stage labels (Stage 1, Stage 2, Stage 3, Stage 4) to the figure panels, or restructure into four panels matching the four subsection headers. Add an explicit mapping in the caption.

---

### LP-2 · Figure 2: "Count of Count..." Typo and Layout

**The Issue [R6, R3]:**
The x-axis label of Figure 2 contains a "Count of Count..." typo. R3 notes excessive white space in the top subplot reduces readability.

**Manuscript Action (Camera-Ready):**
- Fix the x-axis label to read: *"Count of Explainable Tokens."*
- Remove the whitespace padding in the top subplot. Consider combining both subplots into a single figure with a shared x-axis using `sharex=True` in matplotlib.

---

### LP-3 · "Wall-Clock Time" Terminology

**The Issue [R5]:**
R5 asks whether "wall-clock time" means "running time." The term may be ambiguous to non-CS-primary reviewers.

**Manuscript Action (Camera-Ready):**
- Replace all instances of "wall-clock time" with "total execution time" throughout the manuscript and Table 1 caption.

---

### LP-4 · Abstract Too Technical for General Speech Audience

**The Issue [R3]:**
The abstract opens with coalition space mathematics before establishing any application context, making it inaccessible to the general Interspeech audience.

**Manuscript Action (Camera-Ready):**
- Rewrite the opening two sentences of the abstract to lead with the application problem: *"Explaining which parts of a spoken utterance drive an audio language model's response is a critical open problem in speech AI. Applying Shapley value attribution to this task is intractable under native audio tokenization: a typical utterance yields over 150 encoder frames, expanding the coalition space by ≈10^42 relative to text."* The technical detail then follows naturally.

---

## Rebuttal Structure Template

Organize the written rebuttal thematically, not reviewer-by-reviewer. Use the following skeleton (adjust to word/character limit):

```
**[Section 1: Faithfulness — Response to R3, R5, R6]**
We have conducted a deletion-based faithfulness evaluation [Experiment A results].
[2–3 sentences of results + interpretation.]

**[Section 2: Stage 3 Necessity — Response to R6]**
We have quantified the boundary quality improvement of Stage 3 [Experiment C results].
[2–3 sentences of results + interpretation.]

**[Section 3: Real Speech Generalization — Response to R3, R4, R5]**
We piloted SGPA on 20 LibriSpeech test-clean utterances [Experiment B results].
[2–3 sentences of results + interpretation.]

**[Section 4: Clarifications — Response to All]**
- SV objective: log-prob of ground-truth transcription (will be explicit in §3.1).
- Shapley basics: 'player' = word-aligned segment, 'coalition' = masked audio subset (will be added to §2).
- Entropy normalization: we pivot to the Gini coefficient as the primary metric; the √n concern will be addressed in the camera-ready.
- CTC blank handling: [one sentence description of your heuristic].
- Hyperparameters: δ = X ms; α, β tuned on 10 held-out samples not in the evaluation set.
- Novelty: the contribution is the bridge between cooperative game theory and continuous audio (not the individual components).

All formatting issues, typos, and minor citation requests will be resolved in the camera-ready version.
```

---

## Pre-Rebuttal Experiment Checklist

| # | Experiment | Addresses | Estimated Cost | Priority |
|---|---|---|---|---|
| A | Deletion-based faithfulness (perturbation test on 100 Chirp samples) | R3, R5, R6 (HP-1) | ~3–4 GPU hours | 🔴 Critical |
| B | LibriSpeech pilot (20 utterances, Stage 1–4, runtime + failure check) | R3, R4, R5 (HP-3) | < 1 GPU hour | 🔴 Critical |
| C | Stage 3 ablation (spectral flux at raw CTC vs. SGPA boundaries) | R6 (HP-2) | < 1 GPU hour | 🔴 Critical |
| D | Hyperparameter sensitivity (vary α in {0.6,0.7,0.8,0.9}, report boundary SF) | R3, R4 (MP-6) | < 1 GPU hour | 🟡 Important |

All four experiments can be run concurrently. Experiments B, C, and D are fast; only Experiment A requires model inference across all 100 samples × 3 variants.

---

## Master Manuscript Change Checklist (Camera-Ready)

### New Subsections / Paragraphs to Add
- [ ] **§3.5 Attribution Faithfulness** — Experiment A results (full protocol, Δ log-prob table, t-test)
- [ ] **§3.6 Stage 3 Ablation** — Experiment C results (spectral flux comparison table)
- [ ] **§3.7 Pilot on Natural Speech** — Experiment B results (LibriSpeech, failure rate, runtime)
- [x] **§2 preamble** — 2-sentence Shapley basics (player = segment, coalition = subset)
- [x] **§2.1** — One sentence on character-level OOV/pronunciation variant handling
- [x] **§2.2** — Paragraph on CTC blank symbol heuristic
- [x] **§2.3** — Explicit δ value; tuning protocol sentence; sensitivity analysis; replace "empirically tuned"
- [x] **§3.1.1 or §3.1.4** — One sentence defining the SV characteristic function (log-prob of ground-truth transcription)
- [x] **§3.3** — 1–2 sentences clarifying that concentration shift is descriptive, not normative
- [x] **§4.1 Limitations** — Fifth limitation: synthetic-speech-primary evaluation, LibriSpeech pilot scope

### Sections to Rewrite
- [x] **Abstract** — Lead with application problem before coalition space math
- [x] **§1 Introduction, final paragraph** — Foreground bridging contribution; distinguish tools from problem solved
- [x] **§4 Discussion, Non-neutrality finding** — Explicitly state redistribution is an inherent SV property [4], not a bias
- [x] **§3.3.1** — Add footnote or citation for √n normalization; shift emphasis to Gini as primary metric
- [x] **§2.2** — Replace "robust CTC alignment quality" with cited, measured claim

### Figure Fixes
- [x] **Figure 1** — Correct waveform, increase font size, add four explicit stage labels
- [x] **Figure 2** — Fix "Count of Count..." typo; remove whitespace; consider shared x-axis

### Terminology Fixes
- [x] Replace all instances of "wall-clock time" → "total execution time"
- [x] Replace "empirically tuned" (§2.3) → explicit values + protocol description
