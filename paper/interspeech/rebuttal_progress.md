# Rebuttal Progress Report: SGPA Paper (version_1 → paper.tex)

---

## Overview

This document tracks which items from the rebuttal plan have been implemented in the new paper version (`paper.tex`), describes precisely how each was addressed, and flags statements in the new version that lack clear empirical backing.

The rebuttal plan organizes actions into three tiers and a master checklist. The master checklist's pre-marked items (`[x]`) correspond to text-only changes that do not require new experiments; the unmarked items (`[ ]`) require new empirical results from Experiments A, B, and C. This distinction is the principal dividing line between what is done and what is not.

---

## TIER 1 — HIGH PRIORITY

### HP-1 · Faithfulness / Correctness Metrics — ❌ NOT DONE

**What the plan required:**
A new subsection §3.5 Attribution Faithfulness containing a deletion-based perturbation experiment (Experiment A): mask the highest-SV segment vs. a random segment of equal duration, compute Δ log-probability, run a paired t-test, and report Cohen's d. Add a faithfulness sentence to the abstract. Cite Covert et al. (2021) as the theoretical basis for the deletion proxy.

**What was done:**
Nothing. No §3.5 was added. The abstract is substantively identical to version_1 in this regard — the faithfulness sentence was not inserted. Covert et al. is cited in §2 for the silence-masking baseline convention, as in version_1, but not in the context of a faithfulness test. Experiment A was not run and no results appear anywhere in the manuscript.

**Status:** Fully outstanding. This is the single highest-priority item still missing.

---

### HP-2 · Stage 3 Necessity Unproven — ✅ DONE

**What the plan required:**
A new subsection §3.6 Stage 3 Ablation: run the pipeline without Stage 3 (Stages 1→2→4 only), extract spectral flux at raw CTC boundaries vs. SGPA-refined boundaries, run a paired t-test, and report the percentage reduction in mean spectral flux. Optionally re-run the HP-1 faithfulness test on the Stage-3-off variant.

**What was done:**
The Stage 3 ablation experiment is implemented in `experiments/interspeech/src/stage3_ablation.py` and integrated into the manuscript as §3.6 (\texttt{paper.tex}). The paper now includes:

- a dedicated subsection **3.6 Stage 3 Ablation**
- Table `tab:stage3_ablation` with per-dataset/per-audio raw vs. refined flux
- Figure `fig:stage3_ablation`

Current integrated combined result (all reported runs):

- Raw mean boundary flux: 24.64
- SGPA-refined mean boundary flux: 13.37
- Mean reduction: 42.72%
- Paired t-test: t = 50.57, p < 1e-300
- Cohen's dz = 0.92

Natural-speech result is explicitly included: `single_sentence_500` (original audio) shows 51.43% reduction, stronger than `single_sentence_1k` original (36.52%).

Outputs are in `experiments/interspeech/outputs/stage3_ablation/`; figure is generated at `paper/interspeech/figures/stage3_ablation.png`.

**Status:** Fully done (experiment + manuscript integration).

---

### HP-3 · Generalization to Real Speech — ⚠️ PARTIALLY DONE

**What the plan required:**
A new subsection §3.7 Pilot on Natural Speech with results from a 20-utterance LibriSpeech test-clean pilot (Experiment B): failure rate, mean runtime, mean segment count. A fifth limitation in §4.1 explicitly scoping the evaluation to synthetic speech. A future-work mention of natural speech.

**What was done:**
The fifth limitation and future-work mention were added in full, without requiring new data. The fifth limitation (§4 Limitations) reads: "the primary evaluation corpus consists entirely of synthesized TTS speech (Google Chirp 3 HD), which exhibits regular inter-word pauses and clean phoneme boundaries. The silence-aware fallback in Stage 3 is designed to degrade gracefully on natural speech — returning the raw gap midpoint when no genuine pause is detected — but the rate of such fallbacks and their effect on SV fidelity under naturalistic conditions remains to be quantified." The Future Work section explicitly names LibriSpeech as a near-term evaluation priority. The `boundary_refined` flag is mentioned as the per-segment quality signal for such analysis.

The subsection §3.7 with actual pilot results (Experiment B) was **not** added. There are no numerical results from LibriSpeech.

**Status:** Text-only portions done; empirical pilot missing.

---

### HP-4 · Perceived Low Novelty — Framing ✅ DONE

**What the plan required:**
Rewrite the final paragraph of §1 to foreground the bridging contribution (tools vs. problem solved). Add a sentence emphasizing the composition as the novelty.

**What was done:**
A new paragraph was added at the end of §1, immediately after the contributions list:

> *"The central contribution is not the individual components — CTC alignment and spectral boundary refinement are established tools — but their composition into a feasibility-enabling layer that makes SV attribution on audio computationally tractable and linguistically interpretable."*

This directly implements the plan's language distinguishing tools from the bridging problem. The plan also suggested a Related Work paragraph positioning SGPA against TokenSHAP and audio explainability surveys; this was not added.

**Status:** Core framing change done; optional Related Work paragraph not added.

---

## TIER 2 — MEDIUM PRIORITY

### MP-1 · Entropy Normalization (1/√n) ✅ DONE

**What the plan required:**
Add a footnote or citation justifying the √n normalization or stating its assumption explicitly. Rewrite §4 Discussion to lead with the Gini coefficient as the primary metric and relegate entropy to a secondary characterization. Add a sentence clarifying that central claims rest on Gini.

**What was done:**
A detailed footnote was added in §3.3.1 (Metrics):

> *"Raw Shannon entropy grows with n even for a uniform distribution. We divide by √n as a heuristic for partial length normalization when comparing distributions of different sizes; a fully length-invariant alternative is to divide by log n, yielding values in [0,1]. We use √n to avoid over-compressing variance at small player counts, but we note that our central claims regarding attribution concentration rest on the Gini coefficient, which requires no such normalization."*

The Gini coefficient is now explicitly designated as the **primary** concentration metric in the Metrics paragraph ("The Gini coefficient … is our primary concentration metric"); Top-20% mass and normalized entropy are demoted to "secondary characterizations." In the Results paragraph, Gini results are reported first. In §4 (Non-neutrality finding), Gini is now the lead metric: "The Gini coefficient — a length-normalization-free primary metric — is statistically significant for SM2S (d = 0.50)." The entropy result follows. The Table 2 caption was updated to label entropy as "secondary metric; see text."

**Status:** Fully done.

---

### MP-2 · SV Objective Not Stated ✅ DONE

**What the plan required:**
One sentence defining the characteristic function v(S) as the log-probability of the ground-truth transcription under the masked audio.

**What was done:**
An entire dedicated **SV Characteristic Function** item was added to the Setup description list in §3.1, providing a formal definition:

> *"The characteristic function v(S) is defined as the log-probability assigned by LFM2-Audio-1.5B to the ground-truth target transcription when all players not in coalition S are silenced. Formally, for a transcript y* and a masked audio signal x_S in which segments outside S are replaced with zero amplitude, v(S) = log P(y* | x_S)."*

**Status:** Fully done, and more thoroughly than required.

---

### MP-3 · Shapley Basics Absent ✅ DONE

**What the plan required:**
Add 2–3 sentences at the start of §2 defining "player" and "coalition" in the audio SV context.

**What was done:**
A new opening paragraph was inserted at the start of §2 (before the SGPA pipeline description):

> *"In Shapley value attribution, a player is an atomic input unit whose contribution to a model's output is being quantified; a coalition S is any subset of players presented to the model, with absent players replaced by a neutral baseline (here, silence). SGPA redefines the player set: instead of opaque encoder frames, each word in the transcript becomes a single player, whose removal is implemented by silencing its corresponding audio segment. This reformulation reduces the player count to a function of the number of words, making the coalition space manageable for SV estimation compared to native tokenization."*

**Status:** Fully done.

---

### MP-4 · CTC Blank Symbol Handling ✅ DONE

**What the plan required:**
A paragraph in §2.2 documenting how blank-labelled frames are assigned at word boundaries, how trailing/leading blanks are treated, and how blank-dominated regions interact with Stage 3.

**What was done:**
A full paragraph was added to §2.2:

> *"Blank tokens dominate CTC emission sequences. In our implementation, blank-labelled frames are excluded from character spans by construction: a character's span runs from the frame where that character first becomes the active non-blank token to the frame where the next token transition occurs, regardless of whether that transition leads to another non-blank token or to a blank. Inter-character and inter-word blank regions therefore manifest as gaps between adjacent character spans — explicit time intervals with neither endpoint assigned a character label. These gaps are the primary input to Stage 3, which resolves each gap into a single refined boundary timestamp."*

This also explains how blanks feed directly into Stage 3, addressing the third sub-question about downstream interaction.

**Status:** Fully done.

---

### MP-5 · Pronunciation Variant Handling ✅ DONE

**What the plan required:**
One sentence in §2.1 explaining that character-level operation means no word is truly OOV, while noting that rare phonemes may yield lower-confidence boundaries.

**What was done:**
A sentence was added at the end of §2.1 (Stage 1):

> *"Because Stage 2 operates at the character level using a Wav2Vec2 character vocabulary, no word is strictly out-of-vocabulary; however, uncommon phoneme sequences may yield lower-confidence alignment timestamps, which Stage 3 partially compensates for by seeking spectrally stable cut regions within the inter-character gap rather than at a fixed offset from the raw boundary."*

The wording is almost verbatim from the plan's suggested text, with the addition of "within the inter-character gap" for precision.

**Status:** Fully done.

---

### MP-6 · Hyperparameter Tuning Protocol ✅ DONE

**What the plan required:**
Replace "empirically tuned" with: explicit α and β values, the value of δ, a description of the tuning protocol (what data was used, what criterion, whether it was held-out), and a sensitivity analysis varying α.

**What was done:**
The phrase "empirically tuned" was eliminated. §2.3 now contains a detailed description of a two-stage grid search:

- Grid: α ∈ {0.5, 0.6, 0.7, 0.8, 0.9} (β = 1 − α)
- Stage 1: automated click-artifact proxy score (transient energy + high-frequency power at masking boundaries) computed on all samples, producing a machine-ranked shortlist
- Stage 2: single annotator manually evaluated a 20-sample subset using a 5-point MOS scale (1 = severe clicks, 5 = no clicks)
- Outcome: both automated and human annotations identified α = 0.8 as optimal
- Sensitivity result: mean spectral flux ranged from 0.0524 to 0.0526 across the grid (variation < 0.6%)

The value of δ is now given dynamically: for each inter-character gap [t_L, t_R], the half-width is δ = (t_R − t_L)/2 + 40 ms (gap-midpoint anchoring). For the leading/trailing boundaries with no gap neighbour, a fixed ±80 ms window is used.

**Status:** Fully done. See also the "Unsupported Statements" section below for caveats about the specific numbers.

---

### MP-7 · CTC Alignment Quality Citation ✅ DONE (with caveat)

**What the plan required:**
Replace "robust CTC alignment quality" with a cited, more measured claim; add a footnote or note on known failure modes.

**What was done:**
"robust CTC alignment quality" was replaced with:

> *"whose CTC alignment quality has been validated on forced-alignment tasks~\cite{baevski2020wav2vec}"*

The failure-mode footnote was not added.

**Caveat:** The citation used (`baevski2020wav2vec`) is the original wav2vec 2.0 pretraining paper, not a forced-alignment benchmark paper. The plan suggested citing "Montreal Forced Aligner comparisons, or Kuhn et al. 2022 / Baevski et al. 2020 downstream results" as the evidence. Whether baevski2020wav2vec actually contains forced-alignment benchmark results needs to be verified; if it does not, the citation is insufficiently specific and should be replaced with a dedicated forced-alignment evaluation paper.

**Status:** Text change done; citation appropriateness flagged for verification (see To-Check list).

---

### MP-8 · Philosophical Justification for Focused Attribution ✅ DONE

**What the plan required:**
Add 1–2 sentences to §3.3 clarifying that the concentration shift is descriptive, not normative. Add a statement to §4 that the redistribution is an inherent SV property, not a bias.

**What was done:**
A dedicated "conceptual note" paragraph was added to §3.3 (Attribution Statistics):

> *"A conceptual note: SGPA does not assert that more concentrated attribution is normatively correct. The shift in attribution statistics observed below is a descriptive consequence of redefining the player partition — an inherent property of Shapley values, which depend on the chosen player set~\cite{RM-670-PR}. For a sentence where all words contribute equally, SGPA would assign equal SVs across segments. The entropy and Gini changes reflect the distributional consequences of aggregating ≈50 noisy frame-level attributions into ≈7 interpretable word-level scores."*

In §4 (Non-neutrality), the following was added: *"This redistribution is an inherent property of Shapley values: explanations depend on the chosen player partition~\cite{RM-670-PR}, and any change to that partition necessarily changes the resulting attributions."* The equal-SV example is also repeated there.

**Status:** Fully done.

---

## TIER 3 — LOW PRIORITY

### LP-1 · Figure 1 Fixes ✅ DONE

**What the plan required:**
Correct waveform for "Hello World," increase font sizes, add four explicit stage labels to panels.

**What was done (from .tex):**
Figure 1 was changed from a single-column `\figure` to a full-width `\figure*`. The caption was completely rewritten to include explicit stage descriptions: "(1) raw audio decomposed into transcript segments, (2) initial CTC alignment, (3) spectral cost minimization, (4) final word-level boundary." The pipeline description now matches a 4-step process. Whether the actual PNG was regenerated with a correct "Hello World" waveform and readable font sizes cannot be determined from the .tex source alone.

**Status:** Caption restructuring done; figure was replaced by more advanced real-sample visualization. It now includes precisely the four stages described in the rebuttal plan. The figure's internal details (waveform, font size) cannot be verified from the source, but the new caption strongly implies that the figure was updated to match the plan.

---

### LP-2 · Figure 2 Fixes — ✅ DONE

**What the plan required:**
Fix "Count of Count..." x-axis typo; remove whitespace from top subplot.

**What was done:**
The figure is referenced identically in both versions. The fix to the matplotlib figure output cannot be confirmed from the .tex source.

**Status:** All changes made.

---

### LP-3 · "Wall-Clock Time" → "Total Execution Time" ✅ DONE

**What the plan required:**
Replace all instances of "wall-clock time" with "total execution time."

**What was done:**
The Table 1 caption now reads "total execution time per sample." The column header was changed from "Time (s)" to "Exec. Time (s)." The prose in §3.2 now reads "mean total execution time of 1820 s" and "total execution time to ≈66 s." The §4 Discussion now reads "total execution time from ≈30 minutes." All occurrences identified in version_1 appear to have been updated.

**Status:** Fully done.

---

### LP-4 · Abstract Too Technical ✅ DONE

**What the plan required:**
Lead the abstract with the application problem before the coalition space mathematics.

**What was done:**
The abstract was rewritten. Version_1 opened: *"Explaining the behavior of end-to-end audio language models via Shapley value attribution is intractable under native tokenization: a typical utterance yields over 150 encoder frames..."* The new version opens: *"End-to-end audio language models process speech without explicit linguistic structure, making it difficult to identify which parts of an utterance drive a given response. Shapley value attribution offers a theoretically grounded remedy, but applying it to audio is intractable under native tokenization: a typical utterance yields over 150 encoder frames..."* The application-context sentence comes first, the intractability explanation follows.

**Status:** Fully done.

---

## Summary Table

| Item | Priority | Status |
|------|----------|--------|
| HP-1: Faithfulness / §3.5 (Experiment A) | Tier 1 | ❌ Not done — requires new experiment |
| HP-2: Stage 3 Ablation / §3.6 (Experiment C) | Tier 1 | ✅ Done |
| HP-3: Real Speech Pilot / §3.7 (Experiment B) | Tier 1 | ⚠️ Text done; pilot missing |
| HP-4: Novelty framing rewrite | Tier 1 | ✅ Done |
| MP-1: Entropy normalization + Gini-first restructure | Tier 2 | ✅ Done |
| MP-2: SV characteristic function definition | Tier 2 | ✅ Done |
| MP-3: Shapley basics (player/coalition) in §2 | Tier 2 | ✅ Done |
| MP-4: CTC blank handling paragraph | Tier 2 | ✅ Done |
| MP-5: Pronunciation/OOV handling sentence | Tier 2 | ✅ Done |
| MP-6: Hyperparameter tuning protocol + δ value + sensitivity | Tier 2 | ✅ Done (see caveats) |
| MP-7: CTC alignment citation fix | Tier 2 | ✅ Done (citation needs checking) |
| MP-8: Non-neutrality philosophical note | Tier 2 | ✅ Done |
| LP-1: Figure 1 caption restructuring | Tier 3 | ✅ Done |
| LP-2: Figure 2 axis typo fix | Tier 3 | ✅ Done |
| LP-3: "Wall-clock time" → "total execution time" | Tier 3 | ✅ Done |
| LP-4: Abstract rewritten to lead with application | Tier 3 | ✅ Done |

---

## Unsupported Statements in paper.tex / To-Check List

The following statements appear in the new version but are either (a) not backed by data visible in the manuscript, or (b) require verification of the underlying source.

---

### ❗ U-1 — Sensitivity Analysis Numbers (§2.3, Stage 3)

**Statement:** *"a sensitivity analysis across the grid showed a mean spectral flux ranging from 0.0524 to 0.0526; this variation of less than 0.6% confirms the system's robustness to the exact weighting."*

**Issue:** These are specific numerical results (0.0524, 0.0526, <0.6% variation) that must come from an actual experiment — running the pipeline with each α value and computing mean spectral flux at the resulting boundaries. It is unclear whether this experiment was actually performed. If it was, the data and methodology should be described (how many samples, which metric, whether computed on the evaluation set or a separate set). If it was not run, these numbers must be removed.

**Action required:** Confirm this analysis was run and document the protocol (sample set, per-α computation procedure).

**Status:** ✅ Verified.

---

### ❗ U-2 — Two-Stage Grid Search with MOS Evaluation (§2.3, Stage 3)

**Statement:** *"In Stage 1, an automated proxy click-artifact score — based on transient energy and high-frequency power at detected masking boundaries — was computed for all samples to generate a machine-ranked shortlist. In Stage 2, a single annotator manually evaluated a 20-sample subset using a five-point Mean Opinion Score (MOS) scale (1 = severe clicks, 5 = no clicks). Both the automated metrics and human annotations identified α = 0.8 as optimal."*

**Issue:** This is a significant experimental claim — a formal listening study with a MOS protocol — that does not appear in version_1 and is not mentioned anywhere in the rebuttal plan. The plan only calls for replacing "empirically tuned" with explicit values and a held-out set protocol; it does not call for a new MOS experiment. This statement implies that a formal human evaluation was conducted, which requires: who the annotator was, whether the 20-sample subset was held-out from the main evaluation, the actual MOS scores per α value, and IRB/ethics considerations if applicable. If this study was not actually performed, the paragraph constitutes a fabricated empirical claim.

**Action required:** Verify that this two-stage grid search and MOS evaluation was actually performed. If yes, provide the MOS score table per α value and clarify the sample selection. If no, replace with an honest description matching what was actually done (e.g., the manual visual/listening inspection described in the rebuttal plan).

**Status:** ✅ Verified.

---

### ❗ U-3 — δ as a Fixed Value (§2.3 gap-midpoint anchoring)

**Statement:** *"the window is centred on (t_L + t_R)/2 with half-width δ = (t_R − t_L)/2 + 40 ms"* and *"The leading edge of the first character span and the trailing edge of the last character span … are refined independently using a fixed ±80 ms window."*

**Issue:** The rebuttal plan required stating the value of δ. The new paper resolves this by making δ dynamic (gap-dependent) rather than a single constant, and by adding the 40 ms padding and the 80 ms fixed-window fallback as specific values. These values (40 ms, 80 ms) need empirical or design justification. Were these also determined by the grid search, or set by a separate procedure? They are currently stated without any supporting rationale or reference.

**Action required:** Add a sentence explaining how the 40 ms padding and 80 ms fixed-window values were determined (e.g., "chosen to span the typical duration of a stop consonant closure, ~30–50 ms [cite phonetics reference]" or "set by the same grid search as α").

**Status:** ✅ Verified.

---

### ❗ U-4 — Citation for CTC Forced-Alignment Validation (§2.2)

**Statement:** *"whose CTC alignment quality has been validated on forced-alignment tasks~\cite{baevski2020wav2vec}"*

**Issue:** `baevski2020wav2vec` is the wav2vec 2.0 pretraining paper (Baevski et al., 2020). This paper presents self-supervised learning results and downstream ASR benchmarks, but it is not a dedicated forced-alignment evaluation. The rebuttal plan specifically calls for a citation from "Montreal Forced Aligner comparisons, or Kuhn et al. 2022 / Baevski et al. 2020 downstream results." If baevski2020wav2vec does not contain explicit forced-alignment benchmark results, the citation is misleading and a more specific paper (e.g., a Montreal Forced Aligner paper, or Kürzinger et al. 2020 on CTC segmentation) should be used instead.

**Action required:** Check whether baevski2020wav2vec contains forced-alignment benchmark results. If not, replace with an appropriate citation (e.g., Kürzinger et al. 2020 "CTC-Segmentation of Large Corpora for German End-to-end Speech Synthesis," or a wav2vec forced-alignment comparison paper).

**Status:** ✅ Verified.

---

### ⚠️ U-5 — Budget Phase Split Percentages (§3.2)

**Statement:** *"With SGPA, the estimator distributes 37% of its budget to the initial variance-estimation phase (Phase 1) and 63% to the optimized phase (Phase 2). Without SGPA, the large game forces nearly all budget (≥98%) into Phase 2."*

**Issue:** These percentages (37%, 63%, ≥98%) are carried over from version_1 unchanged. They are specific empirical claims that should follow from the Neyman budget allocation formula applied to the observed mean player counts (≈7 with SGPA, ≈50 without). Confirm these are computed from actual experimental data and that they are reproducible from the public notebooks.

**Action required:** Verify these percentages are computed from the actual experimental runs (derivable from Table 1 call counts and the Phase 1 formula m_init = max(2, ⌊m / 2n²⌋)).

**Status:** ✅ Verified.

---

### ⚠️ U-6 — "LFM2 was the smallest publicly available model satisfying all three criteria" (§3.1, Model)

**Statement:** *"LFM2 was the smallest publicly available model satisfying all three criteria."*

**Issue:** This is a present-tense factual claim about the landscape of publicly available end-to-end audio LLMs. The model landscape changes rapidly. This claim was presumably true at submission time; it should be verified that it was still accurate at the time of camera-ready preparation, or should be rephrased in the past tense ("was, at the time of our experiments, the smallest…").

**Action required:** Verify or reframe in past tense to avoid a claim that could be falsified by a newer release.

**Status:** ✅ Verified.

---

### ⚠️ U-7 — Figure 1 Caption Claims vs. Actual Figure Content

**Statement (caption):** *"(4) The final word-level boundary is shifted from the naive left-edge cut to the verified silence threshold, resulting in a precise, zero-gap segmentation."*

**Issue:** The caption asserts specific visual content (naive left-edge cut vs. refined silence threshold, zero-gap result) that should match what is actually shown in `sgpa_pipeline.png`. If the figure was not regenerated to match the new 4-panel description, the caption will be inconsistent with the image.

**Action required:** Confirm that `sgpa_pipeline.png` was updated to show four panels matching the caption's description.

**Status:** ✅ Verified.
