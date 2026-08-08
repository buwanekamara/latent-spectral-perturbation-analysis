# Thesis Evaluation Notes — "Vision Transformers Based Frequency Spectral Perturbation Analysis for AI-Generated Image Detection"

Reviewed as an evaluator against internal consistency, citation integrity, and stated objectives vs. delivered evidence.

## 1. Citation numbering errors (will be flagged by any examiner who checks references)

- **Ojha et al. cited as both [17] and [27], and [27] is the wrong paper.** Table 2.1 correctly cites "Ojha et al. [17]." But the body text — Sections 1.1, 1.4, 2.3.3, 2.5, 2.6, 4.2 — repeatedly cites "Ojha et al. [27]." Reference [27] in the list is actually *Nichol and Dhariwal, "Improved Denoising Diffusion Probabilistic Models"* — a DDPM paper with nothing to do with CLIP-based forensics. This means most of the thesis's central justification for using a frozen CLIP backbone is attached to the wrong citation.
- **Table 7.8 cites "FIRE [12]."** Reference [12] is *Ho and Salimans, "Classifier-Free Diffusion Guidance."* FIRE is reference [3] (Chu et al.), used correctly everywhere else in the document.
- **Section 7.4 cites GenImage as [7].** Reference [7] is the CLIP paper (Radford et al.). GenImage is correctly cited as [11] in Sections 1.6, 3.4.1, and 6.2. Section 7.4 is the odd one out.

These aren't cosmetic — a supervisor or examiner spot-checking references (which they will) will find the framework's key generalization claim resting on a citation to an unrelated paper.

## 2. Chapter 6 vs. Chapter 7 results contradict each other, unexplained

Chapter 6 (Section 6.8) reports the "yes-man" bias diagnosed during development: at the 0.5 threshold, **Recall 97.63%, Precision 58.59%** — i.e. the model over-predicts "real." It explicitly says the threshold bug's resolution "will determine the true precision/recall trade-off reported in Chapter 7."

Chapter 7's tables (7.4) show the **opposite** pattern: precision near 100% on several generators (ADM, GLIDE) with recall dragging down in the 56–64% range — i.e. the model is now under-predicting "real," not over-predicting it. This is a flipped bias, not a fixed one, and the thesis never states what change caused this flip, whether it's the same model/checkpoint, or whether Chapter 7 uses a different definition of the metrics. As written, the two chapters describe two different models behaving in opposite ways, with no bridging explanation. This is the single biggest substantive gap in the thesis and is very likely to draw direct questioning at defense.

## 3. "Relative Spectral Resilience" appears from nowhere in the conclusion

Section 8.2 names a third core contribution: "Relative Spectral Resilience normalization, which expresses each layer's drift as a proportion of that layer's own content strength." This term is never defined, named, or given a formula anywhere in Chapters 3–6. Section 7.3 gestures at something similar in one sentence ("each layer's shift is expressed as a proportion of that layer's own content strength") but the actual code listing (Listing A.2, Section 6.5.2) computes a plain, unnormalized absolute difference:

```
delta = torch.abs(orig_cls - stressed_cls)
```

There is no normalization step in the implementation as documented. Either the implementation section is out of date, or this "contribution" was added to the conclusion without being implemented/described earlier — either way it's a traceability failure between the claimed method and the shown code.

## 4. Objective 5 is only partially met, and the thesis says so — but frame it clearly

Objective 5 (Ch.1) promises evaluation of "robustness against adversarial post-processing." Chapter 8 admits this was never carried out (JPEG/blur robustness testing, explicitly listed as a to-do in the project context too). This is a legitimate, self-acknowledged incomplete objective — worth stating plainly in any evaluation rather than letting it hide inside otherwise-positive language in 8.3.

## 5. Suspiciously perfect in-domain numbers, no leakage discussion

Table 7.3 (ImageNet-based set) reports **100.00% AUC, Accuracy, Precision, Recall, F1** for the familiar ADM generator, and 100.00% AUC / 100.00% calibrated accuracy even for the *unfamiliar* Stable Diffusion v1. Literal 100% across every metric on multiple rows is a red flag for train/test leakage or an evaluation set that's too easy/too similar to training data (e.g. same real-image source, insufficiently distinct crops). The thesis doesn't discuss how train/test overlap was prevented for this specific set (only briefly touches on it for the GenImage split). This should be interrogated at defense.

## 6. "Familiar" generator isn't a clean explanation for the accuracy drop

Table 7.4 shows ADM — the generator the model was *trained on* — dropping to 81.30% Acc@0.5 in the GenImage subset, only marginally better than several *unfamiliar* generators (BigGAN 78.10%, Midjourney 71.30%). This undercuts the thesis's own framing that accuracy drops are about generator novelty; a big part of the drop is clearly about the real-image source/dataset shift, which the thesis acknowledges in principle (Section 3.4.1) but doesn't reconcile explicitly when presenting Table 7.4/7.5, where "familiar" vs. "unfamiliar" is used as if it were the main axis of difficulty.

## 7. Structural/formatting defects

- **Duplicate section numbers**: two "2.3.1" headers (should be 2.3.1/2.3.2), two "6.6" headers (Workflow Diagrams and Integration of Components — should be 6.6/6.7), two "6.8" headers (Testing During Development and Summary — should be 6.8/6.9), and Section 5.4 ("Data Flow and Interaction Design") exists in the body but is missing from the Table of Contents, which jumps from 5.3.3 to 5.5.
- **Mislabeled section**: "6.3.4 Hardware Specifications" actually describes three *data-handling protocols* (pristine cropping, resolution-aware degradation, secure caching) — not hardware. Hardware specs are covered earlier in 6.3. The heading and content don't match.
- **Figure 6.1 is placed and captioned twice** — once under Section 5.3.1 (page ~26) and again under Section 6.4.1 (page ~34), with identical caption text, as if a Chapter 6 figure was pasted into Chapter 5 as well.
- **A stray editorial note is embedded directly in the thesis body**, in Section 7.8: *"One labeling note for consistency: Tables 7.6–7.8 report numbers taken from other papers, while this new Table 7.9 is something you measured yourself... (I've phrased the intro sentence above as 'measured directly' for exactly that reason)..."* — this reads as leftover feedback from a reviewer/writing assistant addressed directly to the author ("you," "I've phrased"), left inside the submitted document. This needs to be removed before submission; if a supervisor or examiner reads it, it undermines confidence in the document's proofreading and possibly raises AI-assistance-disclosure questions.
- **Typo in a chapter heading**: Section 1.5 "Noval Approach" (should be "Novel"), also wrong in the Table of Contents.
- Table 7.4's printed layout has a misaligned "±100.00" fragment sitting between the Calibrated Accuracy and Precision columns for the ADM/GLIDE rows — looks like a column-alignment error from table formatting rather than a real value; worth re-checking the source table before final submission.
- Supervisor signature and date are left blank on the Declaration page.

## 8. Minor numeric transparency issue

Table 7.5's "Familiar" (99.85% AUC / 98.62% Cal. Acc.) and "Unfamiliar" (95.25% AUC / 91.52% Cal. Acc.) summary rows are pooled averages across *both* Table 7.3 and Table 7.4 results (e.g., "unfamiliar" = the six GenImage generators plus Stable Diffusion v1 from Table 7.3). The numbers check out arithmetically, but the pooling isn't stated anywhere near the table — a reader has to reverse-engineer it, which is exactly the kind of thing an examiner will ask about live.

---

### Priority ranking for what to fix first
1. Chapter 6 → Chapter 7 precision/recall contradiction (Section 2) — this is the credibility-critical one.
2. Citation numbering (Section 1) — quick fix, high embarrassment-if-missed factor.
3. Remove the stray editorial note in 7.8 (Section 7).
4. Define/trace "Relative Spectral Resilience" back into the methodology chapters, or drop the term from 8.2 (Section 3).
5. Structural numbering/ToC/duplicate-figure cleanup (Section 7) — cosmetic but easy wins before binding.
