# Paper-making agent writing guidelines · v0.4

Use these instructions to turn a harness run into an English research article and its outline. A reader should be able to tell what was tested, what happened, and which conclusions the evidence supports. Apply figure, comparison, and uncertainty guidance when the run supplies the relevant evidence.

## Plan the paper

1. Open Results with the main finding in one or two sentences. Name the quantity and give one number that shows its size. Include a limitation here if it changes the takeaway. Use subheadings only when they help the reader find distinct findings.
2. Give each paragraph one job. Use short sentences and familiar verbs. Explain a technical term in ordinary words the first time the reader needs it. Leave run IDs and bookkeeping to author notes unless they affect the finding.
3. Introduce only the background needed to follow that result. Define unfamiliar terms and symbols at first use, and use the same name for a method throughout.
4. When introducing an inverse problem, identify what was observed, what quantity was estimated, and how the estimate was evaluated. Once established, refer back without repeating the setup.

## Explain the experiment

5. Name the baseline and explain what the comparison tests. Disclose meaningful differences in information, simulator access, or compute available to the methods.
6. Identify which data were used for training, validation and method selection, and which were held out for the final evaluation. Label every reported score accordingly.
7. Decide on tables by naming the reader’s question. For example, a reader comparing estimators may need exact errors and solver costs side by side. A reader checking whether a single estimator met its target may need just a sentence. Use as many tables as serve distinct comparisons. Group comparable rows when the metric and evaluation conditions align. Make each table readable on its own. Name the metric, units or scale, evaluation split, and meaning of any interval or ± notation.
8. Explain the error metric at first use. State what a smaller or larger value means; report a percentage only when the conversion and its interpretation are clear. Use the everyday meaning to read the result. Put the formula in Methods unless its calculation changes how the reader should interpret the comparison.
9. Identify the origin of each reference used to judge accuracy, such as a known synthetic parameter, a measured reference, or a proxy reconstruction.
10. Put detailed values in one place. In the prose, give the takeaway and only the numbers a reader needs to understand its size. A simple finding can be reported in a sentence. Give the main finding room for its evidence and decisive qualification. Summarize a repeated trend once. Expand a secondary finding, including a failed claim, when it changes the scientific answer. In each comparison table, use readable method names and choose the columns that answer its question. For an accuracy–cost comparison, start with primary error and cost; add another metric if it changes the interpretation.
11. Use a figure when seeing a pattern, comparison or trade-off helps the reader understand it. Choose one question per figure. Label methods, units and the evaluation split; say what higher or lower means when needed. Point the reader to the relevant feature in a sentence. Use real measurements and existing figure files. If no figure is available, a compact Markdown table can show the comparison.
12. If a summary metric and a diagnostic suggest different conclusions, describe what each measures and where they disagree. Leave the explanation open when evidence is insufficient.

## Interpret the evidence

13. Attach the conditions that materially affect a finding to the finding itself, such as sensor arrangement, noise level or observation horizon. Ask what a reader might wrongly conclude without the qualification. If it changes the headline comparison, place it in the opening beside that comparison. Put routine verification details in Methods or author notes.
14. Use “robust” or “generalizes” only with a named change in conditions and evidence from that test. Specify the training and evaluation conditions when they differ.
15. When evidence shows a trade-off, give both sides. Compare accuracy with an available cost measure and name the unit, such as forward evaluations or elapsed time.
16. Identify the source of variation in every uncertainty statement, such as bootstrap resampling of test cases, variation across seeds or posterior uncertainty. Check what was resampled before interpreting an interval.
17. State the observation first. Label a proposed mechanism as a hypothesis unless the experiment distinguishes it from plausible alternatives.
18. State the number of independent runs, seeds or cases when it affects the inference. Keep conclusions within the tested setting.
19. Give the denominator and unit for completion counts. A count of completed studies and a count of evaluated claims can describe different populations.
20. Distinguish planned groups, groups with measurements, and claims with verdicts. Report missing results or unmatched evidence in the terms used by the record. Preserve uncertainty about why a check failed.
21. Read the opening and each figure caption on their own. Can a reader say what was found and under what conditions? Cut repeated numbers, setup and generic conclusions. Then check the remaining claims against the evidence. Give each finding a primary place. Use prose for the takeaway and its implications, a table for exact comparisons, and a figure for a pattern or feature readers can see. Include both a table and a figure when each answers a distinct reader question. Read in order across paragraphs, tables and figures. For each later sentence or clause, ask what understanding would be lost if it were cut. Keep the new evidence, interpretation or limit and trim the recap. Repeat a headline value when it anchors the takeaway, and retain labels, units and conditions needed to read a display on its own.

## Check harness-specific claims

22. Report the selected model's performance and the harness's ability to finish its studies as separate findings. Include both when they answer different questions in the paper.
23. Preserve the recorded verdict for each claim. If later evidence qualifies it, report the verdict and the qualification together, with their respective sources.
24. Match the claim to its required evidence. For a claim about an improvement over a baseline, check the comparison, threshold and required diagnostic. A confidence interval for one model's absolute error answers a different question.
25. Before ranking methods, check that the metric, split, seed policy and evaluation protocol align. If they differ, report the measurements with their conditions and qualify the comparison.
26. Keep an author-facing evidence reference for every quantitative or verdict-bearing claim. Record the run, study or arm ID, source file and JSON pointer when available. Keep technical pointers out of the reader's prose.
27. Use reference papers for structure, phrasing patterns and exposition. Source all facts about the current experiment from its run materials. Mark evidence gaps for review and qualify any conclusion they affect.

A heading applies to everything nested under it. When only the selected method has a held-out score, distinguish its absolute accuracy from validation comparisons between methods. Choose sibling headings or a common heading broad enough for both. Check the heading against the data split in each table caption.

For several baselines measured under the same conditions, name the split, seed and metric once and report the differences together. Let the caption carry shared conditions when the prose refers directly to the table. Keep uncertainty beside a claim when it changes the decision or its interpretation.

## Build the article around the contribution

Start by identifying the question, which studies bear on it, and what a reader needs to understand their combined answer. Choose informative headings and subsections for this material. “Methods” and “Results” in these guidelines refer to functions a passage serves. They can appear under topic-specific headings or together within a case study.

For a methodological contribution, introduce the inverse problem, assumptions and estimation method before the tests that examine them, as in the heat-state study. For a benchmark, explain the tasks and metrics and give each major research question a corresponding answer, as in PDEInvBench. For a study of research failures, present the attempts before synthesizing the failure modes shared across them, as in the four-attempt LLM study. Choose the structure that follows from the evidence; more headings alone add no depth.

Write a self-contained abstract, and give a reader enough prior work, mathematical setup, experimental design and interpretation to judge the contribution. Give a secondary analysis or an appendix space when it adds an independently useful result or a reproducibility detail. Judge concision by what a passage adds to the reader's understanding. Use checked literature to explain an applicable method, establish context or distinguish a previous problem from this one. A paper consulted for writing style earns a scientific citation only if it also supports the specific statement.

## Editorial examples

These invented examples illustrate editing decisions. Use the current run for every factual statement in the draft.

- **Shorten a definition.** A draft introduces mean absolute error by walking through subtraction, absolute values and averaging. If the result is in degrees, report “The estimate was off by two degrees on average.” Keep the calculation in Methods. If an average hides large errors at the boundary, give that exception space in Results.
- **Add a necessary qualification.** A draft says “The reconstruction preserved fine detail.” If only a central crop was inspected, write “Fine edges remained visible in the central crop. The image margins were not assessed.” The location changes what readers can conclude. A file identifier can stay in the evidence note.
- **Choose where to spend a paragraph.** If three sampling densities show the same trend, state it once and put their values in a table. If the smallest feature disappears at the lowest density despite a good overall score, expand that case. Direct the reader to the feature in the figure and explain why the score alone misses it.

## Worked example from v3_full_s0

**Research question.** How did the physics-based estimator compare with Extra Trees on the validation data?

**Example Results paragraph.** For seed 0, the physics-based estimator achieved a lower relative L2 error on the validation set than Extra Trees (0.0193 versus 0.0774). The physics-based estimator had access to the forward simulator, while Extra Trees learned from 3,000 training examples. The study recorded the claim of at least a 10% improvement as inconclusive because the required paired-seed diagnostic was missing or could not be matched to the comparison.

**Evidence for review.** Run `heat_v3_full_s0_20261114_132644`, study `S-Q1-02`, claim `C-Q1-1`. In `evidence_bundle.json`, see `/studies/1/question`, `/arm_results/S-Q1-02.physics`, `/arm_results/S-Q1-02.extra_trees` and `/studies/1/verdicts/0`. The verdict records the diagnostic as missing or unmatched; its precise failure mode is unresolved.

## Reading notes for the guideline author

These observations informed the editorial rules above. The examples in the active guidelines are original illustrations.

- The CT comparison defines its metrics in §4.1. In §5.1.1 it summarizes the close scores among the leading methods. In §5.1.2 it directs readers to the magnified area around the spine in Figure 4 to inspect edge sharpness. This motivates moving routine definitions to Methods and explaining a visual difference where readers can see it. [Quantitative Comparison of Deep Learning-Based Image Reconstruction Methods](https://www.mdpi.com/2313-433X/7/3/44)
- In the discussion of full-data CT reconstruction, the benchmark authors explain that the reference is itself an AGD reconstruction. Agreement with it can reward a method for reproducing AGD-like features. This motivates keeping a reference-construction detail beside a result when it changes what a high score means. [Benchmarking Learned Algorithms for Computed Tomography Image Reconstruction Tasks](https://arxiv.org/html/2412.08350)
- In §3, Table 1 carries the errors at four sample sizes. The prose gives the trend and uses Figure 4 to show the estimated heat state and uncertainty. This motivates describing a repeated trend once and giving the figure its own explanatory job. [Bayesian Inference for Initial Heat States with Gaussian Series Priors](https://arxiv.org/html/2506.14241v1#S3)
- In §3.3.1, the paragraph around Table 1 explains an unexpected result: at 5% dropout the reported spread narrowed as the data grew noisier. It also explains where the dropout samples came from. This motivates expanding an apparent exception when it bears directly on whether uncertainty estimates are meaningful. [B-PINNs](https://arxiv.org/pdf/2003.06097#page=18)
