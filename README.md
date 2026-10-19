# AI-Assisted Discovery in Graph Inverse Problems

**Working repository name:** `ai-inverse-lab`

## Overview

We study how mathematicians can guide AI toward proofs that are correct, insightful, and useful. Our mathematical setting is inverse problems on finite weighted graphs: what can observations of a random walk or a discrete evolution equation reveal about the hidden network and its parameters?

The project combines mathematical investigation with reproducible studies of AI-assisted research. We will develop examples, test conjectures, examine proof attempts, and record which computational tools and human interventions help.

**Current stage:** establishing baseline results and choosing a focused research question. Candidate extensions require a literature check before being described as new.

## Mathematical starting point

Our initial pilot uses a known finite connected graph with known positive edge conductances and unknown interior vertex weights. An observed vertex set is denoted by B.

For a discrete-time random-walk model, set

$$
P_{ij}=\frac{c_{ij}}{\mu_i}\quad(i\ne j),\qquad
P_{ii}=1-\frac{\sum_{j\ne i}c_{ij}}{\mu_i},
$$

where conductances are symmetric, nonedges have conductance zero, and vertex weights satisfy $\mu_i\geq\sum_{j\ne i}c_{ij}$. We initially take the weights on B as known. The observations are

$$
K_t=(P^t)|_{B\times B},\qquad t=0,\ldots,T.
$$

These assumptions specify a reversible walk with holding probabilities. Each proposed extension must state exactly which parameters, graph structure, and observations are known.

Initial questions:

- Which unknown weights can a small observation set determine?
- How much time-dependent data is needed?
- What ambiguities arise from graph symmetries or other mechanisms?
- How does recovery change near exceptional or poorly conditioned parameters?

We will begin with small examples, treat computations as evidence for precisely stated claims; a general theorem requires a proof.

## Two connected research goals

**Mathematics:** reproduce selected results, identify limitations of existing methods, and pursue a manageable extension, reconstruction formula, or counterexample.

**AI methodology:** compare how proof development changes when an AI receives computational feedback, explicit counterexamples, or targeted mathematical guidance. Use shared tasks and comparable resource budgets. Record correctness, useful mathematical progress, and the human effort required.

Previously published lemmas serve as reproduction tasks. Exploratory extensions are tracked separately. Initial experiments are pilots for designing a later evaluation.

## Starting references

1. Emilia Blåsten, Hiroshi Isozaki, Matti Lassas, and Jinpeng Lu,
   *Inverse Problems for Discrete Heat Equations and Random Walks for a Class of Graphs*.
   [arXiv:2107.00494](https://arxiv.org/abs/2107.00494)

2. Songshuo Li, Yixian Gao, Ru Geng, and Yang Yang,
   *Vertex Weight Reconstruction in the Gel'fand's Inverse Problem on Connected Weighted Graphs*.
   [arXiv:2407.17222](https://arxiv.org/abs/2407.17222)

For each reading task, record the exact theorem assumptions, data, unknowns, proof mechanism, and one worked example. The two papers use different observation and boundary conventions; write down the correspondence before transferring an argument between them.

## Suggested repository organization

| Location | Contents |
| --- | --- |
| notes/ | Literature notes, definitions, and worked examples |
| problems/ | One folder per question: statement, assumptions, proof attempts, counterexamples, and reviews |
| experiments/ | Reproducible scripts or notebooks, inputs, and small relevant outputs |
| ai-runs/ | Task-linked prompts, visible responses, tool results, metadata, and human interventions |
| weekly/ | Short individual updates, organized by meeting date |
| meetings/ | Decisions, agreed tasks, owners, and reviewers |


## Weekly contributions

Each active member should submit the following at least 24 hours before the group meeting:

1. **A short update:** about 200–400 words in weekly/YYYY-MM-DD-name.md.
2. **One reviewable contribution:** a worked example, checked lemma, proof attempt with a precise gap, counterexample, reproducible computation, or documented AI experiment.
3. **One peer review:** check a teammate's calculation, rerun an experiment, or examine a specific proof step, and link to the review.

Submit an update and its associated work through a pull request. Shared work may use a joint pull request, with individual contributions identified. Agree on a smaller scope when availability is limited. A well-documented failed attempt counts as a contribution.

## First milestones

1. Agree on notation and data conventions; reproduce a small example from the papers.
2. Choose one narrow mathematical question with explicit success criteria.
3. Run a small comparison of AI alone, AI with computational feedback, and AI with mathematical guidance.
4. Use the findings to refine both the mathematics and the evaluation protocol.

Success is measured by checked mathematical progress and informative experiments. 
