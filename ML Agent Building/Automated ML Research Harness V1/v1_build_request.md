Autonomous ML Research Harness — V1
1. Project Goal
Build an autonomous, multi-agent machine learning research harness capable of taking a research problem, generating synthetic data through simulations, exploring a large number of potential ML methodologies, and identifying and training the best-performing model.
The system should automate the process of idea generation → experimentation → evaluation → model selection, while minimizing unnecessary computational expenditure.
The objective is to combine two approaches:
Autonomous ML research: Inspired by Sakana AI's The Agentic Scientist, where multiple agents independently propose, implement, and evaluate research ideas.
Compute-efficient experimentation: Inspired by LeanML, where inexpensive feasibility checks and preliminary experiments help eliminate weak approaches before committing substantial resources.
The primary output of V1 is a trained, evaluated ML model, accompanied by its research artifacts and experimental results.
V1 should prioritize a functional, extensible research harness rather than a fully autonomous scientific discovery platform.

2. Overall System Architecture
The system consists of four primary layers:
Research Specification
Shared Research Artifacts
Agent Orchestration
Research and Experimental Pipeline
A central scheduler coordinates the entire research process, manages resources, delegates tasks to agents, and determines when experiments advance through the pipeline.
Agents should operate semi-independently while communicating through shared research artifacts.
The system should support execution through coding agents such as Codex or Claude Code, with access to Python execution, simulation tools, and GPU resources.

3. Layer 1 — Research Specification
The system begins with a user-defined research objective.
The user provides a description of the ML problem and the constraints under which research should be conducted.
Inputs may include:
Research question or problem description
Desired synthetic dataset or simulation requirements
Evaluation metrics
Computational budget
Available hardware
Any relevant methodological constraints
The system translates these inputs into a research specification defining the experimental objectives and success criteria.
This specification serves as the reference point for the entire research process.
The scheduler and critic agents should continually ensure that generated methodologies and experiments remain aligned with the original objective.

4. Layer 2 — Shared Research Artifacts
Rather than relying on a traditional shared conversational memory or requiring a particular database, the system should maintain a persistent, structured collection of research artifacts.
These artifacts represent the current state of research and provide information that agents can access across different stages.
Examples include:
Research hypotheses and proposed methodologies
Experimental plans and implementation details
Generated datasets and simulation configurations
Training scripts and model checkpoints
Evaluation metrics and experimental results
Agent critiques and failure explanations
Computational resource usage
Decisions regarding which methodologies to advance
Each artifact should be associated with the idea or experiment that produced it, allowing the system to trace how the final model was selected.
Agents should be able to read relevant artifacts, contribute new findings, and build on previous experiments.
The scheduler determines which results should influence future research decisions.
Design principle: Artifacts, rather than agent conversation histories, should be the primary source of persistent research knowledge.
The implementation should remain storage-agnostic in V1.

5. Layer 3 — Agent Orchestration
The harness is controlled by an overall research scheduler responsible for coordinating independent research agents.
Overall Scheduler
The scheduler maintains the global research objective, research progress, resource budget, and priorities.
Its responsibilities include:
Spawning research agents
Allocating research and experimental tasks
Coordinating parallel research
Managing computational resources
Deciding which ideas should advance
Tracking research artifacts and outcomes
Determining when research should stop

