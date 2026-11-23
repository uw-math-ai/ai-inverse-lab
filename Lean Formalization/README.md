# Vertex centrality reconstruction formalization

Start with [STATUS](STATUS) for PRs, validation, and the current completion estimate.
[PROOF_AUDIT.md](PROOF_AUDIT.md) maps the paper's proof dependencies to pinned
Mathlib and Tau Ceti code, identifies roadmap boundaries, and records coordinate
issues that must be settled before formalizing the reconstruction theorem.

All work and development tools are contained in this folder. The nested library,
roadmap, and review checkouts are excluded from the outer repository to avoid
vendoring their source trees. Submitted changes are preserved as Git patches in
`patches/`; `prs/` holds descriptions and CI notes. `candidates/` contains checked
but unsubmitted material whose upstream scope is unresolved. A checked candidate
is not a landed library contribution.

In this prepared Windows workspace, load the local toolchain and cache paths:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. './Lean Formalization/scripts/env.ps1'
Set-Location './Lean Formalization/TauCeti'
lake build TauCeti.Probability.MarkovChain.FirstPassage
```

The environment change applies only to the current process. Lean is pinned to
`4.36.0-rc2`; library revisions are recorded in the audit. The prepared tool
distribution and caches are local and excluded by `.gitignore`.
The nested Tau Ceti checkout uses repository-local `core.autocrlf=input` so
branch changes preserve Unix line endings for the text-style checker.

The rational examples in the audit can be reproduced without external packages:

```powershell
python './Lean Formalization/scripts/check_paper_coordinates.py'
```

The official TauCetiReview CLI and Python local-lint wrappers currently require
POSIX modules unavailable on this host. The unchanged trusted Lean checkers can
run directly; local reports distinguish their results from full remote CI and
official rubric reviews. No review or merge gate has been bypassed.

Do not open dependent PRs until their prerequisites land. Do not submit new
roadmaps from this agent: roadmap proposals require human curation. The requested
roadmap tutorial is due after the existing-roadmap contribution work is complete,
including review and merge; that milestone has not been reached.
