# Vertex centrality reconstruction

A blueprint-first Lean project for Gao–Li–Yang, *Vertex Centrality Reconstruction
in an Inverse Problem for Information Diffusion*, arXiv:2603.20710v1.
Work belongs to `uw-math-ai/ai-inverse-lab`, branch `owen-working-branch`.

The initial deliverable is a 24-node mathematical blueprint, a live local progress
dashboard, and a pinned, compiling Lean environment. **No paper theorem is proved
by the scaffold.** The existing parent-folder audit, candidate proofs, patches,
and contribution records are preserved and do not count toward this project's proofs.
The user approved the [documented corrections](CORRECTIONS.md) before proof work.

## Quick start

Run these commands from this directory. Prerequisites: Git, elan/Lean, Python 3.11+
and uv; Graphviz (including development libraries), XeLaTeX and latexmk for the blueprint.
On macOS Graphviz and MacTeX supply the latter tools.

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-tools.txt
lake update
lake exe cache get
.venv/bin/python scripts/build_blueprint.py
.venv/bin/python scripts/check.py
.venv/bin/python scripts/test_dashboard.py
.venv/bin/python scripts/serve.py --port 8766
```

Open <http://127.0.0.1:8766/>. The server is read-only, binds only to loopback,
and serves only the dashboard and generated blueprint assets. Stop it with Ctrl-C.
It is a foreground service, not a login service; restart it after quitting the terminal
or rebooting. The initial desktop session may keep its server in a retained process.

For the setup machine, the tools are already installed in
`/Users/owenparks/IBP/.venv`; substitute that path for `.venv` above.
The requested tool repositories are cloned at `/Users/owenparks/IBP/tools/swarmalize`
and `/Users/owenparks/IBP/tools/leanblueprint`. Their revisions are recorded in
[tools.lock.json](tools.lock.json) and pinned in the installation requirements.

Lean is pinned to 4.35.0, Mathlib to its matching release and exact manifest revisions.
The previous parent-folder Windows/Tau Ceti setup uses a different toolchain; it is
not silently reused. The Lean LSP MCP was successfully exercised on this project's
`VertexCentrality.lean`, returning no diagnostics. Future agents should use the
existing Lean LSP MCP with absolute paths inside this project. A fresh client must
connect its own Lean LSP MCP; repository files cannot install a live MCP connection.

## Blueprint and progress workflow

`blueprint/nodes/*.tex` is the authoritative mathematical source. Each file has
swarmalize YAML metadata, a quantified mathematical statement, and a proof plan.
`blueprint/src/content.tex` is generated; edit nodes and regenerate it rather than
maintaining two versions. Use `scripts/build_blueprint.py` to regenerate content and
build both outputs. It calls the official leanblueprint plasTeX plugin directly:
the upstream `leanblueprint` CLI assumes Lake lives at the Git root, whereas this
project lives inside a larger repository.

- HTML and dependency graph: `blueprint/web/`.
- PDF: `blueprint/print/print.pdf`.
- Live dashboard: node metadata, dependencies, source references, corrections,
  current branch/commit, dirty status, and source-bound local build evidence.
- Mathematical coverage: Sections 1–4 and Appendix A, including cited prerequisites.
- Excluded initial scope: Section 5 simulations, numerical stability and Remark 2.2's
  arithmetic complexity analysis.

The dashboard refreshes node metadata every ten seconds. Rebuild the rendered blueprint
after editing the nodes. Suggested statuses are `planned`, `in-progress`, `stated`,
`proved`, and `needs-review`; these are metadata claims, never independent certificates.
When a node is formalized, record its fully qualified `lean_declarations` and exact
`proof_revision`, check the Lean statement against the prose, inspect its axioms, and
only then update proof status. This initial dashboard displays the labels and counts
linked nodes; it does not yet automatically certify declaration existence or axioms.
`lake build` may accept `sorry`, so a green build is deliberately separate from proof completion.

`scripts/check.py` records its command, time and source fingerprint under ignored
`.local/checks.json`. The dashboard marks it stale if Lean sources, toolchain or Lake
configuration change. It does not certify external package modifications. Keep
pinned dependencies pristine. No verification evidence or secret is published from
`.local/`. Historical candidate files outside this project are not built by this target.

## Swarmalize reference integration

```sh
.venv/bin/swarmalize blueprint --root blueprint/nodes check
.venv/bin/swarmalize blueprint --root blueprint/nodes list
.venv/bin/swarmalize blueprint --root blueprint/nodes get full-recovery
.venv/bin/swarmalize blueprint --root blueprint/nodes dependencies algorithm --transitive
```

The dashboard uses swarmalize's actual blueprint parser and dependency validator.
No coordinator, autonomous worker, credential, overnight run, or extra publication
branch is created by this setup. Worker scheduling can be added after the blueprint
review; node dependencies are informational, not task leases.

## Source provenance

The supplied PDF was read directly from the user's Downloads folder. SHA-256:
`6051e651202b2863c454416fd164db55c8c0cfeab46333a40cb2d225b4f63090`.
The repository already contains a copy under the sibling paper-title directory,
with different PDF bytes/text encoding. It has not been overwritten or assumed
byte-identical. References and displayed formulas were checked against the supplied
v1 paper. The existing [proof audit](../PROOF_AUDIT.md) supplied additional local
analysis; its claims of past library checks remain historical until revalidated.
