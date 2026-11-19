"""Generate leanblueprint content from the authoritative swarmalize nodes."""
from project import ROOT, nodes

INTRO = r"""\chapter{Scope and conventions}
This blueprint prepares the exact mathematical formalization of Gao--Li--Yang,
\emph{Vertex Centrality Reconstruction in an Inverse Problem for Information Diffusion},
arXiv:2603.20710v1 (21 March 2026). All paper references below refer to that version.
The full corrected paper remains incomplete. Per-node metadata records the current
proof frontier; partial prerequisites do not certify their consumers. A passing build
is not by itself a proof or axiom certificate. The existing parent-folder proof audit and historical
Tau Ceti candidates are preserved; they are not counted as checked project proofs.

\section{Approved deviations}
On 19 November 2026 the user approved corrected statements with explicit deviations:
the projection target is Euclidean, not the unknown mass-weighted projection;
the value-coordinate moment formula includes the boundary mass matrix;
and the generic-rank argument uses one nonzero polynomial minor before intersecting
with admissibility. See the project correction register and parent proof audit.
These are local audited corrections, not author-confirmed errata.

\section{Scope and verification}
The target includes Sections 1--4 and Appendix A, including the cited control,
Dirichlet, and linear-algebra prerequisites. Numerical experiments in Section 5,
Monte Carlo accuracy, numerical stability, and the complexity count in Remark 2.2
are outside this initial exact-theorem scope. The renewal algorithm is equivalent
to the paper's composition sum only after that equivalence is proved.
A dependency edge records a mathematical need, not a proof certificate.
Definitions, statements, proofs, and axiom audits must be tracked separately as work progresses.
Throughout, Assumption 1 means the weighted-graph hypotheses,
$\mu_x\geq d_w(x)$ for all vertices, and the unique continuation property
defined below. Status labels describe progress, not independent certificates.

\chapter{Mathematical dependency blueprint}
"""
parts=[INTRO]
for n in nodes():
    body=n['body'].split(r'\paragraph{Proof plan.}',1)
    flags=r'\notready' if n['status']=='needs-review' else ''
    parts += [r"\nodebreak"+"\n", f"\\section{{{n['title']}}}\n",
              f"Source: {n['source']}. Status: {n['status']}.\n",
              f"\\begin{{{n['type']}}}\\label{{{n['id']}}}\n{flags}\n",
              '\\uses{'+','.join(n['dependencies'])+'}\n' if n['dependencies'] else '',
              body[0], '\n\\paragraph{Checkpoint status.}\n'+n.get('handoff_note','')+'\n' if n.get('handoff_note') else '', f"\\end{{{n['type']}}}\n",
              '\\paragraph{Proof plan (not yet formalized).}\n'+body[1] if len(body)>1 else '']
(ROOT/'blueprint/src/content.tex').write_text(''.join(parts))
print(f'Rendered {len(nodes())} blueprint nodes')
