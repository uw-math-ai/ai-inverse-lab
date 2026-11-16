/* Display-only notation for exact, repository-owned mathematical text.
 * Unknown text is returned unchanged. No HTML, proof repair, or source mutation.
 */
(() => {
  "use strict";

  const presentations = new Map([
    [
      "Under the assumptions below, the map W: f -> U^f(T, .) from ell^2({0,...,T-1} x B) onto ell^2(X) is surjective. Equivalently, every real target g on X can be reached at time T by a source f supported in {0,...,T-1} x B. Prove this for every graph and every T satisfying the assumptions.",
      String.raw`Under the assumptions below, the map $W:f\mapsto U^f(T,\cdot)$ from $\ell^2(\{0,\ldots,T-1\}\times B)$ onto $\ell^2(X)$ is surjective. Equivalently, every real target $g$ on $X$ can be reached at time $T$ by a source $f$ supported in $\{0,\ldots,T-1\}\times B$. Prove this for every graph and every $T$ satisfying the assumptions.`,
    ],
    [
      "X is a nonempty finite vertex set; the graph is connected, undirected, and simple.",
      String.raw`$X$ is a nonempty finite vertex set; the graph is connected, undirected, and simple.`,
    ],
    [
      "The masses mu_x are positive; w_xy = w_yx is positive exactly on edges and zero otherwise.",
      String.raw`The masses $\mu_x$ are positive; $w_{xy}=w_{yx}$ is positive exactly on edges and zero otherwise.`,
    ],
    [
      "B is a nonempty subset of X.",
      String.raw`$B$ is a nonempty subset of $X$.`,
    ],
    [
      "For every x, mu_x >= sum_{y adjacent to x} w_xy.",
      String.raw`For every $x$, $\mu_x\geq\sum_{y\sim x}w_{xy}$.`,
    ],
    [
      "No nonzero eigenfunction of -Delta vanishes at every vertex of B; this concerns every vector in every eigenspace, including repeated eigenvalues.",
      String.raw`No nonzero eigenfunction of $-\Delta$ vanishes at every vertex of $B$; this concerns every vector in every eigenspace, including repeated eigenvalues.`,
    ],
    [
      "T is an integer with T >= |X|.",
      String.raw`$T$ is an integer with $T\geq |X|$.`,
    ],
    [
      "(Delta u)(x) = (1/mu_x) sum_y w_xy (u(y)-u(x)).",
      String.raw`$(\Delta u)(x)=\frac{1}{\mu_x}\sum_y w_{xy}\bigl(u(y)-u(x)\bigr)$.`,
    ],
    [
      "<u,v>_X = sum_{x in X} mu_x u(x)v(x); functions are real valued.",
      String.raw`$\langle u,v\rangle_X=\sum_{x\in X}\mu_xu(x)v(x)$; functions are real valued.`,
    ],
    [
      "D_t U(t,x) = U(t+1,x)-U(t,x).",
      String.raw`$D_tU(t,x)=U(t+1,x)-U(t,x)$.`,
    ],
    [
      "f(t,x) is arbitrary for 0 <= t < T and x in B, and zero otherwise.",
      String.raw`$f(t,x)$ is arbitrary for $0\leq t<T$ and $x\in B$, and zero otherwise.`,
    ],
    [
      "U^f(0,x)=0 and D_t U^f(t,x)-Delta U^f(t,x)=f(t,x).",
      String.raw`$U^f(0,x)=0$ and $D_tU^f(t,x)-\Delta U^f(t,x)=f(t,x)$.`,
    ],
    [
      "Wf = U^f(T, .).",
      String.raw`$Wf=U^f(T,\cdot)$.`,
    ],
    [
      "Vertices are 0,...,n-1; states are columns. A=I+Delta acts on functions; E has coordinate columns for B. Controls enter as E f_t without a mass factor.",
      String.raw`Vertices are $0,\ldots,n-1$; states are columns. $A=I+\Delta$ acts on functions; $E$ has coordinate columns for $B$. Controls enter as $Ef_t$ without a mass factor.`,
    ],
    [
      "For each optional finite certificate, rank means the rank of the linear map from the T source vectors to U(T).",
      String.raw`For each optional finite certificate, rank means the rank of the linear map from the $T$ source vectors to $U(T)$.`,
    ],
    [
      "The source lemma has q=0. Vertex mass/centrality mu and a Schrödinger potential q are different quantities.",
      String.raw`The source lemma has $q=0$. Vertex mass/centrality $\mu$ and a Schrödinger potential $q$ are different quantities.`,
    ],
  ]);

  const initialDemo =
    "Set A=I+Delta. Its eigenvectors cannot vanish on B by assumption. " +
    "Consequently a linear combination cannot be invisible on B, so W is onto. " +
    "The last consequence needs justification.";

  const repairedDemo =
    "Put n=|X|, A=I+Delta and let E inject controls on B. Iteration gives " +
    "Wf=sum_{s=0}^{T-1} A^(T-1-s)E f_s. Symmetry of w gives " +
    "<Delta u,v>_mu=-sum_{unordered edges {x,y}}w_xy(u_y-u_x)(v_y-v_x), " +
    "so A is self-adjoint in the positive mass inner product. If W is not onto, " +
    "there exists v!=0 orthogonal to A^k E for 0<=k<T. For 0<=k<n this " +
    "means (A^k v)|_B=0, because each mass on B is positive. Define " +
    "N={z:(A^k z)|_B=0 for 0<=k<n}. Then v belongs to N. For z in N, " +
    "the first n-1 required equalities for Az follow by shifting k. " +
    "The last follows from Cayley-Hamilton, expressing A^n in lower powers. " +
    "Thus N is A-invariant. Its nonzero self-adjoint restriction has a real " +
    "eigenvector z!=0. This is an eigenfunction of -Delta vanishing on B, " +
    "contrary to the assumed unique continuation. Hence W is onto. " +
    "The one-step auxiliary fixture violates T>=n and has rank one.";

  // Replacements run only after full-string equality with these two fixtures.
  // Work on original text in a single pass so inserted LaTeX is never rewritten.
  function literalPresentation(source, replacements) {
    let output = "";
    let offset = 0;
    while (offset < source.length) {
      const match = replacements.find(([before]) => source.startsWith(before, offset));
      if (match) {
        output += match[1];
        offset += match[0].length;
      } else {
        output += source[offset];
        offset += 1;
      }
    }
    return output;
  }

  presentations.set(initialDemo, literalPresentation(initialDemo, [
    ["A=I+Delta", String.raw`$A=I+\Delta$`],
    ["on B", String.raw`on $B$`],
    ["so W is onto", String.raw`so $W$ is onto`],
  ]));

  presentations.set(repairedDemo, literalPresentation(repairedDemo, [
    ["Wf=sum_{s=0}^{T-1} A^(T-1-s)E f_s", String.raw`$Wf=\sum_{s=0}^{T-1}A^{T-1-s}Ef_s$`],
    ["<Delta u,v>_mu=-sum_{unordered edges {x,y}}w_xy(u_y-u_x)(v_y-v_x)", String.raw`$\langle\Delta u,v\rangle_\mu=-\sum_{\substack{\text{unordered edges}\\\{x,y\}}}w_{xy}(u_y-u_x)(v_y-v_x)$`],
    ["N={z:(A^k z)|_B=0 for 0<=k<n}", String.raw`$N=\{z:(A^kz)|_B=0\text{ for }0\leq k<n\}$`],
    ["(A^k v)|_B=0", String.raw`$(A^kv)|_B=0$`],
    ["n=|X|", String.raw`$n=|X|$`],
    ["A=I+Delta", String.raw`$A=I+\Delta$`],
    ["let E inject", String.raw`let $E$ inject`],
    ["on B", String.raw`on $B$`],
    ["of w gives", String.raw`of $w$ gives`],
    ["so A is", String.raw`so $A$ is`],
    ["If W is", String.raw`If $W$ is`],
    ["v!=0", String.raw`$v\neq0$`],
    ["A^k E", String.raw`$A^kE$`],
    ["0<=k<T", String.raw`$0\leq k<T$`],
    ["0<=k<n", String.raw`$0\leq k<n$`],
    ["v belongs to N", String.raw`$v$ belongs to $N$`],
    ["z in N", String.raw`$z$ in $N$`],
    ["n-1", String.raw`$n-1$`],
    ["for Az", String.raw`for $Az$`],
    ["shifting k", String.raw`shifting $k$`],
    ["A^n", String.raw`$A^n$`],
    ["N is A-invariant", String.raw`$N$ is $A$-invariant`],
    ["z!=0", String.raw`$z\neq0$`],
    ["-Delta", String.raw`$-\Delta$`],
    ["Hence W is", String.raw`Hence $W$ is`],
    ["T>=n", String.raw`$T\geq n$`],
  ]));

  window.formatResearchText = text => typeof text === "string" ? presentations.get(text) ?? text : text;
})();
