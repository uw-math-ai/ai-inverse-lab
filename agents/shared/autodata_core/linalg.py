"""Small dense linear algebra in pure Python (the runtime image has no NumPy).

Matrices are lists of row lists. Sizes here are tiny (at most ~30 x 30), so O(m^3) is fine.
"""
from __future__ import annotations

import math
from operator import mul


def dot(a, b) -> float:
    return sum(map(mul, a, b))


def gram(cols: list[list[float]]) -> list[list[float]]:
    """G[p][q] = <cols[p], cols[q]> (symmetric)."""
    m = len(cols)
    G = [[0.0] * m for _ in range(m)]
    for p in range(m):
        cp = cols[p]
        for q in range(p, m):
            G[p][q] = G[q][p] = dot(cp, cols[q])
    return G


def solve(A: list[list[float]], b: list[float]) -> list[float]:
    """x with A x = b, Gaussian elimination with partial pivoting. Raises ZeroDivisionError if singular."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ZeroDivisionError("singular matrix")
        M[c], M[piv] = M[piv], M[c]
        pr = M[c]
        inv = 1.0 / pr[c]
        for r in range(c + 1, n):
            f = M[r][c] * inv
            if f:
                row = M[r]
                for j in range(c, n + 1):
                    row[j] -= f * pr[j]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        row = M[r]
        s = row[n] - sum(row[j] * x[j] for j in range(r + 1, n))
        x[r] = s / row[r]
    return x


def cholesky(A: list[list[float]]) -> list[list[float]]:
    """Lower-triangular L with A = L L^T. Raises ValueError if A is not positive definite."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][t] * L[j][t] for t in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def spd_inverse_diag(A: list[list[float]]) -> list[float]:
    """diag(A^{-1}) for symmetric positive definite A (via Cholesky). Raises ValueError otherwise."""
    n = len(A)
    L = cholesky(A)
    out = []
    for c in range(n):
        # (A^{-1})_{cc} = e_c^T L^{-T} L^{-1} e_c = ||y||^2 with L y = e_c
        y = [0.0] * n
        for i in range(n):
            s = (1.0 if i == c else 0.0) - sum(L[i][t] * y[t] for t in range(i))
            y[i] = s / L[i][i]
        out.append(sum(v * v for v in y))
    return out


def sym_eigenvalues(A: list[list[float]], sweeps: int = 50, tol: float = 1e-14) -> list[float]:
    """Eigenvalues of a symmetric matrix by the cyclic Jacobi method, ascending."""
    n = len(A)
    a = [row[:] for row in A]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        scale = sum(a[i][i] ** 2 for i in range(n)) or 1.0
        if off <= tol * tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    return sorted(a[i][i] for i in range(n))
