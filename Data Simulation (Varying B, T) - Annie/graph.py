from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class GridGraph:
    '''
    The weighted graph G = (X, E, mu, w) built on lattice coordinates.

    grid   : list of (row, col) coordinates; vertex i is grid[i]. Two vertices are
             joined by an edge when their coordinates are horizontal or vertical
             neighbours. Use GridGraph.make_grid or GridGraph.grid_from_pattern.
    B      : observation set, as coordinates (row, col) or vertex indices.
    w      : edge weight. possible inputs:
               float       -> w_xy = w on every edge
               dict        -> {((r1, c1), (r2, c2)): w_xy}, either order; edges
                              not listed get w_default
               callable    -> w_xy = w(p, q) for the coordinates p, q of an edge
               "random"    -> w_xy ~ Uniform(w_low, w_high), drawn with w_seed
               n x n array -> used as given (must be symmetric, zero off the lattice edges)
             A weight of 0 removes that edge.
    mu     : vertex centrality. possible inputs:
               "constant"     -> mu_x = mu_val for all x
               "deg"          -> mu_x = deg(x), the number of edges at x
               "weighted_deg" -> mu_x = sum_y w_xy
               array-like     -> mu_x = mu[x], one positive value per vertex (grid order)
    mu_val : the constant used when mu="constant".
    w_default, w_low, w_high, w_seed : see `w` above.

    Attributes W (n x n, symmetric), mu (n,), deg (n,), B (|B|,) are torch tensors.
    Raises ValueError if the graph is disconnected or Assumption 1 (ii) fails.
    '''

    def __init__(self, grid, B, w=1.0, mu="deg", mu_val=1.0,
                 w_default=1.0, w_low=0.1, w_high=1.0, w_seed=0):
        self.grid = [tuple(p) for p in grid]
        self.index = self._build_index(self.grid)
        self.n = len(self.grid)
        self.edges = self._lattice_edges()
        self.W = self._build_weights(w, w_default, w_low, w_high, w_seed)
        self.deg = (self.W > 0).sum(dim=1).to(torch.float64)
        self.mu = self._build_mu(mu, mu_val)
        self.B = self._build_B(B)
        self._check_connected()
        self._check_assumption()

    @staticmethod
    def make_grid(rows, cols):
        '''
        All lattice coordinates (row, col) of a rows x cols rectangular grid, row by row.
        Row 0 is drawn at the top.
        '''
        return [(r, c) for r in range(rows) for c in range(cols)]

    @staticmethod
    def grid_from_pattern(pattern, node="X"):
        '''
        Coordinates from a picture of the graph, one string per row (row 0 on top).
        Characters equal to `node` (case-insensitive) are vertices, anything else is empty;
        spaces are ignored. Example: ["XooX", "XXXX"].
        '''
        rows = [line.replace(" ", "") for line in pattern]
        return [(r, c) for r, line in enumerate(rows)
                for c, ch in enumerate(line) if ch.upper() == node.upper()]

    @staticmethod
    def _build_index(grid):
        index = {p: i for i, p in enumerate(grid)}
        if len(index) != len(grid):
            raise ValueError("grid contains duplicate coordinates")
        return index

    def _lattice_edges(self):
        '''Pairs (i, j), i < j in grid order, of horizontally/vertically adjacent vertices.'''
        edges = []
        for (r, c), i in self.index.items():
            for dr, dc in [(0, 1), (1, 0)]:
                j = self.index.get((r + dr, c + dc))
                if j is not None:
                    edges.append((min(i, j), max(i, j)))
        return sorted(edges)

    def _build_weights(self, w, w_default, w_low, w_high, w_seed):
        n = self.n
        if isinstance(w, (torch.Tensor, np.ndarray)) and np.ndim(w) == 2:
            W = torch.as_tensor(w, dtype=torch.float64).clone()
            if W.shape != (n, n):
                raise ValueError(f"W has shape {tuple(W.shape)}, expected ({n}, {n})")
            if not torch.allclose(W, W.T):
                raise ValueError("W must be symmetric")
            mask = torch.zeros(n, n, dtype=torch.bool)
            for i, j in self.edges:
                mask[i, j] = mask[j, i] = True
            if (W[~mask] != 0).any():
                raise ValueError("W is nonzero on a pair that is not a lattice edge")
        else:
            if isinstance(w, str):
                if w != "random":
                    raise ValueError("w must be a number, dict, callable, 'random' or an n x n array")
                gen = torch.Generator().manual_seed(w_seed)
                vals = w_low + (w_high - w_low) * torch.rand(len(self.edges), generator=gen,
                                                             dtype=torch.float64)
                weight = dict(zip(self.edges, vals.tolist()))
                edge_weight = lambda i, j: weight[(i, j)]
            elif isinstance(w, dict):
                wd = {(tuple(p), tuple(q)): float(v) for (p, q), v in w.items()}
                for p, q in wd:
                    if p not in self.index or q not in self.index:
                        raise ValueError(f"w has a key {(p, q)} that is not in the grid")
                edge_weight = lambda i, j: wd.get((self.grid[i], self.grid[j]),
                                                  wd.get((self.grid[j], self.grid[i]), w_default))
            elif callable(w):
                edge_weight = lambda i, j: float(w(self.grid[i], self.grid[j]))
            else:
                edge_weight = lambda i, j: float(w)

            W = torch.zeros(n, n, dtype=torch.float64)
            for i, j in self.edges:
                W[i, j] = W[j, i] = edge_weight(i, j)

        if (W < 0).any():
            raise ValueError("edge weights must be nonnegative")
        return W

    def _build_mu(self, mu, mu_val):
        n = self.n
        if isinstance(mu, str):
            if mu == "constant":
                mu = torch.full((n,), float(mu_val), dtype=torch.float64)
            elif mu == "deg":
                mu = self.deg.clone()
            elif mu == "weighted_deg":
                mu = self.W.sum(dim=1)
            else:
                raise ValueError("mu must be 'constant', 'deg', 'weighted_deg' or an array of length n")
        else:
            mu = torch.as_tensor(mu, dtype=torch.float64).flatten()
            if mu.numel() != n:
                raise ValueError(f"mu has {mu.numel()} entries, expected {n}")
        if (mu <= 0).any():
            raise ValueError("vertex centrality mu must be positive")
        return mu

    def _build_B(self, B):
        B = torch.tensor([self.index[tuple(b)] if isinstance(b, (tuple, list)) else int(b)
                          for b in B], dtype=torch.long)
        if len(set(B.tolist())) != len(B):
            raise ValueError("B contains duplicate vertices")
        return B

    def _check_connected(self):
        '''Breadth-first search from vertex 0 over edges with w_xy > 0.'''
        seen, frontier = {0}, [0]
        while frontier:
            x = frontier.pop()
            for y in torch.nonzero(self.W[x]).flatten().tolist():
                if y not in seen:
                    seen.add(y)
                    frontier.append(y)
        if len(seen) != self.n:
            raise ValueError("graph is not connected")

    def _check_assumption(self):
        '''Assumption 1 (ii): mu_x >= sum_y w_xy, so that p_xx >= 0.'''
        bad = torch.nonzero(self.mu < self.W.sum(dim=1) - 1e-12).flatten().tolist()
        if bad:
            raise ValueError(f"Assumption 1 (ii) violated at vertices {bad}: mu_x < sum_y w_xy")

    def transition_matrix(self):
        '''P (n x n) for this graph; see transition_matrix(W, mu).'''
        return transition_matrix(self.W, self.mu)


class Graph:
    '''
    A weighted, undirected graph specified by arbitrary hashable vertex labels
    and an explicit edge list.

    vertices : unique, hashable labels; matrix index i corresponds to vertices[i].
    edges    : iterable of (u, v) pairs; zero-weight edges are removed from the walk.
    B        : observed vertex labels.
    w        : scalar, {(u, v): weight} mapping, callable w(u, v), "random",
               or a symmetric n x n matrix.
    mu       : "constant", "deg", "weighted_deg", or one positive value per vertex.

    The W, mu, deg, B, and transition_matrix() interface matches GridGraph.
    '''

    def __init__(self, vertices, edges, B, w=1.0, mu="deg", mu_val=1.0,
                 w_default=1.0, w_low=0.1, w_high=1.0, w_seed=0):
        self.vertices = list(vertices)
        if not self.vertices:
            raise ValueError("vertices must contain at least one vertex")
        try:
            self.index = {vertex: i for i, vertex in enumerate(self.vertices)}
        except TypeError as exc:
            raise ValueError("vertex labels must be hashable") from exc
        if len(self.index) != len(self.vertices):
            raise ValueError("vertices contains duplicate labels")

        self.n = len(self.vertices)
        self.edges = self._build_edges(edges)
        self.W = self._build_weights(w, w_default, w_low, w_high, w_seed)
        self.deg = (self.W > 0).sum(dim=1).to(torch.float64)
        self.mu = self._build_mu(mu, mu_val)
        self.B = self._build_B(B)
        self._check_connected()
        self._check_assumption()

    def _build_edges(self, edges):
        result = []
        seen = set()
        for edge in edges:
            if len(edge) != 2:
                raise ValueError(f"each edge must contain two vertices: {edge}")
            u, v = edge
            if u not in self.index or v not in self.index:
                raise ValueError(f"edge {(u, v)} contains a vertex not in vertices")
            i, j = self.index[u], self.index[v]
            if i == j:
                raise ValueError(f"self-loops are not supported: {edge}")
            pair = (min(i, j), max(i, j))
            if pair in seen:
                raise ValueError(f"duplicate edge: {edge}")
            seen.add(pair)
            result.append(pair)
        return sorted(result)

    def _build_weights(self, w, w_default, w_low, w_high, w_seed):
        if isinstance(w, (torch.Tensor, np.ndarray)) and np.ndim(w) == 2:
            W = torch.as_tensor(w, dtype=torch.float64).clone()
            if W.shape != (self.n, self.n):
                raise ValueError(f"W has shape {tuple(W.shape)}, expected ({self.n}, {self.n})")
            if not torch.allclose(W, W.T):
                raise ValueError("W must be symmetric")
            mask = torch.zeros(self.n, self.n, dtype=torch.bool)
            for i, j in self.edges:
                mask[i, j] = mask[j, i] = True
            if (W[~mask] != 0).any():
                raise ValueError("W is nonzero on a pair that is not in edges")
            if (W < 0).any():
                raise ValueError("edge weights must be nonnegative")
            return W

        if isinstance(w, str):
            if w != "random":
                raise ValueError("w must be a number, dict, callable, 'random' or an n x n array")
            if w_low < 0 or w_high < w_low:
                raise ValueError("random weight bounds must satisfy 0 <= w_low <= w_high")
            gen = torch.Generator().manual_seed(w_seed)
            values = w_low + (w_high - w_low) * torch.rand(
                len(self.edges), generator=gen, dtype=torch.float64
            )
            edge_weights = dict(zip(self.edges, values.tolist()))
            get_weight = lambda i, j: edge_weights[(i, j)]
        elif isinstance(w, dict):
            edge_weights = {}
            for edge, value in w.items():
                if len(edge) != 2:
                    raise ValueError(f"weight key must contain two vertices: {edge}")
                u, v = edge
                if u not in self.index or v not in self.index:
                    raise ValueError(f"weight key {edge} contains a vertex not in vertices")
                pair = tuple(sorted((self.index[u], self.index[v])))
                if pair not in self.edges:
                    raise ValueError(f"weight key {edge} is not an edge")
                edge_weights[pair] = float(value)
            get_weight = lambda i, j: edge_weights.get((i, j), float(w_default))
        elif callable(w):
            get_weight = lambda i, j: float(w(self.vertices[i], self.vertices[j]))
        else:
            get_weight = lambda i, j: float(w)

        W = torch.zeros(self.n, self.n, dtype=torch.float64)
        for i, j in self.edges:
            weight = get_weight(i, j)
            if weight < 0:
                raise ValueError("edge weights must be nonnegative")
            W[i, j] = W[j, i] = weight
        return W

    def _build_mu(self, mu, mu_val):
        if isinstance(mu, str):
            if mu == "constant":
                mu = torch.full((self.n,), float(mu_val), dtype=torch.float64)
            elif mu == "deg":
                mu = self.deg.clone()
            elif mu == "weighted_deg":
                mu = self.W.sum(dim=1)
            else:
                raise ValueError("mu must be 'constant', 'deg', 'weighted_deg' or an array of length n")
        else:
            mu = torch.as_tensor(mu, dtype=torch.float64).flatten()
            if mu.numel() != self.n:
                raise ValueError(f"mu has {mu.numel()} entries, expected {self.n}")
        if (mu <= 0).any():
            raise ValueError("vertex centrality mu must be positive")
        return mu

    def _build_B(self, B):
        try:
            indices = [self.index[vertex] for vertex in B]
        except (KeyError, TypeError) as exc:
            raise ValueError("B must contain vertex labels from vertices") from exc
        if len(set(indices)) != len(indices):
            raise ValueError("B contains duplicate vertices")
        return torch.tensor(indices, dtype=torch.long)

    def _check_connected(self):
        seen, frontier = {0}, [0]
        while frontier:
            x = frontier.pop()
            for y in torch.nonzero(self.W[x]).flatten().tolist():
                if y not in seen:
                    seen.add(y)
                    frontier.append(y)
        if len(seen) != self.n:
            raise ValueError("graph is not connected")

    def _check_assumption(self):
        bad = torch.nonzero(self.mu < self.W.sum(dim=1) - 1e-12).flatten().tolist()
        if bad:
            raise ValueError(
                f"Assumption 1 (ii) violated at vertices {bad}: mu_x < sum_y w_xy"
            )

    def transition_matrix(self):
        '''P (n x n) for this graph; see transition_matrix(W, mu).'''
        return transition_matrix(self.W, self.mu)


def transition_matrix(W, mu):
    '''
    W size: n x n
    mu size: n

    Function returns P (n x n)
    '''
    P = W / mu[:, None]
    stay = 1.0 - P.sum(dim=1)
    if (stay < -1e-12).any():
        raise ValueError("Assumption violated: mu_x < sum_y w_xy for some x")
    return P + torch.diag(stay.clamp_min(0.0))

def _transition_with_absorption(P, q):
    P = torch.as_tensor(P, dtype=torch.float64).cpu()
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square transition matrix")
    if q is None:
        return P

    q = torch.as_tensor(q, dtype=torch.float64).cpu().flatten()
    if q.numel() == 1:
        q = q.expand(P.shape[0])
    if q.numel() != P.shape[0]:
        raise ValueError(f"q has {q.numel()} entries, expected 1 or {P.shape[0]}")
    if not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("absorption probabilities q must be finite and nonnegative")
    if (q > torch.diagonal(P)).any():
        raise ValueError("q_x must not exceed the holding probability p_xx")
    return P - torch.diag(q)

def first_passage_model(G, q, t_max):
    """Differentiable exact first-passage distributions for all pairs in B."""
    K = G.transition_matrix() - torch.diag(q)
    per_target = []
    for y in G.B.tolist():
        r = K[:, y]
        by_time = [r[G.B]]
        avoid_target = torch.arange(G.n) != y
        for _ in range(1, t_max):
            r = K[:, avoid_target] @ r[avoid_target]
            by_time.append(r[G.B])
        per_target.append(torch.stack(by_time))
    return torch.stack(per_target, dim=2)

def exact_first_passage_literal(P, B, t_max, q=None):
    '''
    Exact first-passage distribution for a walk with optional absorption q.

    If q is supplied, it gives one absorption probability per vertex (or one
    shared probability). Absorption replaces part of the holding probability;
    its lost transition mass is not conditioned out.

    Returns r with r[t-1, i, j] = r(t, B[i], B[j]), shape (t_max, |B|, |B|).
    '''
    P = _transition_with_absorption(P, q)
    n = P.shape[0]

    @lru_cache(maxsize=None)
    def r(t, x, y):
        if t == 1:
            return P[x, y].item()
        return sum(P[x, z].item() * r(t - 1, z, y)
                   for z in range(n) if z != y)

    return torch.tensor([[[r(t, int(x), int(y)) for y in B] for x in B]
                         for t in range(1, t_max + 1)], dtype=torch.float64)

def fit_potential(G, q_upper, observed, t_max):
    best_loss, best_q = float("inf"), None
    for initial_q in (0.08, 0.16):
        initial_fraction = (torch.full_like(q_upper, initial_q) / q_upper).clamp(1e-5, 1 - 1e-5)
        theta = torch.nn.Parameter(torch.logit(initial_fraction))
        optimizer = torch.optim.LBFGS(
            [theta], lr=0.8, max_iter=80, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            q_estimate = q_upper * torch.sigmoid(theta)
            prediction = first_passage_model(G, q_estimate, t_max)
            loss = torch.mean((prediction - observed[:t_max]) ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)
        q_estimate = q_upper * torch.sigmoid(theta.detach())
        final_loss = torch.mean((first_passage_model(G, q_estimate, t_max) - observed[:t_max]) ** 2).item()
        if final_loss < best_loss:
            best_loss, best_q = final_loss, q_estimate
    return best_q

@torch.no_grad()
def empirical_first_passage(P, B, t_max, n_walks=10**6, batch=10**6, seed=0, q=None):
    '''
    Returns r' with
        r'[t-1, i, j] = (# walks from B[i] that first hit B[j] at step t) / n_walks.

    Each ordered pair (i, j) gets its own independent n_walks walks. Walks that
    are absorbed or have not hit the target by step t_max contribute no finite
    first-passage count. Absorbed walks are not conditioned out.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    P = _transition_with_absorption(P, q).to(device)
    B = B.to(device)
    nB = len(B)
    n_pairs = nB * nB

    cdf = torch.cumsum(P, dim=1).float()
    cdf[:, -1] = P.sum(dim=1).float().clamp(max=1.0)
    starts  = B.repeat_interleave(nB)
    targets = B.repeat(nB)
    counts = torch.zeros(n_pairs, t_max + 1, dtype=torch.long, device=device)

    done = 0
    while done < n_walks:
        m = min(batch, n_walks - done)
        state = starts[:, None].expand(n_pairs, m).clone()
        tgt = targets[:, None].expand(n_pairs, m)
        hit_time = torch.zeros(n_pairs, m, dtype=torch.long, device=device)
        alive = torch.ones(n_pairs, m, dtype=torch.bool, device=device)

        for s in range(1, t_max + 1):
            u = torch.rand(n_pairs, m, 1, generator=gen, device=device)
            nxt = (u > cdf[state.clamp_max(P.shape[0] - 1)]).sum(dim=-1)
            survived = nxt < P.shape[0]
            new_hit = alive & survived & (nxt == tgt)
            hit_time[new_hit] = s
            alive &= survived & ~new_hit
            state = torch.where(alive, nxt, state)
            if not alive.any():
                break
        offs = torch.arange(n_pairs, device=device)[:, None] * (t_max + 1)
        counts += torch.bincount((hit_time + offs).flatten(),
                                 minlength=n_pairs * (t_max + 1)).view(n_pairs, t_max + 1)
        done += m

    r_emp = counts[:, 1:].double() / n_walks
    return r_emp.T.reshape(t_max, nB, nB).cpu()

def frne(r, r_emp):
    r, r_emp = r.cpu().double(), r_emp.cpu().double()
    return (torch.linalg.norm(r - r_emp) / torch.linalg.norm(r)).item() * 100

B_COLOR, X_COLOR, EDGE_COLOR = "white", "#4cc3e6", "black"
EXACT_COLOR, EMP_COLOR = "#2a6fdb", "#e8833a"

def plot_graph(grid, B, values=None, title=None, ax=None, cmap="viridis", W=None):
    '''
    Draw the grid graph with row 0 at the top.
    B may be coordinates or vertex indices. If `values` (length n) is given,
    vertices are colored by value with a colorbar; otherwise color by membership in B.
    If `W` (n x n) is given, edge width scales with w_xy and each edge is labelled
    with its weight; edges with w_xy = 0 are not drawn.
    '''
    coords = [tuple(p) for p in grid]
    index = {p: i for i, p in enumerate(coords)}
    B_idx = {index[tuple(b)] if isinstance(b, (tuple, list)) else int(b) for b in B}
    xy = np.array([(c, -r) for r, c in coords], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.3 * (np.ptp(xy[:, 0]) + 2), 1.3 * (np.ptp(xy[:, 1]) + 2)))
    for (r, c), i in index.items():
        for dr, dc in [(0, 1), (1, 0)]:
            j = index.get((r + dr, c + dc))
            if j is None:
                continue
            if W is None:
                ax.plot(*xy[[i, j]].T, color=EDGE_COLOR, lw=1.5, zorder=1)
            elif W[i, j] > 0:
                wij = float(W[i, j])
                ax.plot(*xy[[i, j]].T, color=EDGE_COLOR, lw=0.5 + 3.5 * wij / float(W.max()), zorder=1)
                ax.annotate(f"{wij:.2f}", xy[[i, j]].mean(axis=0), ha="center", va="center",
                            fontsize=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    if values is None:
        colors = [B_COLOR if i in B_idx else X_COLOR for i in range(len(coords))]
        ax.scatter(*xy.T, s=140, c=colors, edgecolors="black", linewidths=1, zorder=2)
        ax.scatter([], [], s=80, c=B_COLOR, edgecolors="black", label="vertex in $B$")
        ax.scatter([], [], s=80, c=X_COLOR, edgecolors="black", label=r"vertex in $X \setminus B$")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    else:
        values = np.asarray(torch.as_tensor(values).cpu(), dtype=float)
        sc = ax.scatter(*xy.T, s=180, c=values, cmap=cmap, edgecolors="black",
                        linewidths=1, zorder=2)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        bi = sorted(B_idx)
        ax.scatter(*xy[bi].T, s=180, facecolors="none", edgecolors="black",
                   linewidths=2.5, zorder=3)

    for i, (x, y) in enumerate(xy):
        ax.annotate(str(i), (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.margins(0.25)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def plot_general_graph(graph, B=None, values=None, title=None, ax=None,
                       cmap="viridis", positions=None, show_weights=True):
    '''Plot a Graph using a circular layout or caller-supplied {vertex: (x, y)} positions.'''
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph instance")
    if positions is None:
        angles = np.linspace(0, 2 * np.pi, graph.n, endpoint=False)
        xy = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        if set(positions) != set(graph.vertices):
            raise ValueError("positions must provide one coordinate for every graph vertex")
        xy = np.asarray([positions[vertex] for vertex in graph.vertices], dtype=float)
        if xy.shape != (graph.n, 2):
            raise ValueError(f"positions must have shape ({graph.n}, 2)")

    B_idx = set(graph.B.tolist()) if B is None else {
        graph.index[vertex] for vertex in B
    }
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    max_weight = float(graph.W.max())
    for i, j in graph.edges:
        weight = float(graph.W[i, j])
        if weight <= 0:
            continue
        linewidth = 1.5 if max_weight == 0 else 0.5 + 3.5 * weight / max_weight
        ax.plot(*xy[[i, j]].T, color=EDGE_COLOR, lw=linewidth, zorder=1)
        if show_weights:
            ax.annotate(f"{weight:.2f}", xy[[i, j]].mean(axis=0),
                        ha="center", va="center", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    if values is None:
        colors = [B_COLOR if i in B_idx else X_COLOR for i in range(graph.n)]
        ax.scatter(*xy.T, s=180, c=colors, edgecolors="black", linewidths=1, zorder=2)
        ax.scatter([], [], s=80, c=B_COLOR, edgecolors="black", label="vertex in $B$")
        ax.scatter([], [], s=80, c=X_COLOR, edgecolors="black",
                   label=r"vertex in $X \setminus B$")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    else:
        values = np.asarray(torch.as_tensor(values).cpu(), dtype=float)
        if values.shape != (graph.n,):
            raise ValueError(f"values must have shape ({graph.n},)")
        scatter = ax.scatter(*xy.T, s=200, c=values, cmap=cmap,
                             edgecolors="black", linewidths=1, zorder=2)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        observed = sorted(B_idx)
        ax.scatter(*xy[observed].T, s=200, facecolors="none",
                   edgecolors="black", linewidths=2.5, zorder=3)

    for i, vertex in enumerate(graph.vertices):
        ax.annotate(str(vertex), xy[i], xytext=(6, 6),
                    textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.margins(0.25)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def _plot_fpt_grid(series, B, title):
    '''One panel per ordered pair (B[i], B[j]); `series` is a list of (r, label, color, kind).'''
    B = [int(b) for b in B]
    nB = len(B)
    t_max = series[0][0].shape[0]
    t = np.arange(1, t_max + 1)
    fig, axes = plt.subplots(nB, nB, figsize=(3.2 * nB, 2.4 * nB),
                             sharex=True, sharey=True, squeeze=False)
    for i in range(nB):
        for j in range(nB):
            ax = axes[i, j]
            for r, label, color, kind in series:
                vals = r[:, i, j].cpu().numpy()
                if kind == "bar":
                    ax.bar(t, vals, width=0.7, color=color, label=label)
                else:
                    ax.plot(t, vals, "o-", ms=4, lw=1.5, color=color, label=label)
            ax.set_title(f"$x = {B[i]},\\ y = {B[j]}$", fontsize=10)
            ax.grid(axis="y", color="0.9", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if i == nB - 1:
                ax.set_xlabel("$t$")
            if j == 0:
                ax.set_ylabel("probability")
    if len(series) > 1:
        axes[0, -1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_exact_distribution(r, B, title="Exact first passage distribution $r(t, x, y)$"):
    '''Bar chart of r(t, x, y) for t = 1..t_max and every (x, y) in B x B.'''
    return _plot_fpt_grid([(r, "exact $r$", EXACT_COLOR, "bar")], B, title)


def plot_empirical_distribution(r_emp, B, r=None,
                                title="Empirical first passage distribution $r'(t, x, y)$"):
    '''Bar chart of r'(t, x, y); if the exact r is given it is overlaid as a line.'''
    series = [(r_emp, "empirical $r'$", EMP_COLOR, "bar")]
    if r is not None:
        series.append((r, "exact $r$", EXACT_COLOR, "line"))
    return _plot_fpt_grid(series, B, title)


def summarize(P, B, r_emp, r_lit, paper_frne=None):
    '''Print FRNE and sanity checks for one experiment.'''
    r, r_emp, B_cpu = r_lit.cpu(), r_emp.cpu(), B.cpu()
    print("data shape (2T-1, |B|, |B|):", tuple(r.shape))
    msg = f"FRNE = {frne(r, r_emp):.6f}%"
    if paper_frne is not None:
        msg += f"   (paper: {paper_frne}%)"
    print(msg)
    print("check r(1, y, y) = p_yy:",
          torch.allclose(torch.diagonal(r[0]), torch.diagonal(P.cpu())[B_cpu]))
    print("probability mass observed by t = 2T-1 (exact):\n", r.sum(0).numpy().round(4))
    print("probability mass observed by t = 2T-1 (empirical):\n", r_emp.sum(0).numpy().round(4))
