"""Seeded random graph generation: families, edge weights, mu, observed set B, potential q.

Graph i is a pure function of (master_seed, i): its generator is
    numpy.random.default_rng(SeedSequence(master_seed, spawn_key=(i, 0)))
which is child 0 of child i of SeedSequence(master_seed) in the spawn tree. Monte Carlo
streams use spawn keys (i, 1, sweep, replicate, chunk); see mc_common.mc_seed.

Each family is a function (rng, **params) -> Graph that returns the unweighted topology
(W = 0/1 adjacency) with drawing coordinates when they exist. The later stages draw
weights, mu, B and q from the same rng.
"""
from __future__ import annotations

import heapq
import json

import numpy as np

from .config import FAMILY_PARAMS, sample_spec
from .graphs import (PRESETS, Graph, grid_coords, hop_distances, is_connected,
                     lattice_from_coords, pattern_coords)
from .kernel import KernelError, build_kernel, spectral_info


class GenerationError(RuntimeError):
    """No valid graph after max_retries, or a config value that cannot fit the graph."""


# ---------------------------------------------------------------- helpers
def _topology(family, params, A, coords=None, boundary=None) -> Graph:
    n = A.shape[0]
    return Graph(W=A, mu=np.ones(n), B=np.zeros(0, dtype=np.int64), q=np.zeros(n),
                 family=family, params=params, coords=coords, boundary=boundary)


def _from_edges(n, edges) -> np.ndarray:
    A = np.zeros((n, n))
    for a, b in edges:
        if a != b:
            A[a, b] = A[b, a] = 1.0
    return A


def _circle(n) -> np.ndarray:
    angle = 2 * np.pi * np.arange(n) / max(n, 1)
    return np.column_stack([np.cos(angle), np.sin(angle)])


def _nx_seed(rng) -> int:
    return int(rng.integers(0, 2 ** 31 - 1))


def _nx_adjacency(G, n) -> np.ndarray:
    import networkx as nx
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=float)
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------- families
def grid(rng, rows, cols):
    A, coords, boundary = lattice_from_coords(grid_coords(rows, cols))
    return _topology("grid", {"rows": rows, "cols": cols}, A, coords, boundary)


def grid_holes(rng, rows, cols, hole_frac):
    """Remove round(hole_frac * n) vertices in random order, skipping any removal that disconnects."""
    rc = grid_coords(rows, cols)
    n = len(rc)
    target = min(int(round(hole_frac * n)), max(n - 2, 0))
    full, _, _ = lattice_from_coords(rc)
    alive = np.ones(n, dtype=bool)
    removed = 0
    for v in rng.permutation(n):
        if removed >= target:
            break
        alive[v] = False
        if is_connected(full[np.ix_(alive, alive)]):
            removed += 1
        else:
            alive[v] = True
    A, coords, boundary = lattice_from_coords([p for p, a in zip(rc, alive) if a])
    return _topology("grid_holes", {"rows": rows, "cols": cols, "hole_frac": hole_frac,
                                    "holes": removed}, A, coords, boundary)


def pattern(rng, patterns):
    index = int(rng.integers(len(patterns)))
    A, coords, boundary = lattice_from_coords(pattern_coords(patterns[index]))
    return _topology("pattern", {"pattern_index": index, "pattern": list(patterns[index])},
                     A, coords, boundary)


def ladder(rng, length):
    A, coords, boundary = lattice_from_coords(grid_coords(2, length))
    return _topology("ladder", {"length": length}, A, coords, boundary)


def path(rng, n):
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    return _topology("path", {"n": n}, _from_edges(n, [(i, i + 1) for i in range(n - 1)]), coords)


def cycle(rng, n):
    return _topology("cycle", {"n": n}, _from_edges(n, [(i, (i + 1) % n) for i in range(n)]), _circle(n))


def star(rng, n):
    coords = np.vstack([[0.0, 0.0], _circle(n - 1)])
    return _topology("star", {"n": n}, _from_edges(n, [(0, i) for i in range(1, n)]), coords)


def complete(rng, n):
    A = np.ones((n, n)) - np.eye(n)
    return _topology("complete", {"n": n}, A, _circle(n))


def random_tree(rng, n):
    """Uniform labelled tree from a random Pruefer sequence."""
    if n == 2:
        return _topology("random_tree", {"n": n}, _from_edges(2, [(0, 1)]))
    seq = rng.integers(0, n, size=n - 2)
    degree = np.ones(n, dtype=int)
    for x in seq:
        degree[x] += 1
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    edges = []
    for x in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, int(x)))
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, int(x))
    edges.append((heapq.heappop(leaves), heapq.heappop(leaves)))
    return _topology("random_tree", {"n": n}, _from_edges(n, edges))


def erdos_renyi(rng, n, p):
    upper = np.triu(rng.random((n, n)) < p, 1)
    return _topology("erdos_renyi", {"n": n, "p": p}, (upper | upper.T).astype(float))


def random_geometric(rng, n, radius):
    pos = rng.random((n, 2))
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    A = ((dist <= radius) & ~np.eye(n, dtype=bool)).astype(float)
    return _topology("random_geometric", {"n": n, "radius": radius}, A, pos)


def watts_strogatz(rng, n, k, beta):
    import networkx as nx
    k = max(2, min(k, n - 1))
    G = nx.watts_strogatz_graph(n, k, beta, seed=_nx_seed(rng))
    return _topology("watts_strogatz", {"n": n, "k": k, "beta": beta}, _nx_adjacency(G, n), _circle(n))


def barabasi_albert(rng, n, m):
    import networkx as nx
    m = max(1, min(m, n - 1))
    G = nx.barabasi_albert_graph(n, m, seed=_nx_seed(rng))
    return _topology("barabasi_albert", {"n": n, "m": m}, _nx_adjacency(G, n))


def random_regular(rng, n, d):
    import networkx as nx
    d = max(1, min(d, n - 1))
    if (n * d) % 2:
        n += 1                                  # n * d must be even
    G = nx.random_regular_graph(d, n, seed=_nx_seed(rng))
    return _topology("random_regular", {"n": n, "d": d}, _nx_adjacency(G, n), _circle(n))


FAMILIES = {"grid": grid, "grid_holes": grid_holes, "pattern": pattern, "ladder": ladder,
            "path": path, "cycle": cycle, "star": star, "complete": complete,
            "random_tree": random_tree, "erdos_renyi": erdos_renyi,
            "random_geometric": random_geometric, "watts_strogatz": watts_strogatz,
            "barabasi_albert": barabasi_albert, "random_regular": random_regular}
assert set(FAMILIES) == set(FAMILY_PARAMS)


# ---------------------------------------------------------------- weights, mu, B, q
def assign_weights(rng, A: np.ndarray, wcfg: dict) -> np.ndarray:
    n = A.shape[0]
    iu, ju = np.triu_indices(n, 1)
    mask = A[iu, ju] > 0
    m = int(mask.sum())
    dist = wcfg["dist"]
    if dist == "constant":
        vals = np.full(m, wcfg["value"])
    elif dist == "uniform":
        vals = rng.uniform(wcfg["low"], wcfg["high"], m)
    elif dist == "lognormal":
        vals = rng.lognormal(wcfg["mean"], wcfg["sigma"], m)
    elif dist == "discrete":
        vals = rng.choice(np.asarray(wcfg["values"]), size=m, p=wcfg["probs"])
    else:
        vals = rng.gamma(wcfg["shape"], wcfg["scale"], m)
    vals = np.maximum(vals, 1e-300)             # a drawn zero would delete the edge
    rescale = wcfg.get("rescale")
    if rescale and m:
        ref = vals.max() if rescale["mode"] == "max" else vals.mean()
        vals = vals * (rescale["value"] / ref)
    W = np.zeros((n, n))
    W[iu[mask], ju[mask]] = vals
    return W + W.T


def assign_mu(rng, W: np.ndarray, mcfg: dict, max_retries: int) -> np.ndarray | None:
    """Draw mu; enforce Assumption 1(ii) mu >= sum_y W per on_violation. None means skip graph."""
    n = W.shape[0]
    wdeg = W.sum(axis=1)
    deg = (W > 0).sum(axis=1).astype(float)
    random_mode = mcfg["mode"] in ("uniform", "slack")
    for _ in range(max_retries if random_mode else 1):
        mode = mcfg["mode"]
        if mode == "constant":
            mu = np.full(n, mcfg["value"])
        elif mode == "deg":
            mu = deg.copy()
        elif mode == "weighted_deg":
            mu = wdeg.copy()
        elif mode == "uniform":
            mu = rng.uniform(mcfg["low"], mcfg["high"], n)
        else:  # slack: mu = weighted_deg (1 + U(a, b))
            mu = wdeg * (1.0 + rng.uniform(mcfg["a"], mcfg["b"], n))
        bad = mu < wdeg * (1 - 1e-12)
        if not bad.any():
            return mu
        if mcfg["on_violation"] == "fix":
            return np.where(bad, wdeg, mu)
        if mcfg["on_violation"] == "skip":
            return None
    return None                                 # resample exhausted (or deterministic mode)


def _take(rng, candidates: np.ndarray, max_k) -> np.ndarray:
    if max_k is not None and candidates.size > max_k:
        candidates = rng.choice(candidates, size=max_k, replace=False)
    return np.sort(candidates).astype(np.int64)


def choose_B(rng, graph: Graph, bcfg: dict) -> np.ndarray:
    n, mode = graph.n, bcfg["mode"]
    deg = graph.degrees
    if mode == "explicit":
        B = np.asarray(bcfg["vertices"], dtype=np.int64)
        if B.max() >= n:
            raise GenerationError(f"graphs.B.vertices: vertex {int(B.max())} does not exist "
                                  f"in a {graph.family} graph with {n} vertices")
        return B
    if mode == "random_k":
        k = sample_spec(rng, bcfg["k"], "int") if bcfg["k"] is not None else round(bcfg["frac"] * n)
        k = int(min(max(k, 1), n))
        return _take(rng, rng.choice(n, size=k, replace=False), bcfg["max_k"])
    if mode == "boundary":
        if graph.boundary is None:
            raise GenerationError(f"graphs.B.mode=boundary: family {graph.family} has no boundary")
        cand = np.flatnonzero(graph.boundary)
        if cand.size == 0:
            cand = np.arange(n)
        return _take(rng, cand, bcfg["max_k"])
    if mode == "leaves":
        cand = np.flatnonzero(deg == 1)
        if cand.size == 0:                      # no leaves: use the lowest-degree vertex
            cand = np.lexsort((rng.random(n), deg))[:1]
        return _take(rng, cand, bcfg["max_k"])
    k = int(min(max(sample_spec(rng, bcfg["k"], "int"), 1), n))
    key = deg if mode == "low_degree" else -deg
    order = np.lexsort((rng.random(n), key))  # random tie-breaking
    return _take(rng, order[:k], bcfg["max_k"])


def assign_q(rng, graph: Graph, B: np.ndarray, qcfg: dict) -> np.ndarray:
    n, mode = graph.n, qcfg["mode"]
    if mode == "zero":
        q = np.zeros(n)
    elif mode == "constant":
        q = np.full(n, qcfg["c"])
    elif mode == "uniform":
        q = rng.uniform(qcfg["low"], qcfg["high"], n)
    elif mode == "sparse":
        q = np.zeros(n)
        count = int(round(qcfg["fraction"] * n))
        if count:
            idx = rng.choice(n, size=min(count, n), replace=False)
            q[idx] = rng.uniform(qcfg["low"], qcfg["high"], idx.size)
    elif mode == "localized":
        center = int(rng.integers(n)) if qcfg["center"] == "random" else qcfg["center"]
        if center >= n:
            raise GenerationError(f"graphs.q.center: vertex {center} does not exist (n = {n})")
        q = qcfg["amplitude"] * np.exp(-hop_distances(graph.W, center) / qcfg["decay"])
    else:  # explicit
        q = np.asarray(qcfg["values"], dtype=float)
        if q.size != n:
            raise GenerationError(f"graphs.q.values: has {q.size} entries but the graph has {n} vertices")
    inB = np.zeros(n, dtype=bool)
    inB[B] = True
    if qcfg["on_B_only"]:
        q = np.where(inB, q, 0.0)
    if qcfg["off_B_only"]:
        q = np.where(inB, 0.0, q)
    return q


# ---------------------------------------------------------------- top level
def graph_rng(master_seed: int, graph_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(master_seed, spawn_key=(graph_id, 0)))


def generate_graph(gcfg: dict, master_seed: int, graph_id: int) -> Graph:
    """Graph `graph_id` of the run: deterministic in (master_seed, graph_id, gcfg)."""
    rng = graph_rng(master_seed, graph_id)
    meta = {"master_seed": int(master_seed), "spawn_key": [int(graph_id), 0]}
    if gcfg["presets"]:
        name = gcfg["presets"][graph_id % len(gcfg["presets"])]
        g = PRESETS[name]()
        g.q = assign_q(rng, g, g.B, gcfg["q"])
        g.graph_id, g.meta = graph_id, {**meta, "attempts": 1}
        build_kernel(g.W, g.mu, g.q, gcfg["dt"])   # raises if the explicit dt is invalid
        g.validate()
        return g
    names = list(gcfg["families"])
    weights = np.array([gcfg["families"][k]["weight"] for k in names], dtype=float)
    probs = weights / weights.sum()
    last_reason = "no attempt made"
    for attempt in range(1, gcfg["max_retries"] + 1):
        name = names[int(rng.choice(len(names), p=probs))]
        entry = gcfg["families"][name]
        if name == "pattern":
            params = {"patterns": entry["patterns"]}
        else:
            params = {key: sample_spec(rng, entry[key], kind)
                      for key, (kind, _, _) in FAMILY_PARAMS[name].items()}
        topo = FAMILIES[name](rng, **params)
        if topo.n < 2 or not is_connected(topo.W):
            last_reason = f"{name} graph was disconnected"
            continue
        W = assign_weights(rng, topo.W, gcfg["weights"])
        mu = assign_mu(rng, W, gcfg["mu"], gcfg["max_retries"])
        if mu is None:
            last_reason = "mu violated mu >= sum_y W (on_violation = skip/resample)"
            continue
        topo.W = W
        B = choose_B(rng, topo, gcfg["B"])
        q = assign_q(rng, topo, B, gcfg["q"])
        g = Graph(W=W, mu=mu, B=B, q=q, family=name, params=topo.params, coords=topo.coords,
                  boundary=topo.boundary, graph_id=graph_id, meta={**meta, "attempts": attempt})
        try:
            build_kernel(W, mu, q, gcfg["dt"])
        except KernelError as exc:
            last_reason = f"invalid dt: {exc}"
            continue
        g.validate()
        return g
    raise GenerationError(f"graph {graph_id}: no valid graph after {gcfg['max_retries']} attempts "
                          f"(last reason: {last_reason})")


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def graph_record(graph: Graph, dt="auto") -> dict:
    """One row of graphs.parquet: identity, parameters, sizes, degree and spectral statistics."""
    kern = build_kernel(graph.W, graph.mu, graph.q, dt)
    deg = graph.degrees.astype(float)
    return {"graph_id": int(graph.graph_id), "family": graph.family,
            "params_json": json.dumps(graph.params, default=_jsonable, sort_keys=True),
            "n": graph.n, "num_edges": graph.num_edges, "k": int(len(graph.B)),
            "B": [int(b) for b in graph.B],
            "deg_min": float(deg.min()), "deg_mean": float(deg.mean()),
            "deg_max": float(deg.max()), "deg_std": float(deg.std()),
            "wdeg_mean": float(graph.weighted_degrees.mean()),
            "mu_min": float(graph.mu.min()), "mu_mean": float(graph.mu.mean()),
            "mu_max": float(graph.mu.max()),
            "q_min": float(graph.q.min()), "q_mean": float(graph.q.mean()),
            "q_max": float(graph.q.max()), "q_nonzero": int(np.count_nonzero(graph.q)),
            "dt": kern.dt, "dt_mode": kern.dt_mode, **spectral_info(kern.K, graph.mu),
            "master_seed": int(graph.meta.get("master_seed", -1)),
            "spawn_key": json.dumps(graph.meta.get("spawn_key")),
            "attempts": int(graph.meta.get("attempts", 1)),
            "has_coords": graph.coords is not None}
