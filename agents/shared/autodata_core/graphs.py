"""Pure-Python port of fptsim's spec schema and graph generation (stdlib only).

The keys, modes and meanings are those of fptsim ("Toggable Simulation of Data Code/README.md",
section 3), so a spec written for the agent also runs in fptsim and vice versa. Randomness comes
from Python's `random.Random`, so a seed gives the same graph every time *here*, but not the same
graph as fptsim's NumPy generator (statistically equivalent, not bit-identical).

Kernel conventions (fptsim.kernel):
    s(x) = sum_y W[x,y] / mu(x);  K[x,y] = h W[x,y] / mu(x);  K[x,x] = 1 - h (s(x) + q(x))
    dt: auto -> h = min(1, 1 / max_x (s(x) + q(x)))
"""
from __future__ import annotations

import heapq
import math
import random
from collections import deque


class SpecError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


FAMILY_PARAMS = {
    "grid": {"rows": ("int", 1, 10_000), "cols": ("int", 1, 10_000)},
    "grid_holes": {"rows": ("int", 1, 10_000), "cols": ("int", 1, 10_000), "hole_frac": ("float", 0.0, 0.9)},
    "pattern": {},
    "ladder": {"length": ("int", 1, 100_000)},
    "path": {"n": ("int", 2, 100_000)},
    "cycle": {"n": ("int", 3, 100_000)},
    "star": {"n": ("int", 2, 100_000)},
    "complete": {"n": ("int", 2, 100_000)},
    "random_tree": {"n": ("int", 2, 100_000)},
    "erdos_renyi": {"n": ("int", 2, 100_000), "p": ("float", 0.0, 1.0)},
    "random_geometric": {"n": ("int", 2, 100_000), "radius": ("float", 0.0, 2.0)},
    "watts_strogatz": {"n": ("int", 3, 100_000), "k": ("int", 2, 100_000), "beta": ("float", 0.0, 1.0)},
    "barabasi_albert": {"n": ("int", 2, 100_000), "m": ("int", 1, 100_000)},
    "random_regular": {"n": ("int", 2, 100_000), "d": ("int", 1, 100_000)},
}
LATTICE = {"grid", "grid_holes", "pattern", "ladder"}
W_DISTS = {"constant": {"value"}, "uniform": {"low", "high"}, "lognormal": {"mean", "sigma"},
           "discrete": {"values", "probs"}, "gamma": {"shape", "scale"}}
MU_MODES = {"constant": {"value"}, "deg": set(), "weighted_deg": set(), "uniform": {"low", "high"},
            "slack": {"a", "b"}}
B_MODES = {"random_k": {"k", "frac"}, "boundary": set(), "leaves": set(), "low_degree": {"k"},
           "high_degree": {"k"}, "explicit": {"vertices"}}
Q_MODES = {"zero": set(), "constant": {"c"}, "uniform": {"low", "high"},
           "sparse": {"fraction", "low", "high"}, "localized": {"center", "decay", "amplitude"},
           "explicit": {"values"}}


# ---------------------------------------------------------------- strict validation
def _err(where, msg):
    raise SpecError("invalid_spec", f"{where}: {msg}")


def _number(v, where):
    if isinstance(v, bool):
        _err(where, f"must be a number, got {v!r}")
    if isinstance(v, (int, float)):
        if not math.isfinite(v):
            _err(where, "must be finite")
        return v
    if isinstance(v, str):
        try:
            f = float(v.replace("_", ""))
        except ValueError:
            _err(where, f"must be a number, got {v!r}")
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    _err(where, f"must be a number, got {type(v).__name__}")


def _num(v, where, lo=None, hi=None, integer=False, lo_open=False):
    x = _number(v, where)
    if integer:
        if float(x) != int(x):
            _err(where, f"must be an integer, got {v!r}")
        x = int(x)
    else:
        x = float(x)
    if lo is not None and (x < lo or (lo_open and x == lo)):
        _err(where, f"must be {'>' if lo_open else '>='} {lo}, got {x}")
    if hi is not None and x > hi:
        _err(where, f"must be <= {hi}, got {x}")
    return x


def _keys(d, allowed, where):
    if not isinstance(d, dict):
        _err(where, f"must be a mapping, got {type(d).__name__}")
    extra = set(d) - set(allowed)
    if extra:
        _err(where, f"unknown keys {sorted(extra)} (allowed: {sorted(allowed)})")


def _param(v, where, kind, lo, hi):
    """fixed value | [lo, hi] inclusive range | {choice: [...]}"""
    integer = kind == "int"
    if isinstance(v, dict):
        _keys(v, {"choice"}, where)
        ch = v.get("choice")
        if not isinstance(ch, list) or not ch:
            _err(where, "choice must be a nonempty list")
        return {"choice": [_num(c, f"{where}.choice[{i}]", lo, hi, integer) for i, c in enumerate(ch)]}
    if isinstance(v, list):
        if len(v) != 2:
            _err(where, "a range must be [low, high]")
        a, b = _num(v[0], f"{where}[0]", lo, hi, integer), _num(v[1], f"{where}[1]", lo, hi, integer)
        if a > b:
            _err(where, f"range low {a} > high {b}")
        return [a, b]
    return _num(v, where, lo, hi, integer)


def sample(rng: random.Random, spec, kind):
    if isinstance(spec, dict):
        return rng.choice(spec["choice"])
    if isinstance(spec, list):
        return rng.randint(spec[0], spec[1]) if kind == "int" else rng.uniform(spec[0], spec[1])
    return spec


def spec_max(spec):
    if isinstance(spec, dict):
        return max(spec["choice"])
    if isinstance(spec, list):
        return spec[1]
    return spec


def pattern_coords(pattern):
    rows = [line.replace(" ", "") for line in pattern]
    return [(r, c) for r, line in enumerate(rows) for c, ch in enumerate(line) if ch.upper() == "X"]


def validate_graphs(g) -> dict:
    where = "graphs"
    _keys(g, {"families", "presets", "weights", "mu", "B", "q", "dt", "max_retries"}, where)
    out = {}
    if g.get("presets") is not None:
        pre = g["presets"]
        if not isinstance(pre, list) or not pre or any(p not in PRESETS for p in pre):
            _err("graphs.presets", f"must be a nonempty list of {sorted(PRESETS)}")
        for key in ("families", "weights", "mu", "B"):
            if g.get(key) is not None:
                _err(f"graphs.{key}", "not allowed together with graphs.presets")
        out.update(presets=list(pre), families=None, weights=None, mu=None, B=None)
    else:
        fams = g.get("families")
        if not isinstance(fams, dict) or not fams:
            _err("graphs.families", "required: a mapping {family: {params}} (or give graphs.presets)")
        out["presets"] = None
        out["families"] = {}
        for name, entry in fams.items():
            w = f"graphs.families.{name}"
            if name not in FAMILY_PARAMS:
                _err(w, f"unknown family (known: {sorted(FAMILY_PARAMS)})")
            entry = entry or {}
            if name == "pattern":
                _keys(entry, {"weight", "patterns"}, w)
                pats = entry.get("patterns")
                if not isinstance(pats, list) or not pats:
                    _err(f"{w}.patterns", "a nonempty list of pictures like ['XooX', 'XXXX']")
                pats = [p if isinstance(p, list) else [p] for p in pats]
                for i, p in enumerate(pats):
                    if not all(isinstance(line, str) for line in p) or not pattern_coords(p):
                        _err(f"{w}.patterns[{i}]", "must be rows of 'X' (vertex) and 'o' (empty)")
                res = {"patterns": pats}
            else:
                params = FAMILY_PARAMS[name]
                _keys(entry, {"weight", *params}, w)
                res = {}
                for key, (kind, lo, hi) in params.items():
                    if key not in entry:
                        _err(f"{w}.{key}", "required")
                    res[key] = _param(entry[key], f"{w}.{key}", kind, lo, hi)
            res["weight"] = _num(entry.get("weight", 1), f"{w}.weight", 0, None, lo_open=True)
            out["families"][name] = res
        wcfg = g.get("weights") or {"dist": "constant", "value": 1.0}
        _keys(wcfg, {"dist", "rescale"} | W_DISTS.get(wcfg.get("dist"), set()), "graphs.weights")
        dist = wcfg.get("dist")
        if dist not in W_DISTS:
            _err("graphs.weights.dist", f"one of {sorted(W_DISTS)}")
        wres = {"dist": dist}
        if dist == "constant":
            wres["value"] = _num(wcfg.get("value"), "graphs.weights.value", 0, None, lo_open=True)
        elif dist == "uniform":
            wres["low"] = _num(wcfg.get("low"), "graphs.weights.low", 0, None, lo_open=True)
            wres["high"] = _num(wcfg.get("high"), "graphs.weights.high", wres["low"])
        elif dist == "lognormal":
            wres["mean"] = _num(wcfg.get("mean"), "graphs.weights.mean")
            wres["sigma"] = _num(wcfg.get("sigma"), "graphs.weights.sigma", 0)
        elif dist == "gamma":
            wres["shape"] = _num(wcfg.get("shape"), "graphs.weights.shape", 0, None, lo_open=True)
            wres["scale"] = _num(wcfg.get("scale"), "graphs.weights.scale", 0, None, lo_open=True)
        else:
            vals, probs = wcfg.get("values"), wcfg.get("probs")
            if not isinstance(vals, list) or not isinstance(probs, list) or len(vals) != len(probs) or not vals:
                _err("graphs.weights", "discrete needs equal-length lists values and probs")
            wres["values"] = [_num(v, f"graphs.weights.values[{i}]", 0, None, lo_open=True) for i, v in enumerate(vals)]
            wres["probs"] = [_num(p, f"graphs.weights.probs[{i}]", 0) for i, p in enumerate(probs)]
            if abs(sum(wres["probs"]) - 1) > 1e-9:
                _err("graphs.weights.probs", "must sum to 1")
        rs = wcfg.get("rescale")
        if rs is not None:
            _keys(rs, {"mode", "value"}, "graphs.weights.rescale")
            if rs.get("mode") not in ("max", "mean"):
                _err("graphs.weights.rescale.mode", "max | mean")
            rs = {"mode": rs["mode"], "value": _num(rs.get("value"), "graphs.weights.rescale.value", 0, None, lo_open=True)}
        wres["rescale"] = rs
        out["weights"] = wres
        mcfg = g.get("mu") or {"mode": "deg"}
        mode = mcfg.get("mode")
        if mode not in MU_MODES:
            _err("graphs.mu.mode", f"one of {sorted(MU_MODES)}")
        _keys(mcfg, {"mode", "on_violation"} | MU_MODES[mode], "graphs.mu")
        mres = {"mode": mode, "on_violation": mcfg.get("on_violation", "fix")}
        if mres["on_violation"] not in ("fix", "resample", "skip"):
            _err("graphs.mu.on_violation", "fix | resample | skip")
        if mode == "constant":
            mres["value"] = _num(mcfg.get("value"), "graphs.mu.value", 0, None, lo_open=True)
        elif mode == "uniform":
            mres["low"] = _num(mcfg.get("low"), "graphs.mu.low", 0, None, lo_open=True)
            mres["high"] = _num(mcfg.get("high"), "graphs.mu.high", mres["low"])
        elif mode == "slack":
            mres["a"] = _num(mcfg.get("a"), "graphs.mu.a", 0)
            mres["b"] = _num(mcfg.get("b"), "graphs.mu.b", mres["a"])
        out["mu"] = mres
        bcfg = g.get("B")
        if bcfg is None:
            _err("graphs.B", "required (for example {mode: random_k, k: [2, 5]})")
        bmode = bcfg.get("mode") if isinstance(bcfg, dict) else None
        if bmode not in B_MODES:
            _err("graphs.B.mode", f"one of {sorted(B_MODES)}")
        _keys(bcfg, {"mode"} | B_MODES[bmode] | ({"max_k"} if bmode != "explicit" else set()), "graphs.B")
        bres = {"mode": bmode, "max_k": None if bcfg.get("max_k") is None
                else _num(bcfg["max_k"], "graphs.B.max_k", 1, None, True)}
        if bmode == "random_k":
            if (bcfg.get("k") is None) == (bcfg.get("frac") is None):
                _err("graphs.B", "random_k needs exactly one of k or frac")
            bres["k"] = None if bcfg.get("k") is None else _param(bcfg["k"], "graphs.B.k", "int", 1, 10 ** 6)
            bres["frac"] = None if bcfg.get("frac") is None else _num(bcfg["frac"], "graphs.B.frac", 0, 1, lo_open=True)
        elif bmode in ("low_degree", "high_degree"):
            if "k" not in bcfg:
                _err("graphs.B.k", f"required for mode={bmode}")
            bres["k"] = _param(bcfg["k"], "graphs.B.k", "int", 1, 10 ** 6)
        elif bmode == "explicit":
            vs = bcfg.get("vertices")
            if not isinstance(vs, list) or not vs:
                _err("graphs.B.vertices", "a nonempty list")
            vs = [_num(v, f"graphs.B.vertices[{i}]", 0, None, True) for i, v in enumerate(vs)]
            if len(set(vs)) != len(vs):
                _err("graphs.B.vertices", "contains duplicates")
            bres["vertices"] = vs
        if bmode == "boundary" and any(f not in LATTICE for f in out["families"]):
            _err("graphs.B.mode", "boundary needs lattice families only (grid, grid_holes, pattern, ladder)")
        out["B"] = bres
    qcfg = g.get("q") or {"mode": "zero"}
    qmode = qcfg.get("mode")
    if qmode not in Q_MODES:
        _err("graphs.q.mode", f"one of {sorted(Q_MODES)}")
    _keys(qcfg, {"mode", "on_B_only", "off_B_only"} | Q_MODES[qmode], "graphs.q")
    qres = {"mode": qmode, "on_B_only": bool(qcfg.get("on_B_only", False)),
            "off_B_only": bool(qcfg.get("off_B_only", False))}
    if qmode == "constant":
        qres["c"] = _num(qcfg.get("c"), "graphs.q.c", 0)
    elif qmode in ("uniform", "sparse"):
        qres["low"] = _num(qcfg.get("low"), "graphs.q.low", 0)
        qres["high"] = _num(qcfg.get("high"), "graphs.q.high", qres["low"])
        if qmode == "sparse":
            qres["fraction"] = _num(qcfg.get("fraction"), "graphs.q.fraction", 0, 1)
    elif qmode == "localized":
        c = qcfg.get("center", "random")
        qres["center"] = "random" if c == "random" else _num(c, "graphs.q.center", 0, None, True)
        qres["decay"] = _num(qcfg.get("decay"), "graphs.q.decay", 0, None, lo_open=True)
        qres["amplitude"] = _num(qcfg.get("amplitude"), "graphs.q.amplitude", 0)
    elif qmode == "explicit":
        vals = qcfg.get("values")
        if not isinstance(vals, list) or not vals:
            _err("graphs.q.values", "a nonempty list")
        qres["values"] = [_num(v, f"graphs.q.values[{i}]", 0) for i, v in enumerate(vals)]
    out["q"] = qres
    dt = g.get("dt", "auto")
    out["dt"] = "auto" if dt == "auto" else _num(dt, "graphs.dt", 0, 1, lo_open=True)
    out["max_retries"] = _num(g.get("max_retries", 100), "graphs.max_retries", 1, 100_000, True)
    return out


def validate_simulation(s) -> dict:
    _keys(s, {"T", "T_mode", "N"}, "simulation")
    for key in ("T", "N"):
        if key not in s:
            _err(f"simulation.{key}", "required")
    mode = s.get("T_mode", "steps")
    if mode not in ("steps", "paper"):
        _err("simulation.T_mode", "steps | paper")
    T = _num(s["T"], "simulation.T", 1, 1_000_000, True)
    return {"T": T, "T_mode": mode, "T_steps": 2 * T - 1 if mode == "paper" else T,
            "N": _num(s["N"], "simulation.N", 1, 10 ** 12, True)}


def vertex_bound(gres: dict) -> int:
    if gres.get("presets"):
        return max(len(PRESETS[p]()["mu"]) for p in gres["presets"])
    worst = 0
    for name, e in gres["families"].items():
        if name == "pattern":
            worst = max(worst, max(len(pattern_coords(p)) for p in e["patterns"]))
        elif name in ("grid", "grid_holes"):
            worst = max(worst, int(spec_max(e["rows"])) * int(spec_max(e["cols"])))
        elif name == "ladder":
            worst = max(worst, 2 * int(spec_max(e["length"])))
        else:
            worst = max(worst, int(spec_max(e["n"])) + (1 if name == "random_regular" else 0))
    return worst


# ---------------------------------------------------------------- topologies (adjacency as sets)
def _empty(n):
    return [set() for _ in range(n)]


def _edges(n, pairs):
    adj = _empty(n)
    for a, b in pairs:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def lattice(rc):
    index = {p: i for i, p in enumerate(rc)}
    adj = _empty(len(rc))
    for (r, c), i in index.items():
        for dr, dc in ((0, 1), (1, 0)):
            j = index.get((r + dr, c + dc))
            if j is not None:
                adj[i].add(j)
                adj[j].add(i)
    return adj, [len(a) < 4 for a in adj]


def connected(adj, alive=None) -> bool:
    nodes = [i for i in range(len(adj)) if alive is None or alive[i]]
    if not nodes:
        return False
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen and (alive is None or alive[y]):
                seen.add(y)
                stack.append(y)
    return len(seen) == len(nodes)


def hops(adj, source):
    dist = [math.inf] * len(adj)
    dist[source] = 0
    q = deque([source])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if dist[y] == math.inf:
                dist[y] = dist[x] + 1
                q.append(y)
    return dist


def topology(rng, name, p):
    """-> (adjacency list of sets, boundary flags or None, params as used)"""
    if name == "grid":
        adj, bd = lattice([(r, c) for r in range(p["rows"]) for c in range(p["cols"])])
        return adj, bd, p
    if name == "ladder":
        adj, bd = lattice([(r, c) for r in range(2) for c in range(p["length"])])
        return adj, bd, p
    if name == "pattern":
        i = rng.randrange(len(p["patterns"]))
        adj, bd = lattice(pattern_coords(p["patterns"][i]))
        return adj, bd, {"pattern_index": i}
    if name == "grid_holes":
        rc = [(r, c) for r in range(p["rows"]) for c in range(p["cols"])]
        n = len(rc)
        full, _ = lattice(rc)
        target = min(int(round(p["hole_frac"] * n)), max(n - 2, 0))
        alive = [True] * n
        removed = 0
        order = list(range(n))
        rng.shuffle(order)
        for v in order:
            if removed >= target:
                break
            alive[v] = False
            if connected(full, alive):
                removed += 1
            else:
                alive[v] = True
        adj, bd = lattice([q for q, a in zip(rc, alive) if a])
        return adj, bd, {**p, "holes": removed}
    n = p.get("n")
    if name == "path":
        return _edges(n, [(i, i + 1) for i in range(n - 1)]), None, p
    if name == "cycle":
        return _edges(n, [(i, (i + 1) % n) for i in range(n)]), None, p
    if name == "star":
        return _edges(n, [(0, i) for i in range(1, n)]), None, p
    if name == "complete":
        return _edges(n, [(i, j) for i in range(n) for j in range(i + 1, n)]), None, p
    if name == "random_tree":                       # uniform labelled tree (Pruefer sequence)
        if n == 2:
            return _edges(2, [(0, 1)]), None, p
        seq = [rng.randrange(n) for _ in range(n - 2)]
        deg = [1] * n
        for x in seq:
            deg[x] += 1
        leaves = [i for i in range(n) if deg[i] == 1]
        heapq.heapify(leaves)
        pairs = []
        for x in seq:
            leaf = heapq.heappop(leaves)
            pairs.append((leaf, x))
            deg[x] -= 1
            if deg[x] == 1:
                heapq.heappush(leaves, x)
        pairs.append((heapq.heappop(leaves), heapq.heappop(leaves)))
        return _edges(n, pairs), None, p
    if name == "erdos_renyi":
        return _edges(n, [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p["p"]]), None, p
    if name == "random_geometric":
        pos = [(rng.random(), rng.random()) for _ in range(n)]
        return _edges(n, [(i, j) for i in range(n) for j in range(i + 1, n)
                          if math.dist(pos[i], pos[j]) <= p["radius"]]), None, p
    if name == "watts_strogatz":                    # ring lattice with k nearest neighbours, rewired
        k = max(2, min(p["k"], n - 1))
        adj = _edges(n, [(i, (i + d) % n) for i in range(n) for d in range(1, k // 2 + 1)])
        for d in range(1, k // 2 + 1):
            for i in range(n):
                j = (i + d) % n
                if rng.random() < p["beta"] and j in adj[i]:
                    choices = [w for w in range(n) if w != i and w not in adj[i]]
                    if choices:
                        w = rng.choice(choices)
                        adj[i].discard(j)
                        adj[j].discard(i)
                        adj[i].add(w)
                        adj[w].add(i)
        return adj, None, {**p, "k": k}
    if name == "barabasi_albert":                   # preferential attachment, m edges per new vertex
        m = max(1, min(p["m"], n - 1))
        adj = _empty(n)
        targets, repeated = list(range(m)), []
        for v in range(m, n):
            for t in set(targets):
                adj[v].add(t)
                adj[t].add(v)
                repeated += [v, t]
            chosen = set()
            while len(chosen) < m:
                chosen.add(rng.choice(repeated))
            targets = list(chosen)
        return adj, None, {**p, "m": m}
    if name == "random_regular":                    # pairing model with restarts
        d = max(1, min(p["d"], n - 1))
        if (n * d) % 2:
            n += 1
        for _ in range(1000):
            stubs = [v for v in range(n) for _ in range(d)]
            rng.shuffle(stubs)
            adj, ok = _empty(n), True
            for a, b in zip(stubs[::2], stubs[1::2]):
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if ok:
                return adj, None, {"n": n, "d": d}
        raise SpecError("generation_failed", f"no simple {d}-regular graph on {n} vertices found")
    raise SpecError("invalid_spec", f"unknown family {name!r}")


# ---------------------------------------------------------------- the paper graphs
def _preset(rows, cols, w, mu_deg, B):
    adj, bd = lattice([(r, c) for r in range(rows) for c in range(cols)])
    n = len(adj)
    W = [[w if j in adj[i] else 0.0 for j in range(n)] for i in range(n)]
    mu = [float(len(adj[i])) for i in range(n)] if mu_deg else [1.0] * n
    return {"W": W, "mu": mu, "B": B, "boundary": bd, "adj": adj}


PRESETS = {"paper8": lambda: _preset(2, 4, 0.25, False, [1, 2, 3]),     # Fig. 1
           "paper9": lambda: _preset(3, 3, 1.0, True, [0, 1, 2])}       # Fig. 4


# ---------------------------------------------------------------- weights, mu, B, q
def _weights(rng, adj, wcfg):
    n = len(adj)
    pairs = [(i, j) for i in range(n) for j in sorted(adj[i]) if j > i]
    dist = wcfg["dist"]
    if dist == "constant":
        vals = [wcfg["value"]] * len(pairs)
    elif dist == "uniform":
        vals = [rng.uniform(wcfg["low"], wcfg["high"]) for _ in pairs]
    elif dist == "lognormal":
        vals = [rng.lognormvariate(wcfg["mean"], wcfg["sigma"]) for _ in pairs]
    elif dist == "gamma":
        vals = [rng.gammavariate(wcfg["shape"], wcfg["scale"]) for _ in pairs]
    else:
        vals = rng.choices(wcfg["values"], weights=wcfg["probs"], k=len(pairs))
    vals = [max(v, 1e-300) for v in vals]
    rs = wcfg.get("rescale")
    if rs and vals:
        ref = max(vals) if rs["mode"] == "max" else sum(vals) / len(vals)
        vals = [v * rs["value"] / ref for v in vals]
    W = [[0.0] * n for _ in range(n)]
    for (i, j), v in zip(pairs, vals):
        W[i][j] = W[j][i] = float(v)
    return W


def _mu(rng, W, adj, mcfg, retries):
    n = len(W)
    wdeg = [sum(row) for row in W]
    for _ in range(retries if mcfg["mode"] in ("uniform", "slack") else 1):
        mode = mcfg["mode"]
        if mode == "constant":
            mu = [mcfg["value"]] * n
        elif mode == "deg":
            mu = [float(len(a)) for a in adj]
        elif mode == "weighted_deg":
            mu = wdeg[:]
        elif mode == "uniform":
            mu = [rng.uniform(mcfg["low"], mcfg["high"]) for _ in range(n)]
        else:
            mu = [wd * (1.0 + rng.uniform(mcfg["a"], mcfg["b"])) for wd in wdeg]
        bad = [m < wd * (1 - 1e-12) for m, wd in zip(mu, wdeg)]
        if not any(bad):
            return mu
        if mcfg["on_violation"] == "fix":
            return [wd if b else m for m, wd, b in zip(mu, wdeg, bad)]
        if mcfg["on_violation"] == "skip":
            return None
    return None


def _take(rng, cand, max_k):
    cand = list(cand)
    if max_k is not None and len(cand) > max_k:
        cand = rng.sample(cand, max_k)
    return sorted(cand)


def _B(rng, adj, boundary, bcfg):
    n, mode = len(adj), bcfg["mode"]
    deg = [len(a) for a in adj]
    if mode == "explicit":
        if max(bcfg["vertices"]) >= n:
            raise SpecError("generation_failed", f"graphs.B.vertices: vertex {max(bcfg['vertices'])} "
                                                 f"does not exist in a graph with {n} vertices")
        return list(bcfg["vertices"])
    if mode == "random_k":
        k = sample(rng, bcfg["k"], "int") if bcfg["k"] is not None else round(bcfg["frac"] * n)
        return _take(rng, rng.sample(range(n), int(min(max(k, 1), n))), bcfg["max_k"])
    if mode == "boundary":
        cand = [i for i in range(n) if boundary[i]] or list(range(n))
        return _take(rng, cand, bcfg["max_k"])
    if mode == "leaves":
        cand = [i for i in range(n) if deg[i] == 1]
        if not cand:
            low = min(deg)
            cand = [rng.choice([i for i in range(n) if deg[i] == low])]
        return _take(rng, cand, bcfg["max_k"])
    k = int(min(max(sample(rng, bcfg["k"], "int"), 1), n))
    tie = [rng.random() for _ in range(n)]
    order = sorted(range(n), key=lambda i: ((deg[i] if mode == "low_degree" else -deg[i]), tie[i]))
    return _take(rng, order[:k], bcfg["max_k"])


def _q(rng, adj, B, qcfg):
    n, mode = len(adj), qcfg["mode"]
    if mode == "zero":
        q = [0.0] * n
    elif mode == "constant":
        q = [qcfg["c"]] * n
    elif mode == "uniform":
        q = [rng.uniform(qcfg["low"], qcfg["high"]) for _ in range(n)]
    elif mode == "sparse":
        q = [0.0] * n
        for i in rng.sample(range(n), min(int(round(qcfg["fraction"] * n)), n)):
            q[i] = rng.uniform(qcfg["low"], qcfg["high"])
    elif mode == "localized":
        c = rng.randrange(n) if qcfg["center"] == "random" else qcfg["center"]
        if c >= n:
            raise SpecError("generation_failed", f"graphs.q.center: vertex {c} does not exist (n = {n})")
        q = [qcfg["amplitude"] * math.exp(-h / qcfg["decay"]) for h in hops(adj, c)]
    else:
        if len(qcfg["values"]) != n:
            raise SpecError("generation_failed", f"graphs.q.values has {len(qcfg['values'])} entries, graph has {n}")
        q = list(qcfg["values"])
    inB = set(B)
    if qcfg["on_B_only"]:
        q = [v if i in inB else 0.0 for i, v in enumerate(q)]
    if qcfg["off_B_only"]:
        q = [0.0 if i in inB else v for i, v in enumerate(q)]
    return [float(v) for v in q]


def resolve_dt(W, mu, q, dt):
    rate = max(sum(row) / m + qq for row, m, qq in zip(W, mu, q))
    if dt == "auto":
        return 1.0 if rate <= 1.0 else 1.0 / rate
    if dt * rate > 1.0 + 1e-12:
        raise SpecError("generation_failed", f"dt={dt} too large for this graph (need dt <= {1 / rate:.6g})")
    return float(dt)


def generate(gres: dict, seed: int) -> dict:
    """One graph: {family, params, W, mu, B, q, dt, boundary, attempts}."""
    rng = random.Random(seed)
    if gres["presets"]:
        name = gres["presets"][rng.randrange(len(gres["presets"]))]
        g = PRESETS[name]()
        q = _q(rng, g["adj"], g["B"], gres["q"])
        return {"family": name, "params": {}, "W": g["W"], "mu": g["mu"], "B": g["B"], "q": q,
                "dt": resolve_dt(g["W"], g["mu"], q, gres["dt"]), "boundary": g["boundary"], "attempts": 1}
    names = list(gres["families"])
    weights = [gres["families"][k]["weight"] for k in names]
    last = "no attempt"
    for attempt in range(1, gres["max_retries"] + 1):
        name = rng.choices(names, weights=weights)[0]
        e = gres["families"][name]
        params = ({"patterns": e["patterns"]} if name == "pattern" else
                  {k: sample(rng, e[k], kind) for k, (kind, _, _) in FAMILY_PARAMS[name].items()})
        adj, boundary, used = topology(rng, name, params)
        if len(adj) < 2 or not connected(adj):
            last = f"{name} graph was disconnected"
            continue
        W = _weights(rng, adj, gres["weights"])
        mu = _mu(rng, W, adj, gres["mu"], gres["max_retries"])
        if mu is None:
            last = "mu violated mu >= sum_y W"
            continue
        B = _B(rng, adj, boundary, gres["B"])
        q = _q(rng, adj, B, gres["q"])
        try:
            dt = resolve_dt(W, mu, q, gres["dt"])
        except SpecError as exc:
            last = str(exc)
            continue
        return {"family": name, "params": used, "W": W, "mu": mu, "B": B, "q": q, "dt": dt,
                "boundary": boundary, "attempts": attempt}
    raise SpecError("generation_failed", f"no valid graph after {gres['max_retries']} attempts ({last})")
