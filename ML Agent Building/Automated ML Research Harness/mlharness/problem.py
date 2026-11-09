"""Controlled access to the problem's forward operator and structure (V2).

``ProblemContext`` is the *only* channel through which a method (built-in or agent-written) may
touch the simulator. It exposes:

* ``forward(theta)``  noise-free observations for a batch of parameters, only when the research
  spec authorises it (``constraints.allow_forward_model``) and the simulator provides one. Every
  call is counted against ``max_forward_evals`` (resource control).
* bounds / prior / noise level / parameter names / structural facts from ``problem_info``.

It never exposes the dataset, file paths, or the simulator module itself to method code. The context
is picklable: it drops the live simulator and rebuilds it lazily from the simulator config, so an
exported physics/hybrid model can be re-loaded in a fresh process.

``diagnostics`` computes cheap numerical identifiability evidence (Jacobian sensitivities, Fisher
condition numbers, Cramér–Rao floor at sampled parameters) that the research critic can use as
empirical evidence instead of verbal argument.
"""
from __future__ import annotations

import threading

import numpy as np


class ForwardAccessDenied(PermissionError):
    pass


class ForwardBudgetExceeded(RuntimeError):
    pass


class ProblemContext:
    def __init__(self, simulator_cfg: dict, authorized: bool = True, max_forward_evals: int | None = None):
        self.simulator_cfg = dict(simulator_cfg or {})
        self.authorized = bool(authorized)
        self.max_forward_evals = int(max_forward_evals) if max_forward_evals else None
        self.forward_evals = 0
        self._lock = threading.Lock()
        self._sim = None
        self._fwd = None
        self._load()

    # -- construction / pickling ------------------------------------------------------------
    fallback_path: str | None = None      # copy of the simulator file shipped with an exported model

    def _load(self):
        from .simulators import get_simulator
        try:
            self._sim = get_simulator(self.simulator_cfg)
        except (FileNotFoundError, OSError, ValueError):
            if not self.fallback_path:
                raise
            cfg = {**self.simulator_cfg, "params": {**(self.simulator_cfg.get("params") or {}),
                                                    "path": self.fallback_path}}
            self._sim = get_simulator(cfg)
        info = self._sim.problem_info() or {}
        self._fwd = self._sim.forward_model() if self.authorized else None
        self.info = info
        self.param_names = list(info.get("param_names") or [])
        self.lower = np.asarray(info["lower"], dtype=float) if info.get("lower") is not None else None
        self.upper = np.asarray(info["upper"], dtype=float) if info.get("upper") is not None else None
        self.noise_std = float(info["noise_std"]) if info.get("noise_std") is not None else None
        self.facts = dict(info.get("facts") or {})
        self.simulator_has_forward = self._sim.forward_model() is not None

    def __getstate__(self):
        st = {k: v for k, v in self.__dict__.items() if k not in ("_sim", "_fwd", "_lock")}
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._lock = threading.Lock()
        self._sim = self._fwd = None
        try:
            self._load()
        except Exception:      # simulator file moved: metadata stays usable, forward becomes unavailable
            self._fwd = None

    # -- public API (what method code may use) ----------------------------------------------
    @property
    def forward_available(self) -> bool:
        return self._fwd is not None

    @property
    def n_params(self) -> int | None:
        return len(self.param_names) if self.param_names else (len(self.lower) if self.lower is not None else None)

    def forward(self, theta) -> np.ndarray:
        if self._fwd is None:
            raise ForwardAccessDenied("forward model access is not authorised for this method/spec "
                                      "or the simulator does not provide one")
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        with self._lock:
            if self.max_forward_evals is not None and self.forward_evals + len(theta) > self.max_forward_evals:
                raise ForwardBudgetExceeded(f"forward-evaluation budget of {self.max_forward_evals} exceeded")
            self.forward_evals += len(theta)
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError("non-finite parameters passed to the forward model")
        return self._fwd(theta)

    def clip(self, theta):
        if self.lower is None or self.upper is None:
            return theta
        return np.clip(theta, self.lower, self.upper)

    def prior_center(self):
        if self.lower is None or self.upper is None:
            return None
        return 0.5 * (self.lower + self.upper)

    def describe(self) -> str:
        bits = [f"forward model: {'AVAILABLE via context.forward(theta)' if self.forward_available else 'not available'}"]
        if self.param_names:
            bits.append(f"parameters: {self.param_names}")
        if self.lower is not None:
            bits.append(f"bounds: lower={self.lower.tolist()} upper={self.upper.tolist()}")
        if self.noise_std is not None:
            bits.append(f"noise std: {self.noise_std} ({self.info.get('noise_model', 'unspecified model')})")
        if self.info.get("prior"):
            bits.append(f"prior: {self.info['prior']}")
        if self.facts:
            bits.append("facts: " + "; ".join(f"{k}={v}" for k, v in self.facts.items()))
        return " | ".join(bits)

    def public_summary(self) -> dict:
        return {"forward_available": self.forward_available, "simulator_has_forward": self.simulator_has_forward,
                "param_names": self.param_names, "lower": None if self.lower is None else self.lower.tolist(),
                "upper": None if self.upper is None else self.upper.tolist(), "noise_std": self.noise_std,
                "facts": self.facts, "prior": self.info.get("prior"), "noise_model": self.info.get("noise_model")}


def fd_jacobian(fwd, theta: np.ndarray, h: float = 1e-5, transform: str = "none") -> np.ndarray:
    """Central-difference Jacobian of fwd at theta (n, p) -> (n, m, p), one batched forward call.

    transform='log' differentiates w.r.t. log(theta) (i.e. returns dF/dlog theta)."""
    n, p = theta.shape
    z = np.log(theta) if transform == "log" else theta
    stack = []
    for j in range(p):
        e = np.zeros(p)
        e[j] = h
        stack += [z + e, z - e]
    Z = np.concatenate(stack, axis=0)
    F = fwd(np.exp(Z) if transform == "log" else Z)
    F = F.reshape(2 * p, n, -1)
    return np.stack([(F[2 * j] - F[2 * j + 1]) / (2 * h) for j in range(p)], axis=2)


def diagnostics(ctx: ProblemContext, n_samples: int = 24, seed: int = 0) -> dict:
    """Local identifiability at parameters drawn from the prior box (cheap: one Jacobian batch).

    Returns per-parameter Cramér–Rao standard deviations (median over samples), relative CRB,
    median Fisher condition number and a ranking of the least/best determined parameters."""
    if not ctx.forward_available or ctx.lower is None or ctx.upper is None:
        return {"available": False, "reason": "needs an authorised forward model and parameter bounds"}
    rng = np.random.default_rng(seed)
    theta = rng.uniform(ctx.lower, ctx.upper, size=(n_samples, len(ctx.lower)))
    try:
        J = fd_jacobian(ctx.forward, theta)
    except Exception as exc:
        return {"available": False, "reason": f"jacobian failed: {type(exc).__name__}: {exc}"}
    sigma = ctx.noise_std or 1.0
    fisher = np.einsum("nmi,nmj->nij", J, J) / sigma ** 2
    cond = np.linalg.cond(fisher)
    try:
        crb = np.linalg.inv(fisher)
        std = np.sqrt(np.clip(np.diagonal(crb, axis1=1, axis2=2), 0, None))
    except np.linalg.LinAlgError:
        std = np.full_like(theta, np.inf)
    rel = std / np.abs(theta)
    names = ctx.param_names or [f"p{i}" for i in range(theta.shape[1])]
    med_rel = np.median(rel, axis=0)
    order = np.argsort(-med_rel)
    rank = np.linalg.matrix_rank(J[0])
    return {"available": True, "n_samples": n_samples, "noise_std": ctx.noise_std,
            "jacobian_rank_at_sample0": int(rank), "n_params": int(theta.shape[1]),
            "fisher_condition_median": float(np.median(cond)), "fisher_condition_max": float(np.max(cond)),
            "crb_std_median": {names[i]: float(np.median(std[:, i])) for i in range(len(names))},
            "crb_rel_median": {names[i]: float(med_rel[i]) for i in range(len(names))},
            "crb_rel_l2_estimate": float(np.sqrt((std ** 2).sum()) / np.linalg.norm(theta)),
            "least_determined": [names[i] for i in order[:2]],
            "best_determined": [names[i] for i in order[::-1][:2]],
            "sensitivity_norm_median": {names[i]: float(np.median(np.linalg.norm(J[:, :, i], axis=1)))
                                        for i in range(len(names))}}
