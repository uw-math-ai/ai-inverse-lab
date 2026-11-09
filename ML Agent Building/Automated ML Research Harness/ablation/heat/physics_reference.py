"""Physics-based reference for the heat-conductivity inverse problem (not a harness arm).

1. Exact forward model F(k) re-implemented from examples/heat_diffusion_simulator.py (verified equal).
2. Batched Levenberg-Marquardt in log-k on every test sample (maximum-likelihood under Gaussian noise),
   multi-start, bounds = the simulator prior [0.5, 2.0].
3. Cramer-Rao bound at the true k: Cov >= sigma^2 (J^T J)^-1  ->  noise floor for rel_l2 per segment.

Usage: python ablation/heat/physics_reference.py <dataset.npz>  -> writes physics_reference.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = dict(n_segments=4, n_grid=40, n_steps=200, sensors=(5, 15, 25, 35), every=10, noise=1e-3,
         k_low=0.5, k_high=2.0)


def forward(k_seg):
    """Noise-free observations for a batch of segment conductivities (n, 4). Mirrors simulate()."""
    n, ns = k_seg.shape
    g = P["n_grid"]
    k = np.repeat(k_seg, int(np.ceil(g / ns)), axis=1)[:, :g]
    dx = 1.0 / g
    dt = 0.4 * dx * dx / P["k_high"]
    x = np.linspace(0, 1, g)
    u = np.exp(-((x - 0.5) ** 2) / 0.005)[None, :].repeat(n, axis=0)
    kf = 0.5 * (k[:, 1:] + k[:, :-1])
    obs = []
    for t in range(P["n_steps"]):
        flux = kf * (u[:, 1:] - u[:, :-1]) / dx
        u[:, 1:-1] += dt * (flux[:, 1:] - flux[:, :-1]) / dx
        u[:, 0] = u[:, -1] = 0.0
        if t % P["every"] == 0:
            obs.append(u[:, list(P["sensors"])].copy())
    return np.concatenate(obs, axis=1)


def jac(theta, h=1e-5):
    """Jacobian of F(exp(theta)) w.r.t. theta = log k, central differences. (n, m, 4)"""
    cols = []
    for j in range(theta.shape[1]):
        e = np.zeros_like(theta); e[:, j] = h
        cols.append((forward(np.exp(theta + e)) - forward(np.exp(theta - e))) / (2 * h))
    return np.stack(cols, axis=2)


def lm(X, theta0, iters=60, lo=np.log(P["k_low"]), hi=np.log(P["k_high"])):
    th = theta0.copy()
    lam = np.full(len(X), 1e-2)
    r = forward(np.exp(th)) - X
    cost = (r ** 2).sum(1)
    for _ in range(iters):
        J = jac(th)
        JtJ = np.einsum("nmi,nmj->nij", J, J)
        g = np.einsum("nmi,nm->ni", J, r)
        A = JtJ + lam[:, None, None] * (np.eye(4)[None] * np.maximum(np.diagonal(JtJ, axis1=1, axis2=2), 1e-12)[:, None, :])
        step = -np.linalg.solve(A, g[..., None])[..., 0]
        cand = np.clip(th + step, lo, hi)
        rc = forward(np.exp(cand)) - X
        cc = (rc ** 2).sum(1)
        ok = cc < cost
        th[ok], r[ok], cost[ok] = cand[ok], rc[ok], cc[ok]
        lam = np.where(ok, lam / 3, lam * 4).clip(1e-9, 1e9)
    return th, cost


def rel_l2(Y, Yh):
    return float(np.linalg.norm(Yh - Y) / np.linalg.norm(Y))


def main(data_path):
    spec = importlib.util.spec_from_file_location("heat", ROOT / "examples" / "heat_diffusion_simulator.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    Xc, kc = mod.simulate(64, 123, noise=0.0)
    assert np.allclose(forward(kc), Xc, atol=1e-14), "forward model mismatch"

    z = np.load(data_path)
    X, Y = z["X_test"], z["y_test"]
    sigma = P["noise"]
    # multi-start LM: prior mean + 4 random starts, keep the lowest residual per sample
    rng = np.random.default_rng(0)
    starts = [np.full_like(Y, np.log(1.25))] + [np.log(rng.uniform(0.5, 2.0, Y.shape)) for _ in range(4)]
    best_th, best_c = None, None
    for s in starts:
        th, c = lm(X, s)
        if best_th is None:
            best_th, best_c = th, c
        else:
            m = c < best_c
            best_th[m], best_c[m] = th[m], c[m]
    Yh = np.exp(best_th)
    # Cramer-Rao bound in k-space at the true parameters
    Jk = jac(np.log(Y)) / Y[:, None, :]                       # dF/dk = dF/dtheta / k
    fisher = np.einsum("nmi,nmj->nij", Jk, Jk) / sigma ** 2
    crb = np.linalg.inv(fisher)
    crb_var = np.diagonal(crb, axis1=1, axis2=2)              # (n, 4)
    crb_rel_l2 = float(np.sqrt(crb_var.sum()) / np.linalg.norm(Y))
    cond = np.linalg.cond(fisher)
    out = {
        "n_test": int(len(Y)),
        "mle_test_rel_l2": rel_l2(Y, Yh),
        "mle_test_rel_l2_per_segment": [rel_l2(Y[:, j], Yh[:, j]) for j in range(4)],
        "mle_mean_chi2_per_dof": float(best_c.mean() / sigma ** 2 / X.shape[1]),
        "crb_rel_l2": crb_rel_l2,
        "crb_rel_l2_per_segment": [float(np.sqrt(crb_var[:, j].sum()) / np.linalg.norm(Y[:, j])) for j in range(4)],
        "crb_std_median_per_segment": np.median(np.sqrt(crb_var), axis=0).round(5).tolist(),
        "fisher_condition_median": float(np.median(cond)),
        "mean_predictor_rel_l2": rel_l2(Y, np.broadcast_to(z["y_train"].mean(0), Y.shape)),
    }
    (HERE / "physics_reference.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
