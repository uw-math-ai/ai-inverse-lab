"""Example user simulator (plug-in via `simulator: {name: python, params: {path: ..., ...}}`).

Inverse problem: a 1-D rod with unknown piecewise-constant conductivity k(x) (n_segments values);
we observe the temperature at a few sensor points over time after an initial heat pulse
(explicit finite differences), with Gaussian sensor noise. Goal: recover k from the readings.
"""
import numpy as np

TASK = "regression"
DESCRIPTION = ("1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant "
               "conductivity k (targets); features are noisy temperatures at sensor points over time.")


def simulate(n, seed, n_segments=4, n_grid=40, n_steps=200, sensors=(5, 15, 25, 35), every=10,
             noise=1e-3, k_low=0.5, k_high=2.0):
    rng = np.random.default_rng(seed)
    k_seg = rng.uniform(k_low, k_high, size=(n, n_segments))
    k = np.repeat(k_seg, int(np.ceil(n_grid / n_segments)), axis=1)[:, :n_grid]
    dx = 1.0 / n_grid
    dt = 0.4 * dx * dx / k_high
    x = np.linspace(0, 1, n_grid)
    u = np.exp(-((x - 0.5) ** 2) / 0.005)[None, :].repeat(n, axis=0)
    kf = 0.5 * (k[:, 1:] + k[:, :-1])                     # face conductivities
    obs = []
    for t in range(n_steps):
        flux = kf * (u[:, 1:] - u[:, :-1]) / dx
        u[:, 1:-1] += dt * (flux[:, 1:] - flux[:, :-1]) / dx
        u[:, 0] = u[:, -1] = 0.0
        if t % every == 0:
            obs.append(u[:, list(sensors)].copy())
    X = np.concatenate(obs, axis=1) + rng.normal(0, noise, size=(n, len(obs) * len(sensors)))
    return X, k_seg
