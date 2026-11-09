"""Example user simulator (plug-in via `simulator: {name: python, params: {path: ..., ...}}`).

Inverse problem: a 1-D rod with unknown piecewise-constant conductivity k(x) (n_segments values);
we observe the temperature at a few sensor points over time after an initial heat pulse
(explicit finite differences), with Gaussian sensor noise. Goal: recover k from the readings.

V2 additions (optional simulator-plugin hooks, see mlharness/simulators/__init__.py):
  forward(theta, **params)      noise-free observations for a batch of parameters (the forward operator)
  problem_info(**params)        parameter bounds, noise level and structural facts about the problem
``simulate`` is numerically identical to the V1 version (same operations in the same order).
"""
import numpy as np

TASK = "regression"
DESCRIPTION = ("1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant "
               "conductivity k (targets); features are noisy temperatures at sensor points over time.")


def _propagate(k_seg, n_grid, n_steps, sensors, every, k_high):
    n, n_segments = k_seg.shape
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
    return np.concatenate(obs, axis=1)


def simulate(n, seed, n_segments=4, n_grid=40, n_steps=200, sensors=(5, 15, 25, 35), every=10,
             noise=1e-3, k_low=0.5, k_high=2.0):
    rng = np.random.default_rng(seed)
    k_seg = rng.uniform(k_low, k_high, size=(n, n_segments))
    clean = _propagate(k_seg, n_grid, n_steps, sensors, every, k_high)
    X = clean + rng.normal(0, noise, size=clean.shape)
    return X, k_seg


def forward(theta, n_segments=4, n_grid=40, n_steps=200, sensors=(5, 15, 25, 35), every=10,
            noise=1e-3, k_low=0.5, k_high=2.0):
    """Forward operator F(k): noise-free sensor readings for conductivities ``theta`` (n, n_segments).

    The time step depends only on the simulator's ``k_high`` (not on theta), so F is exactly the
    operator that generated the data, for any theta (stable while theta <= k_high / 0.8).
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    return _propagate(theta, n_grid, n_steps, sensors, every, k_high)


def problem_info(n_segments=4, n_grid=40, n_steps=200, sensors=(5, 15, 25, 35), every=10,
                 noise=1e-3, k_low=0.5, k_high=2.0):
    n_times = len(range(0, n_steps, every))
    return {
        "param_names": [f"k{i + 1}" for i in range(n_segments)],
        "lower": [k_low] * n_segments,
        "upper": [k_high] * n_segments,
        "prior": f"i.i.d. uniform on [{k_low}, {k_high}] per segment",
        "noise_std": noise,
        "noise_model": "additive i.i.d. Gaussian on every sensor reading",
        "obs_layout": f"{n_times} times x {len(sensors)} sensors, time-major",
        "facts": {
            "dimension": 1,
            "pde": "parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x",
            "coefficient_regularity": "piecewise-constant (discontinuous) conductivity, one value per segment",
            "boundary_conditions": "homogeneous Dirichlet at both ends",
            "initial_condition": "known Gaussian pulse centred at x = 0.5",
            "observations": f"interior point sensors at grid indices {list(sensors)}, discrete times, finite window",
            "discretization": f"explicit finite differences, {n_grid} grid points, {n_steps} steps",
            "noise": f"Gaussian, sigma = {noise}",
            "parameter_count": n_segments,
            "positivity": "conductivity is strictly positive",
        },
    }
