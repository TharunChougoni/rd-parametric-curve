"""Recover theta, M, X for the R&D parametric curve.

The brief's generator is:

    x(t) = t * cos(theta) - exp(M * |t|) * sin(0.3 * t) * sin(theta) + X
    y(t) = 42 + t * sin(theta) + exp(M * |t|) * sin(0.3 * t) * cos(theta)

Unknowns and bounds (from the assignment):
    theta in (0, 50) degrees,   used in radians inside cos/sin
    M     in (-0.05, 0.05)
    X     in (0, 100)
    t     in (6, 60)            dummy along-curve parameter, not submitted

Treat this as a bound-constrained nonlinear program: search
z = (theta_deg, M, X) to minimise mean L1 from the 1500 samples to the
curve that z generates. Several SciPy solvers share the same objective;
keep the lowest L1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import (
    basinhopping,
    differential_evolution,
    dual_annealing,
    minimize,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "xy_data_3.csv"

# Open intervals from the brief, closed a hair inside so iterates stay valid.
T_LO, T_HI = 6.0, 60.0
BOUNDS = [(0.05, 49.95), (-0.0499, 0.0499), (0.05, 99.95)]


def load_xy(path: Path = DATA) -> np.ndarray:
    """Load the CSV as an (N, 2) array of (x, y). Header row skipped."""
    return np.loadtxt(path, delimiter=",", skiprows=1)


def curve(t: np.ndarray, theta_rad: float, M: float, X: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the parametric curve at t.

    Same as a line through (X, 42) with heading theta, plus a sideways
    weave of amplitude exp(M |t|) * sin(0.3 t).
    """
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    # Sideways displacement along n = (-sin, cos).
    w = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * ct - w * st + X
    y = 42.0 + t * st + w * ct
    return x, y


def mean_l1(params: np.ndarray, pts: np.ndarray, t_grid: np.ndarray) -> float:
    """Chamfer L1: each data point to the nearest uniform-t sample.

    Used as a check against the assignment wording ("uniformly sampled
    points"). Too slow / slightly grid-biased for the inner solver loop.
    """
    th_deg, M, X = params
    xs, ys = curve(t_grid, np.deg2rad(th_deg), M, X)
    dx = pts[:, 0][:, None] - xs[None, :]
    dy = pts[:, 1][:, None] - ys[None, :]
    return float(np.mean(np.min(np.abs(dx) + np.abs(dy), axis=1)))


def mean_l1_project(params: np.ndarray, pts: np.ndarray) -> float:
    """O(n) mean L1 using the closed-form t of each point.

    The weave is perpendicular to the heading u = (cos theta, sin theta),
    so the along-track coordinate of point p is just

        t = (p - (X, 42)) · u

    clipped to [6, 60]. Residual is then |p - curve(t)|_1.
    This is what DE / annealing / hopping / Nelder-Mead / L-BFGS-B minimise.
    """
    th_deg, M, X = params
    th = np.deg2rad(th_deg)
    ct, st = np.cos(th), np.sin(th)
    t = (pts[:, 0] - X) * ct + (pts[:, 1] - 42.0) * st
    t = np.clip(t, T_LO, T_HI)
    xs, ys = curve(t, th, M, X)
    return float(np.mean(np.abs(pts[:, 0] - xs) + np.abs(pts[:, 1] - ys)))


def nearest_t(params: np.ndarray, pts: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """t on a dense grid closest in L1 to each data point. Scoring only."""
    th_deg, M, X = params
    xs, ys = curve(t_grid, np.deg2rad(th_deg), M, X)
    d = np.abs(pts[:, 0][:, None] - xs[None, :]) + np.abs(pts[:, 1][:, None] - ys[None, :])
    return t_grid[np.argmin(d, axis=1)]


def geometric_seed(pts: np.ndarray) -> np.ndarray:
    """Cheap start: ignore the weave, fit a line through (X, 42).

    Heading from the first SVD/PCA axis of the cloud. X from the median
    intercept after backing out t from (y - 42) / sin(theta). M starts at 0.
    """
    c = pts.mean(axis=0)
    a = np.column_stack([pts[:, 0] - c[0], pts[:, 1] - c[1]])
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    vx, vy = vt[0]
    # Flip so the heading points to +x (the cloud runs left → right).
    if vx < 0:
        vx, vy = -vx, -vy
    th = float(np.rad2deg(np.arctan2(vy, vx)))
    th = float(np.clip(th, 0.05, 49.95))
    u = np.array([np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))])
    st = u[1]
    if abs(st) < 1e-6:
        X = float(np.median(pts[:, 0]))
    else:
        t_est = (pts[:, 1] - 42.0) / st
        X = float(np.median(pts[:, 0] - t_est * u[0]))
    X = float(np.clip(X, 0.05, 99.95))
    return np.array([th, 0.0, X], dtype=float)


def run_solvers(pts: np.ndarray) -> dict:
    """Run independent searches on the same L1 objective; keep the best.

    Order: geometric seed → coarse grid → differential evolution →
    dual annealing → basin hopping → Nelder-Mead and L-BFGS-B polish
    from each of those starts. SciPy implements the named algorithms;
    this function only calls them and records (theta, M, X, L1).
    """
    t_grid = np.linspace(T_LO, T_HI, 801)
    seed = geometric_seed(pts)
    print(f"geometric seed theta={seed[0]:.4f} deg  M={seed[1]:.5f}  X={seed[2]:.4f}  "
          f"L1_proj={mean_l1_project(seed, pts):.6f}  "
          f"L1_chamfer={mean_l1(seed, pts, t_grid):.6f}", flush=True)

    def f(p):
        # Inner objective: closed-form-t L1. Shape (3,) → scalar.
        return mean_l1_project(p, pts)

    results = []

    def record(name, p, nit=None):
        p = np.asarray(p, dtype=float)
        l1 = f(p)
        row = {"method": name, "theta_deg": float(p[0]), "M": float(p[1]),
               "X": float(p[2]), "L1_proj": l1, "nit": nit}
        results.append(row)
        print(f"{name:22s}  θ={p[0]:8.5f} deg  M={p[1]:+.6f}  X={p[2]:8.4f}  L1={l1:.8f}",
              flush=True)
        return row

    record("geometric", seed)

    # Coarse box around the seed. Confirms the basin is not a pinprick
    # before spending budget on global methods.
    best_g, best_v = seed.copy(), f(seed)
    ths = np.linspace(max(0.05, seed[0] - 8), min(49.95, seed[0] + 8), 25)
    Ms = np.linspace(-0.0499, 0.0499, 21)
    Xs = np.linspace(max(0.05, seed[2] - 15), min(99.95, seed[2] + 15), 25)
    for th in ths:
        for M in Ms:
            for X in Xs:
                p = np.array([th, M, X])
                v = f(p)
                if v < best_v:
                    best_v, best_g = v, p
    record("grid", best_g)

    # Population search. Mutation/crossover in scipy; we only pass f and bounds.
    de = differential_evolution(
        f, BOUNDS, seed=1, popsize=15, mutation=0.7, recombination=0.9,
        polish=True, workers=1, updating="deferred",
    )
    record("diff_evol", de.x, de.nit)

    da = dual_annealing(f, BOUNDS, seed=1, maxiter=400)
    record("dual_anneal", da.x, da.nit)

    rng = np.random.default_rng(1)

    def bh_step(x):
        # Custom hop sizes: theta in degrees, M is O(0.01), X is O(10).
        x = np.array(x, dtype=float)
        x[0] += rng.normal(0, 1.5)
        x[1] += rng.normal(0, 0.008)
        x[2] += rng.normal(0, 2.0)
        lo = np.array([b[0] for b in BOUNDS])
        hi = np.array([b[1] for b in BOUNDS])
        return np.clip(x, lo, hi)

    bh = basinhopping(
        f, seed, niter=80, stepsize=1.0, seed=1,
        minimizer_kwargs={"method": "Nelder-Mead", "options": {"maxiter": 250}},
        take_step=bh_step,
    )
    record("basinhopping", bh.x)

    # Local polish from each distinct-ish start so a lucky global cannot
    # hide a better nearby point.
    starts = [seed, best_g, de.x, da.x, bh.x]
    for i, s in enumerate(starts):
        loc = minimize(f, s, method="Nelder-Mead",
                       options={"maxiter": 800, "xatol": 1e-10, "fatol": 1e-12})
        record(f"nelder_from_{i}", loc.x, loc.nit)

    for i, s in enumerate(starts):
        loc = minimize(f, s, method="L-BFGS-B", bounds=BOUNDS)
        record(f"lbfgs_from_{i}", loc.x)

    best = min(results, key=lambda r: r["L1_proj"])
    p = np.array([best["theta_deg"], best["M"], best["X"]])
    t_hi = np.linspace(T_LO, T_HI, 4001)
    best["L1_chamfer"] = mean_l1(p, pts, t_hi)
    best["L1_proj_final"] = mean_l1_project(p, pts)
    best["theta_rad"] = float(np.deg2rad(best["theta_deg"]))
    return {"all": results, "best": best, "params": p.tolist()}


def desmos_string(theta_rad: float, M: float, X: float) -> str:
    """Assignment paste format (radians in the trig calls)."""
    return (
        rf"\left(t*\cos({theta_rad:.6f})-e^{{{M:.6f}\left|t\right|}}"
        rf"\cdot\sin(0.3t)\sin({theta_rad:.6f})+{X:.6f},"
        rf"42+t*\sin({theta_rad:.6f})+e^{{{M:.6f}\left|t\right|}}"
        rf"\cdot\sin(0.3t)\cos({theta_rad:.6f})\right)"
    )


def save_plot(pts: np.ndarray, params: np.ndarray, path: Path) -> None:
    """Overlay data vs fitted curve. Agg backend so this is headless-safe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(T_LO, T_HI, 2000)
    xs, ys = curve(t, np.deg2rad(params[0]), params[1], params[2])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.35, label="data", c="#4c78a8")
    ax.plot(xs, ys, color="#e45756", lw=1.4, label="fit")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"θ={params[0]:.4f}°  M={params[1]:+.5f}  X={params[2]:.4f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    pts = load_xy()
    print(f"n={len(pts)}  x=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]  "
          f"y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]", flush=True)
    out = run_solvers(pts)
    best = out["best"]
    p = np.array(out["params"])
    t_grid = np.linspace(T_LO, T_HI, 4001)
    ts = nearest_t(p, pts, t_grid)
    xs, ys = curve(ts, np.deg2rad(p[0]), p[1], p[2])
    resid = np.abs(pts[:, 0] - xs) + np.abs(pts[:, 1] - ys)
    best["L1_assigned_t"] = float(resid.mean())
    best["L1_p95"] = float(np.percentile(resid, 95))
    best["L1_max"] = float(resid.max())
    best["desmos"] = desmos_string(best["theta_rad"], best["M"], best["X"])
    print("\nBEST", json.dumps(best, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "fit.json").write_text(json.dumps(out, indent=2))
    save_plot(pts, p, ROOT / "results" / "fit.png")
    print("wrote results/fit.json results/fit.png", flush=True)


if __name__ == "__main__":
    main()
