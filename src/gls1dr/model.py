"""Reduced mechanistic model of the GSH-ROS response to a dose-dependent metabolic input.

Dimensionless system (Supplementary Methods), with g the glutamate-associated state,
a the alpha-KG-associated state, h reduced GSH and r reactive oxygen species:

    dg/dt = Phi(D) - g
    da/dt = alpha (g - a)
    dh/dt = beta (g - h - eta h r)
    dr/dt = delta (1 - r - xi h r)

Phi(D) = 1 / (1 + D / K_I). At steady state g* = a* = Phi(D), r* = 1 / (1 + xi h*) and h* is the
positive root of  xi h^2 + (1 + eta - xi g*) h - g* = 0  (Eq. 3 of the Methods).
"""
import numpy as np
from scipy.integrate import solve_ivp


def phi(D, ki):
    return 1.0 / (1.0 + D / ki)


def equilibrium(g, eta, xi):
    """Stationary GSH (h*) and ROS (r*) for a given metabolic input g* = Phi(D)."""
    b = 1.0 + eta - xi * g
    h = 2.0 * g / (b + np.sqrt(b ** 2 + 4.0 * xi * g))
    r = 1.0 / (1.0 + xi * h)
    return h, r


def propagate(ki, eta, xi, D):
    """Stationary a*, h*, r* for every parameter draw (rows) and dose (columns), relative to D = 0."""
    g = 1.0 / (1.0 + D[None, :] / ki[:, None])
    h, r = equilibrium(g, eta[:, None] * np.ones_like(g), xi[:, None] * np.ones_like(g))
    g0 = np.ones((len(ki), 1))
    h0, r0 = equilibrium(g0, eta[:, None], xi[:, None])
    return g / g0, h / h0, r / r0


def jacobian(alpha, beta, delta, eta, xi, h, r):
    """Jacobian of the dimensionless system at the stationary state (Supplementary Methods)."""
    return np.array([[-1, 0, 0, 0],
                     [alpha, -alpha, 0, 0],
                     [beta, 0, -beta * (1 + eta * r), -beta * eta * h],
                     [0, 0, -delta * xi * r, -delta * (1 + xi * h)]])


def verify_model(ki_samples, cfg):
    """Numerical checks: residuals of the stationary equations, monotonicity in dose, local stability
    (eigenvalues of the Jacobian) and agreement between ODE integration and the closed form."""
    rng = np.random.default_rng(cfg.seed_verify)
    M = cfg.n_verify
    lo, hi = cfg.eta_xi_log10_range
    klo, khi = cfg.kinetic_log10_range
    ki = rng.choice(ki_samples, size=M, replace=True)
    eta, xi = (10 ** rng.uniform(lo, hi, M) for _ in range(2))
    alpha, beta, delta = (10 ** rng.uniform(klo, khi, M) for _ in range(3))
    Dv = rng.uniform(0, cfg.mc_dose_max, M)
    g = phi(Dv, ki)
    h, r = equilibrium(g, eta, xi)
    out = {"equilibrium_max_residual": float(max(np.abs(g - h - eta * h * r).max(), np.abs(1 - r - xi * h * r).max()))}
    g2 = phi(Dv + 1e-5, ki)
    h2, r2 = equilibrium(g2, eta, xi)
    out["monotonic"] = {"a_decreasing": bool(np.all(g2 < g)), "h_decreasing": bool(np.all(h2 < h)),
                        "r_increasing": bool(np.all(r2 > r))}
    maxre = np.empty(M)
    for i in range(M):
        J = jacobian(alpha[i], beta[i], delta[i], eta[i], xi[i], h[i], r[i])
        maxre[i] = np.linalg.eigvals(J).real.max()
    out["stability"] = {"max_real_part": float(maxre.max()), "all_negative": bool(np.all(maxre < 0)), "n": int(M)}
    errs = []
    for i in range(min(cfg.n_ode_checks, M)):
        Phi = phi(Dv[i], ki[i])
        f = lambda t, y: [Phi - y[0], alpha[i] * (y[0] - y[1]),
                          beta[i] * (y[0] - y[2] - eta[i] * y[2] * y[3]),
                          delta[i] * (1 - y[3] - xi[i] * y[2] * y[3])]
        sol = solve_ivp(f, [0, 800], rng.uniform(0.2, 2.0, 4), method="LSODA", rtol=1e-10, atol=1e-12)
        yf = sol.y[:, -1]
        errs.append(max(abs(yf[0] - g[i]), abs(yf[1] - g[i]), abs(yf[2] - h[i]), abs(yf[3] - r[i])))
    out["ode_vs_analytic_max_abs_err"] = float(max(errs))
    return out
