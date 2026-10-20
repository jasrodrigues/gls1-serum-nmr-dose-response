"""Dose-response calibration: hyperbolic, linear and batch-only models (Eqs. 2a-2b), AICc,
profile likelihood and a stratified bootstrap."""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def ki_grid(cfg):
    return np.exp(np.linspace(np.log(cfg.ki_min), np.log(cfg.ki_max), cfg.ki_grid_size))


def rss_profile(ki, d, b, s):
    """RSS with K_I fixed. The scale c and the batch effect gamma follow from linear least squares."""
    f = 1.0 / (1.0 + d / ki)
    A = np.column_stack([f, b])
    beta, *_ = np.linalg.lstsq(A, s, rcond=None)
    r = s - A @ beta
    return float(r @ r), beta


def fit_hyper(d, b, s, grid):
    """S_i = c / (1 + D_i / K_I) + gamma B_i + e_i. One-dimensional search over K_I."""
    rs = np.array([rss_profile(k, d, b, s)[0] for k in grid])
    i = int(np.argmin(rs))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda lk: rss_profile(np.exp(lk), d, b, s)[0],
                          bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 1e-8})
    ki = float(np.exp(res.x))
    r, beta = rss_profile(ki, d, b, s)
    return dict(KI=ki, c=float(beta[0]), gamma=float(beta[1]), rss=r)


def fit_linear(d, b, s):
    A = np.column_stack([np.ones_like(d), d, b])
    beta, *_ = np.linalg.lstsq(A, s, rcond=None)
    r = s - A @ beta
    return dict(c=float(beta[0]), slope=float(beta[1]), gamma=float(beta[2]), rss=float(r @ r))


def fit_null(b, s):
    A = np.column_stack([np.ones_like(b), b])
    beta, *_ = np.linalg.lstsq(A, s, rcond=None)
    r = s - A @ beta
    return dict(c=float(beta[0]), gamma=float(beta[1]), rss=float(r @ r))


def aicc(rss, k_reg, n):
    """Corrected Akaike criterion; the residual variance is counted as one extra parameter."""
    k = k_reg + 1
    return float(n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


def compare_models(d, b, s, grid):
    n = len(s)
    h, l, z = fit_hyper(d, b, s, grid), fit_linear(d, b, s), fit_null(b, s)
    h["aicc"], l["aicc"], z["aicc"] = aicc(h["rss"], 3, n), aicc(l["rss"], 3, n), aicc(z["rss"], 2, n)
    return h, l, z


def profile_interval(d, b, s, grid, hyper):
    """Profile-likelihood limits for K_I.

    Inside the interval: n ln(RSS_p(K_I) / RSS_p(K_hat)) <= chi2(1, 0.95). An F-based
    criterion with n - 3 degrees of freedom is returned as a small-sample check.
    """
    n = len(s)
    pr = np.array([rss_profile(k, d, b, s)[0] for k in grid])
    lr = n * np.log(np.maximum(pr, 1e-300) / max(hyper["rss"], 1e-300))
    chi2_95, f95 = stats.chi2.ppf(0.95, 1), stats.f.ppf(0.95, 1, n - 3)
    inside = grid[lr <= chi2_95]
    inside_f = grid[(pr - hyper["rss"]) / (hyper["rss"] / (n - 3)) <= f95]
    return dict(lr_lo=float(inside.min()), lr_unbounded_above=bool(inside.max() >= grid[-5]),
                f_lo=float(inside_f.min()), lr_at_null=float(lr[-1]), chi2_95=float(chi2_95)), np.column_stack([grid, pr, lr])


def bootstrap_indices(group, batch, cfg):
    """Resample indices stratified by treatment group and batch (same resamples for every score)."""
    rng = np.random.default_rng(cfg.seed_boot)
    cells = []
    for g in cfg.dose_map:
        for bt in sorted(set(batch)):
            ix = np.where((group == g) & (batch == bt))[0]
            if len(ix):
                cells.append(ix)
    return np.array([np.concatenate([rng.choice(v, size=len(v), replace=True) for v in cells])
                     for _ in range(cfg.n_boot)])


def bootstrap_ki(d, b, s, indices, grid):
    return np.array([fit_hyper(d[ix], b[ix], s[ix], grid)["KI"] for ix in indices])
