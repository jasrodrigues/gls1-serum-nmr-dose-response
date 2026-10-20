import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gls1dr.config import Config
from gls1dr.model import equilibrium, jacobian, phi, propagate, verify_model

RNG = np.random.default_rng(0)


def test_equilibrium_satisfies_the_stationary_equations():
    g = RNG.uniform(0.05, 1, 2000)
    eta, xi = 10 ** RNG.uniform(-1, 1, 2000), 10 ** RNG.uniform(-1, 1, 2000)
    h, r = equilibrium(g, eta, xi)
    assert np.abs(g - h - eta * h * r).max() < 1e-12
    assert np.abs(1 - r - xi * h * r).max() < 1e-12
    assert np.all(h > 0) and np.all((r > 0) & (r <= 1))


def test_no_coupling_limit():
    g = np.array([0.3, 0.8]); eta = np.array([2.0, 0.5])
    h, r = equilibrium(g, eta, np.zeros(2))
    assert np.allclose(h, g / (1 + eta)) and np.allclose(r, 1.0)


def test_gsh_falls_and_ros_rise_with_dose_for_any_positive_parameters():
    D = np.linspace(0, 2, 200)
    for _ in range(200):
        ki, eta, xi = 10 ** RNG.uniform(-1, 1), 10 ** RNG.uniform(-1, 1), 10 ** RNG.uniform(-1, 1)
        g = phi(D, ki)
        h, r = equilibrium(g, np.full_like(g, eta), np.full_like(g, xi))
        assert np.all(np.diff(g) < 0) and np.all(np.diff(h) < 0) and np.all(np.diff(r) > 0)


def test_propagation_is_relative_to_the_untreated_state():
    ki = np.array([0.5, 2.0]); D = np.array([0.0, 0.14, 0.28])
    a, h, r = propagate(ki, np.array([1.0, 3.0]), np.array([1.0, 0.2]), D)
    assert np.allclose(a[:, 0], 1) and np.allclose(h[:, 0], 1) and np.allclose(r[:, 0], 1)
    assert np.all(a[:, 2] < a[:, 1]) and np.all(h[:, 2] < h[:, 1]) and np.all(r[:, 2] > r[:, 1])


def test_jacobian_eigenvalues_are_negative():
    for _ in range(2000):
        eta, xi, alpha, beta, delta, ki = (10 ** RNG.uniform(-1, 1) for _ in range(6))
        D = RNG.uniform(0, 1)
        h, r = equilibrium(np.array(phi(D, ki)), np.array(eta), np.array(xi))
        assert np.linalg.eigvals(jacobian(alpha, beta, delta, eta, xi, float(h), float(r))).real.max() < 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ode_integration_converges_to_the_closed_form(seed):
    rng = np.random.default_rng(seed)
    eta, xi, alpha, beta, delta = (10 ** rng.uniform(-1, 1) for _ in range(5))
    Phi = float(phi(rng.uniform(0, 1), 10 ** rng.uniform(-1, 1)))
    f = lambda t, y: [Phi - y[0], alpha * (y[0] - y[1]), beta * (y[0] - y[2] - eta * y[2] * y[3]),
                      delta * (1 - y[3] - xi * y[2] * y[3])]
    y = solve_ivp(f, [0, 800], rng.uniform(0.2, 2, 4), method="LSODA", rtol=1e-10, atol=1e-12).y[:, -1]
    h, r = equilibrium(np.array(Phi), np.array(eta), np.array(xi))
    assert np.allclose(y, [Phi, Phi, float(h), float(r)], atol=1e-8)


def test_verify_model_report():
    cfg = Config(n_verify=500, n_ode_checks=10)
    out = verify_model(np.array([0.3, 0.9, 1.5, 4.0]), cfg)
    assert out["equilibrium_max_residual"] < 1e-12
    assert all(out["monotonic"].values()) and out["stability"]["all_negative"]
    assert out["ode_vs_analytic_max_abs_err"] < 1e-7
