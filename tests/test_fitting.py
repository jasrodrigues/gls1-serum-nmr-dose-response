import numpy as np
import pytest

from gls1dr.config import Config
from gls1dr.fitting import (aicc, bootstrap_indices, bootstrap_ki, compare_models, fit_hyper, fit_linear, fit_null,
                            ki_grid, profile_interval, rss_profile)
from gls1dr.model import phi

DOSE = np.array([0, 0, 0, 0.14, 0.14, 0.14, 0.14, 0, 0, 0, 0.14, 0.14, 0.28, 0.28, 0.28, 0.28])
B = np.array([0] * 7 + [1] * 9, float)
GRID = ki_grid(Config())


def test_grid_covers_the_search_interval():
    cfg = Config()
    assert GRID[0] == pytest.approx(cfg.ki_min) and GRID[-1] == pytest.approx(cfg.ki_max) and len(GRID) == cfg.ki_grid_size


@pytest.mark.parametrize("ki_true", [0.2, 0.5, 1.5, 4.0])
def test_hyperbolic_fit_recovers_parameters_without_noise(ki_true):
    s = 1.02 * phi(DOSE, ki_true) + 0.15 * B
    f = fit_hyper(DOSE, B, s, GRID)
    assert f["KI"] == pytest.approx(ki_true, rel=1e-4)
    assert f["c"] == pytest.approx(1.02, rel=1e-4) and f["gamma"] == pytest.approx(0.15, rel=1e-3)
    assert f["rss"] < 1e-12


def test_linear_and_null_fits_are_exact_for_their_own_data():
    lin = 1.0 - 0.8 * DOSE + 0.2 * B
    f = fit_linear(DOSE, B, lin)
    assert f["slope"] == pytest.approx(-0.8) and f["rss"] < 1e-20
    z = fit_null(B, 1.0 + 0.2 * B)
    assert z["gamma"] == pytest.approx(0.2) and z["rss"] < 1e-20


def test_aicc_formula():
    rss, n, k_reg = 0.37, 16, 3
    k = k_reg + 1
    assert aicc(rss, k_reg, n) == pytest.approx(n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


def test_batch_only_model_wins_when_there_is_no_dose_effect():
    rng = np.random.default_rng(3)
    s = 1.0 + 0.2 * B + rng.normal(0, 0.05, len(B))
    h, l, z = compare_models(DOSE, B, s, GRID)
    assert z["aicc"] < h["aicc"] and z["aicc"] < l["aicc"]


def test_profile_interval_contains_the_truth_and_flags_missing_upper_limit():
    rng = np.random.default_rng(11)
    s = phi(DOSE, 0.3) + 0.1 * B + rng.normal(0, 0.01, len(B))
    h = fit_hyper(DOSE, B, s, GRID)
    p, curve = profile_interval(DOSE, B, s, GRID, h)
    assert p["lr_lo"] <= 0.3 * 1.5 and curve.shape == (len(GRID), 3)
    # the profile is evaluated on a grid, so its minimum is close to, but not exactly, zero
    assert -1e-9 <= curve[:, 2].min() < 0.2
    flat = 1.0 + 0.1 * B + rng.normal(0, 0.05, len(B))
    hf = fit_hyper(DOSE, B, flat, GRID)
    pf, _ = profile_interval(DOSE, B, flat, GRID, hf)
    assert pf["lr_unbounded_above"] is True


def test_profile_rss_is_minimal_at_the_fit():
    rng = np.random.default_rng(5)
    s = phi(DOSE, 1.0) + 0.1 * B + rng.normal(0, 0.03, len(B))
    h = fit_hyper(DOSE, B, s, GRID)
    for k in (h["KI"] * 0.7, h["KI"] * 1.4):
        assert rss_profile(k, DOSE, B, s)[0] >= h["rss"] - 1e-12


def test_bootstrap_is_stratified_and_reproducible():
    cfg = Config(n_boot=50)
    group = np.array(["Control"] * 3 + ["G14"] * 4 + ["Control"] * 3 + ["G14"] * 2 + ["G28"] * 4)
    batch = np.array(["Exp1"] * 7 + ["Exp2"] * 9)
    idx = bootstrap_indices(group, batch, cfg)
    assert idx.shape == (50, 16) and np.array_equal(idx, bootstrap_indices(group, batch, cfg))
    # every resample keeps the number of units of each treatment-by-batch cell
    cells = [np.where((group == g) & (batch == b))[0] for g in ("Control", "G14", "G28") for b in ("Exp1", "Exp2")]
    cells = [c for c in cells if len(c)]
    for row in idx:
        for c in cells:
            assert np.isin(row, c).sum() == len(c)
    s = phi(DOSE, 1.0) + 0.1 * B
    ki = bootstrap_ki(DOSE, B, s, idx[:5], GRID)
    assert ki.shape == (5,) and np.all((ki >= GRID[0] * 0.99) & (ki <= GRID[-1] * 1.01))
