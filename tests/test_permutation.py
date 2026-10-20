from itertools import permutations

import numpy as np
import pytest

from gls1dr.permutation import exact_pair, exact_perm_slope, max_t_selection, unique_perms


def test_number_of_distinct_permutations_of_the_study_design():
    batch1 = [0.0] * 3 + [0.14] * 4                      # 3 control + 4 dose 1
    batch2 = [0.0] * 3 + [0.14] * 2 + [0.28] * 4         # 3 + 2 + 4
    assert len(unique_perms(batch1)) == 35
    assert len(unique_perms(batch2)) == 1260
    assert 35 * 1260 == 44100


def test_unique_perms_are_distinct_and_preserve_the_multiset():
    d = np.array([0, 0, 1, 1, 2.0])
    P = unique_perms(d)
    assert len({tuple(r) for r in P}) == len(P) == 30
    assert all(sorted(r) == sorted(d) for r in P)


def _brute_force_p(S, dose, B):
    i1, i2 = np.where(B == 0)[0], np.where(B == 1)[0]

    def slope(d):
        num = sum((d[i] - d[i].mean()) @ (S[i] - S[i].mean()) for i in (i1, i2))
        den = sum(((d[i] - d[i].mean()) ** 2).sum() for i in (i1, i2))
        return num / den

    obs = slope(dose)
    p1 = {tuple(p) for p in permutations(dose[i1])}
    p2 = {tuple(p) for p in permutations(dose[i2])}
    cnt = tot = 0
    for a in p1:
        for b in p2:
            d = dose.copy(); d[i1] = a; d[i2] = b
            cnt += abs(slope(d)) >= abs(obs) - 1e-12; tot += 1
    return obs, cnt / tot, tot


def test_exact_slope_test_matches_brute_force():
    rng = np.random.default_rng(0)
    dose = np.array([0, 0, 0.14, 0.14, 0, 0.14, 0.28, 0.28])
    B = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    S = rng.normal(1, 0.1, 8)
    assert exact_perm_slope(S, dose, B) == pytest.approx(_brute_force_p(S, dose, B), rel=1e-9, abs=1e-12)


def test_no_dose_effect_gives_p_one_and_a_strong_effect_gives_small_p():
    dose = np.array([0, 0, 0, 0.14, 0.14, 0.14, 0.14, 0, 0, 0, 0.14, 0.14, 0.28, 0.28, 0.28, 0.28])
    B = np.array([0] * 7 + [1] * 9, float)
    flat = 1.0 + 0.3 * B                                  # batch effect only
    assert exact_perm_slope(flat, dose, B)[1] == 1.0
    strong = 1.0 - 2.0 * dose + 0.3 * B
    assert exact_perm_slope(strong, dose, B)[1] < 0.01
    assert exact_perm_slope(strong, dose, B)[2] == 44100


def test_exact_pair_is_symmetric_in_sign_and_bounded():
    group = np.array(["C", "C", "C", "T", "T", "T"]); batch = np.array(["a"] * 6)
    S = np.array([1.0, 1.1, 0.9, 0.5, 0.6, 0.4])
    d1, p1, n1 = exact_pair(S, group, batch, "T", "C", "a")
    d2, p2, _ = exact_pair(S, group, batch, "C", "T", "a")
    assert d1 == -d2 and p1 == p2 and n1 == 20 and 0 < p1 <= 1


def test_family_wise_adjusted_p_is_never_smaller_than_the_unadjusted_p(ds, cfg):
    from gls1dr.transform import estimate_noise, glog
    dose = np.array([cfg.dose_map[g] for g in ds.group])
    B = (ds.batch == sorted(set(ds.batch))[1]).astype(float)
    Y = glog(ds.X, estimate_noise(ds, cfg)["c"])
    t, p_raw, p_adj = max_t_selection(Y, dose, B)
    assert np.all(p_adj >= p_raw - 1e-12) and np.all(p_adj > 0) and np.all(p_adj <= 1)
