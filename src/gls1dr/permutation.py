"""Exact permutation tests within batch and the family-wise (max-|t|) adjustment."""
from itertools import combinations

import numpy as np


def unique_perms(d):
    """All distinct permutations of the dose vector d, enumerated through combinations."""
    d = np.asarray(d, float)
    vals = sorted(set(d))
    n = len(d)
    out = []

    def rec(avail, k, cur):
        if k == len(vals) - 1:
            arr = np.empty(n)
            for v, idx in cur:
                arr[idx] = v
            arr[avail] = vals[-1]
            out.append(arr.copy())
            return
        for c in combinations(avail, int((d == vals[k]).sum())):
            rec([a for a in avail if a not in c], k + 1, cur + [(vals[k], list(c))])

    rec(list(range(n)), 0, [])
    return np.array(out)


def exact_perm_slope(S, dose, B):
    """Batch-adjusted slope of the score on dose and its exact two-sided p value.

    The dose labels are permuted within each batch; all distinct permutations are enumerated
    (44,100 for the design of the study). Returns (slope, p, number of permutations).
    """
    i1, i2 = np.where(B == 0)[0], np.where(B == 1)[0]
    P1, P2 = unique_perms(dose[i1]), unique_perms(dose[i2])

    def parts(P, Sb):
        Dc = P - P.mean(axis=1, keepdims=True)
        return Dc @ (Sb - Sb.mean()), (Dc ** 2).sum(axis=1)

    n1, d1 = parts(P1, S[i1])
    n2, d2 = parts(P2, S[i2])
    slopes = (n1[:, None] + n2[None, :]) / (d1[:, None] + d2[None, :])
    obs = ((dose[i1] - dose[i1].mean()) @ (S[i1] - S[i1].mean()) +
           (dose[i2] - dose[i2].mean()) @ (S[i2] - S[i2].mean())) / \
          (((dose[i1] - dose[i1].mean()) ** 2).sum() + ((dose[i2] - dose[i2].mean()) ** 2).sum())
    return float(obs), float(np.mean(np.abs(slopes) >= abs(obs) - 1e-12)), int(slopes.size)


def exact_pair(S, group, batch, g1, g0, b):
    """Difference of means g1 - g0 within batch b, with an exact permutation p value."""
    m = (batch == b) & ((group == g1) | (group == g0))
    v = S[m]
    lab = (group[m] == g1).astype(int)
    k = lab.sum()
    obs = v[lab == 1].mean() - v[lab == 0].mean()
    cnt = tot = 0
    for c in combinations(range(len(v)), k):
        l = np.zeros(len(v), int)
        l[list(c)] = 1
        s = v[l == 1].mean() - v[l == 0].mean()
        cnt += abs(s) >= abs(obs) - 1e-12
        tot += 1
    return float(obs), cnt / tot, tot


def max_t_selection(Y, dose, B):
    """Family-wise adjustment over all buckets (Westfall-Young maximum-|t|), by exact enumeration.

    t is the t statistic of the dose slope in a regression with a batch term (n - 3 degrees of freedom).
    Returns (t observed, unadjusted exact p, family-wise adjusted p), one value per bucket.
    """
    df = len(dose) - 3
    i1, i2 = np.where(B == 0)[0], np.where(B == 1)[0]
    P1, P2 = unique_perms(dose[i1]), unique_perms(dose[i2])
    Yc1, Yc2 = Y[i1] - Y[i1].mean(axis=0), Y[i2] - Y[i2].mean(axis=0)
    syy = (Yc1 ** 2).sum(axis=0) + (Yc2 ** 2).sum(axis=0)

    def parts(P, Yc):
        Dc = P - P.mean(axis=1, keepdims=True)
        return Dc @ Yc, (Dc ** 2).sum(axis=1)

    N1, D1 = parts(P1, Yc1)
    N2, D2 = parts(P2, Yc2)
    abs_t, tmax = [], []
    for a in range(len(P1)):
        num = N1[a][None, :] + N2
        den = (D1[a] + D2)[:, None]
        beta = num / den
        rss = np.maximum(syy[None, :] - beta ** 2 * den, 1e-12)
        t = np.abs(beta / np.sqrt(rss / df / den))
        abs_t.append(t)
        tmax.append(t.max(axis=1))
    abs_t, tmax = np.concatenate(abs_t), np.concatenate(tmax)
    Dc = np.concatenate([dose[i1] - dose[i1].mean(), dose[i2] - dose[i2].mean()])
    Yc = np.concatenate([Yc1, Yc2])
    den_o = (Dc ** 2).sum()
    beta_o = (Dc @ Yc) / den_o
    t_o = beta_o / np.sqrt(np.maximum(syy - beta_o ** 2 * den_o, 1e-12) / df / den_o)
    p_raw = (abs_t >= np.abs(t_o)[None, :] - 1e-12).mean(axis=0)
    p_adj = np.array([(tmax >= abs(t_o[j]) - 1e-12).mean() for j in range(Y.shape[1])])
    return t_o, p_raw, p_adj
