"""Synthetic bucket table with the design of the study. Used by the tests and by `gls1dr --demo`.

It is NOT the real data: intensities are simulated, with a known hyperbolic response of the signature
buckets, a multiplicative batch effect and a mix of additive and multiplicative noise.
"""
import numpy as np

from .config import Config
from .io import Dataset
from .model import phi

# samples per (batch, group): the design of the study (7 + 9 sera)
DESIGN = [("Exp1", "Control", 3), ("Exp1", "G14", 4), ("Exp2", "Control", 3), ("Exp2", "G14", 2), ("Exp2", "G28", 4)]


def make_example(spec, cfg=None, seed=1, ki_true=1.5, n_other_buckets=60, response=True):
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    sig_buckets = [b for s in spec["signatures"].values() for b in s["buckets"]]
    extra = [b for lst in spec.get("diagnostics", {}).get("second_resonance_buckets", {}).values() for b in lst]
    other = [f"B1_{i:04d}" for i in range(100, 100 + n_other_buckets)]
    buckets = list(dict.fromkeys(sig_buckets + extra + other))
    p = len(buckets)

    group, batch = [], []
    for bt, g, k in DESIGN:
        group += [g] * k
        batch += [bt] * k
    group, batch = np.array(group), np.array(batch)
    n = len(group)
    dose = np.array([cfg.dose_map[g] for g in group])

    base = 10 ** rng.uniform(0.5, 3.5, p)                      # 3 to 3,000 intensity units
    is_sig = np.array([b in sig_buckets for b in buckets])
    base[is_sig] = 10 ** rng.uniform(2.2, 3.2, is_sig.sum())   # signature buckets well above the noise floor
    batch_eff = np.exp(rng.normal(0, 0.3, p))
    X = np.tile(base, (n, 1)) * np.where((batch == "Exp2")[:, None], batch_eff[None, :], 1.0)
    if response:
        resp = phi(dose, ki_true)
        for j, b in enumerate(buckets):
            if b in sig_buckets:
                X[:, j] *= resp
    X = X * np.exp(rng.normal(0, 0.2, X.shape)) + rng.normal(0, 8, X.shape)
    sample = np.array([f"S{i + 1:02d}" for i in range(n)])
    return Dataset(X=X, sample=sample, group=group, batch=batch, buckets=buckets)
