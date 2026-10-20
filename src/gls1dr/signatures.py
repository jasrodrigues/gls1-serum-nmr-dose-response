"""Spectral signatures and the metabolic proxy (Eqs. 1 and 2 of the Methods)."""
import json

import numpy as np
import pandas as pd

from .io import bucket_ppm


def load_signatures(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    for key in ("signatures", "proxy"):
        if key not in spec:
            raise ValueError(f"signature file is missing '{key}'")
    unknown = [m for m in spec["proxy"]["members"] if m not in spec["signatures"]]
    if unknown:
        raise ValueError(f"proxy members not defined as signatures: {unknown}")
    return spec


def _index(ds):
    return {b: i for i, b in enumerate(ds.buckets)}


def signature_scores(ds, spec, reference_group):
    """S_im: mean over the buckets of signature m of intensity / mean intensity of the reference group.

    Returns a dict with one vector per signature and one for the proxy (mean of its members).
    """
    idx_of = _index(ds)
    ref = ds.group == reference_group
    if ref.sum() == 0:
        raise ValueError(f"no samples in the reference group '{reference_group}'")
    scores = {}
    for name, sig in spec["signatures"].items():
        missing = [b for b in sig["buckets"] if b not in idx_of]
        if missing:
            raise ValueError(f"signature '{name}': buckets not in the table: {missing}")
        idx = [idx_of[b] for b in sig["buckets"]]
        ref_mean = ds.X[ref][:, idx].mean(axis=0)
        if (ref_mean <= 0).any():
            raise ValueError(f"signature '{name}': a bucket has a non-positive mean in the reference group")
        scores[name] = (ds.X[:, idx] / ref_mean).mean(axis=1)
    scores[spec["proxy"]["name"]] = np.mean([scores[m] for m in spec["proxy"]["members"]], axis=0)
    return scores


def documentation_table(ds, spec):
    """Table S2: bucket, chemical shift and mean intensity of every bucket of every signature."""
    idx_of = _index(ds)
    rows = []
    for name, sig in spec["signatures"].items():
        for b in sig["buckets"]:
            rows.append(dict(signature=name, bucket=b, ppm=bucket_ppm(b), mean_intensity=float(ds.X[:, idx_of[b]].mean())))
    return pd.DataFrame(rows)


def within_block_correlation(Y, ds, spec):
    """Mean, minimum and maximum pairwise correlation of the (transformed) buckets inside each signature."""
    idx_of = _index(ds)
    out = {}
    for name, sig in spec["signatures"].items():
        idx = [idx_of[b] for b in sig["buckets"]]
        C = np.corrcoef(Y[:, idx].T)
        tri = C[np.triu_indices(len(idx), 1)]
        out[name] = [float(tri.mean()), float(tri.min()), float(tri.max())]
    return out


def second_resonance_checks(Y, ds, spec, scores):
    """Does a second resonance of the assigned metabolite co-vary with the signature?

    Configured in the 'diagnostics' block of the signature file; skipped if the buckets are absent.
    """
    idx_of = _index(ds)
    diag = spec.get("diagnostics", {})
    out = {}
    for sig, blist in diag.get("second_resonance_buckets", {}).items():
        for b in blist:
            if b in idx_of:
                members = spec["signatures"][sig]["buckets"]
                out[f"{b}_vs_{sig}_buckets"] = [float(np.corrcoef(Y[:, idx_of[b]], Y[:, idx_of[m]])[0, 1]) for m in members]
    for sig, targets in diag.get("second_resonance_ppm", {}).items():
        for label, target in targets.items():
            nearest = min(ds.buckets, key=lambda b: abs(bucket_ppm(b) - target))
            out[f"{sig}_{label}"] = dict(bucket=nearest, ppm=bucket_ppm(nearest),
                                         r=float(np.corrcoef(Y[:, idx_of[nearest]], scores[sig])[0, 1]))
    return out


def batch_difference_in_reference(ds, reference_group):
    """Share of buckets whose mean differs more than two-fold between the two batches in the reference group."""
    b0, b1 = sorted(set(ds.batch))
    ref = ds.group == reference_group
    m0, m1 = ds.X[ref & (ds.batch == b0)].mean(axis=0), ds.X[ref & (ds.batch == b1)].mean(axis=0)
    ok = (m0 > 1) & (m1 > 1)
    return dict(fraction_gt2_fold=float(np.mean(np.abs(np.log2(m1[ok] / m0[ok])) > 1)), n_buckets=int(ok.sum()))
