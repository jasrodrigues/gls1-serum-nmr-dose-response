"""Variance-stabilising transform and noise estimation."""
import numpy as np


def estimate_noise(ds, cfg):
    """Additive noise floor (sigma0) and multiplicative CV from within-cell standard deviations.

    A cell is a treatment group within a batch (at least two samples). Buckets with a low mean
    give sigma0; buckets with a high mean give the CV.
    """
    batches = sorted(set(ds.batch))
    cells = [(g, b) for g in cfg.dose_map for b in batches if ((ds.group == g) & (ds.batch == b)).sum() >= 2]
    p = ds.X.shape[1]
    mu, sd = np.zeros(p), np.zeros(p)
    for j in range(p):
        mus, sds = [], []
        for g, b in cells:
            m = (ds.group == g) & (ds.batch == b)
            mus.append(ds.X[m, j].mean())
            sds.append(ds.X[m, j].std(ddof=1))
        mu[j] = np.mean(mus)
        sd[j] = np.sqrt(np.mean(np.square(sds)))
    low, hi = mu < cfg.noise_low_mean, mu > cfg.noise_high_mean
    if low.sum() == 0 or hi.sum() == 0:
        raise ValueError("cannot estimate the noise: no buckets in the low- or high-intensity range "
                         "(adjust noise_low_mean / noise_high_mean in Config)")
    sigma0 = float(np.median(sd[low]))
    cv = float(np.median(sd[hi] / mu[hi]))
    return dict(sigma0=sigma0, cv=cv, c=sigma0 / cv, n_low=int(low.sum()), n_hi=int(hi.sum()))


def glog(x, c):
    """Generalised logarithm: asinh(x / 2c) / ln 2 (Supplementary Methods)."""
    return np.arcsinh(x / (2 * c)) / np.log(2)
