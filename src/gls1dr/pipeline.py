"""End-to-end analysis, in the order of the Methods section.

1. spectral signatures and the metabolic proxy (Eqs. 1-2)
2. exact permutation tests within batch (slope on dose; contrasts within batch)
3. dose-response calibration: hyperbolic vs linear vs batch-only, AICc, profile likelihood,
   stratified bootstrap (Eqs. 2a-2b)
4. reduced mechanistic model: Monte Carlo propagation, fixed-K_I sensitivity, numerical checks (Eq. 3)
5. family-wise (max-|t|) screen of every bucket
"""
import json
import os
import platform
from dataclasses import asdict

import numpy as np
import pandas as pd
import scipy

from . import __version__
from . import signatures as sg
from .config import Config
from .fitting import bootstrap_indices, bootstrap_ki, compare_models, ki_grid, profile_interval
from .io import bucket_ppm
from .model import propagate, verify_model
from .permutation import exact_pair, exact_perm_slope, max_t_selection
from .transform import estimate_noise, glog


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def run_analysis(ds, spec, cfg=None, outdir="results", figures=False, log=print):
    cfg = cfg or Config()
    os.makedirs(outdir, exist_ok=True)
    unknown = sorted(set(ds.group) - set(cfg.dose_map))
    if unknown:
        raise ValueError(f"groups without a dose in Config.dose_map: {unknown}")

    batches = sorted(set(ds.batch))
    dose = np.array([cfg.dose_map[g] for g in ds.group], float)            # ng/uL
    B = (ds.batch == batches[1]).astype(float)
    n = len(dose)
    noise = estimate_noise(ds, cfg)
    Y = glog(ds.X, noise["c"])
    log(f"noise: sigma0={noise['sigma0']:.2f}  CV={noise['cv']:.3f}  c={noise['c']:.1f}")

    R = {"provenance": dict(package_version=__version__, python=platform.python_version(), numpy=np.__version__,
                            scipy=scipy.__version__, pandas=pd.__version__, config=asdict(cfg), batches=batches,
                            n_samples=n, n_buckets=len(ds.buckets)),
         "noise": noise}

    # 1. signatures ------------------------------------------------------------------------------
    S = sg.signature_scores(ds, spec, cfg.reference_group)
    proxy = spec["proxy"]["name"]
    names = list(S)
    doc = sg.documentation_table(ds, spec)
    doc.to_csv(os.path.join(outdir, "table_S2_signature_buckets.csv"), index=False)
    R["r_within_signature"] = sg.within_block_correlation(Y, ds, spec)
    R["r_between_signatures"] = float(np.corrcoef(S[spec["proxy"]["members"][0]], S[spec["proxy"]["members"][1]])[0, 1]) \
        if len(spec["proxy"]["members"]) == 2 else None
    R["second_resonance_checks"] = sg.second_resonance_checks(Y, ds, spec, S)
    R["batch_difference_in_reference"] = sg.batch_difference_in_reference(ds, cfg.reference_group)

    rows = []
    for k in names:
        for g in cfg.dose_map:
            for b in batches:
                m = (ds.group == g) & (ds.batch == b)
                if m.sum():
                    rows.append(dict(score=k, group=cfg.display_names.get(g, g), batch=b, n=int(m.sum()),
                                     mean=float(S[k][m].mean()), sd=float(S[k][m].std(ddof=1)) if m.sum() > 1 else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "group_means.csv"), index=False)

    # 2. permutation tests -------------------------------------------------------------------------
    R["perm"], R["pairs"] = {}, {}
    for k in names:
        slope, p, tot = exact_perm_slope(S[k], dose, B)
        R["perm"][k] = dict(slope_per_ng_ul=slope, p=p, n_perm=tot)
        pairs = {}
        for b in batches:
            for g in cfg.dose_map:
                if g == cfg.reference_group:
                    continue
                if ((ds.batch == b) & (ds.group == g)).any() and ((ds.batch == b) & (ds.group == cfg.reference_group)).any():
                    pairs[f"{b}_{cfg.display_names.get(g, g)}_vs_{cfg.display_names.get(cfg.reference_group, cfg.reference_group)}"] = \
                        exact_pair(S[k], ds.group, ds.batch, g, cfg.reference_group, b)
        R["pairs"][k] = {key: dict(difference=v[0], p=v[1], n_perm=v[2]) for key, v in pairs.items()}

    # 3. calibration ---------------------------------------------------------------------------------
    grid = ki_grid(cfg)
    idx_boot = bootstrap_indices(ds.group, ds.batch, cfg)
    R["fit"], boot, prof = {}, {}, {}
    comp_rows, ki_rows = [], []
    for k in names:
        h, l, z = compare_models(dose, B, S[k], grid)
        pint, curve = profile_interval(dose, B, S[k], grid, h)
        bs = bootstrap_ki(dose, B, S[k], idx_boot, grid)
        boot[k], prof[k] = bs, curve
        at_bound = float((bs >= 0.99 * cfg.ki_max).mean())
        R["fit"][k] = dict(hyper=h, linear=l, null=z, profile=pint,
                           boot=dict(median=float(np.median(bs)), lo=float(np.percentile(bs, 2.5)),
                                     hi=float(np.percentile(bs, 97.5)), frac_at_upper_bound=at_bound,
                                     frac_below_dmax=float((bs < max(cfg.dose_map.values())).mean())))
        best = min(h["aicc"], l["aicc"], z["aicc"])
        for model, f, kreg in (("hyperbolic", h, 3), ("linear", l, 3), ("batch only", z, 2)):
            comp_rows.append(dict(score=k, model=model, n_parameters=kreg + 1, rss=f["rss"], aicc=f["aicc"],
                                  delta_aicc=f["aicc"] - best))
        ki_rows.append(dict(score=k, KI_hat=h["KI"], profile_lower_95=pint["lr_lo"], profile_F_lower_95=pint["f_lo"],
                            profile_unbounded_above=pint["lr_unbounded_above"], boot_median=float(np.median(bs)),
                            boot_p2_5=float(np.percentile(bs, 2.5)), boot_p97_5=float(np.percentile(bs, 97.5)),
                            boot_frac_at_upper_bound=at_bound, perm_slope=R["perm"][k]["slope_per_ng_ul"],
                            perm_p=R["perm"][k]["p"]))
        log(f"[{k}] K_I = {h['KI']:.3g} ng/uL; profile lower limit {pint['lr_lo']:.3g}; "
            f"bootstrap median {np.median(bs):.3g}; AICc hyp/lin/null = {h['aicc']:.2f}/{l['aicc']:.2f}/{z['aicc']:.2f}; "
            f"p(slope) = {R['perm'][k]['p']:.3f}")
    pd.DataFrame(comp_rows).to_csv(os.path.join(outdir, "table_S3_model_comparison.csv"), index=False)
    pd.DataFrame(ki_rows).to_csv(os.path.join(outdir, "table_S3_ki_uncertainty.csv"), index=False)

    # 4. mechanistic model -----------------------------------------------------------------------------
    D = np.arange(0, cfg.mc_dose_max + cfg.mc_dose_step / 2, cfg.mc_dose_step)
    rng = np.random.default_rng(cfg.seed_mc)
    lo, hi = cfg.eta_xi_log10_range
    ki_s = rng.choice(boot[proxy], size=cfg.n_mc, replace=True)
    eta_s, xi_s = 10 ** rng.uniform(lo, hi, cfg.n_mc), 10 ** rng.uniform(lo, hi, cfg.n_mc)
    scen = {"A": (ki_s, np.ones(cfg.n_mc), np.ones(cfg.n_mc)), "B": (ki_s, eta_s, xi_s)}
    MC = {"D": D}
    R["mc"] = {}
    for key, (ki, e, x) in scen.items():
        a, h, r = propagate(ki, e, x, D)
        MC[key] = {v: np.percentile(M, [50, 2.5, 97.5], axis=0) for v, M in (("a", a), ("h", h), ("r", r))}
        R["mc"][key] = {v: {f"{d0:g}": MC[key][v][:, int(round(d0 / cfg.mc_dose_step))].tolist() for d0 in cfg.mc_report_doses}
                        for v in ("a", "h", "r")}
    R["mc_note"] = "each entry is [median, 2.5th, 97.5th percentile] of the value relative to the untreated state"
    R["ki_fixed_sensitivity"] = {}
    for ki in cfg.ki_fixed:
        a, h, r = propagate(np.array([float(ki)]), np.ones(1), np.ones(1), D)
        R["ki_fixed_sensitivity"][f"{ki:g}"] = {v: {f"{d0:g}": float(M[0, int(round(d0 / cfg.mc_dose_step))]) for d0 in cfg.mc_report_doses}
                                                for v, M in (("a", a), ("h", h), ("r", r))}
    R["model_checks"] = verify_model(boot[proxy], cfg)

    # 5. family-wise screen of every bucket -------------------------------------------------------------
    t_o, p_raw, p_adj = max_t_selection(Y, dose, B)
    screen = pd.DataFrame(dict(bucket=ds.buckets, ppm=[bucket_ppm(b) for b in ds.buckets], t=t_o, p_exact=p_raw,
                               p_adj_maxT=p_adj)).sort_values("p_adj_maxT")
    screen.to_csv(os.path.join(outdir, "bucket_screen_maxT.csv"), index=False)
    R["screen_top"] = screen.head(10).to_dict(orient="records")

    # outputs ---------------------------------------------------------------------------------------------
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=_jsonable)
    np.savez(os.path.join(outdir, "mc_percentiles.npz"), D=D, **{f"{k}_{v}": MC[k][v] for k in "AB" for v in "ahr"})
    if figures:
        from .figures import fig5c, fig_s9
        fig5c(os.path.join(outdir, "Fig5C_dose_response.png"), ds, dose, B, S[proxy], R["fit"][proxy]["hyper"], boot[proxy], cfg)
        fig_s9(os.path.join(outdir, "FigS9_model_predictions.png"), MC, cfg)
    log(f"done. Results in {outdir}/")
    return R
