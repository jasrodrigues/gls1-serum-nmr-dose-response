"""Figures. Matplotlib is imported lazily so that the analysis runs without it."""
import numpy as np

from .model import equilibrium, phi

COL_B1, COL_B2, COL_FIT = "#ff7f0e", "#2ca02c", "#1f77b4"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig5c(path, ds, dose, B, score, fit, ki_boot, cfg, title=None):
    """Dose-response of the metabolic proxy: batch-adjusted points, group means, fit, extrapolation,
    bootstrap band and K_I. The score is rescaled so that Phi(0) = 1."""
    plt = _plt()
    batches = sorted(set(ds.batch))
    y = (score - fit["gamma"] * B) / fit["c"]
    ki = fit["KI"]
    dmax = max(cfg.dose_map.values())
    xmax = min(max(1.15 * ki, 4 * dmax), 6 * dmax)   # keep the tested range readable
    dg = np.linspace(0, xmax, 400)
    band = np.array([phi(dg, k) for k in ki_boot])
    lo, hi = np.percentile(band, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.axvspan(0, dmax, color=COL_FIT, alpha=0.10, zorder=0, label=f"Tested dose range (0\u2013{dmax:g} ng/\u00b5L)")
    ax.fill_between(dg, lo, hi, color=COL_FIT, alpha=0.15, zorder=1, label="95% bootstrap interval")
    inside = dg <= dmax
    ax.plot(dg[inside], phi(dg[inside], ki), color=COL_FIT, lw=3, zorder=3,
            label=f"Fitted $\\Phi(D)$, $n=1$\n$\\hat{{K}}_I = {ki:.2f}$ ng/\u00b5L")
    ax.plot(dg[~inside], phi(dg[~inside], ki), color=COL_B1, lw=2.5, ls="--", zorder=3,
            label=f"Model extrapolation (D > {dmax:g} ng/\u00b5L)")
    if ki <= xmax:
        ax.axvline(ki, color=COL_FIT, ls=":", lw=1.5)
        ax.text(ki, 0.6, f"$K_I = {ki:.2f}$ ng/\u00b5L", rotation=90, ha="right", va="center", fontsize=10)
    rng = np.random.default_rng(5)
    for bt, col in zip(batches, (COL_B1, COL_B2)):
        m = ds.batch == bt
        ax.scatter(dose[m] + rng.uniform(-0.004, 0.004, m.sum()), y[m], s=80, color=col, edgecolor="white", zorder=4,
                   label=f"Sera ({bt})")
    for g, d0 in cfg.dose_map.items():
        m = ds.group == g
        if m.any():
            ax.scatter([d0], [y[m].mean()], marker="D", s=170, facecolor="white", edgecolor="black", lw=2, zorder=5)
            ax.text(d0, 1.24, cfg.display_names.get(g, g), ha="center", fontweight="bold", fontsize=11)
    ax.axhline(1, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("Dose (ng/\u00b5L)", fontsize=15)
    ax.set_ylabel("Relative downstream glutaminolysis-associated\nmetabolic proxy (batch-adjusted)", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.55, 0.0))
    if title:
        ax.set_title(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_s9(path, mc, cfg):
    """Model predictions for GSH and ROS relative to the untreated state."""
    plt = _plt()
    D = mc["D"]
    dmax = max(cfg.dose_map.values())
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), sharex=True)
    cols = {"A": "#1B7A3E", "B": "#1B4F72"}
    titles = {"A": "A. Calibration uncertainty only\n(K$_I$ from bootstrap; \u03b7 = \u03be = 1)",
              "B": "B. Combined parametric uncertainty\n(K$_I$ from bootstrap; \u03b7, \u03be log-uniform)"}
    for ax, key in zip([axes[0, 0], axes[0, 1]], "AB"):
        med, lo, hi = mc[key]["h"]
        ax.axvspan(0, dmax, color="#EDEDED", alpha=0.9, label=f"Tested dose range (0\u2013{dmax:g} ng/\u00b5L)")
        ax.fill_between(D, lo, hi, color=cols[key], alpha=0.22, label="2.5\u201397.5th percentile")
        ax.plot(D, med, color=cols[key], lw=2.2, label="Median")
        ax.axhline(1, color="gray", ls=":", lw=0.8)
        ax.set_ylim(0.2, 1.05)
        ax.set_title(titles[key], fontsize=8.5)
        ax.set_ylabel("Predicted GSH relative to untreated")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    ax = axes[1, 0]
    ax.axvspan(0, dmax, color="#EDEDED", alpha=0.9)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(cfg.ki_fixed)))
    h0, _ = equilibrium(np.ones(1), np.ones(1), np.ones(1))
    for ki, c in zip(cfg.ki_fixed, cmap):
        g = phi(D, ki)
        h, _ = equilibrium(g, np.ones_like(g), np.ones_like(g))
        ax.plot(D, h / h0, color=c, lw=2, label=f"K$_I$ = {ki:g} ng/\u00b5L")
    ax.axhline(1, color="gray", ls=":", lw=0.8)
    ax.set_ylim(0.2, 1.05)
    ax.set_title("C. Fixed K$_I$ values (\u03b7 = \u03be = 1)", fontsize=8.5)
    ax.set_ylabel("Predicted GSH relative to untreated")
    ax.set_xlabel("Dose (ng/\u00b5L)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax = axes[1, 1]
    med, lo, hi = mc["B"]["r"]
    ax.axvspan(0, dmax, color="#EDEDED", alpha=0.9)
    ax.fill_between(D, lo, hi, color="#C0622D", alpha=0.22)
    ax.plot(D, med, color="#C0622D", lw=2.2)
    ax.axhline(1, color="gray", ls=":", lw=0.8)
    ax.set_ylim(0.95, max(1.2, float(hi.max()) * 1.05))
    ax.set_title("D. Predicted ROS relative to untreated\n(analysis B)", fontsize=8.5)
    ax.set_xlabel("Dose (ng/\u00b5L)")
    ax.set_ylabel("Predicted ROS relative to untreated")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
