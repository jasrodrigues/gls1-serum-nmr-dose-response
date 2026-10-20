"""Analysis settings.

All doses are in ng/uL (concentration of GLS1 in the injected solution), as in the
Methods of the manuscript. Seeds are fixed, so every number is reproducible.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # group label in the input table -> dose (ng/uL)
    dose_map: dict = field(default_factory=lambda: {"Control": 0.0, "G14": 0.14, "G28": 0.28})
    reference_group: str = "Control"
    display_names: dict = field(default_factory=lambda: {"Control": "CTL", "G14": "GLS14", "G28": "GLS28"})

    # resampling and simulation sizes
    n_boot: int = 2000          # stratified bootstrap resamples
    n_mc: int = 20000           # Monte Carlo draws (mechanistic model)
    n_verify: int = 20000       # parameter sets for the numerical checks of the model
    n_ode_checks: int = 200     # of those, how many are also integrated as ODEs

    # seeds
    seed_boot: int = 20261008
    seed_mc: int = 12345
    seed_verify: int = 777

    # search interval for K_I (ng/uL); a value at ki_max means "no dose effect"
    ki_min: float = 0.005
    ki_max: float = 1000.0
    ki_grid_size: int = 400

    # fixed K_I values for the deterministic sensitivity analysis (ng/uL)
    ki_fixed: tuple = (0.5, 1.0, 2.0, 4.0, 10.0)

    # dose grid for the model predictions (ng/uL)
    mc_dose_max: float = 1.0
    mc_dose_step: float = 0.01
    mc_report_doses: tuple = (0.14, 0.28, 0.5, 1.0)

    # log10 ranges of the parameters that the data cannot determine
    eta_xi_log10_range: tuple = (-1.0, 1.0)
    kinetic_log10_range: tuple = (-1.0, 1.0)       # alpha, beta, delta (stability check only)

    # noise estimation for the generalised logarithm
    noise_low_mean: float = 25.0
    noise_high_mean: float = 200.0
