# From the Methods to the code

Equation numbers are those of the section "Mathematical modeling of the serum metabolic response" of the manuscript.

| Statement in the Methods | Eq. | Where |
|---|---|---|
| Sera of week 4, two batches, dose in ng/µL | | `config.Config.dose_map`, `io.Dataset` |
| Bucket intensity divided by the control mean, averaged within a signature | 1 | `signatures.signature_scores` |
| α-KG- and GSH-associated signatures, proxy as their mean | | `configs/signatures.json`; Table S2 from `signatures.documentation_table` |
| Hyperbolic response Φ(D), Hill coefficient fixed at 1 | 2a | `model.phi` |
| Observation model with scale and additive batch effect; linear least squares for c and γ, search over K<sub>I</sub> | 2b | `fitting.rss_profile`, `fitting.fit_hyper` |
| Linear and batch-only models; ranking by AICc | | `fitting.fit_linear`, `fitting.fit_null`, `fitting.aicc`, `fitting.compare_models` (Table S3) |
| Profile likelihood, 95% point of χ² with one degree of freedom | | `fitting.profile_interval` |
| Bootstrap of 2,000 resamples stratified by treatment and batch | | `fitting.bootstrap_indices`, `fitting.bootstrap_ki` |
| Exact permutation test of the slope within batch (44,100 permutations) | | `permutation.unique_perms`, `permutation.exact_perm_slope`, `permutation.exact_pair` |
| Maximum-\|t\| adjustment over the 221 buckets | | `permutation.max_t_selection` (`bucket_screen_maxT.csv`) |
| Reduced model, steady state (GSH h\*, ROS r\*) | 3 | `model.equilibrium` |
| Monte Carlo, 20,000 draws; η, ξ log-uniform in [0.1, 10]; K<sub>I</sub> from the bootstrap of the proxy | | `pipeline.run_analysis` (step 4), `model.propagate`; Figure S9 from `figures.fig_s9` |
| Local stability from the Jacobian; ODE versus closed form | Suppl. | `model.jacobian`, `model.verify_model` |
| Figure 5C | | `figures.fig5c` |

Seeds (`config.Config`): bootstrap 20261008, Monte Carlo 12345, model checks 777.

## Supplementary items cited in the Methods

| Item | Produced by |
|---|---|
| Supplementary Table S2 (buckets of each signature) | `table_S2_signature_buckets.csv` |
| Supplementary Table S3 (model comparison, K<sub>I</sub> uncertainty) | `table_S3_model_comparison.csv`, `table_S3_ki_uncertainty.csv` |
| Supplementary Figure S9 (model predictions) | `FigS9_model_predictions.png` |
| Supplementary Methods (full model, symbols, stability) | `model.py` docstring and the supplementary text |
