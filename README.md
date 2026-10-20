# gls1-serum-nmr-dose-response

Code for the mathematical analysis in Martins et al. (2026), *Glutaminase 1 (GLS1)-based systemic therapy decreases blood glutamine availability and impairs glutamine-dependent metabolism in glioblastoma (GBM)*.

It takes the serum ¹H-NMR bucket table collected halfway through the treatment (week 4) and does five things:

1. builds two spectral signatures (α-ketoglutarate- and GSH-associated) and their mean, the metabolic proxy;
2. tests the dose effect with exact permutation tests within batch;
3. fits a hyperbolic dose–response, Φ(D) = 1/(1 + D/K<sub>I</sub>) with the Hill coefficient fixed at 1, and compares it with a linear and a batch-only model (AICc), with profile-likelihood and bootstrap limits for K<sub>I</sub>;
4. propagates the uncertainty through a reduced GSH–ROS model (Monte Carlo) and checks it numerically;
5. screens every bucket with a family-wise (max-|t|) permutation adjustment.

Doses are in ng/µL, the concentration of GLS1 in the injected solution (0, 0.14 and 0.28 ng/µL; 100 µL per injection).

**The study data are not in this repository.** Put the bucket table in `data/` (see [`data/README.md`](data/README.md) for the format) or follow the Data availability statement of the article. Everything below also runs on a synthetic table, so you can try the code without the data.

## Install

```bash
git clone https://github.com/<user>/gls1-serum-nmr-dose-response.git
cd gls1-serum-nmr-dose-response
python -m pip install -e ".[plot]"      # Python >= 3.9
```

## Run

Without data (synthetic table with the design of the study):

```bash
gls1dr --demo --out results_demo --figures
```

With the bucket table:

```bash
gls1dr --input data/bucket_table.csv --out results --figures
```

The default sizes are those of the Methods (2,000 bootstrap resamples, 20,000 Monte Carlo draws) and take a few minutes. `--n-boot`, `--n-mc` and `--n-verify` shorten a trial run. The signatures are read from [`configs/signatures.json`](configs/signatures.json); pass `--signatures` to use another file.

An Excel workbook in the NMRProcFlow layout (three header rows, `Samplecode` and `Condition` columns) is also accepted as `--input`. Use `--export-tidy table.csv` to convert it once to the plain CSV format.

## Outputs

| File | What it is |
|---|---|
| `Fig5C_dose_response.png` | Figure 5C: batch-adjusted proxy, group means, fit, extrapolation, bootstrap band |
| `FigS9_model_predictions.png` | Figure S9: predicted GSH and ROS relative to the untreated state |
| `table_S2_signature_buckets.csv` | Buckets of each signature, with chemical shift and mean intensity |
| `table_S3_model_comparison.csv` | Hyperbolic, linear and batch-only models: RSS, AICc, ΔAICc |
| `table_S3_ki_uncertainty.csv` | K<sub>I</sub>, profile and bootstrap limits, permutation slope and p value |
| `group_means.csv` | Mean and SD of each score by group and batch |
| `bucket_screen_maxT.csv` | Every bucket: t, exact p, family-wise adjusted p |
| `results.json` | All numbers, plus the settings, seeds and package versions used |
| `mc_percentiles.npz` | Monte Carlo medians and 2.5th–97.5th percentiles |

[`docs/METHODS_MAP.md`](docs/METHODS_MAP.md) links each statement of the Methods section to the function that implements it.

## Reproducibility

All random numbers come from fixed seeds (`gls1dr/config.py`), so two runs give the same output. `results.json` records the settings and the versions of Python, NumPy, SciPy and pandas. The permutation tests are exact (all 44,100 distinct permutations for the study design), not sampled.

```bash
python -m pytest -q
```

runs 44 tests on synthetic data only. They check, among other things, the closed-form steady state against ODE integration, the stability of the model, the exact tests against brute force, and that a known K<sub>I</sub> is recovered in ng/µL. The same tests run on GitHub Actions for Python 3.9, 3.11 and 3.12.

## What the analysis can and cannot say

- The design is small (16 sera from two batches, with dose 2 present only in the second one), so the analysis is exploratory.
- K<sub>I</sub> is an effective dose scale of the model, not an enzyme inhibition constant. With three dose levels it can usually be bounded only from below, and the bootstrap and profile limits show how far.
- Signature assignments rest on chemical shift alone. Scores are relative spectral intensities, not concentrations.
- In the mechanistic model the α-KG-associated state equals Φ(D) by construction, so the data estimate K<sub>I</sub> and do not validate the model. The fall of GSH and rise of ROS are model predictions.
- Exactly two batches are supported, and each serum sample is one unit of analysis.

## Layout

```
src/gls1dr/        the package (io, transform, signatures, permutation, fitting, model, figures, pipeline, cli)
configs/           signature definitions (editable)
data/              place the bucket table here (not tracked by git)
docs/              mapping from the Methods to the code
tests/             unit and end-to-end tests (synthetic data)
```

## License and citation

See `LICENSE` and `CITATION.cff`.
