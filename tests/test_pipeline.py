import json
from dataclasses import replace

import numpy as np
import pandas as pd

from gls1dr.cli import main
from gls1dr.pipeline import run_analysis
from gls1dr.synthetic import make_example

OUTPUTS = ["results.json", "table_S2_signature_buckets.csv", "table_S3_model_comparison.csv",
           "table_S3_ki_uncertainty.csv", "group_means.csv", "bucket_screen_maxT.csv", "mc_percentiles.npz"]


def test_pipeline_writes_all_outputs_and_is_reproducible(tmp_path, ds, spec, cfg):
    R1 = run_analysis(ds, spec, cfg, outdir=tmp_path / "a", log=lambda *a: None)
    R2 = run_analysis(ds, spec, cfg, outdir=tmp_path / "b", log=lambda *a: None)
    for f in OUTPUTS:
        assert (tmp_path / "a" / f).exists(), f
    assert R1["fit"]["proxy"]["hyper"]["KI"] == R2["fit"]["proxy"]["hyper"]["KI"]
    assert R1["mc"] == R2["mc"] and R1["perm"] == R2["perm"]
    saved = json.loads((tmp_path / "a" / "results.json").read_text())
    assert saved["perm"]["proxy"]["n_perm"] == 44100
    assert saved["provenance"]["n_samples"] == 16 and saved["provenance"]["batches"] == ["Exp1", "Exp2"]


def test_tables_have_the_expected_content(tmp_path, ds, spec, cfg):
    run_analysis(ds, spec, cfg, outdir=tmp_path, log=lambda *a: None)
    comp = pd.read_csv(tmp_path / "table_S3_model_comparison.csv")
    assert set(comp["model"]) == {"hyperbolic", "linear", "batch only"}
    assert set(comp["score"]) == {"aKG", "GSH", "proxy"} and (comp.groupby("score")["delta_aicc"].min() == 0).all()
    ki = pd.read_csv(tmp_path / "table_S3_ki_uncertainty.csv")
    assert (ki["KI_hat"].between(cfg.ki_min * 0.99, cfg.ki_max * 1.01)).all()
    screen = pd.read_csv(tmp_path / "bucket_screen_maxT.csv")
    assert len(screen) == len(ds.buckets) and screen["p_adj_maxT"].is_monotonic_increasing


def test_units_are_ng_per_ul_and_a_known_response_is_recovered(tmp_path, spec, cfg):
    """A strong, almost noiseless response with K_I = 0.2 ng/uL must be estimated in that unit."""
    ds = make_example(spec, cfg, seed=4, ki_true=0.2)
    R = run_analysis(ds, spec, cfg, outdir=tmp_path, log=lambda *a: None)
    est = R["fit"]["aKG"]["hyper"]["KI"]
    assert 0.1 < est < 0.4, est
    assert R["provenance"]["config"]["dose_map"]["G28"] == 0.28


def test_unknown_group_is_rejected(tmp_path, ds, spec, cfg):
    import pytest
    bad = replace(cfg, dose_map={"Control": 0.0, "G14": 0.14})
    with pytest.raises(ValueError, match="without a dose"):
        run_analysis(ds, spec, bad, outdir=tmp_path, log=lambda *a: None)


def test_command_line_demo(tmp_path):
    out = tmp_path / "demo"
    assert main(["--demo", "--out", str(out), "--n-boot", "40", "--n-mc", "1000", "--n-verify", "300"]) == 0
    assert (out / "results.json").exists() and (out / "demo_input_synthetic.csv").exists()
    tidy = tmp_path / "tidy.csv"
    assert main(["--input", str(out / "demo_input_synthetic.csv"), "--export-tidy", str(tidy)]) == 0 and tidy.exists()


def test_figures_are_written(tmp_path, ds, spec, cfg):
    run_analysis(ds, spec, cfg, outdir=tmp_path, figures=True, log=lambda *a: None)
    for f in ("Fig5C_dose_response.png", "FigS9_model_predictions.png"):
        assert (tmp_path / f).stat().st_size > 10_000
