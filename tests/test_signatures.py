import json
from pathlib import Path

import numpy as np
import pytest

from gls1dr.signatures import (batch_difference_in_reference, documentation_table, load_signatures,
                               second_resonance_checks, signature_scores, within_block_correlation)
from gls1dr.transform import estimate_noise, glog


def test_reference_group_scores_average_one(ds, spec, cfg):
    S = signature_scores(ds, spec, cfg.reference_group)
    ref = ds.group == cfg.reference_group
    for k, v in S.items():
        assert np.isclose(v[ref].mean(), 1.0), k


def test_proxy_is_the_mean_of_its_members(ds, spec, cfg):
    S = signature_scores(ds, spec, cfg.reference_group)
    m = spec["proxy"]["members"]
    assert np.allclose(S[spec["proxy"]["name"]], np.mean([S[x] for x in m], axis=0))


def test_missing_bucket_is_reported(ds, spec, cfg):
    bad = json.loads(json.dumps(spec))
    bad["signatures"]["aKG"]["buckets"].append("B9_9999")
    with pytest.raises(ValueError, match="not in the table"):
        signature_scores(ds, bad, cfg.reference_group)


def test_documentation_and_diagnostics(ds, spec, cfg):
    doc = documentation_table(ds, spec)
    assert len(doc) == sum(len(s["buckets"]) for s in spec["signatures"].values())
    assert doc["ppm"].between(0, 10).all()
    Y = glog(ds.X, estimate_noise(ds, cfg)["c"])
    S = signature_scores(ds, spec, cfg.reference_group)
    assert set(within_block_correlation(Y, ds, spec)) == set(spec["signatures"])
    checks = second_resonance_checks(Y, ds, spec, S)
    assert any(k.startswith("GSH_") for k in checks)
    assert 0 <= batch_difference_in_reference(ds, cfg.reference_group)["fraction_gt2_fold"] <= 1


def test_packaged_signatures_match_the_documented_config_file():
    repo = Path(__file__).resolve().parents[1]
    a = json.loads((repo / "configs" / "signatures.json").read_text(encoding="utf-8"))
    b = json.loads((repo / "src" / "gls1dr" / "signatures_default.json").read_text(encoding="utf-8"))
    assert a == b, "configs/signatures.json and src/gls1dr/signatures_default.json must be identical"
