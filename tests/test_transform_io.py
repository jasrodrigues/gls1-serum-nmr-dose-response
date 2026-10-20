import numpy as np
import pandas as pd
import pytest

from gls1dr.io import Dataset, bucket_ppm, load_dataset, load_legacy_xlsx, load_tidy_csv, save_tidy_csv
from gls1dr.transform import estimate_noise, glog


def test_glog_is_monotone_and_invertible():
    x = np.linspace(-50, 5000, 400)
    y = glog(x, 40.0)
    assert np.all(np.diff(y) > 0)
    assert np.allclose(2 * 40.0 * np.sinh(y * np.log(2)), x)


def test_glog_is_linear_near_zero_and_logarithmic_for_large_values():
    c = 40.0
    assert np.isclose(glog(1.0, c), 1.0 / (2 * c * np.log(2)), rtol=1e-3)
    assert np.isclose(glog(1e6, c) - glog(1e5, c), np.log2(10), atol=1e-3)


def test_estimate_noise(ds, cfg):
    n = estimate_noise(ds, cfg)
    assert n["sigma0"] > 0 and n["cv"] > 0 and np.isclose(n["c"], n["sigma0"] / n["cv"])


def test_bucket_ppm():
    assert bucket_ppm("B2_4138") == pytest.approx(2.4138)
    assert bucket_ppm("B0_8820") == pytest.approx(0.8820)
    assert bucket_ppm("B3_0216") == pytest.approx(3.0216)


def test_tidy_csv_round_trip(tmp_path, ds):
    path = tmp_path / "t.csv"
    save_tidy_csv(ds, path)
    back = load_tidy_csv(path)
    assert back.buckets == ds.buckets
    assert np.allclose(back.X, ds.X)
    assert list(back.group) == list(ds.group) and list(back.batch) == list(ds.batch)
    assert load_dataset(path).X.shape == ds.X.shape


def test_legacy_workbook_layout(tmp_path, ds):
    """Workbook with three header rows, Samplecode/Condition columns and buckets."""
    codes = [f"FM_{'halftreat' if b == 'Exp1' else 'miceSerum2'}_{i}" for i, b in enumerate(ds.batch)]
    header = ["Samplecode", "Condition"] + ds.buckets
    body = [[c, g] + list(row) for c, g, row in zip(codes, ds.group, ds.X)]
    rows = [["title"] + [None] * (len(header) - 1), [None] * len(header), header] + body
    path = tmp_path / "legacy.xlsx"
    pd.DataFrame(rows).to_excel(path, sheet_name="Quantification", header=False, index=False)
    back = load_legacy_xlsx(path)
    assert back.buckets == ds.buckets and np.allclose(back.X, ds.X)
    assert list(back.batch) == list(ds.batch) and list(back.group) == list(ds.group)


def test_dataset_validation():
    X = np.ones((4, 2))
    with pytest.raises(ValueError, match="two batches"):
        Dataset(X, np.arange(4).astype(str), np.array(["a"] * 4), np.array(["b1", "b2", "b3", "b3"]), ["B1_0001", "B1_0002"])
    Xn = X.copy(); Xn[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        Dataset(Xn, np.arange(4).astype(str), np.array(["a"] * 4), np.array(["b1", "b1", "b2", "b2"]), ["B1_0001", "B1_0002"])
