"""Reading and validating the bucket table."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Bucket intensities with one row per serum sample."""
    X: np.ndarray        # samples x buckets
    sample: np.ndarray   # sample identifiers
    group: np.ndarray    # treatment group label (a key of Config.dose_map)
    batch: np.ndarray    # batch label (exactly two batches are supported)
    buckets: list        # bucket names, e.g. "B2_4690" = 2.4690 ppm

    def __post_init__(self):
        n = self.X.shape[0]
        if not (len(self.sample) == len(self.group) == len(self.batch) == n):
            raise ValueError("sample, group, batch and X must have the same number of rows")
        if self.X.shape[1] != len(self.buckets):
            raise ValueError("number of bucket names does not match the columns of X")
        if len(set(self.batch)) != 2:
            raise ValueError(f"exactly two batches are supported, found {sorted(set(self.batch))}")
        if not np.isfinite(self.X).all():
            raise ValueError("the bucket table contains missing or non-finite values")


def bucket_ppm(bucket):
    """'B2_4138' -> 2.4138 ppm (integer part after the first character, decimals in the suffix)."""
    head, num = bucket.split("_")
    return int(head[1:]) + float(num) / 10 ** len(num)


def load_tidy_csv(path):
    """Canonical format: columns sample, group, batch, then one column per bucket."""
    df = pd.read_csv(path)
    missing = [c for c in ("sample", "group", "batch") if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    buckets = [c for c in df.columns if c not in ("sample", "group", "batch")]
    return Dataset(X=df[buckets].astype(float).to_numpy(), sample=df["sample"].astype(str).to_numpy(),
                   group=df["group"].astype(str).to_numpy(), batch=df["batch"].astype(str).to_numpy(),
                   buckets=buckets)


def save_tidy_csv(ds, path):
    df = pd.DataFrame(ds.X, columns=ds.buckets)
    df.insert(0, "batch", ds.batch)
    df.insert(0, "group", ds.group)
    df.insert(0, "sample", ds.sample)
    df.to_csv(path, index=False)


def load_legacy_xlsx(path, sheet="Quantification", batch_rule=None):
    """Workbook exported from the NMRProcFlow pipeline: three header rows, then one row per sample,
    with columns Samplecode, Condition and the buckets.

    The batch is not stored in that layout; by default it is inferred from the sample code
    ("halftreat" -> Exp1, otherwise Exp2). Pass batch_rule to change this.
    """
    if batch_rule is None:
        batch_rule = lambda code: "Exp1" if "halftreat" in code else "Exp2"
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header = raw.iloc[2].tolist()
    data = raw.iloc[3:].copy()
    data.columns = header
    data = data[data["Samplecode"].notna()].reset_index(drop=True)
    buckets = [c for c in header if isinstance(c, str) and c not in ("Samplecode", "Condition")]
    return Dataset(X=data[buckets].astype(float).to_numpy(), sample=data["Samplecode"].astype(str).to_numpy(),
                   group=data["Condition"].astype(str).to_numpy(),
                   batch=np.array([batch_rule(str(c)) for c in data["Samplecode"]]), buckets=buckets)


def load_dataset(path):
    """Read a .csv (tidy format) or .xlsx (legacy workbook)."""
    p = str(path).lower()
    if p.endswith(".csv"):
        return load_tidy_csv(path)
    if p.endswith((".xlsx", ".xls")):
        return load_legacy_xlsx(path)
    raise ValueError("input must be .csv (tidy format) or .xlsx (legacy workbook)")
