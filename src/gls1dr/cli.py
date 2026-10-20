"""Command-line interface: `gls1dr --input table.csv --out results --figures`."""
import argparse
import os
import tempfile
from dataclasses import replace
from importlib import resources

from .config import Config
from .io import load_dataset, save_tidy_csv
from .pipeline import run_analysis
from .signatures import load_signatures
from .synthetic import make_example


def _default_signatures():
    return str(resources.files("gls1dr").joinpath("signatures_default.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gls1dr", description="Dose-response analysis of serum 1H-NMR signatures after GLS1 therapy.")
    ap.add_argument("--input", help="bucket table: .csv (tidy format, see data/README.md) or .xlsx (legacy workbook)")
    ap.add_argument("--signatures", help="signature definition file (default: the packaged configs/signatures.json)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--figures", action="store_true", help="also write Fig5C and FigS9 (needs matplotlib)")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic dataset instead of --input")
    ap.add_argument("--export-tidy", metavar="CSV", help="convert --input to the tidy CSV format and exit")
    ap.add_argument("--n-boot", type=int, help="bootstrap resamples (default 2000)")
    ap.add_argument("--n-mc", type=int, help="Monte Carlo draws (default 20000)")
    ap.add_argument("--n-verify", type=int, help="parameter sets for the model checks (default 20000)")
    args = ap.parse_args(argv)

    cfg = Config()
    over = {k: v for k, v in (("n_boot", args.n_boot), ("n_mc", args.n_mc), ("n_verify", args.n_verify)) if v is not None}
    cfg = replace(cfg, **over)
    spec = load_signatures(args.signatures or _default_signatures())

    if args.demo:
        ds = make_example(spec, cfg)
        os.makedirs(args.out, exist_ok=True)
        save_tidy_csv(ds, os.path.join(args.out, "demo_input_synthetic.csv"))
        print("Running on SYNTHETIC data (not the study data).")
    elif args.input:
        ds = load_dataset(args.input)
    else:
        ap.error("give --input FILE or use --demo")

    if args.export_tidy:
        save_tidy_csv(ds, args.export_tidy)
        print(f"wrote {args.export_tidy}")
        return 0
    run_analysis(ds, spec, cfg, outdir=args.out, figures=args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
