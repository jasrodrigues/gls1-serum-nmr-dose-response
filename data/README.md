# Input data

The study data are not distributed with the code. Place the bucket table in this folder; its files are ignored by git (see `.gitignore`).

## Format (`.csv`)

One row per serum sample. Three descriptive columns, then one column per spectral bucket:

| column | content |
|---|---|
| `sample` | sample identifier |
| `group` | treatment group: `Control`, `G14` (GLS1 dose 1, 0.14 ng/µL) or `G28` (GLS1 dose 2, 0.28 ng/µL) |
| `batch` | batch label, exactly two values (for example `Exp1`, `Exp2`) |
| `B2_4690`, `B2_4757`, ... | bucket intensities |

```
sample,group,batch,B0_8820,B1_2395,B2_4690,...
S01,Control,Exp1,412.3,125.0,880.1,...
S02,Control,Exp1,398.7,131.2,845.9,...
```

- A bucket name encodes its chemical shift: `B2_4138` is 2.4138 ppm (the digit after `B` is the integer part, the suffix the decimals).
- Intensities must be finite numbers. Negative values are allowed (integrals after baseline correction can be negative).
- The signature buckets listed in `configs/signatures.json` must be present.
- Use the same bucket table that was analyzed in the manuscript. The Methods state that the NMRProcFlow output was normalized by constant sum; say in the deposited file which version (raw or normalized) it contains.

## Group labels and doses

The mapping from group label to dose (ng/µL) is `Config.dose_map` in `src/gls1dr/config.py`. If your labels differ, edit it there; nothing else assumes the names.

## Legacy workbook

An `.xlsx` in the NMRProcFlow layout (sheet `Quantification`, three header rows, columns `Samplecode` and `Condition`) is read directly. The batch is not stored in that layout, so `io.load_legacy_xlsx` infers it from the sample code (`halftreat` gives `Exp1`, anything else `Exp2`). Check that rule against your files, or convert once and edit the `batch` column:

```bash
gls1dr --input workbook.xlsx --export-tidy data/bucket_table.csv
```

## Synthetic example

`gls1dr --demo --out results_demo` writes `demo_input_synthetic.csv`, a simulated table with the study design, so that the format can be inspected and the pipeline tried. It is not real data.
