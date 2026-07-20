# ptmanchor

**Protein-anchored correction for multi-PTM proteomics cohorts**

In quantitative PTM proteomics, changes in parent-protein abundance confound PTM-site measurements, generating false positives. `ptmanchor` regresses out the protein-level effect per site, separating true PTM-specific regulation from protein-driven artifacts.

## Prerequisites

- Python >= 3.10
- OS: macOS, Linux, Windows
- Dependencies: numpy, pandas, scipy, statsmodels, openpyxl (auto-installed)

## Installation

```bash
git clone https://github.com/joonan-lab/ptmanchor.git
cd ptmanchor
pip install -e .
```

To run tests, install with test dependencies:

```bash
pip install -e ".[test]"
```

### Optional: R backend for variance shrinkage

Empirical Bayes variance shrinkage calls `limma::squeezeVar` through rpy2 when R is
available, and otherwise falls back to an equivalent moment-matching implementation in
Python. Both paths run without further configuration, but they do not agree exactly: the
Python fallback is slightly more conservative, so hit counts differ by a few percent.
**The results reported in the manuscript were produced with the R backend**, so install it
if you intend to reproduce them.

```bash
pip install -e ".[r]"
R -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager"); BiocManager::install("limma")'
```

Which path was used is recorded in `run_config.json` for every run.

## Quick Start

### CLI

Prepare the following files:
- **PTM files**: quantification tables for each modality (e.g., `phospho_ratio.tsv`, `acetyl_ratio.tsv`)
- **Protein file** (`global_proteome.tsv`): global proteome quantification used as the protein-level reference
- **Manifest TSV** (`modalities.tsv`): a simple index that lists which PTM files to process (see [Input Format](#input-format))

ptmanchor reads the manifest to find PTM file paths, so you can run multiple modalities (phospho, acetyl, ubiquitin, etc.) in a single command — just add rows to the manifest.

Then run:

```bash
ptmanchor \
  --manifest data/modalities.tsv \
  --protein-file data/global_proteome.tsv \
  --output-dir results/corrected \
  --min-pairs 8 \
  --fdr-cutoff 0.05
```

See [Input Format](#input-format) below for file specifications.

### Python API

You can also call ptmanchor functions directly in Python scripts or Jupyter notebooks:

```python
from ptmanchor import run_manifest, paired_lm_intercept_test

# Run full pipeline from a namespace/args object
summary_tsv, summary_txt, config_json = run_manifest(args)

# Or use modeling functions directly
intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
    raw_delta, protein_delta, min_n=8
)
```

## Method Overview

`ptmanchor` provides 3 tiers of protein-anchored correction:

1. **Subtraction**: `adjusted_delta = PTM_delta - protein_delta` — simple baseline removal assuming fixed λ = 1
2. **Paired linear model (default)**: `PTM_delta ~ intercept + lambda * protein_delta` — estimates a per-site protein contribution coefficient (λ) and isolates the PTM-specific intercept
3. **Sample-level LM/LMM** (optional): `PTM ~ is_tumor + protein + covariates [+ (1|patient)]` — sample-level regression for unpaired designs or when covariates are needed

Each site is classified as:
- **True PTM change**: significant after correction (FDR < cutoff, |effect| > threshold)
- **Protein-driven**: significant before but not after correction
- **Null**: not significant in either analysis

### Test Direction

By default every site-level test is one-sided for upregulation (`H1: β₀ > 0`). Set
`--alternative` to test the opposite direction (`less`) or both (`two-sided`), which
also populates the corresponding down-regulated classifications:

```bash
ptmanchor --manifest data/modalities.tsv --protein-file data/global_proteome.tsv \
  --alternative two-sided
```

### Protein Matching

PTM sites are matched to parent proteins via a 3-level cascade:
1. Exact UniProt accession
2. Canonical accession (isoform-stripped)
3. Gene symbol fallback

### Fallback Strategies

- **Paired-to-unpaired fallback**: when too few paired samples exist
- **Detection-rate fallback**: Fisher exact test on presence/absence for sparse data

## CLI Reference

```
ptmanchor [OPTIONS]

Required inputs:
  --manifest FILE          TSV with columns: modality, ptm_file, enabled
  --protein-file FILE      Global proteome TSV (log2 recommended)

Output:
  --output-dir DIR         Output directory (default: results/multimodal_ptm_correction)

Thresholds:
  --min-pairs INT          Minimum paired observations per site (default: 8)
  --min-tumor INT          Minimum tumor samples for LM/LMM (default: 20)
  --min-normal INT         Minimum normal samples (default: 8)
  --fdr-cutoff FLOAT       FDR significance cutoff (default: 0.05)
  --min-corrected-delta F  Minimum adjusted effect size (default: 0.2)
  --top-n INT              Top N LM hits to save (default: 50)

Test direction:
  --alternative STR        greater (default) | less | two-sided

Sample metadata:
  --sample-meta-file FILE  Sample metadata (csv/tsv/xlsx)
  --sample-meta-sheet STR  Excel sheet name
  --sample-id-col STR      Sample ID column (default: Sample.ID)
  --patient-id-col STR     Patient ID column
  --covariates STR         Comma-separated covariate columns

Advanced models:
  --enable-sample-lm       Run sample-level OLS
  --enable-sample-lmm      Run sample-level mixed model
  --max-sites-sample-lm N  Limit sample LM to N sites
  --max-sites-sample-lmm N Limit sample LMM to N sites
  --lmm-maxiter INT        Max LMM iterations (default: 100)

Fallbacks:
  --enable-paired-to-unpaired-fallback
  --min-paired-testable-sites INT
  --force-unpaired-if-paired
  --enable-detection-fallback
  --force-detection-fallback
  --min-detection-delta FLOAT
  --min-dual-group-sites-detection INT

Other:
  --version                Show version and exit
```

## Input Format

### Manifest TSV

| modality | ptm_file | enabled |
|----------|----------|---------|
| phospho | data/phospho_ratio.tsv | true |
| acetyl | data/acetyl_ratio.tsv | true |

### PTM / Protein TSV

ptmanchor does not perform any internal normalization or log transformation — values are used as-is in the regression model. PTM and protein matrices must share the **same scale and normalization**. The tool has been validated on TMT-based quantification data from CPTAC.

**We recommend log2-transformed values** (e.g., log2 ratio or log2 intensity), since the default effect-size threshold (`--min-corrected-delta 0.2`) is calibrated on the log2 scale. If using a different scale, adjust this threshold accordingly.

Tab-separated with metadata columns followed by sample columns. Tumor samples end with `-T`, normal samples end with `-N`.

**PTM TSV** — `ID` and `UniProtAccession` are required; `Gene Symbol` and `Description` are optional:

```
ID              UniProtAccession  Gene Symbol  Patient1-T  Patient1-N  Patient2-T  Patient2-N
AAAS_S495       Q9NRG9            AAAS         0.32        -0.15       0.78        0.11
ABI1_S216       Q8IZP0            ABI1         1.05        0.42        0.63        -0.08
```

**Protein TSV** — the accession is read from `ID` itself, so only `ID` is required; `Gene Symbol` and `Description` are optional:

```
ID              Gene Symbol  Patient1-T  Patient1-N  Patient2-T  Patient2-N
Q9NRG9          AAAS         0.11        -0.04       0.25        0.08
Q8IZP0          ABI1         0.40        0.19        0.22        -0.02
```

Supplying `Gene Symbol` enables the third matching tier (see [Protein Matching](#protein-matching)); without it, sites are matched by accession only. Both PTM and protein values must be on the same scale (log2 recommended).

## Output Format

For each modality, ptmanchor creates a directory (e.g., `results/corrected/phosphoproteomics/`) containing:
- `all_sites.tsv` — full results for every PTM site
- `true_increase_lm.tsv` — sites classified as true PTM-specific increases
- `true_increase_subtract.tsv` — sites passing subtraction-based correction

Example rows from `all_sites.tsv`:

```
ID              Accession  Gene   lm_intercept  lm_lambda  lm_q_bh   is_true_lm  protein_driven_lm
AAAS_S495       Q9NRG9     AAAS   0.08          0.91       0.82      False       False
ABI1_S216       Q8IZP0     ABI1   0.65          0.34       0.001     True        False
CDK1_T161       P06493     CDK1   0.92          0.12       1.2e-05   True        False
MKI67_S1031     P46013     MKI67  0.03          1.08       0.91      False       True
```

Key output columns:
- `lm_intercept`: PTM-specific effect (beta) after removing protein contribution
- `lm_lambda`: estimated protein contribution coefficient per site
- `lm_q_bh`: BH-adjusted p-value
- `is_true_lm`: True if PTM-specific increase is significant
- `protein_driven_lm`: True if signal was significant before correction but not after

A `modality_summary.tsv` aggregates hit counts across all modalities.

## Manuscript

In the accompanying manuscript, ptmanchor was applied to paired tumor-normal CPTAC cohorts
obtained through the [`cptac`](https://pypi.org/project/cptac/) Python package (v1.5.14); the
same quantification tables are available from the
[Proteomic Data Commons](https://pdc.cancer.gov/pdc/cptac-pancancer). Kinase-substrate
annotations were taken from PhosphoSitePlus. See the manuscript for the full analysis
description.

The version of this repository as submitted for initial review is tagged
[`v1.0.0`](https://github.com/joonan-lab/ptmanchor/releases/tag/v1.0.0).

### Example: preparing CPTAC data

ptmanchor takes any PTM and protein matrix in the format above; CPTAC is simply the source
used in the manuscript. The `cptac` package returns samples as rows and sites as columns, so
the tables need to be transposed and the identifier columns flattened. For one cohort:

```python
import cptac, pandas as pd

ds = cptac.Ucec()

def export(df, is_ptm):
    df = df.T                                   # samples x sites -> sites x samples
    meta = df.index.to_frame(index=False)       # MultiIndex: Name, Site, Peptide, Database_ID
    out = df.reset_index(drop=True)
    # normal samples carry a ".N" suffix; everything else is tumor
    out.columns = [c[:-2] + "-N" if str(c).endswith(".N") else str(c) + "-T" for c in out.columns]
    out.insert(0, "Gene Symbol", meta["Name"].values)
    if is_ptm:
        out.insert(0, "UniProtAccession", meta["Database_ID"].values)
        out.insert(0, "ID", (meta["Name"].astype(str) + "_" + meta["Site"].astype(str)).values)
    else:
        out.insert(0, "ID", meta["Database_ID"].values)
    return out

export(ds.get_dataframe("phosphoproteomics", "umich"), True).to_csv("phospho.tsv", sep="\t", index=False)
export(ds.get_dataframe("proteomics", "umich"), False).to_csv("proteome.tsv", sep="\t", index=False)
pd.DataFrame({"modality": ["phosphoproteomics"], "ptm_file": ["phospho.tsv"],
              "enabled": [True]}).to_csv("manifest.tsv", sep="\t", index=False)
```

```bash
ptmanchor --manifest manifest.tsv --protein-file proteome.tsv \
  --output-dir results/ucec --min-pairs 8 --fdr-cutoff 0.05
```

Keep every tumor and normal sample rather than pre-filtering to matched pairs: the paired
model selects its own pairs, while the unpaired and detection fallbacks use the remaining
samples. With the R backend installed this reproduces the manuscript's UCEC phosphoproteome
counts exactly.

## Testing

```bash
pytest tests/ -v --cov=ptmanchor
```

## Citation

If you use ptmanchor in your research, please cite:

> Jeong et al. (2026). ptmanchor: protein-anchored correction reveals PTM-specific kinase regulation across multi-cohort cancer proteomics. [Under review]

## Contact

Joon-Yong An — joonan30@korea.ac.kr

## License

MIT. See [LICENSE](LICENSE) for details.
