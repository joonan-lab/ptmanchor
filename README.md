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
- **True PTM increase**: significant after correction (FDR < cutoff, effect > threshold)
- **Protein-driven**: significant before but not after correction
- **Null**: not significant in either analysis

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

Tab-separated with row identifiers followed by sample columns. Tumor samples end with `-T`, normal samples end with `-N`:

```
ID              Accession  Gene   Patient1-T  Patient1-N  Patient2-T  Patient2-N
AAAS_S495       Q9NRG9     AAAS   0.32        -0.15       0.78        0.11
ABI1_S216       Q8IZP0     ABI1   1.05        0.42        0.63        -0.08
```

The protein TSV follows the same format. Both PTM and protein values must be on the same scale (log2 recommended).

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

## Reproducing Manuscript Analyses

The `analysis/` directory contains scripts to reproduce all analyses in the manuscript. See [`analysis/README.md`](analysis/README.md) for details.

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
