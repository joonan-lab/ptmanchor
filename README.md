# ptmanchor

**Protein-anchored correction for multi-PTM proteomics cohorts**

In quantitative PTM proteomics, changes in parent-protein abundance confound PTM-site measurements, generating false positives. `ptmanchor` regresses out the protein-level effect per site, separating true PTM-specific regulation from protein-driven artifacts.

## Installation

```bash
pip install ptmanchor
```

For development:

```bash
git clone https://github.com/jooyoung-an/ptmanchor.git
cd ptmanchor
pip install -e ".[test]"
```

## Quick Start

### CLI

```bash
ptmanchor \
  --manifest data/modalities.tsv \
  --protein-file data/global_proteome.tsv \
  --output-dir results/corrected \
  --min-pairs 8 \
  --fdr-cutoff 0.05
```

### Python API

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

`ptmanchor` applies a 3-tier protein-anchored correction:

1. **Subtraction**: `adjusted_delta = PTM_delta - protein_delta` (simple baseline removal)
2. **Paired linear model**: `PTM_delta ~ intercept + lambda * protein_delta` (site-wise OLS on paired tumor-normal deltas; the intercept captures the PTM-specific effect)
3. **Sample-level LM/LMM**: `PTM ~ is_tumor + protein + covariates [+ (1|patient)]` (sample-level regression with optional mixed effects)

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
  --protein-file FILE      Global proteome TSV (log2-ratio or abundance)

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

Tab-separated with:
- Row identifier columns (ID, UniProt accession, gene symbol, description)
- Sample columns named with `-T` (tumor) or `-N` (normal) suffixes, or `RE-` prefix

## Output Format

Per-modality output TSV includes:
- Site identifiers and protein match info
- Raw, subtraction-corrected, and LM-corrected statistics (delta, p-value, q-value)
- Binary hit classifications (`is_raw_up`, `is_true_subtract`, `is_true_lm`, `protein_driven_*`)
- Optional sample-level model results

A `modality_summary.tsv` aggregates hit counts across all modalities.

## Testing

```bash
pytest tests/ -v --cov=ptmanchor
```

## License

MIT. See [LICENSE](LICENSE) for details.
