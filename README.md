# ptmanchor

**Protein-anchored correction for multi-PTM proteomics cohorts**

A change in PTM-site abundance can reflect a PTM-specific change, a change in parent-protein abundance, or both. `ptmanchor` estimates site-specific protein–PTM coupling and separates the estimated protein contribution from the PTM-specific change. Protein adjustment addresses PTM-specific changes and complements, rather than replaces, the unadjusted PTM readout.

[Quick start](#quick-start) · [Input format](#input-format) · [Results](#output-format) · [Method](#method-overview) · [Advanced usage](#advanced-usage)

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

R is optional; see [Backend and reproducibility](#backend-and-reproducibility) for setup and reproduction details.

## Quick Start

### Try the synthetic demo

The repository includes a deterministic paired dataset generator. From the repository root:

```bash
python examples/generate_demo_data.py
ptmanchor \
  --manifest examples/demo_data/manifest.tsv \
  --protein-file examples/demo_data/protein.tsv \
  --output-dir examples/demo_results \
  --alternative two-sided
```

This exercises protein matching, paired per-site regression, EB and λ shrinkage,
BH correction, and direction-specific output files without downloading external data.

### Run your own data

Prepare the following files:

- **PTM files**: quantification tables for each modality (e.g., `phospho_ratio.tsv`, `acetyl_ratio.tsv`)
- **Protein file** (`global_proteome.tsv`): global proteome quantification used as the protein-level reference
- **Manifest TSV** (`modalities.tsv`): a simple index that lists which PTM files to process (see [Input Format](#input-format))

ptmanchor reads the manifest to find PTM file paths, so you can run multiple modalities (such as phosphoproteomics and acetylproteomics) in a single command — just add rows to the manifest.

Then run:

```bash
ptmanchor \
  --manifest data/modalities.tsv \
  --protein-file data/global_proteome.tsv \
  --output-dir results/corrected \
  --min-pairs 8 \
  --fdr-cutoff 0.05
```

Choose a [test direction](#test-direction) and see [Output Format](#output-format) to read the results.

### Test Direction

By default every site-level test is one-sided for upregulation (`H1: β₀ > 0`). Set
`--alternative` to test the opposite direction (`less`) or both (`two-sided`), which
also populates the corresponding down-regulated classifications:

```bash
ptmanchor --manifest data/modalities.tsv --protein-file data/global_proteome.tsv \
  --output-dir results/corrected --alternative two-sided
```

## Input Format

### Manifest TSV

| modality | ptm_file | enabled |
|----------|----------|---------|
| phospho | data/phospho_ratio.tsv | true |
| acetyl | data/acetyl_ratio.tsv | true |

`enabled` is optional and defaults to true. Relative PTM file paths are resolved
from the working directory first, then from the manifest directory if not found.
Run the examples from the repository root.

### PTM / Protein TSV

Supply preprocessed PTM and protein values on a **common log2 scale with consistent
normalization** (log2 ratios or log2 intensities). The model and default effect-size
threshold (`--min-corrected-delta 0.2`) use this scale. ptmanchor uses values as-is:
it does not normalize, log-transform, or check the input scale. The reported
applications used TMT-based CPTAC data.

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

Supplying `Gene Symbol` enables the third matching tier (see [Protein Matching](#protein-matching)); without it, sites are matched by accession only. Keep the column name `UniProtAccession` even when supplying Ensembl protein identifiers.

### Protein Matching

PTM sites are matched to parent proteins via a 3-level cascade:

1. Exact UniProt accession or Ensembl protein identifier (`ENSP...`), including isoform or version suffixes
2. Canonical identifier after removing UniProt isoform or Ensembl version suffixes
3. Primary gene symbol fallback

For each matching tier, duplicate protein rows are aggregated by their per-sample
median. PTM records retain the peptide-level resolution of the input; a record can
represent one modified residue or a combination of residues.

## Output Format

For each modality, ptmanchor creates a directory (e.g., `results/corrected/phosphoproteomics/`) containing:

- `all_sites.tsv` — full results for every PTM site
- `true_increase_lm.tsv` / `true_decrease_lm.tsv` — direction-specific ptmanchor hits
- `true_hits_lm.tsv` — union of ptmanchor hits in the tested direction(s)
- `true_increase_subtract.tsv` / `true_decrease_subtract.tsv` — direction-specific subtraction hits
- `true_hits_subtract.tsv` — union of subtraction hits in the tested direction(s)

With the default `--alternative greater`, decrease files are empty and `true_hits_lm.tsv` is identical to `true_increase_lm.tsv`. With `--alternative two-sided`, increase and decrease results remain separate.

Key output columns in `all_sites.tsv`:

- `lm_intercept_ptm_specific`: PTM-specific change (β₀) after accounting for protein abundance
- `lm_lambda_protein_dependence`: site-specific protein–PTM coupling coefficient (λ)
- `lm_q_bh`: BH-adjusted p-value
- `is_true_lm_up`, `is_true_lm_down`: direction-specific ptmanchor classifications
- `is_true_lm`: union of significant directions tested
- `protein_driven_lm`: legacy raw-hit/not-corrected-hit flag; see [Result interpretation](#result-interpretation) before using it.

Check `modality_summary.tsv` for each modality’s status, analysis mode, and failure or skip reason; a completed CLI invocation alone does not imply every modality succeeded. It also aggregates hit counts across all modalities, and `run_config.json` records the analysis direction, thresholds, shrinkage settings, and configured EB backend (see [Backend and reproducibility](#backend-and-reproducibility)).

### Result interpretation

With the default increase-oriented test, the manuscript uses these terms:

- **PTM-specific increase**: corrected β₀ ≥ 0.2 and BH-adjusted q ≤ 0.05 at the default thresholds.
- **Not retained after correction**: a raw-up site with an estimable corrected result that does not meet the PTM-specific increase criteria.

An estimable corrected result requires finite β₀ and q and at least the requested
number of paired observations (eight by default). Sites without an estimable result
are excluded from manuscript retention-rate denominators, not interpreted as
protein-driven. Not meeting the increase criteria may reflect a smaller estimated
effect, loss of statistical significance, or both; it does not establish that the
biological change is driven by protein abundance. The legacy `protein_driven_lm`
field must be combined with this testability filter for manuscript-style summaries.

## Method Overview

`ptmanchor` provides 3 tiers of protein-anchored correction:

1. **Subtraction**: `adjusted_delta = PTM_delta - protein_delta` — simple baseline removal assuming fixed λ = 1
2. **Paired linear model (default)**: `PTM_delta ~ intercept + lambda * protein_delta` — estimates the site-specific protein–PTM coupling coefficient (λ) and PTM-specific change (β₀)
3. **Sample-level LM/LMM** (optional): `PTM ~ is_tumor + protein + covariates [+ (1|patient)]` — sample-level regression for unpaired designs or when covariates are needed

The paired model fits tumor–normal differences prepared as described in [Input Format](#input-format). Its intercept
β₀ estimates the PTM change at zero protein change. By default, site-specific λ
estimates are shrunk toward a precision-weighted mean within each cohort–modality
analysis. The intercept and residual variance are then recomputed, followed by
empirical Bayes variance moderation. BH correction is applied separately for each
method within each modality and cohort.

The reported CPTAC analyses used paired data from seven phosphoproteomics cohorts
and three acetylproteomics cohorts (LUAD, LSCC, and UCEC), without fallback analyses.

## Advanced usage

<details>
<summary>Python API</summary>

You can also call ptmanchor functions directly in Python scripts or Jupyter notebooks. After generating the synthetic demo data above:

```python
from ptmanchor import run_manifest

# Build the same complete argument namespace used by the CLI.
from ptmanchor.cli import build_parser

args = build_parser().parse_args([
    "--manifest", "examples/demo_data/manifest.tsv",
    "--protein-file", "examples/demo_data/protein.tsv",
    "--output-dir", "examples/demo_results_api",
    "--alternative", "two-sided",
])
summary_tsv, summary_txt, config_json = run_manifest(args)
```

For lower-level modeling, prepare `raw_delta` and `protein_delta` as NumPy arrays
of shape `(PTM records, paired samples)`, containing tumor-minus-normal log2
differences, then call:

```python
from ptmanchor import paired_lm_intercept_test
intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
    raw_delta, protein_delta, min_n=8
)
```

</details>

<details>
<summary>Full CLI reference</summary>

```
ptmanchor [OPTIONS]

Required inputs:
  --manifest FILE          TSV with modality, ptm_file; enabled is optional
  --protein-file FILE      Global proteome TSV (see Input Format)

Output:
  --output-dir DIR         Output directory (required; no default)

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
  --fallback-min-tumor INT
  --fallback-min-normal INT
  --force-unpaired-if-paired
  --enable-detection-fallback
  --force-detection-fallback
  --min-detection-delta FLOAT
  --min-dual-group-sites-detection INT

Shrinkage (enabled by default):
  --no-eb                 Disable empirical Bayes variance moderation
  --no-lambda-shrinkage    Disable paired-model lambda shrinkage

Other:
  --version                Show version and exit
```

</details>

<details>
<summary>Fallback strategies</summary>

- **Paired-to-unpaired fallback** (opt-in): switches the entire modality when too few sites meet the paired-observation threshold. It uses Welch tests for raw and subtraction results and sample-level regression for ptmanchor; it is not a per-site replacement for missing pairs.
- **Detection-rate fallback** (opt-in): Fisher exact tests on presence/absence for sparse data, with a protein-adjusted detection-frequency difference as the effect measure.

If no tumor–normal pairs exist, the primary analysis uses the unpaired model.

</details>

<details>
<summary>Preparing CPTAC data</summary>

The reported analyses used processed quantification tables from the `umich` source
in the `cptac` Python package (v1.5.14). These data are also available through the
[Proteomic Data Commons](https://pdc.cancer.gov/pdc/cptac-pancancer).

Install the pinned CPTAC loader used for the manuscript example with `pip install -e ".[reproduce]"`.

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
        out.insert(0, "ID", meta[["Name", "Site", "Peptide", "Database_ID"]].astype(str).agg("_".join, axis=1).values)
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
model selects its own pairs, while unpaired and detection analyses, when selected, use the available
tumor and normal samples. For exact manuscript reproduction, see [Backend and reproducibility](#backend-and-reproducibility).

</details>

## Backend and reproducibility

Empirical Bayes variance moderation uses `limma::squeezeVar` through rpy2 when
available. A Python moment-matching fallback is used if R/limma is unavailable or
its call fails. **The two backends are not guaranteed to produce identical results.**
Exact manuscript reproduction requires the same source tables, R/limma environment,
and analysis settings, plus the estimability filters in [Result interpretation](#result-interpretation).

```bash
pip install -e ".[r]"
R -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager"); BiocManager::install("limma")'
```

`run_config.json` records the configured/available backend. In v1.1.2 it does not
audit every individual R call; an R-call failure can trigger the Python fallback.
When exact reproduction is required, verify successful R/limma execution.

Documentation note for v1.1.2: this guidance supersedes the README in the release
tag; the tagged source code is unchanged.

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v --cov=ptmanchor
```

## Citation

If you use ptmanchor in your research, please cite:

> Jeong et al. (2026). ptmanchor: per-site protein-anchored correction refines PTM-specific quantification in multi-cohort cancer proteomics. [Under review]

## Contact

Joon-Yong An — joonan30@korea.ac.kr

## License

MIT. See [LICENSE](LICENSE) for details.
