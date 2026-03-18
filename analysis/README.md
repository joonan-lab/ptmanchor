# Analysis Scripts

Scripts to reproduce the analyses presented in the ptmanchor manuscript.

**Prerequisites:** Install the ptmanchor package first (see [Installation](../README.md#installation)). All scripts below import from ptmanchor.

## Directory Structure

```
analysis/
├── benchmark/
│   └── benchmark_vs_competitors.py   # Synthetic data benchmark
└── cptac/
    ├── 1_run_pipeline.py             # CPTAC PTM correction pipeline
    ├── 2_kinase_substrate.py         # Kinase-substrate enrichment analysis
    └── 3_ksea_comparison.py          # KSEA method comparison
```

---

## [1] Synthetic Benchmark

We evaluated ptmanchor against five competing correction methods (Subtract, MSstatsPTM, msqrob2PTM, ANOVA residualization, and Raw t-test) using synthetic datasets. Six scenarios were designed to test different signal structures, including varying protein-PTM coupling strengths, noise levels, sample sizes, and heterogeneous lambda distributions.

### Run benchmark

```bash
python analysis/benchmark/benchmark_vs_competitors.py --output-dir results/benchmark
```

The script generates synthetic paired and unpaired datasets (5,000 PTM sites, 30 tumor + 30 normal samples), applies all correction methods, and computes precision-recall AUC, ROC AUC, observed FDR, and false-positive removal rate for each scenario.

**Requirements:** R with `limma` and `msqrob2` packages (accessed via `rpy2`).

---

## [2] CPTAC Pan-Cancer PTM Correction

We applied ptmanchor to paired tumor-normal data from seven CPTAC cancer cohorts (BRCA, CCRCC, COAD, GBM, HNSCC, LSCC, LUAD, OV, PDAC, UCEC) covering phosphoproteomics and acetylproteomics modalities. For each cohort, the pipeline exports data via the CPTAC Python API, runs per-site protein-anchored correction, and classifies sites as PTM-specific or protein-driven.

### Run CPTAC pipeline

```bash
python analysis/cptac/1_run_pipeline.py
```

The script performs three steps: (1) export CPTAC cohort data to standardized TSV format, (2) run ptmanchor correction for each cohort and modality, and (3) build a cross-cohort summary. Output is written to `results/cptac_<cohort>_ptm_correction/` directories.

**Requirements:** [`cptac`](https://pypi.org/project/cptac/) Python package (`pip install cptac`).

---

## [3] Gene-Level Enrichment and Kinase-Substrate Enrichment Analysis (KSEA)

After protein correction, we performed two complementary enrichment analyses. First, gene-level phosphosite enrichment identifies proteins whose own phosphosites are preferentially represented among true PTM-specific increases using Fisher's exact test. Second, kinase-substrate enrichment analysis (KSEA) identifies upstream kinases with recurrent substrate activation across cohorts using curated kinase-substrate relationships from PhosphoSitePlus.

### Run KSEA

```bash
python analysis/cptac/2_kinase_substrate.py \
    --psp-file data/Kinase_Substrate_Dataset.gz
```

This script depends on the output of the CPTAC pipeline (step [2]). It maps kinase-substrate pairs to corrected PTM sites and computes per-cohort and cross-cohort enrichment statistics.

### PhosphoSitePlus data

The Kinase-Substrate Dataset is required but not included in this repository due to licensing:

1. Visit https://www.phosphosite.org/staticDownloads
2. Download `Kinase_Substrate_Dataset.gz`
3. Place it in `analysis/data/`

---

## [4] KSEA Method Comparison

To assess the impact of protein correction on kinase inference, we compared KSEA results across three correction strategies: no correction (raw), global subtraction, and ptmanchor per-site correction. This analysis identifies kinases that are inflated by protein-level confounding (raw-only false positives) and kinases recovered exclusively by per-site correction (ptmanchor-only hits).

### Run KSEA comparison

```bash
python analysis/cptac/3_ksea_comparison.py \
    --psp-file data/Kinase_Substrate_Dataset.gz
```

This script depends on the output of the CPTAC pipeline (step [2]).

---

## Command-Line Options

All scripts support `--help` for detailed usage information:

```bash
python analysis/benchmark/benchmark_vs_competitors.py --help
python analysis/cptac/1_run_pipeline.py --help
python analysis/cptac/2_kinase_substrate.py --help
python analysis/cptac/3_ksea_comparison.py --help
```
