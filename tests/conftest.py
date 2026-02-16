"""Shared fixtures for ptmanchor tests."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def rng():
    """Reproducible random generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def paired_samples():
    """10 paired tumor-normal sample names."""
    bases = [f"P{i:03d}" for i in range(1, 11)]
    tumor = [f"{b}-T" for b in bases]
    normal = [f"{b}-N" for b in bases]
    return tumor + normal, bases


@pytest.fixture()
def synthetic_ptm_tsv(tmp_path, rng, paired_samples):
    """Create a small PTM TSV file with known properties.

    Returns (path, n_sites, sample_cols).
    """
    samples, _bases = paired_samples
    n_sites = 50
    n_samples = len(samples)

    rows = []
    for i in range(n_sites):
        row = {
            "ID": f"site_{i}",
            "UniProtAccession": f"P{10000 + i}",
            "Gene Symbol": f"GENE{i}",
            "Description": f"Description {i}",
        }
        for j, s in enumerate(samples):
            if rng.random() < 0.1:
                row[s] = np.nan  # ~10% missing
            else:
                row[s] = float(rng.normal(0, 1))
        rows.append(row)

    df = pd.DataFrame(rows)
    path = tmp_path / "ptm_test.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path, n_sites, samples


@pytest.fixture()
def synthetic_protein_tsv(tmp_path, rng, paired_samples):
    """Create a small global-proteome TSV file.

    Returns (path, n_proteins, sample_cols).
    """
    samples, _bases = paired_samples
    n_proteins = 30
    n_samples = len(samples)

    rows = []
    for i in range(n_proteins):
        row = {
            "ID": f"prot_{i}",
            "UniProtAccession": f"P{10000 + i}",
            "Gene Symbol": f"GENE{i}",
            "Description": f"Protein {i}",
        }
        for s in samples:
            if rng.random() < 0.05:
                row[s] = np.nan
            else:
                row[s] = float(rng.normal(0, 0.5))
        rows.append(row)

    df = pd.DataFrame(rows)
    path = tmp_path / "protein_test.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path, n_proteins, samples


@pytest.fixture()
def synthetic_manifest_tsv(tmp_path, synthetic_ptm_tsv):
    """Create a manifest TSV pointing to the synthetic PTM file."""
    ptm_path, _, _ = synthetic_ptm_tsv
    df = pd.DataFrame({
        "modality": ["phospho"],
        "ptm_file": [str(ptm_path)],
        "enabled": ["true"],
    })
    path = tmp_path / "manifest.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture()
def mock_args(tmp_path, synthetic_manifest_tsv, synthetic_protein_tsv):
    """Argparse namespace mimicking CLI arguments for a small test run."""
    protein_path, _, _ = synthetic_protein_tsv
    out_dir = tmp_path / "output"
    out_dir.mkdir(exist_ok=True)

    return argparse.Namespace(
        manifest=str(synthetic_manifest_tsv),
        protein_file=str(protein_path),
        output_dir=str(out_dir),
        min_pairs=3,
        min_tumor=3,
        min_normal=3,
        fdr_cutoff=0.05,
        min_corrected_delta=0.2,
        top_n=10,
        sample_meta_file=None,
        sample_meta_sheet=None,
        sample_id_col="Sample.ID",
        patient_id_col=None,
        covariates="",
        enable_sample_lm=False,
        enable_sample_lmm=False,
        max_sites_sample_lm=0,
        max_sites_sample_lmm=0,
        lmm_maxiter=50,
        enable_paired_to_unpaired_fallback=False,
        min_paired_testable_sites=1,
        fallback_min_tumor=0,
        fallback_min_normal=0,
        force_unpaired_if_paired=False,
        enable_detection_fallback=False,
        force_detection_fallback=False,
        min_detection_delta=0.10,
        min_dual_group_sites_detection=100,
    )
