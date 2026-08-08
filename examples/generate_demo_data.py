"""Generate a deterministic paired PTM/protein dataset for a CLI smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="examples/demo_data")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260808)
    n_sites, n_pairs = 24, 10
    accessions = [f"P{i:05d}" for i in range(1, n_sites + 1)]
    genes = [f"DEMO{i:02d}" for i in range(1, n_sites + 1)]
    protein = {"ID": accessions, "Gene Symbol": genes}
    ptm = {
        "ID": [f"{g}_S100" for g in genes],
        "UniProtAccession": accessions,
        "Gene Symbol": genes,
    }

    beta = np.r_[np.full(6, 0.8), np.full(6, -0.8), np.zeros(12)]
    coupling = np.linspace(0.3, 1.1, n_sites)
    for pair in range(1, n_pairs + 1):
        normal_protein = rng.normal(0, 0.15, n_sites)
        protein_delta = rng.normal(0.5, 0.25, n_sites)
        normal_ptm = rng.normal(0, 0.2, n_sites)
        ptm_delta = beta + coupling * protein_delta + rng.normal(0, 0.12, n_sites)
        protein[f"P{pair:03d}-N"] = normal_protein
        protein[f"P{pair:03d}-T"] = normal_protein + protein_delta
        ptm[f"P{pair:03d}-N"] = normal_ptm
        ptm[f"P{pair:03d}-T"] = normal_ptm + ptm_delta

    pd.DataFrame(protein).to_csv(out / "protein.tsv", sep="\t", index=False)
    pd.DataFrame(ptm).to_csv(out / "ptm.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"modality": ["phosphoproteomics"], "ptm_file": ["ptm.tsv"], "enabled": [True]}
    ).to_csv(out / "manifest.tsv", sep="\t", index=False)
    print(f"Wrote demo inputs to {out}")


if __name__ == "__main__":
    main()
