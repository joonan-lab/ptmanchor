"""Integration tests for ptmanchor.pipeline using synthetic data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ptmanchor.pipeline import run_manifest, run_modality, build_protein_lookup


class TestBuildProteinLookup:
    def test_basic_lookup(self, synthetic_protein_tsv):
        protein_path, n_proteins, samples = synthetic_protein_tsv
        sample_list, exact, canonical, gene = build_protein_lookup(protein_path)

        assert len(sample_list) > 0
        assert not exact.empty or not canonical.empty or not gene.empty

    def test_sample_detection(self, synthetic_protein_tsv):
        protein_path, _, expected_samples = synthetic_protein_tsv
        sample_list, _, _, _ = build_protein_lookup(protein_path)
        # All expected samples should be detected
        assert set(expected_samples).issubset(set(sample_list))


class TestRunModality:
    def test_basic_run(self, synthetic_ptm_tsv, synthetic_protein_tsv, mock_args, tmp_path):
        """run_modality should complete without error and produce output."""
        ptm_path, _, _ = synthetic_ptm_tsv
        protein_path, _, _ = synthetic_protein_tsv

        sample_list, exact, canonical, gene = build_protein_lookup(protein_path)

        out_dir = tmp_path / "modality_out"
        out_dir.mkdir()

        result = run_modality(
            modality="phospho",
            ptm_file=ptm_path,
            protein_samples=sample_list,
            protein_exact=exact,
            protein_canonical=canonical,
            protein_gene=gene,
            args=mock_args,
            output_dir=out_dir,
            sample_design_all=None,
            covariates=[],
        )

        assert isinstance(result, dict)
        assert "modality" in result
        assert result["modality"] == "phospho"


class TestRunManifest:
    def test_end_to_end(self, mock_args):
        """Full manifest pipeline should complete and write output files."""
        summary_tsv, summary_txt, config_json = run_manifest(mock_args)

        assert Path(summary_tsv).exists()
        assert Path(summary_txt).exists()
        assert Path(config_json).exists()

        # Summary TSV should be readable
        df = pd.read_csv(summary_tsv, sep="\t")
        assert len(df) >= 1
        assert "modality" in df.columns

    def test_output_columns(self, mock_args):
        """Output per-modality TSV should contain expected columns."""
        summary_tsv, _, _ = run_manifest(mock_args)

        # Find the per-modality output
        out_dir = Path(mock_args.output_dir)
        modality_files = list(out_dir.glob("*_corrected.tsv"))

        if modality_files:
            df = pd.read_csv(modality_files[0], sep="\t")
            expected_cols = ["raw_delta", "protein_delta", "subtract_delta"]
            for col in expected_cols:
                assert col in df.columns, f"Missing column: {col}"
