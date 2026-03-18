"""Integration tests for ptmanchor.pipeline using synthetic data."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptmanchor.pipeline import run_manifest, run_modality, build_protein_lookup, _select_lmm_indices


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
        modality_files = list(out_dir.glob("*/all_sites.tsv"))

        if modality_files:
            df = pd.read_csv(modality_files[0], sep="\t")
            expected_cols = ["raw_paired_delta_t_minus_n", "protein_paired_delta_t_minus_n", "subtract_paired_delta_t_minus_n"]
            for col in expected_cols:
                assert col in df.columns, f"Missing column: {col}"


class TestUnpairedPipeline:
    """Test unpaired analysis path (no paired samples)."""

    def test_unpaired_run(self, mock_args_unpaired):
        """Pipeline should complete in unpaired mode."""
        summary_tsv, summary_txt, config_json = run_manifest(mock_args_unpaired)

        assert Path(summary_tsv).exists()
        df = pd.read_csv(summary_tsv, sep="\t")
        assert len(df) >= 1
        assert df.iloc[0]["status"] == "ok"
        assert df.iloc[0]["analysis_mode"] == "unpaired"

    def test_unpaired_output_columns(self, mock_args_unpaired):
        """Unpaired output should contain LM columns."""
        run_manifest(mock_args_unpaired)
        out_dir = Path(mock_args_unpaired.output_dir)
        modality_files = list(out_dir.glob("*/all_sites.tsv"))
        assert len(modality_files) >= 1

        df = pd.read_csv(modality_files[0], sep="\t")
        for col in ["lm_intercept_ptm_specific", "lm_lambda_protein_dependence",
                     "lm_p_intercept_one_sided", "lm_q_bh", "is_true_lm"]:
            assert col in df.columns, f"Missing column: {col}"


class TestForceUnpairedFallback:
    """Test paired-to-unpaired fallback path."""

    def test_force_unpaired_if_paired(self, mock_args):
        """force_unpaired_if_paired should switch to unpaired analysis."""
        args = copy.copy(mock_args)
        args.force_unpaired_if_paired = True

        summary_tsv, _, _ = run_manifest(args)
        df = pd.read_csv(summary_tsv, sep="\t")
        assert df.iloc[0]["analysis_mode"] == "forced_unpaired"

    def test_paired_to_unpaired_fallback_triggered(self, mock_args):
        """When min_paired_testable_sites is very high, fallback should trigger."""
        args = copy.copy(mock_args)
        args.enable_paired_to_unpaired_fallback = True
        args.min_paired_testable_sites = 999999  # impossibly high

        summary_tsv, _, _ = run_manifest(args)
        df = pd.read_csv(summary_tsv, sep="\t")
        assert df.iloc[0]["analysis_mode"] == "paired_fallback_unpaired"


class TestDetectionFallback:
    """Test detection-rate fallback (Fisher's exact test)."""

    def test_force_detection_fallback(self, mock_args):
        """force_detection_fallback should run Fisher's exact test."""
        args = copy.copy(mock_args)
        args.enable_detection_fallback = True
        args.force_detection_fallback = True
        args.min_dual_group_sites_detection = 0

        summary_tsv, _, _ = run_manifest(args)
        df = pd.read_csv(summary_tsv, sep="\t")
        assert df.iloc[0]["status"] == "ok"
        assert df.iloc[0]["detection_fallback_used"] == True

        # Check detection columns in output
        out_dir = Path(args.output_dir)
        modality_files = list(out_dir.glob("*/all_sites.tsv"))
        if modality_files:
            sites = pd.read_csv(modality_files[0], sep="\t")
            assert "detection_p_one_sided" in sites.columns
            assert "detection_q_bh" in sites.columns
            # Some sites should have finite detection p-values
            assert sites["detection_p_one_sided"].notna().sum() > 0


class TestMatchingCascade:
    """Test protein-PTM matching cascade: exact → canonical → gene_symbol."""

    def test_canonical_fallback(self, tmp_path, rng):
        """Sites with isoform accessions should fall back to canonical matching."""
        samples = ["P001-T", "P001-N", "P002-T", "P002-N"]

        # PTM file: uses isoform accession P12345-2
        ptm_df = pd.DataFrame({
            "ID": ["site_1"],
            "UniProtAccession": ["P12345-2"],
            "Gene Symbol": ["GENEX"],
            "Description": ["test"],
            **{s: [float(rng.normal(0, 1))] for s in samples},
        })
        ptm_path = tmp_path / "ptm_cascade.tsv"
        ptm_df.to_csv(ptm_path, sep="\t", index=False)

        # Protein file: uses canonical accession P12345 (no isoform)
        protein_df = pd.DataFrame({
            "ID": ["P12345"],
            "Gene Symbol": ["GENEX"],
            "Description": ["test protein"],
            **{s: [float(rng.normal(0, 0.5))] for s in samples},
        })
        protein_path = tmp_path / "protein_cascade.tsv"
        protein_df.to_csv(protein_path, sep="\t", index=False)

        sample_list, exact, canonical, gene = build_protein_lookup(protein_path)

        # P12345-2 should NOT be in exact (exact has P12345)
        # But canonical of P12345-2 is P12345, which IS in canonical
        from ptmanchor.utils import extract_accession, canonical_accession
        ptm_acc = extract_accession("P12345-2")
        ptm_can = canonical_accession(ptm_acc)
        assert ptm_acc not in exact.index  # exact miss
        assert ptm_can in canonical.index   # canonical hit

    def test_gene_symbol_fallback(self, tmp_path, rng):
        """Sites with no accession match should fall back to gene symbol."""
        samples = ["P001-T", "P001-N", "P002-T", "P002-N"]

        ptm_df = pd.DataFrame({
            "ID": ["site_1"],
            "UniProtAccession": ["XXXXX"],  # no valid accession
            "Gene Symbol": ["TP53"],
            "Description": ["test"],
            **{s: [float(rng.normal(0, 1))] for s in samples},
        })
        ptm_path = tmp_path / "ptm_gene.tsv"
        ptm_df.to_csv(ptm_path, sep="\t", index=False)

        protein_df = pd.DataFrame({
            "ID": ["P04637"],
            "Gene Symbol": ["TP53"],
            "Description": ["Tumor protein p53"],
            **{s: [float(rng.normal(0, 0.5))] for s in samples},
        })
        protein_path = tmp_path / "protein_gene.tsv"
        protein_df.to_csv(protein_path, sep="\t", index=False)

        sample_list, exact, canonical, gene = build_protein_lookup(protein_path)

        # TP53 should be in gene index
        assert "TP53" in gene.index


class TestSampleLmPipeline:
    """Test enable_sample_lm pipeline path with sample design."""

    def test_sample_lm_enabled(self, mock_args, tmp_path, paired_samples):
        """Pipeline with enable_sample_lm=True should produce sample_lm columns."""
        samples, bases = paired_samples

        # Create sample metadata
        meta_df = pd.DataFrame({
            "Sample.ID": samples,
            "Age": np.random.default_rng(42).normal(60, 10, len(samples)),
        })
        meta_path = tmp_path / "sample_meta.tsv"
        meta_df.to_csv(meta_path, sep="\t", index=False)

        args = copy.copy(mock_args)
        args.enable_sample_lm = True
        args.sample_meta_file = str(meta_path)
        args.covariates = "Age"
        # Need separate output dir to avoid conflicts
        out_dir = tmp_path / "output_sample_lm"
        out_dir.mkdir(exist_ok=True)
        args.output_dir = str(out_dir)

        summary_tsv, _, _ = run_manifest(args)
        df = pd.read_csv(summary_tsv, sep="\t")
        assert df.iloc[0]["status"] == "ok"

        # Check sample LM columns in output
        modality_files = list(out_dir.glob("*/all_sites.tsv"))
        assert len(modality_files) >= 1
        sites = pd.read_csv(modality_files[0], sep="\t")
        assert "sample_lm_beta_condition" in sites.columns
        assert "sample_lm_p_one_sided" in sites.columns
        assert "sample_lm_q_bh" in sites.columns


class TestErrorHandling:
    """Test pipeline error handling paths."""

    def test_missing_ptm_file(self, mock_args, tmp_path):
        """Missing PTM file should be recorded as skipped."""
        # Create manifest pointing to non-existent file
        manifest_df = pd.DataFrame({
            "modality": ["phospho"],
            "ptm_file": [str(tmp_path / "nonexistent.tsv")],
            "enabled": ["true"],
        })
        manifest_path = tmp_path / "bad_manifest.tsv"
        manifest_df.to_csv(manifest_path, sep="\t", index=False)

        args = copy.copy(mock_args)
        args.manifest = str(manifest_path)

        summary_tsv, _, _ = run_manifest(args)
        df = pd.read_csv(summary_tsv, sep="\t")
        assert df.iloc[0]["status"] == "skipped_missing_file"

    def test_disabled_modality(self, mock_args, tmp_path, synthetic_ptm_tsv):
        """Disabled modality should be skipped."""
        ptm_path, _, _ = synthetic_ptm_tsv
        manifest_df = pd.DataFrame({
            "modality": ["phospho"],
            "ptm_file": [str(ptm_path)],
            "enabled": ["false"],
        })
        manifest_path = tmp_path / "disabled_manifest.tsv"
        manifest_df.to_csv(manifest_path, sep="\t", index=False)

        args = copy.copy(mock_args)
        args.manifest = str(manifest_path)

        summary_tsv, _, _ = run_manifest(args)
        df = pd.read_csv(summary_tsv, sep="\t")
        assert df.iloc[0]["status"] == "skipped_disabled"


class TestSelectLmmIndices:
    """Test _select_lmm_indices site selection logic."""

    def test_no_limit(self):
        """max_sites=0 should return all indices."""
        n = 100
        raw_up = np.zeros(n, dtype=bool)
        lm_mask = np.zeros(n, dtype=bool)
        score = np.random.default_rng(42).random(n)

        idx = _select_lmm_indices(raw_up, lm_mask, score, max_sites=0)
        assert len(idx) == n

    def test_limit_exceeds_n(self):
        """max_sites > n should return all indices."""
        n = 50
        raw_up = np.zeros(n, dtype=bool)
        lm_mask = np.zeros(n, dtype=bool)
        score = np.random.default_rng(42).random(n)

        idx = _select_lmm_indices(raw_up, lm_mask, score, max_sites=200)
        assert len(idx) == n

    def test_preferred_sites_prioritized(self):
        """raw_up and lm_mask sites should be selected first."""
        n = 20
        raw_up = np.zeros(n, dtype=bool)
        raw_up[:3] = True  # sites 0,1,2 are raw-up
        lm_mask = np.zeros(n, dtype=bool)
        lm_mask[5] = True  # site 5 is lm-significant
        score = np.arange(n, dtype=float)  # higher index = higher score

        idx = _select_lmm_indices(raw_up, lm_mask, score, max_sites=5)
        assert len(idx) == 5
        # All preferred sites (0,1,2,5) should be included
        for s in [0, 1, 2, 5]:
            assert s in idx

    def test_preferred_exceeds_max(self):
        """When preferred sites > max_sites, top scored preferred are returned."""
        n = 20
        raw_up = np.ones(n, dtype=bool)  # all preferred
        lm_mask = np.zeros(n, dtype=bool)
        score = np.arange(n, dtype=float)

        idx = _select_lmm_indices(raw_up, lm_mask, score, max_sites=5)
        assert len(idx) == 5
        # Should be top 5 by score (indices 15-19)
        for s in [15, 16, 17, 18, 19]:
            assert s in idx


class TestSampleLmmPipeline:
    """Test enable_sample_lmm pipeline path."""

    def test_sample_lmm_enabled(self, mock_args, tmp_path, paired_samples):
        """Pipeline with enable_sample_lmm=True should produce LMM columns."""
        samples, bases = paired_samples

        meta_df = pd.DataFrame({
            "Sample.ID": samples,
        })
        meta_path = tmp_path / "sample_meta_lmm.tsv"
        meta_df.to_csv(meta_path, sep="\t", index=False)

        args = copy.copy(mock_args)
        args.enable_sample_lmm = True
        args.sample_meta_file = str(meta_path)
        args.max_sites_sample_lmm = 10  # limit to speed up
        out_dir = tmp_path / "output_sample_lmm"
        out_dir.mkdir(exist_ok=True)
        args.output_dir = str(out_dir)

        summary_tsv, _, _ = run_manifest(args)
        df = pd.read_csv(summary_tsv, sep="\t")
        assert df.iloc[0]["status"] == "ok"

        modality_files = list(out_dir.glob("*/all_sites.tsv"))
        assert len(modality_files) >= 1
        sites = pd.read_csv(modality_files[0], sep="\t")
        assert "sample_lmm_beta_condition" in sites.columns
        assert "sample_lmm_p_one_sided" in sites.columns
        assert "sample_lmm_fitted" in sites.columns


class TestMatchingCascadeEndToEnd:
    """Test matching cascade through full pipeline run_modality."""

    def test_canonical_and_gene_matching_in_pipeline(self, tmp_path, rng):
        """Sites using canonical and gene matching should get protein values."""
        samples = ["P001-T", "P001-N", "P002-T", "P002-N", "P003-T", "P003-N"]

        # PTM: 3 sites with different matching scenarios
        ptm_rows = []
        for i, (acc, gene) in enumerate([
            ("P12345", "GENEA"),    # exact match
            ("P12345-2", "GENEB"),  # canonical match (isoform)
            ("XXXXX", "TP53"),      # gene symbol match only
        ]):
            row = {"ID": f"site_{i}", "UniProtAccession": acc, "Gene Symbol": gene, "Description": f"d{i}"}
            for s in samples:
                row[s] = float(rng.normal(1.0 if s.endswith("-T") else 0.0, 0.5))
            ptm_rows.append(row)
        ptm_df = pd.DataFrame(ptm_rows)
        ptm_path = tmp_path / "ptm_mixed.tsv"
        ptm_df.to_csv(ptm_path, sep="\t", index=False)

        # Protein: covers all three matching paths
        prot_rows = []
        for acc, gene in [("P12345", "GENEA"), ("P04637", "TP53")]:
            row = {"ID": acc, "Gene Symbol": gene, "Description": f"prot"}
            for s in samples:
                row[s] = float(rng.normal(0, 0.3))
            prot_rows.append(row)
        protein_df = pd.DataFrame(prot_rows)
        protein_path = tmp_path / "protein_mixed.tsv"
        protein_df.to_csv(protein_path, sep="\t", index=False)

        sample_list, exact, canonical, gene = build_protein_lookup(protein_path)

        import argparse
        args = argparse.Namespace(
            min_pairs=2, min_tumor=2, min_normal=2,
            fdr_cutoff=0.05, min_corrected_delta=0.2, top_n=10,
            no_eb=True, no_lambda_shrinkage=True,
            enable_sample_lm=False, enable_sample_lmm=False,
            max_sites_sample_lm=0, max_sites_sample_lmm=0, lmm_maxiter=50,
            enable_paired_to_unpaired_fallback=False,
            min_paired_testable_sites=1,
            fallback_min_tumor=0, fallback_min_normal=0,
            force_unpaired_if_paired=False,
            enable_detection_fallback=False,
            force_detection_fallback=False,
            min_detection_delta=0.10,
            min_dual_group_sites_detection=100,
        )

        out_dir = tmp_path / "out_mixed"
        out_dir.mkdir()
        result = run_modality(
            "phospho", ptm_path, sample_list, exact, canonical, gene,
            args, out_dir, sample_design_all=None, covariates=[],
        )

        assert result["status"] == "ok"
        # Check matching sources
        sites = pd.read_csv(out_dir / "phospho" / "all_sites.tsv", sep="\t")
        sources = sites["protein_match_source"].tolist()
        assert "exact" in sources
        assert "canonical" in sources or "gene_symbol" in sources


class TestDetectionFallbackEdgeCases:
    """Test detection fallback edge cases."""

    def test_detection_fallback_dual_group_insufficient(self, mock_args, tmp_path):
        """When dual_group sites are too few, detection should not run."""
        args = copy.copy(mock_args)
        args.enable_detection_fallback = True
        args.force_detection_fallback = False
        # Set impossible threshold so primary testable is 0
        args.min_pairs = 999999
        # But also require many dual_group sites
        args.min_dual_group_sites_detection = 999999

        out_dir = tmp_path / "output_det_edge"
        out_dir.mkdir(exist_ok=True)
        args.output_dir = str(out_dir)

        summary_tsv, _, _ = run_manifest(args)
        df = pd.read_csv(summary_tsv, sep="\t")
        # Detection should NOT have been used (insufficient dual_group sites)
        assert df.iloc[0]["status"] == "ok"
