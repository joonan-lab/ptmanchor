"""Tests for ptmanchor.utils."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ptmanchor.utils import (
    extract_accession,
    canonical_accession,
    parse_bool,
    parse_csv_list,
    sample_columns,
    bh_qvalues,
    one_sided_ttest_1samp,
    one_sided_ttest_ind,
    get_paired_indices,
    nanmean_axis1,
)


# --- accession extraction ---


class TestExtractAccession:
    def test_standard_uniprot(self):
        assert extract_accession("sp|P12345|PROT_HUMAN") == "P12345"

    def test_ensembl_with_version(self):
        assert extract_accession("ENSP00000369497.3") == "ENSP00000369497.3"

    def test_isoform(self):
        assert extract_accession("P12345-2") == "P12345-2"

    def test_none(self):
        assert extract_accession(None) is None

    def test_no_match(self):
        assert extract_accession("random_string_123") is None

    def test_embedded_in_text(self):
        assert extract_accession("prefix_Q9Y6K9_suffix") == "Q9Y6K9"


class TestCanonicalAccession:
    def test_strip_isoform(self):
        assert canonical_accession("P12345-2") == "P12345"

    def test_strip_ensembl_version(self):
        assert canonical_accession("ENSP00000369497.3") == "ENSP00000369497"

    def test_already_canonical(self):
        assert canonical_accession("P12345") == "P12345"

    def test_none(self):
        assert canonical_accession(None) is None


# --- parsing helpers ---


class TestParseBool:
    @pytest.mark.parametrize("val", [1, "1", "true", "True", "TRUE", "t", "yes", "y"])
    def test_truthy(self, val):
        assert parse_bool(val) is True

    @pytest.mark.parametrize("val", [0, "0", "false", "False", "FALSE", "f", "no", "n"])
    def test_falsy(self, val):
        assert parse_bool(val) is False

    def test_none_default_true(self):
        assert parse_bool(None, default=True) is True

    def test_nan_default_false(self):
        assert parse_bool(float("nan"), default=False) is False


class TestParseCsvList:
    def test_basic(self):
        assert parse_csv_list("a, b ,c") == ["a", "b", "c"]

    def test_none(self):
        assert parse_csv_list(None) == []

    def test_empty(self):
        assert parse_csv_list("") == []


# --- sample column detection ---


class TestSampleColumns:
    def test_internal_prefix(self):
        cols = ["ID", "Gene", "RE-001-T", "RE-001-N", "RE-002-T"]
        assert sample_columns(cols) == ["RE-001-T", "RE-001-N", "RE-002-T"]

    def test_external_suffix(self):
        cols = ["ID", "Gene", "TCGA-A1-T", "TCGA-A1-N", "TCGA-A2-T"]
        assert sample_columns(cols) == ["TCGA-A1-T", "TCGA-A1-N", "TCGA-A2-T"]

    def test_dot_suffix(self):
        cols = ["ID", "S1.T", "S1.N"]
        assert sample_columns(cols) == ["S1.T", "S1.N"]

    def test_no_samples(self):
        assert sample_columns(["ID", "Gene"]) == []


# --- BH FDR ---


class TestBhQvalues:
    def test_monotone(self):
        pvals = np.array([0.001, 0.01, 0.05, 0.5])
        qvals = bh_qvalues(pvals)
        assert np.all(np.diff(qvals) >= 0), "q-values should be non-decreasing"

    def test_nan_preserved(self):
        pvals = np.array([0.01, np.nan, 0.05])
        qvals = bh_qvalues(pvals)
        assert np.isnan(qvals[1])
        assert np.isfinite(qvals[0])
        assert np.isfinite(qvals[2])

    def test_all_nan(self):
        pvals = np.array([np.nan, np.nan])
        qvals = bh_qvalues(pvals)
        assert np.all(np.isnan(qvals))

    def test_single_value(self):
        qvals = bh_qvalues(np.array([0.03]))
        assert qvals[0] == pytest.approx(0.03)


# --- t-tests ---


class TestOneSidedTtest1samp:
    def test_positive_shift(self, rng):
        """Large positive shift should yield small p-values."""
        matrix = rng.normal(loc=3.0, scale=1.0, size=(5, 30))
        pvals, n = one_sided_ttest_1samp(matrix, min_n=5)
        assert np.all(pvals < 0.01)
        assert np.all(n == 30)

    def test_zero_mean(self, rng):
        """Zero-centered data should have p ~ 0.5."""
        matrix = rng.normal(loc=0.0, scale=1.0, size=(5, 100))
        pvals, n = one_sided_ttest_1samp(matrix, min_n=5)
        assert np.all(pvals > 0.1)

    def test_min_n_mask(self, rng):
        """Sites with fewer than min_n observations should be NaN."""
        matrix = np.full((3, 10), np.nan)
        matrix[0, :5] = rng.normal(3, 1, 5)   # n=5
        matrix[1, :2] = rng.normal(3, 1, 2)   # n=2
        matrix[2, :8] = rng.normal(3, 1, 8)   # n=8
        pvals, n = one_sided_ttest_1samp(matrix, min_n=5)
        assert np.isfinite(pvals[0])
        assert np.isnan(pvals[1])
        assert np.isfinite(pvals[2])


class TestOneSidedTtestInd:
    def test_different_means(self, rng):
        """Tumor higher than normal should yield small p-value."""
        tumor = rng.normal(loc=2.0, scale=1.0, size=(3, 20))
        normal = rng.normal(loc=0.0, scale=1.0, size=(3, 15))
        pvals, nt, nn = one_sided_ttest_ind(tumor, normal, min_tumor=5, min_normal=5)
        assert np.all(pvals < 0.01)

    def test_min_n_mask(self, rng):
        """Insufficient samples should be NaN."""
        tumor = rng.normal(0, 1, (2, 3))   # only 3 tumor
        normal = rng.normal(0, 1, (2, 20))
        pvals, nt, nn = one_sided_ttest_ind(tumor, normal, min_tumor=5, min_normal=5)
        assert np.all(np.isnan(pvals))


# --- paired indices ---


class TestGetPairedIndices:
    def test_basic_pairs(self):
        samples = ["P001-T", "P002-T", "P003-T", "P001-N", "P002-N", "P003-N"]
        t_idx, n_idx, bases = get_paired_indices(samples)
        assert len(bases) == 3
        assert list(bases) == ["P001", "P002", "P003"]
        assert list(t_idx) == [0, 1, 2]
        assert list(n_idx) == [3, 4, 5]

    def test_partial_pairs(self):
        samples = ["P001-T", "P002-T", "P001-N"]
        t_idx, n_idx, bases = get_paired_indices(samples)
        assert bases == ["P001"]

    def test_no_pairs_raises(self):
        with pytest.raises(ValueError, match="No paired"):
            get_paired_indices(["A-T", "B-N"])


# --- nanmean ---


class TestNanmeanAxis1:
    def test_basic(self):
        m = np.array([[1.0, 2.0, np.nan], [np.nan, 4.0, 6.0]])
        result = nanmean_axis1(m)
        assert result[0] == pytest.approx(1.5)
        assert result[1] == pytest.approx(5.0)
