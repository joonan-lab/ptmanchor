"""Tests for ptmanchor.modeling — verify beta recovery on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from ptmanchor.modeling import (
    paired_lm_intercept_test,
    sample_lm_condition_test,
)


class TestPairedLmInterceptTest:
    """paired_lm_intercept_test: raw_delta ~ intercept + lambda * protein_delta."""

    def test_recover_intercept_no_protein_effect(self, rng):
        """When protein_delta is pure noise, intercept should recover the true PTM shift."""
        n_sites, n_pairs = 5, 40
        true_intercept = 1.5

        protein_delta = rng.normal(0, 0.3, (n_sites, n_pairs)).astype(np.float32)
        noise = rng.normal(0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        raw_delta = true_intercept + noise  # lambda=0

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5
        )

        for i in range(n_sites):
            assert intercepts[i] == pytest.approx(true_intercept, abs=0.5)
            assert pvals[i] < 0.01

    def test_recover_intercept_with_protein_effect(self, rng):
        """When protein has strong effect (lambda=0.8), intercept should still recover."""
        n_sites, n_pairs = 3, 50
        true_intercept = 1.0
        true_lambda = 0.8

        protein_delta = rng.normal(0, 1.0, (n_sites, n_pairs)).astype(np.float32)
        noise = rng.normal(0, 0.3, (n_sites, n_pairs)).astype(np.float32)
        raw_delta = true_intercept + true_lambda * protein_delta + noise

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta.astype(np.float32), protein_delta, min_n=5
        )

        for i in range(n_sites):
            assert intercepts[i] == pytest.approx(true_intercept, abs=0.4)
            assert lambdas[i] == pytest.approx(true_lambda, abs=0.3)
            assert pvals[i] < 0.01

    def test_null_sites_not_significant(self, rng):
        """Sites with no true intercept (beta_ptm=0) should not be significant."""
        n_sites, n_pairs = 10, 30
        true_lambda = 0.5

        protein_delta = rng.normal(0, 1.0, (n_sites, n_pairs)).astype(np.float32)
        noise = rng.normal(0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        raw_delta = true_lambda * protein_delta + noise  # intercept=0

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta.astype(np.float32), protein_delta, min_n=5
        )

        # Most null sites should NOT be significant at 0.05
        sig_count = np.sum(pvals < 0.05)
        assert sig_count <= 3, f"Too many false positives: {sig_count}/10"

    def test_min_n_filtering(self, rng):
        """Sites with too few observations should return NaN p-values."""
        n_sites, n_pairs = 2, 10
        raw_delta = rng.normal(1.0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        protein_delta = rng.normal(0, 0.5, (n_sites, n_pairs)).astype(np.float32)

        # Mask most values for site 0
        raw_delta[0, 3:] = np.nan
        protein_delta[0, 3:] = np.nan

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5
        )

        assert np.isnan(pvals[0])      # too few obs
        assert np.isfinite(pvals[1])   # enough obs

    def test_protein_driven_fp_removed(self, rng):
        """A site where signal is entirely protein-driven should have ~0 intercept."""
        n_pairs = 50
        true_lambda = 1.0

        protein_delta = rng.normal(1.0, 0.5, (1, n_pairs)).astype(np.float32)
        raw_delta = (true_lambda * protein_delta +
                     rng.normal(0, 0.2, (1, n_pairs))).astype(np.float32)

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5
        )

        # Intercept should be close to 0 (protein explains the signal)
        assert abs(intercepts[0]) < 0.5
        # p-value for one-sided (intercept > 0) should not be significant
        assert pvals[0] > 0.05


class TestSampleLmConditionTest:
    """sample_lm_condition_test: y ~ 1 + is_tumor + protein + covariates."""

    def test_recover_beta_condition(self, rng):
        """Known tumor effect should be recovered."""
        n_sites = 5
        n_samples = 60
        true_beta_tumor = 1.2
        true_beta_protein = 0.8

        is_tumor = np.array([1] * 30 + [0] * 30, dtype=float)
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        noise = rng.normal(0, 0.3, (n_sites, n_samples)).astype(np.float32)

        ptm = (true_beta_tumor * is_tumor[np.newaxis, :]
               + true_beta_protein * protein
               + noise).astype(np.float32)

        cov_matrix = np.empty((n_samples, 0))

        beta_cond, beta_prot, pvals, n_obs = sample_lm_condition_test(
            ptm, protein, is_tumor, cov_matrix,
            min_tumor=5, min_normal=5,
        )

        for i in range(n_sites):
            assert beta_cond[i] == pytest.approx(true_beta_tumor, abs=0.4)
            assert beta_prot[i] == pytest.approx(true_beta_protein, abs=0.4)
            assert pvals[i] < 0.01

    def test_null_sites(self, rng):
        """Sites with no condition effect should not be significant."""
        n_sites = 10
        n_samples = 40
        is_tumor = np.array([1] * 20 + [0] * 20, dtype=float)
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        noise = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)

        ptm = (0.5 * protein + noise).astype(np.float32)  # no tumor effect

        cov_matrix = np.empty((n_samples, 0))

        beta_cond, beta_prot, pvals, n_obs = sample_lm_condition_test(
            ptm, protein, is_tumor, cov_matrix,
            min_tumor=5, min_normal=5,
        )

        sig_count = np.sum(pvals[np.isfinite(pvals)] < 0.05)
        assert sig_count <= 3, f"Too many false positives: {sig_count}/10"

    def test_min_group_filtering(self, rng):
        """Insufficient group sizes should yield NaN p-values."""
        n_sites = 3
        n_samples = 10
        is_tumor = np.array([1] * 2 + [0] * 8, dtype=float)  # only 2 tumor
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        ptm = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        cov_matrix = np.empty((n_samples, 0))

        beta_cond, beta_prot, pvals, n_obs = sample_lm_condition_test(
            ptm, protein, is_tumor, cov_matrix,
            min_tumor=5, min_normal=5,
        )

        assert np.all(np.isnan(pvals))

    def test_with_covariates(self, rng):
        """Model should still recover beta_condition with covariates."""
        n_sites = 3
        n_samples = 80
        true_beta = 1.0

        is_tumor = np.array([1] * 40 + [0] * 40, dtype=float)
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        cov = rng.normal(0, 1, (n_samples, 1)).astype(np.float64)
        noise = rng.normal(0, 0.3, (n_sites, n_samples)).astype(np.float32)

        ptm = (true_beta * is_tumor[np.newaxis, :]
               + 0.5 * protein
               + 0.3 * cov[:, 0][np.newaxis, :]
               + noise).astype(np.float32)

        beta_cond, beta_prot, pvals, n_obs = sample_lm_condition_test(
            ptm, protein, is_tumor, cov,
            min_tumor=5, min_normal=5,
        )

        for i in range(n_sites):
            assert beta_cond[i] == pytest.approx(true_beta, abs=0.4)
            assert pvals[i] < 0.01

    def test_max_sites(self, rng):
        """max_sites should limit how many sites are processed."""
        n_sites = 20
        n_samples = 40
        is_tumor = np.array([1] * 20 + [0] * 20, dtype=float)
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        ptm = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        cov_matrix = np.empty((n_samples, 0))

        beta_cond, beta_prot, pvals, n_obs = sample_lm_condition_test(
            ptm, protein, is_tumor, cov_matrix,
            min_tumor=5, min_normal=5, max_sites=5,
        )

        # Only first 5 sites should have results
        assert np.sum(np.isfinite(pvals[:5])) == 5
        assert np.all(np.isnan(pvals[5:]))
