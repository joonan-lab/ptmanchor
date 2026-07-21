"""Tests for ptmanchor.modeling — verify beta recovery on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch

from ptmanchor.modeling import (
    paired_lm_intercept_test,
    sample_lm_condition_test,
    sample_lmm_condition_test,
    _limma_squeeze_var,
    _check_rpy2,
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


class TestEBShrinkage:
    """Test EB variance shrinkage and lambda shrinkage options."""

    def test_paired_with_eb(self, rng):
        """paired_lm_intercept_test with use_eb=True should run without error."""
        n_sites, n_pairs = 20, 40
        true_intercept = 1.0

        protein_delta = rng.normal(0, 1.0, (n_sites, n_pairs)).astype(np.float32)
        noise = rng.normal(0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        raw_delta = true_intercept + 0.5 * protein_delta + noise

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta.astype(np.float32), protein_delta, min_n=5,
            use_eb=True, lambda_shrinkage=True,
        )

        # Should still recover intercept
        median_intercept = np.nanmedian(intercepts)
        assert median_intercept == pytest.approx(true_intercept, abs=0.5)

    def test_paired_without_eb(self, rng):
        """paired_lm_intercept_test with use_eb=False should match v2 behavior."""
        n_sites, n_pairs = 20, 40
        true_intercept = 1.0

        protein_delta = rng.normal(0, 1.0, (n_sites, n_pairs)).astype(np.float32)
        noise = rng.normal(0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        raw_delta = true_intercept + 0.5 * protein_delta + noise

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta.astype(np.float32), protein_delta, min_n=5,
            use_eb=False, lambda_shrinkage=False,
        )

        median_intercept = np.nanmedian(intercepts)
        assert median_intercept == pytest.approx(true_intercept, abs=0.5)

    def test_sample_lm_with_eb(self, rng):
        """sample_lm_condition_test with use_eb=True should run without error."""
        n_sites, n_samples = 20, 60
        true_beta = 1.0
        is_tumor = np.array([1] * 30 + [0] * 30, dtype=float)
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        noise = rng.normal(0, 0.3, (n_sites, n_samples)).astype(np.float32)
        ptm = (true_beta * is_tumor[np.newaxis, :] + 0.5 * protein + noise).astype(np.float32)
        cov_matrix = np.empty((n_samples, 0))

        beta_cond, beta_prot, pvals, n_obs = sample_lm_condition_test(
            ptm, protein, is_tumor, cov_matrix,
            min_tumor=5, min_normal=5, use_eb=True,
        )

        median_beta = np.nanmedian(beta_cond)
        assert median_beta == pytest.approx(true_beta, abs=0.5)

    def test_lambda_shrinkage_effect(self, rng):
        """Lambda shrinkage should pull extreme lambdas toward global mean."""
        n_sites, n_pairs = 30, 20
        true_lambda = 0.6

        protein_delta = rng.normal(0, 1.0, (n_sites, n_pairs)).astype(np.float32)
        noise = rng.normal(0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        raw_delta = 0.5 + true_lambda * protein_delta + noise

        _, lambdas_shrunk, _, _ = paired_lm_intercept_test(
            raw_delta.astype(np.float32), protein_delta, min_n=5,
            use_eb=False, lambda_shrinkage=True,
        )
        _, lambdas_raw, _, _ = paired_lm_intercept_test(
            raw_delta.astype(np.float32), protein_delta, min_n=5,
            use_eb=False, lambda_shrinkage=False,
        )

        # Shrunk lambdas should have smaller variance than raw
        var_shrunk = np.nanvar(lambdas_shrunk)
        var_raw = np.nanvar(lambdas_raw)
        assert var_shrunk <= var_raw


class TestSampleLmmConditionTest:
    """sample_lmm_condition_test: y ~ is_tumor + protein + (1|patient_id)."""

    def test_recover_beta_condition(self, rng):
        """Known tumor effect should be recovered by LMM."""
        n_sites = 3
        n_samples = 40
        true_beta = 1.5

        is_tumor = np.array([1] * 20 + [0] * 20, dtype=float)
        patient_ids = np.array(
            [f"P{i:03d}" for i in range(20)] + [f"P{i:03d}" for i in range(20)]
        )
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        noise = rng.normal(0, 0.3, (n_sites, n_samples)).astype(np.float32)
        # Per-patient random intercept, shared by that patient's tumor and normal sample.
        patient_effect = rng.normal(0, 0.8, (n_sites, n_samples // 2))
        patient_effect = np.concatenate([patient_effect, patient_effect], axis=1)
        ptm = (
            true_beta * is_tumor[np.newaxis, :] + 0.5 * protein + patient_effect + noise
        ).astype(np.float32)

        covariate_df = pd.DataFrame(index=range(n_samples))

        beta_cond, beta_prot, pvals, n_obs, fitted = sample_lmm_condition_test(
            ptm, protein, is_tumor, patient_ids, covariate_df,
            min_tumor=5, min_normal=5,
        )

        for i in range(n_sites):
            assert fitted[i]
            assert beta_cond[i] == pytest.approx(true_beta, abs=0.6)
            assert pvals[i] < 0.05

    def test_null_sites(self, rng):
        """Sites with no tumor effect should not be significant in LMM."""
        n_sites = 5
        n_samples = 40
        is_tumor = np.array([1] * 20 + [0] * 20, dtype=float)
        patient_ids = np.array(
            [f"P{i:03d}" for i in range(20)] + [f"P{i:03d}" for i in range(20)]
        )
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        noise = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        ptm = (0.5 * protein + noise).astype(np.float32)

        covariate_df = pd.DataFrame(index=range(n_samples))

        beta_cond, _, pvals, _, fitted = sample_lmm_condition_test(
            ptm, protein, is_tumor, patient_ids, covariate_df,
            min_tumor=5, min_normal=5,
        )

        sig_count = np.sum(pvals[np.isfinite(pvals)] < 0.05)
        assert sig_count <= 2, f"Too many false positives: {sig_count}/5"

    def test_min_group_filtering(self, rng):
        """Insufficient group sizes should yield NaN p-values."""
        n_sites = 2
        n_samples = 10
        is_tumor = np.array([1] * 2 + [0] * 8, dtype=float)
        patient_ids = np.array([f"P{i}" for i in range(10)])
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        ptm = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        covariate_df = pd.DataFrame(index=range(n_samples))

        beta_cond, _, pvals, _, fitted = sample_lmm_condition_test(
            ptm, protein, is_tumor, patient_ids, covariate_df,
            min_tumor=5, min_normal=5,
        )

        assert np.all(np.isnan(pvals))

    def test_with_covariates(self, rng):
        """LMM should recover beta_condition even with covariates."""
        n_sites = 2
        n_samples = 60
        true_beta = 1.2

        is_tumor = np.array([1] * 30 + [0] * 30, dtype=float)
        patient_ids = np.array(
            [f"P{i:03d}" for i in range(30)] + [f"P{i:03d}" for i in range(30)]
        )
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        age = rng.normal(60, 10, n_samples)
        noise = rng.normal(0, 0.3, (n_sites, n_samples)).astype(np.float32)
        # Per-patient random intercept, shared by that patient's tumor and normal sample.
        patient_effect = rng.normal(0, 0.8, (n_sites, n_samples // 2))
        patient_effect = np.concatenate([patient_effect, patient_effect], axis=1)
        ptm = (
            true_beta * is_tumor[np.newaxis, :]
            + 0.5 * protein
            + 0.01 * age[np.newaxis, :]
            + patient_effect
            + noise
        ).astype(np.float32)

        covariate_df = pd.DataFrame({"Age": age})

        beta_cond, _, pvals, _, fitted = sample_lmm_condition_test(
            ptm, protein, is_tumor, patient_ids, covariate_df,
            min_tumor=5, min_normal=5,
        )

        for i in range(n_sites):
            assert fitted[i]
            assert beta_cond[i] == pytest.approx(true_beta, abs=0.6)


class TestLimmaSqueezeVar:
    """Test the Python fallback for EB variance shrinkage."""

    def test_python_fallback_runs(self, rng):
        """_limma_squeeze_var should return valid results via Python fallback."""
        n = 50
        s2 = rng.exponential(1.0, n)
        df = np.full(n, 10.0)

        squeezed_s2, squeezed_df, d0, s0_sq = _limma_squeeze_var(s2, df)

        assert squeezed_s2.shape == (n,)
        assert squeezed_df.shape == (n,)
        assert np.all(np.isfinite(squeezed_s2))
        assert np.all(np.isfinite(squeezed_df))
        # Shrunk variances should be less extreme than raw
        assert np.var(squeezed_s2) <= np.var(s2)

    def test_handles_nan_and_zero(self, rng):
        """Should handle NaN and zero variance gracefully."""
        s2 = np.array([1.0, np.nan, 0.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3, 0.9])
        df = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        squeezed_s2, squeezed_df, d0, s0_sq = _limma_squeeze_var(s2, df)

        assert squeezed_s2.shape == (10,)
        # NaN input should remain NaN in output
        assert np.isnan(squeezed_s2[1])

    def test_few_sites_no_shrinkage(self):
        """With < 3 valid sites, should return original values."""
        s2 = np.array([1.0, np.nan])
        df = np.array([10.0, 10.0])

        squeezed_s2, squeezed_df, d0, s0_sq = _limma_squeeze_var(s2, df)

        assert squeezed_s2[0] == pytest.approx(1.0)
        assert d0 == 0.0

    def test_forced_python_fallback(self, rng):
        """When rpy2 is unavailable, Python fallback should be used."""
        import ptmanchor.modeling as mod
        # Force rpy2 check to return False
        old_avail = mod._R_AVAIL
        try:
            mod._R_AVAIL = False

            n = 30
            s2 = rng.exponential(1.0, n)
            df = np.full(n, 15.0)

            squeezed_s2, squeezed_df, d0, s0_sq = _limma_squeeze_var(s2, df)

            assert squeezed_s2.shape == (n,)
            assert np.all(np.isfinite(squeezed_s2))
            # d0 should be positive (shrinkage applied)
            assert d0 >= 0.0
            assert s0_sq > 0.0
        finally:
            mod._R_AVAIL = old_avail

    def test_mock_rpy2_success_path(self, rng):
        """Simulate rpy2+limma available via mock, testing the R code path."""
        pytest.importorskip("rpy2.robjects")  # the R branch is unreachable without it
        import ptmanchor.modeling as mod
        from unittest.mock import MagicMock

        n = 20
        s2 = rng.exponential(1.0, n)
        df_arr = np.full(n, 10.0)
        valid = np.isfinite(s2) & (s2 > 0)

        # Create mock R result
        mock_result = MagicMock()
        mock_result.rx2.side_effect = lambda key: {
            "var.post": s2[valid] * 0.9,  # slightly shrunk
            "df.prior": np.array([5.0]),
            "var.prior": np.array([1.0]),
        }[key]

        mock_limma = MagicMock()
        mock_limma.squeezeVar.return_value = mock_result

        old_avail = mod._R_AVAIL
        old_pkgs = mod._R_PKGS.copy()
        try:
            mod._R_AVAIL = True
            mod._R_PKGS["limma"] = mock_limma

            squeezed_s2, squeezed_df, d0, s0_sq = _limma_squeeze_var(s2, df_arr)

            # Should have called limma
            mock_limma.squeezeVar.assert_called_once()
            assert d0 == pytest.approx(5.0)
            assert s0_sq == pytest.approx(1.0)
            assert np.all(np.isfinite(squeezed_s2))
        finally:
            mod._R_AVAIL = old_avail
            mod._R_PKGS.clear()
            mod._R_PKGS.update(old_pkgs)

    def test_mock_rpy2_exception_falls_back(self, rng):
        """When R limma call raises an exception, should fall back to Python."""
        import ptmanchor.modeling as mod
        from unittest.mock import MagicMock

        n = 25
        s2 = rng.exponential(1.0, n)
        df_arr = np.full(n, 12.0)

        mock_limma = MagicMock()
        mock_limma.squeezeVar.side_effect = RuntimeError("R crashed")

        old_avail = mod._R_AVAIL
        old_pkgs = mod._R_PKGS.copy()
        try:
            mod._R_AVAIL = True
            mod._R_PKGS["limma"] = mock_limma

            # Should not crash — falls back to Python
            squeezed_s2, squeezed_df, d0, s0_sq = _limma_squeeze_var(s2, df_arr)

            assert squeezed_s2.shape == (n,)
            assert np.all(np.isfinite(squeezed_s2))
        finally:
            mod._R_AVAIL = old_avail
            mod._R_PKGS.clear()
            mod._R_PKGS.update(old_pkgs)

    def test_brentq_fallback_path(self, rng):
        """When brentq cannot find root, should use safe fallback d0."""
        import ptmanchor.modeling as mod

        old_avail = mod._R_AVAIL
        try:
            mod._R_AVAIL = False

            # Create data with very high excess variance to trigger brentq edge case
            n = 30
            s2 = np.concatenate([
                rng.exponential(0.01, n // 2),
                rng.exponential(100.0, n // 2),
            ])
            df_arr = np.full(n, 5.0)

            squeezed_s2, squeezed_df, d0, s0_sq = _limma_squeeze_var(s2, df_arr)

            # Should not crash regardless of brentq outcome
            assert squeezed_s2.shape == (n,)
            assert d0 >= 0.0
        finally:
            mod._R_AVAIL = old_avail

    def test_python_fallback_no_excess_variance(self):
        """When all variances are identical, no shrinkage should be applied."""
        n = 20
        s2 = np.full(n, 1.0)  # identical variances → excess = 0
        df = np.full(n, 10.0)

        import ptmanchor.modeling as mod
        old_avail = mod._R_AVAIL
        try:
            mod._R_AVAIL = False
            squeezed_s2, squeezed_df, d0, s0_sq = _limma_squeeze_var(s2, df)
            # No excess variance → d0 should be 0
            assert d0 == 0.0
            # Original values should be returned unchanged
            assert np.allclose(squeezed_s2, s2)
        finally:
            mod._R_AVAIL = old_avail


class TestEdgeCases:
    """Test edge cases in modeling functions."""

    def test_paired_lm_all_nan(self):
        """All NaN input should return all NaN p-values."""
        n_sites, n_pairs = 3, 10
        raw_delta = np.full((n_sites, n_pairs), np.nan, dtype=np.float32)
        protein_delta = np.full((n_sites, n_pairs), np.nan, dtype=np.float32)

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5
        )

        assert np.all(np.isnan(pvals))
        assert np.all(n_obs == 0)

    def test_paired_lm_zero_protein_variance(self, rng):
        """When protein_delta has zero variance (constant), should still work."""
        n_sites, n_pairs = 2, 20
        raw_delta = rng.normal(1.0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        # Constant protein delta → sxx = 0
        protein_delta = np.full((n_sites, n_pairs), 0.5, dtype=np.float32)

        intercepts, lambdas, pvals, n_obs = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5
        )

        # Should still get valid p-values (falls back to mean-only model)
        assert np.all(np.isfinite(pvals))

    def test_sample_lm_rank_deficient(self, rng):
        """Rank-deficient design matrix should be handled gracefully."""
        n_sites = 3
        n_samples = 10
        # is_tumor is all 1 → no contrast → rank deficient
        is_tumor = np.ones(n_samples, dtype=float)
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        ptm = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        cov_matrix = np.empty((n_samples, 0))

        beta_cond, _, pvals, n_obs = sample_lm_condition_test(
            ptm, protein, is_tumor, cov_matrix,
            min_tumor=3, min_normal=3,
        )

        # All normal_n = 0, so min_normal filter should exclude all
        assert np.all(np.isnan(pvals))

    def test_lmm_single_patient_group(self, rng):
        """LMM with only one unique patient_id should skip."""
        n_sites = 2
        n_samples = 20
        is_tumor = np.array([1] * 10 + [0] * 10, dtype=float)
        patient_ids = np.array(["P001"] * 20)  # all same patient
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        ptm = rng.normal(1.0, 0.5, (n_sites, n_samples)).astype(np.float32)
        covariate_df = pd.DataFrame(index=range(n_samples))

        beta_cond, _, pvals, _, fitted = sample_lmm_condition_test(
            ptm, protein, is_tumor, patient_ids, covariate_df,
            min_tumor=5, min_normal=5,
        )

        # Single patient group → should skip (nunique < 2)
        assert np.all(np.isnan(pvals))
        assert not np.any(fitted)

    def test_lmm_with_missing_values(self, rng):
        """LMM should handle NaN values in PTM/protein gracefully."""
        n_sites = 2
        n_samples = 40
        true_beta = 1.0
        is_tumor = np.array([1] * 20 + [0] * 20, dtype=float)
        patient_ids = np.array(
            [f"P{i:03d}" for i in range(20)] + [f"P{i:03d}" for i in range(20)]
        )
        protein = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        noise = rng.normal(0, 0.3, (n_sites, n_samples)).astype(np.float32)
        # Per-patient random intercept, shared by that patient's tumor and normal sample.
        patient_effect = rng.normal(0, 0.8, (n_sites, n_samples // 2))
        patient_effect = np.concatenate([patient_effect, patient_effect], axis=1)
        ptm = (
            true_beta * is_tumor[np.newaxis, :] + 0.5 * protein + patient_effect + noise
        ).astype(np.float32)

        # Add 20% missing values
        mask = rng.random((n_sites, n_samples)) < 0.2
        ptm[mask] = np.nan

        covariate_df = pd.DataFrame(index=range(n_samples))
        beta_cond, _, pvals, n_obs, fitted = sample_lmm_condition_test(
            ptm, protein, is_tumor, patient_ids, covariate_df,
            min_tumor=5, min_normal=5,
        )

        # Should still produce results (enough non-NaN data)
        for i in range(n_sites):
            assert fitted[i]
            assert n_obs[i] < n_samples  # some were dropped

    def test_paired_lm_with_high_missing(self, rng):
        """Paired LM with heavy missingness — some sites should be skipped."""
        n_sites, n_pairs = 5, 15
        raw_delta = rng.normal(1.0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        protein_delta = rng.normal(0, 0.5, (n_sites, n_pairs)).astype(np.float32)

        # Site 0: 90% missing → should be skipped
        mask0 = rng.random(n_pairs) < 0.9
        raw_delta[0, mask0] = np.nan
        protein_delta[0, mask0] = np.nan

        intercepts, _, pvals, n_obs = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5
        )

        assert np.isnan(pvals[0])  # too few observations
        assert np.isfinite(pvals[1])  # enough observations

    def test_paired_lm_exactly_min_n(self, rng):
        """Site with exactly min_n observations should still be processed."""
        n_pairs = 10
        raw_delta = rng.normal(2.0, 0.5, (1, n_pairs)).astype(np.float32)
        protein_delta = rng.normal(0, 0.5, (1, n_pairs)).astype(np.float32)
        # Keep only 5 obs (min_n=5)
        raw_delta[0, 5:] = np.nan
        protein_delta[0, 5:] = np.nan

        intercepts, _, pvals, n_obs = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5
        )

        assert n_obs[0] == 5
        assert np.isfinite(pvals[0])

    def test_sample_lm_shape_validation(self, rng):
        """Mismatched shapes should raise ValueError."""
        n_sites, n_samples = 5, 20
        ptm = rng.normal(0, 1, (n_sites, n_samples)).astype(np.float32)
        protein = rng.normal(0, 1, (n_sites, n_samples + 5)).astype(np.float32)  # wrong shape
        is_tumor = np.array([1] * 10 + [0] * 10, dtype=float)
        cov = np.empty((n_samples, 0))

        with pytest.raises(ValueError, match="shape"):
            sample_lm_condition_test(ptm, protein, is_tumor, cov, min_tumor=3, min_normal=3)

    def test_paired_lm_df_equals_zero(self, rng):
        """Site with exactly 2 obs (df=0 after fitting slope) should be handled."""
        n_pairs = 10
        raw_delta = rng.normal(1.0, 0.5, (1, n_pairs)).astype(np.float32)
        protein_delta = rng.normal(0, 1.0, (1, n_pairs)).astype(np.float32)
        # Keep only 2 obs → df = n-2 = 0 → should skip
        raw_delta[0, 2:] = np.nan
        protein_delta[0, 2:] = np.nan

        intercepts, _, pvals, n_obs = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=2
        )

        # With only 2 obs: n-2=0, df<=0, should skip
        assert np.isnan(pvals[0])


class TestBidirectionalInference:
    """Direction of the intercept test (`alternative`)."""

    def test_less_detects_negative_intercept(self, rng):
        """A true PTM-specific decrease is significant under 'less', not under 'greater'."""
        n_sites, n_pairs = 4, 40
        protein_delta = rng.normal(0, 0.3, (n_sites, n_pairs)).astype(np.float32)
        raw_delta = (-1.0 + rng.normal(0, 0.3, (n_sites, n_pairs))).astype(np.float32)

        _, _, p_less, _ = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5, alternative="less"
        )
        _, _, p_greater, _ = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5, alternative="greater"
        )

        for i in range(n_sites):
            assert p_less[i] < 0.01
            assert p_greater[i] > 0.99

    def test_two_sided_detects_both_directions(self, rng):
        n_pairs = 40
        protein_delta = rng.normal(0, 0.3, (2, n_pairs)).astype(np.float32)
        raw_delta = np.vstack([
            1.0 + rng.normal(0, 0.3, (1, n_pairs)),
            -1.0 + rng.normal(0, 0.3, (1, n_pairs)),
        ]).astype(np.float32)

        intercepts, _, pvals, _ = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5, alternative="two-sided"
        )

        assert intercepts[0] > 0 and intercepts[1] < 0
        assert (pvals < 0.01).all()

    def test_two_sided_pvalue_is_double_the_one_sided_tail(self, rng):
        """Same t-statistic, only the tail conversion differs."""
        n_sites, n_pairs = 6, 30
        protein_delta = rng.normal(0, 0.5, (n_sites, n_pairs)).astype(np.float32)
        raw_delta = (0.8 + 0.5 * protein_delta + rng.normal(0, 0.4, (n_sites, n_pairs))).astype(np.float32)

        intercepts, _, p_two, _ = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5, alternative="two-sided"
        )
        _, _, p_greater, _ = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5, alternative="greater"
        )

        for i in range(n_sites):
            assert intercepts[i] > 0, "fixture should produce positive intercepts"
            assert p_two[i] == pytest.approx(2.0 * p_greater[i])

    def test_default_is_greater(self, rng):
        """Omitting `alternative` must reproduce the one-sided upregulation test."""
        protein_delta = rng.normal(0, 0.3, (3, 30)).astype(np.float32)
        raw_delta = (0.9 + rng.normal(0, 0.3, (3, 30))).astype(np.float32)

        _, _, p_default, _ = paired_lm_intercept_test(raw_delta, protein_delta, min_n=5)
        _, _, p_greater, _ = paired_lm_intercept_test(
            raw_delta, protein_delta, min_n=5, alternative="greater"
        )
        assert p_default == pytest.approx(p_greater, nan_ok=True)

    def test_sample_lm_honours_direction(self, rng):
        n_sites, n_samples = 3, 40
        is_tumor = np.zeros(n_samples, dtype=bool)
        is_tumor[: n_samples // 2] = True
        protein = rng.normal(0, 0.3, (n_sites, n_samples))
        ptm = rng.normal(0, 0.3, (n_sites, n_samples))
        ptm[:, is_tumor] -= 1.5  # tumour lower than normal

        _, _, p_less, _ = sample_lm_condition_test(
            ptm, protein, is_tumor, np.empty((n_samples, 0)),
            min_tumor=5, min_normal=5, alternative="less",
        )
        assert (p_less < 0.01).all()

    def test_invalid_alternative_raises(self, rng):
        with pytest.raises(ValueError):
            paired_lm_intercept_test(
                rng.normal(0, 1, (2, 10)), rng.normal(0, 1, (2, 10)),
                min_n=5, alternative="negative",
            )


class TestEBBackendEquivalence:
    """The Python EB implementation must match limma::squeezeVar."""

    def _both_backends(self, s2, df):
        import ptmanchor.modeling as m

        saved = m._R_AVAIL
        try:
            m._R_AVAIL = None
            m._R_PKGS = {}
            if not m._check_rpy2():
                pytest.skip("R/limma not available")
            r = _limma_squeeze_var(s2, df)
            m._R_AVAIL = False
            p = _limma_squeeze_var(s2, df)
        finally:
            m._R_AVAIL = saved
        return r, p

    @pytest.mark.parametrize("df_value", [8, 28])
    def test_constant_df_matches_limma(self, rng, df_value):
        n = 3000
        df = np.full(n, float(df_value))
        true_var = 0.25 * 4 / rng.chisquare(4, n)
        s2 = true_var * rng.chisquare(df, n) / df
        r, p = self._both_backends(s2, df)
        assert r[2] == pytest.approx(p[2], rel=1e-6)
        assert r[3] == pytest.approx(p[3], rel=1e-6)
        np.testing.assert_allclose(r[0], p[0], rtol=1e-6)

    def test_varying_df_matches_limma(self, rng):
        n = 3000
        df = rng.integers(6, 29, n).astype(float)
        true_var = 0.25 * 4 / rng.chisquare(4, n)
        s2 = true_var * rng.chisquare(df, n) / df
        r, p = self._both_backends(s2, df)
        assert r[3] == pytest.approx(p[3], rel=1e-6)
        np.testing.assert_allclose(r[0], p[0], rtol=1e-6)
