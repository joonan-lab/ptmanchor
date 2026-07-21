"""modeling: site-wise OLS with optional EB variance shrinkage and lambda shrinkage."""
from __future__ import annotations

import os
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import digamma, polygamma

from .utils import _validate_alternative, t_pvalue_from_stat

# ---------------------------------------------------------------------------
# R_HOME setup for rpy2
# ---------------------------------------------------------------------------
if "R_HOME" not in os.environ:
    try:
        _r_home = subprocess.check_output(["R", "RHOME"], text=True).strip()
        os.environ["R_HOME"] = _r_home
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Empirical Bayes variance shrinkage (limma squeezeVar)
# ---------------------------------------------------------------------------
_R_PKGS: dict = {}
_R_AVAIL: bool | None = None


def _check_rpy2() -> bool:
    global _R_AVAIL, _R_PKGS
    if _R_AVAIL is not None:
        return _R_AVAIL
    try:
        import rpy2.robjects
        from rpy2.robjects.packages import importr
        _R_PKGS["limma"] = importr("limma")
        _R_AVAIL = True
    except Exception:
        _R_AVAIL = False
    return _R_AVAIL


def _limma_squeeze_var(s2, df):
    """Empirical Bayes variance shrinkage via limma::squeezeVar (rpy2) or Python fallback."""
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    valid = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    if valid.sum() < 3:
        return s2.copy(), df.copy(), 0.0, float(np.nanmedian(s2))

    if _check_rpy2():
        try:
            import rpy2.robjects as ro
            sv, dv = s2[valid], df[valid]
            result = _R_PKGS["limma"].squeezeVar(
                ro.FloatVector(sv.tolist()),
                ro.FloatVector(dv.tolist()),
            )
            var_post = np.array(result.rx2("var.post"), dtype=float)
            df_prior = float(np.array(result.rx2("df.prior"))[0])
            var_prior = float(np.array(result.rx2("var.prior"))[0])
            squeezed_s2 = s2.copy()
            squeezed_df = df.copy()
            squeezed_s2[valid] = var_post
            squeezed_df[valid] = dv + df_prior
            return squeezed_s2, squeezed_df, df_prior, var_prior
        except Exception:
            pass

    # Python fallback (moment-matching)
    sv, dv = s2[valid], df[valid]
    z = np.log(sv) - (digamma(dv / 2.0) - np.log(dv / 2.0))
    s0_sq = float(np.exp(np.mean(z)))
    obs_var = float(np.var(z, ddof=1))
    chi2_var = float(np.mean(polygamma(1, dv / 2.0)))
    excess = obs_var - chi2_var
    if excess <= 0:
        return s2.copy(), df.copy(), 0.0, s0_sq
    try:
        def _f(d0h):
            return float(polygamma(1, d0h)) - excess
        lo, hi = 1e-4, 1e5
        if _f(lo) * _f(hi) >= 0:
            d0 = float(min(2.0 / excess, 1e4))
        else:
            d0 = 2.0 * float(brentq(_f, lo, hi, xtol=1e-6))
    except Exception:
        d0 = 10.0
    # limma fitFDist rescales the prior variance once df.prior is known:
    #   s20 <- exp(emean + digamma(df2/2) - log(df2/2))
    s0_sq = float(np.exp(np.mean(z) + digamma(d0 / 2.0) - np.log(d0 / 2.0)))

    squeezed_s2 = s2.copy()
    squeezed_df = df.copy()
    squeezed_s2[valid] = (d0 * s0_sq + dv * sv) / (d0 + dv)
    squeezed_df[valid] = dv + d0
    return squeezed_s2, squeezed_df, d0, s0_sq


# ---------------------------------------------------------------------------
# paired_lm_intercept_test  (EB + lambda shrinkage)
# ---------------------------------------------------------------------------

def paired_lm_intercept_test(
    raw_delta: np.ndarray,
    protein_delta: np.ndarray,
    min_n: int,
    use_eb: bool = True,
    lambda_shrinkage: bool = True,
    alternative: str = "greater",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Site-wise OLS on paired deltas with optional EB + lambda shrinkage.

    Model: raw_delta_i = beta_i + lambda_i * protein_delta_i + epsilon_i

    Pass 1: raw OLS per site -> intercept, lambda, sigma2, df
    Pass 2 (optional): shrink lambda toward precision-weighted global mean
    Pass 3 (optional): EB shrinkage of sigma2 via limma squeezeVar
    Final: moderated t-test on intercept, one- or two-sided per `alternative`
    """
    _validate_alternative(alternative)
    n_sites = raw_delta.shape[0]
    tiny = 1e-12

    # --- Pass 1: OLS per site ---
    intercepts = np.full(n_sites, np.nan, dtype=np.float32)
    lambdas = np.full(n_sites, np.nan, dtype=np.float32)
    sigma2_arr = np.full(n_sites, np.nan, dtype=float)
    df_arr = np.full(n_sites, np.nan, dtype=float)
    n_obs = np.zeros(n_sites, dtype=int)
    xm_arr = np.full(n_sites, np.nan, dtype=float)
    sxx_arr = np.full(n_sites, np.nan, dtype=float)

    for i in range(n_sites):
        y = raw_delta[i, :]
        x = protein_delta[i, :]
        valid = np.isfinite(y) & np.isfinite(x)
        n = int(np.sum(valid))
        n_obs[i] = n
        if n < min_n:
            continue

        xv = x[valid].astype(float)
        yv = y[valid].astype(float)
        xm = float(np.mean(xv))
        ym = float(np.mean(yv))
        dx = xv - xm
        sxx = float(np.dot(dx, dx))
        xm_arr[i] = xm
        sxx_arr[i] = sxx

        if sxx <= tiny:
            intercepts[i] = np.float32(ym)
            lambdas[i] = np.float32(0.0)
            if n >= 2:
                sigma2_arr[i] = float(np.var(yv, ddof=1))
                df_arr[i] = float(n - 1)
            continue

        slope = float(np.dot(dx, yv - ym) / sxx)
        intercept = float(ym - slope * xm)
        residuals = yv - (intercept + slope * xv)
        df = n - 2
        if df <= 0:
            continue

        rss = float(np.dot(residuals, residuals))
        intercepts[i] = np.float32(intercept)
        lambdas[i] = np.float32(slope)
        sigma2_arr[i] = rss / df
        df_arr[i] = float(df)

    # --- Lambda shrinkage ---
    if lambda_shrinkage:
        valid_lam = (
            np.isfinite(lambdas.astype(float))
            & np.isfinite(sigma2_arr)
            & (sxx_arr > tiny)
        )
        if valid_lam.sum() > 10:
            # Global lambda: precision-weighted average
            weights = np.where(
                valid_lam & (sigma2_arr > tiny), sxx_arr / sigma2_arr, 0.0
            )
            lambda_global = float(
                np.average(lambdas[valid_lam].astype(float), weights=weights[valid_lam])
            )

            # Per-site Var(lambda_hat) = sigma2 / Sxx
            var_lambda = np.where(
                valid_lam & (sxx_arr > tiny), sigma2_arr / sxx_arr, np.inf
            )

            # Prior variance of lambda across sites
            lam_vals = lambdas[valid_lam].astype(float)
            var_lam_finite = var_lambda[valid_lam]
            obs_spread = float(np.var(lam_vals, ddof=1))
            mean_sampling_var = float(
                np.mean(var_lam_finite[np.isfinite(var_lam_finite)])
            )
            tau2 = max(obs_spread - mean_sampling_var, 0.01)

            # Shrinkage weight: tau2 / (tau2 + Var(lambda_hat))
            shrink_weight = np.where(
                valid_lam & np.isfinite(var_lambda),
                tau2 / (tau2 + var_lambda),
                0.0,  # fully shrink to global when Var(lambda) is inf
            )
            lambdas_shrunk = np.where(
                valid_lam,
                shrink_weight * lambdas.astype(float)
                + (1.0 - shrink_weight) * lambda_global,
                lambdas.astype(float),
            )

            # Recompute intercepts and sigma2 with shrunk lambda
            for i in range(n_sites):
                if not valid_lam[i]:
                    continue
                y = raw_delta[i, :]
                x = protein_delta[i, :]
                valid = np.isfinite(y) & np.isfinite(x)
                n = int(np.sum(valid))
                if n < min_n:
                    continue
                xv = x[valid].astype(float)
                yv = y[valid].astype(float)
                lam_s = lambdas_shrunk[i]
                intercept = float(np.mean(yv) - lam_s * np.mean(xv))
                residuals = yv - (intercept + lam_s * xv)
                df = n - 2
                if df <= 0:
                    continue
                rss = float(np.dot(residuals, residuals))
                intercepts[i] = np.float32(intercept)
                lambdas[i] = np.float32(lam_s)
                sigma2_arr[i] = rss / df

    # --- EB variance shrinkage ---
    if use_eb:
        eb_valid = (
            np.isfinite(sigma2_arr)
            & (sigma2_arr > 0)
            & np.isfinite(df_arr)
            & (df_arr > 0)
        )
        if eb_valid.sum() >= 3:
            sigma2_arr, df_arr, _, _ = _limma_squeeze_var(sigma2_arr, df_arr)

    # --- Compute p-values (moderated t-test) ---
    pvals = np.full(n_sites, np.nan, dtype=float)
    for i in range(n_sites):
        # df can be inf when EB prior df is inf (t -> normal)
        if not (
            np.isfinite(float(intercepts[i]))
            and np.isfinite(sigma2_arr[i])
            and (np.isfinite(df_arr[i]) or np.isinf(df_arr[i]))
            and sigma2_arr[i] > 0
            and df_arr[i] > 0
            and n_obs[i] >= min_n
        ):
            continue

        sxx = sxx_arr[i] if np.isfinite(sxx_arr[i]) else 0.0
        xm = xm_arr[i] if np.isfinite(xm_arr[i]) else 0.0

        if sxx <= tiny:
            se = float(np.sqrt(sigma2_arr[i] / n_obs[i]))
        else:
            var_intercept = sigma2_arr[i] * (1.0 / n_obs[i] + (xm ** 2) / sxx)
            if var_intercept <= tiny:
                pvals[i] = _degenerate_pvalue(float(intercepts[i]), alternative)
                continue
            se = float(np.sqrt(var_intercept))

        if se < tiny:
            pvals[i] = _degenerate_pvalue(float(intercepts[i]), alternative)
            continue

        t_stat = float(intercepts[i]) / se
        pvals[i] = t_pvalue_from_stat(t_stat, df_arr[i], alternative=alternative)

    return intercepts, lambdas, pvals, n_obs


def _degenerate_pvalue(effect: float, alternative: str) -> float:
    """Limiting p-value when the standard error collapses to zero."""
    if alternative == "greater":
        return 0.0 if effect > 0 else 1.0
    if alternative == "less":
        return 0.0 if effect < 0 else 1.0
    # two-sided
    return 0.0 if effect != 0 else 1.0


# ---------------------------------------------------------------------------
# sample_lm_condition_test  (EB variance shrinkage)
# ---------------------------------------------------------------------------

def sample_lm_condition_test(
    ptm_values: np.ndarray,
    protein_values: np.ndarray,
    is_tumor: np.ndarray,
    covariate_matrix: np.ndarray,
    min_tumor: int,
    min_normal: int,
    max_sites: int = 0,
    use_eb: bool = True,
    alternative: str = "greater",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Site-wise OLS: y ~ 1 + is_tumor + protein + covariates, with optional EB."""
    _validate_alternative(alternative)
    n_sites, n_samples = ptm_values.shape
    if protein_values.shape != ptm_values.shape:
        raise ValueError("protein_values shape must match ptm_values shape")
    if is_tumor.shape[0] != n_samples:
        raise ValueError("is_tumor length must match number of samples")
    if covariate_matrix.shape[0] != n_samples:
        raise ValueError("covariate_matrix rows must match number of samples")

    idx = np.arange(n_sites, dtype=int)
    if max_sites > 0 and max_sites < n_sites:
        idx = idx[:max_sites]

    beta_condition = np.full(n_sites, np.nan, dtype=np.float32)
    beta_protein = np.full(n_sites, np.nan, dtype=np.float32)
    pvals = np.full(n_sites, np.nan, dtype=float)
    n_obs = np.zeros(n_sites, dtype=int)
    sigma2_arr = np.full(n_sites, np.nan, dtype=float)
    df_arr = np.full(n_sites, np.nan, dtype=float)
    xtx_inv_11 = np.full(n_sites, np.nan, dtype=float)

    tiny = 1e-12
    for i in idx:
        y = ptm_values[i, :]
        p = protein_values[i, :]

        valid = np.isfinite(y) & np.isfinite(p)
        if covariate_matrix.size > 0:
            valid &= np.all(np.isfinite(covariate_matrix), axis=1)
        n = int(np.sum(valid))
        n_obs[i] = n
        if n == 0:
            continue

        tumor_n = int(np.sum((is_tumor == 1) & valid))
        normal_n = int(np.sum((is_tumor == 0) & valid))
        if tumor_n < min_tumor or normal_n < min_normal:
            continue

        yv = y[valid].astype(float)
        tv = is_tumor[valid].astype(float)
        pv = p[valid].astype(float)
        cols = [np.ones_like(tv), tv, pv]
        if covariate_matrix.size > 0:
            cols.extend(
                [covariate_matrix[valid, j].astype(float) for j in range(covariate_matrix.shape[1])]
            )
        X = np.column_stack(cols)
        p_dim = X.shape[1]
        if n <= p_dim:
            continue

        XtX = X.T @ X
        rank = np.linalg.matrix_rank(XtX)
        if rank < p_dim:
            continue

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue

        beta = XtX_inv @ X.T @ yv
        residual = yv - X @ beta
        df = n - p_dim
        if df <= 0:
            continue
        rss = float(np.dot(residual, residual))
        sigma2 = rss / df

        beta_condition[i] = np.float32(beta[1])
        beta_protein[i] = np.float32(beta[2])
        sigma2_arr[i] = sigma2
        df_arr[i] = float(df)
        xtx_inv_11[i] = float(XtX_inv[1, 1])

    # --- EB variance shrinkage ---
    if use_eb:
        eb_valid = (
            np.isfinite(sigma2_arr)
            & (sigma2_arr > 0)
            & np.isfinite(df_arr)
            & (df_arr > 0)
        )
        if eb_valid.sum() >= 3:
            sigma2_arr, df_arr, _, _ = _limma_squeeze_var(sigma2_arr, df_arr)

    # --- Compute p-values ---
    for i in idx:
        if not (
            np.isfinite(float(beta_condition[i]))
            and np.isfinite(sigma2_arr[i])
            and (np.isfinite(df_arr[i]) or np.isinf(df_arr[i]))
            and sigma2_arr[i] > 0
            and df_arr[i] > 0
            and np.isfinite(xtx_inv_11[i])
        ):
            continue
        var_cond = float(sigma2_arr[i] * xtx_inv_11[i])
        if not np.isfinite(var_cond) or var_cond <= tiny:
            pvals[i] = _degenerate_pvalue(float(beta_condition[i]), alternative)
        else:
            se = float(np.sqrt(var_cond))
            t_stat = float(beta_condition[i] / se)
            pvals[i] = t_pvalue_from_stat(t_stat, df_arr[i], alternative=alternative)

    return beta_condition, beta_protein, pvals, n_obs


# ---------------------------------------------------------------------------
# sample_lmm_condition_test
# ---------------------------------------------------------------------------

def sample_lmm_condition_test(
    ptm_values: np.ndarray,
    protein_values: np.ndarray,
    is_tumor: np.ndarray,
    patient_ids: np.ndarray,
    covariate_df: pd.DataFrame,
    min_tumor: int,
    min_normal: int,
    selected_indices: np.ndarray | None = None,
    maxiter: int = 100,
    alternative: str = "greater",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Site-wise mixed model: y ~ is_tumor + protein + covariates + (1|patient_id)."""
    _validate_alternative(alternative)
    import statsmodels.formula.api as smf

    n_sites, n_samples = ptm_values.shape
    if selected_indices is None:
        selected_indices = np.arange(n_sites, dtype=int)

    beta_condition = np.full(n_sites, np.nan, dtype=np.float32)
    beta_protein = np.full(n_sites, np.nan, dtype=np.float32)
    pvals = np.full(n_sites, np.nan, dtype=float)
    n_obs = np.zeros(n_sites, dtype=int)
    fitted = np.zeros(n_sites, dtype=bool)

    base = pd.DataFrame(
        {
            "sample_idx": np.arange(n_samples, dtype=int),
            "is_tumor": is_tumor.astype(float),
            "patient_id": pd.Series(patient_ids).astype(str),
        }
    )
    cov_cols = list(covariate_df.columns)
    for col in cov_cols:
        base[col] = pd.to_numeric(covariate_df[col], errors="coerce")

    rhs = ["is_tumor", "protein"] + cov_cols
    formula = "y ~ " + " + ".join(rhs)

    for i in selected_indices:
        y = ptm_values[i, :]
        p = protein_values[i, :]

        work = base.copy()
        work["y"] = y
        work["protein"] = p

        valid = np.isfinite(work["y"].to_numpy()) & np.isfinite(work["protein"].to_numpy())
        if cov_cols:
            valid &= np.isfinite(work[cov_cols].to_numpy(dtype=float)).all(axis=1)
        work = work.loc[valid].copy()

        n = int(work.shape[0])
        n_obs[i] = n
        if n == 0:
            continue
        tumor_n = int(np.sum(work["is_tumor"] > 0.5))
        normal_n = int(np.sum(work["is_tumor"] < 0.5))
        if tumor_n < min_tumor or normal_n < min_normal:
            continue
        if work["patient_id"].nunique() < 2:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, data=work, groups=work["patient_id"], re_formula="1")
                fit = model.fit(reml=False, method="lbfgs", maxiter=maxiter, disp=False)
        except Exception:
            continue

        beta_c = fit.params.get("is_tumor", np.nan)
        beta_p = fit.params.get("protein", np.nan)
        p_two = fit.pvalues.get("is_tumor", np.nan)
        if not np.isfinite(beta_c) or not np.isfinite(p_two):
            continue

        if alternative == "two-sided":
            p_directed = float(p_two)
        elif alternative == "greater":
            p_directed = float(p_two / 2.0) if beta_c >= 0 else float(1.0 - p_two / 2.0)
        else:  # "less"
            p_directed = float(p_two / 2.0) if beta_c <= 0 else float(1.0 - p_two / 2.0)
        beta_condition[i] = np.float32(beta_c)
        beta_protein[i] = np.float32(beta_p) if np.isfinite(beta_p) else np.float32(np.nan)
        pvals[i] = p_directed
        fitted[i] = True

    return beta_condition, beta_protein, pvals, n_obs, fitted
