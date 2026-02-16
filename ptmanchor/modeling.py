from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def paired_lm_intercept_test(
    raw_delta: np.ndarray,
    protein_delta: np.ndarray,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Site-wise OLS on paired deltas: raw_delta ~ intercept + lambda * protein_delta."""
    n_sites = raw_delta.shape[0]
    intercepts = np.full(n_sites, np.nan, dtype=np.float32)
    lambdas = np.full(n_sites, np.nan, dtype=np.float32)
    pvals = np.full(n_sites, np.nan, dtype=float)
    n_obs = np.zeros(n_sites, dtype=int)

    tiny = 1e-12
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
        dy = yv - ym
        sxx = float(np.dot(dx, dx))

        if sxx <= tiny:
            intercepts[i] = np.float32(ym)
            lambdas[i] = np.float32(0.0)
            if n < 2:
                continue
            sd = float(np.std(yv, ddof=1))
            if sd <= tiny or not np.isfinite(sd):
                pvals[i] = 0.0 if ym > 0 else 1.0
            else:
                t_stat = ym / (sd / np.sqrt(n))
                pvals[i] = float(stats.t.sf(t_stat, df=n - 1))
            continue

        slope = float(np.dot(dx, dy) / sxx)
        intercept = float(ym - slope * xm)
        residuals = yv - (intercept + slope * xv)
        df = n - 2
        if df <= 0:
            continue

        rss = float(np.dot(residuals, residuals))
        sigma2 = rss / df
        var_intercept = sigma2 * (1.0 / n + (xm * xm) / sxx)
        if var_intercept <= tiny or not np.isfinite(var_intercept):
            pvals[i] = 0.0 if intercept > 0 else 1.0
        else:
            se = float(np.sqrt(var_intercept))
            t_stat = intercept / se
            pvals[i] = float(stats.t.sf(t_stat, df=df))

        intercepts[i] = np.float32(intercept)
        lambdas[i] = np.float32(slope)

    return intercepts, lambdas, pvals, n_obs


def sample_lm_condition_test(
    ptm_values: np.ndarray,
    protein_values: np.ndarray,
    is_tumor: np.ndarray,
    covariate_matrix: np.ndarray,
    min_tumor: int,
    min_normal: int,
    max_sites: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Site-wise OLS: y ~ 1 + is_tumor + protein + covariates."""
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
            cols.extend([covariate_matrix[valid, j].astype(float) for j in range(covariate_matrix.shape[1])])
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
        var_cond = float(sigma2 * XtX_inv[1, 1])
        if not np.isfinite(var_cond) or var_cond <= tiny:
            p_one = 0.0 if beta[1] > 0 else 1.0
        else:
            se = float(np.sqrt(var_cond))
            t_stat = float(beta[1] / se)
            p_one = float(stats.t.sf(t_stat, df=df))

        beta_condition[i] = np.float32(beta[1])
        beta_protein[i] = np.float32(beta[2])
        pvals[i] = p_one

    return beta_condition, beta_protein, pvals, n_obs


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Site-wise mixed model: y ~ is_tumor + protein + covariates + (1|patient_id)."""
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

        p_one = float(p_two / 2.0) if beta_c >= 0 else float(1.0 - p_two / 2.0)
        beta_condition[i] = np.float32(beta_c)
        beta_protein[i] = np.float32(beta_p) if np.isfinite(beta_p) else np.float32(np.nan)
        pvals[i] = p_one
        fitted[i] = True

    return beta_condition, beta_protein, pvals, n_obs, fitted

