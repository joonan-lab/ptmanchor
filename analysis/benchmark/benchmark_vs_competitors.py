#!/usr/bin/env python3
"""benchmark_vs_competitors.py

Two-version benchmark comparing ptmanchor vs. competing methods on synthetic data.
rpy2 is required for limma::squeezeVar (MSstatsPTM) and msqrob2 (msqrob2PTM).

VERSION 1 — Main Figure (Paired benchmark):
  - Data: 30 tumor + 30 normal, ALL paired.
  - All six methods use the paired structure.
  - Output: benchmark_*_paired.tsv

VERSION 2 — Supplementary Figure (Unpaired benchmark):
  - Data: 30 tumor + 20 normal, ALL UNPAIRED (n_paired=0).
  - All six methods use their unpaired counterparts.
  - ptmanchor uses sample_lm_condition_test (PTM ~ is_tumor + protein).
  - MSstatsPTM uses Welch-based two-sample models + EB.
  - msqrob2PTM uses DPU normalisation + Huber two-sample test + EB.
  - Output: benchmark_*_unpaired.tsv

Output directory: results/benchmark/

Usage:
  cd paper_ptmanchor/
  R_HOME=$(R RHOME) python analysis/benchmark/benchmark_vs_competitors.py
  R_HOME=$(R RHOME) python analysis/benchmark/benchmark_vs_competitors.py --n-reps 20 --seed 42
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import digamma, polygamma
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

from ptmanchor.utils import bh_qvalues, one_sided_ttest_1samp, one_sided_ttest_ind
from ptmanchor.modeling import paired_lm_intercept_test, sample_lm_condition_test


# ---------------------------------------------------------------------------
# R_HOME setup — ensure rpy2 can find the Homebrew R installation
# ---------------------------------------------------------------------------

if "R_HOME" not in os.environ:
    import subprocess
    try:
        r_home = subprocess.check_output(["R", "RHOME"], text=True).strip()
        os.environ["R_HOME"] = r_home
    except Exception:
        pass  # will fail at _check_rpy2() below


# ---------------------------------------------------------------------------
# rpy2 / R package cache
# ---------------------------------------------------------------------------

_R_PKGS: dict = {}
_R_AVAIL: bool | None = None


def _check_rpy2() -> bool:
    """Return True if rpy2 and the required R packages are importable.

    Required R packages: limma, msqrob2, SummarizedExperiment.
    Result is cached so the check runs only once per session.

    rpy2 is required; a RuntimeError is raised if unavailable.
    """
    global _R_AVAIL, _R_PKGS
    if _R_AVAIL is not None:
        return _R_AVAIL
    try:
        import rpy2.robjects  # noqa: F401
        from rpy2.robjects.packages import importr
        _R_PKGS["limma"] = importr("limma")
        _R_PKGS["msqrob2"] = importr("msqrob2")
        _R_PKGS["SummarizedExperiment"] = importr("SummarizedExperiment")
        _R_AVAIL = True
        print("[rpy2] Loaded: limma, msqrob2, SummarizedExperiment")
    except Exception as exc:
        raise RuntimeError(
            f"rpy2 / R packages are REQUIRED but unavailable: {exc}\n"
            "Install rpy2 (`pip install rpy2`) and the R packages:\n"
            "  BiocManager::install(c('limma', 'msqrob2', 'SummarizedExperiment'))\n"
            "Set R_HOME if R is not on the default path:\n"
            "  export R_HOME=$(R RHOME)"
        ) from exc
    return _R_AVAIL


# ---------------------------------------------------------------------------
# Empirical Bayes & robust helpers
# ---------------------------------------------------------------------------

def _squeeze_var(
    s2: np.ndarray,
    df: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Approximate limma::squeezeVar (Smyth 2004, Stat. Appl. Genet. Mol. Biol.).

    Shrinks per-site sample variances toward a global prior via Empirical Bayes.

    Hierarchical model
    ------------------
      s²_i | sigma²_i  ~  sigma²_i * chi²(d_i) / d_i
      sigma²_i          ~  Inv-chi²(d_0, s_0²)       [prior]

    Posterior (closed form)
    -----------------------
      tilde_s²_i = (d_0 * s_0² + d_i * s²_i) / (d_0 + d_i)
      tilde_df_i = d_i + d_0

    Prior estimation
    ----------------
      z_i  = log(s²_i) - [psi(d_i/2) - log(d_i/2)]   (mean ≈ log s_0²)
      s_0² = exp(mean(z))
      excess_var = Var(z) - mean(psi'(d_i/2))
      Solve:  psi'(d_0/2) = excess_var   →  d_0  (via brentq)

    Returns
    -------
    squeezed_s2 : ndarray  posterior variances
    squeezed_df : ndarray  updated degrees of freedom
    d0          : float    estimated prior df  (0 if no shrinkage needed)
    s0_sq       : float    estimated prior variance
    """
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    squeezed_s2 = s2.copy()
    squeezed_df = df.copy()

    valid = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    if valid.sum() < 3:
        return squeezed_s2, squeezed_df, 0.0, float(np.nanmedian(s2))

    sv, dv = s2[valid], df[valid]

    # Standardise: z_i = log(s²_i) - [psi(d_i/2) - log(d_i/2)]
    z = np.log(sv) - (digamma(dv / 2.0) - np.log(dv / 2.0))
    s0_sq = float(np.exp(np.mean(z)))

    # Estimate d_0 from excess variance of z across sites
    obs_var = float(np.var(z, ddof=1))
    chi2_var = float(np.mean(polygamma(1, dv / 2.0)))
    excess = obs_var - chi2_var

    if excess <= 0:
        # Inter-site variance not detected; no shrinkage
        return squeezed_s2, squeezed_df, 0.0, s0_sq

    def _f(d0_half: float) -> float:
        return float(polygamma(1, d0_half)) - excess

    try:
        lo, hi = 1e-4, 1e5
        if _f(lo) * _f(hi) >= 0:
            # excess too small/large for the grid; use a safe fallback
            d0 = float(min(2.0 / excess, 1e4))
        else:
            d0 = 2.0 * float(brentq(_f, lo, hi, xtol=1e-6))
    except Exception:
        d0 = 10.0

    # Posterior
    squeezed_s2[valid] = (d0 * s0_sq + dv * sv) / (d0 + dv)
    squeezed_df[valid] = dv + d0

    return squeezed_s2, squeezed_df, d0, s0_sq


def _huber_one_sample(
    delta_matrix: np.ndarray,
    c: float = 1.345,
    min_n: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-site robust one-sample location via IRLS with Huber weights.

    Approximates msqrob2's M-estimation step (robust=TRUE).
    Reference: Phipson et al. (2016); msqrob2 package.

    Parameters
    ----------
    delta_matrix : (n_sites, n_pairs) array of paired differences
    c            : Huber tuning constant (default 1.345 → 95 % efficiency)
    min_n        : minimum finite observations required

    Returns
    -------
    mu : (n_sites,) robust location estimates
    se : (n_sites,) robust standard errors of the location
    nv : (n_sites,) number of valid (finite) observations used
    """
    n_sites = delta_matrix.shape[0]
    mu = np.full(n_sites, np.nan)
    se = np.full(n_sites, np.nan)
    nv = np.zeros(n_sites, dtype=int)

    for i in range(n_sites):
        y = delta_matrix[i]
        ok = np.isfinite(y)
        n = int(ok.sum())
        if n < min_n:
            continue
        yv = y[ok].astype(float)

        # Initialise at median
        loc = float(np.median(yv))

        # IRLS
        for _ in range(50):
            r = yv - loc
            mad = float(np.median(np.abs(r)))
            scale = mad / 0.6745 if mad > 1e-12 else 1.0
            u = r / scale
            w = np.where(np.abs(u) <= c, 1.0, c / np.abs(u))
            denom = float(w.sum())
            loc_new = float(np.dot(w, yv) / max(denom, 1e-12))
            if abs(loc_new - loc) < 1e-8 * (abs(loc) + 1e-10):
                loc = loc_new
                break
            loc = loc_new

        # Robust SE: weighted residual variance / n
        r = yv - loc
        mad = float(np.median(np.abs(r)))
        scale = mad / 0.6745 if mad > 1e-12 else 1.0
        u = r / scale
        w = np.where(np.abs(u) <= c, 1.0, c / np.abs(u))
        rss_w = float(np.dot(w, r ** 2))
        dof_w = max(float(w.sum()) - 1.0, 1.0)
        se_loc = float(np.sqrt(rss_w / dof_w / n))

        mu[i] = loc
        se[i] = se_loc
        nv[i] = n

    return mu, se, nv


# ---------------------------------------------------------------------------
# rpy2 helper functions
# ---------------------------------------------------------------------------

def _limma_squeeze_var(
    s2: np.ndarray,
    df: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Variance shrinkage via exact limma::squeezeVar (rpy2) or Python fallback.

    If rpy2 and limma are available, calls limma::squeezeVar exactly
    (Smyth 2004, Newton-Raphson MLE for prior df).
    Otherwise falls back to the Python moment-matching approximation.

    Returns
    -------
    Same signature as _squeeze_var: (squeezed_s2, squeezed_df, d0, s0_sq)
    """
    if not _check_rpy2():
        return _squeeze_var(s2, df)

    try:
        import rpy2.robjects as ro
        s2a = np.asarray(s2, dtype=float)
        dfa = np.asarray(df, dtype=float)
        valid = np.isfinite(s2a) & (s2a > 0) & np.isfinite(dfa) & (dfa > 0)
        if valid.sum() < 3:
            return _squeeze_var(s2, df)

        sv, dv = s2a[valid], dfa[valid]
        limma_r = _R_PKGS["limma"]
        result = limma_r.squeezeVar(
            ro.FloatVector(sv.tolist()),
            ro.FloatVector(dv.tolist()),
        )
        var_post = np.array(result.rx2("var.post"), dtype=float)
        df_prior = float(np.array(result.rx2("df.prior"))[0])
        var_prior = float(np.array(result.rx2("var.prior"))[0])

        squeezed_s2 = s2a.copy()
        squeezed_df = dfa.copy()
        squeezed_s2[valid] = var_post
        squeezed_df[valid] = dv + df_prior
        return squeezed_s2, squeezed_df, df_prior, var_prior

    except Exception as exc:
        warnings.warn(f"limma::squeezeVar failed ({exc}); using Python fallback.")
        return _squeeze_var(s2, df)


def _msqrob2_fit_and_test(
    assay_matrix: np.ndarray,
    condition: np.ndarray | None,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run msqrob2::msqrob + hypothesisTest on *assay_matrix* via rpy2.

    Parameters
    ----------
    assay_matrix : (n_sites, n_samples) — DPU-adjusted log-intensities.
                   NaN = missing; converted to NA_real_ in R.
    condition    : (n_samples,) int (1 = tumor, 0 = normal) for a two-sample
                   test (unpaired), or None for a one-sample intercept test
                   (paired delta matrix).
    robust       : Passed to msqrob2::msqrob.

    Notes
    -----
    Paired   (condition=None): formula=~1,            ridge=FALSE,
             contrast="(Intercept) = 0"
    Unpaired (condition arr):  formula=~condition, ridge=FALSE,
             contrast="conditiontumor = 0"

    Returns
    -------
    pvals_one : (n_sites,) one-sided p-values (H1: logFC > 0).
    logFC     : (n_sites,) log2 fold-changes from msqrob2.
    """
    import rpy2.robjects as ro

    n_sites, n_samples = assay_matrix.shape

    # Assign rownames (required — msqrob2 re-indexes by rowname).
    site_names = [f"site{i+1}" for i in range(n_sites)]
    samp_names = [f"s{j+1}" for j in range(n_samples)]

    # Build R matrix (column-major fill); Python float('nan') → R NaN.
    flat = assay_matrix.ravel(order="F")
    ro.globalenv["._mqp_assay"] = ro.r["matrix"](
        ro.FloatVector([float(v) for v in flat]),
        nrow=n_sites, ncol=n_samples,
    )
    ro.globalenv["._mqp_rnames"] = ro.StrVector(site_names)
    ro.globalenv["._mqp_cnames"] = ro.StrVector(samp_names)
    ro.r("""
        rownames(._mqp_assay) <- ._mqp_rnames
        colnames(._mqp_assay) <- ._mqp_cnames
        ._mqp_assay[is.nan(._mqp_assay)] <- NA_real_
    """)

    robust_r = "TRUE" if robust else "FALSE"

    if condition is None:
        # ── Paired: one-sample test on delta matrix ──────────────────────────
        # formula=~1, ridge=FALSE, contrast=(Intercept)=0
        ro.r("""
            ._mqp_cd <- S4Vectors::DataFrame(dummy = factor(rep("a", ncol(._mqp_assay))))
            rownames(._mqp_cd) <- colnames(._mqp_assay)
            ._mqp_se <- SummarizedExperiment::SummarizedExperiment(
                assays  = list(logintensities = ._mqp_assay),
                colData = ._mqp_cd
            )
        """)
        ro.r(f"""
            ._mqp_se <- msqrob2::msqrob(
                object  = ._mqp_se,
                formula = ~1,
                robust  = {robust_r},
                ridge   = FALSE
            )
            ._mqp_models <- SummarizedExperiment::rowData(._mqp_se)[["msqrobModels"]]
            ._mqp_params <- names(msqrob2::getCoef(._mqp_models[[1]]))
            ._mqp_L <- msqrob2::makeContrast(
                "(Intercept) = 0", parameterNames = ._mqp_params
            )
            ._mqp_se     <- msqrob2::hypothesisTest(object = ._mqp_se, contrast = ._mqp_L)
            ._mqp_rescol <- colnames(._mqp_L)
            ._mqp_res    <- as.data.frame(
                SummarizedExperiment::rowData(._mqp_se)[[._mqp_rescol]]
            )
        """)
    else:
        # ── Unpaired: two-sample test (robust + EB, no ridge) ──────────────
        # formula=~condition (treatment contrast), ridge=FALSE
        # Note: ridge=TRUE with ~condition crashes msqrob2 ("must have more
        # than two parameters for ridge regression").  ridge=FALSE still
        # applies robust regression + EB variance moderation via msqrob2.
        cond_labels = ["tumor" if int(c) == 1 else "normal" for c in condition]
        ro.globalenv["._mqp_cond"] = ro.StrVector(cond_labels)
        ro.r("""
            ._mqp_cd <- S4Vectors::DataFrame(
                condition = factor(._mqp_cond, levels = c("normal", "tumor"))
            )
            rownames(._mqp_cd) <- colnames(._mqp_assay)
            ._mqp_se <- SummarizedExperiment::SummarizedExperiment(
                assays  = list(logintensities = ._mqp_assay),
                colData = ._mqp_cd
            )
        """)
        ro.r(f"""
            ._mqp_se <- msqrob2::msqrob(
                object  = ._mqp_se,
                formula = ~condition,
                robust  = {robust_r},
                ridge   = FALSE
            )
            ._mqp_models <- SummarizedExperiment::rowData(._mqp_se)[["msqrobModels"]]
            ._mqp_params <- names(msqrob2::getCoef(._mqp_models[[1]]))
            ._mqp_L <- msqrob2::makeContrast(
                "conditiontumor = 0",
                parameterNames = ._mqp_params
            )
            ._mqp_se     <- msqrob2::hypothesisTest(object = ._mqp_se, contrast = ._mqp_L)
            ._mqp_rescol <- colnames(._mqp_L)
            ._mqp_res    <- as.data.frame(
                SummarizedExperiment::rowData(._mqp_se)[[._mqp_rescol]]
            )
        """)

    res_r    = ro.globalenv["._mqp_res"]
    logFC    = np.array(res_r.rx2("logFC"), dtype=float)
    pval_two = np.array(res_r.rx2("pval"),  dtype=float)

    # One-sided p-value (H1: logFC > 0).
    pvals_one = np.where(
        np.isfinite(pval_two),
        np.where(logFC > 0, pval_two / 2.0, 1.0 - pval_two / 2.0),
        np.nan,
    )

    # Clean up R global env.
    ro.r('rm(list = ls(pattern = "^\\\\._mqp_"))')
    return pvals_one, logFC


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_scenario(
    rng,
    n_sites: int = 5000,
    n_tumor: int = 30,
    n_normal: int = 30,
    n_paired: int = 30,
    lambda_mean: float = 0.8,
    lambda_sd: float = 0.2,
    noise_sd: float = 0.5,
    heterogeneous_lambda: bool = False,
    n_pos: int = 350,
    n_fp: int = 350,
    n_mixed: int = 150,
) -> dict:
    """Generate one synthetic dataset.

    Supports both paired (n_paired > 0) and fully unpaired (n_paired = 0)
    designs.  Paired samples are columns 0..n_paired-1 (tumor) paired with
    columns n_tumor..n_tumor+n_paired-1 (normal).  Any remaining columns are
    unpaired.

    For the unpaired benchmark call with n_tumor=30, n_normal=20, n_paired=0;
    t_idx and n_idx will be empty arrays and 'is_tumor' encodes the grouping.
    """
    n_null = n_sites - n_pos - n_fp - n_mixed
    n_samples = n_tumor + n_normal

    labels = np.array([1] * n_pos + [2] * n_fp + [3] * n_mixed + [0] * n_null)
    perm = rng.permutation(n_sites)
    labels = labels[np.argsort(perm)]

    gt_positive = ((labels == 1) | (labels == 3)).astype(float)
    is_tumor = np.array([1.0] * n_tumor + [0.0] * n_normal)

    beta_ptm = np.zeros(n_sites)
    lam = np.zeros(n_sites)
    prot_shift = np.zeros(n_sites)

    for i in range(n_sites):
        site_lam = (rng.uniform(0.2, 1.5) if heterogeneous_lambda
                    else max(0.0, rng.normal(lambda_mean, lambda_sd)))

        if labels[i] == 1:
            beta_ptm[i] = max(0.2, rng.normal(1.0, 0.3))
            lam[i] = site_lam
            prot_shift[i] = rng.normal(0, 0.15)
        elif labels[i] == 2:
            beta_ptm[i] = 0.0
            lam[i] = max(0.3, site_lam)
            prot_shift[i] = max(0.3, rng.normal(0.8, 0.3))
        elif labels[i] == 3:
            beta_ptm[i] = max(0.15, rng.normal(0.7, 0.3))
            lam[i] = max(0.3, site_lam)
            prot_shift[i] = max(0.2, rng.normal(0.5, 0.3))
        else:
            beta_ptm[i] = 0.0
            lam[i] = rng.uniform(0, 0.5)
            prot_shift[i] = rng.normal(0, 0.1)

    prot_base = rng.normal(0, 0.5, n_sites)
    protein = np.zeros((n_sites, n_samples), dtype=np.float64)
    for j in range(n_samples):
        protein[:, j] = (prot_base + prot_shift * is_tumor[j]
                         + rng.normal(0, 0.3, n_sites))

    ptm = np.zeros((n_sites, n_samples), dtype=np.float64)
    for j in range(n_samples):
        ptm[:, j] = (beta_ptm * is_tumor[j]
                     + lam * protein[:, j]
                     + rng.normal(0, noise_sd, n_sites))

    ptm[rng.random((n_sites, n_samples)) < 0.05] = np.nan
    protein[rng.random((n_sites, n_samples)) < 0.02] = np.nan

    # Paired indices (empty when n_paired=0)
    t_idx = np.arange(n_paired)
    n_idx = np.arange(n_tumor, n_tumor + n_paired)

    return {
        "ptm": ptm,
        "protein": protein,
        "is_tumor": is_tumor,
        "t_idx": t_idx,
        "n_idx": n_idx,
        "labels": labels,
        "gt_positive": gt_positive,
        "beta_ptm": beta_ptm,
        "lambda": lam,
        "prot_shift": prot_shift,
        "n_paired": n_paired,
        "n_tumor": n_tumor,
        "n_normal": n_normal,
    }


# ---------------------------------------------------------------------------
# VERSION 1 — Paired methods
# ---------------------------------------------------------------------------

def method_raw(data, min_n: int = 5):
    """Raw paired t-test on PTM deltas (no protein correction)."""
    delta = data["ptm"][:, data["t_idx"]] - data["ptm"][:, data["n_idx"]]
    p, _ = one_sided_ttest_1samp(delta, min_n=min_n)
    d = np.nanmean(delta, axis=1)
    return p, d


def method_subtract(data, min_n: int = 5):
    """Subtract: (PTM_delta - Protein_delta) → paired t-test [Wu et al. 2011]."""
    ptm_d = data["ptm"][:, data["t_idx"]] - data["ptm"][:, data["n_idx"]]
    prot_d = data["protein"][:, data["t_idx"]] - data["protein"][:, data["n_idx"]]
    sub = ptm_d - prot_d
    p, _ = one_sided_ttest_1samp(sub, min_n=min_n)
    d = np.nanmean(sub, axis=1)
    return p, d


def method_ptmanchor(data, min_n: int = 5):
    """ptmanchor paired LM: PTM_delta ~ intercept + lambda * protein_delta."""
    ptm_d = data["ptm"][:, data["t_idx"]] - data["ptm"][:, data["n_idx"]]
    prot_d = data["protein"][:, data["t_idx"]] - data["protein"][:, data["n_idx"]]
    intercepts, lambdas, p, n_obs = paired_lm_intercept_test(
        ptm_d.astype(np.float32), prot_d.astype(np.float32), min_n=min_n
    )
    return p, intercepts.astype(np.float64)


def method_msstatsPTM(data, min_n: int = 5, eb: bool = True):
    """MSstatsPTM: separate models for PTM and protein, FC subtraction,
    SE combination (independence assumption), Satterthwaite df, + EB.

    Statistical model (Kohler et al. 2023, Mol. Cell. Proteomics):
      PTM_delta     ~ N(mu_ptm,  sigma_ptm²)
      Protein_delta ~ N(mu_prot, sigma_prot²)
      adj_FC = mu_ptm - mu_prot
      SE_adj = sqrt(SE_ptm² + SE_prot²)
      df_adj = Satterthwaite

    For balanced paired design LMM((1|SUBJECT)) ≡ paired t-test (numerically
    identical), so FC and SE below match what the R package would produce.

    Empirical Bayes (eb=True, default):
      Per-site residual variances are shrunk toward a global prior
      (Smyth 2004 / limma::squeezeVar), mimicking MSstatsPTM moderated=TRUE.
    """
    n_sites = data["ptm"].shape[0]
    t_idx, n_idx = data["t_idx"], data["n_idx"]
    ptm_d = data["ptm"][:, t_idx] - data["ptm"][:, n_idx]
    prot_d = data["protein"][:, t_idx] - data["protein"][:, n_idx]

    fc_ptm = np.full(n_sites, np.nan)
    s2_ptm = np.full(n_sites, np.nan)   # residual variance = SD²
    n_ptm = np.zeros(n_sites)
    df_ptm = np.full(n_sites, np.nan)

    fc_prot = np.full(n_sites, np.nan)
    s2_prot = np.full(n_sites, np.nan)
    n_prot = np.zeros(n_sites)
    df_prot = np.full(n_sites, np.nan)

    for i in range(n_sites):
        yp = ptm_d[i];  vp = np.isfinite(yp);  np_ = int(vp.sum())
        yq = prot_d[i]; vq = np.isfinite(yq);  nq_ = int(vq.sum())
        if np_ >= min_n:
            fc_ptm[i] = float(np.mean(yp[vp]))
            s2_ptm[i] = float(np.var(yp[vp], ddof=1))
            n_ptm[i] = float(np_)
            df_ptm[i] = float(np_ - 1)
        if nq_ >= min_n:
            fc_prot[i] = float(np.mean(yq[vq]))
            s2_prot[i] = float(np.var(yq[vq], ddof=1))
            n_prot[i] = float(nq_)
            df_prot[i] = float(nq_ - 1)

    # EB: shrink residual variances across sites (exact limma::squeezeVar if rpy2 available)
    if eb:
        s2_ptm, df_ptm, *_ = _limma_squeeze_var(s2_ptm, df_ptm)
        s2_prot, df_prot, *_ = _limma_squeeze_var(s2_prot, df_prot)

    pvals = np.full(n_sites, np.nan)
    adj_fc = np.full(n_sites, np.nan)

    for i in range(n_sites):
        if not (np.isfinite(fc_ptm[i]) and np.isfinite(fc_prot[i])):
            continue
        if not (np.isfinite(s2_ptm[i]) and np.isfinite(s2_prot[i])):
            continue
        if n_ptm[i] < 1 or n_prot[i] < 1:
            continue
        se_p = float(np.sqrt(s2_ptm[i] / n_ptm[i]))
        se_q = float(np.sqrt(s2_prot[i] / n_prot[i]))
        if se_p < 1e-12 or se_q < 1e-12:
            continue
        adj = fc_ptm[i] - fc_prot[i]
        se_adj = float(np.sqrt(se_p ** 2 + se_q ** 2))
        num = (se_p ** 2 + se_q ** 2) ** 2
        den = se_p ** 4 / max(df_ptm[i], 1.0) + se_q ** 4 / max(df_prot[i], 1.0)
        df_sat = float(num / max(den, 1e-12))
        t_stat = adj / se_adj
        pvals[i] = float(stats.t.sf(t_stat, df=df_sat))
        adj_fc[i] = adj

    return pvals, adj_fc


def method_msqrob2PTM_paired(data, min_n: int = 5,
                              robust: bool = True, eb: bool = True):
    """msqrob2PTM DPU, paired design (Demeulemeester et al. 2024).

    Pipeline
    --------
    1. DPU normalisation (per sample): adjusted_ij = PTM_ij - Protein_ij
    2. Paired differences: delta_i = adjusted_T - adjusted_N

    If rpy2 is available:
      3–5. msqrob2::msqrob(formula=~1, robust=TRUE, ridge=TRUE) +
           hypothesisTest(Intercept = 0) — exact ridge M-estimation + EB.

    Python fallback (when rpy2 unavailable):
      3a. Robust location: Huber M-estimation (IRLS) on delta  [robust=True]
          or simple mean                                        [robust=False]
      4a. Empirical Bayes variance shrinkage across sites       [eb=True]
      5a. One-sided t-test on the estimated location

    With robust=False, eb=False the fallback is mathematically identical to
    Subtract. Huber step (3a) + EB (4a) differentiate msqrob2PTM from Subtract.
    """
    # Step 1: DPU normalisation
    adjusted = data["ptm"] - data["protein"]
    # Step 2: paired differences
    delta = adjusted[:, data["t_idx"]] - adjusted[:, data["n_idx"]]

    # --- rpy2 path: exact msqrob2 (ridge regression + EB via R package) ---
    if _check_rpy2():
        try:
            return _msqrob2_fit_and_test(delta, None, robust=robust)
        except Exception as exc:
            warnings.warn(
                f"msqrob2 rpy2 call failed ({exc}); "
                "falling back to Python Huber+EB."
            )

    # --- Python fallback ---
    if robust:
        loc, se_loc, nv = _huber_one_sample(delta, min_n=min_n)
        s2 = np.where(nv >= min_n, se_loc ** 2 * nv.astype(float), np.nan)
        df = np.where(nv >= min_n, nv.astype(float) - 1.0, np.nan)
    else:
        nv = np.sum(np.isfinite(delta), axis=1).astype(int)
        loc = np.nanmean(delta, axis=1)
        s2 = np.where(nv >= min_n,
                      np.nanvar(delta, axis=1, ddof=1),
                      np.nan)
        df = np.where(nv >= min_n, nv.astype(float) - 1.0, np.nan)

    if eb:
        s2, df, *_ = _limma_squeeze_var(s2, df)

    n_sites = delta.shape[0]
    pvals = np.full(n_sites, np.nan)
    for i in range(n_sites):
        if not (np.isfinite(loc[i]) and np.isfinite(s2[i])
                and np.isfinite(df[i]) and s2[i] > 0 and nv[i] >= min_n):
            continue
        se = float(np.sqrt(s2[i] / nv[i]))
        if se < 1e-12:
            continue
        t_stat = loc[i] / se
        pvals[i] = float(stats.t.sf(t_stat, df=df[i]))

    return pvals, loc


def method_anova_resid(data, min_n: int = 5):
    """ANOVA-style: residualise protein via per-site regression → paired t-test."""
    ptm_d = data["ptm"][:, data["t_idx"]] - data["ptm"][:, data["n_idx"]]
    prot_d = data["protein"][:, data["t_idx"]] - data["protein"][:, data["n_idx"]]

    resid = np.full_like(ptm_d, np.nan)
    for i in range(ptm_d.shape[0]):
        y = ptm_d[i]; x = prot_d[i]
        v = np.isfinite(y) & np.isfinite(x)
        if v.sum() < min_n:
            continue
        slope, *_ = stats.linregress(x[v], y[v])
        resid[i, v] = y[v] - slope * x[v]

    p, _ = one_sided_ttest_1samp(resid, min_n=min_n)
    d = np.nanmean(resid, axis=1)
    return p, d


# ---------------------------------------------------------------------------
# VERSION 2 — Unpaired methods  (30T + 20N, n_paired=0)
# ---------------------------------------------------------------------------

def _tumor_normal_split(data):
    """Return (tumor_mask, normal_mask) boolean arrays over sample axis."""
    t_mask = data["is_tumor"].astype(bool)
    return t_mask, ~t_mask


def method_raw_unpaired(data, min_n: int = 5):
    """Raw Welch two-sample t-test on PTM values (no protein correction)."""
    t_mask, n_mask = _tumor_normal_split(data)
    ptm_T = data["ptm"][:, t_mask]
    ptm_N = data["ptm"][:, n_mask]
    p, _, _ = one_sided_ttest_ind(ptm_T, ptm_N, min_tumor=min_n, min_normal=min_n)
    d = np.nanmean(ptm_T, axis=1) - np.nanmean(ptm_N, axis=1)
    return p, d


def method_subtract_unpaired(data, min_n: int = 5):
    """Subtract: per-sample (PTM - Protein) → Welch two-sample t-test."""
    t_mask, n_mask = _tumor_normal_split(data)
    adj = data["ptm"] - data["protein"]
    adj_T = adj[:, t_mask]
    adj_N = adj[:, n_mask]
    p, _, _ = one_sided_ttest_ind(adj_T, adj_N, min_tumor=min_n, min_normal=min_n)
    d = np.nanmean(adj_T, axis=1) - np.nanmean(adj_N, axis=1)
    return p, d


def method_ptmanchor_unpaired(data, min_n: int = 5):
    """ptmanchor sample-level LM: PTM ~ 1 + is_tumor + protein.

    Uses ptmanchor.modeling.sample_lm_condition_test — the same function
    called by the ptmanchor pipeline when no paired samples are detected.
    The is_tumor coefficient (beta_condition) is the PTM-specific effect.
    """
    is_tumor = data["is_tumor"].astype(int)
    n_samples = len(is_tumor)
    beta_cond, _, pvals, _ = sample_lm_condition_test(
        ptm_values=data["ptm"].astype(np.float32),
        protein_values=data["protein"].astype(np.float32),
        is_tumor=is_tumor,
        covariate_matrix=np.empty((n_samples, 0), dtype=float),
        min_tumor=min_n,
        min_normal=min_n,
    )
    return pvals, beta_cond.astype(np.float64)


def method_msstatsPTM_unpaired(data, min_n: int = 5, eb: bool = True):
    """MSstatsPTM unpaired: Welch-based two-sample models + FC subtraction
    + SE combination + Satterthwaite df + Empirical Bayes.

    Without paired structure, LMM(y ~ GROUP) reduces to a two-sample t-test.
    FC_ptm  = mean(PTM_T)  - mean(PTM_N)
    SE_ptm  = sqrt(var(PTM_T)/n_T + var(PTM_N)/n_N)
    df_ptm  = Welch df
    (same for protein)
    Then: adj_FC, SE_adj, df_Satterthwaite as in the paired version.
    EB applied to the per-group residual variances.
    """
    t_mask, n_mask = _tumor_normal_split(data)
    ptm_T = data["ptm"][:, t_mask]
    ptm_N = data["ptm"][:, n_mask]
    prot_T = data["protein"][:, t_mask]
    prot_N = data["protein"][:, n_mask]
    n_sites = ptm_T.shape[0]

    # Per-site statistics for each group × modality (4 arrays)
    def _group_stats(mat_T, mat_N):
        n_T = np.sum(np.isfinite(mat_T), axis=1).astype(float)
        n_N = np.sum(np.isfinite(mat_N), axis=1).astype(float)
        mu_T = np.nanmean(mat_T, axis=1)
        mu_N = np.nanmean(mat_N, axis=1)
        s2_T = np.where(n_T >= min_n,
                        np.nanvar(mat_T, axis=1, ddof=1), np.nan)
        s2_N = np.where(n_N >= min_n,
                        np.nanvar(mat_N, axis=1, ddof=1), np.nan)
        df_T = np.where(n_T >= min_n, n_T - 1.0, np.nan)
        df_N = np.where(n_N >= min_n, n_N - 1.0, np.nan)
        fc = np.where(np.isfinite(mu_T) & np.isfinite(mu_N),
                      mu_T - mu_N, np.nan)
        # Welch SE and df for the group contrast
        se2 = np.where(n_T >= min_n, s2_T / n_T, np.nan)
        se2 += np.where(n_N >= min_n, s2_N / n_N, np.nan)
        se = np.sqrt(np.where(se2 > 0, se2, np.nan))
        return fc, se, s2_T, s2_N, df_T, df_N, n_T, n_N

    fc_ptm, se_ptm, s2_ptm_T, s2_ptm_N, df_ptm_T, df_ptm_N, n_ptm_T, n_ptm_N = \
        _group_stats(ptm_T, ptm_N)
    fc_prot, se_prot, s2_prot_T, s2_prot_N, df_prot_T, df_prot_N, n_prot_T, n_prot_N = \
        _group_stats(prot_T, prot_N)

    # EB: shrink per-group residual variances (exact limma::squeezeVar if rpy2 available)
    if eb:
        s2_ptm_T, df_ptm_T, *_ = _limma_squeeze_var(s2_ptm_T, df_ptm_T)
        s2_ptm_N, df_ptm_N, *_ = _limma_squeeze_var(s2_ptm_N, df_ptm_N)
        s2_prot_T, df_prot_T, *_ = _limma_squeeze_var(s2_prot_T, df_prot_T)
        s2_prot_N, df_prot_N, *_ = _limma_squeeze_var(s2_prot_N, df_prot_N)
        # Recompute SE after EB
        se_ptm = np.sqrt(
            np.where(n_ptm_T >= 1, s2_ptm_T / n_ptm_T, np.nan)
            + np.where(n_ptm_N >= 1, s2_ptm_N / n_ptm_N, np.nan)
        )
        se_prot = np.sqrt(
            np.where(n_prot_T >= 1, s2_prot_T / n_prot_T, np.nan)
            + np.where(n_prot_N >= 1, s2_prot_N / n_prot_N, np.nan)
        )
        df_ptm_T = np.where(np.isfinite(df_ptm_T), df_ptm_T, 1.0)
        df_ptm_N = np.where(np.isfinite(df_ptm_N), df_ptm_N, 1.0)
        df_prot_T = np.where(np.isfinite(df_prot_T), df_prot_T, 1.0)
        df_prot_N = np.where(np.isfinite(df_prot_N), df_prot_N, 1.0)

    pvals = np.full(n_sites, np.nan)
    adj_fc = np.full(n_sites, np.nan)

    for i in range(n_sites):
        if not (np.isfinite(fc_ptm[i]) and np.isfinite(fc_prot[i])):
            continue
        sp = float(se_ptm[i]) if np.isfinite(se_ptm[i]) else np.nan
        sq = float(se_prot[i]) if np.isfinite(se_prot[i]) else np.nan
        if not (np.isfinite(sp) and np.isfinite(sq) and sp > 1e-12 and sq > 1e-12):
            continue
        adj = fc_ptm[i] - fc_prot[i]
        se_adj = float(np.sqrt(sp ** 2 + sq ** 2))
        # Satterthwaite df for the combined SE
        # (treat each SE as Welch SE with its effective df)
        df_p = float(sp ** 4 / (
            (s2_ptm_T[i] / n_ptm_T[i]) ** 2 / max(df_ptm_T[i], 1.0)
            + (s2_ptm_N[i] / n_ptm_N[i]) ** 2 / max(df_ptm_N[i], 1.0)
        )) if (n_ptm_T[i] >= 1 and n_ptm_N[i] >= 1) else max(n_ptm_T[i] + n_ptm_N[i] - 2, 1.0)
        df_q = float(sq ** 4 / (
            (s2_prot_T[i] / n_prot_T[i]) ** 2 / max(df_prot_T[i], 1.0)
            + (s2_prot_N[i] / n_prot_N[i]) ** 2 / max(df_prot_N[i], 1.0)
        )) if (n_prot_T[i] >= 1 and n_prot_N[i] >= 1) else max(n_prot_T[i] + n_prot_N[i] - 2, 1.0)
        num = (sp ** 2 + sq ** 2) ** 2
        den = sp ** 4 / max(df_p, 1.0) + sq ** 4 / max(df_q, 1.0)
        df_sat = float(num / max(den, 1e-12))
        t_stat = adj / se_adj
        pvals[i] = float(stats.t.sf(t_stat, df=df_sat))
        adj_fc[i] = adj

    return pvals, adj_fc


def method_msqrob2PTM_unpaired(data, min_n: int = 5,
                                robust: bool = True, eb: bool = True):
    """msqrob2PTM DPU, unpaired design (Demeulemeester et al. 2024).

    Pipeline
    --------
    1. DPU normalisation (per sample): adjusted_ij = PTM_ij - Protein_ij

    If rpy2 is available:
      2–4. msqrob2::msqrob(formula=~condition, robust=TRUE, ridge=FALSE) +
           hypothesisTest(conditiontumor = 0) — robust + EB (no ridge).

    Python fallback (when rpy2 unavailable):
      2a. Split into tumor and normal groups
      3a. Robust (Huber) or simple location estimate per group per site
      4a. EB variance shrinkage (per group, across sites)
      5a. Welch/Satterthwaite test on the difference in group locations
    """
    t_mask, n_mask = _tumor_normal_split(data)
    adjusted = data["ptm"] - data["protein"]

    # --- rpy2 path: exact msqrob2 (robust + EB via R package, no ridge) ---
    if _check_rpy2():
        try:
            return _msqrob2_fit_and_test(
                adjusted,
                data["is_tumor"].astype(int),
                robust=robust,
            )
        except Exception as exc:
            warnings.warn(
                f"msqrob2 rpy2 call failed ({exc}); "
                "falling back to Python Huber+EB."
            )

    # --- Python fallback ---
    adj_T = adjusted[:, t_mask]
    adj_N = adjusted[:, n_mask]
    n_sites = adj_T.shape[0]

    if robust:
        mu_T, se_T, nv_T = _huber_one_sample(adj_T, min_n=min_n)
        mu_N, se_N, nv_N = _huber_one_sample(adj_N, min_n=min_n)
        s2_T = np.where(nv_T >= min_n, se_T ** 2 * nv_T.astype(float), np.nan)
        s2_N = np.where(nv_N >= min_n, se_N ** 2 * nv_N.astype(float), np.nan)
        df_T = np.where(nv_T >= min_n, nv_T.astype(float) - 1.0, np.nan)
        df_N = np.where(nv_N >= min_n, nv_N.astype(float) - 1.0, np.nan)
    else:
        nv_T = np.sum(np.isfinite(adj_T), axis=1).astype(float)
        nv_N = np.sum(np.isfinite(adj_N), axis=1).astype(float)
        mu_T = np.nanmean(adj_T, axis=1)
        mu_N = np.nanmean(adj_N, axis=1)
        s2_T = np.where(nv_T >= min_n, np.nanvar(adj_T, axis=1, ddof=1), np.nan)
        s2_N = np.where(nv_N >= min_n, np.nanvar(adj_N, axis=1, ddof=1), np.nan)
        df_T = np.where(nv_T >= min_n, nv_T - 1.0, np.nan)
        df_N = np.where(nv_N >= min_n, nv_N - 1.0, np.nan)

    if eb:
        s2_T, df_T, *_ = _limma_squeeze_var(s2_T, df_T)
        s2_N, df_N, *_ = _limma_squeeze_var(s2_N, df_N)

    pvals = np.full(n_sites, np.nan)
    effect = np.full(n_sites, np.nan)

    for i in range(n_sites):
        if not (np.isfinite(mu_T[i]) and np.isfinite(mu_N[i])):
            continue
        if not (np.isfinite(s2_T[i]) and np.isfinite(s2_N[i])):
            continue
        if nv_T[i] < min_n or nv_N[i] < min_n:
            continue
        se_t = float(np.sqrt(s2_T[i] / nv_T[i]))
        se_n = float(np.sqrt(s2_N[i] / nv_N[i]))
        if se_t < 1e-12 or se_n < 1e-12:
            continue
        diff = mu_T[i] - mu_N[i]
        se_diff = float(np.sqrt(se_t ** 2 + se_n ** 2))
        num = (se_t ** 2 + se_n ** 2) ** 2
        den = se_t ** 4 / max(df_T[i], 1.0) + se_n ** 4 / max(df_N[i], 1.0)
        df_sat = float(num / max(den, 1e-12))
        t_stat = diff / se_diff
        pvals[i] = float(stats.t.sf(t_stat, df=df_sat))
        effect[i] = diff

    return pvals, effect


def method_anova_resid_unpaired(data, min_n: int = 5):
    """ANOVA-style unpaired: residualise protein via per-site regression
    (ignoring group), then Welch two-sample t-test on residuals.

    For each site:
      1. Regress PTM on Protein across all samples (ignoring group label)
         to estimate the protein coupling slope.
      2. Compute residuals: resid = PTM - slope * Protein
      3. Compare residuals between tumor and normal via Welch t-test.
    """
    t_mask, n_mask = _tumor_normal_split(data)
    n_sites = data["ptm"].shape[0]
    pvals = np.full(n_sites, np.nan)
    effect = np.full(n_sites, np.nan)

    for i in range(n_sites):
        ptm_all = data["ptm"][i]
        prot_all = data["protein"][i]
        valid = np.isfinite(ptm_all) & np.isfinite(prot_all)
        if valid.sum() < min_n:
            continue
        slope, *_ = stats.linregress(prot_all[valid], ptm_all[valid])
        resid = ptm_all - slope * prot_all

        r_t = resid[t_mask & valid]
        r_n = resid[n_mask & valid]
        if len(r_t) < min_n or len(r_n) < min_n:
            continue
        _, p = stats.ttest_ind(r_t, r_n, equal_var=False, alternative="greater")
        pvals[i] = float(p)
        effect[i] = float(np.mean(r_t) - np.mean(r_n))

    return pvals, effect


# ---------------------------------------------------------------------------
# Method registries
# ---------------------------------------------------------------------------

# VERSION 1: Main Figure — paired, all six methods with improved MSstatsPTM/msqrob2
METHODS_PAIRED = {
    "Raw":             method_raw,
    "Subtract":        method_subtract,
    "ptmanchor_LM":    method_ptmanchor,
    "MSstatsPTM":      method_msstatsPTM,          # + EB
    "msqrob2PTM_DPU":  method_msqrob2PTM_paired,   # + Huber + EB
    "ANOVA_resid":     method_anova_resid,
}

# VERSION 2: Supplementary — unpaired (30T + 20N), all six methods
METHODS_UNPAIRED = {
    "Raw":             method_raw_unpaired,
    "Subtract":        method_subtract_unpaired,
    "ptmanchor_LM":    method_ptmanchor_unpaired,   # sample_lm_condition_test
    "MSstatsPTM":      method_msstatsPTM_unpaired,  # Welch + EB
    "msqrob2PTM_DPU":  method_msqrob2PTM_unpaired,  # DPU + Huber + EB
    "ANOVA_resid":     method_anova_resid_unpaired,
}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

# VERSION 1 scenarios: all paired (n_tumor = n_normal = n_paired)
SCENARIOS_PAIRED = {
    "A_default":         dict(n_tumor=30, n_normal=30, n_paired=30,
                              lambda_mean=0.8, noise_sd=0.5),
    "B_weak_coupling":   dict(n_tumor=30, n_normal=30, n_paired=30,
                              lambda_mean=0.3, noise_sd=0.5),
    "C_strong_coupling": dict(n_tumor=30, n_normal=30, n_paired=30,
                              lambda_mean=1.2, noise_sd=0.5),
    "D_high_noise":      dict(n_tumor=30, n_normal=30, n_paired=30,
                              lambda_mean=0.8, noise_sd=1.0),
    "E_small_sample":    dict(n_tumor=10, n_normal=10, n_paired=10,
                              lambda_mean=0.8, noise_sd=0.5),
    "F_hetero_lambda":   dict(n_tumor=30, n_normal=30, n_paired=30,
                              lambda_mean=0.8, noise_sd=0.5,
                              heterogeneous_lambda=True),
}

# VERSION 2 scenarios: 30T + 20N, fully unpaired (n_paired=0)
SCENARIOS_UNPAIRED = {
    "A_default":         dict(n_tumor=30, n_normal=20, n_paired=0,
                              lambda_mean=0.8, noise_sd=0.5),
    "B_weak_coupling":   dict(n_tumor=30, n_normal=20, n_paired=0,
                              lambda_mean=0.3, noise_sd=0.5),
    "C_strong_coupling": dict(n_tumor=30, n_normal=20, n_paired=0,
                              lambda_mean=1.2, noise_sd=0.5),
    "D_high_noise":      dict(n_tumor=30, n_normal=20, n_paired=0,
                              lambda_mean=0.8, noise_sd=1.0),
    "E_small_sample":    dict(n_tumor=15, n_normal=10, n_paired=0,
                              lambda_mean=0.8, noise_sd=0.5),
    "F_hetero_lambda":   dict(n_tumor=30, n_normal=20, n_paired=0,
                              lambda_mean=0.8, noise_sd=0.5,
                              heterogeneous_lambda=True),
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(gt, pvals, deltas, fdr: float = 0.05, delta_cut: float = 0.2):
    """Compute PR-AUC, ROC-AUC, sensitivity, precision, observed FDR, TP/FP/FN/TN."""
    valid = np.isfinite(pvals) & np.isfinite(gt)
    if valid.sum() < 10:
        return {k: np.nan for k in [
            "PR_AUC", "ROC_AUC", "sensitivity", "specificity",
            "precision", "observed_FDR", "n_called", "TP", "FP", "FN", "TN",
        ]}
    g = gt[valid].astype(int)
    p = pvals[valid]
    d = deltas[valid] if deltas is not None else np.ones_like(p)
    scores = -np.log10(np.clip(p, 1e-300, 1.0))
    q = bh_qvalues(p)

    called = (q < fdr) & (d > delta_cut)
    tp = int(np.sum(called & (g == 1)))
    fp = int(np.sum(called & (g == 0)))
    fn = int(np.sum(~called & (g == 1)))
    tn = int(np.sum(~called & (g == 0)))

    try:
        pr_auc = float(average_precision_score(g, scores))
    except ValueError:
        pr_auc = np.nan
    try:
        roc_auc = float(roc_auc_score(g, scores))
    except ValueError:
        roc_auc = np.nan

    return {
        "PR_AUC":       round(pr_auc, 4) if np.isfinite(pr_auc) else np.nan,
        "ROC_AUC":      round(roc_auc, 4) if np.isfinite(roc_auc) else np.nan,
        "sensitivity":  round(tp / max(tp + fn, 1), 4),
        "specificity":  round(tn / max(tn + fp, 1), 4),
        "precision":    round(tp / max(tp + fp, 1), 4),
        "observed_FDR": round(fp / max(tp + fp, 1), 4),
        "n_called":     int(called.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def evaluate_fp_removal(gt_labels, pvals, fdr: float = 0.05):
    """Count how many protein-driven FP sites each method calls."""
    fp_mask = gt_labels == 2
    q = bh_qvalues(pvals)
    valid = np.isfinite(q)
    called_fp = int(np.sum((q < fdr) & fp_mask & valid))
    total_fp = int(fp_mask.sum())
    return called_fp, total_fp


def pr_roc_points(gt, pvals, method: str):
    """Extract PR and ROC curve points for plotting."""
    valid = np.isfinite(pvals) & np.isfinite(gt)
    g = gt[valid].astype(int)
    scores = -np.log10(np.clip(pvals[valid], 1e-300, 1.0))
    rows = []
    try:
        prec, rec, _ = precision_recall_curve(g, scores)
        for p, r in zip(prec, rec):
            rows.append({"method": method, "curve": "PR",
                         "x": float(r), "y": float(p)})
    except ValueError:
        pass
    try:
        fpr, tpr, _ = roc_curve(g, scores)
        for f, t in zip(fpr, tpr):
            rows.append({"method": method, "curve": "ROC",
                         "x": float(f), "y": float(t)})
    except ValueError:
        pass
    return rows


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(methods, scenarios, rng, n_reps: int, version_tag: str):
    """Run all scenarios × reps × methods. Returns raw DataFrames."""
    all_metrics, all_fp, all_curves, all_runtimes = [], [], [], []

    for scenario_name, scenario_kwargs in scenarios.items():
        print(f"  [{version_tag}] Scenario: {scenario_name}")
        for rep in range(n_reps):
            data = generate_scenario(rng, **scenario_kwargs)
            for method_name, method_fn in methods.items():
                t0 = time.perf_counter()
                pvals, deltas = method_fn(data)
                elapsed = time.perf_counter() - t0

                metrics = evaluate(data["gt_positive"], pvals, deltas)
                metrics.update(method=method_name, scenario=scenario_name,
                               rep=rep, runtime_sec=round(elapsed, 4),
                               version=version_tag)
                all_metrics.append(metrics)

                all_runtimes.append(dict(
                    method=method_name, scenario=scenario_name,
                    rep=rep, runtime_sec=round(elapsed, 4),
                    version=version_tag,
                ))

                called_fp, total_fp = evaluate_fp_removal(data["labels"], pvals)
                all_fp.append(dict(
                    method=method_name, scenario=scenario_name, rep=rep,
                    called_fp=called_fp, total_fp=total_fp,
                    version=version_tag,
                ))

                if rep == 0:
                    pts = pr_roc_points(data["gt_positive"], pvals, method_name)
                    for pt in pts:
                        pt["scenario"] = scenario_name
                    all_curves.extend(pts)

        print(f"    Completed {n_reps} reps × {len(methods)} methods")

    return (pd.DataFrame(all_metrics), pd.DataFrame(all_fp),
            pd.DataFrame(all_curves), pd.DataFrame(all_runtimes))


def make_summaries(metrics_df, fp_df, runtime_df):
    """Aggregate raw results into per-scenario per-method summaries."""
    summary = (
        metrics_df.groupby(["scenario", "method"])
        .agg(
            PR_AUC_mean=("PR_AUC", "mean"),
            PR_AUC_std=("PR_AUC", "std"),
            ROC_AUC_mean=("ROC_AUC", "mean"),
            ROC_AUC_std=("ROC_AUC", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            observed_FDR_mean=("observed_FDR", "mean"),
            observed_FDR_std=("observed_FDR", "std"),
            n_called_mean=("n_called", "mean"),
            runtime_mean=("runtime_sec", "mean"),
        )
        .reset_index()
    )

    fp_summary = (
        fp_df.groupby(["scenario", "method"])
        .agg(
            called_fp_mean=("called_fp", "mean"),
            called_fp_std=("called_fp", "std"),
            total_fp=("total_fp", "first"),
        )
        .reset_index()
    )
    fp_summary["fp_removal_pct"] = (
        100 * (1 - fp_summary["called_fp_mean"] / fp_summary["total_fp"].clip(1))
    ).round(1)

    runtime_summary = (
        runtime_df.groupby("method")
        .agg(
            runtime_mean=("runtime_sec", "mean"),
            runtime_std=("runtime_sec", "std"),
            runtime_median=("runtime_sec", "median"),
            runtime_min=("runtime_sec", "min"),
            runtime_max=("runtime_sec", "max"),
            n_runs=("runtime_sec", "count"),
        )
        .reset_index()
        .sort_values("runtime_mean")
    )

    return summary, fp_summary, runtime_summary


def print_summary(summary_df, scenarios, tag: str):
    """Print a human-readable summary table to stdout."""
    print(f"\n{'='*70}")
    print(f"  {tag} — SUMMARY")
    print(f"{'='*70}")
    for scenario in scenarios:
        print(f"\n  --- {scenario} ---")
        s = (summary_df[summary_df["scenario"] == scenario]
             .sort_values("PR_AUC_mean", ascending=False))
        print(f"  {'Method':<22s} {'PR-AUC':>8s} {'Sens':>7s} {'Prec':>7s} {'FDR':>7s}")
        for _, r in s.iterrows():
            print(f"  {r['method']:<22s} {r['PR_AUC_mean']:>8.4f} "
                  f"{r['sensitivity_mean']:>7.3f} {r['precision_mean']:>7.3f} "
                  f"{r['observed_FDR_mean']:>7.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark ptmanchor vs. competitors "
                    "(paired main + unpaired supplementary)."
    )
    p.add_argument(
        "--output-dir",
        default="results/benchmark",
        help="Directory for output TSV files.",
    )
    p.add_argument("--n-reps", type=int, default=10,
                   help="Replicates per scenario (default: 10).")
    p.add_argument("--seed", type=int, default=2024,
                   help="Random seed (default: 2024).")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("BENCHMARK: ptmanchor vs. Competitors")
    print(f"  Scenarios (paired)   : {len(SCENARIOS_PAIRED)}")
    print(f"  Scenarios (unpaired) : {len(SCENARIOS_UNPAIRED)}")
    print(f"  Reps: {args.n_reps}  |  Seed: {args.seed}")
    print(f"  Output: {out}/")
    print("=" * 70)

    # -------------------------------------------------------------------
    # VERSION 1: Main Figure — Paired benchmark (all six methods + EB)
    # -------------------------------------------------------------------
    print("\n[1/2] Paired benchmark (Main Figure) ...")
    print("      MSstatsPTM: + Empirical Bayes")
    print("      msqrob2PTM: + Huber M-estimation + Empirical Bayes")
    m_p, fp_p, cur_p, rt_p = run_benchmark(
        METHODS_PAIRED, SCENARIOS_PAIRED, rng, args.n_reps, version_tag="paired"
    )
    s_p, fps_p, rts_p = make_summaries(m_p, fp_p, rt_p)

    m_p.to_csv(out / "benchmark_full_paired.tsv",            sep="\t", index=False)
    s_p.to_csv(out / "benchmark_summary_paired.tsv",         sep="\t", index=False)
    fp_p.to_csv(out / "benchmark_fp_paired.tsv",             sep="\t", index=False)
    fps_p.to_csv(out / "benchmark_fp_summary_paired.tsv",    sep="\t", index=False)
    cur_p.to_csv(out / "benchmark_curves_paired.tsv",        sep="\t", index=False)
    rt_p.to_csv(out / "benchmark_runtime_paired.tsv",        sep="\t", index=False)
    rts_p.to_csv(out / "benchmark_runtime_summary_paired.tsv", sep="\t", index=False)

    print_summary(s_p, SCENARIOS_PAIRED, "Main Figure (Paired)")

    # -------------------------------------------------------------------
    # VERSION 2: Supplementary — Unpaired benchmark (30T + 20N, n_paired=0)
    # -------------------------------------------------------------------
    print("\n[2/2] Unpaired benchmark (Supplementary, 30T + 20N) ...")
    print("      ptmanchor: sample_lm_condition_test (PTM ~ is_tumor + protein)")
    print("      MSstatsPTM: Welch two-sample + EB")
    print("      msqrob2PTM: DPU + Huber two-sample + EB")
    m_u, fp_u, cur_u, rt_u = run_benchmark(
        METHODS_UNPAIRED, SCENARIOS_UNPAIRED, rng, args.n_reps, version_tag="unpaired"
    )
    s_u, fps_u, rts_u = make_summaries(m_u, fp_u, rt_u)

    m_u.to_csv(out / "benchmark_full_unpaired.tsv",            sep="\t", index=False)
    s_u.to_csv(out / "benchmark_summary_unpaired.tsv",         sep="\t", index=False)
    fp_u.to_csv(out / "benchmark_fp_unpaired.tsv",             sep="\t", index=False)
    fps_u.to_csv(out / "benchmark_fp_summary_unpaired.tsv",    sep="\t", index=False)
    cur_u.to_csv(out / "benchmark_curves_unpaired.tsv",        sep="\t", index=False)
    rt_u.to_csv(out / "benchmark_runtime_unpaired.tsv",        sep="\t", index=False)
    rts_u.to_csv(out / "benchmark_runtime_summary_unpaired.tsv", sep="\t", index=False)

    print_summary(s_u, SCENARIOS_UNPAIRED, "Supplementary (Unpaired 30T+20N)")

    print(f"\n[OK] All results written to {out}/")
    print("  Main Figure  : benchmark_summary_paired.tsv")
    print("  Supp. Figure : benchmark_summary_unpaired.tsv")


if __name__ == "__main__":
    main()
