from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from . import __version__

from .metadata import build_sample_design, encode_covariates
from .modeling import (
    _check_rpy2,
    paired_lm_intercept_test,
    sample_lm_condition_test,
    sample_lmm_condition_test,
)
from .utils import (
    bh_qvalues,
    canonical_accession,
    classify_hit,
    extract_accession,
    nanmean_axis1,
    one_sided_ttest_ind,
    one_sided_ttest_1samp,
    parse_bool,
    safe_numeric_frame,
    sample_columns,
)


def _primary_gene_symbol(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    # Keep the first token when multiple genes are present (e.g., "A; B").
    text = text.replace("|", ";").replace(",", ";")
    first = text.split(";")[0].strip()
    return first.upper() if first else None


def build_protein_lookup(
    protein_file: Path,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    header = pd.read_csv(protein_file, sep="\t", nrows=0).columns.tolist()
    samples = sample_columns(header)
    if not samples:
        raise ValueError(f"No sample columns found in protein file: {protein_file}")

    protein_meta_keep = [c for c in ["ID", "Gene Symbol", "Description"] if c in header]
    protein = pd.read_csv(
        protein_file,
        sep="\t",
        usecols=protein_meta_keep + samples,
        low_memory=False,
    )
    protein_numeric = safe_numeric_frame(protein, samples)
    protein_acc = protein["ID"].map(extract_accession)
    protein_can = protein_acc.map(canonical_accession)

    protein_exact = (
        protein_numeric.assign(protein_accession_exact=protein_acc)
        .dropna(subset=["protein_accession_exact"])
        .groupby("protein_accession_exact", sort=False)[samples]
        .median()
    )
    protein_canonical = (
        protein_numeric.assign(protein_accession_canonical=protein_can)
        .dropna(subset=["protein_accession_canonical"])
        .groupby("protein_accession_canonical", sort=False)[samples]
        .median()
    )
    # Gene symbol is only the last matching tier, so the column stays optional.
    protein_gene_symbol = (
        protein["Gene Symbol"].map(_primary_gene_symbol)
        if "Gene Symbol" in protein.columns
        else pd.Series([None] * len(protein))
    )
    protein_gene = (
        protein_numeric.assign(protein_gene_symbol=protein_gene_symbol)
        .dropna(subset=["protein_gene_symbol"])
        .groupby("protein_gene_symbol", sort=False)[samples]
        .median()
    )
    return samples, protein_exact, protein_canonical, protein_gene


def _select_lmm_indices(
    raw_up_mask: np.ndarray,
    lm_sample_mask: np.ndarray,
    score: np.ndarray,
    max_sites: int,
) -> np.ndarray:
    n_sites = raw_up_mask.shape[0]
    if max_sites <= 0 or max_sites >= n_sites:
        return np.arange(n_sites, dtype=int)

    preferred = np.where(raw_up_mask | lm_sample_mask)[0]
    if preferred.size >= max_sites:
        order = preferred[np.argsort(score[preferred])[::-1]]
        return order[:max_sites]

    remaining = np.setdiff1d(np.arange(n_sites, dtype=int), preferred, assume_unique=False)
    out = preferred.tolist()
    if remaining.size > 0:
        order = remaining[np.argsort(score[remaining])[::-1]]
        out.extend(order[: max_sites - len(out)].tolist())
    return np.array(out, dtype=int)


def run_modality(
    modality: str,
    ptm_file: Path,
    protein_samples: list[str],
    protein_exact: pd.DataFrame,
    protein_canonical: pd.DataFrame,
    protein_gene: pd.DataFrame,
    args,
    output_dir: Path,
    sample_design_all: pd.DataFrame | None,
    covariates: list[str],
) -> dict[str, object]:
    header = pd.read_csv(ptm_file, sep="\t", nrows=0).columns.tolist()
    ptm_samples = sample_columns(header)
    samples = [c for c in ptm_samples if c in set(protein_samples)]
    if not samples:
        raise ValueError(f"No overlapping sample columns between {ptm_file} and protein file.")

    ptm_meta_keep = [c for c in ["ID", "UniProtAccession", "Gene Symbol", "Description"] if c in header]
    ptm = pd.read_csv(
        ptm_file,
        sep="\t",
        usecols=ptm_meta_keep + samples,
        low_memory=False,
    )
    ptm_numeric = safe_numeric_frame(ptm, samples)
    ptm_values = ptm_numeric.to_numpy(dtype=np.float32)
    n_sites = ptm_values.shape[0]

    ptm_acc = ptm["UniProtAccession"].map(extract_accession)
    ptm_can = ptm_acc.map(canonical_accession)
    ptm_gene = ptm["Gene Symbol"].map(_primary_gene_symbol) if "Gene Symbol" in ptm.columns else pd.Series([None] * n_sites)
    exact_hit = ptm_acc.isin(protein_exact.index).to_numpy()
    can_hit = ptm_can.isin(protein_canonical.index).to_numpy()
    canonical_hit = (~exact_hit) & can_hit
    gene_hit = (~exact_hit) & (~canonical_hit) & ptm_gene.isin(protein_gene.index).to_numpy()

    protein_by_site = np.full((n_sites, len(samples)), np.nan, dtype=np.float32)
    match_source = np.full(n_sites, "none", dtype=object)
    matched_protein_accession = np.full(n_sites, "", dtype=object)

    if exact_hit.any():
        rows = np.where(exact_hit)[0]
        keys = ptm_acc.iloc[rows].tolist()
        protein_by_site[rows, :] = protein_exact.loc[keys, samples].to_numpy(dtype=np.float32)
        match_source[rows] = "exact"
        matched_protein_accession[rows] = keys

    if canonical_hit.any():
        rows = np.where(canonical_hit)[0]
        keys = ptm_can.iloc[rows].tolist()
        protein_by_site[rows, :] = protein_canonical.loc[keys, samples].to_numpy(dtype=np.float32)
        match_source[rows] = "canonical"
        matched_protein_accession[rows] = keys

    if gene_hit.any():
        rows = np.where(gene_hit)[0]
        keys = ptm_gene.iloc[rows].tolist()
        protein_by_site[rows, :] = protein_gene.loc[keys, samples].to_numpy(dtype=np.float32)
        match_source[rows] = "gene_symbol"
        matched_protein_accession[rows] = keys

    tumor_cols = [c for c in samples if str(c).endswith("-T")]
    normal_cols = [c for c in samples if str(c).endswith("-N")]
    if not tumor_cols or not normal_cols:
        raise ValueError("Both tumor and normal sample columns are required.")

    tumor_idx = np.array([samples.index(c) for c in tumor_cols], dtype=int)
    normal_idx = np.array([samples.index(c) for c in normal_cols], dtype=int)
    is_tumor_vec = np.array([1 if c in set(tumor_cols) else 0 for c in samples], dtype=int)

    raw_t = ptm_values[:, tumor_idx]
    raw_n_matrix = ptm_values[:, normal_idx]
    protein_t = protein_by_site[:, tumor_idx]
    protein_n = protein_by_site[:, normal_idx]
    corrected_t = raw_t - protein_t
    corrected_n = raw_n_matrix - protein_n
    raw_n_t_all = np.sum(np.isfinite(raw_t), axis=1).astype(int)
    raw_n_n_all = np.sum(np.isfinite(raw_n_matrix), axis=1).astype(int)

    raw_mean_delta = nanmean_axis1(raw_t) - nanmean_axis1(raw_n_matrix)
    protein_mean_delta = nanmean_axis1(protein_t) - nanmean_axis1(protein_n)
    # -- Tier 1: Subtraction (adjusted_delta = PTM_delta - protein_delta) --
    subtract_mean_delta = nanmean_axis1(corrected_t) - nanmean_axis1(corrected_n)

    tumor_bases = {c.rsplit("-", 1)[0] for c in tumor_cols}
    normal_bases = {c.rsplit("-", 1)[0] for c in normal_cols}
    pair_bases = sorted(tumor_bases & normal_bases)
    paired_mode = len(pair_bases) > 0
    analysis_mode = "paired" if paired_mode else "unpaired"
    fallback_used = False
    fallback_reason = ""
    effective_min_tumor = int(args.min_tumor)
    effective_min_normal = int(args.min_normal)
    testable_label = f"n>={args.min_pairs}"
    paired_testable_raw_strict = 0

    # Direction of the alternative hypothesis for every site-level test.
    alternative = str(getattr(args, "alternative", "greater"))

    if paired_mode:
        pair_t_cols = [f"{b}-T" for b in pair_bases]
        pair_n_cols = [f"{b}-N" for b in pair_bases]
        pair_t_idx = np.array([samples.index(c) for c in pair_t_cols], dtype=int)
        pair_n_idx = np.array([samples.index(c) for c in pair_n_cols], dtype=int)

        raw_pair_delta = ptm_values[:, pair_t_idx] - ptm_values[:, pair_n_idx]
        protein_pair_delta = protein_by_site[:, pair_t_idx] - protein_by_site[:, pair_n_idx]
        subtract_pair_delta = raw_pair_delta - protein_pair_delta

        raw_p_paired, raw_n_metric_paired = one_sided_ttest_1samp(
            raw_pair_delta, min_n=args.min_pairs, alternative=alternative,
        )
        subtract_p_paired, subtract_n_metric_paired = one_sided_ttest_1samp(
            subtract_pair_delta, min_n=args.min_pairs, alternative=alternative,
        )
        # -- Tier 2: Paired linear model (PTM_delta ~ intercept + lambda * protein_delta) --
        lm_intercept_paired, lm_lambda_paired, lm_p_paired, lm_n_metric_paired = paired_lm_intercept_test(
            raw_pair_delta,
            protein_pair_delta,
            min_n=args.min_pairs,
            use_eb=not getattr(args, "no_eb", False),
            lambda_shrinkage=not getattr(args, "no_lambda_shrinkage", False),
            alternative=alternative,
        )
        paired_testable_raw_strict = int(np.sum(raw_n_metric_paired >= args.min_pairs))

        force_unpaired = bool(getattr(args, "force_unpaired_if_paired", False))
        fallback_enabled = bool(getattr(args, "enable_paired_to_unpaired_fallback", False))
        min_paired_sites = max(int(getattr(args, "min_paired_testable_sites", 1)), 0)
        use_unpaired = force_unpaired
        if force_unpaired:
            fallback_reason = "force_unpaired_if_paired=true"
        elif fallback_enabled and paired_testable_raw_strict < min_paired_sites:
            use_unpaired = True
            fallback_reason = (
                f"paired_testable_raw={paired_testable_raw_strict} "
                f"< min_paired_testable_sites={min_paired_sites}"
            )

        if use_unpaired:
            fallback_used = True
            fallback_min_tumor = int(getattr(args, "fallback_min_tumor", 0))
            fallback_min_normal = int(getattr(args, "fallback_min_normal", 0))
            effective_min_tumor = fallback_min_tumor if fallback_min_tumor > 0 else int(args.min_tumor)
            effective_min_normal = fallback_min_normal if fallback_min_normal > 0 else int(args.min_normal)

            raw_p, raw_n_t, raw_n_n = one_sided_ttest_ind(
                raw_t,
                raw_n_matrix,
                min_tumor=effective_min_tumor,
                min_normal=effective_min_normal,
                alternative=alternative,
            )
            subtract_p, subtract_n_t, subtract_n_n = one_sided_ttest_ind(
                corrected_t,
                corrected_n,
                min_tumor=effective_min_tumor,
                min_normal=effective_min_normal,
                alternative=alternative,
            )
            raw_n_metric = np.minimum(raw_n_t, raw_n_n)
            subtract_n_metric = np.minimum(subtract_n_t, subtract_n_n)

            # Tier 2 fallback: sample-level LM when paired mode is unavailable
            lm_intercept, lm_lambda, lm_p, lm_n_metric = sample_lm_condition_test(
                ptm_values=ptm_values,
                protein_values=protein_by_site,
                is_tumor=is_tumor_vec,
                covariate_matrix=np.empty((len(samples), 0), dtype=float),
                min_tumor=effective_min_tumor,
                min_normal=effective_min_normal,
                max_sites=0,
                use_eb=not getattr(args, "no_eb", False),
                alternative=alternative,
            )
            analysis_mode = "forced_unpaired" if force_unpaired else "paired_fallback_unpaired"
            testable_label = f"tumor>={effective_min_tumor},normal>={effective_min_normal}"
        else:
            raw_p = raw_p_paired
            subtract_p = subtract_p_paired
            lm_p = lm_p_paired
            raw_n_metric = raw_n_metric_paired
            subtract_n_metric = subtract_n_metric_paired
            lm_n_metric = lm_n_metric_paired
            lm_intercept = lm_intercept_paired
            lm_lambda = lm_lambda_paired
            analysis_mode = "paired"
            testable_label = f"n>={args.min_pairs}"
    else:
        raw_p, raw_n_t, raw_n_n = one_sided_ttest_ind(
            raw_t,
            raw_n_matrix,
            min_tumor=effective_min_tumor,
            min_normal=effective_min_normal,
            alternative=alternative,
        )
        subtract_p, subtract_n_t, subtract_n_n = one_sided_ttest_ind(
            corrected_t,
            corrected_n,
            min_tumor=effective_min_tumor,
            min_normal=effective_min_normal,
            alternative=alternative,
        )
        raw_n_metric = np.minimum(raw_n_t, raw_n_n)
        subtract_n_metric = np.minimum(subtract_n_t, subtract_n_n)

        # -- Tier 2: Sample-level LM for unpaired design --
        lm_intercept, lm_lambda, lm_p, lm_n_metric = sample_lm_condition_test(
            ptm_values=ptm_values,
            protein_values=protein_by_site,
            is_tumor=is_tumor_vec,
            covariate_matrix=np.empty((len(samples), 0), dtype=float),
            min_tumor=effective_min_tumor,
            min_normal=effective_min_normal,
            max_sites=0,
            use_eb=not getattr(args, "no_eb", False),
            alternative=alternative,
        )
        analysis_mode = "unpaired"
        testable_label = f"tumor>={effective_min_tumor},normal>={effective_min_normal}"

    dual_group_observed_sites = int(np.sum((raw_n_t_all > 0) & (raw_n_n_all > 0)))
    dual_group_effective_sites = int(
        np.sum((raw_n_t_all >= effective_min_tumor) & (raw_n_n_all >= effective_min_normal))
    )
    if analysis_mode == "paired":
        primary_testable_raw = int(np.sum(raw_n_metric >= args.min_pairs))
    else:
        primary_testable_raw = int(np.sum(np.isfinite(raw_p)))

    raw_q = bh_qvalues(raw_p)
    subtract_q = bh_qvalues(subtract_p)
    lm_q = bh_qvalues(lm_p)

    # Direction-aware classification; "hit" is the union of up and down.
    raw_cls = classify_hit(
        raw_mean_delta, raw_q, alpha=args.fdr_cutoff,
        threshold=args.min_corrected_delta, alternative=alternative,
    )
    subtract_cls = classify_hit(
        subtract_mean_delta, subtract_q, alpha=args.fdr_cutoff,
        threshold=args.min_corrected_delta, alternative=alternative,
    )
    lm_cls = classify_hit(
        lm_intercept, lm_q, alpha=args.fdr_cutoff,
        threshold=args.min_corrected_delta, alternative=alternative,
    )

    raw_up = raw_cls["up"]
    raw_down = raw_cls["down"]
    raw_hit = raw_cls["hit"]
    subtract_up = subtract_cls["up"]
    subtract_down = subtract_cls["down"]
    subtract_hit = subtract_cls["hit"]
    lm_up = lm_cls["up"]
    lm_down = lm_cls["down"]
    lm_hit = lm_cls["hit"]

    # Aggregate "true" mask = hit in the tested direction(s).
    subtract_true = subtract_hit
    lm_true = lm_hit

    # Protein-driven: significant in raw but no longer after correction.
    protein_driven_subtract = raw_hit & (~subtract_hit)
    protein_driven_lm = raw_hit & (~lm_hit)

    detection_rate_t = np.full(n_sites, np.nan, dtype=np.float32)
    detection_rate_n = np.full(n_sites, np.nan, dtype=np.float32)
    protein_detection_rate_t = np.full(n_sites, np.nan, dtype=np.float32)
    protein_detection_rate_n = np.full(n_sites, np.nan, dtype=np.float32)
    detection_delta = np.full(n_sites, np.nan, dtype=np.float32)
    protein_detection_delta = np.full(n_sites, np.nan, dtype=np.float32)
    detection_corrected_delta = np.full(n_sites, np.nan, dtype=np.float32)
    detection_p = np.full(n_sites, np.nan, dtype=float)
    detection_q = np.full(n_sites, np.nan, dtype=float)
    detection_true = np.zeros(n_sites, dtype=bool)
    detection_fallback_used = False
    detection_fallback_reason = ""
    detection_tested_sites = 0

    if bool(getattr(args, "enable_detection_fallback", False)):
        trigger_detection = (primary_testable_raw == 0) or bool(getattr(args, "force_detection_fallback", False))
        if trigger_detection:
            min_dual_sites = max(int(getattr(args, "min_dual_group_sites_detection", 100)), 0)
            allow_detection = bool(getattr(args, "force_detection_fallback", False)) or (
                dual_group_observed_sites >= min_dual_sites
            )
            if allow_detection:
                n_tumor_total = max(len(tumor_cols), 1)
                n_normal_total = max(len(normal_cols), 1)
                det_t = np.sum(np.isfinite(raw_t), axis=1).astype(int)
                det_n = np.sum(np.isfinite(raw_n_matrix), axis=1).astype(int)
                det_pt = np.sum(np.isfinite(protein_t), axis=1).astype(int)
                det_pn = np.sum(np.isfinite(protein_n), axis=1).astype(int)

                detection_rate_t = (det_t / n_tumor_total).astype(np.float32)
                detection_rate_n = (det_n / n_normal_total).astype(np.float32)
                protein_detection_rate_t = (det_pt / n_tumor_total).astype(np.float32)
                protein_detection_rate_n = (det_pn / n_normal_total).astype(np.float32)
                detection_delta = detection_rate_t - detection_rate_n
                protein_detection_delta = protein_detection_rate_t - protein_detection_rate_n
                detection_corrected_delta = detection_delta - protein_detection_delta

                # Fisher exact direction mirrors the site-level alternative.
                fisher_alt = "greater" if alternative == "greater" else (
                    "less" if alternative == "less" else "two-sided"
                )
                for i in range(n_sites):
                    a = int(det_t[i])
                    c = int(det_n[i])
                    b = n_tumor_total - a
                    d = n_normal_total - c
                    try:
                        _odds, pval = fisher_exact([[a, b], [c, d]], alternative=fisher_alt)
                    except Exception:
                        pval = np.nan
                    detection_p[i] = pval

                detection_q = bh_qvalues(detection_p)
                det_min_delta = float(getattr(args, "min_detection_delta", 0.10))
                detection_cls = classify_hit(
                    detection_corrected_delta, detection_q,
                    alpha=args.fdr_cutoff, threshold=det_min_delta,
                    alternative=alternative,
                )
                detection_true = detection_cls["hit"]
                detection_fallback_used = True
                if bool(getattr(args, "force_detection_fallback", False)):
                    detection_fallback_reason = "force_detection_fallback=true"
                elif primary_testable_raw == 0:
                    detection_fallback_reason = "primary_testable_raw=0"
                else:
                    detection_fallback_reason = "requested"
                detection_tested_sites = int(np.sum(np.isfinite(detection_p)))
            else:
                detection_fallback_reason = (
                    f"dual_group_observed_sites={dual_group_observed_sites} "
                    f"< min_dual_group_sites_detection={min_dual_sites}"
                )

    # Optional sample-level LM/LMM models with covariates.
    lm_sample_beta = np.full(n_sites, np.nan, dtype=np.float32)
    lm_sample_beta_protein = np.full(n_sites, np.nan, dtype=np.float32)
    lm_sample_p = np.full(n_sites, np.nan, dtype=float)
    lm_sample_q = np.full(n_sites, np.nan, dtype=float)
    lm_sample_n = np.zeros(n_sites, dtype=int)
    lm_sample_true = np.zeros(n_sites, dtype=bool)

    lmm_beta = np.full(n_sites, np.nan, dtype=np.float32)
    lmm_beta_protein = np.full(n_sites, np.nan, dtype=np.float32)
    lmm_p = np.full(n_sites, np.nan, dtype=float)
    lmm_q = np.full(n_sites, np.nan, dtype=float)
    lmm_n = np.zeros(n_sites, dtype=int)
    lmm_fitted = np.zeros(n_sites, dtype=bool)
    lmm_true = np.zeros(n_sites, dtype=bool)

    design = None
    cov_encoded = pd.DataFrame(index=np.arange(len(samples)))
    if sample_design_all is not None:
        design = sample_design_all.set_index("sample_id").reindex(samples).reset_index()
        cov_encoded = encode_covariates(design, covariates)

    # -- Tier 3: Sample-level regression (PTM ~ is_tumor + protein + covariates) --
    if args.enable_sample_lm:
        if design is None:
            raise ValueError("sample_design_all is required for sample-level LM.")
        lm_sample_beta, lm_sample_beta_protein, lm_sample_p, lm_sample_n = sample_lm_condition_test(
            ptm_values=ptm_values,
            protein_values=protein_by_site,
            is_tumor=design["is_tumor"].to_numpy(dtype=int),
            covariate_matrix=cov_encoded.to_numpy(dtype=float) if not cov_encoded.empty else np.empty((len(samples), 0)),
            min_tumor=effective_min_tumor,
            min_normal=effective_min_normal,
            max_sites=args.max_sites_sample_lm,
            use_eb=not getattr(args, "no_eb", False),
            alternative=alternative,
        )
        lm_sample_q = bh_qvalues(lm_sample_p)
        lm_sample_cls = classify_hit(
            lm_sample_beta, lm_sample_q, alpha=args.fdr_cutoff,
            threshold=args.min_corrected_delta, alternative=alternative,
        )
        lm_sample_true = lm_sample_cls["hit"]

    if args.enable_sample_lmm:
        if design is None:
            raise ValueError("sample_design_all is required for sample-level LMM.")
        priority_score = np.nan_to_num(raw_mean_delta, nan=-1e9)
        selected_idx = _select_lmm_indices(
            raw_up_mask=raw_up,
            lm_sample_mask=lm_sample_true,
            score=priority_score,
            max_sites=args.max_sites_sample_lmm,
        )
        lmm_beta, lmm_beta_protein, lmm_p, lmm_n, lmm_fitted = sample_lmm_condition_test(
            ptm_values=ptm_values,
            protein_values=protein_by_site,
            is_tumor=design["is_tumor"].to_numpy(dtype=int),
            patient_ids=design["patient_id"].astype(str).to_numpy(),
            covariate_df=cov_encoded,
            min_tumor=effective_min_tumor,
            min_normal=effective_min_normal,
            selected_indices=selected_idx,
            maxiter=args.lmm_maxiter,
            alternative=alternative,
        )
        lmm_q = bh_qvalues(lmm_p)
        lmm_cls = classify_hit(
            lmm_beta, lmm_q, alpha=args.fdr_cutoff,
            threshold=args.min_corrected_delta, alternative=alternative,
        )
        lmm_true = lmm_cls["hit"]

    both_true = subtract_true & lm_true
    subtract_only = subtract_true & (~lm_true)
    lm_only = lm_true & (~subtract_true)

    results = ptm.loc[:, ptm_meta_keep].copy()
    results["site_accession_exact"] = ptm_acc
    results["site_accession_canonical"] = ptm_can
    results["protein_match_source"] = match_source
    results["matched_protein_accession"] = matched_protein_accession
    results["paired_n_raw"] = raw_n_metric
    results["paired_n_subtract"] = subtract_n_metric
    results["paired_n_lm"] = lm_n_metric
    results["raw_paired_delta_t_minus_n"] = raw_mean_delta
    results["protein_paired_delta_t_minus_n"] = protein_mean_delta
    results["subtract_paired_delta_t_minus_n"] = subtract_mean_delta
    results["lm_intercept_ptm_specific"] = lm_intercept
    results["lm_lambda_protein_dependence"] = lm_lambda
    results["raw_p_one_sided"] = raw_p
    results["raw_q_bh"] = raw_q
    results["subtract_p_one_sided"] = subtract_p
    results["subtract_q_bh"] = subtract_q
    results["lm_p_intercept_one_sided"] = lm_p
    results["lm_q_bh"] = lm_q
    results["detection_rate_tumor"] = detection_rate_t
    results["detection_rate_normal"] = detection_rate_n
    results["protein_detection_rate_tumor"] = protein_detection_rate_t
    results["protein_detection_rate_normal"] = protein_detection_rate_n
    results["detection_delta_t_minus_n"] = detection_delta
    results["protein_detection_delta_t_minus_n"] = protein_detection_delta
    results["detection_corrected_delta_t_minus_n"] = detection_corrected_delta
    results["detection_p_one_sided"] = detection_p
    results["detection_q_bh"] = detection_q
    results["is_raw_up"] = raw_up
    results["is_raw_down"] = raw_down
    results["is_raw_hit"] = raw_hit
    results["is_true_subtract"] = subtract_true
    results["is_true_subtract_up"] = subtract_up
    results["is_true_subtract_down"] = subtract_down
    results["is_true_lm"] = lm_true
    results["is_true_lm_up"] = lm_up
    results["is_true_lm_down"] = lm_down
    results["is_true_detection"] = detection_true
    results["protein_driven_subtract"] = protein_driven_subtract
    results["protein_driven_lm"] = protein_driven_lm
    results["hit_overlap_subtract_and_lm"] = both_true
    results["hit_subtract_only"] = subtract_only
    results["hit_lm_only"] = lm_only

    # Sample-level LM/LMM outputs.
    results["sample_lm_n"] = lm_sample_n
    results["sample_lm_beta_condition"] = lm_sample_beta
    results["sample_lm_beta_protein"] = lm_sample_beta_protein
    results["sample_lm_p_one_sided"] = lm_sample_p
    results["sample_lm_q_bh"] = lm_sample_q
    results["is_true_sample_lm"] = lm_sample_true

    results["sample_lmm_n"] = lmm_n
    results["sample_lmm_fitted"] = lmm_fitted
    results["sample_lmm_beta_condition"] = lmm_beta
    results["sample_lmm_beta_protein"] = lmm_beta_protein
    results["sample_lmm_p_one_sided"] = lmm_p
    results["sample_lmm_q_bh"] = lmm_q
    results["is_true_sample_lmm"] = lmm_true

    results = results.sort_values(
        by=["is_true_lm", "is_true_sample_lm", "lm_q_bh", "lm_intercept_ptm_specific"],
        ascending=[False, False, True, False],
    )

    modality_dir = output_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)
    all_path = modality_dir / "all_sites.tsv"
    subtract_path = modality_dir / "true_increase_subtract.tsv"
    subtract_down_path = modality_dir / "true_decrease_subtract.tsv"
    subtract_hits_path = modality_dir / "true_hits_subtract.tsv"
    lm_path = modality_dir / "true_increase_lm.tsv"
    lm_down_path = modality_dir / "true_decrease_lm.tsv"
    lm_hits_path = modality_dir / "true_hits_lm.tsv"
    detection_path = modality_dir / "true_increase_detection.tsv"
    lm_sample_path = modality_dir / "true_increase_sample_lm.tsv"
    lmm_path = modality_dir / "true_increase_sample_lmm.tsv"
    top_path = modality_dir / f"top{args.top_n}_lm.tsv"
    summary_path = modality_dir / "summary.txt"

    subtract_up_hits = results.loc[results["is_true_subtract_up"]].copy()
    subtract_down_hits = results.loc[results["is_true_subtract_down"]].copy()
    subtract_hits = results.loc[results["is_true_subtract"]].copy()
    lm_up_hits = results.loc[results["is_true_lm_up"]].copy()
    lm_down_hits = results.loc[results["is_true_lm_down"]].copy()
    lm_hits = results.loc[results["is_true_lm"]].copy()
    detection_hits = results.loc[results["is_true_detection"]].copy()
    lm_sample_hits = results.loc[results["is_true_sample_lm"]].copy()
    lmm_hits = results.loc[results["is_true_sample_lmm"]].copy()
    top_hits = lm_hits.head(args.top_n).copy()

    results.to_csv(all_path, sep="\t", index=False)
    subtract_up_hits.to_csv(subtract_path, sep="\t", index=False)
    subtract_down_hits.to_csv(subtract_down_path, sep="\t", index=False)
    subtract_hits.to_csv(subtract_hits_path, sep="\t", index=False)
    lm_up_hits.to_csv(lm_path, sep="\t", index=False)
    lm_down_hits.to_csv(lm_down_path, sep="\t", index=False)
    lm_hits.to_csv(lm_hits_path, sep="\t", index=False)
    detection_hits.to_csv(detection_path, sep="\t", index=False)
    lm_sample_hits.to_csv(lm_sample_path, sep="\t", index=False)
    lmm_hits.to_csv(lmm_path, sep="\t", index=False)
    top_hits.to_csv(top_path, sep="\t", index=False)

    summary_lines = [
        f"modality: {modality}",
        f"ptm_file: {ptm_file}",
        f"sample_count_overlap: {len(samples)}",
        f"analysis_mode: {analysis_mode}",
        f"fallback_used: {fallback_used}",
        f"fallback_reason: {fallback_reason}",
        f"effective_min_tumor: {effective_min_tumor}",
        f"effective_min_normal: {effective_min_normal}",
        f"tumor_samples: {len(tumor_cols)}",
        f"normal_samples: {len(normal_cols)}",
        f"paired_samples: {len(pair_bases)}",
        f"total_sites: {n_sites}",
        f"protein_match_exact: {int(np.sum(exact_hit))}",
        f"protein_match_canonical: {int(np.sum(canonical_hit))}",
        f"protein_match_gene: {int(np.sum(gene_hit))}",
        f"protein_unmatched: {int(np.sum(match_source == 'none'))}",
        f"paired_testable_raw_strict (n>={args.min_pairs}): {paired_testable_raw_strict}",
        f"primary_testable_raw ({testable_label}): {primary_testable_raw}",
        f"dual_group_observed_sites (>=1T and >=1N): {dual_group_observed_sites}",
        (
            "dual_group_effective_sites "
            f"(tumor>={effective_min_tumor},normal>={effective_min_normal}): {dual_group_effective_sites}"
        ),
        f"detection_fallback_used: {detection_fallback_used}",
        f"detection_fallback_reason: {detection_fallback_reason}",
        f"detection_tested_sites: {detection_tested_sites}",
        f"true_increase_detection: {int(np.sum(detection_true))}",
        f"alternative: {alternative}",
        f"raw_up_sites: {int(np.sum(raw_up))}",
        f"raw_down_sites: {int(np.sum(raw_down))}",
        f"raw_hit_sites: {int(np.sum(raw_hit))}",
        f"true_increase_subtract: {int(np.sum(subtract_up))}",
        f"true_decrease_subtract: {int(np.sum(subtract_down))}",
        f"true_subtract_hits: {int(np.sum(subtract_true))}",
        f"true_subtract_up: {int(np.sum(subtract_up))}",
        f"true_subtract_down: {int(np.sum(subtract_down))}",
        f"true_increase_lm: {int(np.sum(lm_up))}",
        f"true_decrease_lm: {int(np.sum(lm_down))}",
        f"true_lm_hits: {int(np.sum(lm_true))}",
        f"true_lm_up: {int(np.sum(lm_up))}",
        f"true_lm_down: {int(np.sum(lm_down))}",
        f"true_increase_sample_lm: {int(np.sum(lm_sample_true))}",
        f"true_increase_sample_lmm: {int(np.sum(lmm_true))}",
        f"sample_lmm_fitted_sites: {int(np.sum(lmm_fitted))}",
        f"protein_driven_subtract: {int(np.sum(protein_driven_subtract))}",
        f"protein_driven_lm: {int(np.sum(protein_driven_lm))}",
        f"both_methods_true: {int(np.sum(both_true))}",
        f"subtract_only_true: {int(np.sum(subtract_only))}",
        f"lm_only_true: {int(np.sum(lm_only))}",
        f"all_sites_table: {all_path}",
        f"true_subtract_up_table: {subtract_path}",
        f"true_subtract_down_table: {subtract_down_path}",
        f"true_subtract_hits_table: {subtract_hits_path}",
        f"true_lm_up_table: {lm_path}",
        f"true_lm_down_table: {lm_down_path}",
        f"true_lm_hits_table: {lm_hits_path}",
        f"true_detection_table: {detection_path}",
        f"true_sample_lm_table: {lm_sample_path}",
        f"true_sample_lmm_table: {lmm_path}",
        f"top_lm_table: {top_path}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "modality": modality,
        "status": "ok",
        "ptm_file": str(ptm_file),
        "sample_overlap": len(samples),
        "analysis_mode": analysis_mode,
        "tumor_samples": len(tumor_cols),
        "normal_samples": len(normal_cols),
        "paired_samples": len(pair_bases),
        "total_sites": int(n_sites),
        "protein_match_exact": int(np.sum(exact_hit)),
        "protein_match_canonical": int(np.sum(canonical_hit)),
        "protein_match_gene": int(np.sum(gene_hit)),
        "protein_unmatched": int(np.sum(match_source == "none")),
        "paired_testable_raw_strict": paired_testable_raw_strict,
        "paired_testable_raw": primary_testable_raw,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "effective_min_tumor": effective_min_tumor,
        "effective_min_normal": effective_min_normal,
        "dual_group_observed_sites": dual_group_observed_sites,
        "dual_group_effective_sites": dual_group_effective_sites,
        "detection_fallback_used": detection_fallback_used,
        "detection_fallback_reason": detection_fallback_reason,
        "detection_tested_sites": detection_tested_sites,
        "true_increase_detection": int(np.sum(detection_true)),
        "alternative": alternative,
        "raw_up_sites": int(np.sum(raw_up)),
        "raw_down_sites": int(np.sum(raw_down)),
        "raw_hit_sites": int(np.sum(raw_hit)),
        "true_increase_subtract": int(np.sum(subtract_up)),
        "true_decrease_subtract": int(np.sum(subtract_down)),
        "true_subtract_hits": int(np.sum(subtract_true)),
        "true_subtract_up": int(np.sum(subtract_up)),
        "true_subtract_down": int(np.sum(subtract_down)),
        "true_increase_lm": int(np.sum(lm_up)),
        "true_decrease_lm": int(np.sum(lm_down)),
        "true_lm_hits": int(np.sum(lm_true)),
        "true_lm_up": int(np.sum(lm_up)),
        "true_lm_down": int(np.sum(lm_down)),
        "true_increase_sample_lm": int(np.sum(lm_sample_true)),
        "true_increase_sample_lmm": int(np.sum(lmm_true)),
        "sample_lmm_fitted_sites": int(np.sum(lmm_fitted)),
        "protein_driven_subtract": int(np.sum(protein_driven_subtract)),
        "protein_driven_lm": int(np.sum(protein_driven_lm)),
        "both_methods_true": int(np.sum(both_true)),
        "subtract_only_true": int(np.sum(subtract_only)),
        "lm_only_true": int(np.sum(lm_only)),
        "summary_file": str(summary_path),
    }


def run_manifest(args) -> tuple[Path, Path, Path]:
    manifest_path = Path(args.manifest)
    protein_file = Path(args.protein_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    required_cols = {"modality", "ptm_file"}
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_cols)}")

    covariates = [c.strip() for c in str(args.covariates).split(",") if c.strip()] if args.covariates else []
    protein_samples, protein_exact, protein_canonical, protein_gene = build_protein_lookup(protein_file)

    sample_design = build_sample_design(
        samples=protein_samples,
        sample_meta_file=Path(args.sample_meta_file) if args.sample_meta_file else None,
        sample_meta_sheet=args.sample_meta_sheet,
        sample_id_col=args.sample_id_col,
        patient_id_col=args.patient_id_col,
        covariates=covariates,
    )

    summary_rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        modality = str(row["modality"]).strip()
        enabled = parse_bool(row["enabled"], default=True) if "enabled" in manifest.columns else True
        ptm_file = Path(str(row["ptm_file"]).strip())
        if not ptm_file.is_absolute() and not ptm_file.exists():
            manifest_relative = manifest_path.parent / ptm_file
            if manifest_relative.exists():
                ptm_file = manifest_relative
        if not enabled:
            summary_rows.append(
                {
                    "modality": modality,
                    "status": "skipped_disabled",
                    "ptm_file": str(ptm_file),
                    "reason": "enabled=false",
                }
            )
            continue
        if not ptm_file.exists():
            summary_rows.append(
                {
                    "modality": modality,
                    "status": "skipped_missing_file",
                    "ptm_file": str(ptm_file),
                    "reason": "file_not_found",
                }
            )
            continue

        try:
            row_summary = run_modality(
                modality=modality,
                ptm_file=ptm_file,
                protein_samples=protein_samples,
                protein_exact=protein_exact,
                protein_canonical=protein_canonical,
                protein_gene=protein_gene,
                args=args,
                output_dir=output_dir,
                sample_design_all=sample_design,
                covariates=covariates,
            )
        except Exception as exc:  # pragma: no cover
            row_summary = {
                "modality": modality,
                "status": "failed",
                "ptm_file": str(ptm_file),
                "reason": str(exc),
            }
        summary_rows.append(row_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_tsv = output_dir / "modality_summary.tsv"
    summary_txt = output_dir / "summary.txt"
    run_config = output_dir / "run_config.json"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)

    status_series = summary_df["status"] if "status" in summary_df.columns else pd.Series(dtype=str)
    n_ok = int(np.sum(status_series == "ok")) if not summary_df.empty else 0
    n_failed = int(np.sum(status_series == "failed")) if not summary_df.empty else 0
    n_skipped = int(np.sum(status_series.str.startswith("skipped", na=False))) if not summary_df.empty else 0

    lines = [
        "Multimodal protein-adjusted PTM correction summary",
        "",
        f"manifest: {manifest_path}",
        f"protein_file: {protein_file}",
        f"output_dir: {output_dir}",
        "",
        f"n_modalities_in_manifest: {manifest.shape[0]}",
        f"n_modalities_ok: {n_ok}",
        f"n_modalities_skipped: {n_skipped}",
        f"n_modalities_failed: {n_failed}",
        "",
        f"summary_table: {summary_tsv}",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    run_meta = {
        "ptmanchor_version": __version__,
        "manifest": str(manifest_path),
        "protein_file": str(protein_file),
        "output_dir": str(output_dir),
        "alternative": str(getattr(args, "alternative", "greater")),
        # Record the backend for provenance even though the implementations agree.
        "eb_backend": (
            "disabled" if getattr(args, "no_eb", False)
            else ("limma" if _check_rpy2() else "python")
        ),
        "eb_enabled": not bool(getattr(args, "no_eb", False)),
        "lambda_shrinkage_enabled": not bool(
            getattr(args, "no_lambda_shrinkage", False)
        ),
        "min_pairs": args.min_pairs,
        "min_tumor": args.min_tumor,
        "min_normal": args.min_normal,
        "fdr_cutoff": args.fdr_cutoff,
        "min_corrected_delta": args.min_corrected_delta,
        "top_n": args.top_n,
        "sample_meta_file": args.sample_meta_file,
        "sample_meta_sheet": args.sample_meta_sheet,
        "sample_id_col": args.sample_id_col,
        "patient_id_col": args.patient_id_col,
        "covariates": covariates,
        "enable_sample_lm": args.enable_sample_lm,
        "enable_sample_lmm": args.enable_sample_lmm,
        "max_sites_sample_lm": args.max_sites_sample_lm,
        "max_sites_sample_lmm": args.max_sites_sample_lmm,
        "lmm_maxiter": args.lmm_maxiter,
        "enable_paired_to_unpaired_fallback": args.enable_paired_to_unpaired_fallback,
        "min_paired_testable_sites": args.min_paired_testable_sites,
        "fallback_min_tumor": args.fallback_min_tumor,
        "fallback_min_normal": args.fallback_min_normal,
        "force_unpaired_if_paired": args.force_unpaired_if_paired,
        "enable_detection_fallback": args.enable_detection_fallback,
        "force_detection_fallback": args.force_detection_fallback,
        "min_detection_delta": args.min_detection_delta,
        "min_dual_group_sites_detection": args.min_dual_group_sites_detection,
    }
    run_config.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return summary_tsv, summary_txt, run_config
