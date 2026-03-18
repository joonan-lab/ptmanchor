#!/usr/bin/env python3
from __future__ import annotations
"""
Kinase-Substrate Enrichment Analysis (KSEA) comparison across three
protein-correction methods: raw (no correction), subtraction, and ptmanchor LM.

Uses Fisher's exact test per kinase-cohort, BH correction within cohort,
and Fisher's combination method to combine p-values across cohorts.
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Tee: write to stdout AND a file simultaneously
# ---------------------------------------------------------------------------

class _Tee:
    """Context manager that mirrors sys.stdout to a file.

    Usage
    -----
    with _Tee(path) as tee:
        print("goes to console AND file")
    # sys.stdout is restored on exit

    All print() / sys.stdout.write() calls inside the with-block are
    forwarded to both the original stdout and the opened file.
    No existing print() calls need to be changed.
    """

    def __init__(self, filepath: str | Path):
        self._filepath = Path(filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._file = open(self._filepath, "w", encoding="utf-8")
        sys.stdout = self
        return self

    def write(self, data: str) -> None:
        self._original_stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._original_stdout.flush()
        self._file.flush()

    # Delegate any other attribute access to the original stdout so that
    # libraries that inspect sys.stdout (e.g. rich, tqdm) don't break.
    def __getattr__(self, name: str):
        return getattr(self._original_stdout, name)

    def __exit__(self, *args) -> None:
        sys.stdout = self._original_stdout
        if self._file:
            self._file.close()


# ---------------------------------------------------------------------------
# Site key extraction
# ---------------------------------------------------------------------------

def extract_site_key(row) -> str:
    """Build GENE_SITE key from a row of all_sites.tsv.

    ID format: GENE|SITE|PEPTIDE|ACCESSION  →  GENE_SITE  (e.g. "RB1_S807")
    Falls back to GENE_ if SITE part is missing.
    """
    gene = row["Gene Symbol"]
    id_parts = row["ID"].split("|")
    site = id_parts[1] if len(id_parts) > 1 else ""
    return f"{gene}_{site}"


# ---------------------------------------------------------------------------
# Fisher's combination method
# ---------------------------------------------------------------------------

def fisher_combine(pvals):
    """Combine p-values using Fisher's combination method (chi-squared).

    chi2 = -2 * sum(log(p_i)), df = 2k.
    p=1.0 contributes log(1)=0 (neutral), no upper-bound clipping needed.
    Only the lower bound is clipped (1e-300) to avoid log(0).
    """
    pvals = np.array(pvals)
    pvals = np.clip(pvals, 1e-300, 1.0)
    chi2_stat = -2.0 * np.sum(np.log(pvals))
    combined_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(pvals))
    return chi2_stat, combined_p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="KSEA comparison: raw vs subtract vs ptmanchor LM"
    )
    p.add_argument(
        "--psp-file", required=True,
        help="Path to PhosphoSitePlus Kinase_Substrate_Dataset.gz",
    )
    p.add_argument(
        "--results-base", required=True,
        help=(
            "Root directory containing per-cohort ptmanchor result directories "
            "(e.g. /path/to/results, which contains cptac_luad_ptm_correction/)"
        ),
    )
    p.add_argument(
        "--output-dir", default=None,
        help=(
            "Output directory for TSV results. "
            "Default: <results-base>/kinase_substrate_enrichment"
        ),
    )
    p.add_argument(
        "--report-file", default=None,
        help=(
            "Path to the text file where all console output is preserved. "
            "Default: <output-dir>/ksea_comparison_report.txt"
        ),
    )
    p.add_argument(
        "--cohorts",
        nargs="+",
        default=["luad", "ccrcc", "coad", "hnscc", "lscc", "pdac", "ucec"],
        help="CPTAC cohort names to include (default: 7 paired cohorts)",
    )
    p.add_argument(
        "--min-substrates", type=int, default=3,
        help="Minimum substrate sites in background to test a kinase (default: 3)",
    )
    p.add_argument(
        "--q-threshold", type=float, default=0.05,
        help="FDR threshold for significance calls (default: 0.05)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir  = args.output_dir or os.path.join(
        args.results_base, "kinase_substrate_enrichment"
    )
    report_file = args.report_file or os.path.join(
        output_dir, "ksea_comparison_report.txt"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Tee stdout to both console and report file.
    with _Tee(report_file):
        _run(args, output_dir)

    # Print outside the Tee so the path message always appears on the console
    # even if _run raised an exception (it won't reach here, but makes intent clear).
    print(f"\n[Report] Console output saved to: {report_file}", flush=True)


def _run(args, output_dir: str) -> None:
    """Execute the full KSEA comparison pipeline.

    Separated from main() so that _Tee is already active when _run starts,
    ensuring every print() call — including any raised exception tracebacks —
    is captured in the report file.
    """
    ks_file        = args.psp_file
    results_base   = args.results_base
    cohorts_list   = args.cohorts
    MIN_SUBSTRATES = args.min_substrates
    Q_THRESHOLD    = args.q_threshold

    print("=" * 80)
    print("KSEA METHOD COMPARISON")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Load PhosphoSitePlus kinase-substrate data (human-human only)
    # -----------------------------------------------------------------------
    print("\n[1] Loading PhosphoSitePlus kinase-substrate data ...")

    kinase_substrates = defaultdict(set)   # kinase → set of site_keys
    all_ks_sites = set()

    with gzip.open(ks_file, "rt", encoding="latin-1") as fh:
        for i, line in enumerate(fh):
            # Skip first 3 lines (date, license, blank) and header (line 3)
            if i < 4:
                continue
            fields = line.strip().split("\t")
            if len(fields) < 10:
                continue
            gene_kinase  = fields[0]   # GENE (kinase gene symbol)
            sub_gene     = fields[7]   # SUB_GENE
            sub_mod_rsd  = fields[9]   # SUB_MOD_RSD
            kin_org      = fields[3]   # KIN_ORGANISM
            sub_org      = fields[8]   # SUB_ORGANISM

            if kin_org != "human" or sub_org != "human":
                continue

            site_key = f"{sub_gene}_{sub_mod_rsd}"
            kinase_substrates[gene_kinase].add(site_key)
            all_ks_sites.add(site_key)

    print(f"  Loaded {len(kinase_substrates)} kinases with "
          f"{len(all_ks_sites)} unique substrate sites (human-human)")

    sub_counts = [len(v) for v in kinase_substrates.values()]
    print(f"  Substrate count distribution: "
          f"min={min(sub_counts)}, median={int(np.median(sub_counts))}, "
          f"max={max(sub_counts)}, mean={np.mean(sub_counts):.1f}")

    # -----------------------------------------------------------------------
    # 2. Identify CPTAC cohorts
    # -----------------------------------------------------------------------
    cohort_files = {}
    for cohort in cohorts_list:
        fp = os.path.join(results_base,
                          f"cptac_{cohort}_ptm_correction",
                          "phosphoproteomics", "all_sites.tsv")
        if os.path.exists(fp):
            cohort_files[cohort] = fp

    print(f"\n[2] Found {len(cohort_files)} CPTAC cohorts: "
          f"{', '.join(sorted(cohort_files.keys()))}")

    # -----------------------------------------------------------------------
    # 3. Per-cohort Fisher's exact test
    # -----------------------------------------------------------------------
    METHODS = ["raw", "subtract", "lm"]
    METHOD_COLS = {
        "raw":      "is_raw_up",
        "subtract": "is_true_subtract",
        "lm":       "is_true_lm",
    }
    # Continuous effect columns used for KSEA z-score per method
    METHOD_EFFECT_COLS = {
        "raw":      "raw_paired_delta_t_minus_n",
        "subtract": "subtract_paired_delta_t_minus_n",
        "lm":       "lm_intercept_ptm_specific",
    }

    all_results = []

    print("\n[3] Running per-cohort Fisher's exact tests ...")

    for cohort, filepath in sorted(cohort_files.items()):
        print(f"\n  --- {cohort.upper()} ---")
        df = pd.read_csv(filepath, sep="\t", dtype=str)

        df["site_key"] = df.apply(extract_site_key, axis=1)

        for method, col in METHOD_COLS.items():
            df[col] = df[col].map(
                {"True": True, "true": True, "False": False, "false": False}
            )
            df[col] = df[col].fillna(False).astype(bool)

        background = set(df["site_key"].values)
        print(f"    Total sites: {len(df)}, Background site_keys: {len(background)}")

        for method in METHODS:
            col = METHOD_COLS[method]
            true_sites = set(df.loc[df[col], "site_key"].values)
            n_true = len(true_sites)
            n_bg   = len(background)

            print(f"    [{method}] True-up sites: {n_true}")

            # Precompute KSEA global statistics for this cohort-method pair
            effect_col = METHOD_EFFECT_COLS[method]
            if effect_col in df.columns:
                eff_numeric = pd.to_numeric(df[effect_col], errors="coerce")
                site_effect_map = dict(zip(df["site_key"], eff_numeric))
                valid_eff = eff_numeric.dropna()
                ksea_global_mean = float(valid_eff.mean()) if len(valid_eff) > 0 else np.nan
                ksea_global_sd   = float(valid_eff.std())  if len(valid_eff) > 0 else np.nan
            else:
                site_effect_map  = {}
                ksea_global_mean = np.nan
                ksea_global_sd   = np.nan

            for kinase, substrates in kinase_substrates.items():
                subs_in_bg = substrates & background
                if len(subs_in_bg) < MIN_SUBSTRATES:
                    continue

                # 2×2 table (true_sites ⊆ background guaranteed: same df)
                overlap   = subs_in_bg & true_sites
                a = len(overlap)
                b = len(true_sites  - subs_in_bg)
                c = len(subs_in_bg  - true_sites)
                d = n_bg - a - b - c

                table = np.array([[a, b], [c, d]])
                odds_ratio, pval = stats.fisher_exact(table, alternative="greater")

                # KSEA z-score for this method
                if site_effect_map and pd.notna(ksea_global_sd) and ksea_global_sd > 0:
                    sub_effs = [v for sk in subs_in_bg
                                if pd.notna(v := site_effect_map.get(sk, np.nan))]
                    n_scored = len(sub_effs)
                    if n_scored > 0:
                        sub_mean = float(np.mean(sub_effs))
                        ksea_z = (sub_mean - ksea_global_mean) * np.sqrt(n_scored) / ksea_global_sd
                        ksea_p = float(1 - stats.norm.cdf(ksea_z))
                    else:
                        ksea_z, ksea_p, n_scored = np.nan, np.nan, 0
                else:
                    ksea_z, ksea_p, n_scored = np.nan, np.nan, 0

                all_results.append({
                    "cohort":             cohort,
                    "method":             method,
                    "kinase":             kinase,
                    "n_substrates_in_bg": len(subs_in_bg),
                    "n_substrates_true":  a,
                    "n_true_total":       n_true,
                    "n_background":       n_bg,
                    "odds_ratio":         odds_ratio,
                    "pvalue":             pval,
                    "overlap_sites":      ";".join(sorted(overlap)) if overlap else "",
                    "ksea_z":             ksea_z,
                    "ksea_p":             ksea_p,
                    "n_scored":           n_scored,
                })

    results_df = pd.DataFrame(all_results)
    print(f"\n  Total tests performed: {len(results_df)}")

    # -----------------------------------------------------------------------
    # 4. BH correction within each cohort-method combination
    # -----------------------------------------------------------------------
    print("\n[4] Applying BH correction within each cohort-method ...")

    results_df_list = []
    for (cohort, method), grp in pd.DataFrame(all_results).groupby(
            ["cohort", "method"]):
        grp = grp.copy()
        _, qvals, _, _ = multipletests(grp["pvalue"].values, method="fdr_bh")
        grp["qvalue"] = qvals
        # BH correction for KSEA p-values (NaN treated as 1.0, then restored)
        ksea_p_fill = grp["ksea_p"].fillna(1.0).values
        _, ksea_qvals, _, _ = multipletests(ksea_p_fill, method="fdr_bh")
        grp["ksea_q"] = np.where(grp["ksea_p"].isna(), np.nan, ksea_qvals)
        results_df_list.append(grp)
    results_df = pd.concat(results_df_list, ignore_index=True)

    # -----------------------------------------------------------------------
    # 5. Fisher's combination across cohorts
    # -----------------------------------------------------------------------
    print("\n[5] Combining across cohorts using Fisher's combination method ...")

    combined_rows = []
    for (method, kinase), grp in results_df.groupby(["method", "kinase"]):
        n_cohorts = len(grp)
        if n_cohorts < 2:
            continue
        combined_chi2, combined_p = fisher_combine(grp["pvalue"].values)
        mean_or          = grp["odds_ratio"].replace([np.inf], np.nan).mean()
        total_subs_true  = grp["n_substrates_true"].sum()
        total_subs_bg    = grp["n_substrates_in_bg"].sum()
        n_sig_cohorts    = int((grp["qvalue"] < Q_THRESHOLD).sum())

        # Fisher's combination of per-cohort KSEA p-values
        ksea_p_finite = grp["ksea_p"].dropna()
        if len(ksea_p_finite) >= 2:
            combined_ksea_chi2, combined_ksea_p = fisher_combine(ksea_p_finite.values)
        else:
            combined_ksea_chi2, combined_ksea_p = np.nan, np.nan

        combined_rows.append({
            "method":                 method,
            "kinase":                 kinase,
            "n_cohorts":              n_cohorts,
            "n_sig_cohorts":          n_sig_cohorts,
            "combined_chi2":          combined_chi2,
            "combined_p":             combined_p,
            "combined_ksea_chi2":     combined_ksea_chi2,
            "combined_ksea_p":        combined_ksea_p,
            "mean_odds_ratio":        mean_or,
            "total_substrates_true":  total_subs_true,
            "total_substrates_in_bg": total_subs_bg,
        })

    combined_df = pd.DataFrame(combined_rows)

    combined_df_list = []
    for method, grp in combined_df.groupby("method"):
        grp = grp.copy()
        _, qvals, _, _ = multipletests(grp["combined_p"].values, method="fdr_bh")
        grp["combined_q"] = qvals
        # BH correction for combined KSEA p-values
        ksea_p_fill = grp["combined_ksea_p"].fillna(1.0).values
        _, ksea_qvals, _, _ = multipletests(ksea_p_fill, method="fdr_bh")
        grp["combined_ksea_q"] = np.where(grp["combined_ksea_p"].isna(), np.nan, ksea_qvals)
        combined_df_list.append(grp)
    combined_df = pd.concat(combined_df_list, ignore_index=True)

    print(f"  Combined results: {len(combined_df)} kinase-method pairs")

    # -----------------------------------------------------------------------
    # 6. Compare the three methods
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    sig_kinases = {}
    for method in METHODS:
        mask = ((combined_df["method"] == method) &
                (combined_df["combined_q"] < Q_THRESHOLD))
        sig_kinases[method] = set(combined_df.loc[mask, "kinase"].values)

    print("\n--- Significant kinases (combined q < 0.05) per method ---")
    for method in METHODS:
        print(f"  {method:12s}: {len(sig_kinases[method]):3d} kinases")

    print("\n--- Overlap between methods ---")
    for m1 in METHODS:
        for m2 in METHODS:
            if m1 >= m2:
                continue
            overlap = sig_kinases[m1] & sig_kinases[m2]
            union   = sig_kinases[m1] | sig_kinases[m2]
            jaccard = len(overlap) / len(union) if len(union) > 0 else 0
            print(f"  {m1} & {m2}: overlap={len(overlap)}, "
                  f"union={len(union)}, Jaccard={jaccard:.3f}")

    raw_only_vs_subtract = sig_kinases["raw"] - sig_kinases["subtract"]
    raw_only_vs_lm       = sig_kinases["raw"] - sig_kinases["lm"]
    raw_only_vs_both     = sig_kinases["raw"] - (sig_kinases["subtract"] |
                                                  sig_kinases["lm"])

    print("\n--- Potential false positives (in raw but not in corrected) ---")
    print(f"  Raw-only vs subtract : {len(raw_only_vs_subtract)} kinases")
    if raw_only_vs_subtract:
        print(f"    {sorted(raw_only_vs_subtract)}")
    print(f"  Raw-only vs lm       : {len(raw_only_vs_lm)} kinases")
    if raw_only_vs_lm:
        print(f"    {sorted(raw_only_vs_lm)}")
    print(f"  Raw-only vs both     : {len(raw_only_vs_both)} kinases")
    if raw_only_vs_both:
        print(f"    {sorted(raw_only_vs_both)}")

    lm_only       = sig_kinases["lm"]       - sig_kinases["raw"]
    subtract_only = sig_kinases["subtract"] - sig_kinases["raw"]

    print("\n--- Signals revealed by correction (not in raw) ---")
    print(f"  LM-only (not in raw)       : {len(lm_only)} kinases")
    if lm_only:
        print(f"    {sorted(lm_only)}")
    print(f"  Subtract-only (not in raw) : {len(subtract_only)} kinases")
    if subtract_only:
        print(f"    {sorted(subtract_only)}")

    KEY_KINASES = ["CDK1", "CDK2", "PRKAA1", "CHEK1", "AURKB", "ATM", "ATR",
                   "CSNK2A1", "MAPK1", "MAPK3", "AKT1", "MTOR", "PLK1"]

    print("\n--- Key kinases comparison (combined chi2 / q-value) ---")
    print(f"{'Kinase':<10s} | {'raw chi2':>8s} {'raw q':>10s} {'raw sig':>7s} | "
          f"{'sub chi2':>8s} {'sub q':>10s} {'sub sig':>7s} | "
          f"{'lm chi2':>8s} {'lm q':>10s} {'lm sig':>7s}")
    print("-" * 110)

    for kinase in KEY_KINASES:
        parts = []
        for method in METHODS:
            row = combined_df[(combined_df["method"] == method) &
                              (combined_df["kinase"] == kinase)]
            if len(row) == 0:
                parts.append(f"{'N/A':>8s} {'N/A':>10s} {'':>7s}")
            else:
                chi2_val = row["combined_chi2"].values[0]
                q   = row["combined_q"].values[0]
                sig = "*" if q < Q_THRESHOLD else ""
                parts.append(f"{chi2_val:8.2f} {q:10.2e} {sig:>7s}")
        print(f"{kinase:<10s} | {' | '.join(parts)}")

    print("\n--- Per-cohort enrichment summary (q < 0.05) ---")
    cohort_method_sig = (
        results_df[results_df["qvalue"] < Q_THRESHOLD]
        .groupby(["cohort", "method"])
        .size()
        .unstack(fill_value=0)
    )
    if not cohort_method_sig.empty:
        cohort_method_sig = cohort_method_sig.reindex(columns=METHODS, fill_value=0)
        print(cohort_method_sig.to_string())

    # -----------------------------------------------------------------------
    # 7. Save results
    # -----------------------------------------------------------------------
    out_file        = os.path.join(output_dir, "ksea_method_comparison.tsv")
    per_cohort_file = os.path.join(output_dir, "ksea_per_cohort_results.tsv")

    combined_df.sort_values(["method", "combined_p"]).to_csv(
        out_file, sep="\t", index=False)
    results_df.to_csv(per_cohort_file, sep="\t", index=False)

    print(f"\n[7] Combined results saved to   : {out_file}")
    print(f"    Per-cohort results saved to : {per_cohort_file}")

    # -----------------------------------------------------------------------
    # 8. Final summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    summary_data = []
    for method in METHODS:
        n_sig  = len(sig_kinases[method])
        others = set()
        for m2 in METHODS:
            if m2 != method:
                others |= sig_kinases[m2]
        unique     = sig_kinases[method] - others
        shared_all = (sig_kinases["raw"] & sig_kinases["subtract"] &
                      sig_kinases["lm"])
        summary_data.append({
            "Method":                method,
            "Sig. kinases (q<0.05)": n_sig,
            "Unique to method":      len(unique),
            "Shared across all 3":   len(shared_all),
        })

    print(pd.DataFrame(summary_data).to_string(index=False))

    print("\n--- Jaccard similarity matrix ---")
    print(f"{'':>12s}", end="")
    for m in METHODS:
        print(f"  {m:>10s}", end="")
    print()
    for m1 in METHODS:
        print(f"{m1:>12s}", end="")
        for m2 in METHODS:
            overlap = sig_kinases[m1] & sig_kinases[m2]
            union   = sig_kinases[m1] | sig_kinases[m2]
            j = len(overlap) / len(union) if len(union) > 0 else 1.0
            print(f"  {j:10.3f}", end="")
        print()

    print("\n--- Interpretation ---")
    n_raw  = len(sig_kinases["raw"])
    n_sub  = len(sig_kinases["subtract"])
    n_lm   = len(sig_kinases["lm"])
    n_raw_only_both = len(raw_only_vs_both)
    n_corrected_new = len((sig_kinases["subtract"] | sig_kinases["lm"]) -
                           sig_kinases["raw"])
    print(f"  Raw phospho (no correction) yields {n_raw} significant kinases.")
    print(f"  Subtraction correction yields {n_sub} significant kinases.")
    print(f"  ptmanchor LM correction yields {n_lm} significant kinases.")
    print(f"  {n_raw_only_both} kinases are significant ONLY in raw (not in either corrected),")
    print(f"    suggesting these are false positives driven by protein-level changes.")
    print(f"  {n_corrected_new} kinases are found by correction but missed by raw,")
    print(f"    suggesting correction reveals genuine phospho-specific signals.")

    print("\nDone.")


if __name__ == "__main__":
    main()
