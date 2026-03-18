#!/usr/bin/env python3
"""Kinase-Substrate Enrichment Analysis (KSEA) using PhosphoSitePlus.

Loads PSP Kinase_Substrate_Dataset, maps kinase→substrate sites to
CPTAC/NSCLC-PDIA true-increase phosphosites, and performs Fisher exact test +
KSEA z-score enrichment per cohort with cross-cohort aggregation via
Fisher's combination method.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import chi2 as chi2_dist
from scipy.stats import fisher_exact, norm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptmanchor.utils import bh_qvalues
from Scripts.figure_style import COLORS, apply_style, save_figure

# ---------------------------------------------------------------------------
# 1. Load PSP kinase-substrate database
# ---------------------------------------------------------------------------

def load_psp(psp_path: str | Path) -> dict[str, set[str]]:
    """Load PhosphoSitePlus Kinase_Substrate_Dataset.

    Parameters
    ----------
    psp_path : path to Kinase_Substrate_Dataset.gz

    Returns
    -------
    dict mapping kinase gene symbol → set of site keys (e.g. "RB1_S807")
    """
    df = pd.read_csv(psp_path, sep="\t", skiprows=3, low_memory=False,
                      encoding="latin-1")
    # Keep only human kinase → human substrate pairs
    df = df[(df["KIN_ORGANISM"] == "human") & (df["SUB_ORGANISM"] == "human")].copy()
    # Build site key: SUB_GENE + "_" + SUB_MOD_RSD (e.g. "RB1_S807")
    df["site_key"] = df["SUB_GENE"].astype(str) + "_" + df["SUB_MOD_RSD"].astype(str)
    # Build kinase → substrate set mapping
    ks_db: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        kinase = str(row["GENE"])
        site = row["site_key"]
        if kinase and site and kinase != "nan" and site != "nan":
            ks_db[kinase].add(site)
    print(f"PSP loaded: {len(ks_db)} kinases, "
          f"{sum(len(v) for v in ks_db.values())} kinase-substrate pairs "
          f"(human-human)")
    return dict(ks_db)


# ---------------------------------------------------------------------------
# 2. Site key extraction helpers
# ---------------------------------------------------------------------------

def extract_site_key_cptac(site_id: str, gene: str) -> str | None:
    """Extract site key from CPTAC site ID.

    ID format: "GENE|SITE_POS|PEPTIDE|ENSEMBL"  e.g. "BCR|S122|...|ENSP..."
    Returns "BCR_S122".
    """
    parts = site_id.split("|")
    if len(parts) >= 2:
        g = parts[0].strip()
        s = parts[1].strip()
        if g and s and re.match(r"^[STY]\d+$", s):
            return f"{g}_{s}"
    return None


def extract_site_key_hyu(site_id: str, gene: str) -> str | None:
    """Extract site key from NSCLC-PDIA (HYU) site ID.

    ID format: "ACCESSION_SITE" e.g. "O95810_S398"
    Gene Symbol column: e.g. "SDPR; CAVIN2" or "SDPR"
    Returns "SDPR_S398" (using first gene symbol).
    """
    # Extract site residue+position from ID (last part after underscore)
    m = re.search(r"_([STY]\d+)$", site_id)
    if not m:
        return None
    site = m.group(1)
    # Use first gene symbol (before ";")
    if pd.isna(gene) or not str(gene).strip():
        return None
    g = str(gene).split(";")[0].strip()
    if not g:
        return None
    return f"{g}_{site}"


# ---------------------------------------------------------------------------
# 3. Statistical tests
# ---------------------------------------------------------------------------

def ksea_fisher(substrates: set[str], true_keys: set[str],
                bg_keys: set[str]) -> dict:
    """2x2 Fisher exact test: substrate × true-increase.

    Contract: true_keys MUST be a subset of bg_keys. Call run_cohort() which
    enforces true_keys = true_keys & bg_keys before invoking this function.

    Returns dict with odds_ratio, fisher_p, overlap count, etc.
    """
    # Raise ValueError on negative cell counts instead of silently masking.
    a = len(substrates & true_keys)           # substrate AND true-increase
    b = len(substrates & bg_keys) - a         # substrate AND NOT true-increase
    c = len(true_keys) - a                    # NOT substrate AND true-increase
    d = len(bg_keys) - a - b - c             # NOT substrate AND NOT true-increase
    if b < 0 or d < 0:
        raise ValueError(
            f"Negative contingency table cell (b={b}, d={d}). "
            "true_keys must be a subset of bg_keys before calling ksea_fisher()."
        )
    table = np.array([[a, b], [c, d]])
    odds_ratio, pval = fisher_exact(table, alternative="greater")
    return {
        "n_overlap": a,
        "n_substrate_in_bg": a + b,
        "odds_ratio": odds_ratio,
        "fisher_p": pval,
    }


def ksea_zscore(substrates: set[str], all_sites_df: pd.DataFrame,
                effect_col: str, site_key_col: str = "site_key") -> dict:
    """KSEA z-score: mean substrate effect vs global distribution.

    z = (mean_substrate - global_mean) * sqrt(n) / global_sd
    """
    vals = all_sites_df[effect_col].dropna()
    global_mean = vals.mean()
    global_sd = vals.std()
    if global_sd == 0 or pd.isna(global_sd):
        return {"ksea_z": np.nan, "ksea_p": np.nan, "n_scored": 0,
                "mean_substrate_effect": np.nan}
    sub_mask = all_sites_df[site_key_col].isin(substrates)
    sub_vals = all_sites_df.loc[sub_mask, effect_col].dropna()
    n = len(sub_vals)
    if n == 0:
        return {"ksea_z": np.nan, "ksea_p": np.nan, "n_scored": 0,
                "mean_substrate_effect": np.nan}
    sub_mean = sub_vals.mean()
    z = (sub_mean - global_mean) * np.sqrt(n) / global_sd
    p = 1 - norm.cdf(z)  # one-sided (enrichment = positive z)
    return {"ksea_z": z, "ksea_p": p, "n_scored": n,
            "mean_substrate_effect": sub_mean}


# ---------------------------------------------------------------------------
# 4. Per-cohort analysis
# ---------------------------------------------------------------------------

def _build_site_keys(df: pd.DataFrame, extract_fn, gene_col: str) -> pd.Series:
    """Row-wise site key extraction."""
    ids = df["ID"].astype(str)
    genes = df[gene_col].astype(str) if gene_col in df.columns else pd.Series(
        [""] * len(df), index=df.index)
    return pd.Series(
        [extract_fn(i, g) for i, g in zip(ids, genes)],
        index=df.index,
    )


def run_cohort(cohort: str, all_sites_path: Path, true_increase_path: Path,
               ks_db: dict[str, set[str]], source: str,
               min_substrates: int = 3) -> pd.DataFrame:
    """Run KSEA for a single cohort.

    Parameters
    ----------
    source : "cptac" or "hyu"
    """
    all_df = pd.read_csv(all_sites_path, sep="\t", low_memory=False)
    true_df = pd.read_csv(true_increase_path, sep="\t", low_memory=False)

    # Determine extract function and effect column
    if source == "cptac":
        extract_fn = extract_site_key_cptac
        gene_col = "Gene Symbol"
        effect_col = "lm_intercept_ptm_specific"
    else:
        extract_fn = extract_site_key_hyu
        gene_col = "Gene Symbol"
        effect_col = "paired_mean_delta_t_minus_n"

    # Build site keys (row-wise)
    all_df["site_key"] = _build_site_keys(all_df, extract_fn, gene_col)
    true_df["site_key"] = _build_site_keys(true_df, extract_fn, gene_col)
    # Drop rows where site_key is None
    all_df = all_df.dropna(subset=["site_key"])
    true_df = true_df.dropna(subset=["site_key"])

    bg_keys = set(all_df["site_key"])
    true_keys_raw = set(true_df["site_key"])

    # Intersect true_keys with bg_keys to prevent overcounting
    # when site-key extraction edge cases cause mismatches.
    true_keys = true_keys_raw & bg_keys
    n_dropped = len(true_keys_raw) - len(true_keys)
    if n_dropped > 0:
        warnings.warn(
            f"[{cohort}] {n_dropped} true_keys not found in bg_keys and excluded "
            f"from Fisher test ({len(true_keys_raw)} → {len(true_keys)}). "
            "Check for file version mismatch between all_sites.tsv and "
            "true_increase_lm.tsv.",
            stacklevel=2,
        )

    # Report matching stats
    all_psp_sites = set()
    for subs in ks_db.values():
        all_psp_sites |= subs
    matched_bg = len(bg_keys & all_psp_sites)
    matched_true = len(true_keys & all_psp_sites)
    print(f"  {cohort}: {len(bg_keys)} bg sites ({matched_bg} PSP match), "
          f"{len(true_keys)} true-increase ({matched_true} PSP match)")

    # Run for each kinase
    rows = []
    for kinase, substrates in ks_db.items():
        overlap_bg = substrates & bg_keys
        if len(overlap_bg) < min_substrates:
            continue
        fisher_res = ksea_fisher(substrates, true_keys, bg_keys)
        zscore_res = ksea_zscore(substrates, all_df, effect_col)
        overlap_sites = sorted(substrates & true_keys)
        rows.append({
            "cohort": cohort,
            "kinase": kinase,
            "n_substrates_psp": len(substrates),
            "n_substrates_in_bg": fisher_res["n_substrate_in_bg"],
            "n_overlap_true": fisher_res["n_overlap"],
            "odds_ratio": fisher_res["odds_ratio"],
            "fisher_p": fisher_res["fisher_p"],
            "ksea_z": zscore_res["ksea_z"],
            "ksea_p": zscore_res["ksea_p"],
            "n_scored": zscore_res["n_scored"],
            "mean_substrate_effect": zscore_res["mean_substrate_effect"],
            "overlap_sites": ";".join(overlap_sites) if overlap_sites else "",
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # BH correction per cohort
    result["fisher_q"] = bh_qvalues(result["fisher_p"].values)
    result["ksea_q"] = bh_qvalues(result["ksea_p"].values)
    return result


# ---------------------------------------------------------------------------
# 5. Cross-cohort aggregation
# ---------------------------------------------------------------------------

def aggregate_cross_cohort(per_cohort: pd.DataFrame,
                           q_threshold: float = 0.05) -> pd.DataFrame:
    """Aggregate KSEA results across cohorts using Fisher's combination method.

    Fisher's method: chi2 = -2 * sum(log(p_i)), df = 2k.
    p = 1.0 contributes log(1) = 0 (neutral), no clipping needed.
    Only the lower bound is clipped (1e-300) to avoid log(0).
    """
    records = []
    for kinase, grp in per_cohort.groupby("kinase"):
        n_cohorts = len(grp)
        # Fisher's combination for per-cohort Fisher p-values
        fisher_ps = grp["fisher_p"].clip(lower=1e-300).values
        fisher_chi2 = -2.0 * np.sum(np.log(fisher_ps))
        combined_fisher_p = 1 - chi2_dist.cdf(fisher_chi2, df=2 * n_cohorts)
        # Fisher's combination for per-cohort KSEA p-values
        ksea_ps = grp["ksea_p"].dropna()
        if len(ksea_ps) > 0:
            ksea_ps_clipped = np.clip(ksea_ps.values, 1e-300, 1.0)
            ksea_chi2 = -2.0 * np.sum(np.log(ksea_ps_clipped))
            combined_ksea_p = 1 - chi2_dist.cdf(ksea_chi2, df=2 * len(ksea_ps))
        else:
            ksea_chi2 = np.nan
            combined_ksea_p = np.nan
        n_sig_fisher = (grp["fisher_q"] < q_threshold).sum()
        n_sig_ksea = (grp["ksea_q"] < q_threshold).sum()
        records.append({
            "kinase": kinase,
            "n_cohorts": n_cohorts,
            "n_sig_fisher": n_sig_fisher,
            "n_sig_ksea": n_sig_ksea,
            "combined_fisher_chi2": fisher_chi2,
            "combined_fisher_p": combined_fisher_p,
            "combined_ksea_chi2": ksea_chi2,
            "combined_ksea_p": combined_ksea_p,
            "total_overlap_true": grp["n_overlap_true"].sum(),
            "median_odds_ratio": grp["odds_ratio"].median(),
            "mean_ksea_z": grp["ksea_z"].mean(),
        })
    result = pd.DataFrame(records)
    if not result.empty:
        result["combined_fisher_q"] = bh_qvalues(result["combined_fisher_p"].values)
        result["combined_ksea_q"] = bh_qvalues(result["combined_ksea_p"].values)
        result = result.sort_values("combined_fisher_p")
    return result


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def _clean_cohort(name: str) -> str:
    return name.replace("cptac_", "").replace("_ptm_correction", "").upper()


def _build_site_effect_matrix(
    kinases: list[str],
    per_cohort: pd.DataFrame,
    ks_db: dict[str, set[str]],
    all_sites_dict: dict[str, pd.DataFrame],
    max_sites_per_kinase: int = 15,
    min_cohorts: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build substrate-site × cohort effect-size matrix for selected kinases.

    Returns
    -------
    effect_mat : site × cohort effect sizes (NaN = not detected)
    is_true    : site × cohort boolean (in true-increase set)
    kinase_labels : Series mapping site row label → kinase
    """
    cohort_order = sorted(all_sites_dict.keys())
    cohort_clean = [_clean_cohort(c) for c in cohort_order]

    # Collect true-increase site keys per cohort from per_cohort overlap_sites
    true_sites_per_cohort: dict[str, set[str]] = {}
    for cohort in cohort_order:
        rows = per_cohort[per_cohort["cohort"] == cohort]
        sites = set()
        for _, r in rows.iterrows():
            if pd.notna(r["overlap_sites"]) and r["overlap_sites"]:
                sites |= set(r["overlap_sites"].split(";"))
        true_sites_per_cohort[cohort] = sites

    all_rows = []          # (kinase, site_label, {cohort: effect}, {cohort: is_true})
    seen_sites: set[str] = set()   # deduplicate across kinases
    for kinase in kinases:
        substrates = ks_db.get(kinase, set())
        # Gather all substrate sites with their effects across cohorts
        site_data: dict[str, dict] = {}  # site -> {cohort: effect}
        site_true: dict[str, dict] = {}  # site -> {cohort: True/False}
        for cohort in cohort_order:
            df = all_sites_dict[cohort]
            effect_col = ("lm_intercept_ptm_specific"
                          if "lm_intercept_ptm_specific" in df.columns
                          else "paired_mean_delta_t_minus_n")
            sub = df[df["site_key"].isin(substrates)]
            true_set = true_sites_per_cohort.get(cohort, set())
            for _, row in sub.iterrows():
                sk = row["site_key"]
                if pd.isna(sk):
                    continue
                val = row.get(effect_col)
                if pd.isna(val):
                    continue
                site_data.setdefault(sk, {})[cohort] = val
                site_true.setdefault(sk, {})[cohort] = sk in true_set

        # Select sites: present in true-increase in >= min_cohorts, or top by
        # mean effect if not enough recurrent sites
        n_true_cohorts = {
            s: sum(1 for v in site_true.get(s, {}).values() if v)
            for s in site_data
        }
        recurrent = [s for s, n in n_true_cohorts.items() if n >= min_cohorts]
        recurrent.sort(key=lambda s: -n_true_cohorts[s])
        if len(recurrent) > max_sites_per_kinase:
            recurrent = recurrent[:max_sites_per_kinase]
        elif len(recurrent) < max_sites_per_kinase:
            # Fill with top single-cohort true-increase sites by mean effect
            remaining = [s for s in site_data if s not in recurrent
                         and n_true_cohorts.get(s, 0) >= 1]
            remaining.sort(key=lambda s: -np.nanmean(
                list(site_data[s].values())))
            recurrent.extend(remaining[:max_sites_per_kinase - len(recurrent)])

        if not recurrent:
            continue

        for site in recurrent:
            if site in seen_sites:
                continue                # already assigned to a higher-ranked kinase
            seen_sites.add(site)
            # Format: "GENE_SITE" -> "GENE S123"  (split on last underscore)
            parts = site.rsplit("_", 1)
            label = f"{parts[0]} {parts[1]}" if len(parts) == 2 else site
            effects = {_clean_cohort(c): v
                       for c, v in site_data.get(site, {}).items()}
            trues = {_clean_cohort(c): v
                     for c, v in site_true.get(site, {}).items()}
            all_rows.append((kinase, label, effects, trues))

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=str)

    # Build matrices
    row_labels = [r[1] for r in all_rows]
    kinase_labels = pd.Series([r[0] for r in all_rows], index=range(len(all_rows)))
    effect_mat = pd.DataFrame(
        [{c: r[2].get(c, np.nan) for c in cohort_clean} for r in all_rows],
        index=range(len(all_rows)),
    )
    is_true = pd.DataFrame(
        [{c: r[3].get(c, False) for c in cohort_clean} for r in all_rows],
        index=range(len(all_rows)),
    )
    effect_mat.index = row_labels
    is_true.index = row_labels
    return effect_mat, is_true, kinase_labels


def plot_ksea_heatmap(per_cohort: pd.DataFrame, cross_cohort: pd.DataFrame,
                      ks_db: dict[str, set[str]],
                      all_sites_dict: dict[str, pd.DataFrame],
                      output_dir: Path, n_top: int = 10,
                      max_sites: int = 15) -> None:
    """Cell-quality kinase-substrate site x cohort heatmap."""
    from Scripts.figure_style import (CANCER_COLORS, CELL_FULL, CELL_MAX_H,
                                      italicize, panel_label)
    apply_style()

    top_kinases = cross_cohort.head(n_top)["kinase"].tolist()
    if not top_kinases:
        print("  No significant kinases for heatmap, skipping.")
        return

    effect_mat, _is_true, kinase_labels = _build_site_effect_matrix(
        top_kinases, per_cohort, ks_db, all_sites_dict,
        max_sites_per_kinase=max_sites, min_cohorts=1,
    )

    # Export CSVs to the user-specified output directory.
    data_export_dir = output_dir
    data_export_dir.mkdir(parents=True, exist_ok=True)
    if not effect_mat.empty:
        effect_mat.to_csv(data_export_dir / "ksea_effect_matrix.csv")
        kinase_labels.to_frame("kinase").to_csv(
            data_export_dir / "ksea_kinase_labels.csv")
        print(f"  Exported effect matrix ({effect_mat.shape}) for R plotting")

    if effect_mat.empty:
        print("  No substrate sites for heatmap, skipping.")
        return

    n_rows = len(effect_mat)
    n_cols = len(effect_mat.columns)

    # --- Cell-press dimensions ---
    ROW_H = 0.115
    fig_w = CELL_FULL
    fig_h = min(ROW_H * n_rows + 1.4, CELL_MAX_H)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Gridspec: [annot_track] / [kinase_label | heatmap | site_labels]
    # + colorbar at bottom
    annot_h = 0.3   # annotation track height ratio
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.0, n_cols * 0.65, 1.8],
        height_ratios=[annot_h, n_rows, 0.6],
        hspace=0.02, wspace=0.02,
    )

    ax_annot = fig.add_subplot(gs[0, 1])      # cancer type color track
    ax_label = fig.add_subplot(gs[1, 0])       # kinase group labels
    ax_heat = fig.add_subplot(gs[1, 1])        # main heatmap
    ax_site = fig.add_subplot(gs[1, 2])        # site labels (separate axis)
    ax_cbar = fig.add_subplot(gs[2, 1])        # colorbar

    # Hide unused corner cells
    for pos in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        fig.add_subplot(gs[pos[0], pos[1]]).axis("off")

    # ── Annotation track: cancer type color bar ──
    cohort_names = list(effect_mat.columns)
    annot_colors = [CANCER_COLORS.get(c, "#cccccc") for c in cohort_names]
    for j, col in enumerate(annot_colors):
        ax_annot.add_patch(plt.Rectangle(
            (j, 0), 1, 1, facecolor=col, edgecolor="white", lw=0.5))
    ax_annot.set_xlim(0, n_cols)
    ax_annot.set_ylim(0, 1)
    ax_annot.set_xticks(np.arange(n_cols) + 0.5)
    ax_annot.set_xticklabels(cohort_names, fontsize=6, fontweight="bold",
                              rotation=45, ha="right")
    ax_annot.xaxis.set_ticks_position("top")
    ax_annot.set_yticks([0.5])
    ax_annot.set_yticklabels(["Cancer\ntype"], fontsize=5, va="center")
    ax_annot.tick_params(axis="both", length=0, pad=2)
    for spine in ax_annot.spines.values():
        spine.set_visible(False)

    # ── Main heatmap ──
    data = effect_mat.values.astype(float)
    vmax = np.nanpercentile(np.abs(data[np.isfinite(data)]), 95) if np.any(
        np.isfinite(data)) else 2
    vmax = max(vmax, 0.5)
    cnorm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # NaN → light grey background
    ax_heat.set_facecolor("#f0f0f0")
    masked = np.ma.masked_invalid(data)
    im = ax_heat.pcolormesh(
        np.arange(n_cols + 1), np.arange(n_rows + 1), masked,
        cmap="RdBu_r", norm=cnorm, edgecolors="white", linewidths=0.3,
    )
    ax_heat.set_xlim(0, n_cols)
    # set_ylim(n_rows, 0) places row 0 at top; no additional invert needed.
    ax_heat.set_ylim(n_rows, 0)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # ── Site labels (right panel, italic gene names) ──
    ax_site.set_xlim(0, 1)
    ax_site.set_ylim(n_rows, 0)
    ax_site.axis("off")
    for i, label in enumerate(effect_mat.index):
        parts = label.split(" ", 1)
        if len(parts) == 2:
            display = f"{italicize(parts[0])} {parts[1]}"
        else:
            display = italicize(label)
        ax_site.text(0.03, i + 0.5, display, va="center", ha="left",
                     fontsize=5.0)

    # ── Kinase group labels (left panel) ──
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(n_rows, 0)
    ax_label.axis("off")

    kinase_palette = ["#2166AC", "#B2182B", "#1B7837", "#D6604D",
                      "#762A83", "#35978F", "#E08214", "#525252"]

    prev_kinase = None
    group_start = 0
    groups: list[tuple[str, int, int, str]] = []
    for i in range(n_rows):
        k = kinase_labels.iloc[i]
        if k != prev_kinase:
            if prev_kinase is not None:
                groups.append((prev_kinase, group_start, i,
                               kinase_palette[len(groups) % len(kinase_palette)]))
            group_start = i
            prev_kinase = k
    if prev_kinase is not None:
        groups.append((prev_kinase, group_start, n_rows,
                       kinase_palette[len(groups) % len(kinase_palette)]))

    for kinase_name, start, end, color in groups:
        mid = (start + end) / 2
        height = end - start
        # Thin colored bar
        ax_label.add_patch(plt.Rectangle(
            (0.88, start + 0.15), 0.07, height - 0.3,
            facecolor=color, edgecolor="none", clip_on=False,
        ))
        # Kinase name (italic)
        ax_label.text(0.83, mid, italicize(kinase_name), ha="right",
                      va="center", fontsize=6.5, fontweight="bold",
                      color=color)
        # Separator on heatmap
        if start > 0:
            ax_heat.axhline(start, color="white", lw=1.5)

    # ── Colorbar (horizontal, slim) ──
    cb = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cb.set_label("Effect size (LM intercept, log$_2$)", fontsize=6)
    cb.ax.tick_params(labelsize=5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)

    save_figure(fig, "figure5_ksea_heatmap", output_dir)


# ---------------------------------------------------------------------------
# 7. Cohort discovery
# ---------------------------------------------------------------------------

def discover_cohorts(cptac_glob: str, hyu_dir: str | None
                     ) -> list[tuple[str, Path, Path, str]]:
    """Discover available cohorts.

    Returns list of (cohort_name, all_sites_path, true_increase_path, source).

    glob_pattern is initialised to cptac_glob before the for-loop so
    that it is always defined even if the loop body never executes.
    """
    cohorts = []
    # CPTAC cohorts
    if "*" in cptac_glob:
        # Locate the first path component that contains a wildcard, then
        # split into a root directory and a glob sub-pattern.
        parts = cptac_glob.split("/")
        root = Path(".")
        glob_pattern = cptac_glob
        for i, p in enumerate(parts):
            if "*" in p:
                root = Path("/".join(parts[:i])) if i > 0 else Path(".")
                glob_pattern = "/".join(parts[i:])
                break
        for d in sorted(root.glob(glob_pattern)):
            phospho_dir = d / "phosphoproteomics"
            all_sites = phospho_dir / "all_sites.tsv"
            true_inc = phospho_dir / "true_increase_lm.tsv"
            if all_sites.exists() and true_inc.exists():
                # Skip cohorts with no true-increase sites (e.g. unpaired cohorts
                # like OV and GBM where paired LM correction cannot run).
                if true_inc.stat().st_size > 0:
                    n_lines = sum(1 for _ in open(true_inc))
                    if n_lines <= 1:  # header only → no true-increase sites
                        print(f"  Skipping {d.name}: no true-increase sites "
                              f"(unpaired cohort)")
                        continue
                cohorts.append((d.name, all_sites, true_inc, "cptac"))
    else:
        # Single directory
        phospho_dir = Path(cptac_glob) / "phosphoproteomics"
        if phospho_dir.exists():
            all_sites = phospho_dir / "all_sites.tsv"
            true_inc = phospho_dir / "true_increase_lm.tsv"
            if all_sites.exists() and true_inc.exists():
                # Use Path(cptac_glob).name instead of the removed `base` variable
                cohorts.append((Path(cptac_glob).name, all_sites, true_inc, "cptac"))

    return cohorts


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kinase-Substrate Enrichment Analysis (KSEA)")
    parser.add_argument("--psp-file", required=True,
                        help="Path to Kinase_Substrate_Dataset.gz")
    parser.add_argument("--cptac-glob", default="results/cptac_*_ptm_correction",
                        help="Glob pattern for CPTAC result directories")
    parser.add_argument("--hyu-dir", default=None,
                        help="(deprecated, unused)")
    parser.add_argument("--output-dir", default="results/kinase_substrate_enrichment",
                        help="Output directory for tables and R export CSVs")
    parser.add_argument("--figure-dir", default="manuscript/figures",
                        help="Output directory for figures")
    parser.add_argument("--min-substrates", type=int, default=3,
                        help="Minimum substrate sites in background to test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load PSP database
    print("Loading PhosphoSitePlus kinase-substrate database...")
    ks_db = load_psp(args.psp_file)

    # Discover cohorts
    print("\nDiscovering cohorts...")
    cohorts = discover_cohorts(args.cptac_glob, args.hyu_dir)
    print(f"Found {len(cohorts)} cohorts: "
          f"{', '.join(c[0] for c in cohorts)}")

    if not cohorts:
        print("ERROR: No cohorts found. Check paths.", file=sys.stderr)
        sys.exit(1)

    # Run per-cohort analysis
    print("\nRunning per-cohort KSEA...")
    per_cohort_dfs = []
    all_sites_dict: dict[str, pd.DataFrame] = {}
    for cohort_name, all_sites_path, true_inc_path, source in cohorts:
        print(f"\n--- {cohort_name} ---")
        result = run_cohort(cohort_name, all_sites_path, true_inc_path,
                            ks_db, source, args.min_substrates)
        if not result.empty:
            per_cohort_dfs.append(result)
        # Keep all_sites for substrate detail plot
        all_df = pd.read_csv(all_sites_path, sep="\t", low_memory=False)
        gene_col = "Gene Symbol"
        extract_fn = extract_site_key_cptac if source == "cptac" else extract_site_key_hyu
        all_df["site_key"] = _build_site_keys(all_df, extract_fn, gene_col)
        all_df = all_df.dropna(subset=["site_key"])
        all_sites_dict[cohort_name] = all_df

    if not per_cohort_dfs:
        print("ERROR: No KSEA results produced.", file=sys.stderr)
        sys.exit(1)

    per_cohort = pd.concat(per_cohort_dfs, ignore_index=True)

    # Cross-cohort aggregation
    print("\nAggregating across cohorts (Fisher's combination method)...")
    cross_cohort = aggregate_cross_cohort(per_cohort)

    # Save tables
    per_cohort_path = output_dir / "ksea_per_cohort.tsv"
    cross_cohort_path = output_dir / "ksea_cross_cohort.tsv"
    per_cohort.to_csv(per_cohort_path, sep="\t", index=False)
    cross_cohort.to_csv(cross_cohort_path, sep="\t", index=False)
    print(f"\nSaved: {per_cohort_path}")
    print(f"Saved: {cross_cohort_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"Kinases tested: {per_cohort['kinase'].nunique()}")
    sig_fisher = cross_cohort[cross_cohort["combined_fisher_q"] < 0.05]
    print(f"Significant kinases (combined Fisher q<0.05): {len(sig_fisher)}")
    if not sig_fisher.empty:
        print("Top 10 kinases:")
        for _, row in sig_fisher.head(10).iterrows():
            print(f"  {row['kinase']:12s}  n_sig={row['n_sig_fisher']:.0f}  "
                  f"OR={row['median_odds_ratio']:.2f}  "
                  f"chi2={row['combined_fisher_chi2']:.2f}  "
                  f"q={row['combined_fisher_q']:.2e}")

    # Plots
    print("\nGenerating figures...")
    plot_ksea_heatmap(per_cohort, cross_cohort, ks_db, all_sites_dict,
                      figure_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
