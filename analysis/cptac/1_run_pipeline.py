#!/usr/bin/env python3
"""Unified CPTAC pipeline: export all cohorts, generate manifest, run ptmanchor,
and build cross-cohort summary.

Replaces the multi-step workflow of:
  1. build_cptac_manifest.py               (manifest generation)
  2. export_cptac_to_ptmanchor.py --cohort X  (per-cohort export)
  3. multimodal_ptm_correction.py ...         (per-cohort ptmanchor)
  4. build_cptac_cross_cohort_summary.py      (cross-cohort summary)

Usage:
  python export_all_cptac.py                          # export + pipeline + summary
  python export_all_cptac.py --export-only             # export only
  python export_all_cptac.py --pipeline-only            # skip export, run pipeline + summary
  python export_all_cptac.py --summary-only             # skip export & pipeline, build summary only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_COHORTS = [
    "brca",
    "ccrcc",
    "coad",
    "hnscc",
    "lscc",
    "luad",
    "ov",
    "pdac",
    "ucec",
]

COHORT_CLASS = {
    "brca": "Brca",
    "ccrcc": "Ccrcc",
    "coad": "Coad",
    "colon": "Coad",
    "hnscc": "Hnscc",
    "lscc": "Lscc",
    "luad": "Luad",
    "ov": "Ov",
    "ovarian": "Ov",
    "pdac": "Pdac",
    "ucec": "Ucec",
}

MODALITY_GETTER = {
    "proteomics": "get_proteomics",
    "phosphoproteomics": "get_phosphoproteomics",
    "acetylproteomics": "get_acetylproteomics",
}

SOURCE_PRIORITY = ["umich", "bcm", "washu", "broad", "mssm", "harmonized"]

# Override automatic source selection for specific cohort+modality combos.
# Format: { ("cohort", "modality"): "source" }
COHORT_SOURCE_OVERRIDES: dict[tuple[str, str], str] = {
    ("ov", "phosphoproteomics"): "bcm",  # umich download fails for OV phospho
}

TARGET_MODALITIES = ["proteomics", "phosphoproteomics", "acetylproteomics"]

COHORT_DIR_RE = re.compile(r"cptac_(?P<cohort>[a-z0-9]+)_ptm_correction(?:_v\d+)?$")

ACC_RE = re.compile(
    r"(A0A[A-Z0-9]{3}[A-Z0-9]{4}(?:-[0-9]+)?"
    r"|[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?"
    r"|[OPQ][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?"
    r"|ENSP[0-9]+(?:\.[0-9]+)?)"
)


# ---------------------------------------------------------------------------
# Utility helpers (from export_cptac_to_ptmanchor.py)
# ---------------------------------------------------------------------------

def extract_accession(value: object) -> str | None:
    if value is None:
        return None
    matches = ACC_RE.findall(str(value))
    return matches[0] if matches else None


def canonical_cohort_name(cohort: str) -> str:
    cohort = str(cohort).strip().lower()
    if cohort == "colon":
        return "coad"
    if cohort == "ovarian":
        return "ov"
    return cohort


def normalize_condition_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"tumor", "tumour", "primary", "t"}:
        return "T"
    if text in {"normal", "nat", "adjacent normal", "n"}:
        return "N"
    if text.endswith("-t") or text.endswith(".t") or text.endswith("_t"):
        return "T"
    if text.endswith("-n") or text.endswith(".n") or text.endswith("_n"):
        return "N"
    return ""


def make_sample_table(index: pd.Index) -> pd.DataFrame:
    if isinstance(index, pd.MultiIndex):
        lvl_names = [str(n) if n is not None else "" for n in index.names]
        tuples = list(index)
        if len(index.levels) >= 2:
            patient_vals = [str(t[0]) for t in tuples]
            cond_level = 1
            for i, nm in enumerate(lvl_names):
                if "tumor" in nm.lower() or "normal" in nm.lower() or "sample_status" in nm.lower():
                    cond_level = i
                    break
            cond_vals = [normalize_condition_label(t[cond_level]) if len(t) > cond_level else "" for t in tuples]
        else:
            patient_vals = [str(t[0]) for t in tuples]
            cond_vals = ["" for _ in tuples]
    else:
        raw_vals = [str(v) for v in index]
        patient_vals = []
        cond_vals = []
        for raw in raw_vals:
            cond = normalize_condition_label(raw)
            cond_vals.append(cond)
            if cond in {"T", "N"} and raw[-2:] in {"-T", "-N", ".T", ".N", "_T", "_N"}:
                patient_vals.append(raw[:-2])
            else:
                patient_vals.append(raw)

    # Default unknown condition to Tumor (cptac flat index: normal samples
    # have -N suffix, tumor samples often have no suffix).
    cond_vals = ["T" if c == "" else c for c in cond_vals]

    sample_ids = []
    for patient, cond in zip(patient_vals, cond_vals):
        sid = f"{patient}-{cond}"
        sample_ids.append(sid)

    sample_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient_id": patient_vals,
            "condition": ["Tumor" if c == "T" else "Normal" for c in cond_vals],
            "is_tumor": [1 if c == "T" else 0 for c in cond_vals],
        }
    )
    if sample_df["sample_id"].duplicated().any():
        counts: dict[str, int] = {}
        out_ids = []
        for sid in sample_df["sample_id"]:
            counts[sid] = counts.get(sid, 0) + 1
            out_ids.append(sid if counts[sid] == 1 else f"{sid}_{counts[sid]}")
        sample_df["sample_id"] = out_ids
    return sample_df


def flatten_feature(col: object, modality: str) -> tuple[str, str, str]:
    """Return (feature_id, uniprot_accession, gene_symbol)."""
    gene = ""
    acc = ""
    feature_id = ""

    if isinstance(col, tuple):
        parts = [str(x) for x in col if str(x) not in {"", "nan", "None"}]
        joined = "|".join(parts)
        if modality == "proteomics" and len(col) >= 2:
            gene = str(col[0])
            db = str(col[1])
            feature_id = db
            acc = db
        elif modality != "proteomics" and len(col) >= 4:
            gene = str(col[0])
            site = str(col[1])
            peptide = str(col[2])
            db = str(col[3])
            feature_id = f"{gene}|{site}|{peptide}|{db}"
            acc = db
        else:
            maybe_acc = extract_accession(joined)
            if maybe_acc:
                acc = maybe_acc
            if parts:
                gene = parts[0]
            feature_id = joined
    else:
        text = str(col)
        maybe_acc = extract_accession(text)
        if maybe_acc:
            acc = maybe_acc
        feature_id = text

    if not feature_id:
        feature_id = "NA"
    return feature_id, acc, gene


def call_getter(dataset, getter_name: str, source: str | None):
    getter = getattr(dataset, getter_name)
    if source:
        try:
            return getter(source=source)
        except TypeError:
            return getter()
    return getter()


def infer_source(dataset, cohort: str, modality: str) -> str | None:
    override = COHORT_SOURCE_OVERRIDES.get((cohort, modality))
    if override:
        return override
    if not hasattr(dataset, "list_data_sources"):
        return None
    table = dataset.list_data_sources()
    if not isinstance(table, pd.DataFrame):
        return None
    if "Data type" not in table.columns or "Available sources" not in table.columns:
        return None

    row = table.loc[table["Data type"].astype(str) == modality]
    if row.empty:
        return None
    sources = row.iloc[0]["Available sources"]
    if not isinstance(sources, (list, tuple)):
        return None
    values = [str(x).strip().lower() for x in sources if str(x).strip()]
    if not values:
        return None
    for preferred in SOURCE_PRIORITY:
        if preferred in values:
            return preferred
    return values[0]


# ---------------------------------------------------------------------------
# Export functions (from export_cptac_to_ptmanchor.py)
# ---------------------------------------------------------------------------

def export_modality_table(
    dataset,
    modality: str,
    source: str | None,
    output_file: Path,
    expected_samples: pd.DataFrame | None = None,
) -> tuple[Path, int, int]:
    getter_name = MODALITY_GETTER[modality]
    if not hasattr(dataset, getter_name):
        raise RuntimeError(f"Cohort does not support getter: {getter_name}")

    df = call_getter(dataset, getter_name, source=source)
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"{getter_name} did not return DataFrame")

    sample_df = make_sample_table(df.index)
    feature_count = df.shape[1]

    values = df.to_numpy(dtype=float).T
    sample_ids = sample_df["sample_id"].tolist()
    out = pd.DataFrame(values, columns=sample_ids)

    feature_ids = []
    accessions = []
    genes = []
    for col in df.columns:
        fid, acc, gene = flatten_feature(col, modality=modality)
        feature_ids.append(fid)
        accessions.append(acc)
        genes.append(gene)

    out.insert(0, "ID", feature_ids)
    if modality != "proteomics":
        out.insert(1, "UniProtAccession", accessions)
        out.insert(2, "Gene Symbol", genes)
    else:
        out.insert(1, "Gene Symbol", genes)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, sep="\t", index=False)

    if expected_samples is not None:
        keep = set(expected_samples["sample_id"].astype(str))
        have = set(sample_ids)
        overlap = len(keep & have)
        if overlap == 0:
            raise RuntimeError(
                f"No overlapping samples between proteomics and {modality} export."
            )
    return output_file, feature_count, len(sample_ids)


def export_clinical_metadata(
    dataset,
    sample_df: pd.DataFrame,
    output_file: Path,
    source: str | None = None,
) -> Path:
    clinical = call_getter(dataset, "get_clinical", source=source)
    if not isinstance(clinical, pd.DataFrame):
        clinical = pd.DataFrame(clinical)

    clin = clinical.copy()
    if isinstance(clin.index, pd.MultiIndex):
        clin["patient_id"] = clin.index.get_level_values(0).astype(str)
    else:
        clin["patient_id"] = clin.index.astype(str)

    clin = clin.reset_index(drop=True)
    clin = clin.drop_duplicates(subset=["patient_id"], keep="first")

    merged = sample_df.merge(clin, on="patient_id", how="left")
    merged = merged.rename(columns={"sample_id": "Sample.ID", "patient_id": "Patient.ID"})
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, sep="\t", index=False)
    return output_file


def build_modality_manifest(
    cohort_dir: Path,
    modalities: list[str],
    output_file: Path,
) -> Path:
    rows = []
    for m in modalities:
        ptm_path = cohort_dir / f"{m}.tsv"
        rows.append({"modality": m, "ptm_file": str(ptm_path), "enabled": True})
    manifest = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_file, sep="\t", index=False)
    return output_file


# ---------------------------------------------------------------------------
# Manifest helpers (from build_cptac_manifest.py)
# ---------------------------------------------------------------------------

def _empty_manifest_row(cohort: str) -> dict[str, object]:
    return {
        "cohort_id": cohort,
        "has_proteomics": False,
        "has_phosphoproteomics": False,
        "has_acetylproteomics": False,
        "proteomics_sources": "",
        "phosphoproteomics_sources": "",
        "acetylproteomics_sources": "",
        "n_tumor": 0,
        "n_normal": 0,
        "has_normal_samples": False,
        "enabled": True,
        "notes": "",
    }


def _add_source(row: dict[str, object], modality: str, source: str) -> None:
    row[f"has_{modality}"] = True
    src_col = f"{modality}_sources"
    prev = row[src_col]
    values = [v for v in str(prev).split(";") if v]
    if source not in values:
        values.append(source)
    row[src_col] = ";".join(values)


def _merge_existing(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Preserve manually curated 'enabled' and 'notes' from an existing file.

    Only overrides rows that exist in the old manifest; rows absent from
    the old file keep their newly computed values.
    """
    if not out.exists():
        return df
    existing = pd.read_csv(out, sep="\t")
    if "cohort_id" not in existing.columns:
        return df
    overrides = existing[["cohort_id"]].copy()
    for col in ("enabled", "notes"):
        if col in existing.columns:
            overrides[f"_old_{col}"] = existing[col]
    df = df.merge(overrides, on="cohort_id", how="left")
    for col in ("enabled", "notes"):
        old_col = f"_old_{col}"
        if old_col in df.columns:
            mask = df[old_col].notna()
            df.loc[mask, col] = df.loc[mask, old_col]
            df = df.drop(columns=[old_col])
    if "enabled" not in df.columns:
        df["enabled"] = True
    if "notes" not in df.columns:
        df["notes"] = ""
    df["enabled"] = df["enabled"].fillna(True)
    df["notes"] = df["notes"].fillna("")
    log.info("Preserved 'enabled' and 'notes' from existing %s", out)
    return df


# ---------------------------------------------------------------------------
# Per-cohort export logic
# ---------------------------------------------------------------------------

def export_cohort(
    cohort: str,
    output_root: Path,
    ptm_modalities: list[str],
) -> dict[str, object]:
    """Export a single cohort and return a manifest row dict."""
    import cptac  # type: ignore

    cohort = canonical_cohort_name(cohort)
    cls_name = COHORT_CLASS.get(cohort)
    if not cls_name:
        raise ValueError(f"Unsupported cohort id: {cohort}")
    if not hasattr(cptac, cls_name):
        raise RuntimeError(f"cptac class not found: {cls_name}")

    cohort_dir = output_root / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Loading dataset (%s) ...", cohort, cls_name)
    dataset = getattr(cptac, cls_name)()

    # --- Infer sources -------------------------------------------------
    chosen_sources = {
        "proteomics": infer_source(dataset, cohort, "proteomics"),
        "clinical": infer_source(dataset, cohort, "clinical"),
    }
    for mod in ptm_modalities:
        chosen_sources[mod] = infer_source(dataset, cohort, mod)

    # --- Proteomics (mandatory) ----------------------------------------
    protein_file = cohort_dir / "proteomics.tsv"
    protein_path, n_protein_features, n_samples = export_modality_table(
        dataset=dataset,
        modality="proteomics",
        source=chosen_sources["proteomics"],
        output_file=protein_file,
    )

    # Build sample table from exported proteomics for consistency checks.
    protein_wide = pd.read_csv(protein_path, sep="\t", nrows=0)
    protein_samples = [c for c in protein_wide.columns if c.endswith("-T") or c.endswith("-N")]
    sample_df = pd.DataFrame({"sample_id": protein_samples})
    sample_df["patient_id"] = sample_df["sample_id"].str[:-2]
    sample_df["condition"] = np.where(sample_df["sample_id"].str.endswith("-T"), "Tumor", "Normal")
    sample_df["is_tumor"] = np.where(sample_df["condition"] == "Tumor", 1, 0)

    n_tumor = int((sample_df["condition"] == "Tumor").sum())
    n_normal = int((sample_df["condition"] == "Normal").sum())
    has_normal = n_normal > 0

    log.info("[%s] proteomics: %d features, %d samples (T=%d, N=%d)",
             cohort, n_protein_features, n_samples, n_tumor, n_normal)

    # --- Clinical metadata ---------------------------------------------
    sample_meta_file = export_clinical_metadata(
        dataset,
        sample_df=sample_df,
        output_file=cohort_dir / "sample_metadata.tsv",
        source=chosen_sources.get("clinical"),
    )

    # --- PTM modalities ------------------------------------------------
    exported_modalities: list[str] = []
    modality_stats: list[dict] = []
    for modality in ptm_modalities:
        out_file = cohort_dir / f"{modality}.tsv"
        try:
            out_path, feat_n, sample_n = export_modality_table(
                dataset=dataset,
                modality=modality,
                source=chosen_sources.get(modality),
                output_file=out_file,
                expected_samples=sample_df,
            )
            exported_modalities.append(modality)
            modality_stats.append({
                "modality": modality,
                "status": "ok",
                "source": chosen_sources.get(modality),
                "output_file": str(out_path),
                "n_features": int(feat_n),
                "n_samples": int(sample_n),
            })
            log.info("[%s] %s: %d features, %d samples", cohort, modality, feat_n, sample_n)
            if sample_n != n_samples:
                diff = sample_n - n_samples
                log.warning("[%s] %s: sample count differs from proteomics (%+d; %d ptm vs %d proteomics)",
                            cohort, modality, diff, sample_n, n_samples)
        except Exception as exc:
            modality_stats.append({
                "modality": modality,
                "status": "failed",
                "reason": str(exc),
                "output_file": str(out_file),
            })
            log.warning("[%s] %s failed: %s", cohort, modality, exc)

    # --- ptm_manifest.tsv ----------------------------------------------
    manifest_file = build_modality_manifest(
        cohort_dir=cohort_dir,
        modalities=exported_modalities,
        output_file=cohort_dir / "ptm_manifest.tsv",
    )

    # --- export_summary.json -------------------------------------------
    summary = {
        "cohort": cohort,
        "dataset_class": cls_name,
        "selected_sources": chosen_sources,
        "protein_file": str(protein_file),
        "sample_metadata_file": str(sample_meta_file),
        "ptm_manifest_file": str(manifest_file),
        "n_protein_features": int(n_protein_features),
        "n_samples": int(n_samples),
        "n_tumor": n_tumor,
        "n_normal": n_normal,
        "has_normal_samples": has_normal,
        "modalities_requested": ptm_modalities,
        "modalities_exported": exported_modalities,
        "selected_proteomics_source": chosen_sources.get("proteomics"),
        "selected_phospho_source": chosen_sources.get("phosphoproteomics"),
        "selected_acetyl_source": chosen_sources.get("acetylproteomics"),
        "modality_stats": modality_stats,
    }
    summary_file = cohort_dir / "export_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("[%s] Export complete → %s", cohort, cohort_dir)

    # --- Manifest row --------------------------------------------------
    mrow = _empty_manifest_row(cohort)
    mrow["has_proteomics"] = True
    mrow["proteomics_sources"] = chosen_sources.get("proteomics") or ""
    mrow["n_tumor"] = n_tumor
    mrow["n_normal"] = n_normal
    mrow["has_normal_samples"] = has_normal

    if not has_normal:
        mrow["enabled"] = False
        mrow["notes"] = "no normal samples"

    for mod in ptm_modalities:
        src = chosen_sources.get(mod) or ""
        if mod in exported_modalities:
            _add_source(mrow, mod, src) if src else None
            mrow[f"has_{mod}"] = True
        # else: already False from _empty_manifest_row

    return mrow


# ---------------------------------------------------------------------------
# Cross-cohort summary (from build_cptac_cross_cohort_summary.py)
# ---------------------------------------------------------------------------

def _extract_cohort_from_path(summary_path: Path) -> str:
    match = COHORT_DIR_RE.fullmatch(summary_path.parent.name)
    if not match:
        raise ValueError(f"Cannot parse cohort from path: {summary_path}")
    return match.group("cohort").lower()


def load_modality_summaries(summary_paths: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in summary_paths:
        frame = pd.read_csv(path, sep="\t")
        frame.insert(0, "cohort", _extract_cohort_from_path(path))
        rows.append(frame)

    if not rows:
        return pd.DataFrame()

    table = pd.concat(rows, axis=0, ignore_index=True)
    numeric_cols = [
        "sample_overlap", "tumor_samples", "normal_samples", "paired_samples",
        "total_sites", "protein_match_exact", "protein_match_canonical",
        "protein_match_gene", "protein_unmatched", "paired_testable_raw",
        "paired_testable_raw_strict", "dual_group_observed_sites",
        "dual_group_effective_sites", "effective_min_tumor", "effective_min_normal",
        "detection_tested_sites", "true_increase_detection", "raw_up_sites",
        "true_increase_subtract", "true_increase_lm",
        "protein_driven_subtract", "protein_driven_lm",
    ]
    for col in numeric_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")

    for col in ["status", "analysis_mode", "reason"]:
        if col not in table.columns:
            table[col] = pd.NA

    if "raw_up_sites" in table.columns:
        raw_up = table["raw_up_sites"]
        table["retention_subtract_pct"] = (table["true_increase_subtract"] * 100.0 / raw_up).where(raw_up > 0)
        table["retention_lm_pct"] = (table["true_increase_lm"] * 100.0 / raw_up).where(raw_up > 0)
    else:
        table["retention_subtract_pct"] = pd.NA
        table["retention_lm_pct"] = pd.NA
    return table


def load_export_summaries(exports_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not exports_root.exists():
        return pd.DataFrame(rows)

    for cohort_dir in sorted(exports_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        summary_path = cohort_dir / "export_summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text())
        selected_sources = data.get("selected_sources", {})
        if not isinstance(selected_sources, dict):
            selected_sources = {}

        row: dict[str, object] = {"cohort": cohort_dir.name.lower()}
        for key in [
            "n_samples", "n_protein_features",
            "modalities_requested", "modalities_exported",
            "selected_proteomics_source", "selected_phospho_source",
            "selected_acetyl_source",
        ]:
            if key == "selected_proteomics_source":
                value = data.get(key, selected_sources.get("proteomics"))
            elif key == "selected_phospho_source":
                value = data.get(key, selected_sources.get("phosphoproteomics"))
            elif key == "selected_acetyl_source":
                value = data.get(key, selected_sources.get("acetylproteomics"))
            else:
                value = data.get(key)
            if isinstance(value, list):
                row[key] = ";".join(str(x) for x in value)
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_cross_cohort_summary(
    results_root: Path,
    exports_root: Path,
    output_dir: Path,
) -> None:
    """Build cross-cohort summary tables from ptmanchor pipeline outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_glob = str(results_root / "cptac_*_ptm_correction" / "modality_summary.tsv")
    summary_paths = sorted(Path().glob(summary_glob))
    modality_table = load_modality_summaries(summary_paths)

    if modality_table.empty:
        modality_out = output_dir / "cross_cohort_modality_summary.tsv"
        modality_out.write_text("cohort\tmodality\tstatus\treason\n")
        log.warning("No modality summaries found with glob: %s", summary_glob)
        log.info("Wrote: %s", modality_out)
        return

    preferred_cols = [
        "cohort", "modality", "status", "analysis_mode",
        "sample_overlap", "tumor_samples", "normal_samples", "paired_samples",
        "total_sites", "protein_match_exact", "protein_match_canonical",
        "protein_match_gene", "protein_unmatched",
        "paired_testable_raw", "paired_testable_raw_strict",
        "dual_group_observed_sites", "dual_group_effective_sites",
        "fallback_used", "fallback_reason",
        "effective_min_tumor", "effective_min_normal",
        "detection_fallback_used", "detection_fallback_reason",
        "detection_tested_sites", "true_increase_detection",
        "raw_up_sites", "true_increase_subtract", "true_increase_lm",
        "protein_driven_subtract", "protein_driven_lm",
        "retention_subtract_pct", "retention_lm_pct", "reason",
    ]
    for col in preferred_cols:
        if col not in modality_table.columns:
            modality_table[col] = pd.NA
    modality_out = output_dir / "cross_cohort_modality_summary.tsv"
    modality_table = modality_table[preferred_cols].sort_values(["cohort", "modality"])
    modality_table.to_csv(modality_out, sep="\t", index=False)
    log.info("Wrote: %s", modality_out)

    ok_rows = modality_table[modality_table["status"] == "ok"].copy()
    pivot_cols = ["raw_up_sites", "retention_lm_pct", "true_increase_lm"]
    if ok_rows.empty:
        pivot = pd.DataFrame(columns=["cohort"])
    else:
        pivot = ok_rows.pivot_table(
            index="cohort", columns="modality", values=pivot_cols, aggfunc="first",
        )
        pivot.columns = [f"{metric}__{modality}" for metric, modality in pivot.columns]
        pivot = pivot.reset_index()
    pivot_out = output_dir / "cross_cohort_pivot.tsv"
    pivot.to_csv(pivot_out, sep="\t", index=False)
    log.info("Wrote: %s", pivot_out)

    export_table = load_export_summaries(exports_root)
    export_out = output_dir / "cross_cohort_export_summary.tsv"
    if export_table.empty:
        export_out.write_text("cohort\n")
    else:
        export_table = export_table.sort_values("cohort")
        export_table.to_csv(export_out, sep="\t", index=False)
    log.info("Wrote: %s", export_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add ptmanchor pipeline arguments (mirrors ptmanchor.cli)."""
    g = parser.add_argument_group("ptmanchor pipeline")
    g.add_argument("--min-pairs", type=int, default=8)
    g.add_argument("--min-tumor", type=int, default=20)
    g.add_argument("--min-normal", type=int, default=8)
    g.add_argument("--fdr-cutoff", type=float, default=0.05)
    g.add_argument("--min-corrected-delta", type=float, default=0.2)
    g.add_argument("--top-n", type=int, default=50)
    g.add_argument("--covariates", default="")
    g.add_argument("--sample-id-col", default="Sample.ID")
    g.add_argument("--patient-id-col", default=None)
    g.add_argument("--sample-meta-sheet", default=None)
    g.add_argument("--enable-sample-lm", action="store_true")
    g.add_argument("--enable-sample-lmm", action="store_true")
    g.add_argument("--max-sites-sample-lm", type=int, default=0)
    g.add_argument("--max-sites-sample-lmm", type=int, default=0)
    g.add_argument("--lmm-maxiter", type=int, default=100)
    g.add_argument("--enable-paired-to-unpaired-fallback", action="store_true")
    g.add_argument("--min-paired-testable-sites", type=int, default=1)
    g.add_argument("--fallback-min-tumor", type=int, default=0)
    g.add_argument("--fallback-min-normal", type=int, default=0)
    g.add_argument("--force-unpaired-if-paired", action="store_true")
    g.add_argument("--enable-detection-fallback", action="store_true")
    g.add_argument("--force-detection-fallback", action="store_true")
    g.add_argument("--min-detection-delta", type=float, default=0.10)
    g.add_argument("--min-dual-group-sites-detection", type=int, default=100)


def _build_pipeline_args(
    cohort_dir: Path,
    results_dir: Path,
    cli_args: argparse.Namespace,
) -> SimpleNamespace:
    """Build a namespace that pipeline.run_manifest() expects."""
    return SimpleNamespace(
        manifest=str(cohort_dir / "ptm_manifest.tsv"),
        protein_file=str(cohort_dir / "proteomics.tsv"),
        output_dir=str(results_dir),
        sample_meta_file=str(cohort_dir / "sample_metadata.tsv"),
        sample_meta_sheet=cli_args.sample_meta_sheet,
        sample_id_col=cli_args.sample_id_col,
        patient_id_col=cli_args.patient_id_col,
        covariates=cli_args.covariates,
        min_pairs=cli_args.min_pairs,
        min_tumor=cli_args.min_tumor,
        min_normal=cli_args.min_normal,
        fdr_cutoff=cli_args.fdr_cutoff,
        min_corrected_delta=cli_args.min_corrected_delta,
        top_n=cli_args.top_n,
        enable_sample_lm=cli_args.enable_sample_lm,
        enable_sample_lmm=cli_args.enable_sample_lmm,
        max_sites_sample_lm=cli_args.max_sites_sample_lm,
        max_sites_sample_lmm=cli_args.max_sites_sample_lmm,
        lmm_maxiter=cli_args.lmm_maxiter,
        enable_paired_to_unpaired_fallback=cli_args.enable_paired_to_unpaired_fallback,
        min_paired_testable_sites=cli_args.min_paired_testable_sites,
        fallback_min_tumor=cli_args.fallback_min_tumor,
        fallback_min_normal=cli_args.fallback_min_normal,
        force_unpaired_if_paired=cli_args.force_unpaired_if_paired,
        enable_detection_fallback=cli_args.enable_detection_fallback,
        force_detection_fallback=cli_args.force_detection_fallback,
        min_detection_delta=cli_args.min_detection_delta,
        min_dual_group_sites_detection=cli_args.min_dual_group_sites_detection,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all CPTAC cohorts, generate manifest, and run ptmanchor pipeline.",
    )
    parser.add_argument(
        "--output-root",
        default="data/cptac_exports",
        help="Root directory for cohort exports (default: data/cptac_exports).",
    )
    parser.add_argument(
        "--manifest",
        default="data/cptac_modalities_manifest.tsv",
        help="Output manifest TSV (default: data/cptac_modalities_manifest.tsv).",
    )
    parser.add_argument(
        "--modalities",
        default="phosphoproteomics,acetylproteomics",
        help="Comma-separated PTM modalities to export (default: phosphoproteomics,acetylproteomics).",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory for pipeline results (default: results).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export data, skip ptmanchor pipeline.",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Skip export, run pipeline on existing exports.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip export & pipeline, build cross-cohort summary only.",
    )
    parser.add_argument(
        "--cross-cohort-dir",
        default="results/cptac_cross_cohort",
        help="Output directory for cross-cohort summary (default: results/cptac_cross_cohort).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    _add_pipeline_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)

    ptm_modalities = [
        m.strip()
        for m in args.modalities.split(",")
        if m.strip() in MODALITY_GETTER and m.strip() != "proteomics"
    ]
    if not ptm_modalities:
        log.error("No valid PTM modality selected. Use phosphoproteomics and/or acetylproteomics.")
        sys.exit(1)

    cross_cohort_dir = Path(args.cross_cohort_dir)

    # Handle --summary-only: skip to Step 3
    if args.summary_only:
        log.info("=== Step 3: Cross-cohort summary (summary-only) ===")
        build_cross_cohort_summary(results_root, output_root, cross_cohort_dir)
        log.info("Done (summary-only).")
        return

    # ===================================================================
    # Step 1: Export
    # ===================================================================
    manifest_rows: list[dict[str, object]] = []
    succeeded: list[str] = []
    failed: list[str] = []

    if not args.pipeline_only:
        try:
            import cptac  # noqa: F401  type: ignore
        except Exception as exc:
            log.error("Failed to import cptac: %s", exc)
            sys.exit(1)

        log.info("=== Step 1: Export ===")
        log.info("Cohorts: %s", ", ".join(KNOWN_COHORTS))
        log.info("PTM modalities: %s", ", ".join(ptm_modalities))

        for cohort in KNOWN_COHORTS:
            try:
                mrow = export_cohort(cohort, output_root, ptm_modalities)
                manifest_rows.append(mrow)
                succeeded.append(cohort)
            except Exception as exc:
                log.error("[%s] FAILED: %s", cohort, exc, exc_info=True)
                failed.append(cohort)
                mrow = _empty_manifest_row(cohort)
                mrow["enabled"] = False
                mrow["notes"] = f"export failed: {exc}"
                manifest_rows.append(mrow)

        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df = manifest_df.sort_values("cohort_id").reset_index(drop=True)
        manifest_df = _merge_existing(manifest_df, manifest_path)
        manifest_df.to_csv(manifest_path, sep="\t", index=False)

        log.info("Manifest written: %s", manifest_path)
        log.info("Export succeeded: %s", ", ".join(succeeded) if succeeded else "(none)")
        if failed:
            log.warning("Export failed: %s", ", ".join(failed))

    if args.export_only:
        log.info("Done (export-only).")
        return

    # ===================================================================
    # Step 2: Run ptmanchor pipeline
    # ===================================================================
    REPO_ROOT = Path(__file__).resolve().parents[1]
    ptmanchor_pkg = REPO_ROOT / "ptmanchor"
    for p in (str(ptmanchor_pkg), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from ptmanchor.pipeline import run_manifest

    log.info("=== Step 2: ptmanchor pipeline ===")

    # Determine which cohorts to run: use manifest if available, else succeeded list.
    if manifest_path.exists():
        mdf = pd.read_csv(manifest_path, sep="\t")
        pipeline_cohorts = mdf.loc[mdf["enabled"].astype(str).str.lower().isin({"true", "1"}), "cohort_id"].tolist()
    else:
        pipeline_cohorts = succeeded

    pipeline_ok: list[str] = []
    pipeline_fail: list[str] = []

    for cohort in pipeline_cohorts:
        cohort_dir = output_root / cohort
        ptm_manifest = cohort_dir / "ptm_manifest.tsv"
        protein_file = cohort_dir / "proteomics.tsv"

        if not ptm_manifest.exists() or not protein_file.exists():
            log.warning("[%s] Missing export files, skipping pipeline.", cohort)
            pipeline_fail.append(cohort)
            continue

        results_dir = results_root / f"cptac_{cohort}_ptm_correction"
        pipeline_ns = _build_pipeline_args(cohort_dir, results_dir, args)

        log.info("[%s] Running ptmanchor pipeline → %s", cohort, results_dir)
        try:
            summary_tsv, summary_txt, run_config = run_manifest(pipeline_ns)
            log.info("[%s] Pipeline done → %s", cohort, summary_tsv)
            pipeline_ok.append(cohort)
        except Exception as exc:
            log.error("[%s] Pipeline FAILED: %s", cohort, exc, exc_info=True)
            pipeline_fail.append(cohort)

    log.info("Pipeline succeeded: %s", ", ".join(pipeline_ok) if pipeline_ok else "(none)")
    if pipeline_fail:
        log.warning("Pipeline failed: %s", ", ".join(pipeline_fail))

    # ===================================================================
    # Step 3: Cross-cohort summary
    # ===================================================================
    log.info("=== Step 3: Cross-cohort summary ===")
    build_cross_cohort_summary(results_root, output_root, cross_cohort_dir)

    log.info("Done. Export %d/%d, Pipeline %d/%d.",
             len(succeeded) if not args.pipeline_only else len(pipeline_cohorts),
             len(KNOWN_COHORTS),
             len(pipeline_ok), len(pipeline_cohorts))


if __name__ == "__main__":
    main()
