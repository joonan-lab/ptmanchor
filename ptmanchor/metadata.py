from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def derive_patient_id(sample_id: str) -> str:
    sample_id = str(sample_id)
    if sample_id.endswith("-T") or sample_id.endswith("-N"):
        return sample_id[:-2]
    return sample_id


def read_table(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


def _fill_covariate_within_patient(df: pd.DataFrame, cov: str) -> pd.Series:
    def _fill(series: pd.Series) -> pd.Series:
        return series.ffill().bfill()

    return df.groupby("patient_id", sort=False)[cov].transform(_fill)


def build_sample_design(
    samples: list[str],
    sample_meta_file: Path | None = None,
    sample_meta_sheet: str | None = None,
    sample_id_col: str = "Sample.ID",
    patient_id_col: str | None = None,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    covariates = covariates or []
    design = pd.DataFrame({"sample_id": [str(s) for s in samples]})
    design["is_tumor"] = design["sample_id"].str.endswith("-T").astype(int)
    design["patient_id"] = design["sample_id"].map(derive_patient_id)

    if sample_meta_file is not None:
        meta = read_table(sample_meta_file, sheet_name=sample_meta_sheet)
        if sample_id_col not in meta.columns:
            raise ValueError(f"sample_id_col not found in metadata: {sample_id_col}")

        meta = meta.copy()
        meta[sample_id_col] = meta[sample_id_col].astype(str)
        keep_cols = [sample_id_col]
        for c in covariates:
            if c in meta.columns:
                keep_cols.append(c)
        if patient_id_col and patient_id_col in meta.columns:
            keep_cols.append(patient_id_col)
        meta = meta.loc[:, keep_cols].drop_duplicates(subset=[sample_id_col], keep="first")

        joined = design.merge(meta, left_on="sample_id", right_on=sample_id_col, how="left")
        if patient_id_col and patient_id_col in joined.columns:
            has_meta_patient = joined[patient_id_col].notna()
            joined.loc[has_meta_patient, "patient_id"] = joined.loc[has_meta_patient, patient_id_col].astype(str)
        design = joined.drop(columns=[sample_id_col], errors="ignore")
    else:
        for cov in covariates:
            design[cov] = np.nan

    for cov in covariates:
        if cov not in design.columns:
            design[cov] = np.nan
        design[cov] = _fill_covariate_within_patient(design, cov)

        # If covariate still missing, use neutral fallback.
        if pd.api.types.is_numeric_dtype(design[cov]):
            med = pd.to_numeric(design[cov], errors="coerce").median()
            if pd.isna(med):
                med = 0.0
            design[cov] = pd.to_numeric(design[cov], errors="coerce").fillna(float(med))
        else:
            design[cov] = design[cov].astype("string").fillna("Unknown").replace("<NA>", "Unknown")

    return design


def encode_covariates(design: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    def _safe_name(name: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name))
        if not safe:
            safe = "x"
        if safe[0].isdigit():
            safe = f"x_{safe}"
        return safe

    if not covariates:
        return pd.DataFrame(index=design.index)

    encoded_parts: list[pd.DataFrame] = []
    for cov in covariates:
        if cov not in design.columns:
            continue
        series = design[cov]
        if pd.api.types.is_numeric_dtype(series):
            vec = pd.to_numeric(series, errors="coerce")
            med = vec.median()
            if pd.isna(med):
                med = 0.0
            encoded_parts.append(pd.DataFrame({f"cov_{_safe_name(cov)}": vec.fillna(float(med)).astype(float)}))
            continue

        cat = series.astype("string").fillna("Unknown").replace("<NA>", "Unknown")
        dummies = pd.get_dummies(cat, prefix=f"cov_{_safe_name(cov)}", drop_first=True, dtype=float)
        dummies.columns = [_safe_name(c) for c in dummies.columns]
        if dummies.shape[1] == 0:
            # Constant category; no informative columns to add.
            continue
        encoded_parts.append(dummies)

    if not encoded_parts:
        return pd.DataFrame(index=design.index)
    out = pd.concat(encoded_parts, axis=1)
    if out.columns.duplicated().any():
        counts: dict[str, int] = {}
        new_cols = []
        for col in out.columns:
            counts[col] = counts.get(col, 0) + 1
            new_cols.append(col if counts[col] == 1 else f"{col}_{counts[col]}")
        out.columns = new_cols
    out.index = design.index
    return out
