from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

ACC_RE = re.compile(
    r"(A0A[A-Z0-9]{3}[A-Z0-9]{4}(?:-[0-9]+)?"
    r"|[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?"
    r"|[OPQ][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?"
    r"|ENSP[0-9]+(?:\.[0-9]+)?)"
)


def extract_accession(value: object) -> str | None:
    if value is None:
        return None
    matches = ACC_RE.findall(str(value))
    return matches[0] if matches else None


def canonical_accession(accession: str | None) -> str | None:
    if accession is None:
        return None
    if accession.startswith("ENSP") and "." in accession:
        return accession.split(".")[0]
    return accession.split("-")[0]


def parse_bool(value: object, default: bool = True) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def sample_columns(columns: list[str]) -> list[str]:
    primary = [c for c in columns if c.startswith("RE-")]
    if primary:
        return primary

    # Fallback for external cohorts (e.g., CPTAC exports) using suffix labels.
    fallback = [
        c
        for c in columns
        if str(c).endswith(("-T", "-N", ".T", ".N", "_T", "_N"))
    ]
    return fallback


def safe_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)


def nanmean_axis1(matrix: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(matrix, axis=1)


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(pvals.shape, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if valid.any():
        _, q = fdrcorrection(pvals[valid], alpha=0.05, method="indep")
        qvals[valid] = q
    return qvals


def one_sided_ttest_1samp(matrix: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.sum(np.isfinite(matrix), axis=1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _stat, pvals = stats.ttest_1samp(
            matrix,
            popmean=0.0,
            axis=1,
            nan_policy="omit",
            alternative="greater",
        )
    pvals = np.asarray(pvals, dtype=float)
    pvals[n < min_n] = np.nan
    return pvals, n


def one_sided_ttest_ind(
    tumor_matrix: np.ndarray,
    normal_matrix: np.ndarray,
    min_tumor: int,
    min_normal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_t = np.sum(np.isfinite(tumor_matrix), axis=1).astype(int)
    n_n = np.sum(np.isfinite(normal_matrix), axis=1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _stat, pvals = stats.ttest_ind(
            tumor_matrix,
            normal_matrix,
            axis=1,
            nan_policy="omit",
            equal_var=False,
            alternative="greater",
        )
    pvals = np.asarray(pvals, dtype=float)
    pvals[(n_t < min_tumor) | (n_n < min_normal)] = np.nan
    return pvals, n_t, n_n


def get_paired_indices(samples: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    tumor_cols = [c for c in samples if c.endswith("-T")]
    normal_cols = [c for c in samples if c.endswith("-N")]
    tumor_bases = {c.rsplit("-", 1)[0] for c in tumor_cols}
    normal_bases = {c.rsplit("-", 1)[0] for c in normal_cols}
    pair_bases = sorted(tumor_bases & normal_bases)
    if not pair_bases:
        raise ValueError("No paired tumor-normal sample bases found.")
    pair_t_cols = [f"{b}-T" for b in pair_bases]
    pair_n_cols = [f"{b}-N" for b in pair_bases]
    pair_t_idx = np.array([samples.index(c) for c in pair_t_cols], dtype=int)
    pair_n_idx = np.array([samples.index(c) for c in pair_n_cols], dtype=int)
    return pair_t_idx, pair_n_idx, pair_bases
