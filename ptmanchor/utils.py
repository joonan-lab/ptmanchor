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
    # Missing values arrive as None or as float NaN depending on the pandas version.
    if not isinstance(accession, str):
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


_ALT_CHOICES = ("greater", "less", "two-sided")


def _validate_alternative(alternative: str) -> str:
    if alternative not in _ALT_CHOICES:
        raise ValueError(
            f"alternative must be one of {_ALT_CHOICES!r}, got {alternative!r}"
        )
    return alternative


def one_sided_ttest_1samp(
    matrix: np.ndarray,
    min_n: int,
    alternative: str = "greater",
) -> tuple[np.ndarray, np.ndarray]:
    _validate_alternative(alternative)
    n = np.sum(np.isfinite(matrix), axis=1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _stat, pvals = stats.ttest_1samp(
            matrix,
            popmean=0.0,
            axis=1,
            nan_policy="omit",
            alternative=alternative,
        )
    pvals = np.asarray(pvals, dtype=float)
    pvals[n < min_n] = np.nan
    return pvals, n


def one_sided_ttest_ind(
    tumor_matrix: np.ndarray,
    normal_matrix: np.ndarray,
    min_tumor: int,
    min_normal: int,
    alternative: str = "greater",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_alternative(alternative)
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
            alternative=alternative,
        )
    pvals = np.asarray(pvals, dtype=float)
    pvals[(n_t < min_tumor) | (n_n < min_normal)] = np.nan
    return pvals, n_t, n_n


def t_pvalue_from_stat(
    t_stat: float,
    df: float,
    alternative: str = "greater",
) -> float:
    """Convert a t-statistic + df into a p-value under the chosen alternative."""
    _validate_alternative(alternative)
    if np.isinf(df):
        dist = stats.norm
    else:
        dist = stats.t(df=df)
    if alternative == "greater":
        return float(dist.sf(t_stat))
    if alternative == "less":
        return float(dist.cdf(t_stat))
    # two-sided
    return float(2.0 * dist.sf(abs(t_stat)))


def classify_hit(
    effect: np.ndarray,
    q: np.ndarray,
    alpha: float,
    threshold: float,
    alternative: str = "greater",
) -> dict[str, np.ndarray]:
    """Classify sites into up / down / hit (union) masks by direction.

    greater   -> only up populated (effect >= threshold & q <= alpha)
    less      -> only down populated (effect <= -threshold & q <= alpha)
    two-sided -> both populated
    """
    _validate_alternative(alternative)
    effect = np.asarray(effect, dtype=float)
    q = np.asarray(q, dtype=float)
    finite = np.isfinite(q) & np.isfinite(effect)
    sig = finite & (q <= alpha)

    up = np.zeros(effect.shape, dtype=bool)
    down = np.zeros(effect.shape, dtype=bool)
    if alternative in ("greater", "two-sided"):
        up = sig & (effect >= threshold)
    if alternative in ("less", "two-sided"):
        down = sig & (effect <= -threshold)
    hit = up | down
    return {"up": up, "down": down, "hit": hit}


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
