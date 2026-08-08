"""ptmanchor — protein-anchored correction for multi-PTM proteomics cohorts."""

__version__ = "1.1.2"

from .pipeline import run_manifest, run_modality
from .modeling import (
    paired_lm_intercept_test,
    sample_lm_condition_test,
    sample_lmm_condition_test,
)
from .utils import (
    extract_accession,
    canonical_accession,
    bh_qvalues,
    classify_hit,
    one_sided_ttest_1samp,
    one_sided_ttest_ind,
    t_pvalue_from_stat,
    get_paired_indices,
    sample_columns,
)
from .metadata import (
    build_sample_design,
    encode_covariates,
    derive_patient_id,
)

__all__ = [
    "__version__",
    # pipeline
    "run_manifest",
    "run_modality",
    # modeling
    "paired_lm_intercept_test",
    "sample_lm_condition_test",
    "sample_lmm_condition_test",
    # utils
    "extract_accession",
    "canonical_accession",
    "bh_qvalues",
    "classify_hit",
    "one_sided_ttest_1samp",
    "one_sided_ttest_ind",
    "t_pvalue_from_stat",
    "get_paired_indices",
    "sample_columns",
    # metadata
    "build_sample_design",
    "encode_covariates",
    "derive_patient_id",
]
