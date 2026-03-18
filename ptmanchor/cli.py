from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptmanchor",
        description="Run protein-adjusted PTM correction for multiple modalities from a manifest.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="TSV with columns: modality, ptm_file, enabled(optional).",
    )
    parser.add_argument(
        "--protein-file",
        required=True,
        help="Global proteome TSV file used as protein anchor.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--no-eb",
        action="store_true",
        help="Disable empirical Bayes variance shrinkage.",
    )
    parser.add_argument(
        "--no-lambda-shrinkage",
        action="store_true",
        help="Disable James-Stein lambda shrinkage.",
    )
    parser.add_argument("--min-pairs", type=int, default=8, help="Minimum paired observations per site.")
    parser.add_argument("--min-tumor", type=int, default=20, help="Minimum tumor observations for sample LM/LMM.")
    parser.add_argument("--min-normal", type=int, default=8, help="Minimum normal observations for sample LM/LMM.")
    parser.add_argument("--fdr-cutoff", type=float, default=0.05, help="FDR cutoff.")
    parser.add_argument(
        "--min-corrected-delta",
        type=float,
        default=0.2,
        help="Minimum adjusted effect size for true-increase call.",
    )
    parser.add_argument("--top-n", type=int, default=50, help="Top N LM hits to save.")

    parser.add_argument(
        "--sample-meta-file",
        default=None,
        help="Sample metadata file (csv/tsv/xlsx). Optional.",
    )
    parser.add_argument(
        "--sample-meta-sheet",
        default=None,
        help="Excel sheet name when sample-meta-file is xlsx.",
    )
    parser.add_argument("--sample-id-col", default="Sample.ID", help="Sample ID column in metadata.")
    parser.add_argument(
        "--patient-id-col",
        default=None,
        help="Optional patient ID column in metadata. If omitted, inferred from sample suffix.",
    )
    parser.add_argument(
        "--covariates",
        default="",
        help="Comma-separated covariate columns from metadata (example: Subtype,DX).",
    )

    parser.add_argument(
        "--enable-sample-lm",
        action="store_true",
        help="Run sample-level OLS model: y ~ is_tumor + protein + covariates.",
    )
    parser.add_argument(
        "--enable-sample-lmm",
        action="store_true",
        help="Run sample-level mixed model with patient random intercept.",
    )
    parser.add_argument(
        "--max-sites-sample-lm",
        type=int,
        default=0,
        help="If >0, run sample LM only for first N sites.",
    )
    parser.add_argument(
        "--max-sites-sample-lmm",
        type=int,
        default=0,
        help="If >0, run sample LMM for up to N selected sites.",
    )
    parser.add_argument("--lmm-maxiter", type=int, default=100, help="Max iterations for each LMM fit.")
    parser.add_argument(
        "--enable-paired-to-unpaired-fallback",
        action="store_true",
        help=(
            "When paired mode has too few testable sites, fallback to unpaired tests "
            "on the same modality."
        ),
    )
    parser.add_argument(
        "--min-paired-testable-sites",
        type=int,
        default=1,
        help=(
            "Minimum number of paired-testable sites required to keep paired mode. "
            "Used only when --enable-paired-to-unpaired-fallback is set."
        ),
    )
    parser.add_argument(
        "--fallback-min-tumor",
        type=int,
        default=0,
        help="Unpaired fallback tumor minimum. If 0, reuse --min-tumor.",
    )
    parser.add_argument(
        "--fallback-min-normal",
        type=int,
        default=0,
        help="Unpaired fallback normal minimum. If 0, reuse --min-normal.",
    )
    parser.add_argument(
        "--force-unpaired-if-paired",
        action="store_true",
        help="Force unpaired tests even when paired samples exist.",
    )
    parser.add_argument(
        "--enable-detection-fallback",
        action="store_true",
        help=(
            "If continuous-value tests are not testable, run detection-rate fallback "
            "using one-sided Fisher tests (tumor > normal)."
        ),
    )
    parser.add_argument(
        "--force-detection-fallback",
        action="store_true",
        help="Run detection fallback even when overlap diagnostics suggest sparse/disjoint data.",
    )
    parser.add_argument(
        "--min-detection-delta",
        type=float,
        default=0.10,
        help="Minimum protein-adjusted detection-rate delta for detection fallback hits.",
    )
    parser.add_argument(
        "--min-dual-group-sites-detection",
        type=int,
        default=100,
        help=(
            "Minimum number of sites observed in both tumor and normal groups (>=1 each) "
            "required to allow detection fallback, unless forced."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        parser.error(f"Manifest file not found: {manifest}")
    protein = Path(args.protein_file)
    if not protein.exists():
        parser.error(f"Protein file not found: {protein}")
    if args.sample_meta_file and not Path(args.sample_meta_file).exists():
        parser.error(f"Sample metadata file not found: {args.sample_meta_file}")

    try:
        summary_tsv, summary_txt, run_config = run_manifest(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Wrote: {summary_tsv}")
    print(f"[OK] Wrote: {summary_txt}")
    print(f"[OK] Wrote: {run_config}")


if __name__ == "__main__":
    main()
