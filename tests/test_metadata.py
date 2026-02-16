"""Tests for ptmanchor.metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptmanchor.metadata import (
    derive_patient_id,
    build_sample_design,
    encode_covariates,
)


class TestDerivePatientId:
    def test_tumor_suffix(self):
        assert derive_patient_id("P001-T") == "P001"

    def test_normal_suffix(self):
        assert derive_patient_id("P001-N") == "P001"

    def test_no_suffix(self):
        assert derive_patient_id("SAMPLE_X") == "SAMPLE_X"

    def test_numeric_input(self):
        assert derive_patient_id(123) == "123"


class TestBuildSampleDesign:
    def test_basic_design(self):
        samples = ["P001-T", "P001-N", "P002-T", "P002-N"]
        design = build_sample_design(samples)

        assert list(design.columns) == ["sample_id", "is_tumor", "patient_id"]
        assert list(design["is_tumor"]) == [1, 0, 1, 0]
        assert list(design["patient_id"]) == ["P001", "P001", "P002", "P002"]

    def test_covariates_without_metadata(self):
        """Covariates without metadata should be filled with NaN."""
        samples = ["P001-T", "P001-N"]
        design = build_sample_design(samples, covariates=["Age", "Sex"])
        assert "Age" in design.columns
        assert "Sex" in design.columns

    def test_with_metadata_file(self, tmp_path):
        """Should merge metadata correctly."""
        samples = ["P001-T", "P001-N", "P002-T", "P002-N"]

        meta = pd.DataFrame({
            "Sample.ID": ["P001-T", "P001-N", "P002-T", "P002-N"],
            "Age": [55, 55, 60, 60],
            "Subtype": ["LUAD", "LUAD", "LUSC", "LUSC"],
        })
        meta_path = tmp_path / "meta.tsv"
        meta.to_csv(meta_path, sep="\t", index=False)

        design = build_sample_design(
            samples,
            sample_meta_file=meta_path,
            covariates=["Age", "Subtype"],
        )

        assert list(design["Age"]) == [55, 55, 60, 60]
        assert list(design["Subtype"]) == ["LUAD", "LUAD", "LUSC", "LUSC"]

    def test_patient_id_from_metadata(self, tmp_path):
        """Should use patient_id_col from metadata when provided."""
        samples = ["S1-T", "S1-N"]
        meta = pd.DataFrame({
            "Sample.ID": ["S1-T", "S1-N"],
            "PatientID": ["PAT_A", "PAT_A"],
        })
        meta_path = tmp_path / "meta.tsv"
        meta.to_csv(meta_path, sep="\t", index=False)

        design = build_sample_design(
            samples,
            sample_meta_file=meta_path,
            patient_id_col="PatientID",
        )

        assert list(design["patient_id"]) == ["PAT_A", "PAT_A"]


class TestEncodeCovariates:
    def test_numeric_covariate(self):
        design = pd.DataFrame({
            "sample_id": ["A", "B", "C"],
            "Age": [50, 60, 70],
        })
        encoded = encode_covariates(design, ["Age"])
        assert "cov_Age" in encoded.columns
        # Should be centered around median (60)
        assert encoded["cov_Age"].iloc[0] == pytest.approx(50.0)
        assert encoded["cov_Age"].iloc[1] == pytest.approx(60.0)

    def test_categorical_covariate(self):
        design = pd.DataFrame({
            "sample_id": ["A", "B", "C"],
            "Subtype": ["LUAD", "LUSC", "LUAD"],
        })
        encoded = encode_covariates(design, ["Subtype"])
        # One-hot with drop_first: should have 1 column
        assert encoded.shape[1] == 1

    def test_empty_covariates(self):
        design = pd.DataFrame({"sample_id": ["A", "B"]})
        encoded = encode_covariates(design, [])
        assert encoded.shape == (2, 0)

    def test_missing_covariate_column(self):
        design = pd.DataFrame({"sample_id": ["A", "B"]})
        encoded = encode_covariates(design, ["NonExistent"])
        assert encoded.shape == (2, 0)
