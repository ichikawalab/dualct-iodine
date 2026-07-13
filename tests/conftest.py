# -*- coding: utf-8 -*-
import pytest

from tests._dicom_fixtures import (
    make_synthetic_dataset,
    make_synthetic_kvp_dataset,
    make_synthetic_nested_kvp_dataset,
)


@pytest.fixture
def synthetic_root(tmp_path):
    """A tiny synthetic dataset (4 patients, 8 slices, 16x16) mirroring the real layout."""
    patient_ids = ["01", "02", "03", "04"]
    make_synthetic_dataset(tmp_path, patient_ids, n_slices=8, rows=16, cols=16)
    return tmp_path, patient_ids


@pytest.fixture
def synthetic_kvp_root(tmp_path):
    """A tiny synthetic kVp-task dataset (flat <root>/<kv>/<patient>/, no mask)."""
    patient_ids = ["01", "02", "03", "04"]
    make_synthetic_kvp_dataset(tmp_path, patient_ids, n_slices=8, rows=16, cols=16)
    return tmp_path, patient_ids


@pytest.fixture
def synthetic_nested_kvp_root(tmp_path):
    """A tiny nested DE-group kVp dataset: <root>/DE{1,2,3}/<kv>/<patient 1,2>/."""
    groups = ["DE1", "DE2", "DE3"]
    make_synthetic_nested_kvp_dataset(tmp_path, groups, patients_per_group=2, n_slices=8, rows=16, cols=16)
    return tmp_path, groups
