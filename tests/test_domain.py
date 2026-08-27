import numpy as np
import pytest

from gmes_plot.domain.models import Dataset, Threshold, ThresholdMode


def sample_dataset() -> Dataset:
    return Dataset(
        name="sample",
        columns={"X": np.array([0.0, 1.0, 0.0]), "Y": np.array([0.0, 0.0, 1.0]), "V": np.array([1.0, 2.0, 3.0])},
        roles={"x": "X", "y": "Y", "value": "V"},
    )


def test_original_arrays_are_read_only():
    dataset = sample_dataset()
    with pytest.raises(ValueError):
        dataset.columns["V"][0] = 99


def test_derive_creates_new_dataset_and_lineage():
    original = sample_dataset()
    derived = original.derive("filtered", np.array([True, False, True]), {"type": "filter"})
    assert derived.parent_id == original.id
    assert derived.row_count == 2
    assert original.row_count == 3


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (Threshold(ThresholdMode.KEEP_RANGE, 1.5, 2.5), [False, True, False]),
        (Threshold(ThresholdMode.HIDE_RANGE, 1.5, 2.5), [True, False, True]),
        (Threshold(ThresholdMode.ABOVE, upper=2.0), [True, True, False]),
        (Threshold(ThresholdMode.BELOW, lower=2.0), [False, True, True]),
    ],
)
def test_threshold_modes(threshold, expected):
    assert threshold.mask(np.array([1.0, 2.0, 3.0])).tolist() == expected


