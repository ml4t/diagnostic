"""Tests for shared formatting helpers."""

import pytest

from ml4t.diagnostic.utils import format_finite


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_format_finite_marks_non_finite_values_unavailable(value: float) -> None:
    assert format_finite(value, ".4%") == "N/A"


def test_format_finite_applies_standard_format_specs() -> None:
    assert format_finite(0.125, "+.1%") == "+12.5%"
