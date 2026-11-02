from __future__ import annotations

import pytest

from tokenverse.language.number_tokenization import (
    ARITHMETIC_EXAMPLE,
    segment_expression,
    segment_number,
)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("single", ["1", "2", "3", "4", "5"]),
        ("l2r", ["123", "45"]),
        ("r2l", ["12", "345"]),
    ],
)
def test_spec_examples(strategy, expected):
    assert segment_number("12345", strategy) == expected


def test_signs_are_separate_segments():
    assert segment_number("-1234", "r2l") == ["-", "1", "234"]
    assert segment_number("+1234", "l2r") == ["+", "123", "4"]


def test_comma_separated_integers():
    assert segment_number("1,234,567", "r2l") == ["1", "234", "567"]
    assert segment_number("1,234,567", "l2r") == ["123", "456", "7"]
    assert segment_number("-12,345", "single") == ["-", "1", "2", "3", "4", "5"]


def test_decimal_part_kept_whole():
    assert segment_number("12345.678", "r2l") == ["12", "345", ".", "678"]
    assert segment_number("-0.5", "single") == ["-", "0", ".", "5"]


def test_exact_multiple_of_three():
    assert segment_number("123456", "r2l") == segment_number("123456", "l2r")


@pytest.mark.parametrize(
    "bad",
    ["", "abc", "12a", "1,23", "12,3456", ",123", "1.", ".5", "--1", "1 2", "1.2.3"],
)
def test_malformed_numbers_raise(bad):
    with pytest.raises(ValueError, match="not a supported number"):
        segment_number(bad, "single")


def test_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        segment_number("123", "middle")


def test_arithmetic_example_matches_article():
    groups = segment_expression(ARITHMETIC_EXAMPLE)
    assert groups["l2r"] == [["378", "9"], ["879", "1"], ["125", "80"]]
    assert groups["r2l"] == [["3", "789"], ["8", "791"], ["12", "580"]]
