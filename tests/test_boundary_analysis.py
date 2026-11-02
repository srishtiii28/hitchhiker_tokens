from __future__ import annotations

import pytest

from tokenverse.language.boundary_analysis import find_crossing_tokens, inspect_boundary


def test_find_crossing_with_local_offsets():
    offsets = [(0, 3), (3, 7), (7, 9)]
    assert find_crossing_tokens(offsets, 5) == [1]


def test_no_crossing_on_exact_edges():
    offsets = [(0, 3), (3, 7)]
    assert find_crossing_tokens(offsets, 3) == []
    assert find_crossing_tokens(offsets, 0) == []
    assert find_crossing_tokens(offsets, 7) == []


def test_zero_length_and_missing_offsets_never_cross():
    assert find_crossing_tokens([(0, 0), None, (2, 2)], 1) == []


def test_crossing_with_real_tokenizer(local_tokenizer):
    report = inspect_boundary("becau", "se", local_tokenizer)
    assert report.boundary_index == 5
    assert report.combined == "because"
    for i in report.crossing_indices:
        s, e = report.combined_offsets[i]
        assert s < 5 < e
    # The tiny corpus repeats "because", so it is one token there.
    assert report.has_crossing
    assert not report.prefix_tokenization_matches
    assert "crosses the boundary" in report.describe()


def test_whitespace_boundary(local_tokenizer):
    report = inspect_boundary("hello", " world", local_tokenizer)
    assert not report.has_crossing
    assert report.prefix_tokenization_matches


def test_empty_prefix(fake_tokenizer):
    report = inspect_boundary("", "hello world", fake_tokenizer)
    assert report.prefix_tokens == []
    assert not report.has_crossing
    assert report.combined_prefix_ids == []


def test_empty_continuation(local_tokenizer):
    report = inspect_boundary("hello world", "", local_tokenizer)
    assert not report.has_crossing
    assert report.prefix_ids == report.combined_ids
    assert report.prefix_tokenization_matches


def test_unicode_offsets_are_code_points(local_tokenizer):
    report = inspect_boundary("caf", "é", local_tokenizer)
    assert report.boundary_index == 3
    assert all(e <= len(report.combined) for _, e in report.combined_offsets)


def test_fake_tokenizer_crossing(fake_tokenizer):
    report = inspect_boundary("hel", "lo there", fake_tokenizer)
    assert report.crossing_indices == [0]
    assert report.combined_tokens[0] == "hello"


def test_requires_fast_tokenizer(fake_tokenizer):
    fake_tokenizer.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizer"):
        inspect_boundary("a", "b", fake_tokenizer)
