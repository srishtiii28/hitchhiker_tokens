from __future__ import annotations

import pytest

from tokenverse.language.byte_analysis import byte_summary, inspect_bytes


@pytest.mark.parametrize("text", ["A", "café", "你好", "🙂", "é", "👍🏽", ""])
def test_bytes_match_python_encoding(text):
    records = inspect_bytes(text)
    assert len(records) == len(text)
    assert b"".join(r.utf8_bytes for r in records) == text.encode("utf-8")
    for r in records:
        assert bytes.fromhex(r.utf8_hex.replace(" ", "")) == r.char.encode("utf-8")
        assert r.num_bytes == len(r.char.encode("utf-8"))


def test_byte_lengths_by_script():
    lengths = {r.char: r.num_bytes for r in inspect_bytes("Aé你🙂")}
    assert lengths == {"A": 1, "é": 2, "你": 3, "🙂": 4}


def test_code_point_format():
    records = inspect_bytes("A🙂")
    assert records[0].code_point == "U+0041"
    assert records[1].code_point == "U+1F642"
    assert records[1].utf8_hex == "f0 9f 99 82"


def test_combining_sequence_is_two_code_points():
    # One visible "é" written as e + combining acute accent.
    records = inspect_bytes("é")
    assert [r.code_point for r in records] == ["U+0065", "U+0301"]
    assert byte_summary("é") == {"code_points": 2, "utf8_bytes": 3}
