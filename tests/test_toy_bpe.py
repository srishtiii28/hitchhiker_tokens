from __future__ import annotations

import pytest

from tokenverse.language.toy_bpe import END_OF_WORD, ToyBPE

CORPUS = (
    "low low low low low lower lower newest newest newest newest newest newest "
    "widest widest widest"
)


def test_merges_are_deterministic():
    a = ToyBPE().train(CORPUS, num_merges=10)
    b = ToyBPE().train(CORPUS, num_merges=10)
    assert a.merges == b.merges
    assert a.merges[:3] == [("e", "s"), ("es", "t"), ("est", END_OF_WORD)]


def test_ties_break_lexicographically():
    bpe = ToyBPE().train("ab cd", num_merges=1)
    # ('a','b'), ('b','</w>'), ('c','d'), ('d','</w>') all occur once.
    assert bpe.merges == [("a", "b")]


def test_merge_history_records_steps():
    bpe = ToyBPE().train(CORPUS, num_merges=5)
    assert [m.step for m in bpe.merge_history] == [1, 2, 3, 4, 5]
    first = bpe.merge_history[0]
    assert first.frequency == 9 and first.merged == "es"
    assert first.samples["newest"] == ["n", "e", "w", "es", "t", END_OF_WORD]
    assert "es" in bpe.vocab


def test_encode_decode_round_trip():
    bpe = ToyBPE().train(CORPUS, num_merges=10)
    tokens = bpe.encode("lower newer")
    assert tokens[0] == "low"
    assert bpe.decode(tokens) == "lower newer"


def test_encode_applies_merges_in_priority_order():
    bpe = ToyBPE().train(CORPUS, num_merges=10)
    assert bpe.encode("newest") == ["newest</w>"]


def test_whitespace_is_normalised():
    bpe = ToyBPE().train(CORPUS, num_merges=10)
    assert bpe.decode(bpe.encode("  lower \n\t newest ")) == "lower newest"


def test_empty_input():
    bpe = ToyBPE().train("", num_merges=5)
    assert bpe.merges == [] and bpe.vocab == []
    assert bpe.encode("") == []
    assert bpe.decode([]) == ""


def test_unseen_characters_stay_single():
    bpe = ToyBPE().train(CORPUS, num_merges=10)
    tokens = bpe.encode("lowΩz")
    assert "Ω" in tokens and "z" in tokens
    assert bpe.decode(tokens) == "lowΩz"


def test_stops_when_no_pairs_remain():
    bpe = ToyBPE().train("a a a", num_merges=50)
    assert len(bpe.merges) == 1


def test_negative_merges_rejected():
    with pytest.raises(ValueError):
        ToyBPE().train(CORPUS, num_merges=-1)
