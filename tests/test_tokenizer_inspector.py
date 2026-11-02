from __future__ import annotations

from unittest import mock

import pytest

from tokenverse.language import tokenizer_loader
from tokenverse.language.tokenizer_inspector import (
    EXAMPLES,
    compare_tokenizers,
    escape_whitespace,
    inspect_text,
)
from tokenverse.language.tokenizer_loader import TokenizerLoadError, load_tokenizer


@pytest.fixture(autouse=True)
def _clear_cache():
    tokenizer_loader.clear_tokenizer_cache()
    yield
    tokenizer_loader.clear_tokenizer_cache()


def _patch_auto(**kwargs):
    pytest.importorskip("transformers")
    return mock.patch("transformers.AutoTokenizer.from_pretrained", **kwargs)


def test_load_tokenizer_uses_expected_options_and_caches():
    fake = mock.Mock(is_fast=True)
    with _patch_auto(return_value=fake) as patched:
        assert load_tokenizer("gpt2") is fake
        assert load_tokenizer("gpt2") is fake
    patched.assert_called_once_with(
        "gpt2", use_fast=True, trust_remote_code=False, local_files_only=False
    )


def test_load_tokenizer_passes_local_files_only():
    with _patch_auto(return_value=mock.Mock(is_fast=True)) as patched:
        load_tokenizer("t5-small", local_files_only=True)
    assert patched.call_args.kwargs["local_files_only"] is True


def test_load_tokenizer_reports_network_failure():
    with (
        _patch_auto(side_effect=OSError("connection refused")),
        pytest.raises(TokenizerLoadError, match="gpt2.*internet"),
    ):
        load_tokenizer("gpt2")


def test_load_tokenizer_reports_missing_sentencepiece():
    with (
        _patch_auto(side_effect=ImportError("sentencepiece not found")),
        pytest.raises(TokenizerLoadError, match="SentencePiece"),
    ):
        load_tokenizer("t5-small")


def test_load_tokenizer_rejects_slow_tokenizer():
    with (
        _patch_auto(return_value=mock.Mock(is_fast=False)),
        pytest.raises(TokenizerLoadError, match="fast"),
    ):
        load_tokenizer("bert-base-uncased")


def test_inspect_text_is_coherent(local_tokenizer):
    text = "hello world 2026"
    report = inspect_text(text, local_tokenizer)
    assert report.num_tokens == len(report.tokens) == len(report.token_ids) > 0
    assert report.num_code_points == len(text)
    assert report.chars_per_token == pytest.approx(len(text) / report.num_tokens)
    assert len(report.offsets) == report.num_tokens
    # Offsets cover the text in order and the spans rebuild it.
    assert "".join(text[s:e] for s, e in report.offsets) == text
    assert report.decoded == text and report.round_trip
    # IDs are exactly what the tokenizer returns.
    assert (
        report.token_ids == local_tokenizer(text, add_special_tokens=False)["input_ids"]
    )
    assert "␣" in "".join(report.display_labels)
    assert not any(report.special_tokens_mask)


def test_inspect_text_handles_multibyte(local_tokenizer):
    report = inspect_text("café 你好", local_tokenizer)
    assert report.round_trip
    assert any("(cont.)" in label for label in report.display_labels)


def test_inspect_text_empty(local_tokenizer):
    report = inspect_text("", local_tokenizer)
    assert report.num_tokens == 0
    assert report.chars_per_token is None


def test_escape_whitespace_only_changes_labels():
    assert escape_whitespace("a b\nc\t") == "a␣b\\nc\\t"


def test_compare_tokenizers_keeps_ids_separate(local_tokenizer, fake_tokenizer):
    comparison = compare_tokenizers(
        "hello world", {"tiny": local_tokenizer, "fake": fake_tokenizer}
    )
    assert set(comparison.reports) == {"tiny", "fake"}
    assert comparison.token_counts() == {
        name: r.num_tokens for name, r in comparison.reports.items()
    }
    # Only counts and strings are compared across tokenizers, not IDs.
    assert not any("id" in attr for attr in dir(comparison) if not attr.startswith("_"))
    assert "specific to each vocabulary" in comparison.summary()


def test_examples_cover_numbers_and_multilingual():
    assert "847291038475" in EXAMPLES
    assert any(ord(ch) > 0x3000 for text in EXAMPLES for ch in text)
