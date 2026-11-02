"""Shared offline fixtures. Nothing here touches the network."""

from __future__ import annotations

import os

import pytest

# Belt and braces: make any accidental Hub access fail fast.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="session")
def local_tokenizer():
    """Tiny byte-level BPE fast tokenizer trained in memory."""
    pytest.importorskip("transformers")
    from tokenverse.language.local_tokenizer import build_local_tokenizer

    return build_local_tokenizer()


class FakeTokenizer:
    """Minimal tokenizer that splits on spaces and reports offsets.

    Used where a test needs full control over token spans.
    """

    is_fast = True
    name_or_path = "fake-whitespace"
    special_tokens_map = {"unk_token": "[UNK]"}

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab) + 100)

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        return_special_tokens_mask=False,
    ):
        ids, offsets, pos = [], [], 0
        for piece in text.split(" "):
            if piece:
                ids.append(self._id(piece))
                offsets.append((pos, pos + len(piece)))
            pos += len(piece) + 1
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        if return_special_tokens_mask:
            out["special_tokens_mask"] = [0] * len(ids)
        return out

    def convert_ids_to_tokens(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return [inverse[i] for i in ids]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(self.convert_ids_to_tokens(ids))


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
