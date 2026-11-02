"""Diagnose how a prompt boundary lines up with tokenizer boundaries.

Related reading: the Partial Token Problem (Xu et al., "Are You Going to
Finish That?") and the Prompt Boundary Problem (Hayase et al., "Sampling from
Your Language Model One Byte at a Time").

This module only looks at offsets. It tokenises ``prefix + continuation``,
reports which tokens straddle the point where the prefix ends, and checks
whether the prefix tokenised on its own gives the same pieces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tokenverse.language.tokenizer_inspector import escape_whitespace


def find_crossing_tokens(
    offsets: Sequence[tuple[int, int] | None], boundary: int
) -> list[int]:
    """Indices of tokens whose span strictly crosses ``boundary``.

    A token crosses when ``start < boundary < end``. Tokens with zero-length
    or missing offsets (often special tokens) never cross.

    Args:
        offsets: ``(start, end)`` character offsets, one per token.
        boundary: Character index of the boundary.
    """
    crossing = []
    for i, span in enumerate(offsets):
        if span is None:
            continue
        start, end = span
        if end > start and start < boundary < end:
            crossing.append(i)
    return crossing


@dataclass
class BoundaryReport:
    """Result of :func:`inspect_boundary`.

    Attributes:
        prefix: Text before the boundary.
        continuation: Text after the boundary.
        combined: ``prefix + continuation``.
        boundary_index: ``len(prefix)``, the boundary position in ``combined``.
        combined_tokens: Token strings for ``combined``.
        combined_ids: Token IDs for ``combined``.
        combined_offsets: Offsets of each combined token into ``combined``.
        crossing_indices: Indices into ``combined_tokens`` that cross the
            boundary.
        prefix_tokens: Token strings when ``prefix`` is tokenised alone.
        prefix_ids: Token IDs when ``prefix`` is tokenised alone.
        prefix_offsets: Offsets for the prefix-only tokenisation.
        combined_prefix_ids: IDs of combined tokens that lie entirely before
            the boundary (``end <= boundary``, non-empty span).
    """

    prefix: str
    continuation: str
    combined: str
    boundary_index: int
    combined_tokens: list[str]
    combined_ids: list[int]
    combined_offsets: list[tuple[int, int]]
    crossing_indices: list[int]
    prefix_tokens: list[str]
    prefix_ids: list[int]
    prefix_offsets: list[tuple[int, int]]
    combined_prefix_ids: list[int]

    @property
    def has_crossing(self) -> bool:
        """``True`` if at least one combined token straddles the boundary."""
        return bool(self.crossing_indices)

    @property
    def prefix_tokenization_matches(self) -> bool:
        """Whether the prefix tokenised alone equals the prefix part of the
        combined tokenisation (and no token crosses the boundary)."""
        return not self.has_crossing and self.prefix_ids == self.combined_prefix_ids

    def describe(self) -> str:
        """Plain-language summary of the diagnostic."""

        def labels(tokens: list[str], offsets: list[tuple[int, int]], text: str):
            out = []
            for tok, (s, e) in zip(tokens, offsets, strict=True):
                out.append(escape_whitespace(text[s:e]) if e > s else repr(tok))
            return " | ".join(out) or "(none)"

        lines = [
            f"Prefix {self.prefix!r} + continuation {self.continuation!r} "
            f"(boundary at character {self.boundary_index})",
            "  prefix alone : "
            + labels(self.prefix_tokens, self.prefix_offsets, self.prefix),
            "  combined     : "
            + labels(self.combined_tokens, self.combined_offsets, self.combined),
        ]
        if self.has_crossing:
            for i in self.crossing_indices:
                s, e = self.combined_offsets[i]
                lines.append(
                    f"  token {i} {self.combined_tokens[i]!r} spans {s}-{e} and "
                    f"crosses the boundary: '{self.combined[s : self.boundary_index]}'"
                    f" is in the prefix, '{self.combined[self.boundary_index : e]}' in "
                    "the continuation."
                )
        else:
            lines.append("  no combined token crosses the boundary.")
        lines.append(
            "  prefix tokenisation "
            + ("matches" if self.prefix_tokenization_matches else "differs from")
            + " the prefix part of the combined tokenisation."
        )
        return "\n".join(lines)


def _encode(text: str, tokenizer: Any) -> tuple[list[str], list[int], list]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "inspect_boundary needs a fast tokenizer that returns offset mappings."
        )
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(i) for i in enc["input_ids"]]
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    tokens = list(tokenizer.convert_ids_to_tokens(ids))
    return tokens, ids, offsets


def inspect_boundary(prefix: str, continuation: str, tokenizer: Any) -> BoundaryReport:
    """Compare tokenising ``prefix`` alone with tokenising ``prefix + continuation``.

    Steps:

    1. Tokenise ``prefix`` on its own.
    2. Tokenise ``prefix + continuation`` with offset mappings.
    3. Mark combined tokens with ``start < len(prefix) < end``.
    4. Collect combined tokens that sit entirely inside the prefix, so they can
       be compared with step 1.

    No special tokens are added. The outcome depends on the tokenizer: the
    same strings can produce a crossing token in one vocabulary and a clean
    split in another.

    Args:
        prefix: The prompt text, possibly ending mid-word.
        continuation: The text that follows.
        tokenizer: A fast Hugging Face tokenizer.

    Raises:
        ValueError: If the tokenizer cannot return offsets.
    """
    combined = prefix + continuation
    boundary = len(prefix)
    c_tokens, c_ids, c_offsets = _encode(combined, tokenizer)
    p_tokens, p_ids, p_offsets = _encode(prefix, tokenizer) if prefix else ([], [], [])
    crossing = find_crossing_tokens(c_offsets, boundary)
    combined_prefix_ids = [
        tid
        for tid, (s, e) in zip(c_ids, c_offsets, strict=True)
        if e > s and e <= boundary
    ]
    return BoundaryReport(
        prefix=prefix,
        continuation=continuation,
        combined=combined,
        boundary_index=boundary,
        combined_tokens=c_tokens,
        combined_ids=c_ids,
        combined_offsets=c_offsets,
        crossing_indices=crossing,
        prefix_tokens=p_tokens,
        prefix_ids=p_ids,
        prefix_offsets=p_offsets,
        combined_prefix_ids=combined_prefix_ids,
    )
