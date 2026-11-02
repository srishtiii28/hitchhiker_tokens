"""Inspect how a tokenizer segments a piece of text.

The main entry point is :func:`inspect_text`, which returns a
:class:`TokenizationReport`. :func:`compare_tokenizers` runs the same text
through several tokenizers and returns one report per tokenizer.

Token IDs are vocabulary-specific. Reports from different tokenizers can be
compared on token count, token strings and spans, but their IDs are only
meaningful inside their own vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

EXAMPLES: list[str] = [
    "Time is an illusion. Lunchtime doubly so.",
    "Broken bottles break.",
    "Tokenisation is fascinating!",
    "847291038475",
    "2026",
    "Hello, world!",
    "café naïve résumé",
    "你好，世界",
    "print('hello, world')",
]

_SPACE_MARKERS = "▁Ġ"  # SentencePiece and GPT-2 byte-level space markers
_WHITESPACE_MARKERS = {" ": "␣", "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def escape_whitespace(text: str) -> str:
    """Make whitespace visible for display (``' '`` becomes ``'␣'``)."""
    return "".join(_WHITESPACE_MARKERS.get(ch, ch) for ch in text)


@dataclass
class TokenizationReport:
    """What a tokenizer did to one string.

    Attributes:
        text: The original input.
        tokenizer_name: ``name_or_path`` of the tokenizer, when available.
        tokens: Token strings from the vocabulary (e.g. ``'Ġworld'`` in GPT-2).
            These are vocabulary entries, not necessarily readable text.
        token_ids: Integer IDs, exactly as returned by the tokenizer.
        offsets: ``(start, end)`` code-point offsets into ``text`` for each
            token, or ``None`` if the tokenizer cannot provide them.
        special_tokens_mask: ``1`` where a token is a special token.
        display_labels: Readable labels: the original text span when offsets
            are available, otherwise the token string, with whitespace made
            visible.
        decoded: The result of decoding the complete ID sequence.
        round_trip: ``True`` if ``decoded == text``. Normalising tokenizers
            (for example BERT uncased, which lowercases and strips accents)
            cannot always reproduce the input exactly.
        special_tokens: The tokenizer's special tokens, by role.
    """

    text: str
    tokenizer_name: str
    tokens: list[str]
    token_ids: list[int]
    offsets: list[tuple[int, int]] | None
    special_tokens_mask: list[int]
    display_labels: list[str]
    decoded: str
    round_trip: bool
    special_tokens: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        """Number of tokens."""
        return len(self.token_ids)

    @property
    def num_code_points(self) -> int:
        """Number of Unicode code points (``len(text)``).

        This is not a count of user-perceived characters: one visible symbol
        such as a flag emoji can be several code points.
        """
        return len(self.text)

    @property
    def chars_per_token(self) -> float | None:
        """Unicode code points divided by token count (``None`` for no tokens)."""
        if self.num_tokens == 0:
            return None
        return self.num_code_points / self.num_tokens

    def rows(self) -> list[dict[str, Any]]:
        """One dict per token, convenient for printing a table."""
        rows = []
        for i, (tok, tid) in enumerate(zip(self.tokens, self.token_ids, strict=True)):
            rows.append(
                {
                    "index": i,
                    "token": tok,
                    "id": tid,
                    "offsets": self.offsets[i] if self.offsets else None,
                    "label": self.display_labels[i],
                    "special": bool(self.special_tokens_mask[i]),
                }
            )
        return rows

    def format_table(self) -> str:
        """Plain-text table of tokens, IDs and offsets."""
        lines = [f"{'#':>3}  {'id':>7}  {'offsets':>9}  {'token':<16} label"]
        for row in self.rows():
            off = row["offsets"]
            off_s = f"{off[0]}-{off[1]}" if off is not None else "n/a"
            flag = " [special]" if row["special"] else ""
            lines.append(
                f"{row['index']:>3}  {row['id']:>7}  {off_s:>9}  "
                f"{row['token']!r:<16} {row['label']}{flag}"
            )
        return "\n".join(lines)


def _display_labels(
    text: str, tokens: list[str], offsets: list[tuple[int, int]] | None
) -> list[str]:
    """Readable label for each token, preferring the original text span.

    Two cases need care:

    - SentencePiece can emit a bare space marker ``"▁"`` whose offsets point
      at the next character. Pure whitespace-marker tokens (``"▁"``, ``"Ġ"``)
      are labelled ``"␣"``.
    - Byte-level BPE can split one code point (e.g. a CJK character) across
      several tokens that share the same offsets. The first token shows the
      character; the rest are marked ``"(cont.)"``.
    """
    labels: list[str] = []
    previous: tuple[int, int] | None = None
    for i, tok in enumerate(tokens):
        span = offsets[i] if offsets is not None else None
        if tok and not tok.strip(_SPACE_MARKERS):
            labels.append("␣" * len(tok))
            continue
        if span is not None and span[1] > span[0]:
            label = escape_whitespace(text[span[0] : span[1]])
            if span == previous:
                label = f"{label}(cont.)"
        else:
            label = escape_whitespace(tok)
        labels.append(label)
        previous = span
    return labels


def inspect_text(
    text: str, tokenizer: Any, *, add_special_tokens: bool = False
) -> TokenizationReport:
    """Tokenise ``text`` and collect everything needed to explain the result.

    Args:
        text: The string to tokenise.
        tokenizer: A Hugging Face tokenizer. Offsets are requested only when
            ``tokenizer.is_fast`` is true.
        add_special_tokens: Defaults to ``False`` so the report measures the
            supplied text rather than, say, BERT's ``[CLS]``/``[SEP]`` wrapper.

    Returns:
        A :class:`TokenizationReport`.
    """
    is_fast = bool(getattr(tokenizer, "is_fast", False))
    encoding = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=is_fast,
        return_special_tokens_mask=True,
    )
    token_ids = [int(i) for i in encoding["input_ids"]]
    tokens = list(tokenizer.convert_ids_to_tokens(token_ids))
    offsets = (
        [(int(s), int(e)) for s, e in encoding["offset_mapping"]] if is_fast else None
    )
    special_mask = list(encoding.get("special_tokens_mask", [0] * len(token_ids)))
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=not add_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    return TokenizationReport(
        text=text,
        tokenizer_name=str(getattr(tokenizer, "name_or_path", "")),
        tokens=tokens,
        token_ids=token_ids,
        offsets=offsets,
        special_tokens_mask=special_mask,
        display_labels=_display_labels(text, tokens, offsets),
        decoded=decoded,
        round_trip=decoded == text,
        special_tokens=dict(getattr(tokenizer, "special_tokens_map", {}) or {}),
    )


@dataclass
class TokenizerComparison:
    """Reports for one text from several tokenizers.

    IDs from different vocabularies can't be compared, so this only lines up
    token counts and token strings.
    """

    text: str
    reports: dict[str, TokenizationReport]

    def token_counts(self) -> dict[str, int]:
        """Token count per tokenizer."""
        return {name: r.num_tokens for name, r in self.reports.items()}

    def token_strings(self) -> dict[str, list[str]]:
        """Readable token labels per tokenizer."""
        return {name: r.display_labels for name, r in self.reports.items()}

    def summary(self) -> str:
        """Short text summary for printing."""
        lines = [f"Text: {self.text!r}  ({len(self.text)} code points)"]
        for name, r in self.reports.items():
            joined = " | ".join(r.display_labels)
            lines.append(f"  {name:<18} {r.num_tokens:>3} tokens  {joined}")
        lines.append(
            "  (IDs are specific to each vocabulary, so they aren't compared here.)"
        )
        return "\n".join(lines)


def compare_tokenizers(
    text: str,
    tokenizers: Mapping[str, Any],
    *,
    add_special_tokens: bool = False,
) -> TokenizerComparison:
    """Run the same text through several tokenizers.

    Args:
        text: Input string.
        tokenizers: Mapping from a display name to a tokenizer.
        add_special_tokens: Passed to :func:`inspect_text`.
    """
    reports = {
        name: inspect_text(text, tok, add_special_tokens=add_special_tokens)
        for name, tok in tokenizers.items()
    }
    return TokenizerComparison(text=text, reports=reports)
