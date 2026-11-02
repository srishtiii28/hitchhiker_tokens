"""Inspect text as Unicode code points and UTF-8 bytes.

Three different things are often all called "characters":

- a **grapheme** is what a reader sees as one symbol (e.g. a flag emoji);
- a **code point** is one Unicode value such as ``U+00E9``; Python's
  ``len(text)`` counts these;
- a **byte** is one of the 1-4 UTF-8 bytes used to store a code point.

One grapheme can be several code points (``"é"`` can be ``e`` plus a combining
accent; a family emoji joins several emoji with zero-width joiners), and one
code point can be several bytes. Byte-level tokenizers such as GPT-2's start
from the bytes.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class ByteRecord:
    """One Unicode code point and its UTF-8 encoding.

    Attributes:
        char: The code point as a one-character string.
        code_point: ``"U+XXXX"`` notation (at least four hex digits).
        utf8_hex: The UTF-8 bytes as space-separated hex, e.g. ``"c3 a9"``.
        num_bytes: Number of UTF-8 bytes.
        name: The Unicode character name, or ``""`` if it has none.
    """

    char: str
    code_point: str
    utf8_hex: str
    num_bytes: int
    name: str

    @property
    def utf8_bytes(self) -> bytes:
        """The raw UTF-8 bytes."""
        return self.char.encode("utf-8")


def inspect_bytes(text: str) -> list[ByteRecord]:
    """Return one :class:`ByteRecord` per Unicode code point in ``text``."""
    records = []
    for ch in text:
        encoded = ch.encode("utf-8")
        records.append(
            ByteRecord(
                char=ch,
                code_point=f"U+{ord(ch):04X}",
                utf8_hex=encoded.hex(" "),
                num_bytes=len(encoded),
                name=unicodedata.name(ch, ""),
            )
        )
    return records


def byte_summary(text: str) -> dict[str, int]:
    """Counts of code points and UTF-8 bytes for ``text``."""
    return {
        "code_points": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
    }


def format_byte_table(text: str) -> str:
    """Plain-text table of code points and bytes."""
    lines = [f"{'char':<6}{'code point':<12}{'bytes':>5}  utf-8 hex      name"]
    for r in inspect_bytes(text):
        shown = repr(r.char) if r.char.isspace() or not r.char.isprintable() else r.char
        lines.append(
            f"{shown:<6}{r.code_point:<12}{r.num_bytes:>5}  {r.utf8_hex:<14} {r.name}"
        )
    return "\n".join(lines)
