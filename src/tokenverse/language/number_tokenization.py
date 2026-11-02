"""Three ways to group the digits of a number.

These show how a digit string could be split into tokens. For what GPT-2
actually does, use :func:`tokenverse.language.tokenizer_inspector.inspect_text`.

Supported input format:

- optional leading ``+`` or ``-``;
- an integer part, either plain digits (``1234567``) or with comma thousands
  separators in the usual positions (``1,234,567``);
- an optional decimal fraction (``.25``).

Commas are dropped before grouping. The sign and the decimal point each
become their own segment, and the fractional digits stay together as one.

Strategies, using ``12345``:

- ``"single"``: ``1 | 2 | 3 | 4 | 5``
- ``"l2r"`` (three digits, left to right): ``123 | 45``
- ``"r2l"`` (three digits, right to left): ``12 | 345``
"""

from __future__ import annotations

import re
from typing import Literal

Strategy = Literal["single", "l2r", "r2l"]
STRATEGIES: tuple[Strategy, ...] = ("single", "l2r", "r2l")

_NUMBER_RE = re.compile(
    r"""
    ^(?P<sign>[+-])?
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)
    (?:\.(?P<frac>\d+))?$
    """,
    re.VERBOSE,
)

ARITHMETIC_EXAMPLE = "3789 + 8791 = 12580"


def _group(digits: str, size: int, *, from_right: bool) -> list[str]:
    if not from_right:
        return [digits[i : i + size] for i in range(0, len(digits), size)]
    head = len(digits) % size
    groups = [digits[:head]] if head else []
    groups.extend(digits[i : i + size] for i in range(head, len(digits), size))
    return groups


def segment_number(value: str, strategy: Strategy) -> list[str]:
    """Split a number string into segments using one grouping strategy.

    Args:
        value: A number such as ``"12345"``, ``"-1,234,567"`` or ``"3.14"``.
            Surrounding whitespace is ignored.
        strategy: ``"single"``, ``"l2r"`` or ``"r2l"``.

    Returns:
        The segments in order, e.g. ``["-", "1", "234", "567"]``.

    Raises:
        ValueError: If ``value`` is not in the supported format or the
            strategy is unknown.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; choose one of {', '.join(STRATEGIES)}"
        )
    if not isinstance(value, str):
        raise ValueError(f"Expected a string, got {type(value).__name__}")
    match = _NUMBER_RE.match(value.strip())
    if match is None:
        raise ValueError(
            f"{value!r} is not a supported number. Use digits with an optional "
            "leading sign, optional comma thousands separators (e.g. 1,234,567) "
            "and an optional decimal part (e.g. 3.14)."
        )
    digits = match["int"].replace(",", "")
    if strategy == "single":
        groups = list(digits)
    else:
        groups = _group(digits, 3, from_right=strategy == "r2l")

    segments: list[str] = []
    if match["sign"]:
        segments.append(match["sign"])
    segments.extend(groups)
    if match["frac"] is not None:
        segments.extend([".", match["frac"]])
    return segments


def segment_expression(
    expression: str = ARITHMETIC_EXAMPLE,
) -> dict[Strategy, list[list[str]]]:
    """Segment every number in a simple ``a + b = c`` style expression.

    Numbers are found by splitting on whitespace and keeping the parts that
    parse as numbers; operators are skipped.

    Returns:
        ``{strategy: [segments_for_each_number]}``.
    """
    numbers = []
    for part in expression.split():
        if _NUMBER_RE.match(part):
            numbers.append(part)
    if not numbers:
        raise ValueError(f"No numbers found in {expression!r}")
    return {s: [segment_number(n, s) for n in numbers] for s in STRATEGIES}


def format_segments(segments: list[str]) -> str:
    """Render segments as ``'12 | 345'``."""
    return " | ".join(segments)
