"""Figures for the language-side experiments.

Every function returns a Matplotlib ``Figure`` and accepts an optional ``ax``
where it makes sense. Tokens are drawn as bordered boxes with text labels so
the figures stay readable without relying on colour.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colormaps
from matplotlib.patches import Rectangle

from tokenverse.language.byte_analysis import inspect_bytes
from tokenverse.language.number_tokenization import (
    STRATEGIES,
    segment_expression,
    segment_number,
)
from tokenverse.language.tokenizer_inspector import escape_whitespace
from tokenverse.visualization._theme import (
    PALETTE,
    ROLE_COLOURS,
    blank,
    box_width,
    draw_box,
    draw_token_row,
    finish,
    new_axes,
    plot_safe,
    quiet_glyph_warnings,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from tokenverse.language.boundary_analysis import BoundaryReport
    from tokenverse.language.tokenizer_inspector import (
        TokenizationReport,
        TokenizerComparison,
    )
    from tokenverse.language.toy_bpe import ToyBPE

STRATEGY_TITLES = {
    "single": "Single digit",
    "l2r": "Three-digit, left to right (L2R)",
    "r2l": "Three-digit, right to left (R2L)",
}


def _wrap(
    labels: Sequence[str], sublabels: Sequence[str] | None, max_width: float
) -> list[list[int]]:
    """Split token indices into rows no wider than ``max_width`` data units."""
    rows: list[list[int]] = [[]]
    width = 0.0
    for i, label in enumerate(labels):
        w = box_width(label, sublabels[i] if sublabels else "") + 0.15
        if rows[-1] and width + w > max_width:
            rows.append([])
            width = 0.0
        rows[-1].append(i)
        width += w
    return rows


def _token_rows(
    ax,
    labels: Sequence[str],
    *,
    sublabels: Sequence[str] | None = None,
    max_width: float = 30.0,
    row_height: float = 1.6,
    y0: float = 0.0,
    edges: Sequence[str] | None = None,
    hatches: Sequence[str | None] | None = None,
) -> float:
    """Draw wrapped token rows starting at ``y0`` going down. Returns bottom y."""
    rows = _wrap(labels, sublabels, max_width)
    y = y0
    for row in rows:
        draw_token_row(
            ax,
            [labels[i] for i in row],
            y,
            sublabels=[sublabels[i] for i in row] if sublabels else None,
            edges=[edges[i] for i in row] if edges else None,
            hatches=[hatches[i] for i in row] if hatches else None,
        )
        y -= row_height
    return y


def plot_token_overlay(
    report: TokenizationReport, *, show_ids: bool = True, max_width: float = 30.0
) -> Figure:
    """Show each token as a box labelled with its text span and (optionally) ID.

    Labels come from the report's offset-based ``display_labels``, so spaces
    appear as ``␣``.
    """
    labels = report.display_labels
    subs = [str(i) for i in report.token_ids] if show_ids else None
    n_rows = len(_wrap(labels, subs, max_width)) or 1
    fig, ax = new_axes(10, 1.2 + 1.3 * n_rows)
    with quiet_glyph_warnings():
        bottom = _token_rows(ax, labels, sublabels=subs, max_width=max_width)
        ax.set_xlim(-0.3, max_width + 0.3)
        ax.set_ylim(bottom + 0.6, 1.2)
        blank(ax)
        name = report.tokenizer_name or "tokenizer"
        title = (
            f"{name}: {report.num_tokens} tokens for "
            f"{report.num_code_points} code points"
        )
        cap = "Box = one token (text span shown, ␣ = space)"
        if show_ids:
            cap += "; number below = token ID in this tokenizer's vocabulary"
        finish(fig, title, caption=cap)
    return fig


def plot_tokenizer_rows(
    comparison: TokenizerComparison, *, max_width: float = 30.0
) -> Figure:
    """Draw the same text segmented by each tokenizer, one block per tokenizer.

    Token IDs are left out on purpose: they belong to different vocabularies.
    """
    blocks = []
    total_rows = 0
    for name, rep in comparison.reports.items():
        n = len(_wrap(rep.display_labels, None, max_width)) or 1
        blocks.append((name, rep, n))
        total_rows += n
    fig, ax = new_axes(10, 1.0 + 0.9 * total_rows + 0.6 * len(blocks))
    y = 0.0
    with quiet_glyph_warnings():
        for name, rep, _ in blocks:
            ax.text(
                0,
                y + 1.0,
                f"{name}  ·  {rep.num_tokens} tokens",
                fontsize=11,
                color=PALETTE["cyan"],
                fontweight="bold",
            )
            y = _token_rows(
                ax, rep.display_labels, max_width=max_width, row_height=1.05, y0=y
            )
            y -= 0.7
        ax.set_xlim(-0.3, max_width + 0.3)
        ax.set_ylim(y + 0.5, 1.6)
        blank(ax)
        finish(
            fig,
            plot_safe(f"Same text, different tokenizers: {comparison.text!r}"),
            caption="Token IDs are omitted: each tokenizer has its own vocabulary.",
        )
    return fig


def plot_token_counts(
    comparison: TokenizerComparison | dict[str, dict[str, int]],
) -> Figure:
    """Bar chart of token counts per tokenizer.

    Args:
        comparison: A single :class:`TokenizerComparison`, or
            ``{text: {tokenizer: count}}`` for several texts (grouped bars).
    """
    if hasattr(comparison, "token_counts"):
        data = {comparison.text: comparison.token_counts()}
    else:
        data = comparison
    texts = list(data)
    names = list(next(iter(data.values())))
    colours = [PALETTE["cyan"], PALETTE["amber"], PALETTE["violet"], PALETTE["green"]]
    hatches = ["", "//", "..", "xx"]
    fig, ax = new_axes(10, 0.8 + 0.55 * len(texts) * len(names))
    height = 0.8 / len(names)
    y = np.arange(len(texts))
    with quiet_glyph_warnings():
        for j, name in enumerate(names):
            counts = [data[t][name] for t in texts]
            bars = ax.barh(
                y + j * height,
                counts,
                height=height,
                label=name,
                color=colours[j % len(colours)],
                hatch=hatches[j % len(hatches)],
                edgecolor=PALETTE["bg"],
            )
            for bar, c in zip(bars, counts, strict=True):
                ax.text(
                    bar.get_width() + 0.15,
                    bar.get_y() + bar.get_height() / 2,
                    str(c),
                    va="center",
                    fontsize=9,
                    color=PALETTE["text"],
                )
        ax.set_yticks(y + height * (len(names) - 1) / 2)
        ax.set_yticklabels(
            [plot_safe(repr(t) if len(t) < 32 else repr(t[:29] + "…")) for t in texts]
        )
        ax.invert_yaxis()
        ax.set_xlabel("number of tokens")
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=9)
        finish(fig, "Token counts by tokenizer")
    return fig


def plot_bpe_merges(
    bpe: ToyBPE,
    words: Sequence[str] | None = None,
    *,
    steps: Sequence[int] | None = None,
) -> Figure:
    """Show how a few words are segmented after successive BPE merges.

    Args:
        bpe: A trained :class:`~tokenverse.language.toy_bpe.ToyBPE`.
        words: Words to show; defaults to the words recorded in the history.
        steps: Merge steps to show (0 = before any merge). Defaults to all.
    """
    from tokenverse.language.toy_bpe import ToyBPE as _ToyBPE

    history = bpe.merge_history
    if words is None:
        words = list(history[0].samples) if history else []
    if steps is None:
        steps = list(range(len(history) + 1))
    n_rows = len(steps)
    segmentations = []
    for step in steps:
        partial = _ToyBPE(merges=list(bpe.merges[:step]))
        segmentations.append(
            [[p.replace("</w>", "_") for p in partial.encode_word(w)] for w in words]
        )
    gap = 0.08
    # Each word gets a column as wide as its widest (earliest) segmentation.
    col_widths = [
        max(sum(box_width(p) + gap for p in seg[c]) for seg in segmentations)
        for c in range(len(words))
    ]
    col_x = np.concatenate([[0.0], np.cumsum([w + 0.9 for w in col_widths])])
    total_w = float(col_x[-1])
    fig, ax = new_axes(10, 0.9 + 0.62 * n_rows)
    for r, step in enumerate(steps):
        y = -r * 1.1
        if step == 0:
            tag = "start"
        else:
            m = history[step - 1]
            tag = f"#{step} {m.pair[0]}+{m.pair[1]} ({m.frequency})"
        ax.text(
            -0.4,
            y + 0.38,
            tag,
            ha="right",
            va="center",
            fontsize=9,
            family="monospace",
            color=PALETTE["amber"] if step else PALETTE["muted"],
        )
        for c, pieces in enumerate(segmentations[r]):
            draw_token_row(
                ax, pieces, y, x0=float(col_x[c]), height=0.75, fontsize=9.5, gap=gap
            )
    for c, word in enumerate(words):
        ax.text(
            col_x[c] + col_widths[c] / 2,
            1.25,
            word,
            ha="center",
            fontsize=10,
            color=PALETTE["cyan"],
            fontweight="bold",
        )
    label_w = (
        max(len(f"#{len(history)} {a}+{b} (00)") for a, b in bpe.merges) * 0.62
        if bpe.merges
        else 4
    )
    ax.set_xlim(-label_w - 0.8, total_w)
    ax.set_ylim(-(n_rows - 1) * 1.1 - 0.5, 1.8)
    blank(ax)
    finish(
        fig,
        "BPE merges, one step at a time",
        caption=(
            "Left: merge number, merged pair and its weighted count. "
            "'_' marks </w> (end of word)."
        ),
    )
    return fig


def plot_number_groupings(values: Sequence[str] | str) -> Figure:
    """Compare single-digit, L2R and R2L grouping for one or more numbers.

    Args:
        values: A number string, a list of them, or an arithmetic expression
            such as ``"3789 + 8791 = 12580"``.
    """
    if isinstance(values, str) and any(op in values for op in "+=") and " " in values:
        expr = segment_expression(values)
        numbers = [p for p in values.split() if p not in {"+", "-", "=", "*", "/"}]
        title = f"Grouping the digits of {values}"
    else:
        numbers = [values] if isinstance(values, str) else list(values)
        expr = {s: [segment_number(n, s) for n in numbers] for s in STRATEGIES}
        title = "Three ways to group the same digits"

    fig, axes = new_axes(8, 1.0 + 0.55 * len(numbers) * 3 + 0.9)
    ax = axes
    y = 0.0
    for strategy in STRATEGIES:
        colour = ROLE_COLOURS[strategy]
        ax.text(
            0,
            y + 0.9,
            STRATEGY_TITLES[strategy],
            fontsize=11.5,
            color=colour,
            fontweight="bold",
        )
        for number, segs in zip(numbers, expr[strategy], strict=True):
            ax.text(
                2.6,
                y + 0.35,
                number,
                ha="right",
                va="center",
                family="monospace",
                fontsize=10,
                color=PALETTE["muted"],
            )
            # Right-align the integer digits so place values line up across rows.
            total = sum(box_width(s) + 0.12 for s in segs)
            draw_token_row(
                ax,
                segs,
                y,
                x0=9 - total,
                edges=[colour] * len(segs),
                gap=0.12,
                height=0.7,
            )
            y -= 0.95
        y -= 0.75
    ax.set_xlim(0, 12)
    ax.set_ylim(y + 0.5, 1.6)
    blank(ax)
    finish(
        fig,
        title,
        caption=("Rows are right-aligned so the units digits line up."),
    )
    return fig


def plot_boundary(report: BoundaryReport) -> Figure:
    """Show prefix-only and combined tokenisations with the boundary marked.

    Tokens that cross the boundary get a highlighted border and hatching.
    """
    fig, ax = new_axes(9, 3.6)
    combined_labels = [
        escape_whitespace(report.combined[s:e]) if e > s else tok
        for tok, (s, e) in zip(
            report.combined_tokens, report.combined_offsets, strict=True
        )
    ]
    prefix_labels = [
        escape_whitespace(report.prefix[s:e]) if e > s else tok
        for tok, (s, e) in zip(report.prefix_tokens, report.prefix_offsets, strict=True)
    ]
    crossing = set(report.crossing_indices)
    edges = [
        ROLE_COLOURS["boundary"] if i in crossing else PALETTE["edge"]
        for i in range(len(combined_labels))
    ]
    hatches = ["//" if i in crossing else None for i in range(len(combined_labels))]

    # Place tokens by character offset so the boundary line is meaningful.
    char_w = 0.62
    with quiet_glyph_warnings():
        ax.text(
            0,
            2.35,
            plot_safe(f"prefix alone: {report.prefix!r}"),
            fontsize=10.5,
            color=PALETTE["muted"],
        )
        for label, (s, e) in zip(prefix_labels, report.prefix_offsets, strict=True):
            draw_box(ax, s * char_w, 1.35, max(e - s, 1) * char_w - 0.06, 0.7, label)
        ax.text(
            0,
            0.65,
            plot_safe(f"prefix + continuation: {report.combined!r}"),
            fontsize=10.5,
            color=PALETTE["muted"],
        )
        for i, (label, (s, e)) in enumerate(
            zip(combined_labels, report.combined_offsets, strict=True)
        ):
            draw_box(
                ax,
                s * char_w,
                -0.35,
                max(e - s, 1) * char_w - 0.06,
                0.7,
                label,
                edge=edges[i],
                hatch=hatches[i],
                linewidth=2 if i in crossing else 1.2,
            )
        bx = report.boundary_index * char_w - 0.03
        ax.vlines(
            bx,
            -0.55,
            2.2,
            color=ROLE_COLOURS["boundary"],
            linestyle="--",
            linewidth=1.6,
        )
        ax.text(
            bx,
            -0.65,
            "prompt ends here",
            ha="center",
            va="top",
            fontsize=9.5,
            color=ROLE_COLOURS["boundary"],
        )
        width = max(len(report.combined), 8) * char_w
        ax.set_xlim(-0.4, max(width + 0.4, 9))
        ax.set_ylim(-1.3, 2.8)
        blank(ax)
        verdict = (
            f"{len(crossing)} token(s) cross the boundary (hatched)"
            if crossing
            else "no token crosses the boundary"
        )
        finish(
            fig,
            f"Token boundary diagnostic: {verdict}",
            caption=(
                "Boxes are placed at their character offsets. "
                "No probabilities are computed."
            ),
        )
    return fig


def plot_utf8_bytes(text: str) -> Figure:
    """Show each code point of ``text`` above its UTF-8 bytes."""
    records = inspect_bytes(text)
    fig, ax = new_axes(10, 2.8)
    x = 0.0
    with quiet_glyph_warnings():
        for i, r in enumerate(records):
            hexes = r.utf8_hex.split()
            w = max(len(hexes) * 0.95, 1.6)
            face = (PALETTE["token_a"], PALETTE["token_b"])[i % 2]
            shown = escape_whitespace(r.char)
            draw_box(ax, x, 1.2, w - 0.1, 0.8, shown, face=face, fontsize=13)
            ax.text(
                x + (w - 0.1) / 2,
                1.05,
                r.code_point,
                ha="center",
                va="top",
                fontsize=8,
                family="monospace",
                color=PALETTE["muted"],
            )
            for j, hx in enumerate(hexes):
                draw_box(
                    ax,
                    x + j * 0.95,
                    -0.15,
                    0.85,
                    0.65,
                    hx,
                    face=PALETTE["panel"],
                    edge=PALETTE["cyan"],
                    fontsize=9,
                )
            x += w + 0.2
        ax.text(
            -0.2,
            1.6,
            "code point",
            ha="right",
            va="center",
            fontsize=9,
            color=PALETTE["muted"],
        )
        ax.text(
            -0.2,
            0.17,
            "UTF-8 bytes",
            ha="right",
            va="center",
            fontsize=9,
            color=PALETTE["cyan"],
        )
        ax.set_xlim(-2.6, max(x, 10))
        ax.set_ylim(-0.6, 2.4)
        blank(ax)
        n_bytes = sum(r.num_bytes for r in records)
        finish(fig, f"{len(records)} code points, {n_bytes} UTF-8 bytes")
    return fig


def plot_tokenization_pipeline(
    tokens: Sequence[str],
    token_ids: Sequence[int],
    embeddings: np.ndarray,
    text: str,
) -> Figure:
    """Diagram: text -> token strings -> token IDs -> embedding vectors.

    Args:
        tokens: Readable token labels.
        token_ids: IDs from the same tokenizer.
        embeddings: Array of shape ``[len(tokens), d]`` (small ``d``).
        text: The original text.
    """
    fig, ax = new_axes(10, 5.6)
    stage_x = 0.0
    with quiet_glyph_warnings():
        draw_box(
            ax,
            3.0,
            4.6,
            16,
            0.85,
            text,
            face=PALETTE["panel"],
            edge=PALETTE["muted"],
            fontsize=12,
        )
        rows = [
            ("token strings", 3.2, [escape_whitespace(t) for t in tokens]),
            ("token IDs", 1.8, [str(i) for i in token_ids]),
        ]
        widths = [
            max(box_width(a), box_width(b))
            for a, b in zip(*[r[2] for r in rows], strict=True)
        ]
        total = sum(widths) + 0.2 * (len(widths) - 1)
        x0 = 11 - total / 2
        for name, y, labels in rows:
            ax.text(
                stage_x,
                y + 0.35,
                name,
                ha="left",
                va="center",
                fontsize=10,
                color=PALETTE["muted"],
            )
            x = x0
            for w, lab in zip(widths, labels, strict=True):
                draw_box(ax, x, y, w, 0.75, lab)
                x += w + 0.2
        # Embedding heat strips.
        x = x0
        vmax = float(np.abs(embeddings).max()) or 1.0
        ax.text(
            stage_x,
            0.0,
            "embedding\nvectors",
            ha="left",
            va="center",
            fontsize=10,
            color=PALETTE["muted"],
        )
        for w, vec in zip(widths, embeddings, strict=True):
            d = len(vec)
            cell = 1.5 / d
            for k, v in enumerate(vec):
                ax.add_patch(
                    Rectangle(
                        (x + w / 2 - 0.3, 0.75 - (k + 1) * cell),
                        0.6,
                        cell,
                        facecolor=colormaps["coolwarm"](0.5 + 0.5 * v / vmax),
                        edgecolor=PALETTE["bg"],
                        linewidth=0.5,
                    )
                )
            x += w + 0.2
        for y_from, y_to in ((4.55, 4.05), (3.15, 2.6), (1.75, 0.85)):
            ax.annotate(
                "",
                xy=(11, y_to),
                xytext=(11, y_from),
                arrowprops={"arrowstyle": "->", "color": PALETTE["cyan"], "lw": 1.5},
            )
        ax.text(
            11.3,
            4.3,
            "tokenizer splits",
            fontsize=9,
            color=PALETTE["cyan"],
            va="center",
        )
        ax.text(
            11.3,
            2.88,
            "vocabulary lookup",
            fontsize=9,
            color=PALETTE["cyan"],
            va="center",
        )
        ax.text(
            11.3,
            1.3,
            "embedding table row",
            fontsize=9,
            color=PALETTE["cyan"],
            va="center",
        )
        ax.set_xlim(-0.2, 22)
        ax.set_ylim(-1.0, 5.7)
        blank(ax)
        finish(
            fig,
            "From text to vectors",
            caption=(
                "IDs are row numbers into an embedding table; the vectors (random "
                "here) are what the model learns and uses."
            ),
        )
    return fig
