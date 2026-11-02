"""Shared dark theme and drawing helpers for every figure in the project."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PALETTE = {
    "bg": "#0f1117",
    "panel": "#1a1e29",
    "grid": "#2a2f3d",
    "text": "#e8e8ec",
    "muted": "#9aa3b2",
    "token_a": "#24324a",
    "token_b": "#2c2a45",
    "edge": "#6b7a99",
    "cyan": "#4fc3f7",
    "amber": "#ffb74d",
    "pink": "#f48fb1",
    "green": "#81c784",
    "violet": "#b39ddb",
}

# Accent per role, kept consistent across figures.
ROLE_COLOURS = {
    "cls": PALETTE["amber"],
    "patch": PALETTE["cyan"],
    "position": PALETTE["violet"],
    "boundary": PALETTE["pink"],
    "single": PALETTE["green"],
    "l2r": PALETTE["pink"],
    "r2l": PALETTE["cyan"],
}

MONO_FONTS = [
    "DejaVu Sans Mono",
    "Menlo",
    "Hiragino Sans",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Color Emoji",
]


def apply_dark_theme() -> None:
    """Set Matplotlib rcParams for the project's dark style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["text"],
            "grid.color": PALETTE["grid"],
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", *MONO_FONTS[1:]],
            "font.monospace": MONO_FONTS,
            "legend.facecolor": PALETTE["panel"],
            "legend.edgecolor": PALETTE["grid"],
        }
    )


@contextmanager
def quiet_glyph_warnings():
    """Silence 'Glyph missing' warnings when a font lacks CJK/emoji glyphs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Gg]lyph.*missing.*")
        warnings.filterwarnings("ignore", message=".*Matplotlib currently does not.*")
        yield


@lru_cache(maxsize=1)
def _supported_code_points() -> frozenset[int]:
    """Code points the bundled DejaVu fonts can draw.

    DejaVu ships with Matplotlib, so this is the same locally and in Colab.
    """
    from fontTools.ttLib import TTFont
    from matplotlib import font_manager

    points: set[int] = set()
    for family in ("DejaVu Sans Mono", "DejaVu Sans"):
        path = font_manager.findfont(font_manager.FontProperties(family=family))
        points.update(TTFont(path, lazy=True).getBestCmap())
    return frozenset(points)


def plot_safe(text: str) -> str:
    """Replace characters the plot font cannot draw with ``U+XXXX``.

    CJK characters and emoji would otherwise render as empty boxes.
    """
    supported = _supported_code_points()
    return "".join(
        ch if ord(ch) in supported or ch.isspace() else f"U+{ord(ch):04X}"
        for ch in text
    )


def new_axes(width: float, height: float, ax=None):
    """Return ``(fig, ax)``, creating a themed figure when ``ax`` is ``None``."""
    apply_dark_theme()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def blank(ax) -> None:
    """Hide axes spines and ticks for diagram-style plots."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def box_width(label: str, sublabel: str = "", char_w: float = 0.62) -> float:
    """Width in data units for a token box holding ``label`` (monospace)."""
    return max(len(plot_safe(label)), len(plot_safe(sublabel)), 1) * char_w + 0.5


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    face: str | None = None,
    edge: str | None = None,
    sublabel: str = "",
    hatch: str | None = None,
    fontsize: float = 11,
    text_colour: str | None = None,
    linewidth: float = 1.2,
    mono: bool = True,
) -> None:
    """Draw a rounded box with a centred label and an optional sublabel below."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=face or PALETTE["token_a"],
        edgecolor=edge or PALETTE["edge"],
        linewidth=linewidth,
        hatch=hatch,
    )
    ax.add_patch(patch)
    family = "monospace" if mono else "sans-serif"
    ax.text(
        x + w / 2,
        y + h / 2,
        plot_safe(label),
        ha="center",
        va="center",
        fontsize=fontsize,
        family=family,
        color=text_colour or PALETTE["text"],
        bbox=(
            {
                "boxstyle": "round,pad=0.15",
                "facecolor": face or PALETTE["token_a"],
                "edgecolor": "none",
            }
            if hatch
            else None
        ),
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y - 0.18,
            plot_safe(sublabel),
            ha="center",
            va="top",
            fontsize=fontsize * 0.78,
            family="monospace",
            color=PALETTE["muted"],
        )


def draw_token_row(
    ax,
    labels: Sequence[str],
    y: float,
    *,
    x0: float = 0.0,
    sublabels: Sequence[str] | None = None,
    faces: Sequence[str] | None = None,
    edges: Sequence[str] | None = None,
    hatches: Sequence[str | None] | None = None,
    gap: float = 0.15,
    height: float = 0.8,
    fontsize: float = 11,
) -> list[tuple[float, float]]:
    """Draw a row of token boxes. Returns ``(x_start, x_end)`` for each box.

    Faces alternate between two shades by default and every box has a visible
    border, so neighbouring tokens are distinguishable without colour.
    """
    spans = []
    x = x0
    for i, label in enumerate(labels):
        sub = sublabels[i] if sublabels else ""
        w = box_width(label, sub)
        face = faces[i] if faces else (PALETTE["token_a"], PALETTE["token_b"])[i % 2]
        draw_box(
            ax,
            x,
            y,
            w,
            height,
            label,
            face=face,
            edge=edges[i] if edges else None,
            sublabel=sub,
            hatch=hatches[i] if hatches else None,
            fontsize=fontsize,
        )
        spans.append((x, x + w))
        x += w + gap
    return spans


def finish(fig, title: str | None = None, *, caption: str | None = None) -> None:
    """Add a title and optional caption, then tighten the layout."""
    if title:
        fig.suptitle(title, color=PALETTE["text"], fontsize=14, fontweight="bold")
    if caption:
        fig.text(
            0.5,
            0.01,
            caption,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=PALETTE["muted"],
            wrap=True,
        )
    fig.tight_layout(rect=(0, 0.05 if caption else 0, 1, 0.95 if title else 1))
