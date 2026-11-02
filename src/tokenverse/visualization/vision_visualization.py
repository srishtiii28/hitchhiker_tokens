"""Figures for the Vision Transformer walkthrough.

The token sequence is drawn as ``[CLS, patch_0, ..., patch_{N-1}]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from tokenverse.visualization._theme import (
    PALETTE,
    ROLE_COLOURS,
    blank,
    draw_box,
    finish,
    new_axes,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT_SHAPES: list[tuple[str, str]] = [
    ("Input batch", "[2, 3, 224, 224]"),
    ("Flattened patches (before projection)", "[2, 196, 768]"),
    ("Patch embeddings", "[2, 196, 192]"),
    ("Expanded CLS token", "[2, 1, 192]"),
    ("Sequence after CLS concatenation", "[2, 197, 192]"),
    ("Sequence after positional embeddings", "[2, 197, 192]"),
    ("Encoded sequence", "[2, 197, 192]"),
    ("CLS representation", "[2, 192]"),
    ("Classifier logits (10 classes)", "[2, 10]"),
]


def _to_array(image) -> np.ndarray:
    """Accept a PIL image, an ``[H, W, 3]`` array or a ``[3, H, W]`` tensor."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if hasattr(image, "detach"):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        return np.clip(np.transpose(arr, (1, 2, 0)), 0, 1)
    return np.asarray(image)


def plot_patch_grid(
    image, patch_size: int, *, show_indices: bool = True, ax=None
) -> Figure:
    """Overlay the patch grid on an image, numbering patches row-major."""
    arr = _to_array(image)
    h, w = arr.shape[:2]
    fig, ax = new_axes(6.5, 6.5, ax)
    ax.imshow(arr)
    for x in range(0, w + 1, patch_size):
        ax.axvline(x - 0.5, color=PALETTE["text"], linewidth=0.8, alpha=0.8)
    for y in range(0, h + 1, patch_size):
        ax.axhline(y - 0.5, color=PALETTE["text"], linewidth=0.8, alpha=0.8)
    cols = w // patch_size
    if show_indices:
        size = 7 if cols > 10 else 9
        for r in range(h // patch_size):
            for c in range(cols):
                ax.text(
                    c * patch_size + 2,
                    r * patch_size + 2,
                    str(r * cols + c),
                    fontsize=size,
                    va="top",
                    ha="left",
                    color="white",
                    bbox={
                        "boxstyle": "round,pad=0.1",
                        "facecolor": "#000000aa",
                        "edgecolor": "none",
                    },
                )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    blank(ax)
    n = (h // patch_size) * cols
    ax.set_title(f"{h}×{w} image → {n} patches of {patch_size}×{patch_size}")
    return fig


def plot_patch_thumbnails(
    image, patch_size: int, indices: Sequence[int] = (0, 1, 2, 14, 97, 195)
) -> Figure:
    """Show selected patches enlarged, labelled with their row-major index."""
    arr = _to_array(image)
    cols = arr.shape[1] // patch_size
    fig, axes = new_axes(10, 2.6)
    blank(axes)
    for k, idx in enumerate(indices):
        sub = fig.add_subplot(1, len(indices), k + 1)
        r, c = divmod(idx, cols)
        patch = arr[
            r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size
        ]
        sub.imshow(patch, interpolation="nearest")
        sub.set_title(f"patch {idx}\n(row {r}, col {c})", fontsize=10)
        blank(sub)
        for spine in sub.spines.values():
            spine.set_visible(True)
            spine.set_color(ROLE_COLOURS["patch"])
    finish(fig, "Individual patches (row-major index)")
    return fig


def plot_cls_sequence(num_patches: int = 196, *, shown: int = 5) -> Figure:
    """Diagram of the token sequence: one CLS token followed by every patch."""
    fig, ax = new_axes(10, 3.0)
    x = 0.0
    w, h = 1.35, 0.9
    labels = ["[CLS]"] + [f"P{i}" for i in range(shown)] + ["…"]
    labels += [f"P{num_patches - 2}", f"P{num_patches - 1}"]
    positions = ["pos 0"] + [f"pos {i + 1}" for i in range(shown)] + [""]
    positions += [f"pos {num_patches - 1}", f"pos {num_patches}"]
    for label, pos in zip(labels, positions, strict=True):
        if label == "…":
            ax.text(x + w / 2, h / 2, "…", ha="center", va="center", fontsize=18)
        else:
            is_cls = label == "[CLS]"
            draw_box(
                ax,
                x,
                0,
                w,
                h,
                label,
                face=PALETTE["panel"] if is_cls else PALETTE["token_a"],
                edge=ROLE_COLOURS["cls"] if is_cls else ROLE_COLOURS["patch"],
                hatch="//" if is_cls else None,
                linewidth=2.2 if is_cls else 1.2,
                fontsize=11,
            )
            ax.text(
                x + w / 2,
                -0.2,
                pos,
                ha="center",
                va="top",
                fontsize=8.5,
                color=ROLE_COLOURS["position"],
                family="monospace",
            )
        x += w + 0.18
    ax.annotate(
        "learned vector, the same for every image\n(prepended, not replacing a patch)",
        xy=(w / 2, h + 0.05),
        xytext=(0.0, h + 1.0),
        fontsize=9.5,
        color=ROLE_COLOURS["cls"],
        arrowprops={"arrowstyle": "->", "color": ROLE_COLOURS["cls"]},
    )
    ax.text(
        x - 0.2,
        h + 0.55,
        f"{num_patches} patch embeddings, all kept",
        ha="right",
        fontsize=9.5,
        color=ROLE_COLOURS["patch"],
    )
    ax.set_xlim(-0.3, x + 0.1)
    ax.set_ylim(-0.9, h + 1.6)
    blank(ax)
    finish(
        fig,
        f"Sequence length = 1 CLS + {num_patches} patches = {num_patches + 1}",
        caption="Each position also gets a learned positional embedding (pos k).",
    )
    return fig


def plot_shape_flow(shapes: Sequence[tuple[str, str]] = DEFAULT_SHAPES) -> Figure:
    """Table-like diagram of tensor shapes through the ViT pipeline."""
    fig, ax = new_axes(9, 0.6 + 0.62 * len(shapes))
    for i, (stage, shape) in enumerate(shapes):
        y = -i * 1.0
        is_cls = "CLS" in stage
        draw_box(
            ax,
            0,
            y,
            9.5,
            0.72,
            stage,
            face=PALETTE["panel"],
            edge=ROLE_COLOURS["cls"] if is_cls else PALETTE["edge"],
            fontsize=10.5,
            mono=False,
        )
        draw_box(
            ax,
            10.0,
            y,
            4.6,
            0.72,
            shape,
            face=PALETTE["token_a"],
            edge=ROLE_COLOURS["patch"],
            fontsize=10.5,
        )
        if i < len(shapes) - 1:
            ax.annotate(
                "",
                xy=(4.75, y - 0.27),
                xytext=(4.75, y + 0.0),
                arrowprops={"arrowstyle": "->", "color": PALETTE["muted"]},
            )
    ax.set_xlim(-0.2, 14.8)
    ax.set_ylim(-(len(shapes) - 1) - 0.4, 1.0)
    blank(ax)
    finish(fig, "Tensor shapes through the miniature ViT")
    return fig


def plot_vit_architecture() -> Figure:
    """Simplified ViT pipeline from image to classification head."""
    fig, ax = new_axes(10, 6.2)
    stages = [
        ("Image\n[B, 3, 224, 224]", PALETTE["muted"]),
        (
            "Patch projection\nConv2d 16×16, stride 16\n→ [B, 196, D]",
            ROLE_COLOURS["patch"],
        ),
        ("Prepend [CLS]\n→ [B, 197, D]", ROLE_COLOURS["cls"]),
        ("Add positional embeddings\n→ [B, 197, D]", ROLE_COLOURS["position"]),
        ("Transformer encoder × L\n(self-attention + MLP)", PALETTE["green"]),
        ("Final LayerNorm", PALETTE["muted"]),
        ("Take position 0 (CLS)\n→ [B, D]", ROLE_COLOURS["cls"]),
        ("Linear head\n→ logits [B, classes]", PALETTE["pink"]),
    ]
    # Two columns of four, snaking down then up for compactness.
    coords = [(0, 3 - i) for i in range(4)] + [(1, i) for i in range(4)]
    w, h = 5.6, 1.0
    centres = []
    for (label, colour), (col, row) in zip(stages, coords, strict=True):
        x, y = col * 7.4, row * 1.45
        is_cls = "CLS" in label
        draw_box(
            ax,
            x,
            y,
            w,
            h,
            label,
            face=PALETTE["panel"],
            edge=colour,
            linewidth=2.2 if is_cls else 1.4,
            fontsize=10,
            mono=False,
            hatch="//" if is_cls else None,
        )
        centres.append((x, y))
    for (x1, y1), (x2, y2) in zip(centres, centres[1:], strict=False):
        if x1 == x2:
            start = (x1 + w / 2, y1 + (0 if y2 < y1 else h))
            end = (x2 + w / 2, y2 + (h if y2 < y1 else 0))
        else:
            start, end = (x1 + w, y1 + h / 2), (x2, y2 + h / 2)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": PALETTE["text"], "lw": 1.4},
        )
    ax.text(
        6.5,
        -0.65,
        "CLS takes part in self-attention in every layer; that is how it gathers\n"
        "information from the patches. Some ViTs average patch tokens instead.",
        ha="center",
        va="top",
        fontsize=9.5,
        color=PALETTE["muted"],
    )
    ax.set_xlim(-0.3, 13.3)
    ax.set_ylim(-1.6, 5.6)
    blank(ax)
    finish(fig, "A Vision Transformer classifier, simplified")
    return fig
