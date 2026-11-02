"""A deterministic synthetic dataset of circles, squares and triangles.

Labels: ``0 = circle``, ``1 = square``, ``2 = triangle``.

Each sample is generated from ``base_seed`` and its index alone, so sample
``i`` is the same image regardless of iteration order, shuffling or the number
of DataLoader workers. Size, position, foreground colour and background shade
vary per sample.
"""

from __future__ import annotations

import random

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from tokenverse.vision.image_utils import pil_to_tensor

CLASS_NAMES: tuple[str, ...] = ("circle", "square", "triangle")


def draw_shape(label: int, image_size: int, rng: random.Random) -> Image.Image:
    """Draw one shape of class ``label`` using randomness from ``rng``."""
    if label not in range(len(CLASS_NAMES)):
        raise ValueError(f"label must be 0, 1 or 2, got {label}")
    bg = rng.randint(0, 90)
    image = Image.new("RGB", (image_size, image_size), (bg, bg, bg))
    draw = ImageDraw.Draw(image)

    size = rng.uniform(0.35, 0.7) * image_size
    cx = rng.uniform(size / 2, image_size - size / 2)
    cy = rng.uniform(size / 2, image_size - size / 2)
    colour = (rng.randint(130, 255), rng.randint(130, 255), rng.randint(130, 255))
    half = size / 2
    box = [cx - half, cy - half, cx + half, cy + half]
    if label == 0:
        draw.ellipse(box, fill=colour)
    elif label == 1:
        draw.rectangle(box, fill=colour)
    else:
        draw.polygon(
            [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)],
            fill=colour,
        )
    return image


class SyntheticShapes(Dataset):
    """Generated shape images as ``(tensor [3, S, S] in [0, 1], label)`` pairs.

    Args:
        num_samples: Dataset length.
        image_size: Side length of each square image.
        base_seed: Seed combined with the index to generate each sample. Use
            different seeds for train and validation splits.
    """

    def __init__(
        self, num_samples: int = 600, image_size: int = 32, base_seed: int = 0
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if image_size < 8:
            raise ValueError("image_size must be at least 8")
        self.num_samples = num_samples
        self.image_size = image_size
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.base_seed * 1_000_003 + index)

    def label_for(self, index: int) -> int:
        """Label of sample ``index`` (classes cycle, so splits stay balanced)."""
        return index % len(CLASS_NAMES)

    def image(self, index: int) -> Image.Image:
        """The PIL image for sample ``index``."""
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        return draw_shape(self.label_for(index), self.image_size, self._rng(index))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return pil_to_tensor(self.image(index)), self.label_for(index)
