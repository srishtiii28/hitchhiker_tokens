"""Image loading and conversion helpers.

Normalisation convention: tensors hold ``float32`` values in ``[0, 1]``
(pixel / 255), channel-first. There's no ImageNet mean/std normalisation,
since none of the models here are pretrained.

Shapes:

- :func:`pil_to_tensor` returns ``[C, H, W]``.
- :func:`load_image_tensor` returns ``[C, H, W]``; add a batch dimension with
  ``x.unsqueeze(0)`` to get ``[1, C, H, W]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_image(path: str | Path, size: int | None = None) -> Image.Image:
    """Open an image as RGB and optionally resize it to ``size x size``.

    Resizing ignores aspect ratio, which is fine for the patch demos.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a ``[3, H, W]`` float tensor in ``[0, 1]``."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_image_tensor(path: str | Path, size: int = 224) -> torch.Tensor:
    """Load an image file as a ``[3, size, size]`` tensor in ``[0, 1]``."""
    return pil_to_tensor(load_image(path, size))


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``[3, H, W]`` (or ``[1, 3, H, W]``) tensor in ``[0, 1]`` to PIL."""
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError("tensor_to_pil expects a single image, not a batch")
        tensor = tensor[0]
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected shape [3, H, W], got {list(tensor.shape)}")
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round()
    return Image.fromarray(array.astype(np.uint8))


def make_sample_image(size: int = 224) -> Image.Image:
    """Draw a simple synthetic scene (sky, sun, hills, house) for demos.

    Distinct regions make the patch grid easy to read. No file is needed.
    """
    image = Image.new("RGB", (size, size), (40, 70, 120))
    draw = ImageDraw.Draw(image)
    s = size / 224
    # Sky gradient.
    for y in range(int(130 * s)):
        shade = int(70 + 80 * y / (130 * s))
        draw.line([(0, y), (size, y)], fill=(40, shade, 160))
    draw.ellipse([160 * s, 20 * s, 205 * s, 65 * s], fill=(250, 210, 80))
    draw.ellipse([-60 * s, 110 * s, 150 * s, 300 * s], fill=(60, 140, 80))
    draw.ellipse([90 * s, 125 * s, 300 * s, 300 * s], fill=(45, 115, 65))
    draw.rectangle([40 * s, 120 * s, 100 * s, 175 * s], fill=(200, 170, 140))
    draw.polygon(
        [(32 * s, 122 * s), (70 * s, 88 * s), (108 * s, 122 * s)], fill=(170, 60, 60)
    )
    draw.rectangle([62 * s, 145 * s, 78 * s, 175 * s], fill=(90, 60, 40))
    return image


def select_device(prefer: str | None = None) -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU.

    Args:
        prefer: Force a device name such as ``"cpu"``.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
