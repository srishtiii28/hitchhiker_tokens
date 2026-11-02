"""Vision-side tools: patches, patch embeddings with [CLS], a mini ViT, shapes data.

Requires PyTorch (``pip install -e ".[vision]"``).
"""

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "tokenverse.vision needs PyTorch. Install it with "
        '`pip install -e ".[vision]"` or follow https://pytorch.org/get-started/.'
    ) from exc

from tokenverse.vision.image_utils import (
    load_image,
    load_image_tensor,
    make_sample_image,
    pil_to_tensor,
    select_device,
    tensor_to_pil,
)
from tokenverse.vision.mini_vit import MiniViT, ViTConfig
from tokenverse.vision.patch_embedding import PatchEmbeddingWithCLS, patchify
from tokenverse.vision.synthetic_dataset import CLASS_NAMES, SyntheticShapes

__all__ = [
    "CLASS_NAMES",
    "MiniViT",
    "PatchEmbeddingWithCLS",
    "SyntheticShapes",
    "ViTConfig",
    "load_image",
    "load_image_tensor",
    "make_sample_image",
    "patchify",
    "pil_to_tensor",
    "select_device",
    "tensor_to_pil",
]
