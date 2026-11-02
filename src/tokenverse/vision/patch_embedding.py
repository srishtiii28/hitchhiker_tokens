"""Turn images into patch sequences and patch embeddings with a [CLS] token.

Pipeline implemented by :class:`PatchEmbeddingWithCLS`::

    images [B, C, H, W]
      -> Conv2d(kernel=stride=P)      [B, D, H/P, W/P]
      -> flatten + transpose          [B, N, D]       patch embeddings
      -> prepend expanded CLS         [B, N+1, D]
      -> add positional embeddings    [B, N+1, D]

The [CLS] token is one learned vector, shared by every image, placed in front
of the patch tokens. All N patches stay in the sequence.
"""

from __future__ import annotations

import torch
from torch import nn


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Cut images into non-overlapping square patches and flatten each patch.

    Patches are ordered row-major: left to right along the top row of
    patches, then the next row down, and so on. Within a patch, values are
    flattened channel-first (all of channel 0, then channel 1, ...), each
    channel row by row, which matches how ``Conv2d`` weights are laid out.

    Args:
        images: Tensor of shape ``[batch, channels, height, width]``.
        patch_size: Side length ``P`` of each square patch.

    Returns:
        Tensor of shape ``[batch, (H/P) * (W/P), channels * P * P]``.
        For ``[2, 3, 224, 224]`` and ``P=16`` that is ``[2, 196, 768]``.

    Raises:
        ValueError: For non-4D input, non-positive ``patch_size``, or image
            sides not divisible by ``patch_size``.
    """
    if images.dim() != 4:
        raise ValueError(
            "patchify expects [batch, channels, height, width], got shape "
            f"{list(images.shape)}"
        )
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    b, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError(
            f"Image size {h}x{w} is not divisible by patch_size={patch_size}"
        )
    p = patch_size
    # [B, C, H/P, W/P, P, P]
    patches = images.unfold(2, p, p).unfold(3, p, p)
    # [B, H/P, W/P, C, P, P] -> [B, N, C*P*P]
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches.reshape(b, (h // p) * (w // p), c * p * p)


class PatchEmbeddingWithCLS(nn.Module):
    """Patch projection, learnable [CLS] token and learnable positions.

    The ``Conv2d`` with ``kernel_size = stride = patch_size`` applies the same
    linear map to every non-overlapping patch. It is equivalent to
    ``patchify(images) @ W.T + b`` with ``W`` the conv weight reshaped to
    ``[embed_dim, C * P * P]``.

    Args:
        image_size: Height and width of the (square) input image.
        patch_size: Side of each square patch.
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Embedding dimension ``D``.

    Attributes:
        cls_token: Learnable parameter of shape ``[1, 1, D]``.
        pos_embedding: Learnable parameter of shape ``[1, N + 1, D]``.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or image_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if image_size % patch_size:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}"
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2

        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    @property
    def sequence_length(self) -> int:
        """Number of patches plus one CLS token."""
        return self.num_patches + 1

    def _check_input(self, images: torch.Tensor) -> None:
        if images.dim() != 4:
            raise ValueError(
                f"Expected [batch, channels, height, width], got {list(images.shape)}"
            )
        _, c, h, w = images.shape
        if c != self.in_channels or h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Expected images of shape [B, {self.in_channels}, "
                f"{self.image_size}, {self.image_size}], got {list(images.shape)}"
            )

    def embed_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Project patches: ``[B, C, H, W] -> [B, N, D]``."""
        self._check_input(images)
        x = self.projection(images)  # [B, D, H/P, W/P]
        return x.flatten(2).transpose(1, 2)  # [B, N, D]

    def forward(
        self, images: torch.Tensor, *, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Build the position-enriched token sequence.

        Args:
            images: ``[B, C, image_size, image_size]``.
            return_intermediates: If ``True``, return a dict with
                ``patch_embeddings`` ``[B, N, D]``, ``cls_token_expanded``
                ``[B, 1, D]``, ``sequence_with_cls`` ``[B, N+1, D]`` and
                ``sequence_with_positions`` ``[B, N+1, D]``.

        Returns:
            ``sequence_with_positions`` or the dict described above.
        """
        patch_embeddings = self.embed_patches(images)
        batch = patch_embeddings.shape[0]
        # One shared parameter, viewed (not copied) once per example.
        cls_expanded = self.cls_token.expand(batch, -1, -1)
        sequence_with_cls = torch.cat([cls_expanded, patch_embeddings], dim=1)
        sequence_with_positions = sequence_with_cls + self.pos_embedding
        if not return_intermediates:
            return sequence_with_positions
        return {
            "patch_embeddings": patch_embeddings,
            "cls_token_expanded": cls_expanded,
            "sequence_with_cls": sequence_with_cls,
            "sequence_with_positions": sequence_with_positions,
        }
