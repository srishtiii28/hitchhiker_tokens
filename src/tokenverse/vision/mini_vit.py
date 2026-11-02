"""A small Vision Transformer classifier that reads its answer from [CLS].

Architecture::

    images -> PatchEmbeddingWithCLS -> TransformerEncoder -> LayerNorm
           -> take position 0 (the CLS token) -> Linear head -> logits

The CLS token starts as the same learned vector for every image. It carries
no information about a particular image until self-attention runs: in each
encoder layer, the CLS position attends to the patch tokens (and they to it),
so its representation can pick up information from the image. Training
pushes it to collect what the classification head needs.

Using CLS is one design choice. The original ViT used a class token; many
later models instead average the patch tokens (global average pooling).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tokenverse.vision.patch_embedding import PatchEmbeddingWithCLS


@dataclass(frozen=True)
class ViTConfig:
    """Configuration for :class:`MiniViT`.

    The defaults give the 224 x 224, patch 16, dim 192 model used in the
    shape walkthrough (196 patches + 1 CLS = 197 tokens).
    """

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 3
    mlp_dim: int = 768
    num_classes: int = 10
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "image_size",
            "patch_size",
            "in_channels",
            "embed_dim",
            "depth",
            "num_heads",
            "mlp_dim",
            "num_classes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by "
                f"patch_size={self.patch_size}"
            )
        if self.embed_dim % self.num_heads:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by "
                f"num_heads={self.num_heads}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def num_patches(self) -> int:
        """Number of patches per image."""
        return (self.image_size // self.patch_size) ** 2


class MiniViT(nn.Module):
    """Miniature ViT: conv patch projection, CLS, positions, encoder, head."""

    def __init__(self, config: ViTConfig | None = None) -> None:
        super().__init__()
        self.config = config or ViTConfig()
        cfg = self.config
        self.patch_embed = PatchEmbeddingWithCLS(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.mlp_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm, as in ViT
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

    @property
    def cls_token(self) -> nn.Parameter:
        """The learnable CLS parameter, shape ``[1, 1, embed_dim]``."""
        return self.patch_embed.cls_token

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run everything except the classification head.

        Returns a dict with:

        - ``patch_embeddings`` ``[B, N, D]``: projected patches.
        - ``cls_token_expanded`` ``[B, 1, D]``: the shared CLS vector per example.
        - ``sequence_with_cls`` ``[B, N+1, D]``: CLS followed by all patches.
        - ``sequence_with_positions`` ``[B, N+1, D]``: plus positional embeddings.
        - ``encoded_sequence`` ``[B, N+1, D]``: encoder output after the final
          LayerNorm.
        - ``cls_representation`` ``[B, D]``: ``encoded_sequence[:, 0, :]``.
        """
        features = self.patch_embed(images, return_intermediates=True)
        encoded = self.encoder(features["sequence_with_positions"])
        encoded = self.norm(encoded)
        features["encoded_sequence"] = encoded
        features["cls_representation"] = encoded[:, 0, :]
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``[B, num_classes]``."""
        return self.head(self.forward_features(images)["cls_representation"])
