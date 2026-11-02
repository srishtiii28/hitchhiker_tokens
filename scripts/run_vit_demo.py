"""Run one forward pass of the miniature ViT and print every tensor shape.

The model is randomly initialised, so the logits are just noise. The point is
to see the shape at each stage.

    python scripts/run_vit_demo.py
    python scripts/run_vit_demo.py --batch-size 4 --device cpu
"""

from __future__ import annotations

import argparse

import torch

from tokenverse.vision.image_utils import select_device
from tokenverse.vision.mini_vit import MiniViT, ViTConfig
from tokenverse.vision.patch_embedding import patchify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mini ViT shape walkthrough")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (auto)")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    config = ViTConfig()
    model = MiniViT(config).to(device).eval()
    images = torch.rand(
        args.batch_size, 3, config.image_size, config.image_size, device=device
    )

    with torch.no_grad():
        flat = patchify(images, config.patch_size)
        projected = model.patch_embed.projection(images)
        feats = model.forward_features(images)
        logits = model.head(feats["cls_representation"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}   parameters: {n_params:,}")
    print(
        f"Config: image {config.image_size}, patch {config.patch_size}, "
        f"dim {config.embed_dim}, depth {config.depth}, heads {config.num_heads}, "
        f"classes {config.num_classes}"
    )
    print()
    rows = [
        ("Input images", images),
        ("Flattened patches (raw pixels)", flat),
        ("Patch projection", projected),
        ("Patch embeddings", feats["patch_embeddings"]),
        ("Expanded CLS token", feats["cls_token_expanded"]),
        ("Sequence after CLS concatenation", feats["sequence_with_cls"]),
        ("Sequence after positional embeddings", feats["sequence_with_positions"]),
        ("Encoded sequence (after final LayerNorm)", feats["encoded_sequence"]),
        ("CLS representation  encoded[:, 0, :]", feats["cls_representation"]),
        ("Classifier logits", logits),
    ]
    for name, tensor in rows:
        print(f"  {name:<42} {list(tensor.shape)}")
    print()
    print(
        f"Sequence length = {config.num_patches} patches + 1 CLS = "
        f"{feats['sequence_with_cls'].shape[1]}"
    )
    print("The model is untrained, so the logits don't mean anything yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
