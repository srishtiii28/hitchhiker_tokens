from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tokenverse.vision.mini_vit import MiniViT, ViTConfig  # noqa: E402

SMALL = ViTConfig(
    image_size=32,
    patch_size=8,
    embed_dim=64,
    depth=2,
    num_heads=4,
    mlp_dim=128,
    num_classes=3,
)

FEATURE_KEYS = {
    "patch_embeddings",
    "cls_token_expanded",
    "sequence_with_cls",
    "sequence_with_positions",
    "encoded_sequence",
    "cls_representation",
}


def test_default_forward_shapes():
    torch.manual_seed(0)
    model = MiniViT().eval()
    with torch.no_grad():
        x = torch.rand(2, 3, 224, 224)
        feats = model.forward_features(x)
        logits = model(x)
    assert set(feats) == FEATURE_KEYS
    assert feats["patch_embeddings"].shape == (2, 196, 192)
    assert feats["cls_token_expanded"].shape == (2, 1, 192)
    assert feats["sequence_with_cls"].shape == (2, 197, 192)
    assert feats["sequence_with_positions"].shape == (2, 197, 192)
    assert feats["encoded_sequence"].shape == (2, 197, 192)
    assert feats["cls_representation"].shape == (2, 192)
    assert logits.shape == (2, 10)


def test_head_reads_cls_position():
    torch.manual_seed(0)
    model = MiniViT(SMALL).eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        feats = model.forward_features(x)
        assert torch.equal(feats["cls_representation"], feats["encoded_sequence"][:, 0])
        assert torch.allclose(model(x), model.head(feats["encoded_sequence"][:, 0]))


def test_gradient_reaches_cls_token():
    torch.manual_seed(0)
    model = MiniViT(SMALL)
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 0])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grad = model.cls_token.grad
    assert grad is not None
    assert grad.shape == (1, 1, 64)
    assert grad.abs().sum() > 0


def test_cls_changes_through_attention():
    # Before the encoder the CLS slot is identical for every image; after it differs.
    torch.manual_seed(0)
    model = MiniViT(SMALL).eval()
    with torch.no_grad():
        feats = model.forward_features(torch.rand(2, 3, 32, 32))
    before = feats["sequence_with_cls"][:, 0]
    assert torch.equal(before[0], before[1])
    assert not torch.allclose(
        feats["cls_representation"][0], feats["cls_representation"][1]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dim": 190, "num_heads": 3},
        {"image_size": 30, "patch_size": 16},
        {"depth": 0},
        {"num_classes": 0},
        {"dropout": 1.5},
    ],
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        ViTConfig(**kwargs)


def test_config_num_patches():
    assert ViTConfig().num_patches == 196
    assert SMALL.num_patches == 16
