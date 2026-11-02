from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tokenverse.vision.patch_embedding import PatchEmbeddingWithCLS, patchify  # noqa: E402


def test_patchify_shape_default():
    x = torch.zeros(2, 3, 224, 224)
    assert patchify(x, 16).shape == (2, 196, 768)


def test_patchify_rectangular():
    x = torch.zeros(1, 1, 8, 12)
    assert patchify(x, 4).shape == (1, 6, 16)


def test_patch_order_is_row_major():
    # 1 channel, 4x4 image, 2x2 patches. Fill each patch with its expected index.
    image = torch.zeros(1, 1, 4, 4)
    image[0, 0, :2, :2] = 0
    image[0, 0, :2, 2:] = 1
    image[0, 0, 2:, :2] = 2
    image[0, 0, 2:, 2:] = 3
    patches = patchify(image, 2)
    assert patches[0, :, 0].tolist() == [0, 1, 2, 3]
    assert torch.all(patches[0] == patches[0, :, :1])


def test_within_patch_values_are_row_major():
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    patches = patchify(image, 2)
    assert patches[0, 0].tolist() == [0, 1, 4, 5]
    assert patches[0, 1].tolist() == [2, 3, 6, 7]
    assert patches[0, 2].tolist() == [8, 9, 12, 13]


def test_channels_flatten_channel_first():
    image = torch.stack([torch.full((2, 2), float(c)) for c in range(3)]).unsqueeze(0)
    assert patchify(image, 2)[0, 0].tolist() == [0] * 4 + [1] * 4 + [2] * 4


@pytest.mark.parametrize("shape", [(3, 224, 224), (224, 224), (1, 1, 3, 224, 224)])
def test_patchify_rejects_wrong_rank(shape):
    with pytest.raises(ValueError, match="batch, channels"):
        patchify(torch.zeros(shape), 16)


def test_patchify_rejects_indivisible():
    with pytest.raises(ValueError, match="divisible"):
        patchify(torch.zeros(1, 3, 30, 32), 16)


def test_patchify_rejects_bad_patch_size():
    with pytest.raises(ValueError, match="positive"):
        patchify(torch.zeros(1, 3, 32, 32), 0)


def test_patch_embedding_shapes():
    embed = PatchEmbeddingWithCLS(224, 16, 3, 192)
    out = embed(torch.rand(2, 3, 224, 224), return_intermediates=True)
    assert out["patch_embeddings"].shape == (2, 196, 192)
    assert out["cls_token_expanded"].shape == (2, 1, 192)
    assert out["sequence_with_cls"].shape == (2, 197, 192)
    assert out["sequence_with_positions"].shape == (2, 197, 192)
    assert embed.sequence_length == 197


def test_cls_parameter_is_learnable_and_shared():
    embed = PatchEmbeddingWithCLS(32, 8, 3, 64)
    assert isinstance(embed.cls_token, torch.nn.Parameter)
    assert embed.cls_token.shape == (1, 1, 64)
    assert embed.cls_token.requires_grad
    assert embed.pos_embedding.shape == (1, 17, 64)


@pytest.mark.parametrize("batch", [1, 3, 5])
def test_cls_expansion_for_batch_sizes(batch):
    embed = PatchEmbeddingWithCLS(32, 8, 3, 64)
    out = embed(torch.rand(batch, 3, 32, 32), return_intermediates=True)
    cls = out["cls_token_expanded"]
    assert cls.shape == (batch, 1, 64)
    for b in range(batch):
        assert torch.equal(cls[b], embed.cls_token[0])
    # CLS is at position 0, before the patches, before positions are added.
    assert torch.equal(out["sequence_with_cls"][:, 0], cls[:, 0])
    expected = out["sequence_with_cls"] + embed.pos_embedding
    assert torch.allclose(out["sequence_with_positions"], expected)


def test_conv_projection_equals_linear_on_patches():
    torch.manual_seed(0)
    embed = PatchEmbeddingWithCLS(32, 8, 3, 16)
    x = torch.rand(2, 3, 32, 32)
    weight = embed.projection.weight.reshape(16, -1)
    linear = patchify(x, 8) @ weight.T + embed.projection.bias
    assert torch.allclose(linear, embed.embed_patches(x), atol=1e-5)


def test_patch_embedding_rejects_wrong_size():
    embed = PatchEmbeddingWithCLS(32, 8, 3, 16)
    with pytest.raises(ValueError, match="Expected images"):
        embed(torch.rand(1, 3, 64, 64))


def test_patch_embedding_rejects_bad_config():
    with pytest.raises(ValueError, match="divisible"):
        PatchEmbeddingWithCLS(30, 8)
