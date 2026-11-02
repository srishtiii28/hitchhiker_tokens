from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tokenverse.vision.synthetic_dataset import CLASS_NAMES, SyntheticShapes  # noqa: E402


def test_labels_and_shapes():
    ds = SyntheticShapes(num_samples=9, image_size=32, base_seed=1)
    assert len(ds) == 9
    for i in range(9):
        x, y = ds[i]
        assert x.shape == (3, 32, 32)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y in range(len(CLASS_NAMES))
    assert [ds[i][1] for i in range(6)] == [0, 1, 2, 0, 1, 2]


def test_deterministic_regardless_of_order():
    a = SyntheticShapes(20, 32, base_seed=7)
    b = SyntheticShapes(20, 32, base_seed=7)
    for i in reversed(range(20)):
        assert torch.equal(a[i][0], b[i][0])
    assert torch.equal(a[5][0], a[5][0])


def test_seeds_change_images():
    a = SyntheticShapes(3, 32, base_seed=0)
    b = SyntheticShapes(3, 32, base_seed=1)
    assert not torch.equal(a[0][0], b[0][0])


def test_samples_vary():
    ds = SyntheticShapes(6, 32, base_seed=0)
    assert not torch.equal(ds[0][0], ds[3][0])  # both circles, different draws


def test_class_names():
    assert CLASS_NAMES == ("circle", "square", "triangle")


def test_invalid_arguments():
    with pytest.raises(ValueError):
        SyntheticShapes(0)
    with pytest.raises(IndexError):
        SyntheticShapes(3)[3]
