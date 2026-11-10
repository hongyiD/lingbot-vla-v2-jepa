"""frames.resize_224 matches torchvision v2 Resize, and parquet reads round-trip."""
import io

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from research.jepa_probe.frames import frame_image, resize_224, to_teacher_input
from research.jepa_probe.tests.helpers import write_dataset

torchvision = pytest.importorskip("torchvision")
from torchvision.transforms import v2  # noqa: E402


def test_resize_matches_torchvision():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8), "RGB")
    expected = v2.Resize((224, 224))(image)
    got = resize_224(image)
    assert got.shape == (3, 224, 224) and got.dtype == torch.uint8
    assert torch.equal(got, v2.ToImage()(expected))


def test_resize_is_identity_at_224():
    image = Image.new("RGB", (224, 224), (10, 20, 30))
    got = resize_224(image)
    assert torch.equal(got[:, 0, 0], torch.tensor([10, 20, 30], dtype=torch.uint8))


def test_frame_roundtrip_from_parquet(tmp_path):
    dataset = write_dataset(tmp_path, "click_bell-aloha-agilex_clean_50-50", episodes=[4, 5],
                            image_size=(32, 24))
    first = np.asarray(frame_image(dataset, 0))
    later = np.asarray(frame_image(dataset, 6))  # episode 1 frame 2, past the first parquet file
    assert first.shape == (24, 32, 3)
    assert not np.array_equal(first, later)


def test_teacher_input_normalization():
    frames = torch.zeros(2, 3, 224, 224, dtype=torch.uint8)
    frames[:] = torch.tensor([255, 0, 0]).view(3, 1, 1)
    out = to_teacher_input(frames, input_size=256)
    assert out.shape == (2, 3, 256, 256)
    expected_red = (1.0 - 0.485) / 0.229
    assert torch.allclose(out[:, 0], torch.full((256, 256), expected_red), atol=1e-4)
