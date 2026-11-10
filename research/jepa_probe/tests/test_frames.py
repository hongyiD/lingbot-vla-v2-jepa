"""frames.resize_224 follows the training pixel path, and parquet reads round-trip."""
import numpy as np
import pytest
from PIL import Image

from research.jepa_probe.frames import FrameReader
from research.jepa_probe.tests.helpers import write_dataset


def _dataset(tmp_path):
    return write_dataset(tmp_path, "click_bell-aloha-agilex_clean_50-50", episodes=[4, 5], image_size=(32, 24))


def test_frame_roundtrip_from_parquet(tmp_path):
    reader = FrameReader(_dataset(tmp_path))
    assert reader.num_rows == 9
    first, later, again = reader.images([0, 6, 0])  # row 6 is episode 1 frame 2, in the second file
    assert np.asarray(first).shape == (24, 32, 3)
    assert not np.array_equal(np.asarray(first), np.asarray(later))
    assert np.array_equal(np.asarray(first), np.asarray(again))
    # Fixture color is ((episode * 40) % 256, (frame * 20) % 256, 128).
    assert tuple(np.asarray(later)[0, 0]) == (40, 40, 128)
    with pytest.raises(IndexError):
        reader.images([9])


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_resize_matches_training_path():
    """lerobot ToTensor -> v2.Resize on the float tensor -> pad_and_concat's uint8 conversion."""
    from torchvision import transforms
    from torchvision.transforms import v2

    from research.jepa_probe.frames import resize_224

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8), "RGB")
    expected = v2.Resize((224, 224))(transforms.ToTensor()(image))
    expected = (expected * 255).round().clamp(0, 255).to(torch.uint8)
    got = resize_224(image)
    assert got.shape == (3, 224, 224) and got.dtype == torch.uint8
    assert torch.equal(got, expected)


def test_resize_is_identity_at_224():
    from research.jepa_probe.frames import resize_224

    image = Image.new("RGB", (224, 224), (10, 20, 30))
    assert torch.equal(resize_224(image)[:, 0, 0], torch.tensor([10, 20, 30], dtype=torch.uint8))


def test_teacher_input_normalization():
    from research.jepa_probe.frames import to_teacher_input

    frames = torch.zeros(2, 3, 224, 224, dtype=torch.uint8)
    frames[:] = torch.tensor([255, 0, 0]).view(3, 1, 1)
    out = to_teacher_input(frames, input_size=256)
    assert out.shape == (2, 3, 256, 256)
    expected_red = (1.0 - 0.485) / 0.229
    assert torch.allclose(out[:, 0], torch.full((256, 256), expected_red), atol=1e-4)
