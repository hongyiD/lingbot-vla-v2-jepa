"""VJepaTeacher shape contract, exercised with the toy encoder so no weights are needed.

Skipped where torch cannot be imported. Importing the teacher module goes through
`lingbotvla.models`, which loads transformers, so this needs the training environment.
"""
import pytest

try:
    import torch
    from lingbotvla.models.vla.vision_models.jepa.teacher import ToyEncoder, VJepaTeacher
except Exception as exc:  # the dev machine's torch/numpy is broken; the pod image is fine
    pytest.skip(f"teacher tests need a working torch: {exc}", allow_module_level=True)

from research.jepa_probe.common import EMBED_DIM  # noqa: E402


def _teacher() -> VJepaTeacher:
    return VJepaTeacher(ToyEncoder(), backend="vjepa2_1")


def test_future_and_current_shapes():
    teacher = _teacher()
    video = torch.randn(2, 3, 4, 256, 256)
    future, current = teacher.get_future_feature(video, return_current=True, current_index=0)
    assert future.shape == (2, 256, EMBED_DIM)
    assert current.shape == (2, 256, EMBED_DIM)


def test_current_index_selects_the_frame():
    teacher = _teacher()
    torch.manual_seed(0)
    video = torch.randn(1, 3, 4, 64, 64)
    _, current = teacher.get_future_feature(video, return_current=True, current_index=2)
    tokens = teacher._tokens(video, hidden_layer=None)
    per_frame = tokens.shape[1] // 2
    assert torch.equal(current, tokens[:, per_frame:])


def test_rejects_bad_shapes():
    teacher = _teacher()
    for shape in ((2, 256, 256), (1, 3, 3, 256, 256), (1, 3, 2, 250, 250)):
        with pytest.raises(ValueError):
            teacher.get_future_feature(torch.randn(*shape))


def test_no_class_token():
    teacher = _teacher()
    with pytest.raises(NotImplementedError):
        teacher.get_future_feature(torch.randn(1, 3, 2, 64, 64), return_cls=True)
