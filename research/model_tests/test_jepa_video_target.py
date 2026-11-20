"""A2 (plan/03): the V-JEPA 2 teacher in the training target path, with the toy encoder instead of weights."""
import pytest
import torch

import lingbotvla.models.vla.vision_models.module_utils as module_utils
from lingbotvla.data.vla_data.transform import prepare_images
from lingbotvla.data.vla_data.utils import FeatureTransform
from lingbotvla.models.vla.vision_models.jepa.teacher import ToyEncoder, VJepaTeacher
from lingbotvla.models.vla.vision_models.module_utils import (
    build_video_model,
    check_video_teacher_config,
    get_video_target,
)
from research.jepa_probe.teachers import build_runner

MIDS = [8, 16, 24, 33, 41]
V2_CLIP = [0, 0, *MIDS, 49]


def jepa_cfg(**extra):
    cfg = {"teacher_type": "vjepa2", "jepa_model_path": "/weights/vjepa2", "input_size": 64, "use_warmup_frame": False,
           "use_patch_loss": True, "use_current_patch_loss": True, "use_cls_loss": False}
    return {**cfg, **extra}


class RecordingTeacher:
    def __init__(self):
        self.calls = []

    def get_future_feature(self, video, **kwargs):
        self.calls.append((video, kwargs))
        tokens = torch.zeros(video.shape[0], 16, 8)
        return tokens, tokens


def frames(batch, count, size=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, count, 3, size, size), dtype=torch.uint8, generator=generator)


def normalized(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images.float() / 255.0 - mean) / std


def test_vjepa2_clip_is_current_twice_then_mid_then_future():
    pil, future, mid = frames(2, 3), frames(2, 1, seed=1), frames(2, len(MIDS), seed=2)
    teacher = RecordingTeacher()
    get_video_target(teacher, pil, future, jepa_cfg(), effective_fps=torch.ones(2), mid_pil_images=mid)
    (video, kwargs), = teacher.calls
    assert video.shape == (2, 3, 8, 64, 64)
    expected = [pil[:, 0], pil[:, 0], *mid.unbind(1), future[:, 0]]
    for index, frame in enumerate(expected):
        torch.testing.assert_close(video[:, :, index], normalized(frame), rtol=0, atol=1e-6)
    assert kwargs.get("current_index", 0) == 0 and kwargs["return_current"] is True


def test_vjepa2_without_mid_frames_pads_to_whole_tubelets():
    pil, future = frames(1, 3), frames(1, 1, seed=1)
    teacher = RecordingTeacher()
    get_video_target(teacher, pil, future, jepa_cfg())
    video = teacher.calls[0][0]
    assert video.shape[2] == 4  # [cur, cur, fut, fut], the Stage 1 V1 clip
    assert torch.equal(video[:, :, 2], video[:, :, 3]) and torch.equal(video[:, :, 0], video[:, :, 1])


@pytest.mark.parametrize("size", [64, 48])  # 48 px goes through the bilinear resize in both paths
def test_training_target_matches_the_stage1_v2_runner(size):
    runner = build_runner({"teacher": "toy", "embed_dim": 32, "input_size": 64, "clip_frames": V2_CLIP,
                           "current_index": 0, "layer": "last"}, "cpu")
    clip = frames(2, len(runner.positions), size=size, seed=3)  # positions (0, 8, ..., 41, 49)
    probe_current, probe_future = runner(clip, torch.ones(2))
    bundle = get_video_target(runner.teacher, clip[:, :1], clip[:, -1:], jepa_cfg(), mid_pil_images=clip[:, 1:-1])
    # The probe keeps float32; training returns bf16, as it does for DINO-Video.
    assert torch.equal(bundle["current_patch"], probe_current.to(torch.bfloat16))
    assert torch.equal(bundle["patch"], probe_future.to(torch.bfloat16))


def test_dino_path_takes_no_mid_frames():
    pil, future = frames(1, 3), frames(1, 1, seed=1)
    teacher = RecordingTeacher()
    dino = {"use_warmup_frame": True, "use_current_patch_loss": True, "input_size": 64}
    get_video_target(teacher, pil, future, dino)
    assert teacher.calls[0][0].shape[2] == 3 and teacher.calls[0][1]["current_index"] == 1
    with pytest.raises(ValueError, match="mid_pil_images"):
        get_video_target(teacher, pil, future, dino, mid_pil_images=frames(1, 5))


def test_teacher_config_checks():
    check_video_teacher_config({"use_warmup_frame": True})  # the official DINO-Video section
    check_video_teacher_config(jepa_cfg(), MIDS)
    bad = [({"use_warmup_frame": True}, MIDS, "future_video_mid_frames"),
           (jepa_cfg(jepa_model_path=None), MIDS, "jepa_model_path"),
           (jepa_cfg(use_warmup_frame=True), MIDS, "use_warmup_frame"),
           (jepa_cfg(use_cls_loss=True), MIDS, "use_cls_loss"),
           (jepa_cfg(num_future_frames=2), MIDS, "num_future_frames"),
           ({"teacher_type": "vjepa3"}, [], "teacher_type")]
    for cfg, mids, message in bad:
        with pytest.raises(ValueError, match=message):
            check_video_teacher_config(cfg, mids)


def test_build_video_model_loads_vjepa2_in_fp32(monkeypatch):
    calls = []

    def from_hf(path, dtype):
        calls.append((path, dtype))
        return VJepaTeacher(ToyEncoder(8), backend="hf_vjepa2")

    monkeypatch.setattr(VJepaTeacher, "from_hf", staticmethod(from_hf))
    teacher = build_video_model({"video": jepa_cfg(device="cpu")})
    assert calls == [("/weights/vjepa2", torch.float32)]
    assert isinstance(teacher, VJepaTeacher)
    assert not any(p.requires_grad for p in teacher.parameters())


def test_mid_frames_share_the_sample_augmentation():
    key = "observation.images.camera_top"
    current = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    torch.manual_seed(0)
    _, _, pil, _, params = prepare_images(None, {"state": torch.zeros(1), "image": {key: current}}, image_keys=[key],
                                          train=True, use_depth_align=True, return_augment_params=True)
    assert params is not None
    mids = FeatureTransform.augment_future_video_mid_images(current.unsqueeze(0).repeat(2, 1, 1, 1), params)
    assert mids.dtype == torch.uint8 and all(torch.equal(m, pil[0]) for m in mids)
    raw = current.unsqueeze(0)
    assert FeatureTransform.augment_future_video_mid_images(raw, None) is raw  # no augmentation: frames as decoded
