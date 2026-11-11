"""Teacher runners for the probe variants (plan section 2.2).

A runner takes `frames` uint8 [B, P, 3, H, W] (training pixels, frames.IMAGE_SIZE), the frames at `runner.positions` (offsets from
t, as produced by frames.FrameReader.load), and `fps` [B], the effective future-video fps. It
returns (current, future) tokens, each [B, grid * grid, C] float32 on a `runner.grid` square grid.

  runner = build_runner(load_variant(cfg, "V1"), device)
  current, future = runner(frames, fps)
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from research.jepa_probe.frames import to_teacher_input


def add_extra_paths(variant: dict) -> None:
    """Append the variant's `extra_paths` to sys.path, after the image's own packages."""
    for path in variant.get("extra_paths", []):
        if path not in sys.path:
            sys.path.append(path)


def frame_positions(variant: dict) -> tuple[int, ...]:
    return tuple(sorted(set(variant["clip_frames"]) | set(variant.get("future_clip_frames", []))))


class Runner:
    grid = 16

    def __init__(self, variant: dict, device: torch.device):
        self.variant = variant
        self.device = torch.device(device)
        self.positions = frame_positions(variant)
        self.bf16 = variant.get("dtype", "bf16") == "bf16" and self.device.type == "cuda"
        self._call = self._forward

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor, fps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[1] != len(self.positions):
            raise ValueError(f"expected frames [B, {len(self.positions)}, 3, H, W], got {tuple(frames.shape)}")
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            current, future = self._call(frames, fps)
        return current.float(), future.float()

    def compile(self) -> None:
        self._call = torch.compile(self._forward)

    def _forward(self, frames, fps):
        raise NotImplementedError

    def _column(self, position: int) -> int:
        return self.positions.index(position)


class DinoVideoRunner(Runner):
    """V0: the training target, through module_utils.get_video_target with the training video config.

    `fps` is not passed on. FeatureTransform.convert_features drops the dataset's
    future_video_effective_fps, so training calls get_video_target without it and the teacher
    uses `effective_fps` from the video config. check_pipeline.py verifies that against the
    real data path.
    """

    def __init__(self, variant: dict, device: torch.device):
        super().__init__(variant, device)
        from lingbotvla.models.vla.vision_models.module_utils import build_video_model

        clip = variant["clip_frames"]
        if len(set(clip)) != 2:
            raise ValueError(f"DINO-Video takes one current and one future frame, got clip {clip}")
        self.video_cfg = dict(variant["video"], device=str(self.device))
        self.teacher = build_video_model(self.video_cfg)
        self.current, self.future = self._column(min(clip)), self._column(max(clip))

    def _forward(self, frames, fps):
        from lingbotvla.models.vla.vision_models.module_utils import get_video_target

        del fps
        bundle = get_video_target(self.teacher, frames[:, self.current:self.current + 1],
                                  frames[:, self.future:self.future + 1], self.video_cfg)
        return bundle["current_patch"], bundle["patch"]


class VJepaRunner(Runner):
    """V1, V1i, V1m, V2, V3, V3h: one clip with current and future tubelets, or one clip each."""

    def __init__(self, variant: dict, device: torch.device, teacher):
        super().__init__(variant, device)
        self.teacher = teacher.to(self.device)
        self.input_size = int(variant["input_size"])
        layer = variant.get("layer", "last")
        self.hidden_layer = None if layer == "last" else int(layer)
        self.current_index = int(variant.get("current_index", 0))
        self.resample_to = variant.get("resample_to")

    def _video(self, normalized: torch.Tensor, clip: list[int]) -> torch.Tensor:
        columns = [self._column(position) for position in clip]
        return normalized[:, columns].permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H, W]

    def _forward(self, frames, fps):
        del fps
        batch, positions = frames.shape[:2]
        size = self.input_size
        normalized = to_teacher_input(frames.flatten(0, 1), size).view(batch, positions, 3, size, size)
        clip = self.variant["clip_frames"]
        future_clip = self.variant.get("future_clip_frames")
        if future_clip is None:
            future, current = self.teacher.get_future_feature(
                self._video(normalized, clip), return_current=True, current_index=self.current_index,
                hidden_layer=self.hidden_layer)
        else:
            # Separate forwards: the current-only clip's last tubelet is the current frame.
            current = self.teacher.get_future_feature(self._video(normalized, clip), hidden_layer=self.hidden_layer)
            future = self.teacher.get_future_feature(self._video(normalized, future_clip),
                                                     hidden_layer=self.hidden_layer)
        return self._resample(current), self._resample(future)

    def _resample(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.resample_to:
            return tokens
        batch, count, channels = tokens.shape
        side = int(round(count ** 0.5))
        if side * side != count:
            raise ValueError(f"cannot resample {count} tokens: not a square grid")
        grid = tokens.transpose(1, 2).reshape(batch, channels, side, side)
        grid = F.adaptive_avg_pool2d(grid.float(), int(self.resample_to))  # area interpolation
        return grid.flatten(2).transpose(1, 2)


class PixelRunner(Runner):
    """B-pixel: each frame shrunk to 16x16 RGB, grouped into 8x8 tokens of 2x2x3 = 12 values."""

    grid = 8

    def __init__(self, variant: dict, device: torch.device):
        super().__init__(variant, device)
        clip = variant["clip_frames"]
        self.current, self.future = self._column(min(clip)), self._column(max(clip))

    def _forward(self, frames, fps):
        del fps
        return self._tokens(frames[:, self.current]), self._tokens(frames[:, self.future])

    @staticmethod
    def _tokens(frame: torch.Tensor) -> torch.Tensor:
        small = F.interpolate(frame.float() / 255.0, size=(16, 16), mode="area")  # [B, 3, 16, 16]
        return F.pixel_unshuffle(small, 2).flatten(2).transpose(1, 2)        # [B, 64, 12]


def build_runner(variant: dict, device: torch.device | str) -> Runner:
    kind = variant["teacher"]
    if kind == "dino_video":
        return DinoVideoRunner(variant, device)
    if kind == "pixel":
        return PixelRunner(variant, device)

    from lingbotvla.models.vla.vision_models.jepa.teacher import ToyEncoder, VJepaTeacher

    if kind == "hf_vjepa2":
        teacher = VJepaTeacher.from_hf(variant["weights"], dtype=torch.float32)
    elif kind == "vjepa2_1":
        add_extra_paths(variant)
        teacher = VJepaTeacher.from_vjepa2_1(variant["source"], variant["weights"],
                                             img_size=int(variant.get("img_size", 384)))
    elif kind == "toy":  # tests only
        torch.manual_seed(0)
        teacher = VJepaTeacher(ToyEncoder(int(variant.get("embed_dim", 16))), backend="vjepa2_1")
    else:
        raise ValueError(f"unknown teacher {kind!r}")
    return VJepaRunner(variant, device, teacher)
