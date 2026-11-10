"""V-JEPA teacher with the same interface as DinoVideoTeacher.

`get_future_feature(video, ...)` takes a clip [B, 3, T, H, W] that is already resized and
ImageNet-normalized (module_utils.get_video_target does that) and returns patch tokens of the
current and future frames, each [B, tokens_per_frame, 1024].

Two backends:
  hf_vjepa2  transformers.VJEPA2Model (V-JEPA 2, 256 px, in the training image).
  vjepa2_1   the official facebookresearch/vjepa2 code (V-JEPA 2.1, trained at 384 px), loaded
             from a local snapshot because that environment has no outbound git access.

T is measured in frames and must be a multiple of the tubelet size (2). The current frame is
frame index `current_index`; the future frame is the last frame.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

TUBELET = 2
PATCH = 16
EMBED_DIM = 1024


class VJepaTeacher(nn.Module):
    def __init__(self, encoder: nn.Module, backend: str, embed_dim: int = EMBED_DIM,
                 interpolate_pos_encoding: bool = False):
        super().__init__()
        if backend not in ("hf_vjepa2", "vjepa2_1"):
            raise ValueError(f"unknown backend {backend!r}")
        self.encoder = encoder
        self.backend = backend
        self.embed_dim = embed_dim
        # V-JEPA 2.1 was trained at 384 and is run here at 256, so its position encoding must adapt.
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_hf(cls, model_dir: str | Path, dtype: torch.dtype = torch.bfloat16) -> "VJepaTeacher":
        from transformers import VJEPA2Model

        model = VJEPA2Model.from_pretrained(str(model_dir), torch_dtype=dtype)
        return cls(model, backend="hf_vjepa2")

    @classmethod
    def from_vjepa2_1(cls, source_dir: str | Path, weights: str | Path, *,
                      img_size: int = 384) -> "VJepaTeacher":
        """Build the official ViT-L encoder and load the `ema_encoder` weights.

        `source_dir` is the unpacked facebookresearch/vjepa2 snapshot; its hub loader points at
        localhost in the pinned revision, so weights are loaded from `weights` instead.
        """
        import sys

        sys.path.insert(0, str(source_dir))
        from app.vjepa_2_1.models.vision_transformer import vit_large

        encoder = vit_large(
            patch_size=PATCH, img_size=(img_size, img_size), num_frames=64, tubelet_size=TUBELET,
            use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
            img_temporal_dim_size=1, interpolate_rope=True)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        encoder.load_state_dict(_strip_prefix(state), strict=True)
        return cls(encoder, backend="vjepa2_1", interpolate_pos_encoding=True)

    @torch.no_grad()
    def get_future_feature(self, video: torch.Tensor, return_cls: bool = False,
                           return_current: bool = False, current_index: int = 0,
                           fps: float | torch.Tensor | None = None,
                           hidden_layer: int | None = None):
        """Mirror of DinoVideoTeacher.get_future_feature. `fps` is accepted and ignored: V-JEPA
        encodes time by tubelet index, not by seconds."""
        del fps
        if return_cls:
            raise NotImplementedError("V-JEPA has no class token; the probe uses patch tokens only.")
        if video.ndim != 5:
            raise ValueError(f"expected video [B, 3, T, H, W], got {tuple(video.shape)}")
        batch, channels, frames, height, width = video.shape
        if channels != 3 or frames < 2 or frames % TUBELET or height % PATCH or width % PATCH:
            raise ValueError(f"video must be [B, 3, k*{TUBELET}, k*{PATCH}, k*{PATCH}], got {tuple(video.shape)}")
        tokens = self._tokens(video, hidden_layer)
        per_frame = tokens.shape[1] // (frames // TUBELET)
        current = tokens[:, (current_index // TUBELET) * per_frame:(current_index // TUBELET + 1) * per_frame]
        future = tokens[:, -per_frame:]
        if return_current:
            return future, current
        return future

    def _tokens(self, video: torch.Tensor, hidden_layer: int | None) -> torch.Tensor:
        if self.backend == "hf_vjepa2":
            outputs = self.encoder(pixel_values_videos=video.permute(0, 2, 1, 3, 4).contiguous(),
                                   skip_predictor=True, output_hidden_states=hidden_layer is not None)
            if hidden_layer is None:
                return outputs.last_hidden_state
            return outputs.hidden_states[hidden_layer]
        outputs = self.encoder(video, interpolate_pos_encoding=self.interpolate_pos_encoding)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return outputs


def _strip_prefix(state: dict) -> dict:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


class ToyEncoder(nn.Module):
    """Stand-in encoder for the shape contract: one conv patch embed, no weights file needed."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed = nn.Conv3d(3, embed_dim, kernel_size=(TUBELET, PATCH, PATCH),
                               stride=(TUBELET, PATCH, PATCH))

    def forward(self, video: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        del interpolate_pos_encoding
        tokens = self.embed(video)  # [B, dim, T/2, H/16, W/16]
        return tokens.flatten(2).transpose(1, 2).contiguous()
