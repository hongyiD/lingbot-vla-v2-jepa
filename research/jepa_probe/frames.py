"""Read top-camera frames and reproduce the pixels LingBot feeds its teachers.

Training stores the uint8 image after Resize((224, 224)) in `pil_images`
(lingbotvla/data/vla_data/transform.py, prepare_images with use_depth_align=True). The teacher then
resizes that to its own input size with bilinear interpolation (module_utils.get_video_target).
Both steps are reproduced here so probe features match training targets.

  frames = load_frames(dataset_dir, [row_t, row_future])   # uint8, [N, 3, 224, 224]
  teacher_input = to_teacher_input(frames, input_size=256) # float, ImageNet-normalized
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from research.jepa_probe.common import IMAGENET_MEAN, IMAGENET_STD, TOP_CAMERA

IMAGE_SIZE = (224, 224)


def frame_image(dataset_dir: Path, dataset_row: int, camera: str = TOP_CAMERA) -> Image.Image:
    """The PIL image `datasets` would decode for one row, read straight from the parquet file.

    LeRobot stores image columns as the `datasets` Image struct: {"bytes": <png bytes>, "path": ...}.
    """
    import io

    files = sorted((Path(dataset_dir) / "data").glob("chunk-*/*.parquet"))
    seen = 0
    for path in files:
        table = pq.read_table(path, columns=[camera])
        if seen + table.num_rows <= dataset_row:
            seen += table.num_rows
            continue
        cell = table.column(camera)[dataset_row - seen].as_py()
        payload = cell["bytes"] if isinstance(cell, dict) else cell
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(f"{camera} row {dataset_row}: expected image bytes, got {type(payload).__name__}")
        return Image.open(io.BytesIO(payload)).convert("RGB")
    raise IndexError(f"row {dataset_row} outside {dataset_dir} ({seen} rows)")


def resize_224(image: Image.Image):
    """torchvision v2 Resize((224, 224)) on a uint8 image: stretch, bilinear, antialias.

    Returns uint8 [3, 224, 224], which is what ends up in `pil_images`.
    """
    import torch
    import torch.nn.functional as F

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if tuple(tensor.shape[-2:]) == IMAGE_SIZE:
        return tensor
    resized = F.interpolate(tensor.unsqueeze(0).float(), size=IMAGE_SIZE, mode="bilinear",
                            align_corners=False, antialias=True)
    return resized.round().clamp(0, 255).to(torch.uint8).squeeze(0)


def load_frames(dataset_dir: Path, dataset_rows: list[int], camera: str = TOP_CAMERA) -> torch.Tensor:
    """uint8 frames at 224, shape [N, 3, 224, 224], in the order of `dataset_rows`."""
    return torch.stack([resize_224(frame_image(dataset_dir, row, camera)) for row in dataset_rows])


def to_teacher_input(frames, input_size: int):
    """uint8 [N, 3, 224, 224] -> ImageNet-normalized float [N, 3, input_size, input_size].

    Mirrors get_video_target: float conversion, bilinear resize (no antialias), then normalization.
    """
    import torch
    import torch.nn.functional as F

    values = frames.float() / 255.0
    if values.shape[-2:] != (input_size, input_size):
        values = F.interpolate(values, size=(input_size, input_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=values.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=values.dtype).view(1, 3, 1, 1)
    return (values - mean) / std
