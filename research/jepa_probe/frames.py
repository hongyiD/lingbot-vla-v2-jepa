"""Read top-camera frames and reproduce the pixels LingBot feeds its teachers.

Training path for parquet-embedded PNG cameras: lerobot decodes to float [0, 1] (ToTensor), the
dataset applies torchvision v2 Resize((224, 224)) to that float tensor, and pad_and_concat turns it
back into uint8 with (x * 255).round().clamp(0, 255). The teacher then resizes that uint8 image to
its own input size with bilinear interpolation (module_utils.get_video_target). Both steps are
reproduced here, in the same order and with the same ops, so probe features match training targets.

  reader = FrameReader(dataset_dir)
  frames = reader.load([row_t, row_future])                # uint8, [N, 3, 224, 224]
  teacher_input = to_teacher_input(frames, input_size=256) # float, ImageNet-normalized
"""
from __future__ import annotations

import bisect
import io
from collections import OrderedDict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from research.jepa_probe.common import IMAGENET_MEAN, IMAGENET_STD, TOP_CAMERA

IMAGE_SIZE = (224, 224)


class FrameReader:
    """Random access to one camera column of a LeRobot v3.0 dataset.

    Only the parquet row groups that hold requested rows are read, so a batch costs a few row
    groups instead of whole image columns. LeRobot stores image columns as the `datasets` Image
    struct: {"bytes": <png bytes>, "path": ...}.
    """

    def __init__(self, dataset_dir: Path, camera: str = TOP_CAMERA, cache_groups: int = 2):
        self.dataset_dir = Path(dataset_dir)
        self.camera = camera
        self._cache: OrderedDict[int, object] = OrderedDict()
        self._cache_groups = cache_groups
        files = sorted((self.dataset_dir / "data").glob("chunk-*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"no data files under {self.dataset_dir / 'data'}")
        self._groups: list[tuple[Path, int, int]] = []  # (file, row group, rows)
        self._starts: list[int] = []
        total = 0
        for path in files:
            metadata = pq.ParquetFile(path).metadata
            for group in range(metadata.num_row_groups):
                rows = metadata.row_group(group).num_rows
                self._groups.append((path, group, rows))
                self._starts.append(total)
                total += rows
        self.num_rows = total

    def raw(self, dataset_rows: list[int]) -> list[bytes]:
        """Encoded image bytes in the order of `dataset_rows`."""
        wanted: dict[int, list[tuple[int, int]]] = {}
        for position, row in enumerate(dataset_rows):
            if not 0 <= row < self.num_rows:
                raise IndexError(f"row {row} outside {self.dataset_dir} ({self.num_rows} rows)")
            group = bisect.bisect_right(self._starts, row) - 1
            wanted.setdefault(group, []).append((position, row - self._starts[group]))
        out: list[bytes | None] = [None] * len(dataset_rows)
        for group, requests in wanted.items():
            column = self._column(group)
            for position, offset in requests:
                out[position] = _payload(column[offset].as_py(), self.camera, dataset_rows[position])
        return out

    def images(self, dataset_rows: list[int]) -> list[Image.Image]:
        """PIL images in the order of `dataset_rows`."""
        return [decode_png(payload) for payload in self.raw(dataset_rows)]

    def _column(self, group: int):
        if group in self._cache:
            self._cache.move_to_end(group)
            return self._cache[group]
        path, group_in_file, _ = self._groups[group]
        column = pq.ParquetFile(path).read_row_group(group_in_file, columns=[self.camera]).column(0)
        self._cache[group] = column
        while len(self._cache) > self._cache_groups:
            self._cache.popitem(last=False)
        return column

    def load(self, dataset_rows: list[int]):
        """uint8 frames at 224, shape [N, 3, 224, 224], in the order of `dataset_rows`."""
        import torch

        return torch.stack([resize_224(image) for image in self.images(dataset_rows)])


def _payload(cell, camera: str, row: int) -> bytes:
    payload = cell["bytes"] if isinstance(cell, dict) else cell
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"{camera} row {row}: expected image bytes, got {type(payload).__name__}")
    return payload


def decode_png(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGB")


def resize_224(image: Image.Image):
    """PIL image -> the uint8 [3, 224, 224] tensor training puts in `pil_images`."""
    import torch
    from torchvision.transforms import functional as TF
    from torchvision.transforms import v2

    values = TF.to_tensor(image.convert("RGB"))
    values = v2.Resize(IMAGE_SIZE)(values)
    return (values * 255).round().clamp(0, 255).to(torch.uint8)


def to_teacher_input(frames, input_size: int):
    """uint8 [N, 3, 224, 224] -> ImageNet-normalized float [N, 3, input_size, input_size].

    Mirrors get_video_target: float conversion, bilinear resize (no antialias), then normalization.
    """
    import torch
    import torch.nn.functional as F

    values = frames.float() / 255.0
    if values.shape[-2:] != (input_size, input_size):
        values = F.interpolate(values, size=(input_size, input_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=values.dtype, device=values.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=values.dtype, device=values.device).view(1, 3, 1, 1)
    return (values - mean) / std
