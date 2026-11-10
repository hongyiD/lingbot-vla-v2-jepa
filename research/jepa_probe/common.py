"""Shared constants and small helpers for the Stage 1 offline probes.

Nothing here imports torch, so the index builder runs in a plain CPU environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# LeRobot v3.0 on-disk layout (lerobot 0.4.2).
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TOP_CAMERA = "observation.images.cam_high"  # LingBot's `camera_top`
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIM = 14
ACTION_DIM = 14

# Matches configs/vla/robotwin/robotwin.yaml: arm/effector use bounds_99_woclip.
NORM_TYPE = "bounds_99_woclip"
NORM_KEYS = ("action.arm.position", "action.effector.position",
             "observation.state.arm.position", "observation.state.effector.position")
# Layout of the 14-dim vectors: left arm (6), left gripper (1), right arm (6), right gripper (1).
# The robot config concatenates the arms into a 12-dim `arm.position` feature and the grippers into
# a 2-dim `effector.position` feature, and the norm stats follow that concatenated order.
ARM_SLICES = (slice(0, 6), slice(7, 13))
GRIPPER_SLICES = (slice(6, 7), slice(13, 14))
# A gripper counts as moving inside a chunk when its normalized value spans more than this.
# Fully open to fully closed is about 2 in normalized units.
GRIPPER_EVENT_SPAN = 0.5

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT = "train=0-34,val=35-39,test=40-49"

IMAGE_SIZE = (224, 224)          # LingBot's dataset image_transforms=Resize(image_size)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# V-JEPA patch geometry. Duplicated in the teacher module, which cannot import this package
# without pulling in lingbotvla's torch-only model registry.
TUBELET = 2
PATCH = 16
EMBED_DIM = 1024


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_split(spec: str) -> dict[str, range]:
    """'train=0-34,val=35-39,test=40-49' -> {name: range of episode indices, end inclusive}."""
    out: dict[str, range] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, bounds = item.split("=", 1)
        start_text, end_text = bounds.split("-", 1)
        start, end = int(start_text), int(end_text)
        if name not in SPLIT_NAMES or end < start:
            raise ValueError(f"bad split entry {item!r}; names must be {SPLIT_NAMES}")
        out[name] = range(start, end + 1)
    missing = [name for name in SPLIT_NAMES if name not in out]
    if missing:
        raise ValueError(f"split spec is missing {missing}")
    return out


def split_of(episode_index: int, splits: dict[str, range]) -> str | None:
    for name, episode_range in splits.items():
        if episode_index in episode_range:
            return name
    return None


def normalize_bounds(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """bounds_99_woclip: map q01/q99 to [-1, 1] without clipping (transform.py)."""
    return (value - low) / (high - low + 1e-6) * 2.0 - 1.0


def load_norm(norm_stats_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    stats = read_json(norm_stats_path)["norm_stats"]
    missing = [key for key in NORM_KEYS if key not in stats]
    if missing:
        raise KeyError(f"norm stats missing keys {missing}")
    return {key: {name: np.asarray(values, dtype=np.float64) for name, values in entry.items()}
            for key, entry in stats.items()}


def normalize_vector(raw: np.ndarray, kind: str, norm: dict) -> np.ndarray:
    """Normalize one 14-dim state or action vector the way LingBot does.

    `kind` is 'action' or 'observation.state'. The output keeps the raw 14-dim layout; each
    dimension uses the stat at its position in the concatenated feature (left part first).
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (STATE_DIM,):
        raise ValueError(f"expected a {STATE_DIM}-dim {kind} vector, got {raw.shape}")
    out = np.empty_like(raw)
    for part, slices in (("arm.position", ARM_SLICES), ("effector.position", GRIPPER_SLICES)):
        entry = norm[f"{kind}.{part}"]
        low, high = entry["q01"], entry["q99"]
        needed = sum(sl.stop - sl.start for sl in slices)
        if low.ndim != 1 or low.shape[0] != needed:
            raise ValueError(f"norm stats for {kind}.{part} have shape {low.shape}, expected ({needed},)")
        offset = 0
        for sl in slices:
            width = sl.stop - sl.start
            out[sl] = normalize_bounds(raw[sl], low[offset:offset + width], high[offset:offset + width])
            offset += width
    return out.astype(np.float32)


def gripper_event(normalized_chunk: np.ndarray, span: float = GRIPPER_EVENT_SPAN) -> bool:
    """True when either gripper moves by more than `span` (normalized units) inside the chunk.

    A threshold rather than exact equality: gripper commands are continuous and jitter slightly.
    """
    for sl in GRIPPER_SLICES:
        column = normalized_chunk[:, sl]
        if float(column.max() - column.min()) > span:
            return True
    return False
