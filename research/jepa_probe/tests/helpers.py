"""A minimal fake LeRobot v3.0 dataset for the probe tests (no lerobot, no torch)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from research.jepa_probe.common import ACTION_DIM, ACTION_KEY, STATE_DIM, STATE_KEY, TOP_CAMERA

CHUNK_ROWS = 6  # small on purpose, so rows span more than one parquet file


def _png(color: tuple[int, int, int], size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def write_dataset(root: Path, name: str, *, episodes: list[int], fps: float = 50.0,
                  image_size: tuple[int, int] = (32, 24)) -> Path:
    """One dataset directory. `episodes` gives the length of each episode.

    State and action are deterministic functions of (episode, frame), so a test can recompute them.
    Images are solid colors that also depend on (episode, frame).
    """
    dataset = root / name
    rows: dict[str, list] = {"episode_index": [], "frame_index": [], STATE_KEY: [], ACTION_KEY: [], TOP_CAMERA: []}
    episode_rows = []
    cursor = 0
    for episode_index, length in enumerate(episodes):
        for frame in range(length):
            rows["episode_index"].append(episode_index)
            rows["frame_index"].append(frame)
            rows[STATE_KEY].append(_vector(episode_index, frame, scale=0.01))
            rows[ACTION_KEY].append(_vector(episode_index, frame, scale=0.02))
            color = ((episode_index * 40) % 256, (frame * 20) % 256, 128)
            rows[TOP_CAMERA].append({"bytes": _png(color, image_size), "path": None})
        episode_rows.append({"episode_index": episode_index, "dataset_from_index": cursor,
                             "dataset_to_index": cursor + length})
        cursor += length

    _write_chunks(dataset / "data", rows, include_images=True)
    _write_chunks(dataset / "meta" / "episodes", episode_rows, include_images=False)
    info = {"codebase_version": "v3.0", "total_episodes": len(episodes), "total_frames": cursor,
            "fps": fps, "features": {TOP_CAMERA: {"dtype": "image", "shape": [3, image_size[1], image_size[0]]}}}
    (dataset / "meta").mkdir(parents=True, exist_ok=True)
    (dataset / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return dataset


def _vector(episode_index: int, frame: int, scale: float) -> list[float]:
    return [scale * (episode_index + 1) * (frame + 1 + dim) for dim in range(STATE_DIM)]


def _write_chunks(directory: Path, rows, include_images: bool) -> None:
    n = len(rows) if isinstance(rows, list) else len(rows["episode_index"])
    for file_index, start in enumerate(range(0, n, CHUNK_ROWS)):
        chunk = directory / "chunk-000"
        chunk.mkdir(parents=True, exist_ok=True)
        end = min(start + CHUNK_ROWS, n)
        if isinstance(rows, list):
            table = pa.Table.from_pylist(rows[start:end])
        else:
            batch = {key: value[start:end] for key, value in rows.items()}
            arrays = {
                "episode_index": pa.array(batch["episode_index"], pa.int64()),
                "frame_index": pa.array(batch["frame_index"], pa.int64()),
            }
            if include_images:
                arrays[STATE_KEY] = pa.array(batch[STATE_KEY], pa.list_(pa.float32(), STATE_DIM))
                arrays[ACTION_KEY] = pa.array(batch[ACTION_KEY], pa.list_(pa.float32(), ACTION_DIM))
                image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
                arrays[TOP_CAMERA] = pa.array(batch[TOP_CAMERA], image_type)
            table = pa.table(arrays)
        pq.write_table(table, chunk / f"file-{file_index:03d}.parquet")


LEFT_BOUNDS = (0.0, 1.0)
RIGHT_BOUNDS = (-1.0, 3.0)


def write_norm_stats(path: Path) -> None:
    """Layout of the real file: arm stats are 12-dim (left 6, right 6), gripper stats 2-dim.

    Left and right use different bounds so a mix-up between the two shows up in the labels.
    """
    (l_low, l_high), (r_low, r_high) = LEFT_BOUNDS, RIGHT_BOUNDS
    entry = {"mean": [0] * 12, "std": [1] * 12, "q01": [l_low] * 6 + [r_low] * 6, "q99": [l_high] * 6 + [r_high] * 6}
    gripper = {"mean": [0, 0], "std": [1, 1], "q01": [l_low, r_low], "q99": [l_high, r_high]}
    stats = {}
    for kind in ("action", "observation.state"):
        stats[f"{kind}.arm.position"] = entry
        stats[f"{kind}.effector.position"] = gripper
    path.write_text(json.dumps({"norm_stats": stats}), encoding="utf-8")


def write_robot_config(path: Path) -> None:
    path.write_text("images:\n- observation.images.camera_top: {}\n", encoding="utf-8")
