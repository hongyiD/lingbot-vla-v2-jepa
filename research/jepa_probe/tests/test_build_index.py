"""Index splits, label normalization and the t + future_offset boundary."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from research.jepa_probe.build_index import build
from research.jepa_probe.common import gripper_event, normalize_vector
from research.jepa_probe.tests.helpers import (LEFT_BOUNDS, RIGHT_BOUNDS, write_dataset, write_norm_stats,
                                               write_robot_config)

EPISODES = [8, 7, 9, 6]  # episode 1 (length 7) cannot hold a chunk of 5 past offset 4
OFFSET = 4
CHUNK = 5
STRIDE = 3


def _setup(tmp_path: Path):
    data = tmp_path / "data"
    first = write_dataset(data, "click_bell-aloha-agilex_clean_50-50", episodes=EPISODES, fps=50)
    second = write_dataset(data, "turn_switch_joint_v30", episodes=EPISODES, fps=25)
    train_list = tmp_path / "train.txt"
    train_list.write_text(f"{first}\n{second}\n")
    norm_stats = tmp_path / "norm.json"
    write_norm_stats(norm_stats)
    robot_config = tmp_path / "robot.yaml"
    write_robot_config(robot_config)
    return train_list, robot_config, norm_stats


def _expected_frames(length: int) -> list[int]:
    return list(range(0, length - OFFSET, STRIDE))


def test_splits_and_boundary(tmp_path):
    train_list, robot_config, norm_stats = _setup(tmp_path)
    out = tmp_path / "index"
    meta = build(train_list, robot_config, norm_stats, out, stride=STRIDE, future_offset=OFFSET,
                 chunk=CHUNK, split_spec="train=0-1,val=2-2,test=3-3")

    table = pq.read_table(out / "index.parquet").to_pylist()
    # episode 1 has length 7: t = 0 only, because t = 3 would need frames through 3+4 = 7.
    short = [row for row in table if row["task"] == "click_bell" and row["episode_index"] == 1]
    assert [row["t"] for row in short] == [0]
    assert short[0]["split"] == "train"

    per_episode = {0: _expected_frames(8), 2: _expected_frames(9), 3: _expected_frames(6)}
    for episode, frames in per_episode.items():
        found = [row["t"] for row in table if row["task"] == "turn_switch" and row["episode_index"] == episode]
        assert found == frames

    splits = {name: sorted({row["episode_index"] for row in table if row["split"] == name})
              for name in ("train", "val", "test")}
    assert splits == {"train": [0, 1], "val": [2], "test": [3]}
    assert [row["sample_id"] for row in table] == list(range(len(table)))
    assert meta["datasets"][1]["fps"] == 25

    labels = np.load(out / "labels.npz")
    assert list(labels["task_names"]) == ["click_bell", "turn_switch"]
    assert len(labels["action_chunk"]) == len(table)


def test_labels_match_training_normalization(tmp_path):
    train_list, robot_config, norm_stats = _setup(tmp_path)
    out = tmp_path / "index"
    build(train_list, robot_config, norm_stats, out, stride=STRIDE, future_offset=OFFSET,
          chunk=CHUNK, split_spec="train=0-3,val=0-0,test=0-0")

    from research.jepa_probe.common import load_norm
    norm = load_norm(norm_stats)
    table = pq.read_table(out / "index.parquet").to_pylist()
    labels = np.load(out / "labels.npz")
    row = next(row for row in table if row["task"] == "click_bell" and row["episode_index"] == 2 and row["t"] == 3)

    def raw_vector(episode, frame, scale):
        return np.array([scale * (episode + 1) * (frame + 1 + dim) for dim in range(14)])

    state = normalize_vector(raw_vector(2, 3, 0.01), "observation.state", norm)
    future = normalize_vector(raw_vector(2, 3 + OFFSET, 0.01), "observation.state", norm)
    action = np.stack([normalize_vector(raw_vector(2, 3 + step, 0.02), "action", norm) for step in range(CHUNK)])
    assert np.allclose(labels["state_t"][row["sample_id"]], state, atol=1e-6)
    assert np.allclose(labels["state_t49"][row["sample_id"]], future, atol=1e-6)
    assert np.allclose(labels["action_chunk"][row["sample_id"]], action, atol=1e-6)
    # No clipping: a value far outside [q01, q99] stays outside [-1, 1].
    far = normalize_vector(np.full(14, 5.0), "observation.state", norm)
    assert far.max() > 1


def test_right_side_uses_right_stats(tmp_path):
    from research.jepa_probe.common import load_norm
    norm_stats = tmp_path / "norm.json"
    write_norm_stats(norm_stats)
    norm = load_norm(norm_stats)
    (l_low, l_high), (r_low, r_high) = LEFT_BOUNDS, RIGHT_BOUNDS
    raw = np.full(14, 1.0)
    out = normalize_vector(raw, "action", norm)
    left = (1.0 - l_low) / (l_high - l_low + 1e-6) * 2 - 1
    right = (1.0 - r_low) / (r_high - r_low + 1e-6) * 2 - 1
    assert np.allclose(out[0:7], left, atol=1e-5)    # left arm and left gripper
    assert np.allclose(out[7:14], right, atol=1e-5)  # right arm and right gripper


def test_gripper_event_threshold():
    chunk = np.zeros((50, 14), dtype=np.float32)
    chunk[:, 6] = 0.9 + 0.01 * np.sin(np.arange(50))  # left gripper jitters while held
    assert not gripper_event(chunk)
    chunk[25:, 13] = -1.0  # right gripper closes halfway through
    chunk[:25, 13] = 1.0
    assert gripper_event(chunk)


def test_gripper_event_column_and_v2_rows(tmp_path):
    train_list, robot_config, norm_stats = _setup(tmp_path)
    out = tmp_path / "index"
    build(train_list, robot_config, norm_stats, out, stride=STRIDE, future_offset=OFFSET,
          chunk=CHUNK, split_spec="train=0-3,val=0-0,test=0-0")
    table = pq.read_table(out / "index.parquet").to_pylist()
    assert all(isinstance(row["gripper_event"], bool) for row in table)
    import json
    row = table[0]
    rows = json.loads(row["rows_v2"])
    assert len(rows) == 8 and rows[0] == rows[1] == row["row_t"] and rows[-1] == row["row_future"]


def test_rejects_non_v3(tmp_path):
    import json
    train_list, robot_config, norm_stats = _setup(tmp_path)
    dataset = Path(train_list.read_text().splitlines()[0])
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v2.1"
    info_path.write_text(json.dumps(info))
    try:
        build(train_list, robot_config, norm_stats, tmp_path / "out", stride=STRIDE,
              future_offset=OFFSET, chunk=CHUNK, split_spec="train=0-3,val=0-0,test=0-0")
    except ValueError as exc:
        assert "v3.0" in str(exc)
    else:
        raise AssertionError("v2.1 dataset should be rejected")
