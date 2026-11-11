"""Build the Stage 1 sample index and labels from the converted clean_50 datasets.

Reads only the non-image columns (the same idea as research/tools/pod/agx_run.py): one row per
sample with the current frame, the frame `future_offset` steps later, the 8 frame rows a V2 clip
needs, and labels normalized exactly like LingBot training (bounds_99_woclip, no clipping).

  python -m research.jepa_probe.build_index \
      --train_list $D/lists/train_c50.txt --robot_config $CFG/robot_configs/robotwin_c50.yaml \
      --norm_stats $CFG/norm_stats/robotwin_c50.json --out $D/jepa/probe/index
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from research.jepa_probe.common import (
    ACTION_DIM,
    ACTION_KEY,
    DEFAULT_SPLIT,
    STATE_DIM,
    STATE_KEY,
    gripper_event,
    load_norm,
    normalize_vector,
    parse_split,
    read_json,
    split_of,
)


def dataset_dirs(train_list: Path) -> list[Path]:
    """Dataset directories of a LingBot train list: lines are '<robot config name> <dataset dir>'."""
    dirs = [Path(line.split()[-1]) for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [str(d) for d in dirs if not (d / "meta" / "info.json").is_file()]
    if missing:
        raise FileNotFoundError("train list entries without meta/info.json:\n  " + "\n  ".join(missing))
    return dirs


def read_episodes(dataset_dir: Path) -> list[dict]:
    files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {dataset_dir / 'meta' / 'episodes'}")
    rows: list[dict] = []
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "dataset_from_index", "dataset_to_index"])
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: row["episode_index"])
    found = [int(row["episode_index"]) for row in rows]
    if found != list(range(len(rows))):
        raise ValueError(f"{dataset_dir.name}: episode indices {found[:5]}... are not 0..{len(rows) - 1}")
    return rows


def read_columns(dataset_dir: Path) -> dict[str, np.ndarray]:
    """Non-image columns of every data file, in file order. That is the dataset row order."""
    files = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data files under {dataset_dir / 'data'}")
    table = pa.concat_tables(
        [pq.read_table(path, columns=["episode_index", "frame_index", STATE_KEY, ACTION_KEY]) for path in files])
    columns = {
        "episode_index": np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64),
        "frame_index": np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64),
    }
    for name, width in ((STATE_KEY, STATE_DIM), (ACTION_KEY, ACTION_DIM)):
        columns[name] = np.stack(table.column(name).to_pylist()).astype(np.float64).reshape(-1, width)
    return columns


def clip_rows(rows: np.ndarray, t: int, future_offset: int) -> list[int]:
    """8 frame rows: f_t twice, five frames spread over (t, t+offset), then f_{t+offset}.

    Doubling f_t keeps the first tubelet (2 frames) free of future frames, matching the V2 clip.
    """
    positions = [t, t] + [t + int(round(k * future_offset / 6)) for k in range(1, 6)] + [t + future_offset]
    return [int(rows[position]) for position in positions]


def task_name(dataset_dir: Path) -> str:
    """'click_bell-aloha-agilex_clean_50-50' and 'click_bell_joint_v30' both give 'click_bell'."""
    return dataset_dir.name.split("-")[0].split("_joint")[0]


def index_dataset(dataset_dir: Path, *, stride: int, future_offset: int, chunk: int,
                  splits: dict, norm: dict) -> tuple[list[dict], dict[str, list]]:
    info = read_json(dataset_dir / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{dataset_dir.name}: expected codebase_version v3.0, got {info.get('codebase_version')!r}")
    fps = float(info["fps"])
    columns = read_columns(dataset_dir)
    n_rows = len(columns["episode_index"])

    samples: list[dict] = []
    labels = {"state_t": [], "state_t49": [], "action_chunk": []}
    for episode in read_episodes(dataset_dir):
        episode_index = int(episode["episode_index"])
        split = split_of(episode_index, splits)
        if split is None:
            continue
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        if not 0 <= start < end <= n_rows:
            raise ValueError(f"{dataset_dir.name} ep {episode_index}: row range {start}:{end} outside 0..{n_rows}")
        rows = np.arange(start, end)
        if not np.all(columns["episode_index"][rows] == episode_index):
            raise ValueError(f"{dataset_dir.name} ep {episode_index}: episode_index column mismatches metadata")
        if not np.array_equal(columns["frame_index"][rows], np.arange(end - start)):
            raise ValueError(f"{dataset_dir.name} ep {episode_index}: frame_index is not 0..length-1")
        length = end - start
        horizon = max(future_offset + 1, chunk)
        for t in range(0, length - horizon + 1, stride):
            row_t = int(rows[t])
            action_raw = columns[ACTION_KEY][row_t:row_t + chunk]
            action = np.stack([normalize_vector(step, "action", norm) for step in action_raw])
            samples.append({
                "task": task_name(dataset_dir),
                "dataset_dir": str(dataset_dir),
                "episode_index": episode_index,
                "t": t,
                "row_t": row_t,
                "row_future": int(rows[t + future_offset]),
                "rows_v2": json.dumps(clip_rows(rows, t, future_offset)),
                "split": split,
                "fps": fps,
                "gripper_event": bool(gripper_event(action)),
            })
            labels["state_t"].append(normalize_vector(columns[STATE_KEY][row_t], "observation.state", norm))
            labels["state_t49"].append(normalize_vector(columns[STATE_KEY][rows[t + future_offset]],
                                                        "observation.state", norm))
            labels["action_chunk"].append(action)
    return samples, labels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(train_list: Path, robot_config: Path, norm_stats: Path, out_dir: Path, *,
          stride: int, future_offset: int, chunk: int, split_spec: str) -> dict:
    splits = parse_split(split_spec)
    norm = load_norm(norm_stats)

    samples: list[dict] = []
    label_rows: list[dict[str, list]] = []
    per_dataset = []
    for dataset_dir in dataset_dirs(train_list):
        found, labels = index_dataset(dataset_dir, stride=stride, future_offset=future_offset,
                                      chunk=chunk, splits=splits, norm=norm)
        samples.extend(found)
        if found:
            label_rows.append(labels)
        per_dataset.append({"dataset": dataset_dir.name, "fps": found[0]["fps"] if found else None,
                            "samples": len(found)})

    tasks = sorted({sample["task"] for sample in samples})
    task_ids = {task: i for i, task in enumerate(tasks)}
    order = sorted(range(len(samples)),
                   key=lambda i: (samples[i]["task"], samples[i]["episode_index"], samples[i]["t"]))
    samples = [samples[i] for i in order]
    for sample_id, sample in enumerate(samples):
        sample["sample_id"] = sample_id
        sample["task_id"] = task_ids[sample["task"]]

    merged = {name: np.stack([row for part in label_rows for row in part[name]])[order] for name in label_rows[0]}
    merged["task_id"] = np.array([sample["task_id"] for sample in samples], dtype=np.int64)

    meta = {"stride": stride, "future_offset": future_offset, "chunk": chunk, "split": split_spec,
            "datasets": per_dataset, "samples": len(samples), "tasks": tasks,
            "sha256": {name: sha256_file(path) for name, path in
                       (("train_list", train_list), ("robot_config", robot_config), ("norm_stats", norm_stats))}}
    write_output(samples, merged, meta, out_dir)
    return meta


def write_output(samples: list[dict], labels: dict[str, np.ndarray], meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(samples), out_dir / "index.parquet")
    np.savez(out_dir / "labels.npz", task_names=np.asarray(meta["tasks"]), **labels)
    (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_list", type=Path, required=True)
    parser.add_argument("--robot_config", type=Path, required=True)
    parser.add_argument("--norm_stats", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--future_offset", type=int, default=49)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    meta = build(args.train_list, args.robot_config, args.norm_stats, args.out, stride=args.stride,
                 future_offset=args.future_offset, chunk=args.chunk, split_spec=args.split)
    counts = {}
    table = pq.read_table(args.out / "index.parquet")
    for name in ("train", "val", "test"):
        counts[name] = table.column("split").to_pylist().count(name)
    events = sum(table.column("gripper_event").to_pylist())
    print(f"{meta['samples']} samples from {len(meta['datasets'])} datasets -> {args.out}")
    print(f"splits {counts}, gripper events {events} ({events / max(meta['samples'], 1):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
