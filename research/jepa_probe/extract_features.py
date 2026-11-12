"""Extract pooled teacher tokens for every sample of one or more variants.

Outputs shards of fp16 tokens under <out>/shard_XXXXXXX.npz, each holding `cur` and `fut` arrays of
shape [shard, 64, C] (the teacher's 16x16 grid pooled to 8x8). Shards that contain visualization
samples also hold their full-grid tokens as `viz_ids`, `viz_cur`, `viz_fut`. Finished shards are
skipped, so a run can resume. `stats.npz` carries the per-channel train-split mean and std that the
probes normalize with, and `meta.json` the variant, checksums and timings.

Reading the frames dominates: parquet is read by whole row groups, so a shard reads about ten times
the frames it uses. With several variants, each shard's frames are read and decoded once and fed to
every teacher that still lacks that shard. `--part k/n` takes every n-th shard, so n processes (one
per GPU) split the reading; the process that writes the last shard of a variant writes its stats.

  python -m research.jepa_probe.extract_features --variant V1 \
      --variants_cfg research/jepa_probe/configs/variants.yaml \
      --index $P/index --out $P/features/V1 --batch 64 --num_workers 16 --viz 1000
  python -m research.jepa_probe.extract_features --variant V0,B-pixel,V1 --features $P/features --part 0/4 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from research.jepa_probe.common import load_index, read_json, split_ids

try:
    import torch
    _IterableDataset = torch.utils.data.IterableDataset
except ImportError:  # the stats and shard helpers are used without torch
    torch = None
    _IterableDataset = object

SHARD_SIZE = 2048
POOLED_GRID = 8


def _expand(value, missing: list[str]):
    if isinstance(value, dict):
        return {key: _expand(item, missing) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, missing) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        missing.extend(re.findall(r"\$\{?(\w+)", expanded))
        return expanded
    return value


def load_variant(config_path: Path, name: str) -> dict:
    """One variant with ${VAR} expanded and relative `weights` / `source` joined to weights_dir."""
    import yaml
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    variants = config["variants"]
    if name not in variants:
        raise KeyError(f"variant {name!r} not in {config_path}; have {sorted(variants)}")
    missing: list[str] = []
    weights_dir = _expand(str(config.get("weights_dir", "")), missing)
    variant = _expand(dict(variants[name]), missing)
    if missing:
        raise ValueError(f"{config_path}: unset environment variables {sorted(set(missing))}; source pod.env first")
    for key in ("weights", "source"):
        if key in variant and not os.path.isabs(variant[key]):
            variant[key] = str(Path(weights_dir) / variant[key])
    variant["name"] = name
    return variant


def pool_tokens(tokens, grid: int, pool: int):
    """[B, grid*grid, C] -> [B, (grid/pool)**2, C] by averaging non-overlapping patches."""
    if tokens.shape[1] != grid * grid:
        raise ValueError(f"expected {grid * grid} tokens, got {tokens.shape[1]}")
    if grid % pool:
        raise ValueError(f"grid {grid} is not divisible by pool {pool}")
    if pool == 1:
        return tokens
    batch, _, channels = tokens.shape
    grouped = tokens.reshape(batch, grid // pool, pool, grid // pool, pool, channels)
    return grouped.mean(dim=(2, 4)).reshape(batch, -1, channels)


def iter_shards(n_samples: int, shard_size: int | None = None):
    shard_size = shard_size or SHARD_SIZE
    for start in range(0, n_samples, shard_size):
        yield start, min(start + shard_size, n_samples)


def shard_path(out_dir: Path, start: int) -> Path:
    return out_dir / f"shard_{start:07d}.npz"


def save_shard(path: Path, cur: np.ndarray, fut: np.ndarray, **extra: np.ndarray) -> None:
    # Write under a name the shard glob does not match, then rename, so a killed run leaves no
    # half-written shard. np.savez keeps a name that already ends with ".npz".
    partial = path.with_name(f".partial_{path.name}")
    np.savez(partial, cur=cur, fut=fut, **extra)
    partial.replace(path)


def pending_samples(out_dir: Path, n_samples: int) -> list[int]:
    done = set()
    for start, end in iter_shards(n_samples):
        path = shard_path(out_dir, start)
        if path.is_file() and _shard_ok(path, end - start):
            done.update(range(start, end))
    return [i for i in range(n_samples) if i not in done]


def _shard_ok(path: Path, expected_rows: int) -> bool:
    try:
        with np.load(path) as data:
            return data["cur"].shape[0] == expected_rows and data["fut"].shape == data["cur"].shape
    except (OSError, ValueError, KeyError):
        return False


def compute_stats(out_dir: Path, train_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Per-channel mean and std over the train split, plus the token variance (for loss scaling)."""
    total = np.zeros(2, dtype=np.float64)
    sum_channel = None
    sum_sq_channel = None
    sum_sq_token = 0.0
    for path in sorted(out_dir.glob("shard_*.npz")):
        with np.load(path) as data:
            start = int(path.stem.split("_")[1])
            rows = np.arange(start, start + len(data["cur"]))
            keep = np.isin(rows, train_ids)
            if not keep.any():
                continue
            both = np.stack([data["cur"][keep], data["fut"][keep]]).astype(np.float64)  # [2, N, 64, C]
            if sum_channel is None:
                sum_channel = np.zeros((2, both.shape[-1]))
                sum_sq_channel = np.zeros_like(sum_channel)
            total += both.shape[1] * both.shape[2]
            sum_channel += both.sum(axis=(1, 2))
            sum_sq_channel += np.square(both).sum(axis=(1, 2))
            sum_sq_token += np.square(both).sum()
    if sum_channel is None:
        raise ValueError(f"no train samples in the shards under {out_dir}")
    count = total.reshape(2, 1)
    mean = sum_channel / count
    var = np.maximum(sum_sq_channel / count - np.square(mean), 0.0)
    return {"mean": mean.astype(np.float32), "std": np.sqrt(var).astype(np.float32),
            "token_var": np.float32(sum_sq_token / total.sum()), "n_tokens": total.astype(np.int64)}


def viz_samples(index: dict[str, np.ndarray], limit: int) -> np.ndarray:
    """Fixed samples kept at full resolution: every sample of each task's first test episode."""
    if limit <= 0:
        return np.zeros(0, dtype=np.int64)
    test = split_ids(index, "test")
    chosen = []
    for task in np.unique(index["task"][test]):
        rows = test[index["task"][test] == task]
        first = index["episode_index"][rows].min()
        chosen.extend(rows[index["episode_index"][rows] == first].tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))[:limit]


class ShardFrames(_IterableDataset):
    """Iterable of (sample ids, frames uint8 [b, P, 3, H, W], fps [b]) batches for one shard.

    Each DataLoader worker takes one contiguous slice of the shard, so consecutive samples come
    from the same parquet row groups and every row group is read about once.
    """

    def __init__(self, ids: np.ndarray, index: dict[str, np.ndarray], positions: tuple[int, ...],
                 batch: int, future_offset: int):
        super().__init__()
        self.ids = ids
        self.dataset_dir = index["dataset_dir"]
        self.row_t = index["row_t"].astype(np.int64)
        self.fps = (index["fps"].astype(np.float32) / float(future_offset))
        self.positions = positions
        self.batch = batch

    def __iter__(self):
        from research.jepa_probe.frames import FrameReader

        info = torch.utils.data.get_worker_info()
        ids = self.ids if info is None else np.array_split(self.ids, info.num_workers)[info.id]
        readers: dict[str, FrameReader] = {}
        for start in range(0, len(ids), self.batch):
            chunk = ids[start:start + self.batch]
            frames = []
            for sample in chunk:
                directory = str(self.dataset_dir[sample])
                if directory not in readers:
                    readers.clear()
                    readers[directory] = FrameReader(Path(directory))
                rows = [int(self.row_t[sample]) + position for position in self.positions]
                frames.append(readers[directory].load(rows))
            yield torch.from_numpy(chunk), torch.stack(frames), torch.from_numpy(self.fps[chunk])


def file_digest(path: Path) -> dict:
    """sha256 of a weights file, or of each weights file inside a model directory."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix in (".safetensors", ".bin", ".pt", ".pth"))
        return {str(p.relative_to(path)): file_digest(p)["sha256"] for p in files}
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def weight_digests(variant: dict) -> dict:
    paths = {key: variant[key] for key in ("weights",) if key in variant}
    if "video" in variant:
        paths.update({key: variant["video"][key] for key in ("ckpt_path", "config_path") if key in variant["video"]})
    return {key: file_digest(Path(value)) for key, value in paths.items() if Path(value).exists()}


def write_meta(out_dir: Path, variant: dict, n_samples: int, **extra) -> None:
    path = out_dir / "meta.json"
    meta = read_json(path) if path.is_file() else {}
    meta.update({"variant": variant, "samples": n_samples, "pooled_tokens": POOLED_GRID ** 2,
                 "dtype": "float16", "code_commit": os.environ.get("LB_COMMIT")}, **extra)
    partial = path.with_name(f".partial_{os.getpid()}_{path.name}")
    partial.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    partial.replace(path)


def write_stats(out_dir: Path, stats: dict[str, np.ndarray]) -> None:
    partial = out_dir / f".partial_{os.getpid()}_stats.npz"
    np.savez(partial, **stats)
    partial.replace(out_dir / "stats.npz")


def parse_part(spec: str | None) -> tuple[int, int]:
    """'k/n' -> (k, n): this process takes shards k, k+n, k+2n, ..."""
    if not spec:
        return 0, 1
    k, n = (int(value) for value in spec.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"bad --part {spec!r}: need 0 <= k < n")
    return k, n


class _Job:
    """One variant inside a multi-variant run; `runner` stays None when this part has nothing to do."""

    def __init__(self, variant: dict, out_dir: Path):
        self.variant, self.out_dir = variant, Path(out_dir)
        self.runner = None
        self.wrote = False

    @property
    def pool(self) -> int:
        return self.runner.grid // POOLED_GRID

    def needs(self, start: int, end: int) -> bool:
        path = shard_path(self.out_dir, start)
        return not (path.is_file() and _shard_ok(path, end - start))


def extract(variant: dict, index_dir: Path, out_dir: Path, **kwargs) -> dict:
    return extract_many([variant], [out_dir], index_dir, **kwargs)[variant["name"]]


def extract_many(variants: list[dict], out_dirs: list[Path], index_dir: Path, *, batch: int, num_workers: int,
                 device: str, viz: int = 1000, max_shards: int | None = None, compile_teacher: bool = False,
                 part: tuple[int, int] = (0, 1)) -> dict[str, dict]:
    from research.jepa_probe.teachers import build_runner

    index = load_index(index_dir)
    index_meta = read_json(index_dir / "index_meta.json")
    n_samples = len(index["sample_id"])
    shards = [s for i, s in enumerate(iter_shards(n_samples)) if i % part[1] == part[0]]
    viz_ids = viz_samples(index, viz)
    started = time.time()

    jobs = []
    for variant, out_dir in zip(variants, out_dirs):
        job = _Job(variant, out_dir)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        write_meta(job.out_dir, variant, n_samples)
        if any(job.needs(*shard) for shard in shards):
            job.runner = build_runner(variant, device)
            if max(job.runner.positions) > index_meta["future_offset"]:
                raise ValueError(f"{variant['name']}: clip position {max(job.runner.positions)} is past "
                                 f"future_offset {index_meta['future_offset']}")
            if compile_teacher:
                job.runner.compile()
        jobs.append(job)

    done_shards = 0
    for start, end in shards:
        todo = [job for job in jobs if job.runner is not None and job.needs(start, end)]
        if not todo:
            continue
        if max_shards is not None and done_shards >= max_shards:
            break
        shard_started = time.time()
        positions = tuple(sorted({p for job in todo for p in job.runner.positions}))
        columns = [[positions.index(p) for p in job.runner.positions] for job in todo]
        frames = ShardFrames(np.arange(start, end), index, positions, batch, index_meta["future_offset"])
        loader = torch.utils.data.DataLoader(
            frames, batch_size=None, num_workers=num_workers,
            pin_memory=torch.device(device).type == "cuda", prefetch_factor=4 if num_workers else None)
        outputs = [{"cur": None, "fut": None, "viz_ids": [], "viz_cur": [], "viz_fut": []} for _ in todo]
        filled = np.zeros(end - start, dtype=bool)
        for batch_ids, batch_frames, fps in loader:
            batch_frames, fps = batch_frames.to(device, non_blocking=True), fps.to(device)
            local = batch_ids.numpy() - start
            keep = np.isin(batch_ids.numpy(), viz_ids)
            for job, cols, out in zip(todo, columns, outputs):
                current, future = job.runner(batch_frames[:, cols], fps)
                pooled_cur = pool_tokens(current, job.runner.grid, job.pool).cpu().numpy().astype(np.float16)
                pooled_fut = pool_tokens(future, job.runner.grid, job.pool).cpu().numpy().astype(np.float16)
                if out["cur"] is None:
                    out["cur"] = np.zeros((end - start, *pooled_cur.shape[1:]), dtype=np.float16)
                    out["fut"] = np.zeros_like(out["cur"])
                out["cur"][local], out["fut"][local] = pooled_cur, pooled_fut
                if keep.any():
                    mask = torch.from_numpy(keep).to(current.device)
                    out["viz_ids"].append(batch_ids.numpy()[keep])
                    out["viz_cur"].append(current[mask].cpu().numpy().astype(np.float16))
                    out["viz_fut"].append(future[mask].cpu().numpy().astype(np.float16))
            filled[local] = True
        if not filled.all():
            raise RuntimeError(f"shard {start}: {int((~filled).sum())} samples were not produced")
        for job, out in zip(todo, outputs):
            extra = {key: np.concatenate(out[key]) for key in ("viz_ids", "viz_cur", "viz_fut") if out[key]}
            save_shard(shard_path(job.out_dir, start), out["cur"], out["fut"], **extra)
            job.wrote = True
        done_shards += 1
        names = ",".join(job.variant["name"] for job in todo)
        print(f"shard {start}-{end} [{names}]: {time.time() - shard_started:.1f}s", flush=True)

    results = {}
    for job in jobs:
        remaining = pending_samples(job.out_dir, n_samples)
        result = {"remaining": len(remaining), "seconds": time.time() - started}
        if not remaining and (job.wrote or not (job.out_dir / "stats.npz").is_file()):
            stats = compute_stats(job.out_dir, split_ids(index, "train"))
            write_stats(job.out_dir, stats)
            write_meta(job.out_dir, job.variant, n_samples, channels=int(stats["mean"].shape[-1]),
                       viz_samples=int(len(viz_ids)), weights=weight_digests(job.variant),
                       extract_seconds=result["seconds"])
        results[job.variant["name"]] = result
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, help="one name, or a comma-separated list with --features")
    parser.add_argument("--variants_cfg", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="directory written by build_index")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--out", type=Path, help="output directory of a single variant")
    outputs.add_argument("--features", type=Path, help="root; each variant writes to <features>/<name>")
    parser.add_argument("--part", default=None, help="k/n: process every n-th shard, starting at shard k")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--viz", type=int, default=1000, help="samples kept at full 16x16 resolution")
    parser.add_argument("--max_shards", type=int, default=None, help="stop after this many new shards")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)

    names = [name for name in args.variant.split(",") if name]
    if args.out is not None and len(names) != 1:
        parser.error("--out takes a single variant; use --features for several")
    variants = []
    for name in names:
        variant = load_variant(args.variants_cfg, name)
        variant["dtype"] = args.dtype
        variants.append(variant)
    out_dirs = [args.out] if args.out is not None else [args.features / name for name in names]
    results = extract_many(variants, out_dirs, args.index, batch=args.batch, num_workers=args.num_workers,
                           device=args.device, viz=args.viz, max_shards=args.max_shards,
                           compile_teacher=args.compile, part=parse_part(args.part))
    for name, out_dir in zip(names, out_dirs):
        result = results[name]
        status = "complete" if not result["remaining"] else f"{result['remaining']} samples left"
        print(f"{name}: {status} in {result['seconds']:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
