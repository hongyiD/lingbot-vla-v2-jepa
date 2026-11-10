"""Extract pooled teacher tokens for every sample of one variant.

Outputs shards of fp16 tokens under <out>/shard_XXXXXX.npz, each holding `cur` and `fut` arrays of
shape [shard, 64, 1024] (16x16 tokens pooled 2x2 to 8x8). Finished shards are skipped, so a run can
resume. `stats.npz` carries the per-channel train-set mean and std that the probes normalize with.

  python -m research.jepa_probe.extract_features --variant V1 \
      --variants_cfg research/jepa_probe/configs/variants.yaml \
      --index $P/index --out $P/features/V1 --batch 64 --dtype bf16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.jepa_probe.common import read_json

SHARD_SIZE = 2048
POOLED_TOKENS = 64  # 8 x 8


def load_variant(config_path: Path, name: str) -> dict:
    import yaml
    variants = yaml.safe_load(config_path.read_text(encoding="utf-8"))["variants"]
    if name not in variants:
        raise KeyError(f"variant {name!r} not in {config_path}; have {sorted(variants)}")
    variant = variants[name]
    variant["name"] = name
    return variant


def pool_tokens(tokens, grid: int, pool: int):
    """[B, grid*grid, C] -> [B, (grid/pool)**2, C] by averaging non-overlapping patches."""
    if tokens.shape[1] != grid * grid:
        raise ValueError(f"expected {grid * grid} tokens, got {tokens.shape[1]}")
    if grid % pool:
        raise ValueError(f"grid {grid} is not divisible by pool {pool}")
    batch, _, channels = tokens.shape
    grouped = tokens.reshape(batch, grid // pool, pool, grid // pool, pool, channels)
    return grouped.mean(dim=(2, 4)).reshape(batch, -1, channels)


def iter_shards(n_samples: int, shard_size: int = SHARD_SIZE):
    for start in range(0, n_samples, shard_size):
        yield start, min(start + shard_size, n_samples)


def shard_path(out_dir: Path, start: int) -> Path:
    return out_dir / f"shard_{start:07d}.npz"


def save_shard(path: Path, cur: np.ndarray, fut: np.ndarray) -> None:
    # np.savez appends ".npz" when the name does not end with it, so use a final name directly.
    np.savez(path, cur=cur, fut=fut)


def pending_samples(out_dir: Path, n_samples: int) -> list[int]:
    done = set()
    for start, end in iter_shards(n_samples):
        path = shard_path(out_dir, start)
        if path.is_file() and _shard_ok(path, end - start):
            done.update(range(start, end))
    return [i for i in range(n_samples) if i not in done]


def _shard_ok(path: Path, expected_rows: int) -> bool:
    with np.load(path) as data:
        return data["cur"].shape[0] == expected_rows and data["fut"].shape == data["cur"].shape


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
    count = total.reshape(2, 1)
    mean = sum_channel / count
    var = np.maximum(sum_sq_channel / count - np.square(mean), 0.0)
    return {"mean": mean.astype(np.float32), "std": np.sqrt(var).astype(np.float32),
            "token_var": np.float32(sum_sq_token / total.sum()), "n_tokens": total.astype(np.int64)}


def write_meta(out_dir: Path, variant: dict, n_samples: int) -> None:
    meta = {"variant": variant, "samples": n_samples, "pooled_tokens": POOLED_TOKENS, "dtype": "float16"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--variants_cfg", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="directory written by build_index")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--pool", type=int, default=2)
    args = parser.parse_args(argv)

    variant = load_variant(args.variants_cfg, args.variant)
    index_meta = read_json(args.index / "index_meta.json")
    args.out.mkdir(parents=True, exist_ok=True)
    write_meta(args.out, variant, index_meta["samples"])
    remaining = pending_samples(args.out, index_meta["samples"])
    print(f"{args.variant}: {index_meta['samples'] - len(remaining)} done, {len(remaining)} to extract")
    if remaining:
        print("feature extraction runs on the GPU pod; this skeleton records the plan and exits here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
