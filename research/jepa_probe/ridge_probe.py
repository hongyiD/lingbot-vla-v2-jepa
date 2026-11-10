"""Closed-form ridge probes: the linear reference for every task, and P3.

The linear probe globally pools each frame's tokens, concatenates them, and fits ridge regression
with the penalty chosen on the val split. P3 predicts the future tokens (pooled to 4x4, then PCA to
256 dims) from the current tokens alone and from the current tokens plus the action chunk; the gap
between the two R^2 values is how much of the future target an action determines.

Tokens are z-scored per channel with the variant's stats.npz first, as in the attention probe.
Writes <out>/<task>/<variant>.json (same metrics as train_probe) and <out>/P3/<variant>.json.

  python -m research.jepa_probe.ridge_probe --variants V0,V1 \
      --features $P/features --index $P/index --out $P/results/ridge
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.jepa_probe.common import load_index, prediction_metrics, split_ids

PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LINEAR_TASKS = ("P1", "P1s", "P1d", "P2")


def fit_ridge(features: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    """Ridge weights [D + 1, T] including a bias column prepended to `features`."""
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    gram = design.T @ design
    gram[1:, 1:] += penalty * len(features) * np.eye(features.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    return design @ weights


def r2(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.mean(np.square(prediction - target))
    variance = np.mean(np.square(target - target.mean(axis=0)))
    return float(1.0 - residual / max(variance, 1e-12))


def select_penalty(train_x, train_y, val_x, val_y) -> float:
    best_penalty, best_score = PENALTIES[0], -np.inf
    for penalty in PENALTIES:
        score = r2(predict(val_x, fit_ridge(train_x, train_y, penalty)), val_y)
        if score > best_score:
            best_penalty, best_score = penalty, score
    return best_penalty


def standardize(train: np.ndarray, others: list[np.ndarray]) -> list[np.ndarray]:
    """z-score with train stats; columns that are constant on train are zeroed rather than blown up."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-6 * max(float(std.max()), 1e-12), std, np.inf)
    return [((array - mean) / std).astype(np.float64) for array in [train, *others]]


def ridge_fit_predict(x: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> tuple[float, dict[str, np.ndarray]]:
    """Standardize on train, choose the penalty on val, return predictions for every split in `x`."""
    names = [name for name in x if name != "train"]
    scaled = dict(zip(["train", *names], standardize(x["train"], [x[name] for name in names])))
    penalty = select_penalty(scaled["train"], y["train"], scaled["val"], y["val"])
    weights = fit_ridge(scaled["train"], y["train"], penalty)
    return penalty, {name: predict(values, weights) for name, values in scaled.items()}


def pca_project(train: np.ndarray, others: list[np.ndarray], dims: int, *,
                oversample: int = 32, power_iters: int = 4, seed: int = 0) -> list[np.ndarray]:
    """Project every array with the top-`dims` PCA basis of `train`. All arrays share the column count.

    Randomized SVD (Halko et al.): a full SVD of a 30k x 16k matrix is far too slow on a CPU.
    """
    center = train.mean(axis=0)
    centered = (train - center).astype(np.float32)
    dims = min(dims, len(centered) - 1, centered.shape[1])  # centering removes one rank
    width = min(dims + oversample, *centered.shape)
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(centered @ rng.standard_normal((centered.shape[1], width)).astype(np.float32))
    for _ in range(power_iters):
        basis, _ = np.linalg.qr(centered @ (centered.T @ basis))
    _, _, components = np.linalg.svd(basis.T @ centered, full_matrices=False)
    components = components[:dims].T
    return [(array - center).astype(np.float32) @ components for array in [train, *others]]


def pool_grid(tokens: np.ndarray, grid: int = 8, pool: int = 2) -> np.ndarray:
    """[N, grid*grid, C] -> [N, (grid/pool)**2 * C] by averaging non-overlapping pool x pool patches."""
    n, _, channels = tokens.shape
    grouped = tokens.astype(np.float32).reshape(n, grid // pool, pool, grid // pool, pool, channels)
    return grouped.mean(axis=(2, 4)).reshape(n, -1)


def p3_scores(current: np.ndarray, future: np.ndarray, action: np.ndarray, *,
              train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray,
              dims: int = 256, input_dims: int = 512) -> dict:
    """R^2 of the future tokens from the current tokens, with and without the action chunk.

    `current` / `future` are tokens [N, 64, C] or already pooled to 4x4 and flattened [N, 16 * C].
    Both are PCA-projected on the train split (future to `dims`, current to `input_dims`); a ridge
    on the raw 64 x 1024 inputs would need a 65k x 65k Gram matrix.
    """
    if current.ndim == 3:
        current, future = pool_grid(current), pool_grid(future)
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    future_parts = dict(zip(ids, pca_project(future[train_ids], [future[val_ids], future[test_ids]], dims)))
    current_parts = dict(zip(ids, pca_project(current[train_ids], [current[val_ids], current[test_ids]],
                                              input_dims)))
    flat_action = action.reshape(len(action), -1)
    scores = {}
    for name, with_action in (("cur", False), ("cur_act", True)):
        x = {split: np.concatenate([current_parts[split], flat_action[ids[split]]], axis=1) if with_action
             else current_parts[split] for split in ids}
        penalty, predictions = ridge_fit_predict(x, future_parts)
        scores[name] = r2(predictions["test"], future_parts["test"])
        scores[f"{name}_penalty"] = penalty
    scores["delta"] = scores["cur_act"] - scores["cur"]
    return scores


def load_pooled(directory: Path, n_samples: int) -> dict[str, np.ndarray]:
    """Stream one variant's shards into global means [N, C] and 4x4-pooled tokens [N, 16 * C]."""
    stats = np.load(directory / "stats.npz")
    mean, std = stats["mean"], np.maximum(stats["std"], 1e-6)
    out: dict[str, np.ndarray] = {}
    filled = 0
    for path in sorted(directory.glob("shard_*.npz")):
        start = int(path.stem.split("_")[1])
        with np.load(path) as data:
            for row, stream in enumerate(("cur", "fut")):
                tokens = (data[stream].astype(np.float32) - mean[row]) / std[row]
                if f"mean_{stream}" not in out:
                    channels = tokens.shape[-1]
                    out[f"mean_{stream}"] = np.zeros((n_samples, channels), dtype=np.float32)
                    out[f"pool_{stream}"] = np.zeros((n_samples, 16 * channels), dtype=np.float32)
                stop = start + len(tokens)
                out[f"mean_{stream}"][start:stop] = tokens.mean(axis=1)
                out[f"pool_{stream}"][start:stop] = pool_grid(tokens)
            filled += len(data["cur"])
    if filled != n_samples:
        raise ValueError(f"{directory}: shards hold {filled} samples, index has {n_samples}")
    return out


def linear_inputs(task: str, pooled: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
    if task == "P2":
        return pooled["mean_cur"]
    parts = [pooled["mean_cur"], pooled["mean_fut"]]
    if task == "P1s":
        parts.append(state)
    return np.concatenate(parts, axis=1)


def run_variant(name: str, features: Path, index: dict, labels: dict, out: Path, *, p3: bool = True) -> None:
    from research.jepa_probe.train_probe import target_array

    started = time.time()
    n_samples = len(index["sample_id"])
    pooled = load_pooled(features / name, n_samples)
    ids = {split: split_ids(index, split) for split in ("train", "val", "test")}
    for task in LINEAR_TASKS:
        target = target_array(labels, task)
        x_all = linear_inputs(task, pooled, labels["state_t"].astype(np.float32))
        penalty, predictions = ridge_fit_predict({split: x_all[ids[split]] for split in ids},
                                                 {split: target[ids[split]] for split in ids})
        record = {"task": task, "input": name, "probe": "ridge", "penalty": penalty, "lr": None, "seed": 0}
        for split in ("val", "test"):
            record[split] = prediction_metrics(predictions[split], target[ids[split]],
                                               episode_keys=index["episode_key"][ids[split]],
                                               events=index["gripper_event"][ids[split]].astype(bool))
        _write(out / task / f"{name}.json", record)
        print(f"ridge {task}/{name}: val {record['val']['nmse']:.4f} test {record['test']['nmse']:.4f}", flush=True)
    if p3:
        scores = p3_scores(pooled["pool_cur"], pooled["pool_fut"], labels["action_chunk"].astype(np.float32),
                           train_ids=ids["train"], val_ids=ids["val"], test_ids=ids["test"])
        _write(out / "P3" / f"{name}.json", {"task": "P3", "input": name, **scores})
        print(f"P3 {name}: R2 cur {scores['cur']:.3f}, cur+act {scores['cur_act']:.3f}, "
              f"delta {scores['delta']:.3f}", flush=True)
    print(f"{name}: {time.time() - started:.0f}s", flush=True)


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", required=True, help="comma-separated variant names")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no_p3", action="store_true")
    args = parser.parse_args(argv)

    index = load_index(args.index)
    labels = dict(np.load(args.index / "labels.npz"))
    for name in args.variants.split(","):
        if not (args.features / name / "stats.npz").is_file():
            print(f"skip {name}: no finished features under {args.features / name}", flush=True)
            continue
        run_variant(name, args.features, index, labels, args.out, p3=not args.no_p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
