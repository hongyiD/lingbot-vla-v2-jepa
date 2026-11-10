"""Closed-form ridge probes: the linear reference for every task, and P3.

The linear probe globally pools each frame's tokens, concatenates them, and fits ridge regression
with the penalty chosen on the val split. P3 predicts the future tokens (pooled to 4x4, then PCA to
256 dims) from the current tokens alone and from the current tokens plus the action chunk; the gap
between the two R^2 values is how much of the future target an action determines.

  python -m research.jepa_probe.ridge_probe --variants V0,V1 \
      --features $P/features --index $P/index --out $P/results/ridge
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PENALTIES = (1e-4, 1e-2, 1.0, 1e2)


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


def pca_project(train: np.ndarray, others: list[np.ndarray], dims: int, *,
                oversample: int = 32, power_iters: int = 4, seed: int = 0) -> list[np.ndarray]:
    """Project every array with the top-`dims` PCA basis of `train`. All arrays share the column count.

    Randomized SVD (Halko et al.): a full SVD of a 30k x 16k matrix is far too slow on a CPU.
    """
    center = train.mean(axis=0)
    centered = (train - center).astype(np.float32)
    dims = min(dims, *centered.shape)
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

    Both token sets are pooled to 4x4 and PCA-projected on the train split (future to `dims`,
    current to `input_dims`); a ridge on the raw 64 x 1024 inputs would need a 65k x 65k Gram matrix.
    """
    future_train, future_val, future_test = pca_project(
        pool_grid(future[train_ids]), [pool_grid(future[val_ids]), pool_grid(future[test_ids])], dims)
    current_parts = pca_project(
        pool_grid(current[train_ids]), [pool_grid(current[val_ids]), pool_grid(current[test_ids])], input_dims)
    current_by_split = dict(zip(("train", "val", "test"), current_parts))
    ids_by_split = {"train": train_ids, "val": val_ids, "test": test_ids}
    scores = {}
    for name, extra in (("cur", None), ("cur_act", action)):
        def take(split, values=extra):
            base = current_by_split[split]
            if values is None:
                return base
            ids = ids_by_split[split]
            return np.concatenate([base, values[ids].reshape(len(ids), -1)], axis=1)
        penalty = select_penalty(take("train"), future_train, take("val"), future_val)
        scores[name] = r2(predict(take("test"), fit_ridge(take("train"), future_train, penalty)), future_test)
    scores["delta"] = scores["cur_act"] - scores["cur"]
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", required=True, help="comma-separated variant names")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"ridge probes for {args.variants}: features {args.features}, out {args.out}")
    raise SystemExit("ridge_probe: the driver is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
