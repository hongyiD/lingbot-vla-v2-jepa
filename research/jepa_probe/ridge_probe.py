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


def pca_project(train: np.ndarray, others: list[np.ndarray], dims: int) -> list[np.ndarray]:
    """Project every array with the PCA basis of `train`. All arrays share the column count."""
    center = train.mean(axis=0)
    _, _, basis = np.linalg.svd(train - center, full_matrices=False)
    basis = basis[:dims].T
    return [(array - center) @ basis for array in [train, *others]]


def p3_scores(current: np.ndarray, future: np.ndarray, action: np.ndarray, *,
              train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray, dims: int = 256) -> dict:
    """R^2 of the future tokens from the current tokens, with and without the action chunk."""
    pooled_future = future.reshape(len(future), 8, 8, -1)[:, ::2, ::2, :].reshape(len(future), -1)
    pooled_current = current.reshape(len(current), -1)
    future_train, future_val, future_test = pca_project(
        pooled_future[train_ids], [pooled_future[val_ids], pooled_future[test_ids]], dims)
    scores = {}
    for name, extra in (("cur", None), ("cur_act", action)):
        def take(ids, values=extra):
            base = pooled_current[ids]
            return base if values is None else np.concatenate([base, values[ids].reshape(len(ids), -1)], axis=1)
        penalty = select_penalty(take(train_ids), future_train, take(val_ids), future_val)
        scores[name] = r2(predict(take(test_ids), fit_ridge(take(train_ids), future_train, penalty)), future_test)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
