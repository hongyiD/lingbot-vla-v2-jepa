"""Attention probes for P1, P1s, P1-delta, P2 and P4, and the B-mean / B-state baselines (plan 2.3-2.4).

Tokens are channel-wise z-scored with the train-split stats from extract_features, then a 2-layer
Transformer (width 256, 4 heads, one learned pooling query) predicts the target. Training stops when
the val nMSE has not improved for `patience` epochs; the best epoch is evaluated on val and test.

Inputs are feature directories under --features (V0, V1, ..., B-pixel), a channel concat of two of
them written "V0+V1" (P4), or the baselines B-mean (train mean) and B-state (MLP on state[t] and the
task one-hot). Each run writes <out>/<task>/<input>/lr<lr>_s<seed>.json.

  # the whole sweep, one worker process per GPU
  python -m research.jepa_probe.train_probe --sweep research/jepa_probe/configs/sweep_main.yaml \
      --features $P/features --index $P/index --out $P/results --gpus 0,1,2,3
  # one run
  python -m research.jepa_probe.train_probe --task P1s --input V1 --lr 1e-3 --seed 0 \
      --features $P/features --index $P/index --out $P/results
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from research.jepa_probe.common import STATE_DIM, load_index, nmse, prediction_metrics, split_ids

try:
    import torch
    from torch import nn
except ImportError:  # index building and the ridge probe never need torch
    torch = None
    nn = None

TASKS = {
    # target: which label array; state: add a state token; frames: token streams used.
    "P1": {"target": "action_chunk", "state": False, "frames": ("cur", "fut")},
    "P1s": {"target": "action_chunk", "state": True, "frames": ("cur", "fut")},
    "P1d": {"target": "state_delta", "state": False, "frames": ("cur", "fut")},
    "P2": {"target": "state_t", "state": False, "frames": ("cur",)},
}
BASELINES = ("B-mean", "B-state")
# B-state's input is state[t], which is P2's target.
SKIP = {("P2", "B-state")}

__all__ = ["TASKS", "build_attention_probe", "nmse", "normalize_tokens", "target_array"]


def build_attention_probe(in_dim: int, out_dim: int, d_model: int = 256, n_heads: int = 4,
                          n_layers: int = 2, dropout: float = 0.1, n_streams: int = 2):
    """The probe class is built on demand, so this module imports without torch."""

    class AttentionProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model))
            self.pos = nn.Parameter(torch.zeros(64, d_model))
            self.stream = nn.Parameter(torch.zeros(n_streams, d_model))
            self.state_proj = nn.Linear(STATE_DIM, d_model)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.query = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pool = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, out_dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.stream, std=0.02)
            nn.init.trunc_normal_(self.query, std=0.02)

        def forward(self, tokens, state=None):
            """tokens: [B, streams, 64, C]. state: [B, 14] or None."""
            batch, streams, positions, _ = tokens.shape
            hidden = self.proj(tokens) + self.pos + self.stream[:streams].view(1, streams, 1, -1)
            hidden = hidden.reshape(batch, streams * positions, -1)
            if state is not None:
                hidden = torch.cat([hidden, self.state_proj(state).unsqueeze(1)], dim=1)
            hidden = self.encoder(hidden)
            pooled, _ = self.pool(self.query.expand(batch, -1, -1), hidden, hidden)
            return self.head(pooled.squeeze(1))

    return AttentionProbe()


def build_state_mlp(n_tasks: int, out_dim: int, width: int = 512, dropout: float = 0.1):
    """B-state: state[t] and the task one-hot through a 2-hidden-layer MLP."""
    return nn.Sequential(nn.Linear(STATE_DIM + n_tasks, width), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, out_dim))


def target_array(labels: dict, task: str) -> np.ndarray:
    spec = TASKS[task]
    if spec["target"] == "state_delta":
        values = labels["state_t49"] - labels["state_t"]
    else:
        values = labels[spec["target"]]
    return values.reshape(len(values), -1).astype(np.float32)


def normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """tokens [N, 64, C], mean/std [C] -> z-scored float32."""
    return ((tokens.astype(np.float32) - mean) / np.maximum(std, 1e-6)).astype(np.float32)


# ---------------------------------------------------------------- features


class Features:
    """One input's pooled tokens on the device, fp16, z-scored on the fly per batch."""

    def __init__(self, cur, fut, mean, std):
        self.streams = {"cur": cur, "fut": fut}
        self.mean = mean  # [2, C]
        self.std = std

    @property
    def channels(self) -> int:
        return int(self.mean.shape[-1])

    def batch(self, names: tuple[str, ...], ids):
        parts = []
        for name in names:
            row = 0 if name == "cur" else 1
            values = self.streams[name][ids].float()
            parts.append((values - self.mean[row]) / self.std[row].clamp_min(1e-6))
        return torch.stack(parts, dim=1)  # [B, streams, 64, C]


def load_features(root: Path, name: str, n_samples: int, device) -> Features:
    """`name` is a feature directory under `root`, or 'A+B' for their channel concat."""
    if "+" in name:
        parts = [load_features(root, part, n_samples, device) for part in name.split("+")]
        cat = lambda key: torch.cat([part.streams[key] for part in parts], dim=-1)  # noqa: E731
        return Features(cat("cur"), cat("fut"), torch.cat([p.mean for p in parts], dim=-1),
                        torch.cat([p.std for p in parts], dim=-1))
    directory = root / name
    if not (directory / "stats.npz").is_file():
        raise FileNotFoundError(f"no finished features at {directory} (stats.npz missing)")
    shards = sorted(directory.glob("shard_*.npz"))
    cur = fut = None
    filled = 0
    for path in shards:
        start = int(path.stem.split("_")[1])
        with np.load(path) as data:
            block_cur, block_fut = data["cur"], data["fut"]
        if cur is None:
            cur = torch.empty((n_samples, *block_cur.shape[1:]), dtype=torch.float16, device=device)
            fut = torch.empty_like(cur)
        cur[start:start + len(block_cur)] = torch.from_numpy(block_cur).to(device)
        fut[start:start + len(block_fut)] = torch.from_numpy(block_fut).to(device)
        filled += len(block_cur)
    if filled != n_samples:
        raise ValueError(f"{directory}: shards hold {filled} samples, index has {n_samples}")
    stats = np.load(directory / "stats.npz")
    return Features(cur, fut, torch.from_numpy(stats["mean"]).to(device), torch.from_numpy(stats["std"]).to(device))


# ---------------------------------------------------------------- training


def fit(model, forward, target, *, train_ids, val_ids, lr: float, seed: int, epochs: int = 40,
        batch_size: int = 256, patience: int = 5) -> dict:
    """Train with AdamW + cosine schedule; keep the epoch with the best val nMSE.

    `forward(model, ids)` returns predictions for a LongTensor of sample ids on the model's device.
    """
    device = target.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_tensor = torch.from_numpy(train_ids).to(device)
    val_target = target[torch.from_numpy(val_ids).to(device)].cpu().numpy()
    best = {"val": float("inf"), "epoch": -1, "state": None}
    curve, idle = [], 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_ids), generator=generator).to(device)
        for start in range(0, len(train_ids), batch_size):
            ids = train_tensor[order[start:start + batch_size]]
            loss = nn.functional.mse_loss(forward(model, ids), target[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        schedule.step()
        score = nmse(predict(model, forward, val_ids, device), val_target)
        curve.append(score)
        if score < best["val"]:
            best = {"val": score, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            idle = 0
        else:
            idle += 1
            if idle >= patience:
                break
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "val_curve": curve}


def predict(model, forward, ids: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    ids_tensor = torch.from_numpy(ids).to(device)
    with torch.no_grad():
        out = [forward(model, ids_tensor[start:start + batch_size]).float().cpu()
               for start in range(0, len(ids), batch_size)]
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- jobs


def result_path(out: Path, job: dict) -> Path:
    name = "mean.json" if job["input"] == "B-mean" else f"lr{job['lr']:g}_s{job['seed']}.json"
    return out / job["task"] / job["input"] / name


def sweep_jobs(sweep: dict, inputs: list[str] | None = None, tasks: list[str] | None = None) -> list[dict]:
    jobs = []
    for task in tasks or sweep["tasks"]:
        for name in inputs or sweep["inputs"]:
            if (task, name) in SKIP:
                continue
            if name == "B-mean":
                jobs.append({"task": task, "input": name, "lr": None, "seed": 0})
                continue
            for lr in sweep["lrs"]:
                for seed in sweep["seeds"]:
                    jobs.append({"task": task, "input": name, "lr": float(lr), "seed": int(seed)})
    return jobs


class Context:
    def __init__(self, features_root: Path, index_dir: Path, out: Path, device: str, train_cfg: dict):
        self.features_root = features_root
        self.out = out
        self.device = torch.device(device)
        self.train_cfg = train_cfg
        self.index = load_index(index_dir)
        self.labels = dict(np.load(index_dir / "labels.npz"))
        self.n_samples = len(self.index["sample_id"])
        self.ids = {name: split_ids(self.index, name) for name in ("train", "val", "test")}
        self._features: tuple[str, Features] | None = None

    def features(self, name: str) -> Features:
        if self._features is None or self._features[0] != name:
            self._features = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._features = (name, load_features(self.features_root, name, self.n_samples, self.device))
        return self._features[1]

    def evaluate(self, prediction_by_split: dict[str, np.ndarray], target: np.ndarray) -> dict:
        out = {}
        for split, prediction in prediction_by_split.items():
            ids = self.ids[split]
            out[split] = prediction_metrics(prediction, target[ids], episode_keys=self.index["episode_key"][ids],
                                            events=self.index["gripper_event"][ids].astype(bool))
        return out


def run_job(job: dict, ctx: Context) -> dict:
    task, name = job["task"], job["input"]
    spec = TASKS[task]
    target_np = target_array(ctx.labels, task)
    started = time.time()
    record = {**job, "probe": "attention", "train": dict(ctx.train_cfg)}
    if name == "B-mean":
        mean = target_np[ctx.ids["train"]].mean(axis=0, keepdims=True)
        predictions = {split: np.repeat(mean, len(ctx.ids[split]), axis=0) for split in ("val", "test")}
        record.update(best_epoch=None, val_curve=[])
    else:
        torch.manual_seed(job["seed"])
        target = torch.from_numpy(target_np).to(ctx.device)
        state = torch.from_numpy(ctx.labels["state_t"].astype(np.float32)).to(ctx.device)
        if name == "B-state":
            n_tasks = int(ctx.labels["task_id"].max()) + 1
            onehot = nn.functional.one_hot(torch.from_numpy(ctx.labels["task_id"]).long(), n_tasks).float()
            inputs = torch.cat([state, onehot.to(ctx.device)], dim=1)
            model = build_state_mlp(n_tasks, target.shape[1]).to(ctx.device)
            forward = lambda m, ids: m(inputs[ids])  # noqa: E731
        else:
            features = ctx.features(name)
            model = build_attention_probe(features.channels, target.shape[1],
                                          n_streams=len(spec["frames"])).to(ctx.device)
            use_state = spec["state"]
            forward = lambda m, ids: m(features.batch(spec["frames"], ids),  # noqa: E731
                                       state[ids] if use_state else None)
        history = fit(model, forward, target, train_ids=ctx.ids["train"], val_ids=ctx.ids["val"],
                      lr=job["lr"], seed=job["seed"], **ctx.train_cfg)
        predictions = {split: predict(model, forward, ctx.ids[split], ctx.device) for split in ("val", "test")}
        record.update(history)
    record.update(ctx.evaluate(predictions, target_np))
    record["seconds"] = time.time() - started
    return record


def run_jobs(jobs: list[dict], ctx: Context) -> None:
    jobs = sorted(jobs, key=lambda job: (job["input"], job["task"], job["lr"] or 0, job["seed"]))
    for job in jobs:
        path = result_path(ctx.out, job)
        if path.is_file():
            continue
        try:
            record = run_job(job, ctx)
        except FileNotFoundError as exc:
            print(f"skip {job['task']}/{job['input']}: {exc}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".partial_{path.name}")
        partial.write_text(json.dumps(record, indent=1), encoding="utf-8")
        partial.replace(path)
        print(f"{job['task']}/{job['input']} lr={job['lr']} seed={job['seed']}: "
              f"val {record['val']['nmse']:.4f} test {record['test']['nmse']:.4f} ({record['seconds']:.0f}s)",
              flush=True)


def best_inputs(out: Path, task: str, candidates: list[str], top: int) -> list[str]:
    """The `top` candidates with the lowest mean val nMSE (best lr, averaged over seeds)."""
    scores = {}
    for name in candidates:
        by_lr: dict[str, list[float]] = {}
        for path in (out / task / name).glob("lr*_s*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            by_lr.setdefault(str(record["lr"]), []).append(record["val"]["nmse"])
        if by_lr:
            scores[name] = min(np.mean(values) for values in by_lr.values())
    return sorted(scores, key=scores.get)[:top]


def dispatch(jobs: list[dict], gpus: list[str], args, phase: str) -> None:
    """Run `jobs` in one worker process per GPU; all jobs of one input go to the same GPU."""
    pending = [job for job in jobs if not result_path(args.out, job).is_file()]
    if not pending:
        return
    by_input: dict[str, list[dict]] = {}
    for job in pending:
        by_input.setdefault(job["input"], []).append(job)
    buckets: list[list[dict]] = [[] for _ in gpus]
    for name in sorted(by_input, key=lambda key: -len(by_input[key])):
        min(buckets, key=len).extend(by_input[name])
    job_dir = args.out / "_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu, bucket in zip(gpus, buckets):
        if not bucket:
            continue
        job_file = job_dir / f"{phase}_gpu{gpu}.json"
        job_file.write_text(json.dumps(bucket), encoding="utf-8")
        command = [sys.executable, "-m", "research.jepa_probe.train_probe", "--jobs", str(job_file),
                   "--features", str(args.features), "--index", str(args.index), "--out", str(args.out),
                   "--device", "cuda", "--train_cfg", json.dumps(args.train_cfg)]
        log = open(job_dir / f"{phase}_gpu{gpu}.log", "a", encoding="utf-8")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        processes.append((gpu, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"{phase}: gpu {gpu} runs {len(bucket)} jobs, log {log.name}", flush=True)
    failed = []
    for gpu, process, log in processes:
        if process.wait() != 0:
            failed.append(gpu)
        log.close()
    if failed:
        raise SystemExit(f"{phase}: workers on gpus {failed} failed; see {job_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, help="sweep YAML: tasks, inputs, lrs, seeds, p4, train")
    parser.add_argument("--gpus", help="comma-separated GPU ids; one worker process each")
    parser.add_argument("--jobs", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--train_cfg", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--input")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if torch is None:
        raise SystemExit("train_probe needs torch")

    sweep = {}
    if args.sweep:
        import yaml
        sweep = yaml.safe_load(args.sweep.read_text(encoding="utf-8"))
    defaults = {"epochs": 40, "batch_size": 256, "patience": 5}
    args.train_cfg = {**defaults, **sweep.get("train", {}), **(json.loads(args.train_cfg) if args.train_cfg else {})}

    if args.jobs:  # worker process started by dispatch
        run_jobs(json.loads(args.jobs.read_text(encoding="utf-8")),
                 Context(args.features, args.index, args.out, args.device, args.train_cfg))
        return 0
    if args.task:
        if not args.input:
            parser.error("--task needs --input")
        job = {"task": args.task, "input": args.input, "lr": None if args.input == "B-mean" else args.lr,
               "seed": args.seed}
        run_jobs([job], Context(args.features, args.index, args.out, args.device, args.train_cfg))
        return 0
    if not sweep:
        parser.error("give --sweep or --task/--input")

    main_jobs = sweep_jobs(sweep)
    p4 = sweep.get("p4")
    gpus = args.gpus.split(",") if args.gpus else None
    if gpus:
        dispatch(main_jobs, gpus, args, "main")
    else:
        run_jobs(main_jobs, Context(args.features, args.index, args.out, args.device, args.train_cfg))
    if p4:
        chosen = best_inputs(args.out, p4["task"], p4["candidates"], int(p4.get("top", 2)))
        pairs = [f"{p4['base']}+{name}" for name in chosen]
        print(f"P4 on {p4['task']}: {pairs}", flush=True)
        p4_jobs = sweep_jobs(sweep, inputs=pairs, tasks=[p4["task"]])
        if gpus:
            dispatch(p4_jobs, gpus, args, "p4")
        else:
            run_jobs(p4_jobs, Context(args.features, args.index, args.out, args.device, args.train_cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
