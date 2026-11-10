"""Attention probe for P1, P1s, P1-delta and P2 (plan section 2.4).

Tokens are channel-wise z-scored with the train-split stats from extract_features, then a 2-layer
Transformer (width 256, 4 heads, one learned pooling query) predicts the target. The learning rate
is chosen on the val split and the test nMSE is reported over seeds.

  python -m research.jepa_probe.train_probe --task P1s --variant V0 \
      --features $P/features/V0 --index $P/index --out $P/results --seeds 0,1,2 --lrs 3e-4,1e-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # index building and the ridge probe never need torch
    torch = None
    nn = None

TASKS = {
    # target: which label array; extra inputs: state token or not; frames used.
    "P1": {"target": "action_chunk", "state": False, "frames": ("cur", "fut")},
    "P1s": {"target": "action_chunk", "state": True, "frames": ("cur", "fut")},
    "P1d": {"target": "state_delta", "state": False, "frames": ("cur", "fut")},
    "P2": {"target": "state_t", "state": False, "frames": ("cur",)},
}


def build_attention_probe(in_dim: int, out_dim: int, d_model: int = 256, n_heads: int = 4,
                          n_layers: int = 2, dropout: float = 0.1, n_streams: int = 2):
    """The probe class is built on demand, so this module imports without torch."""

    class AttentionProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model))
            self.pos = nn.Parameter(torch.zeros(64, d_model))
            self.stream = nn.Parameter(torch.zeros(n_streams, d_model))
            self.state_proj = nn.Linear(14, d_model)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, n_layers)
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


def target_array(labels: dict, task: str) -> np.ndarray:
    spec = TASKS[task]
    if spec["target"] == "state_delta":
        values = labels["state_t49"] - labels["state_t"]
    else:
        values = labels[spec["target"]]
    return values.reshape(len(values), -1).astype(np.float32)


def nmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """MSE divided by the target variance: 0 is perfect, 1 matches predicting the mean."""
    var = np.mean(np.square(target - target.mean(axis=0)))
    return float(np.mean(np.square(prediction - target)) / max(var, 1e-12))


def normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """tokens [N, 64, C], mean/std [C] -> z-scored float32."""
    return ((tokens.astype(np.float32) - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def train_one(tokens: dict[str, np.ndarray], target: np.ndarray, state: np.ndarray | None, *,
              train_ids: np.ndarray, val_ids: np.ndarray, lr: float, seed: int, epochs: int = 40,
              batch_size: int = 256, patience: int = 5, device: str = "cpu") -> dict:
    """Train until the val nMSE stops improving. Returns val curve and the best prediction."""
    torch.manual_seed(seed)
    streams = len(tokens)
    model = build_attention_probe(next(iter(tokens.values())).shape[-1], target.shape[-1],
                                  n_streams=streams).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    inputs = {name: torch.from_numpy(values) for name, values in tokens.items()}
    target_tensor = torch.from_numpy(target)
    state_tensor = None if state is None else torch.from_numpy(state)
    best = {"val": float("inf"), "epoch": -1, "state": None}
    curve = []
    idle = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_ids))
        for start in range(0, len(train_ids), batch_size):
            batch = torch.from_numpy(train_ids)[order[start:start + batch_size]]
            stacked = torch.stack([inputs[name][batch] for name in tokens], dim=1).to(device)
            batch_state = None if state_tensor is None else state_tensor[batch].to(device)
            loss = nn.functional.mse_loss(model(stacked, batch_state), target_tensor[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        score = _eval_nmse(model, inputs, target_tensor, state_tensor, val_ids, target, device)
        curve.append(score)
        if score < best["val"]:
            best = {"val": score, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            idle = 0
        else:
            idle += 1
            if idle >= patience:
                break
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "val_curve": curve, "model": model}


def _eval_nmse(model, inputs, target_tensor, state_tensor, ids, target_np, device) -> float:
    model.eval()
    with torch.no_grad():
        index = torch.from_numpy(ids)
        stacked = torch.stack([inputs[name][index] for name in inputs], dim=1).to(device)
        state = None if state_tensor is None else state_tensor[index].to(device)
        prediction = model(stacked, state).cpu().numpy()
    return nmse(prediction, target_np[ids])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--lrs", default="3e-4,1e-3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if not (args.features / "stats.npz").is_file():
        raise SystemExit(f"no features at {args.features}; run extract_features first")
    print(f"probe {args.task} on {args.variant}: seeds {args.seeds}, lrs {args.lrs}, device {args.device}")
    raise SystemExit("train_probe: the training driver is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
