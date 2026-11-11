"""T1.2: load each teacher with its real weights and check the shape contract.

  V0 (dino_video)  random clip [2, 3, 3, 256, 256] -> current and future [2, 256, 1024], no NaN;
                   the runner path (uint8 frames -> get_video_target) gives the same shapes. GPU only.
  hf_vjepa2        last_hidden_state is [B, 2 x 256, 1024] for a 4-frame clip, and the wrapper's
                   current / future are its tubelet 0 / tubelet 1; a middle layer (V1m) is
                   LayerNorm-ed. Also runs on CPU.
  vjepa2_1         strict load of the extracted ema_encoder; 512 tokens at 256 px and 1152 at 384.
                   Also runs on CPU.

  python -m research.jepa_probe.selfcheck_teacher --variants V0,V1,V1m,V3 --device cuda \
      --out $R/jepa/s1_teacher_selfcheck.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

EMBED_DIM = 1024


def _finite(*tensors) -> bool:
    import torch
    return all(bool(torch.isfinite(t.float()).all()) for t in tensors)


def check_dino_video(variant: dict, device: str) -> list[str]:
    import torch

    from research.jepa_probe.teachers import build_runner

    runner = build_runner(variant, device)
    clip = torch.randn(2, 3, 3, 256, 256, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        future, current = runner.teacher.get_future_feature(clip, return_current=True, current_index=1,
                                                            fps=torch.full((2,), 50 / 49, device=device))
    assert future.shape == current.shape == (2, 256, EMBED_DIM), (future.shape, current.shape)
    assert _finite(future, current), "NaN or inf in DINO-Video tokens"
    frames = torch.randint(0, 256, (2, len(runner.positions), 3, 224, 224), dtype=torch.uint8, device=device)
    cur, fut = runner(frames, torch.full((2,), 50 / 49, device=device))
    assert cur.shape == fut.shape == (2, 256, EMBED_DIM) and _finite(cur, fut)
    return [f"get_future_feature -> {tuple(future.shape)} x2; runner -> {tuple(cur.shape)} x2"]


def check_hf_vjepa2(variant: dict, device: str, batch: int) -> list[str]:
    import torch

    from lingbotvla.models.vla.vision_models.jepa.teacher import VJepaTeacher

    teacher = VJepaTeacher.from_hf(variant["weights"], dtype=torch.float32).to(device)
    video = torch.randn(batch, 3, 4, 256, 256, device=device)
    with torch.no_grad():
        outputs = teacher.encoder(pixel_values_videos=video.permute(0, 2, 1, 3, 4).contiguous(),
                                  skip_predictor=True, output_hidden_states=True)
    hidden = outputs.last_hidden_state
    assert hidden.shape == (batch, 512, EMBED_DIM), hidden.shape
    future, current = teacher.get_future_feature(video, return_current=True, current_index=0)
    assert torch.allclose(current, hidden[:, :256], atol=1e-4), "current is not tubelet 0"
    assert torch.allclose(future, hidden[:, 256:], atol=1e-4), "future is not tubelet 1"
    notes = [f"last_hidden_state {tuple(hidden.shape)}; current == [:, :256], future == [:, 256:]",
             f"{len(outputs.hidden_states) - 1} layers"]
    layer = variant.get("layer", "last")
    if layer != "last":
        middle, _ = teacher.get_future_feature(video, return_current=True, hidden_layer=int(layer))
        assert middle.shape == (batch, 256, EMBED_DIM) and _finite(middle)
        mean, std = float(middle.mean(-1).abs().max()), float(middle.std(-1).mean())
        assert mean < 1e-3 and abs(std - 1) < 1e-2, (mean, std)
        notes.append(f"layer {layer}: {tuple(middle.shape)}, per-token mean ~0, std {std:.3f}")
    return notes


def check_vjepa2_1(variant: dict, device: str, batch: int) -> list[str]:
    import torch

    from lingbotvla.models.vla.vision_models.jepa.teacher import VJepaTeacher

    teacher = VJepaTeacher.from_vjepa2_1(variant["source"], variant["weights"],
                                         img_size=int(variant.get("img_size", 384))).to(device)
    notes = ["strict load of ema_encoder ok"]
    for size in (256, 384):
        with torch.no_grad():
            tokens = teacher._tokens(torch.randn(batch, 3, 4, size, size, device=device), None)
        expected = 2 * (size // 16) ** 2
        assert tokens.shape == (batch, expected, EMBED_DIM), (size, tokens.shape)
        assert _finite(tokens), f"NaN or inf at {size}"
        notes.append(f"{size} px -> {tokens.shape[1]} tokens")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="V0,V1,V1m,V3")
    parser.add_argument("--variants_cfg", type=Path, default=Path("research/jepa_probe/configs/variants.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=None, help="default 2 on GPU, 1 on CPU")
    parser.add_argument("--pylib", default=os.path.join(os.environ.get("D", "/mnt/work"), "pylib"),
                        help="extra site dir with timm for the V-JEPA 2.1 code")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if os.path.isdir(args.pylib) and args.pylib not in sys.path:
        sys.path.append(args.pylib)

    from research.jepa_probe.extract_features import load_variant

    batch = args.batch or (2 if args.device.startswith("cuda") else 1)
    checks = {"dino_video": lambda v: check_dino_video(v, args.device),
              "hf_vjepa2": lambda v: check_hf_vjepa2(v, args.device, batch),
              "vjepa2_1": lambda v: check_vjepa2_1(v, args.device, batch)}
    lines, failed = [f"teacher self-check, device {args.device}, commit {os.environ.get('LB_COMMIT')}"], []
    for name in args.variants.split(","):
        started = time.time()
        try:
            variant = load_variant(args.variants_cfg, name)
            notes = checks[variant["teacher"]](variant)
            lines.append(f"PASS {name} ({time.time() - started:.0f}s): " + "; ".join(notes))
        except Exception as exc:  # report every variant, then fail at the end
            failed.append(name)
            lines.append(f"FAIL {name}: {type(exc).__name__}: {exc}")
        print(lines[-1], flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
