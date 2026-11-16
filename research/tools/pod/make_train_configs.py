#!/usr/bin/env python3
"""Write LingBot training configs for the TAIP Pod, starting from the official configs/vla/robotwin/robotwin.yaml.

Only weight/data paths, batch sizes and step counts change. LingBot does not derive gradient accumulation:
global_batch_size must equal micro_batch_size x GPUs x gradient_accumulation_steps or arguments.py raises, so the
official file (global 1024 = micro 32 x 32 GPUs) fails as-is on 4 GPUs. Every config written here sets all three.

  python make_train_configs.py smoke                       # S1, S2, S3, S3b, S3c, S5 into $D/configs
  python make_train_configs.py smoke --train_list $D/lists/pilot.txt
  python make_train_configs.py formal --name train_c50_v1 --micro_batch 32 --accum 2 --max_steps 12000 --save_steps 3000
  python make_train_configs.py smoke --arm B1               # the same smoke set with the arm applied, *_b1.yaml
  python make_train_configs.py formal --arm B1 --seed 43 --ac_weight 0.002 --micro_batch 64 --accum 1 \
      --max_steps 10735 --save_steps 2147                   # name defaults to jepa_B1_s43

Arms: A0 is the official recipe. B0 and B1 add the action-conditioned head (align_params.action_cond), fed zero
actions (B0) or the true action chunk (B1).

Paths default to the pod.env variables D, W, CFG, LB; --no_check skips the existence checks.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import yaml

ROBOT = "robotwin_c50"
ARMS = ("A0", "B0", "B1")


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def base_config(lb: Path, weights: Path, cfg_root: Path, train_list: Path) -> dict:
    base = yaml.safe_load((lb / "configs/vla/robotwin/robotwin.yaml").read_text(encoding="utf-8"))
    model, data, train = base["model"], base["data"], base["train"]
    depth, video = train["align_params"]["depth"], train["align_params"]["video"]
    model["model_path"] = str(weights / "lingbot-vla-v2-6b")
    model["tokenizer_path"] = str(weights / "Qwen3-VL-4B-Instruct")
    depth["moge_path"] = str(weights / "moge-2-vitb-normal/model.pt")
    depth["morgbd_path"] = str(weights / "lingbot-vla-v2-6b/depth/model.pt")
    video["ckpt_path"] = str(weights / "lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth")
    video["config_path"] = str(weights / "lingbot-vla-v2-6b/dino_video/config.yaml")
    data.update(data_name="multi", train_path=str(train_list), robot_config_root=str(cfg_root / "robot_configs"),
                norm_stats_file=str(cfg_root / "norm_stats" / f"{ROBOT}.json"))
    return base


def missing_inputs(cfg: dict) -> list[str]:
    model, data, train = cfg["model"], cfg["data"], cfg["train"]
    depth, video = train["align_params"]["depth"], train["align_params"]["video"]
    paths = [model["model_path"], model["tokenizer_path"], depth["moge_path"], depth["morgbd_path"],
             video["ckpt_path"], video["config_path"], data["train_path"], data["norm_stats_file"],
             str(Path(data["robot_config_root"]) / f"{ROBOT}.yaml")]
    return [p for p in paths if not Path(p).exists()]


def variant(base: dict, gpus: int, micro: int, accum: int, max_steps: int, output_dir: Path,
            train: dict | None = None, model: dict | None = None, video: dict | None = None) -> dict:
    cfg = copy.deepcopy(base)
    cfg["train"].update(micro_batch_size=micro, gradient_accumulation_steps=accum,
                        global_batch_size=micro * gpus * accum, max_steps=max_steps, output_dir=str(output_dir))
    cfg["train"].update(train or {})
    cfg["model"].update(model or {})
    cfg["train"]["align_params"]["video"].update(video or {})
    return cfg


def apply_arm(cfg: dict, arm: str, ac_weight: float | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    if arm == "A0" and ac_weight is not None:
        raise ValueError("--ac_weight only applies to the B arms")
    ac = cfg["train"]["align_params"].setdefault("action_cond", {})
    ac["enabled"] = arm != "A0"
    if arm != "A0":
        ac["use_actions"] = arm == "B1"
        if ac_weight is not None:
            ac["loss_weight"] = ac_weight
    return cfg


def smoke_configs(base: dict, smoke_dir: Path) -> dict[str, tuple[int, dict]]:
    reference = dict(train={"use_compile": False, "attention_implementation": "eager"},
                     model={"vit_attn_implementation": "sdpa"}, video={"attention_mode": "sdpa_block_causal"})
    return {
        "smoke_s1": (1, variant(base, 1, 8, 1, 20, smoke_dir / "s1")),
        "smoke_s2_ref": (1, variant(base, 1, 8, 1, 20, smoke_dir / "s2", **reference)),
        "smoke_s3": (4, variant(base, 4, 32, 2, 30, smoke_dir / "s3")),
        "smoke_s3b": (4, variant(base, 4, 64, 1, 30, smoke_dir / "s3b")),
        "smoke_s3c_distmuon": (4, variant(base, 4, 64, 1, 30, smoke_dir / "s3c", train={"optimizer": "dist_muon"})),
        "smoke_s5_resume": (1, variant(base, 1, 8, 1, 40, smoke_dir / "s1")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("smoke", "formal"))
    ap.add_argument("--lingbot_root", type=Path, default=_env_path("LB"))
    ap.add_argument("--weights", type=Path, default=_env_path("W"))
    ap.add_argument("--cfg", type=Path, default=_env_path("CFG"), help="dir with robot_configs/ and norm_stats/")
    ap.add_argument("--work", type=Path, default=_env_path("D"), help="work PVC root (configs/, smoke/, runs/)")
    ap.add_argument("--train_list", type=Path, default=None, help="default: <work>/lists/train_c50.txt")
    ap.add_argument("--out_dir", type=Path, default=None, help="default: <work>/configs")
    ap.add_argument("--name", default=None,
                    help="formal: config name, also the run directory name (default with --arm: jepa_<arm>_s<seed>)")
    ap.add_argument("--arm", choices=ARMS, default=None, help="experiment arm; omitted = A0 without renaming")
    ap.add_argument("--seed", type=int, default=None, help="train.seed (LingBot default 42)")
    ap.add_argument("--ac_weight", type=float, default=None, help="B arms: align_params.action_cond.loss_weight")
    ap.add_argument("--gpus", type=int, default=4, help="formal: GPUs the run will use")
    ap.add_argument("--micro_batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--save_steps", type=int, default=None)
    ap.add_argument("--no_check", action="store_true", help="do not require weights, list and norm stats to exist")
    args = ap.parse_args(argv)

    for flag in ("lingbot_root", "weights", "cfg", "work"):
        if getattr(args, flag) is None:
            ap.error(f"--{flag} not given and its pod.env variable is unset (source pod.env first)")
    train_list = args.train_list or args.work / "lists" / "train_c50.txt"
    out_dir = args.out_dir or args.work / "configs"
    base = base_config(args.lingbot_root, args.weights, args.cfg, train_list)

    arm = args.arm or "A0"
    if args.ac_weight is not None and arm == "A0":
        ap.error("--ac_weight only applies to --arm B0/B1")
    if args.mode == "smoke":
        configs = smoke_configs(base, args.work / "smoke")
        if arm != "A0":
            suffix = arm.lower()
            for _, cfg in configs.values():
                cfg["train"]["output_dir"] += f"_{suffix}"
            configs = {f"{name}_{suffix}": entry for name, entry in configs.items()}
    else:
        if args.name is None and args.arm is not None:
            args.name = f"jepa_{arm}_s{42 if args.seed is None else args.seed}"
        needed = ("name", "micro_batch", "accum", "max_steps", "save_steps")
        missing = [f"--{k}" for k in needed if getattr(args, k) is None]
        if missing:
            ap.error(f"formal needs {' '.join(missing)}")
        cfg = variant(base, args.gpus, args.micro_batch, args.accum, args.max_steps, args.work / "runs" / args.name,
                      train={"save_steps": args.save_steps})
        configs = {args.name: (args.gpus, cfg)}
    for _, cfg in configs.values():
        apply_arm(cfg, arm, args.ac_weight)
        if args.seed is not None:
            cfg["train"]["seed"] = args.seed

    if not args.no_check:
        absent = missing_inputs(base)
        if absent:
            print("missing inputs:\n  " + "\n  ".join(absent), file=sys.stderr)
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (gpus, cfg) in configs.items():
        t = cfg["train"]
        assert t["global_batch_size"] == t["micro_batch_size"] * gpus * t["gradient_accumulation_steps"]
        path = out_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        ac = t["align_params"].get("action_cond") or {}
        arm_note = (f", action_cond use_actions={ac['use_actions']} weight={ac.get('loss_weight')}"
                    if ac.get("enabled") else "")
        print(f"{path}: {gpus} GPU x mb {t['micro_batch_size']} x accum {t['gradient_accumulation_steps']} = "
              f"global {t['global_batch_size']}, max_steps {t['max_steps']}, output_dir {t['output_dir']}{arm_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
