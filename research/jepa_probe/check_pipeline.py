"""Steps 1 and 2 of the Stage 1 plan: the probe's labels, pixels and V0 targets against LingBot's
own training data path.

  data   CPU (lingbot-data). For --samples random index samples, LingBot's VLADataset (same robot
         config, norm stats and data settings as training) builds the item the training loop sees:
           labels  normalized action chunk and state vs labels.npz, max abs error <= 1e-5
           rows    episode_index / frame_index match the index, no padded action steps
           pixels  pil_images and future_pil_images of the top camera vs frames.FrameReader, bit-equal;
                   with data.future_video_mid_frames also the mid frames the V-JEPA clip reads
           fps     the item carries no future_video_effective_fps, so the teacher uses the config's
           config  the --variant's teacher settings in variants.yaml vs the training config (V0: the
                   DINO-Video keys; a V-JEPA variant: clip frames, input size, layer, weights dir);
                   norm stats vs the index
  video  GPU (lingbot-train). --batches x --batch train samples. Target A is get_video_target on the
         dataset's pil_images (and mid frames), called the way the training loop calls it, with the
         teacher the training config builds; target B is the --variant runner on the frames
         extract_features feeds it. Per-token cosine >= 0.999, relative L2 <= 1%. For a V-JEPA
         variant (plan/03, arm A2) it also reports the target scale against DINO-Video on the same
         samples: S = mean squared deviation from the per-token, per-dim sample mean, and the
         suggested video.mse_loss_weight S_dino / S_jepa.

The Qwen image processor and tokenizer are skipped: they do not touch pil_images. Items are read
through research/tools/pod/agx_run.py's query patch, as training runs.

  python -m research.jepa_probe.check_pipeline data --index $P/index --out $R/jepa/s1_check_data.json
  python -m research.jepa_probe.check_pipeline video --index $P/index --out $R/jepa/s1_check_video.json
  python -m research.jepa_probe.check_pipeline data --variant V2 --train_config $D/configs/jepa_A2_s42.yaml ...
  python -m research.jepa_probe.check_pipeline video --variant V2 --train_config $D/configs/jepa_A2_s42.yaml ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from research.jepa_probe.common import load_index, read_json, split_ids

TOP_IMAGE = "observation.images.camera_top"
VIDEO_KEYS = ("attention_mode", "input_size", "num_future_frames", "use_warmup_frame", "effective_fps",
              "n_blocks", "cls_pool", "use_patch_loss", "use_current_patch_loss", "use_cls_loss")
LABEL_TOL = 1e-5
COSINE_MIN = 0.999
REL_L2_MAX = 0.01


def load_train_config(path: Path) -> tuple[SimpleNamespace, dict, int]:
    """Data arguments as tasks/vla/train_lingbotvla.py parses them, its video config and chunk size."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = dict(cfg["data"])
    for key in ("joints", "cameras", "norm_type"):  # parse_args passes list items through str()
        if data.get(key) is not None:
            data[key] = [str(item) for item in data[key]]
    for key, default in (("img_size", 256), ("image_augment", False), ("use_future_image", False)):
        data.setdefault(key, default)  # MyDataArguments / DataArguments defaults
    train = cfg.get("train", {})
    return SimpleNamespace(**data), dict(train["align_params"]["video"]), int(train.get("chunk_size", 50))


def feature_order(vectors: np.ndarray) -> np.ndarray:
    """14-dim raw layout -> LingBot's [arm.position (12), effector.position (2)] order."""
    return np.concatenate([vectors[..., 0:6], vectors[..., 7:13], vectors[..., 6:7], vectors[..., 13:14]], -1)


class TrainingItems:
    """Training items from LingBot's VLADataset; one dataset is open at a time."""

    def __init__(self, data_args: SimpleNamespace, robot_config_root: Path, robot_name: str, chunk: int):
        import research.tools.pod.agx_run  # noqa: F401  (patches the chunk query, values unchanged)
        from lingbotvla.data.vla_data.base_dataset import VLADataset

        self._cls = VLADataset
        self.data_args = data_args
        self.robot_config_root = str(robot_config_root)
        self.robot_name = robot_name
        self.chunk = chunk
        self._open: tuple[str | None, object] = (None, None)

    def dataset(self, directory: str):
        if self._open[0] != directory:
            size = int(self.data_args.img_size)
            vla = self._cls(directory, self.robot_name, self.data_args, self.robot_config_root,
                            config=SimpleNamespace(), processor=None, chunk_size=self.chunk,
                            image_size=(size, size), do_nomalize=True, return_item=True,
                            image_augment=bool(self.data_args.image_augment), use_depth_align=True,
                            use_future_image=bool(self.data_args.use_future_image))
            self._open = (directory, vla)
        return self._open[1]

    def build(self, directory: str, row: int) -> dict:
        """The normalized item, plus pil_images / future_pil_images [cameras, 3, H, W] uint8 and the
        effective fps as FeatureTransform.apply would put them in the batch."""
        from lingbotvla.data.vla_data.transform import prepare_images

        vla = self.dataset(directory)
        transform = vla.feature_transform
        item = vla.getitem(row)  # return_item=True stops right after normalization
        batch = transform.pad_and_concat(dict(item))
        keys = transform.feature_config.images
        train = bool(transform.image_augment)
        _, _, pil, _, params = prepare_images(None, batch, image_keys=keys, train=train, use_depth_align=True,
                                              return_augment_params=True)
        future = None
        if transform.use_future_image and len(batch.get("future_image", {})) > 0:
            _, _, future, _ = prepare_images(None, {**batch, "image": batch["future_image"]}, image_keys=keys,
                                             train=train, use_depth_align=True, augment_params=params)
        mid = batch.get("future_video_mid_image")
        if mid is not None:
            mid = transform.augment_future_video_mid_images(mid, params)
        return {"item": item, "pil_images": pil, "future_pil_images": future, "future_video_mid_pil_images": mid,
                "image_keys": list(keys), "effective_fps": batch.get("future_video_effective_fps")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pick(ids: np.ndarray, count: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(ids, size=min(count, len(ids)), replace=False))


def _video_mismatch(ours: dict, training: dict) -> dict:
    return {key: [ours.get(key), training.get(key)] for key in VIDEO_KEYS if ours.get(key) != training.get(key)}


def mid_frames(data_args) -> list[int]:
    return [int(o) for o in (getattr(data_args, "future_video_mid_frames", None) or [])]


def is_jepa(video_cfg: dict) -> bool:
    return video_cfg.get("teacher_type", "dino_video") == "vjepa2"


def teacher_mismatch(variant: dict, video_cfg: dict, mids: list[int], chunk: int) -> dict:
    """Differences between a probe variant's teacher and the one the training config builds."""
    if variant["teacher"] == "dino_video":
        if is_jepa(video_cfg):
            return {"teacher_type": ["dino_video", video_cfg.get("teacher_type")]}
        return _video_mismatch(variant["video"], video_cfg)
    if variant.get("future_clip_frames") or variant["teacher"] != "hf_vjepa2":
        return {"teacher": [variant["teacher"], "training builds hf_vjepa2 on one clip"]}
    clip = [0, 0, *mids, chunk - 1]
    if len(clip) % 2:
        clip.append(chunk - 1)
    pairs = {"teacher_type": ("vjepa2", video_cfg.get("teacher_type")),
             "clip_frames": (list(variant["clip_frames"]), clip),
             "input_size": (int(variant["input_size"]), int(video_cfg.get("input_size", 256))),
             "current_index": (int(variant.get("current_index", 0)), 0),
             "layer": (variant.get("layer", "last"), "last"),
             "weights": (Path(variant["weights"]).name, Path(str(video_cfg.get("jepa_model_path"))).name)}
    return {key: list(pair) for key, pair in pairs.items() if pair[0] != pair[1]}


def target_scale(targets: list) -> float:
    """Mean squared deviation of [N, tokens, dim] targets from their per-token, per-dim mean over samples."""
    import torch

    values = torch.cat(targets).double()
    return float((values - values.mean(0, keepdim=True)).square().mean())


def check_data(args) -> dict:
    from research.jepa_probe.frames import FrameReader

    index = load_index(args.index)
    labels = np.load(args.index / "labels.npz")
    meta = read_json(args.index / "index_meta.json")
    data_args, video_cfg, chunk = load_train_config(args.train_config)
    if chunk != int(meta["chunk"]) or int(meta["future_offset"]) != chunk - 1:
        raise ValueError(f"index built with chunk {meta['chunk']} / offset {meta['future_offset']}, training chunk {chunk}")
    items = TrainingItems(data_args, args.robot_config_root, args.robot_name, chunk)
    mids = mid_frames(data_args)

    action_err = state_err = 0.0
    bad_rows, bad_pixels, fps_values, top_positions, shapes = [], [], set(), set(), set()
    reader: tuple[str | None, FrameReader | None] = (None, None)
    for sample in _pick(np.arange(len(index["sample_id"])), args.samples, args.seed):
        directory = str(index["dataset_dir"][sample])
        row_t, row_future = int(index["row_t"][sample]), int(index["row_future"][sample])
        built = items.build(directory, row_t)
        item = built["item"]
        action = np.concatenate([item["action.arm.position"].numpy(), item["action.effector.position"].numpy()], -1)
        state = np.concatenate([item["observation.state.arm.position"].numpy(),
                                item["observation.state.effector.position"].numpy()], -1)
        action_err = max(action_err, float(np.abs(action - feature_order(labels["action_chunk"][sample])).max()))
        state_err = max(state_err, float(np.abs(state - feature_order(labels["state_t"][sample])).max()))
        if (int(item["episode_index"]) != int(index["episode_index"][sample])
                or int(item["frame_index"]) != int(index["t"][sample]) or bool(item["action_is_pad"].any())):
            bad_rows.append(int(sample))

        if reader[0] != directory:
            reader = (directory, FrameReader(Path(directory)))
        frames = reader[1].load([row_t, *[row_t + o for o in mids], row_future])
        top = built["image_keys"].index(TOP_IMAGE)
        top_positions.add(top)
        current, future = built["pil_images"][top], built["future_pil_images"][top]
        mid = built["future_video_mid_pil_images"]
        shapes.add(tuple(current.shape))
        mid_ok = (mid is None and not mids) or (mid is not None and mid.dtype == frames.dtype
                                                and mid.shape == frames[1:-1].shape and bool((mid == frames[1:-1]).all()))
        if not (current.dtype == frames.dtype and current.shape == frames[0].shape
                and bool((current == frames[0]).all()) and bool((future == frames[-1]).all()) and mid_ok):
            bad_pixels.append(int(sample))
        fps = built["effective_fps"]
        fps_values.add(None if fps is None else round(float(fps), 6))

    from research.jepa_probe.extract_features import load_variant

    mismatch = teacher_mismatch(load_variant(args.variants_cfg, args.variant), video_cfg, mids, chunk)
    robot_cfg = yaml.safe_load((Path(args.robot_config_root) / f"{args.robot_name}.yaml").read_text(encoding="utf-8"))
    norm_sha = _sha256(Path(robot_cfg["norm_stats"]))
    checks = {
        "labels": {"pass": action_err <= LABEL_TOL and state_err <= LABEL_TOL,
                   "max_abs_action": action_err, "max_abs_state": state_err},
        "rows": {"pass": not bad_rows, "bad_samples": bad_rows[:20]},
        "pixels": {"pass": not bad_pixels and top_positions == {0}, "bad_samples": bad_pixels[:20],
                   "top_camera_position": sorted(top_positions), "shape": sorted(shapes), "mid_frames": mids,
                   "img_size": int(data_args.img_size), "image_augment": bool(data_args.image_augment)},
        "fps": {"pass": fps_values == {None}, "batch_values": sorted(fps_values, key=str),
                "teacher_fps": video_cfg.get("effective_fps")},
        "config": {"pass": not mismatch and norm_sha == meta["sha256"]["norm_stats"],
                   "variant": args.variant, "teacher_mismatch": mismatch,
                   "norm_stats_matches_index": norm_sha == meta["sha256"]["norm_stats"]},
    }
    return {"mode": "data", "samples": min(args.samples, len(index["sample_id"])), "checks": checks}


def check_video(args) -> dict:
    import torch
    import torch.nn.functional as F

    from lingbotvla.models.vla.vision_models.module_utils import build_video_model, get_video_target
    from research.jepa_probe.extract_features import ShardFrames, load_variant
    from research.jepa_probe.teachers import build_runner

    index = load_index(args.index)
    meta = read_json(args.index / "index_meta.json")
    data_args, video_cfg, chunk = load_train_config(args.train_config)
    items = TrainingItems(data_args, args.robot_config_root, args.robot_name, chunk)
    device = torch.device(args.device)
    variant = load_variant(args.variants_cfg, args.variant)
    mids = mid_frames(data_args)
    mismatch = teacher_mismatch(variant, video_cfg, mids, chunk)
    if mismatch:
        raise ValueError(f"variant {args.variant} does not match the training teacher: {mismatch}")
    runner = build_runner(variant, device)
    jepa = is_jepa(video_cfg)
    teacher = build_video_model(dict(video_cfg, device=str(device))) if jepa else runner.teacher
    dino = build_runner(load_variant(args.variants_cfg, "V0"), device) if jepa else None
    columns = [runner._column(0), *[runner._column(o) for o in mids], runner._column(chunk - 1)]
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    ids = _pick(split_ids(index, "train"), args.batches * args.batch, args.seed)

    cosines, sq_diff, sq_ref, max_abs, pixels_equal = [], 0.0, 0.0, 0.0, True
    scale = {"dino_current": [], "dino_future": [], "jepa_current": [], "jepa_future": []}
    for start in range(0, len(ids), args.batch):
        chunk_ids = ids[start:start + args.batch]
        built = [items.build(str(index["dataset_dir"][s]), int(index["row_t"][s])) for s in chunk_ids]
        pil = torch.stack([b["pil_images"] for b in built]).to(device)
        future_pil = torch.stack([b["future_pil_images"] for b in built]).to(device)
        mid = torch.stack([b["future_video_mid_pil_images"] for b in built]).to(device) if mids else None
        fps = None if any(b["effective_fps"] is None for b in built) else \
            torch.stack([torch.as_tensor(b["effective_fps"]) for b in built]).to(device)
        with torch.no_grad(), torch.autocast(**autocast):
            bundle = get_video_target(teacher, pil, future_pil, video_cfg, effective_fps=fps, mid_pil_images=mid)
        target_a = (bundle["current_patch"].float(), bundle["patch"].float())

        _, frames, frame_fps = next(iter(ShardFrames(chunk_ids, index, runner.positions, len(chunk_ids),
                                                     int(meta["future_offset"]))))
        clip = [pil[:, 0], *([] if mid is None else mid.unbind(1)), future_pil[:, 0]]
        pixels_equal &= all(bool(torch.equal(frames[:, c], x.cpu())) for c, x in zip(columns, clip))
        target_b = runner(frames.to(device), frame_fps.to(device))
        if dino is not None:
            with torch.no_grad(), torch.autocast(**autocast):
                ref = get_video_target(dino.teacher, pil, future_pil, dino.video_cfg)
            scale["dino_current"].append(ref["current_patch"].float().cpu())
            scale["dino_future"].append(ref["patch"].float().cpu())
            scale["jepa_current"].append(target_a[0].cpu())
            scale["jepa_future"].append(target_a[1].cpu())
        for a, b in zip(target_a, target_b):
            cosines.append(F.cosine_similarity(a, b, dim=-1).flatten().cpu())
            sq_diff += float((a - b).square().sum())
            sq_ref += float(a.square().sum())
            max_abs = max(max_abs, float((a - b).abs().max()))

    cosine = torch.cat(cosines)
    rel_l2 = (sq_diff / max(sq_ref, 1e-12)) ** 0.5
    result = {"pass": bool(cosine.mean() >= COSINE_MIN and rel_l2 <= REL_L2_MAX and pixels_equal),
              "cosine_mean": float(cosine.mean()), "cosine_min": float(cosine.min()),
              "cosine_p01": float(cosine.quantile(0.01)), "rel_l2": rel_l2, "max_abs": max_abs,
              "identical": max_abs == 0.0, "pixels_equal": pixels_equal}
    checks = {f"{args.variant.lower()}_target": result}
    if dino is not None:
        s = {key: target_scale(values) for key, values in scale.items()}
        ratios = {part: s[f"dino_{part}"] / s[f"jepa_{part}"] for part in ("current", "future")}
        checks["scale"] = {"pass": True, **{f"S_{key}": value for key, value in s.items()},
                           **{f"ratio_{part}": value for part, value in ratios.items()},
                           "suggested_mse_loss_weight": float(f"{sum(ratios.values()) / 2:.2g}")}
    return {"mode": "video", "samples": len(ids), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("data", "video"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--train_config", type=Path, default=Path("configs/vla/robotwin/robotwin.yaml"),
                        help="training config; only its data section and align_params.video are read")
    parser.add_argument("--robot_config_root", type=Path,
                        default=Path(os.environ.get("CFG", "/mnt/work/lingbot_cfg")) / "robot_configs")
    parser.add_argument("--robot_name", default="robotwin_c50", help="first column of the train list")
    parser.add_argument("--variants_cfg", type=Path, default=Path("research/jepa_probe/configs/variants.yaml"))
    parser.add_argument("--variant", default="V0", help="probe variant the training teacher must match (V2 for arm A2)")
    parser.add_argument("--samples", type=int, default=50, help="data mode")
    parser.add_argument("--batches", type=int, default=4, help="video mode")
    parser.add_argument("--batch", type=int, default=16, help="video mode")
    parser.add_argument("--device", default="cuda", help="video mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    report = check_data(args) if args.mode == "data" else check_video(args)
    report["pass"] = all(check["pass"] for check in report["checks"].values())
    report["code_commit"] = os.environ.get("LB_COMMIT")
    for name, check in report["checks"].items():
        details = ", ".join(f"{k}={v}" for k, v in check.items() if k != "pass")
        print(f"{'PASS' if check['pass'] else 'FAIL'} {name}: {details}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
