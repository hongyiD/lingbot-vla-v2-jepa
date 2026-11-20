"""check_pipeline against a dataset written by lerobot itself, read back by LingBot's VLADataset."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
try:
    import lingbotvla.data.vla_data.base_dataset  # noqa: F401
    import lingbotvla.models.vla.vision_models.module_utils as module_utils
except Exception as exc:
    pytest.skip(f"needs LingBot's data stack: {exc}", allow_module_level=True)

from research.jepa_probe import check_pipeline  # noqa: E402
from research.jepa_probe.build_index import build  # noqa: E402
from research.jepa_probe.tests.helpers import write_norm_stats  # noqa: E402

LB = Path(__file__).resolve().parents[3]
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
EPISODES, LENGTH = 3, 60


def _write_lerobot(root: Path) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image = {"dtype": "image", "shape": (48, 64, 3), "names": ["height", "width", "channels"]}
    features = {"observation.state": {"dtype": "float32", "shape": (14,), "names": None},
                "action": {"dtype": "float32", "shape": (14,), "names": None},
                **{f"observation.images.{camera}": image for camera in CAMERAS}}
    name = "click_bell-aloha-agilex_clean_50-50"
    dataset = LeRobotDataset.create(name, fps=50, features=features, root=root / name, robot_type="aloha",
                                    use_videos=False)
    rng = np.random.default_rng(0)
    for _ in range(EPISODES):
        for _ in range(LENGTH):
            frame = {"observation.state": rng.uniform(-1, 3, 14).astype(np.float32),
                     "action": rng.uniform(-1, 3, 14).astype(np.float32), "task": "click the bell"}
            for camera in CAMERAS:
                frame[f"observation.images.{camera}"] = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return root / name


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("check")
    dataset = _write_lerobot(root / "data")
    norm = root / "norm.json"
    write_norm_stats(norm)
    robot_root = root / "robot_configs"
    robot_root.mkdir()
    robot = (LB / "configs/robot_configs/robotwin.yaml").read_text(encoding="utf-8")
    robot = "\n".join(f"norm_stats: {norm}" if line.startswith("norm_stats:") else line for line in robot.splitlines())
    (robot_root / "robotwin_c50.yaml").write_text(robot + "\n", encoding="utf-8")
    train_list = root / "train.txt"
    train_list.write_text(f"robotwin_c50 {dataset}\n", encoding="utf-8")
    build(train_list, robot_root / "robotwin_c50.yaml", norm, root / "index", stride=5, future_offset=49, chunk=50,
          split_spec="train=0-0,val=1-1,test=2-2")
    return root


def _args(root: Path, mode: str, *extra: str) -> list[str]:
    return [mode, "--index", str(root / "index"), "--robot_config_root", str(root / "robot_configs"),
            "--train_config", str(LB / "configs/vla/robotwin/robotwin.yaml"),
            "--variants_cfg", str(LB / "research/jepa_probe/configs/variants.yaml"), *extra]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("D", str(tmp_path))
    monkeypatch.setenv("W", str(tmp_path / "weights"))


def test_data_check_passes(workspace):
    out = workspace / "data.json"
    assert check_pipeline.main(_args(workspace, "data", "--samples", "12", "--out", str(out))) == 0
    report = __import__("json").loads(out.read_text())
    assert report["checks"]["pixels"]["shape"] == [[3, 256, 256]]
    assert report["checks"]["fps"]["batch_values"] == [None]


def test_data_check_catches_a_different_image_size(workspace):
    config = (LB / "configs/vla/robotwin/robotwin.yaml").read_text(encoding="utf-8")
    changed = workspace / "train_224.yaml"
    changed.write_text(config.replace("  use_future_image: true", "  use_future_image: true\n  img_size: 224"),
                       encoding="utf-8")
    args = _args(workspace, "data", "--samples", "4")
    args[args.index("--train_config") + 1] = str(changed)
    assert check_pipeline.main(args) == 1


class _FakeVideoTeacher:
    """Stands in for DINO-Video: patch tokens are the 16x16 average-pooled frame."""

    def get_future_feature(self, video, return_cls=False, return_current=False, current_index=0, fps=None):
        assert fps is None
        def tokens(frame):
            return torch.nn.functional.adaptive_avg_pool2d(frame.float(), 16).flatten(2).transpose(1, 2)
        future = tokens(video[:, :, -1])
        return (future, tokens(video[:, :, current_index])) if return_current else future


def test_video_check_passes_with_identical_targets(workspace, monkeypatch):
    monkeypatch.setattr(module_utils, "build_video_model", lambda cfg: _FakeVideoTeacher())
    out = workspace / "video.json"
    args = _args(workspace, "video", "--batches", "2", "--batch", "3", "--device", "cpu", "--out", str(out))
    assert check_pipeline.main(args) == 0
    result = __import__("json").loads(out.read_text())["checks"]["v0_target"]
    assert result["identical"] and result["pixels_equal"]


# Arm A2 (plan/03): V-JEPA 2 on [t, t, t+8, ..., t+41, t+49], against the Stage 1 V2 variant.

def _a2_config(root: Path, weights: Path) -> Path:
    import yaml

    from research.tools.pod.make_train_configs import apply_arm

    cfg = yaml.safe_load((LB / "configs/vla/robotwin/robotwin.yaml").read_text(encoding="utf-8"))
    apply_arm(cfg, "A2", weights=weights)
    path = root / "train_a2.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _a2_args(root: Path, mode: str, *extra: str) -> list[str]:
    args = _args(root, mode, "--variant", "V2", *extra)
    args[args.index("--train_config") + 1] = str(_a2_config(root, root / "weights"))
    return args


def _a2_data_args(root: Path):
    data_args, video_cfg, chunk = check_pipeline.load_train_config(_a2_config(root, root / "weights"))
    return data_args, chunk


def test_a2_reads_mid_frames_for_the_top_camera_only(workspace):
    data_args, chunk = _a2_data_args(workspace)
    items = check_pipeline.TrainingItems(data_args, workspace / "robot_configs", "robotwin_c50", chunk)
    vla = items.dataset(str(workspace / "data" / "click_bell-aloha-agilex_clean_50-50"))
    delta = vla.get_video_delta_timestamps()
    assert [round(t * 50) for t in delta["observation.images.cam_high"]] == [0, 8, 16, 24, 33, 41, 49]
    assert all([round(t * 50) for t in delta[f"observation.images.{c}"]] == [0, 49] for c in CAMERAS[1:])
    built = items.build(str(workspace / "data" / "click_bell-aloha-agilex_clean_50-50"), 3)
    assert built["future_video_mid_pil_images"].shape == (5, 3, 256, 256)
    assert built["future_video_mid_pil_images"].dtype == torch.uint8
    assert built["future_pil_images"].shape == (3, 3, 256, 256)  # future depth still reads t+49 of every camera


def test_a2_data_check_passes(workspace, monkeypatch):
    monkeypatch.setenv("D", str(workspace))
    monkeypatch.setenv("W", str(workspace / "weights"))
    out = workspace / "data_a2.json"
    assert check_pipeline.main(_a2_args(workspace, "data", "--samples", "8", "--out", str(out))) == 0
    report = __import__("json").loads(out.read_text())
    assert report["checks"]["pixels"]["mid_frames"] == [8, 16, 24, 33, 41]
    assert report["checks"]["config"]["teacher_mismatch"] == {}


def test_a2_data_check_catches_wrong_mid_frames(workspace, monkeypatch):
    import yaml

    monkeypatch.setenv("D", str(workspace))
    args = _a2_args(workspace, "data", "--samples", "2")
    path = Path(args[args.index("--train_config") + 1])
    cfg = yaml.safe_load(path.read_text())
    cfg["data"]["future_video_mid_frames"] = [10, 20, 30, 40, 45]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert check_pipeline.main(args) == 1


def test_mid_frames_are_off_without_future_images(workspace):
    data_args, chunk = _a2_data_args(workspace)
    robot = str(workspace / "robot_configs" / "robotwin_c50.yaml")
    from lingbotvla.data.vla_data.utils import FeatureTransform

    deploy = FeatureTransform(robot, data_args, None, None, do_nomalize=False, chunk_size=chunk)
    assert deploy.future_video_mid_frames == []
    data_args.future_video_mid_frames = [8, 49]
    with pytest.raises(ValueError, match="future_video_mid_frames"):
        FeatureTransform(robot, data_args, None, None, do_nomalize=False, chunk_size=chunk, use_future_image=True)


def test_a2_video_check_matches_the_v2_runner_and_reports_scale(workspace, monkeypatch):
    from lingbotvla.models.vla.vision_models.jepa.teacher import ToyEncoder, VJepaTeacher

    def toy_from_hf(path, dtype=torch.bfloat16):
        assert Path(path).name == "vjepa2-vitl-fpc64-256"
        torch.manual_seed(0)
        return VJepaTeacher(ToyEncoder(), backend="hf_vjepa2")

    original = module_utils.build_video_model
    monkeypatch.setattr(VJepaTeacher, "_tokens", lambda self, video, hidden_layer: self.encoder(video))
    monkeypatch.setattr(VJepaTeacher, "from_hf", staticmethod(toy_from_hf))
    monkeypatch.setattr(module_utils, "build_video_model",
                        lambda cfg: original(cfg) if cfg.get("teacher_type") == "vjepa2" else _FakeVideoTeacher())
    monkeypatch.setenv("D", str(workspace))
    out = workspace / "video_a2.json"
    args = _a2_args(workspace, "video", "--batches", "2", "--batch", "3", "--device", "cpu", "--out", str(out))
    assert check_pipeline.main(args) == 0
    checks = __import__("json").loads(out.read_text())["checks"]
    # Same teacher on the same pixels; the gap is the training path's bf16 rounding of its targets.
    target = checks["v2_target"]
    assert target["pass"] and target["pixels_equal"] and target["rel_l2"] < 4e-3
    scale = checks["scale"]
    assert scale["S_jepa_future"] > 0 and scale["S_dino_future"] > 0
    ratio = (scale["ratio_current"] + scale["ratio_future"]) / 2
    assert scale["suggested_mse_loss_weight"] == pytest.approx(ratio, rel=0.05)
