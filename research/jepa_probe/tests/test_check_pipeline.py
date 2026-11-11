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
