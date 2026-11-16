"""The action-conditioned head (align_params.action_cond): target, head, loss wiring, loader and config tools."""
import types

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from lingbotvla.models.module_utils import _init_post_training_new_modules
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import FlowMatchingV2, LingbotVlaV2Policy
from lingbotvla.models.vla.vision_models.align_heads.action_cond_head import (
    ActionConditionedHead,
    teacher_delta_target,
)

from conftest import ROOT

LLM, STATE, ACTION, CHUNK, GROUP, QUERIES, OUT = 32, 6, 7, 10, 5, 16, 12
IMAGE_TOKENS = 4  # llm_image_token_size 2 -> 2 x 2 top-camera tokens


def make_head(use_actions=True):
    torch.manual_seed(0)
    return ActionConditionedHead(llm_dim=LLM, state_dim=STATE, action_dim=ACTION, chunk_size=CHUNK,
                                 action_group=GROUP, num_queries=QUERIES, dim_out=OUT, num_layers=1, num_heads=2,
                                 dim_head=8, ff_mult=1, use_actions=use_actions)


def randomize_output(head):
    with torch.no_grad():
        nn.init.normal_(head.predictor.proj_out.weight, std=0.1)


def batch(b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(vlm=torch.randn(b, IMAGE_TOKENS + 2, LLM, generator=g), state=torch.randn(b, STATE, generator=g),
                actions=torch.randn(b, CHUNK, ACTION, generator=g))


def test_target_is_layernormed_difference():
    g = torch.Generator().manual_seed(0)
    cur = torch.randn(2, 5, 8, generator=g) * 3 + 1
    fut = cur + 0.1 * torch.randn(2, 5, 8, generator=g)
    target = teacher_delta_target(fut, cur)
    expected = F.layer_norm(fut, (8,), eps=1e-6) - F.layer_norm(cur, (8,), eps=1e-6)
    assert torch.allclose(target, expected, atol=1e-6)
    assert torch.equal(teacher_delta_target(cur, cur), torch.zeros_like(cur))
    assert not target.requires_grad


def test_head_shape_and_zero_start():
    head = make_head()
    x = batch()
    pred = head(x["vlm"], x["state"], x["actions"])
    assert pred.shape == (4, QUERIES, OUT) and pred.dtype == torch.float32
    assert torch.count_nonzero(pred) == 0, "proj_out starts at zero, so the first loss equals mean |target|"


def test_padding_is_ignored_and_b0_ignores_actions():
    head = make_head()
    randomize_output(head)
    x = batch()
    joint_mask = torch.ones(4, CHUNK, ACTION, dtype=torch.bool)
    joint_mask[..., -2:] = False
    is_pad = torch.zeros(4, CHUNK, dtype=torch.bool)
    is_pad[:, -3:] = True
    noisy = x["actions"].clone()
    noisy[..., -2:] += 5.0
    noisy[:, -3:] += 5.0
    a = head(x["vlm"], x["state"], x["actions"], joint_mask, None, is_pad)
    b = head(x["vlm"], x["state"], noisy, joint_mask, None, is_pad)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, head(x["vlm"], x["state"], x["actions"] + 1.0, joint_mask, None, is_pad))

    blind = make_head(use_actions=False)
    randomize_output(blind)
    assert torch.allclose(blind(x["vlm"], x["state"], x["actions"]), blind(x["vlm"], x["state"], noisy))


def test_gradient_reaches_vlm_tokens():
    head = make_head()
    randomize_output(head)
    x = batch()
    vlm = x["vlm"].requires_grad_(True)
    head(vlm, x["state"], x["actions"]).square().mean().backward()
    assert vlm.grad is not None and vlm.grad.abs().sum() > 0


def test_chunk_must_split_into_groups():
    with pytest.raises(ValueError):
        ActionConditionedHead(llm_dim=LLM, state_dim=STATE, action_dim=ACTION, chunk_size=12, action_group=5)


class FakeFlowMatching(nn.Module):
    """Just what FlowMatchingV2.action_cond_emb_forward reads."""

    action_cond_emb_forward = FlowMatchingV2.action_cond_emb_forward

    def __init__(self, use_actions=True, weight=0.5, detach=False, shuffle=True):
        super().__init__()
        self.llm_image_token_size = 2
        self.action_cond_loss_weight = weight
        self.action_cond_detach_image_feats = detach
        self.action_cond_shuffle_metric = shuffle
        self.action_cond_head = make_head(use_actions)

    def _future_video_patch_task_tokens(self, hidden_states):
        return hidden_states[:, -2:, :]


def prefix_batch(b=4, seed=2):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(b, 1 + IMAGE_TOKENS + 5 + 2, LLM, generator=g)  # vision start, image, other, future tokens
    cur = torch.randn(b, QUERIES, OUT, generator=g)
    fut = cur + torch.randn(b, QUERIES, OUT, generator=g)
    return hidden, torch.randn(b, STATE, generator=g), torch.randn(b, CHUNK, ACTION, generator=g), fut, cur


def test_loss_metrics_and_zero_start():
    model = FakeFlowMatching(weight=0.5)
    hidden, state, actions, fut, cur = prefix_batch()
    weighted, preds, metrics = model.action_cond_emb_forward(hidden, state, actions, fut, cur)
    target = teacher_delta_target(fut, cur)
    assert torch.allclose(metrics["align/action_cond_loss"], target.abs().mean())
    assert torch.allclose(metrics["align/action_cond_zero_loss"], target.abs().mean())
    assert torch.allclose(weighted, 0.5 * metrics["align/action_cond_loss"])
    assert torch.allclose(metrics["align/action_cond_loss_weighted"], weighted.detach())
    assert metrics["align/action_cond_shuffle_gap"].abs() < 1e-6, "an all-zero prediction cannot use actions"
    assert weighted.requires_grad and not metrics["align/action_cond_shuffle_loss"].requires_grad
    with pytest.raises(ValueError):
        model.action_cond_emb_forward(hidden, state, actions, fut, None)

    model1 = FakeFlowMatching(shuffle=False)
    _, _, metrics1 = model1.action_cond_emb_forward(hidden, state, actions, fut, cur)
    assert "align/action_cond_shuffle_gap" not in metrics1
    _, _, metrics_b1 = FakeFlowMatching().action_cond_emb_forward(*(t[:1] for t in prefix_batch()))
    assert "align/action_cond_shuffle_gap" not in metrics_b1, "no other sample to borrow actions from"


def test_image_tokens_are_the_top_camera_span_and_detach_works():
    hidden, state, actions, fut, cur = prefix_batch()
    for detach in (False, True):
        model = FakeFlowMatching(detach=detach)
        randomize_output(model.action_cond_head)
        h = hidden.clone().requires_grad_(True)
        weighted, _, _ = model.action_cond_emb_forward(h, state, actions, fut, cur)
        weighted.backward()
        image_grad = h.grad[:, 1:1 + IMAGE_TOKENS].abs().sum()
        assert (image_grad == 0) == detach
        assert h.grad[:, 0].abs().sum() == 0 and h.grad[:, 1 + IMAGE_TOKENS:-2].abs().sum() == 0
        assert h.grad[:, -2:].abs().sum() > 0


def train_on_action_dependent_target(use_actions, steps=150):
    """Every patch moves by a fixed linear map of the mean action, so only a head that sees actions can fit it."""
    torch.manual_seed(3)
    model = FakeFlowMatching(use_actions=use_actions, weight=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    readout = 3 * torch.randn(ACTION, OUT) / ACTION ** 0.5
    metrics = None
    for step in range(steps):
        hidden, state, actions, _, cur = prefix_batch(b=16, seed=100 + step)
        fut = cur + (actions.mean(1) @ readout)[:, None, :]
        weighted, _, metrics = model.action_cond_emb_forward(hidden, state, actions, fut, cur)
        opt.zero_grad()
        weighted.backward()
        opt.step()
    return metrics


def test_shuffle_gap_separates_b1_from_b0():
    b1 = train_on_action_dependent_target(True)
    b0 = train_on_action_dependent_target(False)
    assert b1["align/action_cond_loss"] < 0.9 * b1["align/action_cond_zero_loss"]
    assert b1["align/action_cond_shuffle_gap"] > 0.05
    assert b0["align/action_cond_shuffle_gap"].abs() < 1e-6


def test_policy_adds_action_cond_loss_and_keeps_tuple_length():
    def fake_model_forward(*args, **kwargs):
        assert kwargs["state_joint_mask"] is state_joint_mask and kwargs["joint_mask"] is joint_mask
        losses = torch.ones(2, CHUNK, ACTION)
        metrics = {"align/action_cond_loss": torch.tensor(0.3), "_action_cond_loss": torch.tensor(0.25)}
        zero = torch.tensor(0.0)
        return losses, zero, zero, torch.tensor(0.5), None, zero, zero, metrics, None, None, None

    joint_mask = torch.ones(2, CHUNK, ACTION)
    state_joint_mask = torch.ones(2, STATE, dtype=torch.bool)
    policy = types.SimpleNamespace(config=types.SimpleNamespace(loss_type="L1_fm", action_dim=ACTION),
                                   model=types.SimpleNamespace(forward=fake_model_forward))
    out = LingbotVlaV2Policy.forward(policy, None, None, torch.zeros(2, STATE), None, None, torch.zeros(2, CHUNK, ACTION),
                                     joint_mask=joint_mask, state_joint_mask=state_joint_mask)
    assert len(out) == 11
    total, loss_vla, loss_dict = out[0], out[1], out[6]
    assert torch.allclose(total, loss_vla + 0.5 + 0.25)
    assert "_action_cond_loss" not in loss_dict and "align/action_cond_loss" in loss_dict


class FakePolicy(nn.Module):
    def __init__(self, with_head=True):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Linear(3, 3)
        if with_head:
            self.model.action_cond_head = make_head()

    post_training_new_modules = LingbotVlaV2Policy.post_training_new_modules


def test_loader_initialises_new_head_only_when_entirely_missing():
    policy = FakePolicy()
    with torch.no_grad():
        for p in policy.model.action_cond_head.parameters():
            p.fill_(float("nan"))
    head_names = {f"model.action_cond_head.{n}" for n, _ in policy.model.action_cond_head.named_parameters()}
    left = _init_post_training_new_modules(policy, set(head_names))
    assert left == set()
    head = policy.model.action_cond_head
    assert all(torch.isfinite(p).all() for p in head.parameters())
    assert torch.count_nonzero(head.predictor.proj_out.weight) == 0

    assert _init_post_training_new_modules(policy, {"model.backbone.weight"}) == {"model.backbone.weight"}
    with pytest.raises(KeyError):
        _init_post_training_new_modules(policy, {sorted(head_names)[0]})
    assert _init_post_training_new_modules(FakePolicy(with_head=False), {"model.backbone.bias"}) == {
        "model.backbone.bias"}


def test_flowmatching_init_hook_also_initialises_head_parameters():
    head = make_head()
    with torch.no_grad():
        for p in head.parameters():
            p.fill_(float("nan"))
    fake = types.SimpleNamespace(config=types.SimpleNamespace(initializer_range=0.02))
    head.apply(lambda m: FlowMatchingV2._init_weights(fake, m))
    assert all(torch.isfinite(p).all() for p in head.parameters())
    assert torch.count_nonzero(head.predictor.proj_out.weight) == 0


def test_init_requires_dino_current_and_future_targets():
    fake = types.SimpleNamespace(use_future_video=True, use_current_video_patch=False)
    with pytest.raises(ValueError, match="use_current_patch_loss"):
        FlowMatchingV2.init_action_cond_head(fake, {"action_cond": {"enabled": True}})


def test_official_yaml_keeps_the_head_off():
    cfg = yaml.safe_load((ROOT / "configs/vla/robotwin/robotwin.yaml").read_text())
    align = cfg["train"]["align_params"]
    assert align["action_cond"]["enabled"] is False
    assert align["video"]["use_current_patch_loss"] is True
    assert 50 % align["action_cond"]["action_group"] == 0, "LingBot's chunk is 50 steps"
