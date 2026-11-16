import torch
import torch.nn as nn
import torch.nn.functional as F

from .resampler import TaskTokenResampler


def teacher_delta_target(future_patch, current_patch, eps=1e-6):
    """LN(z_future) - LN(z_current) per patch token, with a parameter-free LayerNorm over channels."""
    dim = future_patch.shape[-1]
    future = F.layer_norm(future_patch.float(), (dim,), eps=eps)
    current = F.layer_norm(current_patch.float(), (dim,), eps=eps)
    return (future - current).detach()


class ActionConditionedHead(nn.Module):
    """Predicts the teacher's per-patch change between the current and the future frame.

    The context is the VLM's top-camera image tokens and future task tokens, plus one state token and one token per
    `action_group` steps of the (clean) action chunk. With `use_actions=False` the action tokens are built from zeros,
    so the head keeps the same shape but sees no action information.
    """

    def __init__(
        self,
        *,
        llm_dim,
        state_dim,
        action_dim,
        chunk_size,
        action_group=5,
        num_queries=256,
        dim_out=1024,
        num_layers=1,
        num_heads=4,
        dim_head=32,
        ff_mult=1,
        use_actions=True,
        init_std=0.02,
    ):
        super().__init__()
        if chunk_size % action_group != 0:
            raise ValueError(f"chunk_size {chunk_size} is not divisible by action_group {action_group}.")
        self.use_actions = bool(use_actions)
        self.action_group = action_group
        self.num_action_tokens = chunk_size // action_group
        self.init_std = init_std

        self.state_proj = nn.Linear(state_dim, llm_dim)
        self.action_proj = nn.Linear(action_group * action_dim, llm_dim)
        self.action_pos_embedding = nn.Parameter(torch.zeros(self.num_action_tokens, llm_dim))
        self.queries = nn.Parameter(torch.zeros(num_queries, llm_dim))
        self.predictor = TaskTokenResampler(
            dim_in=llm_dim,
            dim_mid=llm_dim,
            dim_head=dim_head,
            dim_out=dim_out,
            num_layers=num_layers,
            num_heads=num_heads,
            num_queries=num_queries,
            ff_mult=ff_mult,
        )
        # The target is a per-token difference that is near zero on static patches; a final LayerNorm would force
        # every output token to the same norm.
        self.predictor.norm_out = nn.Identity()
        self.reset_parameters()

    @torch.no_grad()
    def _reset_post_init_parameters(self):
        nn.init.normal_(self.queries, std=1.0)
        nn.init.normal_(self.action_pos_embedding, std=self.init_std)
        nn.init.zeros_(self.predictor.proj_out.weight)
        nn.init.zeros_(self.predictor.proj_out.bias)

    @torch.no_grad()
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self._reset_post_init_parameters()

    def action_tokens(self, actions, joint_mask=None, action_is_pad=None):
        """[B, T, A] actions -> [B, T // action_group, llm_dim] tokens; padded joints and steps are zeroed."""
        actions = actions.float()
        if joint_mask is not None:
            actions = actions * joint_mask.to(actions.dtype)
        if action_is_pad is not None:
            actions = actions * (~action_is_pad.bool()).unsqueeze(-1).to(actions.dtype)
        if not self.use_actions:
            actions = torch.zeros_like(actions)
        bsize, steps, dim = actions.shape
        if steps != self.num_action_tokens * self.action_group:
            raise ValueError(f"expected {self.num_action_tokens * self.action_group} action steps, got {steps}.")
        grouped = actions.reshape(bsize, self.num_action_tokens, self.action_group * dim)
        weight = self.action_proj.weight
        tokens = self.action_proj(grouped.to(weight.dtype))
        return tokens + self.action_pos_embedding.to(tokens.dtype)

    def state_token(self, state, state_joint_mask=None):
        state = state.float()
        if state_joint_mask is not None:
            state = state * state_joint_mask.to(state.dtype)
        weight = self.state_proj.weight
        return self.state_proj(state.to(weight.dtype)).unsqueeze(1)

    def forward(self, vlm_tokens, state, actions, joint_mask=None, state_joint_mask=None, action_is_pad=None):
        dtype = vlm_tokens.dtype
        context = torch.cat(
            [
                vlm_tokens,
                self.state_token(state, state_joint_mask).to(dtype),
                self.action_tokens(actions, joint_mask, action_is_pad).to(dtype),
            ],
            dim=1,
        )
        queries = self.queries.unsqueeze(0).expand(context.shape[0], -1, -1).to(dtype)
        return self.predictor(context, queries).float()
