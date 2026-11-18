"""共同状态增强，动作头缩小；价值读出仅预测标量回报，不负责分割或规划。"""
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from .relational_state import RelationalConfig, RelationalPredictor

ARCHITECTURE = "interactive_state_v5"


@dataclass(frozen=True)
class InteractiveConfig(RelationalConfig):
    supported_horizons: ClassVar[tuple] = (1, 5, 10)
    prediction_steps: int = 10
    report_horizons: tuple = (1, 5, 10)
    horizon_weights: tuple = (.5, .8, 1.2)
    chunk_steps: int = 16
    batch_size: int = 6
    prediction_anchors: int = 2
    validation_episodes: int = 64
    max_optimizer_steps: int = 4000
    max_training_seconds: float = 10800.
    early_stop_start: int = 12000
    learning_rate: float = .0002
    reconstruction_weight: float = .35
    world_weight: float = 1.
    prediction_pyramid: bool = True
    reward_weight: float = .1
    value_weight: float = .1
    advantage_temperature: float = .3
    partial_probability: float = .6
    missing_probability: float = .1
    patch_drop_probability: float = .5
    train_precision: str = "bfloat16"
    seed: int = 202611175

    def validate(self):
        super().validate()
        if self.dimension != 64 or self.world_weight not in (0., 1.) or self.advantage_temperature <= 0:
            raise ValueError("第五版固定 64 通道；world_weight 只用于预先规定的 0/1 对照；优势温度必须为正。")
        if min(self.reward_weight, self.value_weight) < 0:
            raise ValueError("奖惩学习权重不能为负。")


class InteractivePredictor(RelationalPredictor):
    def __init__(self, config):
        super().__init__(config)
        d = config.dimension
        self.refinements = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, d * 2, 3, padding=1), nn.GroupNorm(8, d * 2), nn.SiLU(),
                          nn.Conv2d(d * 2, d, 3, padding=1), nn.GroupNorm(8, d)) for _ in range(4)])
        self.refinement_scale = nn.Parameter(torch.full((4,), .05))
        self.action_head = nn.Sequential(nn.Conv2d(d, 8, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((8, 12)),
                                         nn.Flatten(), nn.Linear(8 * 8 * 12, 96), nn.SiLU(), nn.Linear(96, 18))
        self.value_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(d, 1))

    def cell(self, prior, incoming, event):
        state = super().cell(prior, incoming, event)
        for scale, refinement in zip(self.refinement_scale, self.refinements):
            state = state + scale.tanh() * refinement(state).tanh()
        return state

    def rollout(self, state, actions, dt, horizons=None):
        return super().rollout(state, actions, dt, self.config.report_horizons if horizons is None else horizons)

    def warm_start(self, checkpoint):
        if checkpoint["architecture"] != "relational_state_v4":
            raise ValueError("第五版迁移只接受第四版共同状态，不猜测其它权重布局。")
        own, inherited = self.state_dict(), {}
        for key, value in checkpoint["model"].items():
            if key.startswith("action_head."):
                continue
            if key not in own or own[key].shape != value.shape:
                raise ValueError(f"共同状态权重不兼容：{key}")
            inherited[key] = value
        own.update(inherited)
        self.load_state_dict(own)
        return {"inherited": list(inherited), "new": [k for k in own if k not in inherited],
                "old_action_head_inherited": False, "optimizer_inherited": False}
