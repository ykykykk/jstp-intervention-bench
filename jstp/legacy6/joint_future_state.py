"""第五版修订：联合未来包含自身运动、世界变化和可学习的结果评价。"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .interactive_state import InteractiveConfig, InteractivePredictor

ARCHITECTURE = "joint_future_v5"


@dataclass(frozen=True)
class FutureConfig(InteractiveConfig):
    max_optimizer_steps: int = 2000
    max_training_seconds: float = 5400.
    validation_interval: int = 500
    learning_rate: float = .0001
    imagined_outcome_weight: float = 1.
    imagined_value_weight: float = .5
    imagined_body_weight: float = .1
    imagined_action_weight: float = .1
    seed: int = 2026111755

    def validate(self):
        super().validate()
        weights = (self.imagined_outcome_weight, self.imagined_value_weight, self.imagined_body_weight, self.imagined_action_weight)
        if self.world_weight != 1 or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("联合未来修订必须训练世界预测及全部未来读出，权重为正有限值。")


class FuturePredictor(InteractivePredictor):
    def __init__(self, config):
        super().__init__(config)
        # 同一状态上的两个标量读出，不接收环境真值，不另设评价网络。
        self.outcome_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(config.dimension, 2))

    def outcome(self, predicted_state):
        output = self.outcome_head(predicted_state).float()
        return 2 * output[:, 0].tanh(), output[:, 1]

    def warm_start(self, checkpoint):
        if checkpoint["architecture"] != "interactive_state_v5":
            raise ValueError("联合未来修订只从已冻结的第五版迁移。")
        own = self.state_dict()
        for key, value in checkpoint["model"].items():
            if key not in own or own[key].shape != value.shape:
                raise ValueError(f"第五版权重不兼容：{key}")
            own[key] = value
        self.load_state_dict(own)
        return {"inherited": list(checkpoint["model"]), "new": [k for k in own if k not in checkpoint["model"]],
                "action_readout_role": "candidate future own-motion proposal, never direct commitment", "optimizer_inherited": False}
