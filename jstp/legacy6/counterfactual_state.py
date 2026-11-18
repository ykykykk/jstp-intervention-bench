"""同一共同状态的成组后果学习；没有对象槽、分割头或第二套评价网络。"""
from dataclasses import dataclass

from .interactive_state import InteractiveConfig
from .joint_future_state import FuturePredictor

ARCHITECTURE = "counterfactual_joint_v5"


@dataclass(frozen=True)
class CounterfactualConfig(InteractiveConfig):
    max_optimizer_steps: int = 3000
    max_training_seconds: float = 7200.
    validation_interval: int = 500
    validation_episodes: int = 64
    learning_rate: float = .00015
    missing_probability: float = 0.
    partial_probability: float = 0.
    patch_drop_probability: float = 0.
    closed_validation_interval: int = 1000
    branch_horizon: int = 5
    reward_scale: float = 20.
    reward_contrast_scale: float = 50.
    image_contrast_weight: float = 100.
    own_position_weight: float = 20.
    proposal_weight: float = .2
    rank_weight: float = .1
    rank_temperature: float = .02
    seed: int = 2026111761

    def validate(self):
        super().validate()
        if self.branch_horizon != 5 or self.world_weight != 1:
            raise ValueError("本轮固定五步成组后果预测，并保留图像预测。")
        if min(self.reward_scale, self.reward_contrast_scale, self.image_contrast_weight,
               self.own_position_weight, self.proposal_weight, self.rank_weight, self.rank_temperature) <= 0:
            raise ValueError("成组训练权重必须为正。")


class CounterfactualPredictor(FuturePredictor):
    def __init__(self, config):
        super().__init__(config)
        # 五步评分直接监督同样长度的实际反馈，不再附加旧的远期价值猜测。
        del self.value_head

    def warm_start(self, checkpoint):
        if checkpoint["architecture"] != "joint_future_v5":
            raise ValueError("本轮从上一轮完整联合未来模型迁移，不能猜测其它布局。")
        own, inherited = self.state_dict(), []
        for key, value in checkpoint["model"].items():
            if key.startswith(("value_head.", "outcome_head.")):
                continue
            if key not in own or own[key].shape != value.shape:
                raise ValueError(f"共同状态权重不兼容：{key}")
            own[key] = value
            inherited.append(key)
        self.load_state_dict(own)
        # 丢弃上一轮明显乐观的结果读出，重新用成组真实后果校准。
        import torch
        with torch.no_grad():
            self.outcome_head[-1].weight.zero_()
            self.outcome_head[-1].bias.copy_(torch.tensor([-.0025000052, 3.], device=self.outcome_head[-1].bias.device))
        return {"inherited": inherited, "outcome_readout_reset": True, "old_terminal_value_removed": True,
                "optimizer_inherited": False, "new_networks_or_segmenters": False}
