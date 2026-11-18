"""第六版：同一空间状态，真实/想象闭合，保留空间的身体与完成经历联合读出。"""
from dataclasses import dataclass
import torch
from torch import nn
from .counterfactual_state import CounterfactualConfig, CounterfactualPredictor

ARCHITECTURE = "grounded_joint_v6"


@dataclass(frozen=True)
class GroundedConfig(CounterfactualConfig):
    max_optimizer_steps: int = 6000
    max_training_seconds: float = 10800.
    learning_rate: float = .00015
    validation_interval: int = 2000
    seed: int = 2026111769
    latent_weight: float = 2.
    cost_weight: float = 1.
    sensory_weight: float = 20.
    cost_rank_temperature: float = 1.5
    history_gradient_steps: int = 12
    future_steps: int = 3


class GroundedPredictor(CounterfactualPredictor):
    def __init__(self, config):
        super().__init__(config)
        del self.outcome_head
        # 身体与完成经历共享空间读出；不是另一个接收环境真值的价值网络。
        self.body_head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
                                       nn.Linear(64 * 24, 128), nn.SiLU(), nn.Linear(128, 70))

    def joint(self, state):
        output = self.body_head(state).float()
        bins = torch.arange(65, device=state.device, dtype=torch.float32)
        return {"body": output[:, :3], "cost_logits": output[:, 3:68],
                "cost": (output[:, 3:68].softmax(-1) * bins).sum(-1),
                "done": output[:, 68], "reward": 2 * output[:, 69].tanh()}

    def proprio_logits(self, state):
        return self.joint(state)["body"]

    def warm_start(self, checkpoint):
        if checkpoint["architecture"] != "counterfactual_joint_v5":
            raise ValueError("第六版迁移需要已冻结的第五版成组模型。")
        own = self.state_dict()
        for key, value in checkpoint["model"].items():
            if key.startswith(("body_head.", "outcome_head.")):
                continue
            if key not in own or own[key].shape != value.shape:
                raise ValueError(f"迁移结构不符：{key}")
            own[key] = value
        previous = checkpoint["model"]
        own["body_head.2.weight"][:64] = previous["body_head.2.weight"]
        own["body_head.2.bias"][:64] = previous["body_head.2.bias"]
        own["body_head.4.weight"][:3].zero_()
        own["body_head.4.weight"][:3, :64] = previous["body_head.4.weight"]
        own["body_head.4.bias"][:3] = previous["body_head.4.bias"]
        self.load_state_dict(own)
        return {"shared_core_and_action_proposal_inherited": True, "old_global_average_reward_removed": True,
                "new_completion_distribution": "observed correction suffix steps 0..64, not geometry input", "optimizer_inherited": False}
