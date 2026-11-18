"""第三版：一个空间递归状态，同一更新器处理观察与动作。无对象槽、分割头或规划器。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn

from .model import ConvGRU

ARCHITECTURE = "minimal_state_v3"


@dataclass(frozen=True)
class MinimalConfig:
    image_width: int = 192
    image_height: int = 128
    control_hz: int = 5
    context_frames: int = 8
    prediction_steps: int = 5
    report_horizons: tuple = (1, 3, 5)
    horizon_weights: tuple = (1., .7, .4)
    chunk_steps: int = 16
    batch_size: int = 8
    dimension: int = 48
    learning_rate: float = .0003
    weight_decay: float = .0001
    gradient_clip_norm: float = 1.
    max_optimizer_steps: int = 1500
    max_training_seconds: float = 7200.
    validation_interval: int = 500
    early_stop_start: int = 3000
    early_stop_patience: int = 6
    early_stop_min_relative_improvement: float = .005
    change_threshold: float = 8 / 255
    pixel_objective: str = "mse"
    random_anchor: bool = True
    image_weight: float = 20.
    action_weight: float = 1.
    proprio_weight: float = .2
    reconstruction_weight: float = 1.
    missing_probability: float = .05
    partial_probability: float = .2
    patch_drop_probability: float = .2
    seed: int = 20261117

    @classmethod
    def from_dict(cls, value):
        unknown = value.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"第三版未知配置项：{sorted(unknown)}")
        value = dict(value)
        for key in ("report_horizons", "horizon_weights"):
            if key in value:
                value[key] = tuple(value[key])
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        horizons = getattr(self, "supported_horizons", (1, 3, 5))
        if (self.image_width, self.image_height, self.control_hz, self.prediction_steps, self.report_horizons) != (192, 128, 5, max(horizons), horizons):
            raise ValueError(f"固定使用 192×128、5 Hz 名义时间、{horizons} 步预测。")
        for key in ("dimension", "batch_size", "chunk_steps", "max_optimizer_steps", "validation_interval", "early_stop_start", "early_stop_patience"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} 必须为正整数。")
        for key in ("learning_rate", "max_training_seconds", "gradient_clip_norm", "image_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} 必须为正有限数。")
        for key in ("weight_decay", "action_weight", "proprio_weight", "reconstruction_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} 必须非负有限。")
        if self.pixel_objective != "mse" or self.random_anchor is not True:
            raise ValueError("第三版固定普通像素 MSE 和均匀随机历史锚点。")
        if len(self.horizon_weights) != 3 or any(w <= 0 or not math.isfinite(w) for w in self.horizon_weights):
            raise ValueError("三个跨度权重必须为正有限数。")
        if any(not 0 <= p < 1 for p in (self.missing_probability, self.partial_probability, self.patch_drop_probability, self.early_stop_min_relative_improvement, self.change_threshold)) or self.missing_probability + self.partial_probability > 1:
            raise ValueError("概率或阈值越界。")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed 必须为 uint32。")

    def as_dict(self):
        return asdict(self)


class MinimalPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.dimension
        self.encoder = nn.Sequential(nn.Conv2d(4, 24, 5, 2, 2), nn.SiLU(),
                                     nn.Conv2d(24, 40, 3, 2, 1), nn.SiLU())
        self.body_encoder = nn.Sequential(nn.Linear(3, 8), nn.SiLU())
        self.action_encoder = nn.Embedding(18, 48)
        self.dt_encoder = nn.Linear(1, 48)
        self.event_embedding = nn.Embedding(2, 8)
        self.core = ConvGRU(56, d)
        self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                     nn.Conv2d(d, 32, 3, 1, 1), nn.SiLU(),
                                     nn.Upsample(scale_factor=2, mode="nearest"),
                                     nn.Conv2d(32, 16, 3, 1, 1), nn.SiLU(),
                                     nn.Conv2d(16, 3, 3, 1, 1), nn.Sigmoid())
        self.action_head = nn.Sequential(nn.Conv2d(d, 8, 1), nn.SiLU(), nn.Flatten(),
                                         nn.Linear(8 * 32 * 48, 128), nn.SiLU(), nn.Linear(128, 18))
        self.body_head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
                                       nn.Linear(d * 24, 64), nn.SiLU(), nn.Linear(64, 3))

    def initial_state(self, batch, device):
        return torch.zeros(batch, self.config.dimension, 32, 48, device=device)

    def encode(self, image, proprio, visibility=None):
        if visibility is None:
            visibility = torch.ones_like(image[:, :1])
        visual = self.encoder(torch.cat((image * visibility, visibility), 1))
        body = self.body_encoder(proprio)[:, :, None, None].expand(-1, -1, 32, 48)
        return torch.cat((visual, body), 1)

    def cell(self, prior, incoming, event):
        if prior is None:
            prior = incoming.new_zeros(incoming.shape[0], self.config.dimension, 32, 48)
        kind = self.event_embedding.weight[event].to(incoming.dtype)[None, :, None, None].expand(incoming.shape[0], -1, 32, 48)
        return self.core(prior, torch.cat((incoming, kind), 1))

    def observe(self, image, proprio, prior, visibility=None):
        return self.cell(prior, self.encode(image, proprio, visibility), 0)

    def transition(self, state, action, dt):
        incoming = self.action_encoder(action) + self.dt_encoder(dt[:, None])
        return self.cell(state, incoming[:, :, None, None].expand(-1, -1, 32, 48), 1)

    def decode(self, state):
        return self.decoder(state)

    def proprio_logits(self, state):
        return self.body_head(state)

    def history(self, images, proprio, past_actions, past_dt):
        state, logits = None, []
        for t in range(images.shape[1]):
            if t:
                state = self.transition(state, past_actions[:, t - 1], past_dt[:, t - 1])
            state = self.observe(images[:, t], proprio[:, t], state)
            logits.append(self.action_head(state))
        return state, torch.stack(logits, 1)

    def rollout(self, state, actions, dt, horizons=(1, 3, 5)):
        predictions = {}
        for t in range(actions.shape[1]):
            state = self.transition(state, actions[:, t], dt[:, t])
            if t + 1 in horizons:
                predictions[t + 1] = self.decode(state)
        return predictions

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
