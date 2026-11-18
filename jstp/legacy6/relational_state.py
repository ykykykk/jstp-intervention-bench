"""第四版：一个递归网络中的局部更新和通用全局联系，不预设对象或分割。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .minimal_state import MinimalConfig, MinimalPredictor
from .model import ConvGRU

ARCHITECTURE = "relational_state_v4"


@dataclass(frozen=True)
class RelationalConfig(MinimalConfig):
    dimension: int = 64
    chunk_steps: int = 12
    batch_size: int = 8
    prediction_anchors: int = 3
    validation_episodes: int = 128
    max_optimizer_steps: int = 6000
    max_training_seconds: float = 10800.
    validation_interval: int = 1000
    early_stop_start: int = 12000
    early_stop_patience: int = 8
    missing_probability: float = .05
    partial_probability: float = .65
    patch_drop_probability: float = .6
    action_weight: float = .7
    proprio_weight: float = .1
    train_precision: str = "float16"
    seed: int = 202611174

    def validate(self):
        super().validate()
        if self.dimension % 4 or self.prediction_anchors < 1 or self.validation_episodes < 1:
            raise ValueError("维数需整除四个通用注意力头；预测锚点与验证轨迹数须为正数。")
        if self.train_precision not in ("float16", "bfloat16"):
            raise ValueError("训练精度必须为 float16 或 bfloat16。")


class RelationalPredictor(MinimalPredictor):
    def __init__(self, config):
        # 方法契约复用第三版，但不实例化或加载第三版网络。
        nn.Module.__init__(self)
        self.config = config
        d = config.dimension
        self.encoder = nn.Sequential(nn.Conv2d(4, 32, 5, 2, 2), nn.SiLU(), nn.Conv2d(32, d - 8, 3, 2, 1), nn.SiLU())
        self.body_encoder = nn.Sequential(nn.Linear(3, 8), nn.SiLU())
        self.action_encoder = nn.Embedding(18, d)
        self.dt_encoder = nn.Linear(1, d)
        self.event_embedding = nn.Embedding(2, d)
        self.core = ConvGRU(d, d)
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, d * 3)
        self.projection = nn.Linear(d, d)
        self.global_scale = nn.Parameter(torch.tensor(.1))
        self.position = nn.Parameter(torch.randn(1, 384, d) * .01)
        self.feed_forward = nn.Sequential(nn.Conv2d(d, d * 2, 1), nn.SiLU(), nn.Conv2d(d * 2, d, 1))
        self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(d, 48, 3, 1, 1), nn.SiLU(),
                                     nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(48, 24, 3, 1, 1), nn.SiLU(),
                                     nn.Conv2d(24, 3, 3, 1, 1), nn.Sigmoid())
        self.action_head = nn.Sequential(nn.Conv2d(d, 8, 1), nn.SiLU(), nn.Flatten(),
                                         nn.Linear(8 * 32 * 48, 192), nn.SiLU(), nn.Linear(192, 18))
        self.body_head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
                                       nn.Linear(d * 24, 64), nn.SiLU(), nn.Linear(64, 3))

    def cell(self, prior, incoming, event):
        if prior is None:
            prior = incoming.new_zeros(incoming.shape[0], self.config.dimension, 32, 48)
        incoming = incoming + self.event_embedding.weight[event][None, :, None, None]
        local = self.core(prior, incoming)
        tokens = F.avg_pool2d(local, 2).flatten(2).transpose(1, 2)
        normalized = self.norm(tokens + self.position)
        b, n, d = normalized.shape
        q, k, v = self.qkv(normalized).reshape(b, n, 3, 4, d // 4).permute(2, 0, 3, 1, 4).unbind(0)
        connected = F.scaled_dot_product_attention(q, k, v)
        connected = self.projection(connected.transpose(1, 2).reshape(b, n, d)).transpose(1, 2).reshape(b, d, 16, 24)
        connected = F.interpolate(connected, size=(32, 48), mode="bilinear", align_corners=False)
        state = local + self.global_scale.tanh() * connected.tanh()
        return state + .1 * self.feed_forward(state).tanh()
