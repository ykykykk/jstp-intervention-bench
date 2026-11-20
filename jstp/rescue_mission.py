"""双舱救援的任务卡和操作。底层相机、挂接和胜负仍使用 V7 原装置。"""
from dataclasses import dataclass

import numpy as np

from .v7_world import TABLE, World, setup


@dataclass(frozen=True)
class Order:
    title: str
    instruction: str
    battery: str


# 顺序也是旧演示收益读出的三个任务编号。
ORDERS = (
    Order("只救乘员", "乘员进救援区，电池必须留在原泊位。", "留在原位"),
    Order("双舱转移", "乘员和电池都要进各自的救援区。", "一起送达"),
    Order("紧急撤离", "只要乘员进救援区就算过关，电池去哪都行。", "位置不限"),
)
ROTATION = (0, 1, 2, 0, 1, 2)


def rig(seed):
    board, example, transport, dy = setup(seed)
    lateral = transport.copy()
    lateral[:, 2] = example[2]
    lateral[:, 3] = 0
    return board, np.concatenate([transport, lateral]), int(dy)


def matches(frame, goals, order):
    candidates = goals if order == 2 else goals[order:order + 1]
    return any(np.array_equal(frame[:TABLE], goal) for goal in candidates)


class Sortie:
    def __init__(self, seed, joined, order, parked):
        self._layout = (seed, joined, parked)
        self.board, self.actions, self.dy = rig(seed)
        self.world = World(self.board, joined)
        self.order = order
        self.goals = self.board.goals(self.dy)
        self.start = self.world.camera()
        self.peak = self.start
        self.past = np.zeros(4, dtype=int)
        self.hand = self.board.grips[parked].copy()
        self.scouted = None
        self.sent = None
        self.won = None

    def other_order(self):
        """在同一复位点分出另一张工单，不重演试探，也不重新抽接线。"""
        if self.scouted is None or self.order == 2 or self.won is None:
            raise ValueError("只有带试探经历的定向任务可以换单")
        seed, joined, parked = self._layout
        other = Sortie(seed, joined, 1 - self.order, parked)
        other.scouted, other.peak, other.past = self.scouted, self.peak.copy(), self.past.copy()
        other.hand = self.past[:2].copy()
        return other

    def offset(self, handle):
        return self.world.g[handle] - self.board.grips[handle]

    def press(self, handle):
        # 中途松手后应在绞盘的当前位置重新握住，而不是去抓空的起点。
        self.hand = self.world.g[handle].copy()
        self.world.press(self.hand)

    def step(self, delta):
        self.hand += delta
        self.world.move(delta)

    def remember_peak(self, handle):
        self.scouted = handle
        self.past = np.concatenate((self.board.grips[handle], self.offset(handle)))
        self.peak = self.world.camera()

    def finish_probe(self):
        self.world.release()
        if not np.array_equal(self.world.camera(), self.start):
            raise RuntimeError("横拉试探未能回到起始画面")

    def finish_delivery(self, handle):
        self.world.release()
        self.sent = handle
        self.won = matches(self.world.camera(), self.goals, self.order)
        return self.won
