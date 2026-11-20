"""A playable pair of late-goal cards on the paper's unmodified four-track rig."""
from dataclasses import dataclass
from secrets import randbelow

import numpy as np

from lategoal.rig import EXAM, Rig, brightness, goal_wins, history, lane_travel


@dataclass(frozen=True)
class Deal:
    layout: int
    wiring: int
    gear: int
    probe: int
    card: int

    @classmethod
    def fresh(cls):
        # Separate from all training, validation and published exam seed ranges.
        return cls(500_000_000 + randbelow(400_000_000), randbelow(2), randbelow(2),
                   (-1, 1)[randbelow(2)], randbelow(len(EXAM)))


PAPER_EXAMPLE = Deal(113670000, 0, 0, -1, 0)


def horizontal_centres(frame):
    light = brightness(frame)[:44].reshape(44, 4, 24).sum(0)
    return (light * np.arange(24)).sum(1) / light.sum(1)


@dataclass
class Outcome:
    frame: np.ndarray
    motion: np.ndarray
    hits: int
    won: bool


@dataclass
class Sounding:
    side: int
    peak: np.ndarray
    trace: np.ndarray
    observation: np.ndarray


@dataclass
class Stroke:
    frame: np.ndarray
    motion: np.ndarray


class FreeHarbor:
    """Free play forks the same rig at its berth for each gesture, not the paper's sealed trial."""
    def __init__(self, deal):
        self.deal = deal
        self.rig = Rig(deal.layout)
        self.choices, self.outcomes = [], []

    @property
    def card_ids(self):
        return self.deal.card, self.deal.card ^ 1

    def sound(self, side):
        trial = self.rig.world(self.deal.wiring)
        peak = trial.probe(side)  # both winches, any order; every pull returns to the same berth
        trial.set_clutch(self.deal.gear)
        return Sounding(side, peak, horizontal_centres(peak)-horizontal_centres(self.rig.base),
                        history(self.rig, peak, side, self.deal.gear))

    def voyage(self, side):
        trial = self.rig.world(self.deal.wiring)
        # Arm a disposable fork. This internal reset is not a player's sounding or a model input.
        trial.probe(side)
        trial.set_clutch(self.deal.gear)
        frame = trial.stroke(side)
        return Stroke(frame, lane_travel(self.rig.base, frame))

    def land(self, side, stroke, goal):
        self.choices.append(side)
        self.outcomes.append(Outcome(stroke.frame, stroke.motion, int((goal*stroke.motion > 0).sum()),
                                     bool(goal_wins(stroke.motion, goal))))


class Dispatch:
    def __init__(self, deal):
        self.deal = deal
        self.rig = Rig(deal.layout)
        self.trial = self.rig.world(deal.wiring)
        self.peak = self.trace = self.observation = None
        self.choices = []
        self.outcomes = []

    @property
    def card_ids(self):
        return self.deal.card, self.deal.card ^ 1

    def begin_probe(self):
        if self.peak is None:
            self.peak = self.trial.probe(self.deal.probe)
            self.trace = horizontal_centres(self.peak) - horizontal_centres(self.rig.base)

    def finish_probe(self):
        if self.peak is None or self.observation is not None:
            raise RuntimeError("必须完成一次且仅一次试拉")
        self.trial.set_clutch(self.deal.gear)
        self.observation = history(self.rig, self.peak, self.deal.probe, self.deal.gear)

    def seal(self, side):
        if side not in (-1, 1) or self.observation is None or len(self.choices) >= 2:
            raise ValueError("当前不能签发这个把手")
        self.choices.append(side)
        if len(self.choices) == 1:
            return
        # Neither player gets the first outcome before committing the second action.
        pieces, hands = self.trial.stroke_copies(np.asarray(self.choices))
        for pos, grips, card in zip(pieces, hands, self.card_ids, strict=True):
            final = self.rig.board.render(pos, grips)
            motion = lane_travel(self.rig.base, final)
            goal = EXAM[card]
            self.outcomes.append(Outcome(final, motion, int(((goal * motion) > 0).sum()),
                                         bool(goal_wins(motion, goal))))
