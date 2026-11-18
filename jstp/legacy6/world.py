from __future__ import annotations
import copy,math
from types import SimpleNamespace
import numpy as np
DIRECTIONS=((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1))
LEVELS=("basic","ordinary","motion","occlusion","transfer")

def displacement(action: int) -> tuple[float, float]:
    if not 0 <= action < 18:
        raise ValueError(f"动作编号越界：{action}")
    x, y = DIRECTIONS[action // 2]
    scale = 8 / math.sqrt(2) if x and y else 8
    return x * scale, y * scale

class OwnedEnvironment:
    def __init__(self,seed,task):
        import pygame
        self.pg=pygame
    def camera(self):
        return self.pg.surfarray.array3d(self.draw()).transpose(1,0,2).copy()

class InteractionWorld:
    def __init__(self, seed, variant="interaction", object_count=None):
        if variant not in ("hinted", "interaction", "occluded", "switch", "novel"):
            raise ValueError(f"未知第五版场景：{variant}")
        self.seed, self.variant = seed, variant
        self.rng = np.random.default_rng(seed)
        n = object_count or int(self.rng.integers(4, 6) if variant == "novel" else self.rng.integers(2, 4))
        self.objects = []
        for i in range(n):
            w, h = int(self.rng.choice([20, 24, 28])), int(self.rng.choice([18, 22, 26]))
            for _ in range(1000):
                x, y = self.rng.uniform([20, 20], [172, 108])
                if all(abs(x - o["x"]) > (w + o["w"]) / 2 + 4 or abs(y - o["y"]) > (h + o["h"]) / 2 + 4 for o in self.objects):
                    break
            else:
                raise RuntimeError("无法放置互不重叠的物体。")
            moving = variant in ("occluded", "novel") or self.rng.random() < .35
            self.objects.append(dict(x=float(x), y=float(y), w=w, h=h,
                                     vx=float(self.rng.choice([-1, 1]) * self.rng.uniform(.4, 1.2)) if moving else 0.,
                                     vy=float(self.rng.choice([-1, 1]) * self.rng.uniform(.3, .9)) if moving else 0.))
        self.target_index = int(self.rng.integers(n))
        self.grasped = None
        self.offset = (0., 0.)
        self.cue_until = 8
        self.stage = 0
        self.completed_goals = 0
        self.switched = False
        self.final_done = False
        self.state = SimpleNamespace(task="drag", cursor_x=float(self.rng.uniform(5, 187)),
                                     cursor_y=float(self.rng.uniform(5, 123)), left_held=False, dragging=False,
                                     steps=0, block_x=0., block_y=0., target_x=0., target_y=0.)
        self.new_destination()
        self.sync()
        self.last_reward, self.last_external = 0., False
        self.contacts = 0

    def new_destination(self):
        obj = self.objects[self.target_index]
        for _ in range(1000):
            x, y = self.rng.uniform([24, 22], [168, 106])
            if np.hypot(x - obj["x"], y - obj["y"]) > 50:
                self.state.target_x, self.state.target_y = float(x), float(y)
                return
        raise RuntimeError("无法设置有效目标。")

    def sync(self):
        obj = self.objects[self.target_index]
        self.state.block_x, self.state.block_y = obj["x"], obj["y"]
        self.state.dragging = self.grasped is not None

    def proprio(self):
        s = self.state
        return np.array([s.cursor_x / 191, s.cursor_y / 127, float(s.left_held)], dtype=np.float32)

    def object_box(self):
        return self.box(self.target_index)

    def box(self, i):
        o = self.objects[i]
        return o["x"] - o["w"] / 2, o["y"] - o["h"] / 2, o["x"] + o["w"] / 2, o["y"] + o["h"] / 2

    def hit(self):
        s = self.state
        for i in reversed(range(len(self.objects))):
            a, b, c, d = self.box(i)
            if a <= s.cursor_x <= c and b <= s.cursor_y <= d:
                return i
        return None

    def placed(self):
        o, s = self.objects[self.target_index], self.state
        return abs(o["x"] - s.target_x) <= 22 - o["w"] / 2 and abs(o["y"] - s.target_y) <= 20 - o["h"] / 2

    def success(self):
        return bool(self.final_done)

    def potential(self):
        o, s = self.objects[self.target_index], self.state
        # 标量任务反馈，不作为观察输入；不使用任何分割标签。
        return -float(np.hypot(o["x"] - s.target_x, o["y"] - s.target_y) + .25 * np.hypot(s.cursor_x - o["x"], s.cursor_y - o["y"])) / 192

    def occluder(self):
        if self.variant not in ("occluded", "novel"):
            return None
        return (round(86 + 20 * np.sin(self.state.steps * np.pi / 36)), 27, 24, 76)

    def apply(self, action):
        s = self.state
        old_potential = self.potential()
        old_positions = [(o["x"], o["y"]) for o in self.objects]
        held = bool(action % 2)
        pressed, released = held and not s.left_held, s.left_held and not held
        reward = -.005
        if released:
            self.grasped = None
        if pressed:
            self.grasped = self.hit()
            if self.grasped is not None:
                o = self.objects[self.grasped]
                self.offset = (o["x"] - s.cursor_x, o["y"] - s.cursor_y)
                if self.grasped != self.target_index:
                    reward -= .08
        s.left_held = held
        dx, dy = displacement(action)
        s.cursor_x = float(np.clip(s.cursor_x + dx, 0, 191))
        s.cursor_y = float(np.clip(s.cursor_y + dy, 0, 127))
        for i, o in enumerate(self.objects):
            if i == self.grasped:
                o["x"], o["y"] = s.cursor_x + self.offset[0], s.cursor_y + self.offset[1]
                o["vx"], o["vy"] = 0., 0.
            else:
                o["x"] += o["vx"]
                o["y"] += o["vy"]
            self.clamp(o)
        contact_pairs = set()
        # 矩形刚体接触；被抓取物体是运动学物体，其余物体平分最小穿透修正。
        for _ in range(3):
            for i in range(len(self.objects)):
                for j in range(i + 1, len(self.objects)):
                    a, b = self.objects[i], self.objects[j]
                    px = (a["w"] + b["w"]) / 2 - abs(a["x"] - b["x"])
                    py = (a["h"] + b["h"]) / 2 - abs(a["y"] - b["y"])
                    if min(px, py) <= 0:
                        continue
                    contact_pairs.add((i, j))
                    axis, overlap = ("x", px) if px < py else ("y", py)
                    sign = 1. if b[axis] >= a[axis] else -1.
                    if i == self.grasped:
                        b[axis] += sign * overlap
                    elif j == self.grasped:
                        a[axis] -= sign * overlap
                    else:
                        a[axis] -= sign * overlap / 2
                        b[axis] += sign * overlap / 2
                    a["v" + axis], b["v" + axis] = b["v" + axis], a["v" + axis]
                    self.clamp(a)
                    self.clamp(b)
        self.contacts += len(contact_pairs)
        reward += float(np.clip(.3 * (self.potential() - old_potential), -.08, .08))
        s.steps += 1
        self.last_external = False
        completed = self.placed() and not s.left_held
        needs_switch = self.variant in ("switch", "novel")
        if completed:
            self.completed_goals += 1
            reward += 1.
        if needs_switch and not self.switched and (completed or s.steps == 36):
            if not completed:
                reward -= .25
            self.target_index = (self.target_index + int(self.rng.integers(1, len(self.objects)))) % len(self.objects)
            self.new_destination()
            self.stage, self.switched = 1, True
            self.cue_until = s.steps + 8
            self.last_external = True
        elif completed:
            self.final_done = True
        self.sync()
        self.last_reward = float(reward)
        return {"drag": any(p != (o["x"], o["y"]) for p, o in zip(old_positions, self.objects)),
                "release": released, "click": False, "pressed": pressed, "contacts": len(contact_pairs),
                "goal_completed": completed, "external": self.last_external}

    @staticmethod
    def clamp(o):
        for axis, size, extent in (("x", "w", 192), ("y", "h", 128)):
            low, high = o[size] / 2, extent - o[size] / 2
            if o[axis] < low or o[axis] > high:
                o["v" + axis] *= -1
            o[axis] = float(np.clip(o[axis], low, high))

    def snapshot(self):
        return {**vars(self.state), "variant": self.variant, "objects": copy.deepcopy(self.objects),
                "object_count": len(self.objects), "target_index": self.target_index,
                "completed_goals": self.completed_goals, "switched": self.switched}

class InteractionEnvironment(OwnedEnvironment):
    def __init__(self, seed, variant="interaction", object_count=None):
        super().__init__(seed, "drag")
        self.world = InteractionWorld(seed, variant, object_count)
        self.variant = variant
        rng = np.random.default_rng(seed ^ 0x517E)
        self.background = self.pg.Surface((192, 128))
        for y in range(0, 128, 12):
            for x in range(0, 192, 12):
                g = int(rng.integers(18, 35))
                self.pg.draw.rect(self.background, (g, g + 4, g + 10), (x, y, 12, 12))
        colors = [tuple(int(v) for v in rng.integers(70, 231, 3)) for _ in range(4)]
        self.textures = []
        for obj in self.world.objects:
            texture = self.pg.Surface((obj["w"], obj["h"]))
            cell = int(rng.choice([4, 6, 8]))
            for y in range(0, obj["h"], cell):
                for x in range(0, obj["w"], cell):
                    self.pg.draw.rect(texture, colors[int(rng.integers(4))], (x, y, cell, cell))
            self.pg.draw.rect(texture, (23, 28, 40), (0, 0, obj["w"], obj["h"]), width=1)
            self.textures.append(texture)

    def draw(self, labels=False):
        pg, world, s = self.pg, self.world, self.world.state
        canvas = pg.Surface((192, 128)) if labels else self.background.copy()
        if labels:
            canvas.fill((0, 0, 0))
        color = lambda c, label: (label,) * 3 if labels else c
        pg.draw.rect(canvas, color((210, 228, 239), 6), (round(s.target_x - 24), round(s.target_y - 22), 48, 44), width=2)
        for i, obj in enumerate(world.objects):
            rect = (round(obj["x"] - obj["w"] / 2), round(obj["y"] - obj["h"] / 2), obj["w"], obj["h"])
            if labels:
                pg.draw.rect(canvas, (i + 1,) * 3, rect)
            else:
                canvas.blit(self.textures[i], rect[:2])
                if i == world.target_index and (s.steps < world.cue_until or self.variant == "hinted"):
                    pg.draw.circle(canvas, (12, 15, 20), (round(obj["x"]), round(obj["y"])), 4)
                    pg.draw.circle(canvas, (250, 250, 250), (round(obj["x"]), round(obj["y"])), 2)
        if world.occluder():
            pg.draw.rect(canvas, color((60, 67, 81), 8), world.occluder())
        center = (round(s.cursor_x), round(s.cursor_y))
        pg.draw.circle(canvas, color((0, 0, 0), 7), center, 5)
        pg.draw.circle(canvas, color((255, 242, 106) if s.left_held else (245, 249, 255), 7), center, 3)
        pg.draw.line(canvas, color((255, 255, 255), 7), (center[0] - 6, center[1]), (center[0] + 6, center[1]))
        pg.draw.line(canvas, color((255, 255, 255), 7), (center[0], center[1] - 6), (center[0], center[1] + 6))
        return canvas

class GroundedWorld(InteractionWorld):
    def __init__(self, seed, level):
        if level not in LEVELS:
            raise ValueError(f"未知分级场景：{level}")
        rng = np.random.default_rng(seed ^ 0x601D)
        count = 1 if level == "basic" else (4 if level == "transfer" else int(rng.integers(2, 4)))
        super().__init__(seed, "hinted", count)
        self.level, self.controlled_target, self.controlled_distance = level, False, 0.
        self.cue_until = 10**9
        for obj in self.objects:
            obj["vx"] = float(rng.choice([-1, 1]) * rng.uniform(.25, .65)) if level in ("motion", "occlusion") else 0.
            obj["vy"] = float(rng.choice([-1, 1]) * rng.uniform(.2, .5)) if level in ("motion", "occlusion") else 0.

    def occluder(self):
        if self.level != "occlusion":
            return None
        return (round(86 + 26 * np.sin(self.state.steps * np.pi / 48)), 30, 18, 66)

    def apply(self, action):
        if self.final_done:
            self.last_reward, self.last_external = 0., False
            return dict(drag=False, release=False, click=False, pressed=False, contacts=0, goal_completed=False, external=False)
        before = copy.deepcopy(self.objects[self.target_index])
        event = super().apply(action)
        if self.grasped == self.target_index:
            self.controlled_target = True
            current = self.objects[self.target_index]
            self.controlled_distance += np.hypot(current["x"] - before["x"], current["y"] - before["y"])
        # 动态物体自己飘进目标不算模型完成；必须确实抓取并移动过目标物体。
        if event["goal_completed"] and not (self.controlled_target and self.controlled_distance >= 8):
            self.final_done = False
            self.completed_goals -= 1
            self.last_reward -= 1.
            event["goal_completed"] = False
        return event

    def snapshot(self):
        return {**super().snapshot(), "level": self.level, "controlled_target": self.controlled_target,
                "controlled_distance": float(self.controlled_distance)}

class GroundedEnvironment(InteractionEnvironment):
    def __init__(self, seed, level):
        world = GroundedWorld(seed, level)
        super().__init__(seed, "hinted", len(world.objects))
        self.world, self.level = world, level
