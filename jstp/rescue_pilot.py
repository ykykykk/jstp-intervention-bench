"""小游戏的代驾：冻结 V7 想象画面，旧演示的收益读出决定试探或出航。"""
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import ROOT, read_json, sha
from .rescue_mission import matches, rig
from .v7_model import JointFuture, displacement, picture, rgb
from .v7_world import H, W, World


class ReturnReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Sequential(nn.LayerNorm(240), nn.Linear(240, 96), nn.SiLU())
        self.after = nn.Sequential(nn.Linear(96 + 32 + 14, 128), nn.SiLU(),
                                   nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, base, hist, past, action, goal, x, fee, seen):
        p = past * base.scale + base.offset
        a = action * base.scale + base.offset
        context = base.context(torch.cat([hist, p, a], dim=1))
        public = torch.cat([p, a, F.one_hot(goal, 3), x[:, None] / W, fee[:, None], seen[:, None]], dim=1)
        return self.after(torch.cat([self.readout(hist), context, public], dim=1))[:, 0]


def sense(base, current, peak):
    change = peak - current
    return base.history(F.avg_pool2d(torch.cat([current, change, change.abs()], dim=1), 4))


def cost(x, handle_x, is_probe):
    return abs(x - handle_x) / W * .29 + .17 * is_probe


def learn_returns(base, head, device, progress):
    """旧窗口的两段学习：实际交付收益；再用学出的后续收益估计试探的价值。"""
    n = 1536
    rng = np.random.default_rng(81730032)
    histories, actions, parking, rewards = [], [], [], []
    tensor = lambda a, dtype: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    for i in range(0, n, 32):
        progress(f"学习试航经历  ·  {i} / {n}")
        bases, peaks = [], []
        for seed in range(31_000_000_000 + i, 31_000_000_000 + i + 32):
            board, cmds, dy = rig(seed)
            goals = board.goals(dy)
            start = World(board, 0).camera()
            frames, outcomes = [start], []
            for wiring in (0, 1):
                for cmd in cmds[2:]:
                    frames.append(World(board, wiring).probe(cmd)[1])
                finals = [World(board, wiring).execute(cmd) for cmd in cmds[:2]]
                outcomes.append([[matches(frame, goals, k) for k in range(3)] for frame in finals])
            bases.append(start); peaks.append(frames); actions.append(cmds); rewards.append(outcomes)
            parking.append(np.clip(cmds[rng.integers(2), 0] + rng.integers(-9, 10), 0, W - 1))
        with torch.no_grad():
            c = rgb(np.stack(bases), device).repeat_interleave(5, dim=0)
            e = rgb(np.asarray(peaks).reshape(-1, H, W, 3), device)
            histories.append(sense(base, c, e).reshape(-1, 5, 240))

    hist = torch.cat(histories)
    acts, xs, ys = (tensor(v, torch.float32) for v in (actions, parking, rewards))
    product = lambda *sizes: torch.cartesian_prod(*(torch.arange(k, device=device) for k in sizes))
    terminal = product(n, 2, 3, 2, 3)  # layout, wiring, prior probe (none/A/B), delivery, goal
    tests = product(n, 2, 2, 3)
    probes = torch.stack([tests[:, 0], tests[:, 1], torch.zeros_like(tests[:, 0]), tests[:, 2] + 2, tests[:, 3]], 1)
    t = tests.repeat_interleave(2, dim=0)
    later = torch.stack([t[:, 0], t[:, 1], t[:, 2] + 1, torch.arange(2, device=device).repeat(len(tests)), t[:, 3]], 1)

    def inputs(rows):
        i, wiring, k, a, g = rows.T
        # wiring 只索引真实相机的观测；不是给收益读出的特征。
        obs = torch.where(k == 0, k, 2 * wiring + k)
        past = torch.where((k > 0)[:, None], acts[i, k + 1], 0.)
        x = torch.where(k > 0, acts[i, k + 1, 0], xs[i])
        fee = cost(x, acts[i, a, 0], (a >= 2).float())
        return hist[i, obs], past, acts[i, a], g, x, fee, (k > 0).float()

    opt = torch.optim.AdamW(head.parameters(), lr=.002)
    for phase, steps in enumerate((670, 610)):
        if phase:
            with torch.no_grad():
                continuation = torch.cat([head(base, *inputs(q)) for q in later.split(2048)])
                probe_y = continuation.reshape(-1, 2).max(dim=1).values - inputs(probes)[5]
        for step in range(steps):
            if step % 24 == 0:
                progress(f"{'学习试探代价' if phase else '学习交付后果'}  ·  {step + 1} / {steps}")
            q = terminal[torch.randint(len(terminal), (384,), device=device)]
            i, wiring, _, a, g = q.T
            features = inputs(q)
            loss = F.mse_loss(head(base, *features), ys[i, wiring, a, g] - features[5])
            if phase:
                j = torch.randint(len(probes), (256,), device=device)
                loss = loss + F.mse_loss(head(base, *inputs(probes[j])), probe_y[j])
            opt.zero_grad(); loss.backward(); opt.step()


class Pilot:
    def __init__(self, base, head, device):
        self.base, self.head, self.device = base, head, device

    @classmethod
    def prepare(cls, progress):
        torch.set_num_threads(2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folder = ROOT / "evidence/legacy7"
        path = folder / "frozen.pt"
        digest = read_json(folder / "freeze.json")["sha256"]
        if sha(path) != digest:
            raise ValueError("冻结的 V7 权重与归档指纹不一致")
        base = JointFuture().to(device).eval()
        base.load_state_dict(torch.load(path, map_location=device, weights_only=True)["weights"])
        base.requires_grad_(False)
        torch.manual_seed(81730032)
        head = ReturnReadout().to(device)
        cache = ROOT / "work/rescue-game" / ("returns-v1-" + digest[:16])
        cache.mkdir(parents=True, exist_ok=True)
        saved = sorted(cache.glob("pilot-*.pt"))
        if saved:
            progress("载入本机的试航经验")
            head.load_state_dict(torch.load(saved[0], map_location=device, weights_only=True))
        else:
            learn_returns(base, head, device, progress)
            progress("收好试航经验")
            # 只发布已关闭的唯一文件；同时开两个窗口也不用替换在读的清单或权重。
            with NamedTemporaryFile(dir=cache, prefix="pilot-", suffix=".partial", delete=False) as stream:
                tmp = Path(stream.name)
            try:
                torch.save(head.state_dict(), tmp)
                tmp.rename(tmp.with_suffix(".pt"))
            finally:
                tmp.unlink(missing_ok=True)
        return cls(base, head.eval(), device)

    @torch.inference_mode()
    def decide(self, frame, peak, past, commands, order, hand_x):
        """只接受相机与公开任务/手部信息，不接受 World 或隐藏挂接。"""
        device = self.device
        cur, pk = rgb(frame, device)[None], rgb(peak, device)[None]
        to = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)
        seen = bool(np.any(past[2:]))
        acts = to(commands[:2] if seen else commands)
        n = len(acts)
        p = to(past).expand(n, -1)
        h = sense(self.base, cur, pk).expand(n, -1)
        x = to([hand_x] * n)
        fee = cost(x, acts[:, 0], (acts[:, 2] != 0).float())
        value = self.head(self.base, h, p, acts, torch.full((n,), order, device=device),
                          x, fee, to([seen] * n))
        # 右屏是 V7 的两个事前 RGB 预测；不拿模拟器试跑候选动作。
        c = cur.expand(2, -1, -1, -1)
        futures = self.base(c, pk.expand_as(c), c, p[:2], acts[:2])
        motion = displacement(picture(futures), frame)[:, :, 1]
        return int(value.argmax().item()), motion
