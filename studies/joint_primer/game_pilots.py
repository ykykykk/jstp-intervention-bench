"""Live inference only. No camera judge, hidden rig state, saved answers or network downloads."""
from dataclasses import dataclass

import torch

from lategoal.rig import EXAM, TRAIN
from mechanism.data import sha
from .train import HERE, load


@dataclass(frozen=True)
class Prediction:
    side: int
    # Camera-coordinate displacement under reference handle A, at 12 overlapping sites.
    optics: tuple | None


@dataclass
class ProbeView:
    x: torch.Tensor
    address: torch.Tensor | None
    tokens: torch.Tensor | None
    optics: tuple | None


def checked_file(path, digest):
    if sha(path) != digest:
        raise ValueError(f"冻结文件与实验记录不一致：{path}")


class FrozenPilot:
    def __init__(self, arm, budget, seed):
        torch.set_num_threads(2)
        self.arm = arm
        self.extractor = self.readout = None
        if arm == "joint":
            self.device = "cpu"
            self.net = load(HERE / "evidence/train", arm, budget, seed).requires_grad_(False)
            return
        if not torch.cuda.is_available():
            raise RuntimeError("这五种外部适配沿用论文的 CUDA 推理接口；人工及完整 JSTP 不需要 GPU。")
        self.device = "cuda"
        torch.backends.mha.set_fastpath_enabled(False)
        if arm in ("smol", "dino"):
            from .official_train import DEFAULT, restore
            self.net = restore(DEFAULT, arm, budget, seed).eval().requires_grad_(False)
            return
        from studies.fourtrack_foundations.common import E, jread
        from studies.fourtrack_foundations.features import Video
        from studies.fourtrack_foundations.policies import Policy
        folder = E / arm / "models" / f"{arm}-{budget}-{seed}"
        rec = jread(folder / "freeze.json")
        path = folder / "adapter.pt"
        checked_file(path, rec["weights"])
        self.weights = torch.load(path, map_location="cpu", weights_only=True)
        self.extractor = Video() if arm == "vjepa21" else Policy(arm)
        source = jread(E / arm / "train-features.json")
        if any(self.extractor.meta[k] != source[k] for k in ("base_sha256", "upstream_commit")):
            raise ValueError("视觉基座与论文中的冻结适配不匹配")

    @torch.inference_mode()
    def observe(self, observation):
        """Encode the completed probe once. No target has entered this path."""
        x = torch.as_tensor(observation[None], device=self.device)
        eye = tokens = address = None
        if self.arm == "joint":
            h = self.net.observe(x)
            z = self.net.representation(h)
            eye = self.net.eye(z)[0]
            # The A expectation and direct path meet in the original trained motor cell.
            address = self.net.motor(h, z)
        elif self.arm in ("dino", "smol"):
            from .official_train import dino_eye, visual
            v = visual(self.net, self.arm, x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.arm == "dino":
                    one = torch.ones_like(x[:, :1])
                    a = dino_eye(self.net, v, x, -one)
                    b = dino_eye(self.net, v, x, one)
                    eye = a[0]
                    address = self.net.utility(b) - self.net.utility(a)
                else:
                    tokens = v
        else:
            from studies.fourtrack_foundations.train import model
            f = {k: v.to(x.device) for k, v in self.extractor(x).items()}
            if self.readout is None:
                if self.arm != "vjepa21":
                    f.update(self.extractor.projection())
                self.readout = model(self.arm, f, self.extractor.meta).to(x.device)
                self.readout.load_state_dict(self.weights)
                self.readout.eval().requires_grad_(False)
                del self.weights
            pred = self.readout(f, torch.zeros(1, dtype=torch.long, device=x.device), x[:, 144:])[0]
            address = pred[None, :4]
            eye = pred[4:16]
        if eye is not None and not torch.isfinite(eye).all():
            raise FloatingPointError(f"{self.arm} 给出了非有限的光学预期")
        return ProbeView(x, address, tokens, None if eye is None else tuple(eye.float().tolist()))

    @torch.inference_mode()
    def predict(self, view, card_id):
        """One request, one late card. The next card is not available until this choice is sealed."""
        x = view.x
        goal = torch.as_tensor(EXAM[card_id:card_id+1], dtype=x.dtype, device=x.device)
        if self.arm == "smol":
            from .official_train import TokenPrefix
            ids = torch.tensor([card_id + len(TRAIN)], device=x.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                score = TokenPrefix(self.net.native).score(self.net, view.tokens, x, goal, ids)
        elif self.arm == "dino":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                score = view.address @ goal.T
        elif self.arm == "joint":
            score = (view.address * goal).sum(-1)
        else:
            score = view.address @ goal.T
        if not torch.isfinite(score).all():
            raise FloatingPointError(f"{self.arm} 给出了非有限的动作分数")
        return Prediction(1 if score.item() >= 0 else -1, view.optics)

    def close(self):
        if self.arm in ("act", "diffusion"):
            self.extractor.hook.remove()
