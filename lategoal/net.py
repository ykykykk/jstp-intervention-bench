"""一个网络：探查、自身运动以及对同一运动的未来光学状态。"""
import math
import torch
from torch import nn
from .rig import C,F,HIST

def sign(x):return torch.where(x>=0,1.,-1.)
class Net(nn.Module):
    def __init__(self):
        super().__init__();w=C["width"]
        def enc(d):return nn.Sequential(nn.Linear(d,w),nn.LayerNorm(w),nn.SiLU(),nn.Linear(w,w),nn.LayerNorm(w))
        self.before=enc(F+1);self.observe=enc(HIST)
        self.transition=enc(w+1);self.eye=nn.Linear(w,12)
        self.combine=enc(w*2);self.question=nn.Linear(4,w,bias=False)
        self.port=nn.Linear(w,1);self.motor_bias=nn.Linear(w,1)
    def probe(self,initial):return self.port(self.before(initial)).squeeze(-1)
    def future(self,h,drive):return self.transition(torch.cat((h,drive[:,None]),1))
    def issue(self,h,z,g,arm):
        # 目标到达后仍可直接查看完整原经历；不计算两个把手的候选收益。
        zz=z if arm=="joint" else torch.zeros_like(z)
        state=self.combine(torch.cat((h,zz),1))
        return self.port(state*self.question(g)).squeeze(-1)+self.motor_bias(state).squeeze(-1)
