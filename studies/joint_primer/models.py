"""Controllers see public observations and requested goals, never target motion or rig internals."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from lategoal.rig import HIST, F as GRID

def block(n,w):
    return nn.Sequential(nn.Linear(n,w),nn.LayerNorm(w),nn.SiLU(),nn.Linear(w,w))

class Joint(nn.Module):
    """Observe once; a prospective stroke and a motor commitment reuse the same transition."""
    def __init__(self,arm,w):
        super().__init__()
        self.arm=arm
        self.encoder=block(HIST,w)
        self.cell=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w))
        self.events=nn.Embedding(3,w)  # stroke A, stroke B, motor commitment
        self.eye=nn.Linear(w,12)
        self.return_path=nn.Linear(w,w,bias=False)
        self.address=nn.Linear(w,4)
    def observe(self,x):return self.encoder(x)
    def step(self,h,event):return h+self.cell(h+event)
    def future(self,h,side):return self.step(h,self.events(side))
    def motor(self,h,z):
        if self.arm=='joint':h=h+self.return_path(z)
        return self.address(self.step(h,self.events.weight[2]))
    def representation(self,h):
        return h if self.arm=='direct' else self.step(h,self.events.weight[0])
    def score(self,h,g):
        z=self.step(h,self.events.weight[0]) if self.arm=='joint' else None
        return (self.motor(h,z)*g).sum(-1)

class World(Joint):
    """Explicit forward model: value the two actual action hypotheses, no access to camera truth."""
    def __init__(self,w):
        super().__init__('world',w)
        del self.return_path,self.address
        self.value=nn.Sequential(nn.Linear(12,w),nn.SiLU(),nn.Linear(w,4))
    def score(self,h,g):
        a=self.value(self.eye(self.step(h,self.events.weight[0])))
        b=self.value(self.eye(self.step(h,self.events.weight[1])))
        return ((b-a)*g).sum(-1)

class Masked(nn.Module):
    """Same short token stream for missing actuator or missing optical values; MTM-inspired, not upstream MTM."""
    def __init__(self,w):
        super().__init__()
        self.parts=nn.ModuleList([nn.Linear(k,w) for k in (GRID,GRID,HIST-2*GRID)])
        self.event=nn.Embedding(3,w)
        self.pos=nn.Parameter(torch.randn(4,w)*.04)
        self.block=nn.TransformerEncoderLayer(w,4,2*w,dropout=0,batch_first=True,
                                             activation='gelu',norm_first=True)
        self.norm=nn.LayerNorm(w);self.eye=nn.Linear(w,12);self.address=nn.Linear(w,4)
    def observe(self,x):
        return torch.stack([m(v) for m,v in zip(self.parts,x.split([GRID,GRID,HIST-2*GRID],dim=1))],1)
    def query(self,h,e):
        return self.norm(self.block(torch.cat((h,e[:,None]),1)+self.pos)[:,-1])
    def future(self,h,side):return self.query(h,self.event(side))
    def representation(self,h):return self.query(h,self.event.weight[0].expand(len(h),-1))
    def score(self,h,g):
        return (self.address(self.query(h,self.event.weight[2].expand(len(h),-1)))*g).sum(-1)

class Flow(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.encoder=block(HIST,w); self.card=nn.Linear(4,w,bias=False)
        self.field=nn.Sequential(nn.Linear(w+3,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,1))
    def observe(self,x):return self.encoder(x)
    def representation(self,h):return h
    def velocity(self,h,x,t):
        return self.field(torch.cat((h,x[:,None],torch.sin(math.pi*t)[:,None],torch.cos(math.pi*t)[:,None]),1)).squeeze(1)
    def loss(self,h,g,y):
        t=torch.rand_like(y);noise=torch.randn_like(y)
        return F.mse_loss(self.velocity(h*self.card(g),(1-t)*noise+t*y,t),y-noise)
    def score(self,h,g):
        c=h*self.card(g);x=torch.zeros_like(g[:,0])
        for i in range(8):x=x+self.velocity(c,x,torch.full_like(x,i/8))/8
        return x

def make(arm,w):
    if arm in ('joint','aux','direct'):return Joint(arm,w)
    return {'world':World,'masked':Masked,'flow':Flow}[arm](w)
