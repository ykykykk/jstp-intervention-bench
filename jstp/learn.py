"""同预算 RGB/动作对照；监督来自可见结果。候选把手的顺序本身不携带正确答案。"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .v7_world import H,W,TABLE,D
from .v7_model import JointFuture,pixel_loss,pair_forward

class ResidualRGB(JointFuture):
    """不提供位移搬运公式；空洞卷积使每个输出可读取距它六像素以上的来源。"""
    def __init__(self):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv2d(11,20,3,padding=1),nn.SiLU(),
                    nn.Conv2d(20,20,3,padding=2,dilation=2),nn.SiLU(),
                    nn.Conv2d(20,28,3,padding=4,dilation=4),nn.SiLU())
    def forward(self,before,peak,current,past_action,candidate):
        p=past_action*self.scale+self.offset; q=candidate*self.scale+self.offset
        diff=peak-before
        past=self.history(F.avg_pool2d(torch.cat([current,diff,diff.abs()],1),4))
        z=self.context(torch.cat([past,p,q],1))
        grid=self.grid.expand(len(current),-1,-1,-1); rel=grid-q[:,:2,None,None]
        f=self.spatial(torch.cat([current,diff,grid,rel,torch.exp(-24*rel.square().sum(1,keepdim=True))],1))
        return current+self.decode(torch.cat([f,z[:,:,None,None].expand(-1,-1,H,W)],1))

def costs(pred,goals,mask):
    # [B,J,A,C,H,W] -> [B,J,G,A]. 正常化为指定区域的 MSE，无需知道另一个目标。
    err=(pred[:,:,None,:,:,:TABLE,:]-goals[:,None,:,None]).square().mean(-3)
    w=mask[:,None,None,None]
    return (err*w).sum((-1,-2))/(w.sum((-1,-2)).clamp_min(1)*.035)

def branch_loss(pred,real,cur,goals,mask,paired):
    loss=pixel_loss(pred,real,cur[:,None,None])
    if paired:
        for axis in (1,2):
            t=real.select(axis,0)-real.select(axis,1); dp=pred.select(axis,0)-pred.select(axis,1)
            loss=loss+.275*pixel_loss(dp,t,torch.zeros_like(t))
        c=costs(pred,goals,mask); r=costs(real,goals,mask)
        loss=loss+.12*F.kl_div(F.log_softmax(-6*c,-1),F.softmax(-6*r,-1),reduction="sum")/(len(cur)*4)
    return loss

class HistoryPolicy(nn.Module):
    """读所有视觉历史和目标，但测试时不生成任何候选未来，输出每个动作的兼容分。"""
    def __init__(self):
        super().__init__()
        yy,xx=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing="ij")
        self.register_buffer("grid",torch.stack([xx,yy])[None],persistent=False)
        self.register_buffer("scale",torch.tensor([2/(W-1),2/(H-1),1/D,1/D]),persistent=False)
        self.register_buffer("off",torch.tensor([-1.,-1.,0.,0.]),persistent=False)
        self.obs=nn.Sequential(nn.Conv2d(15,20,3,padding=1),nn.SiLU(),
                               nn.Conv2d(20,24,3,stride=2,padding=1),nn.SiLU(),
                               nn.Conv2d(24,24,3,stride=2,padding=1),nn.SiLU(),
                               nn.AdaptiveMaxPool2d((3,6)),nn.Flatten())
        self.rank=nn.Sequential(nn.Linear(24*18+8,72),nn.SiLU(),nn.Linear(72,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,c,e,p,a,g,mask):
        n,j=e.shape[:2]; k=g.shape[1]; m=a.shape[1]
        def bc(x):return x.expand(n,j,k,*x.shape[3:]).flatten(0,2)
        delta=e-c[:,None]; maskfull=F.pad(mask[:,None],(0,0,0,H-TABLE))
        gd=F.pad((g-c[:,None,:,:TABLE])*mask[:,None,None],(0,0,0,H-TABLE))
        inp=torch.cat([bc(c[:,None,None]),bc(delta[:,:,None]),bc(delta.abs()[:,:,None]),bc(gd[:,None]),
                       bc(maskfull[:,None,None]),self.grid.expand(n*j*k,-1,-1,-1)],1)
        z=self.obs(F.avg_pool2d(inp,2)).reshape(n,j,k,-1)
        past=(p*self.scale+self.off)[:,None,None,None].expand(-1,j,k,m,-1)
        query=(a*self.scale+self.off)[:,None,None].expand(-1,j,k,-1,-1)
        return self.rank(torch.cat([z[:,:,:,None].expand(-1,-1,-1,m,-1),past,query],-1)).squeeze(-1)

def model_for(key,external=None):
    if key in ("jstp","transport_factual"):return JointFuture()
    if key in ("rgb_paired","rgb_factual"):return ResidualRGB()
    if key=="history_policy":return HistoryPolicy()
    if key=="dino_wm":
        from .foundation import DinoDynamics
        return DinoDynamics(external)
    raise KeyError(key)

class Bank:
    def __init__(self,data,device):
        self.d={k:torch.as_tensor(v,device=device) for k,v in data.items()}
        self.n=len(self.d["c"])
    def batch(self,ix):
        out={}
        for k,v in self.d.items():
            t=v[ix]
            if v.dtype==torch.uint8:
                if k!="mask":t=t.movedim(-1,-3)
                t=t.float().div_(255)
            elif v.dtype==torch.float16:t=t.float()
            out[k]=t
        return out

def direct_loss(m,key,b):
    c,a,f=b["c"],b["a"],b["f"]; p=torch.zeros_like(a)
    if key=="dino_wm":
        return F.mse_loss(m.predict(c,c,c,p,a),f)
    if key=="history_policy":
        hold=a.clone(); hold[:,2:]=0
        options=torch.stack([a,hold],1); goals=torch.stack([f[:,:,:TABLE],c[:,:,:TABLE]],1)
        mask=torch.ones((len(c),TABLE,W),device=c.device)
        logits=m(c,c[:,None],p,options,goals,mask)
        truth=costs(torch.stack([f,c],1)[:,None],goals,mask).argmin(-1)
        return F.cross_entropy(logits.reshape(-1,2),truth.flatten())
    return pixel_loss(m(c,c,c,p,a),f,c)

def pair_outputs(m,key,b):
    c,e,p,a,g,mask=[b[k] for k in ("c","e","p","a","g","mask")]
    if key=="history_policy":return None,-m(c,e,p,a,g,mask)
    if key=="dino_wm":
        f=m.branches(c,e,p,a); return f,m.distances(f,g,mask)
    f=pair_forward(m,c,e,p,a)
    return f,costs(f,g,mask)

def pair_loss(m,key,b):
    f,cost=pair_outputs(m,key,b)
    if key=="history_policy":
        true=costs(b["f"],b["g"],b["mask"]).argmin(-1)
        return F.cross_entropy(-cost.reshape(-1,2),true.flatten())
    if key=="dino_wm":return F.mse_loss(f,b["f"])
    return branch_loss(f,b["f"],b["c"],b["g"],b["mask"],key in ("jstp","rgb_paired"))
