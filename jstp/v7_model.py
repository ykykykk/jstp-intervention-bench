"""单一 RGB 联合转移预测器。无对象槽、分割头、价值头、动作头或真值状态输入。"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .v7_world import W,H,TABLE,D


def rgb(x,device):
    return torch.as_tensor(x,device=device).movedim(-1,-3).float().div_(255.)


def picture(t):
    return t.detach().float().clamp(0,1).mul(255).round().byte().movedim(-3,-1).cpu().numpy()


def shift(image,delta):
    b,c,h,w=image.shape; delta=delta.long()
    xx=torch.arange(w,device=image.device)[None,None,:]-delta[:,0,None,None]
    yy=torch.arange(h,device=image.device)[None,:,None]-delta[:,1,None,None]
    j=(yy.clamp(0,h-1)*w+xx.clamp(0,w-1)).reshape(b,1,-1)
    out=image.flatten(2).gather(2,j.expand(-1,c,-1)).reshape(b,c,h,w)
    return out*((xx>=0)&(xx<w)&(yy>=0)&(yy<h))[:,None]


class JointFuture(nn.Module):
    def __init__(self):
        super().__init__()
        yy,xx=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing="ij")
        self.register_buffer("grid",torch.stack([xx,yy])[None],persistent=False)
        self.register_buffer("scale",torch.tensor([2/(W-1),2/(H-1),1/D,1/D]),persistent=False)
        self.register_buffer("offset",torch.tensor([-1.,-1.,0.,0.]),persistent=False)
        self.history=nn.Sequential(nn.Conv2d(9,16,3,padding=1),nn.SiLU(),
                                   nn.Conv2d(16,20,3,stride=2,padding=1),nn.SiLU(),
                                   nn.AdaptiveMaxPool2d((3,4)),nn.Flatten())
        self.context=nn.Sequential(nn.Linear(20*3*4+8,80),nn.SiLU(),
                                   nn.Linear(80,32),nn.SiLU())
        # 共享逐位置计算。没有定义哪部分算对象、哪一侧应跟哪个把手、何时算共动。
        self.spatial=nn.Sequential(nn.Conv2d(3+3+2+2+1,24,1),nn.SiLU(),
                                   nn.Conv2d(24,28,3,padding=1),nn.SiLU())
        self.decode=nn.Sequential(nn.Conv2d(28+32,40,1),nn.SiLU(),
                                  nn.Conv2d(40,32,1),nn.SiLU(),nn.Conv2d(32,3,1))

    def forward(self,before,peak,current,past_action,candidate):
        p=past_action*self.scale+self.offset; q=candidate*self.scale+self.offset
        # 无试探的 proprio 用当前候选点代替是不允许的；全零过去跨所有候选相同。
        change=peak-before
        event=torch.cat([current,change,change.abs()],dim=1)
        past=self.history(F.avg_pool2d(event,4))
        z=self.context(torch.cat([past,p,q],dim=1))
        grid=self.grid.expand(current.shape[0],-1,-1,-1)
        rel=grid-q[:,:2,None,None]
        touch=torch.exp(-24*rel.square().sum(1,keepdim=True))
        f=self.spatial(torch.cat([current,change,grid,rel,touch],dim=1))
        signal=self.decode(torch.cat([f,z[:,:,None,None].expand(-1,-1,H,W)],dim=1))
        # 平移是通用运动先验。网络输出三个无界、有符号的视觉信号通道，不是 alpha/二值掩码。
        # 零填充边界没有被抓物体；去掉任意常数偏置，避免边界伪影伪装成视觉证据。
        signal=signal-signal[:,:,:1,:1]
        return current+shift(signal,candidate[:,2:])-signal


def goal_cost(futures,goals,current):
    # futures [batch,2,3,H,W], goals [batch,2,3,TABLE,W]；只比较公开工作区全幅图像。
    pred=futures[...,:TABLE,:]
    scale=(goals-current[:,None,:,:TABLE,:]).square().mean((-3,-2,-1)).clamp_min(1.e-5)
    return (pred[:,None]-goals[:,:,None]).square().mean((-3,-2,-1))/scale[:,:,None]


def pixel_loss(pred,truth,origin):
    change=(truth-origin).abs().amax(dim=-3,keepdim=True)
    w=1+12*change
    err=F.smooth_l1_loss(pred,truth,reduction="none",beta=.06)
    normal=((truth-origin).abs()*w).mean().detach().clamp_min(.002)
    return (err*w).mean()/normal


def pair_forward(model,cur,peak,probe,cmd):
    n=len(cur)
    def expand(t): return t.flatten(0,2)
    c=expand(cur[:,None,None].expand(-1,2,2,-1,-1,-1))
    e=expand(peak[:,:,None].expand(-1,-1,2,-1,-1,-1))
    p=expand(probe[:,None,None].expand(-1,2,2,-1))
    a=expand(cmd[:,None].expand(-1,2,-1,-1))
    return model(c,e,c,p,a).reshape(n,2,2,3,H,W)


def learned_pair_loss(pred,truth,cur,goal):
    image=pixel_loss(pred,truth,cur[:,None,None])
    # 成对差值的靶也只是实拍 RGB 之差，不使用连接、成员、最佳把手或位移标签。
    contrast=0.
    for axis in (1,2):
        dt=truth.select(axis,0)-truth.select(axis,1)
        dp=pred.select(axis,0)-pred.select(axis,1)
        contrast+=pixel_loss(dp,dt,torch.zeros_like(dt))*.5
    n=len(cur)
    goals=goal[:,None].expand(-1,2,-1,-1,-1,-1).flatten(0,1)
    now=cur[:,None].expand(-1,2,-1,-1,-1).flatten(0,1)
    costs=goal_cost(pred.flatten(0,1),goals,now)
    real=goal_cost(truth.flatten(0,1),goals,now)
    # 目标优劣来自真实画面距离。没有单独的可学习选择器。
    ranking=F.kl_div(F.log_softmax(-6*costs,dim=-1),F.softmax(-6*real,dim=-1),
                     reduction="sum")/(4*n)
    return image+.55*contrast+.12*ranking


def displacement(images,reference):
    """只在验收时用两扇固定观察窗，以亮度质心读出 RGB 可见位移；不碰物体坐标。"""
    def centers(v):
        v=np.asarray(v,dtype=np.float64)[...,:TABLE,:,:]
        bright=np.maximum(v.mean(-1)-46.,0)
        out=[]
        for x in (0,W//2):
            mass=bright[...,x:x+W//2]; total=mass.sum((-1,-2))
            if (total<1.).any(): raise ValueError("评分可见性不足，不能当作零运动")
            yy,xx=np.mgrid[:TABLE,x:x+W//2]
            out.append(np.stack([(mass*xx).sum((-1,-2))/total,(mass*yy).sum((-1,-2))/total],axis=-1))
        return np.stack(out,axis=-2)
    return centers(images)-centers(reference)
