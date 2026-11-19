"""Match the exact ordered downstream rows and goal cards of the official transfer arms."""
import hashlib,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models import make
from .train import C,HERE,DATA,DEFAULT,logloss
from .transfer import P,cu,seal,execute
from mechanism.data import read,sha,dump
from mechanism.train import config
from lategoal.rig import TRAIN
class View(nn.Module):
    def __init__(self,net):super().__init__();self.net=net
    def probe(self,x):return self.net.probe(x)
    def choices(self,x,goals,first_goal):
        h=self.net.observe(x);n,k=len(h),len(goals);p=x[:,156].repeat_interleave(k)
        return self.net.policy(h.repeat_interleave(k,dim=0),p,goals.repeat(n,1)).reshape(n,k)
def run(arm,b):
    seed=P["seed"];config(seed,"cuda");name="matched-"+arm;d=DEFAULT/"transfer"/name;d.mkdir(parents=True,exist_ok=False)
    n=len(b["start"])*8;rng=np.random.default_rng(seed);idx=rng.permutation(n);cards=rng.integers(len(TRAIN),size=n)
    digest=hashlib.sha256(np.stack((idx,cards),1).astype("<i4").tobytes()).hexdigest()
    m=make(arm).cuda();opt=torch.optim.AdamW(m.parameters(),lr=C["lr"],weight_decay=.0001);tic=time.monotonic()
    for k,(rr,gg) in enumerate(zip(idx.reshape(-1,32),cards.reshape(-1,32)),1):
        i,s,g,pi=np.unravel_index(rr,(len(b["start"]),2,2,2));goal=TRAIN[gg];p=cu((2*pi-1).astype("float32"))
        x=cu(b["hist"][i,s,g,pi]);ys=cu(np.where((b["motion"][i,s,g,0]*goal).sum(-1)>0,-1,1)).float()
        start=cu(b["start"][i]);h=m.observe(x);loss=.18*logloss(m.probe(start),start[:,-1])+logloss(m.policy(h,p,cu(goal).float()),ys)
        if arm!="direct":
            one=torch.ones_like(p);f=m.eye(m.future(torch.cat([h,h]),torch.cat([-one,one])));y=cu(b["eye"][i,s,g]).transpose(0,1).reshape(-1,12)
            loss=loss+3*F.mse_loss(f,y)
        if not torch.isfinite(loss):raise FloatingPointError((arm,k))
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
    f=d/"adapter.pt"
    with f.open("xb") as q:torch.save(m.cpu().state_dict(),q)
    dump(d/"freeze.json",dict(arm=arm,adapter=sha(f),data=sha(DATA/"train.npz"),schedule=digest,
        protocol=sha(HERE/"bridge-plan.md"),code=sha(HERE/"bridge.py"),params=sum(p.numel() for p in m.parameters()),seed=seed,updates=k,examples=n,seconds=round(time.monotonic()-tic,2)))
    m=make(arm).cuda();m.load_state_dict(torch.load(f,map_location="cuda",weights_only=True));m=View(m.eval()).eval()
    for panel in C["panels"]:seal(name,m,d,panel)
    for panel in C["panels"]:execute(d,panel)
if __name__=="__main__":
    if sha(DATA/"train.npz")!=C["train_sha256"]:raise ValueError("训练数据已变")
    b=read(DATA/"train.npz")
    for arm in ("partial","direct","world"):run(arm,b)

