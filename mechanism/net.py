"""单份共享权重；同一个更新块和读出口用于两项任务，不使用候选列表/价值/分割器。"""
import torch
from torch import nn
from torch.nn import functional as F
from .data import R,CTX,CFG,CODE

def electrical(v):
    return (v>.35).to(v.dtype)-(v<-.35).to(v.dtype)
def levels(x):return torch.round(x.clamp(0,1)*15)/15
def hard(logits):
    p=torch.sigmoid(logits)
    return p+((p>=.5).to(p.dtype)-p).detach()
def image(out):return torch.sigmoid(out[:,:2*R])
def motor(out):return torch.tanh(out[:,2*R])
def felt(out):return torch.tanh(out[:,2*R+1])*2
def lang_fields(out):
    s=2*R+2
    return out[:,s:s+3],out[:,s+3:s+6],out[:,s+6:s+9],out[:,s+9:s+17]

class Core(nn.Module):
    def __init__(self):
        super().__init__(); w=CFG["width"];self.w=w
        self.obs=nn.Sequential(nn.Linear(CTX,w),nn.LayerNorm(w),nn.SiLU(),nn.Linear(w,w))
        self.loop=nn.Sequential(nn.Linear(w*2+2*R+24,w),nn.LayerNorm(w),nn.SiLU(),nn.Linear(w,w),nn.SiLU())
        self.port=nn.Linear(w,2*R+19)

    def step(self,ctx,phase,prior=None,eye=None,drive=None,card=None):
        n=len(ctx); state=ctx.new_zeros(n,self.w) if prior is None else prior
        retinal=ctx.new_zeros(n,2*R) if eye is None else eye
        cue=ctx.new_zeros(n,24);cue[:,phase]=1
        if drive is not None:cue[:,5]=drive
        if card is not None:cue[:,6:6+card.shape[1]]=card
        h=self.loop(torch.cat((ctx,state,retinal,cue),1))
        return h,self.port(h)

    def propose(self,x):
        ctx=self.obs(x);h,out=self.step(ctx,0)
        return ctx,h,motor(out)
    def imagine(self,ctx,h,drive):
        _,out=self.step(ctx,1,prior=h,drive=drive);return image(out)
    def refine(self,ctx,h,proposal,visual,mode):
        # A direct route from full history and goal remains open even in the joint arm.
        _,out=self.step(ctx,2,prior=h,drive=proposal,eye=levels(visual).detach() if mode=="joint" else None)
        return motor(out)
    def operation(self,x,mode):
        ctx,h,rough=self.propose(x);proposal=electrical(rough)
        imagined=self.imagine(ctx,h,proposal)
        command=self.refine(ctx,h,proposal,imagined,mode)
        return ctx,h,rough,imagined,command
    def consequent(self,ctx,h,command):
        _,out=self.step(ctx,3,prior=h,drive=command);return image(out),felt(out)
    def language(self,initial,cards,query,mode,perturb="natural",donors=None):
        # One read-only, prepresented circuit; each tick addresses one card. No fresh world observations.
        n,depth,_=cards.shape; dev=initial.device
        z=self.port.weight.new_zeros(n,self.w); memory=z.new_zeros(n,3)
        allout=[]; payload=[]; mid=max(1,depth//2)
        for t in range(depth):
            s,d,p=cards[:,t].unbind(-1)
            card=torch.cat((F.one_hot(s,3),F.one_hot(d,3),p[:,None],F.one_hot(query,3),
                            initial if t==0 else torch.zeros_like(initial), memory,
                            z.new_full((n,1),int(t==0))),1).float()
            _,out=self.step(z,4,card=card);lex,whole,latent,_=lang_fields(out)
            memory=hard({"symbol":lex,"holistic":whole,"latent":latent}[mode])
            allout.append(out);payload.append(memory)
            if perturb=="clear":memory=torch.zeros_like(memory)
            elif t+1==mid:
                if perturb=="donor":memory=memory[donors]
                elif perturb=="shuffle":memory=torch.cat((1-memory[:,1:2],memory[:,2:3],memory[:,0:1]),1)
                elif perturb=="repeat":memory=memory.clone()
        return torch.stack(allout,1),torch.stack(payload,1)

def sight_loss(pred,target):
    return ((pred-target).square()*(1+14*target)).mean()*8

def speech_loss(net,seq,mode):
    init,cards,q,trace=seq
    out,_=net.language(init,cards,q,mode)
    flat=out.flatten(0,1);truth=trace.flatten(0,1);a,b,l,display=lang_fields(flat)
    ids=(truth*truth.new_tensor([1,2,4])).sum(-1).long()
    coded=ids.new_tensor(CODE)[ids]
    holistic=((coded[:,None]>>torch.arange(3,device=ids.device))&1).float()
    bitsloss=F.binary_cross_entropy_with_logits(a,truth.float()) if mode=="symbol" else F.binary_cross_entropy_with_logits(b,holistic)
    answer=trace.gather(2,q[:,None,None].expand(-1,trace.shape[1],1)).reshape(-1).float()*2-1
    return bitsloss+F.cross_entropy(display,ids)*.5+F.mse_loss(motor(flat),answer)
