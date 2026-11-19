"""Small, from-scratch policies for the same signed actuator. No simulator or judge imports."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from lategoal.rig import HIST, F as RETINA

def enc(inp,w):
    return nn.Sequential(nn.Linear(inp,w),nn.LayerNorm(w),nn.SiLU(),nn.Linear(w,w),nn.LayerNorm(w))

class Body(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.start=nn.Sequential(nn.Linear(RETINA+1,w),nn.SiLU(),nn.Linear(w,1))
        self.observe=enc(HIST,w)
    def probe(self,initial):
        return self.start(initial).squeeze(-1)
    def representation(self,h,p):
        return h

class Partial(Body):
    """A tied residual transition. A goal may act directly on history even if the prospective state is absent."""
    def __init__(self,arm,w):
        super().__init__(w); self.arm=arm
        self.cell=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w))
        self.drive=nn.Linear(1,w); self.commit=nn.Parameter(torch.zeros(w))
        self.return_path=nn.Linear(w,w,bias=False); self.eye=nn.Linear(w,12)
        self.address=nn.Linear(w,4); self.bias=nn.Linear(w,1)
    def step(self,state,event):
        return state+self.cell(state+event)
    def future(self,h,p):
        return self.step(h,self.drive(p[:,None]))
    def representation(self,h,p):
        return h if self.arm=="direct" else self.future(h,p)
    def issue(self,h,z,g,arm=None):
        s=h+self.return_path(z) if self.arm=="partial" else h
        s=self.step(s,self.commit)
        return (self.address(s)*g).sum(-1)+self.bias(s).squeeze(-1)
    def policy(self,h,p,g):
        return self.issue(h,self.future(h,p) if self.arm=="partial" else None,g)

class CandidateWorld(Body):
    """Train a conditional feature-world model. At inference, explicitly forecast and compare A and B."""
    def __init__(self,w):
        super().__init__(w); self.transition=enc(w+1,w); self.eye=nn.Linear(w,12)
        self.value=nn.Sequential(nn.Linear(12,w),nn.SiLU(),nn.Linear(w,4))
    def future(self,h,p):
        return self.transition(torch.cat([h,p[:,None]],1))
    representation=future
    def policy(self,h,p,g):
        one=torch.ones_like(p)
        pred=self.eye(self.future(torch.cat((h,h)),torch.cat((-one,one))))
        left,right=self.value(pred).chunk(2)
        return ((right-left)*g).sum(-1)

class TokenPolicy(Body):
    """Sensor prefix plus ordered address/sign instruction tokens; emit one discrete action token."""
    def __init__(self,w):
        super().__init__(w)
        self.lex=nn.Embedding(3,w); self.position=nn.Parameter(torch.randn(6,w)*.04)
        self.ask=nn.Parameter(torch.zeros(w))
        self.block=nn.TransformerEncoderLayer(w,4,2*w,dropout=0,batch_first=True,
                                              activation="gelu",norm_first=True)
        self.out=nn.Linear(w,2)
    def policy(self,h,p,g):
        n=len(g); seq=torch.cat((h[:,None],self.lex((g.long()+1)),self.ask.expand(n,1,-1)),1)
        logits=self.out(self.block(seq+self.position)[:,-1])
        return logits[:,1]-logits[:,0]

class FlowPolicy(Body):
    """Conditional flow matching in one motor dimension; common additive goal coordinates plus continuous drive."""
    def __init__(self,w):
        super().__init__(w); self.card=nn.Linear(4,w,bias=False)
        self.field=nn.Sequential(nn.Linear(w+3,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,1))
    def velocity(self,c,x,t):
        return self.field(torch.cat((c,x[:,None],torch.sin(math.pi*t)[:,None],
                                   torch.cos(math.pi*t)[:,None]),-1)).squeeze(-1)
    def loss(self,h,g,y):
        t=torch.rand_like(y); noise=torch.randn_like(y)
        return F.mse_loss(self.velocity(h*self.card(g),(1-t)*noise+t*y,t),y-noise)
    def policy(self,h,p,g):
        c=h*self.card(g); x=torch.zeros_like(p)
        for k in range(8):x=x+self.velocity(c,x,torch.full_like(x,k/8))/8
        return x

def make(arm,w=64):
    if arm in ("partial","detached","direct"):return Partial(arm,w)
    return {"world":CandidateWorld,"token":TokenPolicy,"flow":FlowPolicy}[arm](w)

