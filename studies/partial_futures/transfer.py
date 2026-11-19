"""Two separately budgeted, official-weight adaptations. The scratch comparison does not import these dependencies."""
import argparse,json,time
from dataclasses import fields
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mechanism.data import read,save,sha,dump
from lategoal.rig import TRAIN,EXAM,Rig,goal_wins
from lategoal.exam import actual
from lategoal.control import fingerprint
from .train import HERE,C,DATA,DEFAULT
from .exam import prehistory
from .analyze import count,distribution,summary

PLAN=HERE/"transfer-plan.json"; P=json.loads(PLAN.read_text(encoding="utf8"))
WORK=HERE.parents[1]/"work"
def cu(x):return torch.as_tensor(x,device="cuda")
def picture(x,size):
    base=x[:,:72].reshape(-1,6,12);dx=x[:,72:144].reshape_as(base)
    im=torch.stack((5*base,5*(base+dx),.5+3*dx),1).clamp(0,1)
    return F.interpolate(im,size=(size,size),mode="bilinear",align_corners=False)
def goal_words(g):
    return "Move "+", ".join(f"track {i+1} {'up' if v<0 else 'down'}" for i,v in enumerate(g) if v)+"."
class TaskModel(nn.Module):
    def __init__(self):
        super().__init__();self.start=nn.Linear(73,1)
    def probe(self,b):return self.start(b).squeeze(-1)
class Dino(TaskModel):
    def __init__(self,folder):
        super().__init__()
        from transformers import AutoModel
        from .upstream.dino_wm.vit import ViTPredictor
        self.backbone=AutoModel.from_pretrained(str(folder),local_files_only=True).requires_grad_(False).eval()
        self.img=nn.Linear(384,64);self.body=nn.Linear(19,64)
        self.transition=ViTPredictor(num_patches=16,num_frames=1,dim=64,depth=2,heads=4,dim_head=16,mlp_dim=128)
        self.eye=nn.Linear(16*64,12); self.utility=nn.Sequential(nn.Linear(12,64),nn.SiLU(),nn.Linear(64,4))
    def context(self,x):
        im=picture(x,56);mu=im.new_tensor([.485,.456,.406])[None,:,None,None];sd=im.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.no_grad():patch=self.backbone(pixel_values=(im-mu)/sd).last_hidden_state[:,1:]
        return self.img(patch),x[:,144:]
    def candidates(self,x):
        pat,body=self.context(x);w=[]
        for k in (-1,1):
            pp=self.body(torch.cat((body,torch.full_like(body[:,:1],k)),1))
            w.append(self.eye(self.transition(pat+pp[:,None]).flatten(1)))
        return torch.stack(w,1)
    def loss(self,x,card,y,eyes,card_ix):
        v=self.candidates(x).float()
        issue=((self.utility(v[:,1])-self.utility(v[:,0]))*card).sum(-1)
        return F.softplus(-issue*y).mean()+3*F.mse_loss(v,eyes)
    def choices(self,x,goals,first_goal):
        v=self.candidates(x);score=self.utility(v[:,1])-self.utility(v[:,0])
        return score@goals.T
class Smol(TaskModel):
    def __init__(self,folder,processor):
        super().__init__()
        from safetensors.torch import load_file
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
        doc=json.loads((folder/"config.json").read_text(encoding="utf8"))
        keys={"attention_mode","num_vlm_layers","self_attn_every_n_layers","expert_width_multiplier","add_image_special_tokens","prefix_length"}
        cfg=SmolVLAConfig(**{k:doc[k] for k in keys},vlm_model_name=str(processor.resolve()),load_vlm_weights=False,
            device="cuda",chunk_size=1,n_action_steps=1,num_steps=4,use_cache=True)
        self.native=VLAFlowMatching(cfg)
        raw=load_file(str(folder/"model.safetensors"),device="cpu")
        if not all(k.startswith("model.") for k in raw):raise ValueError("未知官方权重命名")
        self.native.load_state_dict({k[6:]:v for k,v in raw.items()},strict=True)
        self.native.requires_grad_(False);self.native.state_proj.requires_grad_(True)
        self.native.action_out_proj.requires_grad_(True)
        self.native.vlm_with_expert.lm_expert.layers[-1].requires_grad_(True)
        tok=self.native.vlm_with_expert.processor.tokenizer
        texts=[goal_words(c) for c in np.concatenate((TRAIN,EXAM))]
        tt=tok(texts,padding=True,return_tensors="pt")
        self.register_buffer("words",tt.input_ids,persistent=False);self.register_buffer("mask",tt.attention_mask.bool(),persistent=False)
    def inputs(self,x,g,idx):
        im=2*picture(x,128)-1;state=F.pad(torch.cat((x[:,144:],g),1),(0,10))
        return [im],[torch.ones(len(x),dtype=torch.bool,device=x.device)],self.words[idx],self.mask[idx],state
    def loss(self,x,card,y,eyes,card_ix):
        act=x.new_zeros(len(x),1,32);act[:,0,0]=y
        noise=torch.zeros_like(act);noise[:,0,0]=torch.randn_like(y)
        return self.native(*self.inputs(x,card,card_ix),actions=act,noise=noise)[:,:,0].mean()
    def choices(self,x,goals,first_goal):
        n,k=len(x),len(goals);idx=torch.arange(first_goal,first_goal+k,device=x.device).repeat(n)
        xx=x.repeat_interleave(k,dim=0);gg=goals.repeat(n,1);out=[]
        for xb,gb,ib in zip(xx.split(32),gg.split(32),idx.split(32)):
            noise=xb.new_zeros(len(xb),1,32)
            out.append(self.native.sample_actions(*self.inputs(xb,gb,ib),noise=noise)[:,0,0])
        return torch.cat(out).reshape(n,k)
def make(name,folder,processor):
    if sha(folder/"model.safetensors")!=P[name]["sha256"]:raise ValueError("预训练权重不完整："+str(folder))
    return (Dino(folder) if name=="dino" else Smol(folder,processor)).to("cuda")
def params(m):return {n:p for n,p in m.named_parameters() if p.requires_grad}
def fit(name,m,rel):
    if sha(DATA/"train.npz")!=C["train_sha256"]:raise ValueError("训练源不是封存的 v3 数据")
    from importlib.metadata import version
    d=rel/name;d.mkdir(parents=True,exist_ok=False)
    b=read(DATA/"train.npz");N=len(b["start"])*8; rng=np.random.default_rng(P["seed"])
    idx=rng.permutation(N);cards=rng.integers(len(TRAIN),size=N)
    opt=torch.optim.AdamW(params(m).values(),lr=P[name]["lr"],weight_decay=.0001);tic=time.monotonic()
    m.eval()  # Frozen feature extractor, no dropout; gradients remain on for selected layers.
    with (d/"curve.jsonl").open("x",encoding="utf8",buffering=1) as log:
        for k,(rr,gg) in enumerate(zip(idx.reshape(-1,P["batch"]),cards.reshape(-1,P["batch"])),1):
            i,s,g,pi=np.unravel_index(rr,(len(b["start"]),2,2,2));goals=TRAIN[gg]
            good=np.where((b["motion"][i,s,g,0]*goals).sum(-1)>0,-1.,1.).astype("float32")
            xs=cu(b["hist"][i,s,g,pi]);start=cu(b["start"][i])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=m.loss(xs,cu(goals).float(),cu(good),cu(b["eye"][i,s,g]),cu(gg).long())+.18*F.softplus(-m.probe(start)*start[:,-1]).mean()
            if not torch.isfinite(loss):raise FloatingPointError((name,k))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params(m).values(),3);opt.step()
            if k%32==0:log.write(json.dumps(dict(step=k,loss=float(loss.detach()),seconds=round(time.monotonic()-tic,1)))+"\n")
    ck=d/"adapter.pt"
    with ck.open("xb") as f:torch.save({n:p.detach().cpu() for n,p in params(m).items()},f)
    dump(d/"freeze.json",dict(adapter=sha(ck),code=sha(Path(__file__)),external=P[name]["sha256"],data=C["train_sha256"],
        plan=sha(PLAN),updates=k,examples=N,seed=P["seed"],parameters=sum(p.numel() for p in m.parameters()),
        adapted=sum(p.numel() for p in params(m).values()),seconds=round(time.monotonic()-tic,2),device=torch.cuda.get_device_name(0),
        versions={n:version(n) for n in ("torch","transformers","lerobot","numpy","safetensors","einops")}))
@torch.inference_mode()
def seal(name,m,d,panel):
    st,n=C["panels"][panel];ids=np.arange(st,st+n);rigs=[Rig(int(i)) for i in ids]
    s,x,pk=prehistory(rigs);p=np.where(m.probe(cu(s)).cpu().numpy()>=0,1,-1).astype("int8");ii=np.arange(n);k=(p>0).astype(int)
    a=cu(x[ii,k].reshape(n*4,-1));v=[];gg=cu(EXAM).float()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        for b in a.split(16 if name=="smol" else 64):
            v.append((2*(m.choices(b,gg,len(TRAIN))>=0).to(torch.int8)-1).cpu().numpy())
    dst=d/(panel+"-sealed.npz");save(dst,ids=ids,goals=EXAM,probe=p,peak=pk[ii,k],u=np.concatenate(v).reshape(n,4,len(EXAM)))
    dump(d/(panel+"-before.json"),dict(weights=sha(d/"adapter.pt"),intents=sha(dst)))
def execute(d,panel):
    dst=d/(panel+"-sealed.npz");before=json.loads((d/(panel+"-before.json")).read_text(encoding="utf8"))
    if sha(dst)!=before["intents"] or sha(d/"adapter.pt")!=before["weights"]:raise ValueError("封卷不符")
    a=read(dst);wins=[];hashes=[]
    for i,seed in enumerate(a["ids"]):
        r=Rig(int(seed));yy=[];hs=[]
        for state in range(4):
            w,g=divmod(state,2);p=a["probe"][i]
            v,left,eye,h=actual(r,w,g,p,a["u"][i,state],a["peak"][i,w]);yy.append(goal_wins(v,EXAM));hs.append(h)
        wins.append(yy);hashes.append(hs)
    w=np.asarray(wins);n=len(w);dst=d/(panel+"-observed.npz");save(dst,wins=w,frame_sha=np.asarray(hashes))
    draw=np.random.default_rng(44178).multinomial(n,np.full(n,1/n),3000)
    data=dict(panel=panel,layouts=n,observed=sha(dst),before=sha(d/(panel+"-before.json")))
    for k,t in (("task",w),("pair",w[...,::2]&w[...,1::2])):data[k]=summary(*distribution(count(t),draw))
    dump(d/(panel+"-result.json"),data);print("OFFICIAL",d.name,panel,data["pair"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name",choices=("smol","dino"));p.add_argument("--out",type=Path,default=DEFAULT/"transfer")
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--processor",type=Path,default=WORK/"upstream-smol-processor")
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(P["seed"])
    m=make(a.name,a.weights,a.processor);d=a.out/a.name
    if not (d/"freeze.json").exists():fit(a.name,m,a.out)
    meta=json.loads((d/"freeze.json").read_text(encoding="utf8"));ck=d/"adapter.pt"
    if meta["adapter"]!=sha(ck) or meta["code"]!=sha(Path(__file__)):raise ValueError("冻结权重或实现指纹不符")
    adapted=torch.load(ck,map_location="cuda",weights_only=True)
    if set(adapted)!=set(params(m)):raise ValueError("适配参数集合不符")
    with torch.no_grad():
        for key,val in params(m).items():val.copy_(adapted[key])
    m.eval()
    for panel in C["panels"]:
        if not (d/(panel+"-before.json")).exists():seal(a.name,m,d,panel)
    for panel in C["panels"]:
        if not (d/(panel+"-result.json")).exists():execute(d,panel)

