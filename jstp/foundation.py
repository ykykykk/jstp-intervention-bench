"""DINO-WM 的任务适配：官方 DINOv2 ViT-S/14 权重 + 原仓库未修改的 ViTPredictor。
保留 patch 预测与目标距离；时窗、宽度、动作接口和有限枚举规划按本任务适配。
"""
from __future__ import annotations
import importlib.util,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .common import config,sha,read_npz,save_npz,new_json
from .v7_world import TABLE,W,H,D

def repo(root,name):
    root=Path(root); spec=config()["dino"]
    p=root/name
    actual=subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip()
    want=spec["encoder_commit" if name=="dinov2" else "predictor_commit"]
    if actual!=want:raise ValueError(f"{name} 必须检出固定提交 {want}")
    return p

class Encoder:
    def __init__(self,root,device):
        self.device=device; root=Path(root); spec=config()["dino"]
        src=repo(root,"dinov2"); weights=root/spec["checkpoint_name"]
        if sha(weights)!=spec["encoder_sha256"]:raise ValueError("DINOv2 官方权重 SHA256 不匹配")
        sys.path.insert(0,str(src.resolve()))
        from dinov2.hub.backbones import dinov2_vits14
        self.net=dinov2_vits14(pretrained=False)
        self.net.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True),strict=True)
        self.net.to(device).requires_grad_(False).eval()
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    @torch.inference_mode()
    def __call__(self,images):
        prefix=images.shape[:-3]; images=images.reshape(-1,H,W,3)
        dest=np.empty((len(images),54,384),np.float16)
        for i in range(0,len(images),96):
            x=torch.as_tensor(images[i:i+96],device=self.device).permute(0,3,1,2).float()/255
            x=F.interpolate(x,size=(84,126),mode="bicubic",align_corners=False,antialias=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=str(self.device).startswith("cuda")):
                z=self.net.forward_features((x-self.mean)/self.std)["x_norm_patchtokens"]
            dest[i:i+len(x)]=z.half().cpu().numpy()
        return dest.reshape(*prefix,54,384)

def full_goals(goals,current):
    tail=np.broadcast_to(current[:,None,TABLE:],(*goals.shape[:2],H-TABLE,W,3))
    return np.concatenate([goals,tail],axis=-3)

def features(data,enc):
    out={}
    for k,v in data.items():
        if k in ("c","e","f"):out[k]=enc(v)
        elif k=="g":out[k]=enc(full_goals(v,data["c"]))
        else:out[k]=v
    if "g" in data and "f" in data:
        y=[]
        for a,g,m in zip(data["f"],data["g"],data["mask"]):
            err=(a[:,None,:,:TABLE]!=g[None,:,None]).any(-1)&(m[None,None,None]>0)
            y.append(err.sum((-1,-2)).argmin(-1))
        out["answer"]=np.stack(y)
    return out

def cache(work,device):
    data=work/"data"; dst=work/"features"; dst.mkdir(parents=True,exist_ok=True)
    enc=Encoder(work/"external",device); log={}
    for split in ("direct_train","train","val"):
        raw=read_npz(data/(split+".npz")); fs=features(raw,enc); path=dst/(split+".npz")
        # 短期离线缓存不压缩浮点 token，以免压缩遮盖训练时间。
        with path.open("xb") as f:np.savez(f,**fs)
        log[split]={"data":sha(data/(split+".npz")),"features":sha(path)}
        print("DINO FEATURES",split,flush=True)
    new_json(dst/"manifest.json",log)

class DinoDynamics(nn.Module):
    def __init__(self,external):
        super().__init__(); path=repo(external,"dino_wm")/"models"/"vit.py"
        spec=importlib.util.spec_from_file_location("jstp_upstream_dino_vit",path)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        cfg=config()["dino"]; d=cfg["width"]
        self.input=nn.Linear(384,d);self.action=nn.Linear(4,d)
        self.predictor=mod.ViTPredictor(num_patches=54,num_frames=3,dim=d,depth=cfg["depth"],
                         heads=cfg["heads"],mlp_dim=2*d,dim_head=cfg["dim_head"])
        self.out=nn.Linear(d,384)
        self.register_buffer("scale",torch.tensor([2/(W-1),2/(H-1),1/D,1/D]),persistent=False)
        self.register_buffer("off",torch.tensor([-1.,-1.,0.,0.]),persistent=False)
    def predict(self,b,e,c,p,a):
        rev=p.clone();rev[:,:2]+=p[:,2:];rev[:,2:]*=-1
        cmd=torch.stack([p,rev,a],1)*self.scale+self.off
        x=self.input(torch.stack([b,e,c],1))+self.action(cmd)[:,:,None]
        z=self.predictor(x.flatten(1,2))[:,-54:]
        return c+self.out(z)
    def branches(self,c,e,p,a):
        n=len(c)
        cc=c[:,None,None].expand(-1,2,2,-1,-1).flatten(0,2)
        ee=e[:,:,None].expand(-1,-1,2,-1,-1).flatten(0,2)
        pp=p[:,None,None].expand(-1,2,2,-1).flatten(0,2)
        aa=a[:,None].expand(-1,2,-1,-1).flatten(0,2)
        return self.predict(cc,ee,cc,pp,aa).reshape(n,2,2,54,384)
    @staticmethod
    def distances(f,g,mask):
        w=F.adaptive_avg_pool2d(F.pad(mask[:,None],(0,0,0,H-TABLE)),(6,9)).flatten(1)
        err=(f[:,:,None]-g[:,None,:,None]).square().mean(-1)
        return (err*w[:,None,None,None]).sum(-1)/w.sum(-1)[:,None,None,None].clamp_min(1)
