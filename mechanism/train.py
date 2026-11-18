import argparse,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import CFG,PROTO,prepare,read,sha,dump,context,training_programs
from .net import Core,electrical,sight_loss,speech_loss

def config(seed,device):
    torch.manual_seed(seed);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    if str(device).startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("要求的 CUDA 不存在")
def T(x,device):return torch.as_tensor(x,device=device)
class Bank:
    def __init__(self,data,dev):
        r=data["rows"];self.x=T(context(data,r),dev);self.r=T(r.astype('int64'),dev)
        self.f=T(data["futures"],dev);self.n=len(r)
    def eye(self,r,u):
        i,j,k=r[:,:3].T; a=u.long()+1
        return torch.cat((self.f[i,j,a],self.f[i,k,a]),1)

def motor_loss(net,b,ix,explore,mode):
    r=b.r[ix]; x=b.x[ix]; own=r[:,10].float();home=r[:,7].float();ext=r[:,6].float()
    ctx,h,rough,pic,cmd=net.operation(x,mode)
    loss=F.mse_loss(rough,home)*.4+F.mse_loss(cmd,own)
    if mode=="direct":return loss
    # In the rehearsal curriculum the home stroke and reset are actually executed.
    # Calibration also uses measured, goal-independent apparatus sweeps; never test-branch labels.
    loss=loss+sight_loss(net.imagine(ctx,h,home),b.eye(r,home))
    loss=loss+sight_loss(net.imagine(ctx,h,explore),b.eye(r,explore))
    f,body=net.consequent(ctx,h,own)
    loss=loss+sight_loss(f,b.eye(r,own+ext))+F.mse_loss(body,own+ext)*.35
    active=r[:,5]==0
    if active.any():
        f,body=net.consequent(ctx[active],h[active],explore[active])
        loss=loss+.65*sight_loss(f,b.eye(r[active],explore[active]))+.15*F.mse_loss(body,explore[active])
    return loss

def fita(data,release,arm,seed,device,banks):
    mode,lang=CFG["arms"][arm];root=release/"models"/f"{arm}-{seed}"
    root.mkdir(parents=True,exist_ok=False);config(seed,device)
    net=Core().to(device); opt=torch.optim.AdamW(net.parameters(),lr=CFG["lr"],weight_decay=1e-5)
    b=banks[0]; v=banks[1]; n=CFG["batch"]
    rng=np.random.default_rng(seed+923); lrng=np.random.default_rng(CFG["lang_train_seed"]+seed)
    tic=time.monotonic(); pending=[]
    with (root/"curve.jsonl").open("x",encoding="utf8",newline="\n",buffering=1) as log:
        for step in range(1,CFG["steps"]+1):
            ix=T(rng.integers(b.n,size=n),device); exp=T(rng.integers(-1,2,size=n).astype('float32'),device)
            depth=1+(step-1)%CFG["train_max_depth"]
            seq=tuple(T(z,device) for z in training_programs(lrng,n,depth))
            loss=motor_loss(net,b,ix,exp,mode)+speech_loss(net,seq,lang)
            if not torch.isfinite(loss):raise FloatingPointError(f"{arm}/{step}")
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
            pending.append(float(loss.detach()))
            if step%CFG["evaluate_every"]==0:
                with torch.no_grad():
                    ii=torch.arange(0,min(v.n,1200),2,device=device); cmd=net.operation(v.x[ii],mode)[-1]
                    acc=(electrical(cmd)==v.r[ii,10]).float().mean().item()
                row=dict(step=step,loss=float(np.mean(pending)),validation_recorded_drive=acc,seconds=round(time.monotonic()-tic,3))
                log.write(json.dumps(row)+"\n");pending=[]
    wt=root/"last.pt"
    with wt.open("xb") as f:torch.save(net.cpu().state_dict(),f)
    dump(root/"freeze.json",dict(arm=arm,seed=seed,steps=step,trainable=sum(p.numel() for p in net.parameters()),
         weights=sha(wt),protocol=sha(PROTO),data=sha(data/"manifest.json"),python=platform.python_version(),
         torch=str(torch.__version__),numpy=np.__version__,device=(torch.cuda.get_device_name(device) if device.startswith("cuda") else device),
         seconds=round(time.monotonic()-tic,2)))
    print("FROZEN",arm,seed,round(time.monotonic()-tic,1),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["data","fit"]);p.add_argument("--data",type=Path,default=Path("work/m2-data"))
    p.add_argument("--release",type=Path,default=Path("evidence/mechanism"));p.add_argument("--device",default="cuda")
    p.add_argument("--arms",nargs="+",choices=list(CFG["arms"]),default=list(CFG["arms"]));p.add_argument("--seeds",nargs="+",type=int,default=CFG["seeds"])
    a=p.parse_args()
    if a.action=="data":return prepare(a.data)
    banks=tuple(Bank(read(a.data/(s+".npz")),a.device) for s in ("train","validation"))
    for arm in a.arms:
        for seed in a.seeds:fita(a.data,a.release,arm,seed,a.device,banks)
if __name__=="__main__":main()
