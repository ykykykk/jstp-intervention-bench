from __future__ import annotations
import argparse,json,platform,time,sys
from pathlib import Path
import numpy as np
import torch
from .common import config,configure,sha,PROTOCOL,new_json,read_json,read_npz
from .learn import Bank,model_for,direct_loss,pair_loss,pair_outputs,costs

@torch.inference_mode()
def valid(m,key,bank):
    m.eval(); hits=[]
    for i in range(0,bank.n,8):
        b=bank.batch(slice(i,min(i+8,bank.n)));_,v=pair_outputs(m,key,b)
        y=b["answer"] if key=="dino_wm" else costs(b["f"],b["g"],b["mask"]).argmin(-1)
        hits.extend((v.argmin(-1)==y).flatten(1).cpu().tolist())
    return float(np.mean(hits))

def fit(work,release,key,seed,device,banks):
    dest=release/"models"/f"{key}-{seed}"; dest.mkdir(parents=True,exist_ok=True)
    meta=dest/"freeze.json"
    if meta.exists():
        r=read_json(meta)
        if r["protocol"]!=sha(PROTOCOL) or r["sha256"]!=sha(dest/"frozen.pt"):raise ValueError("冻结文件与协议不符")
        print("EXISTING",key,seed,flush=True);return
    cfg=config(); configure(seed,device); m=model_for(key,work/"external").to(device); best=None; best_score=-1
    d,train,v=banks; tic=time.monotonic()
    # 使用独立取样 RNG，使不同网络初始化不改变所见训练记录清单。
    rng=torch.Generator(device="cpu").manual_seed(seed+3107)
    row=[];log=dest/"curve.jsonl"
    with log.open("x",encoding="utf8",buffering=1) as f:
        for ph,nsteps in ((1,cfg["stage1_steps"]),(2,cfg["stage2_steps"])):
            lr=cfg["dino_lr"] if key=="dino_wm" else cfg[f"lr_stage{ph}"]
            opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1.e-5)
            for step in range(1,nsteps+1):
                m.train(); isdirect=ph==1 or step%cfg["direct_rehearsal_every"]==0
                bank=d if isdirect else train
                size=cfg["direct_batch" if isdirect else "groups_per_batch"]
                ix=torch.randint(bank.n,(size,),generator=rng).to(device); b=bank.batch(ix)
                loss=(direct_loss if isdirect else pair_loss)(m,key,b)
                if not torch.isfinite(loss):raise FloatingPointError(f"{key}/{seed}/{ph}/{step}: loss={loss.item()}")
                opt.zero_grad(set_to_none=True); loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
                row.append(float(loss.detach()))
                if step%cfg["validate_every"]==0:
                    acc=valid(m,key,v) if ph==2 else None
                    if ph==2 and acc>=best_score:
                        best_score,best_step=acc,step
                        best={k:t.detach().cpu().clone() for k,t in m.state_dict().items()}
                    msg=dict(phase=ph,step=step,loss=float(np.mean(row)),val_task=acc,seconds=round(time.monotonic()-tic,2))
                    row=[]; f.write(json.dumps(msg,allow_nan=False)+"\n")
                    if ph==2 and step%450==0:print(key,seed,step,acc,flush=True)
    path=dest/"frozen.pt"
    with path.open("xb") as f:torch.save(best,f)
    record=dict(method=key,seed=seed,step=best_step,val=best_score,trainable=sum(p.numel() for p in m.parameters()),
        protocol=sha(PROTOCOL),sha256=sha(path),seconds=round(time.monotonic()-tic,2),
        torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),
        device=(torch.cuda.get_device_name(device) if str(device).startswith("cuda") else str(device)),
        train_manifest=sha(work/"data"/"manifest.json"))
    if key=="dino_wm":record["external"]=cfg["dino"];record["features_manifest"]=sha(work/"features"/"manifest.json")
    new_json(meta,record);print("FROZEN",key,seed,best_step,best_score,flush=True)

def train(work,release,keys,seeds,device):
    cfg=config(); cache={}
    for key in keys:
        if key not in cfg["learned"]:raise KeyError(key)
        src="features" if key=="dino_wm" else "data"
        if src not in cache:
            cache={src:tuple(Bank(read_npz(work/src/(s+".npz")),device) for s in ("direct_train","train","val"))}
        for s in seeds: fit(work,release,key,s,device,cache[src])

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--work",type=Path,default=Path("work"));p.add_argument("--release",type=Path,default=Path("evidence/paper"))
    p.add_argument("--methods",nargs="+",default=config()["learned"]);p.add_argument("--seeds",nargs="+",type=int,default=config()["seeds"]);p.add_argument("--device",default="cuda")
    a=p.parse_args();train(a.work,a.release,a.methods,a.seeds,a.device)
