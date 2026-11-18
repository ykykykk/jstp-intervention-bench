"""两阶段考试。seal 只看试探和目标，所有方法的承诺落盘后 score 才执行正式竖拖。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
import torch
from .common import config,configure,PROTOCOL,sha,arrsha,new_json,read_json,save_npz,read_npz,mname
from .arena import scene,sense,execute
from .v7_world import H,W,TABLE,D
from .v7_model import picture
from .learn import Bank,model_for,pair_outputs,costs
from .rules import rule

def fields(split):
    start,count=config()["splits"][split]
    return range(start,start+count)
def observed(seed,split):
    s=scene(seed,split=="test_link"); c,e=sense(s); gg=[]; mm=[]
    for kind in (False,True):
        goals,mask=s.goals(kind);gg.append(goals);mm.append(mask)
    return s,dict(c=c,e=e,p=s.probe.astype(np.float32),a=s.options.astype(np.float32),gs=np.stack(gg),ms=np.stack(mm))
def observed_batch(split):
    rows=[observed(i,split)[1] for i in fields(split)]
    return {k:np.stack([x[k] for x in rows]) for k in rows[0]}
def evidence_ids(b):
    return dict(current=[arrsha(x) for x in b["c"]],peak=[[arrsha(i) for i in row] for row in b["e"]],
                goals=[[arrsha(gg) for gg in row] for row in b["gs"]],mask=[[arrsha(mm) for mm in row] for row in b["ms"]])
def cy(a):
    a=np.asarray(a)[...,:TABLE,:,:].astype(np.float32)
    lum=np.maximum(a.mean(-1)-46,0); cells=[]
    for k in range(3):
        b=lum[...,32*k:32*k+32]; mass=b.sum((-1,-2))
        moment=(b*np.arange(TABLE,dtype=np.float32)[:,None]).sum((-1,-2))
        cells.append(np.divide(moment,mass,out=np.full_like(mass,np.nan),where=mass>1.))
    return np.stack(cells,-1)
def dy_effect(img,c,cmd):
    v=cy(img)-cy(c)[:,None,None,:]
    return v*np.sign(cmd[:,0,3])[:,None,None,None]
def ids(cfg):
    for k in cfg["learned"]:
        for seed in cfg["seeds"]:yield k,seed
    for k in cfg["rules"]:yield k,0

@torch.inference_mode()
def forecasts(m,key,b,raw,device):
    n=len(b["c"]); physical=key not in ("history_policy","dino_wm")
    # 顺序 [布局,目标类型,经历干预,实际挂接,目标,候选]
    values=np.empty((n,2,3,2,2,2),np.float32)
    motion=np.empty((n,2,2,3),np.float32) if physical else None
    fingerprints=[]; goal_diff=0.; example=None
    for i in range(0,n,8):
        rr={k:v[i:i+8] for k,v in raw.items()}; bi={k:v[i:i+8] for k,v in b.items()}
        for mode in range(3):
            e=bi["e"] if mode==0 else (np.repeat(bi["c"][:,None],2,1) if mode==1 else bi["e"][:,::-1].copy())
            first=None
            for t in range(2):
                for g in range(2):
                    goals=bi["gs"][:,t,g:g+1]; mask=bi["ms"][:,t]
                    dd=dict(c=bi["c"],e=e,p=bi["p"],a=bi["a"],g=goals,mask=mask)
                    batch=Bank(dd,device).batch(slice(None))
                    if m is None:
                        f=torch.as_tensor(rule(dd["c"],dd["e"],dd["p"],dd["a"],key),device=device).movedim(-1,-3)
                        cost=costs(f,batch["g"],batch["mask"])
                    else:f,cost=pair_outputs(m,key,batch)
                    values[i:i+len(e),t,mode,:,g,:]=cost[:,:,0].cpu().numpy()
                    if f is not None:
                        if first is None:first=f
                        else:goal_diff=max(goal_diff,float((first-f).abs().max()))
                    if mode==0 and t==0 and g==0 and f is not None:
                        pics=picture(f) if physical else f.float().cpu().numpy()
                        fingerprints.extend([arrsha(a) for a in pics])
                        if physical:
                            motion[i:i+len(e)]=dy_effect(pics,rr["c"],rr["a"])
                            if i==0:example=pics[0].copy()
    if not np.isfinite(values).all():raise ValueError("候选代价非有限")
    out=dict(cost=values,choice=values.argmin(-1).astype(np.uint8),forecast_sha=np.array(fingerprints,dtype="S64"))
    if motion is not None:out.update(dy=motion,example=example)
    if m is None or key!="history_policy":out["goal_delta"]=np.array(goal_diff)
    return out

def dino_inputs(raw,encoder):
    from .foundation import full_goals
    out=raw.copy(); out["c"]=encoder(raw["c"]); out["e"]=encoder(raw["e"])
    out["gs"]=np.stack([encoder(full_goals(raw["gs"][:,i],raw["c"])) for i in (0,1)],1)
    return out

def seal(work,release,device):
    cfg=config(); configure(1729,device)
    root=release/"exam"; root.mkdir(exist_ok=False)
    weights={}; files={}; pid=sha(PROTOCOL)
    for k,s in ids(cfg):
        if not s:continue
        name=mname(k,s); base=release/"models"/name; meta=read_json(base/"freeze.json")
        if meta["protocol"]!=pid or sha(base/"frozen.pt")!=meta["sha256"]:raise ValueError(f"未冻结：{name}")
        weights[name]=meta["sha256"]
    encoder=None
    for split in ("test_id","test_link"):
        part=root/split; part.mkdir(); raw=observed_batch(split)
        new_json(part/"observations.json",evidence_ids(raw));files[split+"/observations.json"]=sha(part/"observations.json")
        din=None
        for key,s in ids(cfg):
            name=mname(key,s); mod=None; data=raw
            if s:
                mod=model_for(key,work/"external").to(device)
                mod.load_state_dict(torch.load(release/"models"/name/"frozen.pt",map_location=device,weights_only=True))
                mod.eval().requires_grad_(False)
            if key=="dino_wm":
                if encoder is None:
                    from .foundation import Encoder
                    encoder=Encoder(work/"external",device)
                if din is None:din=dino_inputs(raw,encoder)
                data=din
            commit=forecasts(mod,key,data,raw,device); f=part/(name+".npz");save_npz(f,**commit);files[split+"/"+f.name]=sha(f)
            print("COMMITTED",split,name,flush=True)
    new_json(root/"seal-before-action.json",dict(protocol=pid,models=weights,commits=files,utc_ns=time.time_ns()))

def score(release,out=None):
    cfg=config(); root=release/"exam";sealfile=root/"seal-before-action.json"; lock=read_json(sealfile)
    if lock["protocol"]!=sha(PROTOCOL):raise ValueError("协议不匹配")
    for f,h in lock["commits"].items():
        if sha(root/f)!=h:raise ValueError(f"已承诺的数据改变：{f}")
    for name,h in lock["models"].items():
        if sha(release/"models"/name/"frozen.pt")!=h:raise ValueError(f"冻结模型改变：{name}")
    out=out if out is not None else release/"observed";out.mkdir(parents=True,exist_ok=False)
    output={}
    for split in ("test_id","test_link"):
        part=out/split;part.mkdir(); obs=read_json(root/split/"observations.json")
        rs={mname(k,s):read_npz(root/split/(mname(k,s)+".npz"))["choice"] for k,s in ids(cfg)}
        wins={k:np.zeros_like(v,dtype=bool) for k,v in rs.items()}; tru=[];checks=[]
        for i,seed in enumerate(fields(split)):
            s,raw=observed(seed,split)
            if any(obs[k][i]!=v[0] for k,v in evidence_ids({k:v[None] for k,v in raw.items()}).items()):
                raise ValueError("当前观察不等于封存时的历史/目标")
            actual=np.asarray([[execute(s,j,a) for a in s.options] for j in (0,1)])
            tru.append(dy_effect(actual[None],raw["c"][None],raw["a"][None])[0])
            checks.append([[arrsha(x) for x in row] for row in actual])
            if i==0:save_npz(part/"example.npz",current=raw["c"],peaks=raw["e"],goals=raw["gs"],masks=raw["ms"],actual=actual)
            for k,v in rs.items():
                for t in (0,1):
                    m=raw["ms"][t].astype(bool)
                    for mode in (0,1,2):
                        for j in (0,1):
                            for g in (0,1):
                                a=int(v[i,t,mode,j,g]); image=execute(s,j,s.options[a])
                                if not np.array_equal(actual[j,a],image):raise ValueError("执行器分支不一致")
                                wins[k][i,t,mode,j,g]=np.array_equal(image[:TABLE][m],raw["gs"][t,g][m])
        path=part/"outcomes.npz";save_npz(path,dy=np.array(tru),actual_sha=np.array(checks,dtype="S64"),**wins)
        output[split]=sha(path);print("ACTUATED",split,len(tru),flush=True)
    new_json(out/"after.json",dict(seal=sha(sealfile),results=output,utc_ns=time.time_ns()))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["seal","score"]);p.add_argument("--work",type=Path,default=Path("work"))
    p.add_argument("--release",type=Path,default=Path("evidence/paper"));p.add_argument("--out",type=Path);p.add_argument("--device",default="cuda")
    a=p.parse_args();seal(a.work,a.release,a.device) if a.action=="seal" else score(a.release,a.out)
