"""只读已执行的考题。置信区间按布局重采样，三个训练初始化作为固定面板而非 3072 个独立试次。"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from .common import config,sha,new_json,read_json,read_npz,mname

def interval(values,indices):
    mean=np.mean(values); bs=values[indices].mean(1)
    return [float(x) for x in (mean,*np.quantile(bs,[.025,.975]))]

def analyze(release,out):
    cfg=config(); measured=read_json(release/"observed"/"after.json"); seal=release/"exam"/"seal-before-action.json"
    if measured["seal"]!=sha(seal):raise ValueError("实测与预测封存不对应")
    out.mkdir(parents=True,exist_ok=False); methods=cfg["learned"]+cfg["rules"]; rows=[]; seeds_rows=[]; effects=[]
    for split in ("test_id","test_link"):
        op=release/"observed"/split/"outcomes.npz"
        if sha(op)!=measured["results"][split]:raise ValueError("实测指纹不符")
        truth=read_npz(op); dy=truth["dy"]; n=len(dy)
        rng=np.random.default_rng(cfg["bootstrap_seed"]+(split=="test_link")); indices=rng.integers(n,size=(cfg["bootstrap_draws"],n))
        per={}
        for key in methods:
            keys=[mname(key,s) for s in (cfg["seeds"] if key in cfg["learned"] else [0])]
            wins=np.stack([truth[k] for k in keys]) # S,N,T,MODE,J,G
            per[key]=wins[:,:,:,0].mean((0,3,4)) # N,T
            phys=[]; mag=[]; misses=[]; goalmax=[]
            for k in keys:
                z=read_npz(release/"exam"/split/(k+".npz"))
                if "dy" in z:
                    v=z["dy"];t=dy>=cfg["forecast_threshold_pixels"]
                    valid=np.isfinite(v); match=((v>=cfg["forecast_threshold_pixels"])==t)&valid
                    phys.append(match.all(1).mean());mag.append(np.nanmean(np.abs(v-dy)));misses.append((~valid).sum())
                if "goal_delta" in z:goalmax.append(float(z["goal_delta"]))
            for t,name in enumerate(("single","compound")):
                w=wins[:,:,t]; est,lo,hi=interval(per[key][:,t],indices)
                acc=w[:,:,0].mean((1,2,3))
                cast=lambda a:None if not len(a) else float(np.mean(a))
                rows.append(dict(link=split,goals=name,method=key,task=est,ci_low=lo,ci_high=hi,
                   seed_min=float(acc.min()),seed_max=float(acc.max()),quartet=float(w[:,:,0].all((-1,-2)).mean()),
                   flat=float(w[:,:,1].mean()),exchange=float(w[:,:,2].mean()),physics_pair=cast(phys),
                   pixel_mae=cast(mag),invalid_reads=int(sum(misses)) if phys else None,
                   fixed_goal_delta=max(goalmax) if goalmax else None,n_layouts=n,n_seeds=len(keys)))
                for k,ss in zip(keys,w):
                    seeds_rows.append(dict(link=split,goals=name,run=k,total=int(ss[:,0].size),
                         completed=int(ss[:,0].sum()),quartet=int(ss[:,0].all((-1,-2)).sum()),
                         flat=int(ss[:,1].sum()),exchange=int(ss[:,2].sum())))
        for target in methods:
            if target=="jstp":continue
            for t,name in enumerate(("single","compound")):
                e,lo,hi=interval(per["jstp"][:,t]-per[target][:,t],indices)
                effects.append(dict(link=split,goals=name,contrast="jstp - "+target,delta=e,ci_low=lo,ci_high=hi))
    for name,data in (("metrics",rows),("per_seed",seeds_rows),("paired_differences",effects)):
        with (out/(name+".csv")).open("x",encoding="utf8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n");w.writeheader();w.writerows(data)
    new_json(out/"results.json",dict(estimand="layout-paired bootstrap, fixed trained seed panel; rates not percentages",metrics=rows,contrasts=effects))
    for r in rows:
        if r["goals"]=="compound":print(r["link"],r["method"],round(r["task"],3),r["physics_pair"])
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--release",type=Path,default=Path("evidence/paper"));p.add_argument("--out",type=Path)
    a=p.parse_args();analyze(a.release,a.out or a.release/"analysis")
