"""仅需 NumPy；把每个布局作为一个统计单元。"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .rig import C,REL,PROTO,EXAM,TRAIN,read,dump,sha
from mechanism.analyze import interval,csvout
def checked(rel):
    for name in ("frozen","calibrated","before","after"):
        data=json.loads((rel/(name+".json")).read_text(encoding="utf8"))
        if data["protocol"]!=sha(PROTO):raise ValueError("协议已改变")
        for p,h in data['files'].items():
            if sha(rel/p)!=h:raise ValueError("归档已改变："+p)
def vectors(z,o,arm,goals):
    u=z['u'];w=o['wins'];n=len(u);native=w[...,0]
    opt=np.where((o['left'][:,:,None]*goals).sum(-1)>0,-1,1)
    donor=opt[:,[1,0,3,2]];needs=opt!=donor
    if not np.array_equal(w,u==opt[...,None]):raise ValueError("提交与相机裁判不符")
    def mean(v,mask=None):
        v=np.asarray(v,float).reshape(n,-1)
        if mask is None:return v.mean(1)
        m=mask.reshape(n,-1)
        if not m.any(1).all():raise ValueError("条件无有效题")
        return (v*m).sum(1)/m.sum(1)
    signed=np.abs((o['left'][:,:,None]*goals).sum(-1)); conflict=signed==1
    d=mean(native&(u[...,2]==donor),needs);dn=mean(native&(u[...,3]==donor),needs)
    fp=mean(native&~w[...,2],~needs)
    out=dict(task=mean(native),pair=mean(native[...,::2]&native[...,1::2]),
        target_flip=mean(u[:,:,::2,0]!=u[:,:,1::2,0]),contested=mean(native,conflict),
        unanimous=mean(native,~conflict),directed=d,null_directed=dn,false_flip=fp,vs_null=d-dn,vs_stay=d-fp,
        stable_stays=mean(w[...,2],~needs),patched_observed_task=mean(w[...,2]),
        full_directed=mean(native&(u[...,5]==donor),needs),full_false=mean(native&~w[...,5],~needs),
        equal_future_changed=mean(u[...,4]!=u[...,0]),repeat_changed=mean(u[...,1]!=u[...,0]),fraction_eligible=mean(needs))
    if arm!="direct":
        t=o["reference_eye"];dt=t[:,[1,0,3,2]];eye=z["eye"]
        dist=lambda a,b:np.square(a-b).mean(-1)
        good=dist(eye[:,:,0],t)<dist(eye[:,:,0],dt)
        out.update(optical_correct=mean(good),physical_turn=mean(good&(dist(eye[:,:,2],dt)<dist(eye[:,:,2],t))),
                   full_turn=mean(good&(dist(eye[:,:,5],dt)<dist(eye[:,:,5],t))),
                   optical_mae=mean(abs(eye[:,:,0]-t)),patch_norm=mean(z["norm"][:,:,0]),
                   null_eye_max=abs(eye[:,:,0]-eye[:,:,3]).reshape(n,-1).max(1),
                   norm_gap_max=abs(z['norm'][:,:,0]-z['norm'][:,:,1]).max(1))
    if np.any(out["repeat_changed"]):raise ValueError("原样重算改变了动作")
    if arm!="joint" and np.any(u!=u[...,:1]):raise ValueError("断开的通路仍然改变了动作")
    if arm!="direct" and (out["null_eye_max"].max()>2e-4 or out["norm_gap_max"].max()>2e-4):raise ValueError("等幅零空间对照失效")
    return out
def analyze(rel,out):
    checked(rel);out.mkdir(parents=True,exist_ok=False)
    goals=read(rel/"sealed/goals.npz")['goals']
    if not np.array_equal(goals,EXAM) or set(map(tuple,goals))&set(map(tuple,TRAIN)):raise ValueError("考试目标受污染")
    n=C["exam"][1];rng=np.random.default_rng(C["stats_seed"]);draw=rng.integers(n,size=(C["resamples"],n))
    arms={};ss={};vs={};per=[];table=[]
    for arm in C['arms']:
        allseeds=[]
        for seed in C['seeds']:
            name=f"{arm}-{seed}.npz";m=vectors(read(rel/"sealed"/name),read(rel/"observed"/name),arm,goals)
            allseeds.append(m);one={k:float(v.mean()) for k,v in m.items()}
            ss.setdefault(arm,{})[str(seed)]=one;per.extend(dict(arm=arm,seed=seed,metric=k,value=v) for k,v in one.items())
        agg={k:np.mean([m[k] for m in allseeds],0) for k in allseeds[0]};vs[arm]=agg
        arms[arm]={k:interval(v,draw) for k,v in agg.items()}
        table.extend(dict(arm=arm,metric=k,**v) for k,v in arms[arm].items())
    contrasts={o:{k:interval(vs['joint'][k]-vs[o][k],draw) for k in ('pair','task','contested')} for o in ('aux','direct')}
    a=arms['joint'];seeds=list(ss['joint'].values())
    task=bool(a['pair']['value']>=.8 and a['pair']['lo']>.75 and sum(s['pair']>.75 for s in seeds)>=2)
    criteria=dict(task_gate=task,readout_turn_at_least_80pct=a['physical_turn']['value']>=.8,
        both_specificities_at_least_10pp=all(a[k]['value']>=.10 for k in ('vs_null','vs_stay')),
        both_lower_bounds_positive=all(a[k]['lo']>0 for k in ('vs_null','vs_stay')),
        at_least_two_seeds_both_positive=sum(min(s['vs_null'],s['vs_stay'])>0 for s in seeds)>=2)
    gates=dict(task_primary=task,mechanism_primary=all(criteria.values()),criteria=criteria)
    dump(out/"summary.json",dict(protocol=sha(PROTO),goals=len(goals),layouts=n,seeds=C['seeds'],gates=gates,
         arms=arms,paired_contrasts=contrasts,per_seed=ss))
    csvout(out/"metrics.csv",table);csvout(out/"per_seed.csv",per);print(json.dumps(gates),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--release",type=Path,default=REL);p.add_argument("--out",type=Path)
    a=p.parse_args();analyze(a.release,a.out or a.release/"analysis")
