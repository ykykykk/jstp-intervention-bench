"""CPU-only recount; resample whole layouts, paired across conditions, with seeds held fixed."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from mechanism.data import read,dump,sha
from lategoal.rig import EXAM
from .train import C,DEFAULT
from .exam import check,triples

def count(x,den=None):
    n=len(x);x=np.asarray(x);d=np.ones_like(x,dtype=bool) if den is None else den
    return np.stack((x.reshape(n,-1).sum(1),d.reshape(n,-1).sum(1)),1).astype("float64")
def ratios(a):
    return a[...,0]/a[...,1]
def metrics(z,w,truth):
    p=z["probe"];u=z["u"];v=u[...,0];n=len(p);i=np.arange(n)
    native=w[...,0]; left=truth["left"]; good=np.where(left@EXAM.T>0,-1,1)
    if not np.array_equal(native,v==good):raise ValueError("意向与相机裁判不符")
    ref=(-p[:,None,None])*left
    group=(z["group"]==ref).reshape(n,2,2,4).all((2,3))
    blind=(z["blind"]==ref).reshape(n,2,2,4).all((2,3))
    m=dict(task=count(native),pair=count(native[...,::2]&native[...,1::2]),group=count(group),blind=count(blind))
    if "eye" in z:
        y=truth["reference_eye"][i,:,(p>0).astype(int)]
        m["optical_mse"]=count((z["eye"]-y)**2)
    if "patched_eye" in z:
        dg=good[:,[1,0,3,2]];change=good!=dg
        m["directed"]=count(native&change&(u[...,1]==dg),change)
        m["null"]=count(native&change&(u[...,2]==dg),change)
        m["false"]=count(native&~change&(u[...,1]!=good),~change)
        dy=y[:,[1,0,3,2]]; dist=lambda x,t:np.sum((x-t)**2,axis=-1)
        m["physical_fit"]=count((dist(z["eye"],y)<dist(z["eye"],dy))&(dist(z["patched_eye"],dy)<dist(z["patched_eye"],y)))
    return m
def distribution(c,draw):
    # Multinomial multiplicities give exactly the usual paired whole-layout bootstrap.
    return float(ratios(c.sum(0))),ratios(draw@c)
def summary(p,reps):
    lo,hi=np.quantile(reps,[.025,.975]);return dict(value=float(p),lo=float(lo),hi=float(hi))
def analyze(rel):
    dest=rel/"analysis";dest.mkdir();rows=[];seeds=[];all_results={}
    for panel,(_,n) in C["panels"].items():
        check(rel,f"{panel}/after.json");check(rel,f"{panel}/before.json")
        d=rel/panel;truth=read(d/"observed/camera.npz"); c={}
        rng=np.random.default_rng(44178);draw=rng.multinomial(n,np.full(n,1/n),3000).astype("float64")
        for arm,seed,name in triples():
            z=read(d/"sealed"/(name+".npz"));w=read(d/"observed"/(name+".npz"))["wins"]
            ms=metrics(z,w,truth); c.setdefault(arm,{});vals={}
            for k,ct in ms.items():
                c[arm][k]=c[arm].get(k,0)+ct; v=float(ratios(ct.sum(0)))
                vals[k]=v
            seeds.append(dict(panel=panel,arm=arm,seed=seed,metrics=vals))
        table={};dist={}
        for a,ms in c.items():
            dists={k:distribution(v,draw) for k,v in ms.items()};dist[a]=dists
            if "directed" in dists:
                for k,x in (("over_null","null"),("over_false","false")):
                    dists[k]=tuple(i-j for i,j in zip(dists["directed"],dists[x]))
            table[a]={k:summary(*v) for k,v in dists.items()}
            table[a]["group_over_blind"]=summary(*(i-j for i,j in zip(dists["group"],dists["blind"])))
            if "optical_mse" in dists:table[a]["optical_rmse"]=summary(*(np.sqrt(v) for v in dists["optical_mse"]))
        contrasts={a:summary(*(i-j for i,j in zip(dist["partial"]["pair"],dist[a]["pair"]))) for a in c if a!="partial"}
        for a,ms in table.items():
            for k,v in ms.items():rows.append(dict(panel=panel,arm=a,metric=k,**v))
        all_results[panel]=dict(arms=table,partial_pair_difference=contrasts,layouts=n)
    dump(dest/"summary.json",all_results);dump(dest/"per_seed.json",seeds)
    with (dest/"metrics.csv").open("x",encoding="utf8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=["panel","arm","metric","value","lo","hi"]);wr.writeheader();wr.writerows(rows)
    return all_results
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=DEFAULT);a=p.parse_args()
    for k,v in analyze(a.out).items():
        print(k,{x:round(y["pair"]["value"],4) for x,y in v["arms"].items()})

