"""CPU-only layout/pair cluster analysis. No checkpoint or training framework is imported."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .data import CFG,PROTO,CODE,read,dump,sha

def interval(v,draw):
    v=np.asarray(v,dtype=float); boot=v[draw].mean(1)
    return dict(value=float(v.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)))
def csvout(path,rows):
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def motor_metrics(rows,z,obs,n,visual):
    active=rows[:,5]==0; passive=~active;u=z["u"];ex=obs["expert"]
    vec=lambda x,sel=active:np.asarray(x[sel],float).reshape(n,-1).mean(1)
    base=obs["win"][:,0]; relevance=base&(u[:,2]==-ex); noise=base&(u[:,3]==-ex)
    v={"task":vec(base),"relevant_direction":vec(relevance),"irrelevant_direction":vec(noise),
       "specific":vec(relevance.astype(int)-noise.astype(int)),"switched_relevant":vec(u[:,2]!=u[:,0]),
       "switched_irrelevant":vec(u[:,3]!=u[:,0]),"repeat_changed":vec(u[:,1]!=u[:,0]),
       "real_history_swapped":vec(obs["win"][:,4]),"flat_history":vec(obs["win"][:,5]),
       "passive_task":vec(base,passive),"passive_rest":vec(u[:,0]==0,passive),
       "home_match":vec(z["proposal"]==rows[:,7]),"provisional_task":vec(z["proposal"]==ex),
       "goal_pair":base[active].reshape(n,-1,2).all(2).mean(1),
       "goal_switch":(u[active,0].reshape(n,-1,2)[:,:,0]!=u[active,0].reshape(n,-1,2)[:,:,1]).mean(1)}
    if visual:
        v.update(strict=vec(obs["strict"]),terminal_mae=vec(obs["terminal_mae"]),body_mae=vec(obs["bodily_mae"]),
                 clamp_goal_raw_mae=vec(z["goal_delta"]),relevant_amplitude=vec(z["pert"][:,0]),irrelevant_amplitude=vec(z["pert"][:,1]))
        source=(obs["bodily_mae"][active]<.5)&(obs["bodily_mae"][passive]<.5)&base[active]&base[passive]
        v["active_passive_pair"]=source.reshape(n,-1).mean(1)
    return v
def circuit_metrics(z,truth,d,mode):
    a=z[f"d{d}_answer"];real=truth[f"d{d}_answer"];ok=a==real[:,None]
    agg=lambda x:np.asarray(x,dtype=float).reshape(-1,2).mean(1)
    v=dict(accuracy=agg(ok[:,0]),both=ok[:,0].reshape(-1,2).all(1),
           cleared=agg(ok[:,4]),repeat_changed=agg(a[:,0]!=a[:,1]),shuffled=agg(ok[:,3]),
           invalid=agg(a[:,0]==0))
    if d>1:
        v["donor_target"]=agg(a[:,2]==-real)
        v["directional"]=agg(ok[:,0]&(a[:,2]==-real))
    if mode in ("symbol","holistic"):
        state=truth[f"d{d}_states"];expected=state if mode=="symbol" else CODE[state]
        v["report_correct"]=agg((z[f"d{d}_message"]==expected).mean(1))
    return v

def audit_files(rel):
    before=rel/"sealed/before.json";lock=json.loads(before.read_text());after=json.loads((rel/"observed/after.json").read_text())
    if lock["protocol"]!=sha(PROTO) or after["seal"]!=sha(before):raise ValueError("bundle mismatch")
    for sub,items in ((rel/"sealed",lock["files"]),(rel/"observed",after["files"])):
        for f,h in items.items():
            if sha(sub/f)!=h:raise ValueError(f"unmatched evidence: {f}")

def analyze(rel,out):
    audit_files(rel);out.mkdir(parents=True,exist_ok=False); results=[];per=[];vectors={}
    truth=read(rel/"observed/circuits.npz")
    for split in ["test_familiar","test_novel"]+["depth"+str(d) for d in CFG["test_depths"]]:
        physical=split.startswith("test"); n=CFG[split][1] if physical else CFG["language_pairs_per_depth"]
        rng=np.random.default_rng(80119+n);draw=rng.integers(n,size=(6000,n))
        if physical:rows=read(rel/"sealed"/split/"input.npz")["rows"]
        for arm,(m,l) in CFG["arms"].items():
            group=[]
            for seed in CFG["seeds"]:
                name=f"{arm}-{seed}.npz"
                if physical:
                    z=read(rel/"sealed"/split/name);obs=read(rel/"observed"/split/name);vs=motor_metrics(rows,z,obs,n,m!="direct")
                else:
                    d=int(split[5:]);z=read(rel/"sealed/language"/name);vs=circuit_metrics(z,truth,d,l)
                group.append(vs)
                per.extend(dict(split=split,arm=arm,seed=seed,metric=k,value=float(v.mean())) for k,v in vs.items())
            agg={k:np.mean([v[k] for v in group],0) for k in group[0]};vectors[split,arm]=agg
            results.extend(dict(split=split,arm=arm,metric=k,**interval(v,draw)) for k,v in agg.items())
        others=("aux_symbol","direct_symbol") if physical else ("joint_holistic","joint_latent")
        for other in others:
            for k in (("task","specific") if physical else ("accuracy","both")):
                v=vectors[split,"joint_symbol"][k]-vectors[split,other][k]
                results.append(dict(split=split,arm="joint_symbol - "+other,metric=k,**interval(v,draw)))
    def get(s,a,k):return next(v for v in results if (v["split"],v["arm"],v["metric"])==(s,a,k))
    def raw(s,a,k):return [v["value"] for v in per if (v["split"],v["arm"],v["metric"])==(s,a,k)]
    g=lambda k:get("test_familiar","joint_symbol",k)
    motor_checks=dict(specific_at_least_point_10=g("specific")["value"]>=.1,
        lower_positive=g("specific")["lo"]>0,positive_seeds=sum(x>0 for x in raw("test_familiar","joint_symbol","specific"))>=2,
        task_lower_over_point_60=g("task")["lo"]>.6,strict_over_half=g("strict")["value"]>.5)
    a,b=[raw("depth"+str(d),"joint_latent","accuracy") for d in (6,8)]
    decision=dict(motor_gates=motor_checks,motor_primary=all(motor_checks.values()),
                  nonlexical_passing_seeds=[s for s,x,y in zip(CFG["seeds"],a,b) if min(x,y)>.75],
                  explicit_symbol_port_not_necessary=sum(min(x,y)>.75 for x,y in zip(a,b))>=2)
    csvout(out/"metrics.csv",results);csvout(out/"per_seed.csv",per)
    dump(out/"summary.json",dict(protocol=sha(PROTO),gates=decision,metrics=results))
    print(json.dumps(decision),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--release",type=Path,default=Path("evidence/mechanism"));p.add_argument("--out",type=Path)
    a=p.parse_args();analyze(a.release,a.out or a.release/"analysis")
