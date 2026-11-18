"""冻结后续研究：用另一批物理经历标定分析方向，从新题的中间隐状态做局部移植。"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .data import CFG,R,read,save,dump,sha,apparatus,episodes,context
from .net import Core,electrical,levels
from .train import config,T
from .exam import physical,lanes
from .analyze import interval,csvout
PLAN=Path(__file__).with_name("latent-plan.json")
L=json.loads(PLAN.read_text(encoding="utf8"))
def active_rows(b,f):
    rr=episodes(b,f);return rr[rr[:,5]==0]
def alt(rows,col):
    r=rows.copy();r[:,col]=1-r[:,col];return r
@torch.inference_mode()
def embed(m,b,r,dev):
    x=T(context(b,r),dev);return torch.cat(m.propose(x)[:2],1)
def directions(delta,truth,rank):
    mat=(delta.T@truth)/len(truth)
    return np.linalg.svd(mat,full_matrices=False)[0][:,:rank].astype("float32")
def calibrate(m,b,r,dev,seed):
    z=embed(m,b,r,dev).cpu().numpy();sd=np.maximum(z.std(0),L["std_floor"]);i,j,k=r[:,:3].T
    arrays=[]
    for col,wire in ((1,j),(2,k)):
        zz=embed(m,b,alt(r,col),dev).cpu().numpy()
        yes=lanes(b["futures"][i,wire,0])>1e-6; no=lanes(b["futures"][i,1-wire,0])>1e-6
        arrays.append(directions((zz-z)/sd,no.astype(float)-yes.astype(float),L["rank"]))
    rng=np.random.default_rng(L["random_seed"]+seed)
    rand=np.linalg.qr(rng.normal(size=(z.shape[1],L["rank"])))[0].astype("float32")
    return dict(sd=sd,A=arrays[0],N=arrays[1],U=rand)
@torch.inference_mode()
def interventions(m,mode,b,r,basis,dev):
    from .exam import cat
    sd,A,N,U=[T(basis[k],dev) for k in ("sd","A","N","U")];out=[]
    for i in range(0,len(r),384):
        rr=r[i:i+384];x=T(context(b,rr),dev);ctx,h,rough=m.propose(x)
        z=torch.cat((ctx,h),1);p=electrical(rough);vis=m.imagine(ctx,h,p)
        dr=(embed(m,b,alt(rr,1),dev)-z)/sd;dn=(embed(m,b,alt(rr,2),dev)-z)/sd
        proj=lambda v,base:(v@base)@base.T
        relevant=proj(dr,A);norm=relevant.norm(dim=1,keepdim=True)
        def match(v):
            length=v.norm(dim=1,keepdim=True)
            if (length<1e-9).any():raise ValueError("cannot norm-match a zero direction")
            return v*norm/length
        patches=[torch.zeros_like(z),relevant,match(proj(dn,N)),match(proj(dr,U)),torch.zeros_like(z)]
        drive=[];sights=[]
        for delta in patches:
            c,hh=(z+sd*delta).chunk(2,1)
            drive.append(electrical(m.refine(c,hh,p,vis,mode)))
            sights.append((levels(m.imagine(c,hh,p)[:,:R])*15).round().byte())
        out.append(dict(u=torch.stack(drive,1).to(torch.int8).cpu().numpy(),eye=torch.stack(sights,1).cpu().numpy(),
                        prop=p.to(torch.int8).cpu().numpy(),norm=norm[:,0].cpu().numpy()))
    return cat(out)
def study(rel,out,dev):
    out.mkdir(parents=True,exist_ok=False);base,n=L["calibration"]
    calib=apparatus(base,n,False,True);rc=active_rows(calib,base);cache={};files={}
    for sp in ("familiar","novel"):
        a,n=L[sp];b=apparatus(a,n,sp=="novel");r=active_rows(b,a);cache[sp]=(b,r);(out/sp).mkdir()
        save(out/sp/"input.npz",rows=r,base=b["base"],past=b["past"])
    (out/"axes").mkdir()
    for arm in L["arms"]:
        for seed in L["seeds"]:
            name=f"{arm}-{seed}";d=rel/"models"/name;weight=d/"last.pt";config(seed,dev)
            if sha(weight)!=json.loads((d/"freeze.json").read_text())["weights"]:raise ValueError("frozen input")
            net=Core().to(dev).eval();net.load_state_dict(torch.load(weight,map_location=dev,weights_only=True))
            axes=calibrate(net,calib,rc,dev,seed);save(out/"axes"/(name+".npz"),**axes)
            for sp,(b,r) in cache.items():
                f=out/sp/(name+".npz");save(f,**interventions(net,CFG["arms"][arm][0],b,r,axes,dev))
    for f in sorted(out.rglob("*.npz")):files[f.relative_to(out).as_posix()]=sha(f)
    dump(out/"before.json",dict(plan=sha(PLAN),files=files,utc_ns=time.time_ns()))
    results=[]
    for sp,(_,rows) in cache.items():
        start,n=L[sp];cam,good,dig=physical(start,n,sp=="novel");save(out/sp/"physical.npz",camera=cam,frame_sha=dig,goal=good)
        ix,j,k,l,w=rows[:,:5].T; ex=np.where(good[ix,j,1,0,l,w],-1,1);rng=np.random.default_rng(80119+n);draw=rng.integers(n,size=(6000,n))
        for arm in L["arms"]:
            group={}
            for seed in L["seeds"]:
                f=out/sp/f"{arm}-{seed}.npz";assert sha(f)==files[f.relative_to(out).as_posix()]
                z=read(f);u=z["u"];p=z["prop"].astype('int32')+1
                eye=z["eye"].astype('float32')/15
                source=cam[ix,j,1,p];donor=cam[ix,1-j,1,p]
                ds=np.abs(eye-source[:,None]).mean(-1);dd=np.abs(eye-donor[:,None]).mean(-1)
                baseline=u[:,0]==ex;changes=baseline[:,None]&(u==-ex[:,None])
                phys=(ds[:,0]<dd[:,0])[:,None]&(dd<ds);both=changes&phys
                measures=dict(task=baseline,repeat_changed=u[:,0]!=u[:,4],norm=z["norm"])
                for v,key in enumerate(("relevant","nuisance","random"),1):
                    measures[key+"_motor"]=changes[:,v]
                    if arm!="direct_symbol":
                        measures[key+"_physical"]=phys[:,v];measures[key+"_both"]=both[:,v]
                for c,key in enumerate(("nuisance","random"),2):
                    measures["motor_minus_"+key]=changes[:,1].astype(float)-changes[:,c]
                    if arm!="direct_symbol":measures["both_minus_"+key]=both[:,1].astype(float)-both[:,c]
                for key,v in measures.items():
                    vec=v.reshape(n,-1).mean(1);group.setdefault(key,[]).append(vec)
                    results.append(dict(split=sp,arm=arm,seed=str(seed),metric=key,value=float(vec.mean()),lo="",hi=""))
            for key,v in group.items():results.append(dict(split=sp,arm=arm,seed="pooled",metric=key,**interval(np.mean(v,0),draw)))
    csvout(out/"metrics.csv",results)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--release",type=Path,default=Path("evidence/mechanism"));p.add_argument("--out",type=Path);p.add_argument("--device",default="cuda")
    a=p.parse_args();study(a.release,a.out or a.release/"latent_followup",a.device)
