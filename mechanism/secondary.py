"""与自由运行隔离的辅助试次：真实相机移植、未训练结构对照。"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .data import CFG,R,read,save,dump,sha,context
from .exam import SPLITS,lanes,T,config
from .net import Core,levels,electrical
from .analyze import interval,audit_files,csvout

@torch.inference_mode()
def commit(net,empty,b,r,cam,dev):
    choices=[];trains=[];inits=[];amps=[]
    for off in range(0,len(r),384):
        row=r[off:off+384];i,j,k=row[:,:3].T;x=T(context(b,row),dev)
        ctx,h,rough=net.propose(x);p=electrical(rough);q=p.cpu().numpy().astype(int)+1
        def physical(jj,kk):return T(np.concatenate((cam[i,jj,1,q],cam[i,kk,1,q]),1),dev)
        same=physical(j,k);a=physical(1-j,k);n=physical(j,1-k)
        choices.append(torch.stack([electrical(net.refine(ctx,h,p,v,"joint")) for v in (same,a,n)],1).to(torch.int8).cpu().numpy())
        amps.append((levels(same)-levels(a)).abs().mean(1).cpu().numpy()-(levels(same)-levels(n)).abs().mean(1).cpu().numpy())
        rr=row.copy();rr[:,1]=1-j;xd=T(context(b,rr),dev);fixed=T(row[:,7].astype('float32'),dev)
        aq=lanes(cam[i,j,1,row[:,7]+1]);bq=lanes(cam[i,1-j,1,row[:,7]+1]);th=(aq+bq)/2
        for m,rec in ((net,trains),(empty,inits)):
            v=[]
            for xx in (x,xd):
                c,hh,_=m.propose(xx);v.append(lanes(levels(m.imagine(c,hh,fixed)[:,:R]).cpu().numpy()))
            rec.append(((v[0]>th)==(aq>th))&((v[1]>th)==(bq>th))&(th>0))
    return dict(u=np.concatenate(choices),strict_trained=np.concatenate(trains),strict_initial=np.concatenate(inits),amplitude_difference=np.concatenate(amps))

def study(rel,out,dev):
    audit_files(rel);out.mkdir(parents=True,exist_ok=False);files={}; all_metrics=[]
    for split in SPLITS:
        feed=read(rel/"sealed"/split/"input.npz");rows=feed["rows"];b={k:feed[k] for k in ("base","past")}
        cam=read(rel/"observed"/split/"apparatus.npz")["camera"];part=out/split;part.mkdir()
        for seed in CFG["seeds"]:
            config(seed,dev);untrained=Core().to(dev).eval();net=Core().to(dev).eval()
            d=rel/"models"/f"joint_symbol-{seed}";f=d/"last.pt"
            if sha(f)!=json.loads((d/"freeze.json").read_text())["weights"]:raise ValueError("source checkpoint")
            net.load_state_dict(torch.load(f,weights_only=True,map_location=dev));commitment=commit(net,untrained,b,rows,cam,dev)
            p=part/f"{seed}.npz";save(p,**commitment);files[p.relative_to(out).as_posix()]=sha(p)
    dump(out/"second-seal.json",dict(parent=sha(rel/"observed/after.json"),files=files,utc_ns=time.time_ns()))
    for split in SPLITS:
        rows=read(rel/"sealed"/split/"input.npz")["rows"];i,j,k,l,w,auto,ext,home,old,repair=rows.T
        good=read(rel/"observed"/split/"apparatus.npz")["acceptable"];factual=auto==0;n=CFG[split][1]
        draw=np.random.default_rng(80119+n).integers(n,size=(6000,n));store={}
        for seed in CFG["seeds"]:
            p=out/split/f"{seed}.npz"; assert sha(p)==files[p.relative_to(out).as_posix()]
            z=read(p);u=z["u"];win=good[i[:,None],j[:,None],ext[:,None]+1,u+1,l[:,None],w[:,None]]&((u==0)==auto[:,None])
            ex=np.where(good[i,j,1,0,l,w],-1,1)
            d=(win[:,0]&(u[:,1]==-ex)).astype(float)-(win[:,0]&(u[:,2]==-ex))
            ms=dict(real_specific=d,real_native=win[:,0],real_relevant=win[:,1],real_nuisance=win[:,2],
                   trained_strict=z["strict_trained"],initial_strict=z["strict_initial"],
                   absolute_amplitude_difference=np.abs(z["amplitude_difference"]))
            for key,val in ms.items():
                v=val[factual].reshape(n,-1).mean(1);store.setdefault(key,[]).append(v)
                all_metrics.append(dict(split=split,seed=str(seed),metric=key,value=float(v.mean()),lo="",hi=""))
        for key,ss in store.items():all_metrics.append(dict(split=split,seed="pooled",metric=key,**interval(np.mean(ss,0),draw)))
    csvout(out/"metrics.csv",all_metrics)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--release",type=Path,default=Path("evidence/mechanism"))
    p.add_argument("--out",type=Path);p.add_argument("--device",default="cuda");a=p.parse_args();study(a.release,a.out or a.release/"secondary",a.device)
