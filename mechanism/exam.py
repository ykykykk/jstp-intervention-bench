"""封存网络输出在前，实际操作与电路裁判在后；只读真实相机校验任务。"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .data import CFG,PROTO,R,sha,dump,save,read,scene,images,act,retina,apparatus,episodes,context,test_programs,index
from .net import Core,motor,electrical,levels
from .train import T,config

SPLITS=("test_familiar","test_novel")
MM=("natural","matched-repeat","opposite-relevant-panel","opposite-irrelevant-panel","visual-history-twin","flat-history")
LM=("natural","repeat","donor","shuffle","clear")
def specs():return [(arm,seed) for arm in CFG["arms"] for seed in CFG["seeds"]]
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def quant(x):return (levels(x)*15).round().byte().cpu().numpy()
def cat(rows):return {k:np.concatenate([r[k] for r in rows],0) for k in rows[0]}
def names(arm,seed):return f"{arm}-{seed}"

@torch.inference_mode()
def motor_seal(net,mode,bank,rows,dev):
    output=[]
    for start in range(0,len(rows),384):
        r=rows[start:start+384];x=T(context(bank,r),dev)
        def changed(col):
            rr=r.copy();rr[:,col]=1-rr[:,col];return T(context(bank,rr),dev)
        ctx,h,rough,pic,m=net.operation(x,mode);u=electrical(m);p=electrical(rough)
        ta,ha,_=net.propose(changed(1));tb,hb,_=net.propose(changed(2))
        av=net.imagine(ta,ha,p);bv=net.imagine(tb,hb,p)
        def refined(v):return electrical(net.refine(ctx,h,p,v,mode))
        same=refined(net.imagine(ctx,h,p))
        aa=refined(torch.cat((av[:,:R],pic[:,R:]),1));bb=refined(torch.cat((pic[:,:R],bv[:,R:]),1))
        hist=electrical(net.operation(changed(1),mode)[-1]);blank=x.clone();blank[:,R:3*R]=0
        flat=electrical(net.operation(blank,mode)[-1]);sensed,body=net.consequent(ctx,h,u)
        # Measurement trials clamp the same non-adaptive drive. Never used to select a motor.
        goalx=x.clone();goalx[:,3*R+3]*=-1;cg,hg,_=net.propose(goalx)
        fa,_=net.consequent(ctx,h,p);fg,_=net.consequent(cg,hg,p)
        out=dict(u=torch.stack((u,same,aa,bb,hist,flat),1).to(torch.int8).cpu().numpy(),
                 proposal=p.to(torch.int8).cpu().numpy(),felt=body.cpu().numpy().astype('float16'),
                 prospect=quant(pic),donor=quant(av[:,:R]),terminal=quant(sensed),
                 goal_delta=(fa[:,:R]-fg[:,:R]).abs().mean(1).cpu().numpy().astype('float32'),
                 pert=(torch.stack(((levels(pic[:,:R])-levels(av[:,:R])).abs().mean(1),
                                     (levels(pic[:,R:])-levels(bv[:,R:])).abs().mean(1)),1)).cpu().numpy().astype('float16'))
        if not all(np.isfinite(v).all() for v in out.values()):raise ValueError("nonfinite submission")
        output.append(out)
    return cat(output)

@torch.inference_mode()
def language_seal(net,mode,depth,dev):
    initial,cards,q,_=test_programs(depth);init,cs,y=[T(a,dev) for a in (initial,cards,q)]
    donor=torch.arange(len(q),device=dev)^1; answers=[];payload=None
    for action in LM:
        out,mem=net.language(init,cs,y,mode,action,donor)
        answers.append(electrical(motor(out[:,-1])).to(torch.int8).cpu().numpy())
        if action=="natural":payload=index(mem.cpu().numpy().astype('uint8')).astype('uint8')
    return dict(answer=np.stack(answers,1),message=payload,input_sha=np.array(digest(np.column_stack((initial,cards.reshape(len(q),-1),q)))))

def seal(rel,dev):
    dest=rel/"sealed";dest.mkdir(exist_ok=False); files={};wts={};config(80119,dev)
    allbank={}
    for key in SPLITS:
        first,n=CFG[key];b=apparatus(first,n,key.endswith("novel"),False);r=episodes(b,first)
        allbank[key]=(b,r);part=dest/key;part.mkdir()
        save(part/"input.npz",rows=r,base=b["base"],past=b["past"])
        files[str((part/"input.npz").relative_to(dest).as_posix())]=sha(part/"input.npz")
    (dest/"language").mkdir()
    for arm,seed in specs():
        name=names(arm,seed);d=rel/"models"/name;wt=d/"last.pt";meta=json.loads((d/"freeze.json").read_text(encoding="utf8"))
        if sha(wt)!=meta["weights"] or sha(PROTO)!=meta["protocol"]:raise ValueError(name+" freeze")
        wts[name]=sha(wt);net=Core().to(dev).eval();net.load_state_dict(torch.load(wt,map_location=dev,weights_only=True))
        m,l=CFG["arms"][arm]
        for split,(b,r) in allbank.items():
            f=dest/split/(name+".npz");out=motor_seal(net,m,b,r,dev);save(f,**out);files[f.relative_to(dest).as_posix()]=sha(f)
        lang={}
        for depth in CFG["test_depths"]:
            for k,v in language_seal(net,l,depth,dev).items():lang[f"d{depth}_{k}"]=v
        f=dest/"language"/(name+".npz");save(f,**lang);files[f.relative_to(dest).as_posix()]=sha(f)
        print("SEALED",name,flush=True)
    dump(dest/"before.json",dict(protocol=sha(PROTO),weights=wts,files=files,utc_ns=time.time_ns()))

def physical(first,n,novel):
    camera=np.empty((n,2,3,3,R),np.float32); accept=np.empty((n,2,3,3,3,2),bool)
    hashes=np.empty((n,2,3,3),dtype='S64')
    for i in range(n):
        s=scene(first+i,novel);board=s.board
        for j in (0,1):
            b,_=images(s,j);goals={}
            for lane in range(3):
                pos=board.pieces.copy();pos[lane,1]+=4
                sl=np.s_[:43,32*lane:32*(lane+1)]
                goals[lane]=(b[sl],board.render(pos,board.grips)[sl])
            for ext in (-1,0,1):
                for own in (-1,0,1):
                    im=act(s,j,own,ext);camera[i,j,ext+1,own+1]=retina(b,im);hashes[i,j,ext+1,own+1]=digest(im)
                    for lane,(g0,g1) in goals.items():
                        strip=im[:43,32*lane:32*(lane+1)]
                        accept[i,j,ext+1,own+1,lane]=[np.array_equal(strip,g0),np.array_equal(strip,g1)]
    return camera,accept,hashes

def lanes(x):return x.reshape(-1,8,3,4)[:,:5].sum((1,3))
def physical_score(rows,z,cam,acceptable):
    i,j,k,l,w,auto,ext,home,old,repair=rows.T;u=z["u"].astype('int32');p=z["proposal"].astype('int32')
    # Goal: execute a transport with the indicated region moving/staying; in assisted mode the subject must not drive.
    wins=acceptable[i[:,None],j[:,None],ext[:,None]+1,u+1,l[:,None],w[:,None]]&((u==0)==auto[:,None])
    left=acceptable[i,j,1,0,l,w];right=acceptable[i,j,1,2,l,w]
    if not np.all(left^right):raise ValueError("ambiguous target")
    expert=np.where(left,-1,1)
    a=lanes(cam[i,j,1,p+1]);b=lanes(cam[i,1-j,1,p+1]);thr=(a+b)/2
    v=lanes(z["prospect"][:,:R].astype('float32')/15);vv=lanes(z["donor"].astype('float32')/15)
    pair=((v>thr)==(a>thr))&((vv>thr)==(b>thr))&(p[:,None]!=0)&(thr>0)
    target=np.concatenate((cam[i,j,ext+1,u[:,0]+1],cam[i,k,ext+1,u[:,0]+1]),1)
    return dict(win=wins,expert=expert.astype("int8"),strict=pair,
                terminal_mae=np.abs(z["terminal"].astype("float32")/15-target).mean(1),
                bodily_mae=np.abs(z["felt"].astype("float32")-(u[:,0]+ext)))

def reference_circuit(init,cards):
    # Independent scalar executor, not the vectorized training generator.
    states=[]
    for start,program in zip(init,cards):
        lamps=list(map(int,start));history=[]
        for s,d,p in program:
            if lamps[int(s)]==int(p):lamps[int(d)]=1-lamps[int(d)]
            history.append(lamps.copy())
        states.append(history)
    return np.asarray(states,np.uint8)
def verify(rel):
    sealed=rel/"sealed";lock=json.loads((sealed/"before.json").read_text(encoding="utf8"))
    if lock["protocol"]!=sha(PROTO):raise ValueError("protocol changed")
    for name,d in lock["weights"].items():
        if d!=sha(rel/"models"/name/"last.pt"):raise ValueError("weight changed: "+name)
    for file,d in lock["files"].items():
        if d!=sha(sealed/file):raise ValueError("prediction changed: "+file)
    return lock
def score(rel,out):
    verify(rel);sealed=rel/"sealed";out.mkdir(parents=True,exist_ok=False); hashes={}
    for split in SPLITS:
        rows=read(sealed/split/"input.npz")["rows"];first,n=CFG[split]
        cam,good,hist=physical(first,n,split.endswith("novel"));part=out/split;part.mkdir()
        save(part/"apparatus.npz",camera=cam,acceptable=good,frame_sha=hist)
        for arm,seed in specs():
            name=names(arm,seed);rr=physical_score(rows,read(sealed/split/(name+".npz")),cam,good);save(part/(name+".npz"),**rr)
        print("EXECUTED",split,n,flush=True)
    theory={}
    for d in CFG["test_depths"]:
        init,cards,q,t=test_programs(d); actual=reference_circuit(init,cards)
        if not np.array_equal(actual,t):raise ValueError("independent executor disagrees")
        answer=actual[np.arange(len(q)),-1,q].astype('int8')*2-1
        if not np.all(answer[::2]==-answer[1::2]):raise ValueError("circuit pairing")
        theory[f"d{d}_answer"]=answer;theory[f"d{d}_states"]=index(actual).astype("uint8")
    save(out/"circuits.npz",**theory)
    for f in sorted(out.rglob("*.npz")):hashes[f.relative_to(out).as_posix()]=sha(f)
    dump(out/"after.json",dict(seal=sha(sealed/"before.json"),files=hashes,utc_ns=time.time_ns()))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["seal","score"]);p.add_argument("--release",type=Path,default=Path("evidence/mechanism"))
    p.add_argument("--out",type=Path);p.add_argument("--device",default="cuda");a=p.parse_args()
    seal(a.release,a.device) if a.action=="seal" else score(a.release,a.out or a.release/"observed")
