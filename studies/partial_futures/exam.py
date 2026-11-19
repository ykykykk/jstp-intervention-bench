"""Calibrate readouts on old calibration layouts; seal commands before any scored stroke."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from lategoal.rig import Rig,first,history,bank,EXAM,F,goal_wins
from lategoal.control import fingerprint,axes,T
from lategoal.exam import actual
from mechanism.data import sha,dump,save,read
from .train import C,HERE,PROTOCOL,DEFAULT,load,SUP

def index(rel,filename,paths):
    dump(rel/filename,dict(protocol=sha(PROTOCOL),files={p.relative_to(rel).as_posix():sha(p) for p in paths}))
def check(rel,filename):
    doc=json.loads((rel/filename).read_text(encoding="utf8"))
    if sha(PROTOCOL)!=doc["protocol"]:raise ValueError("协议指纹改变")
    for p,d in doc["files"].items():
        if sha(rel/p)!=d:raise ValueError("文件指纹改变："+p)
def triples(arms=None):
    return [(a,s,f"{a}-{s}") for a in (arms or C["arms"]) for s in C["seeds"]]
def prehistory(layouts):
    start=[];states=[];peaks=[]
    for r in layouts:
        start.append(first(r)); states1=[]; peaks1=[]
        for p in (-1,1):
            ims=[r.world(s).probe(p) for s in (0,1)]
            states1.append(np.stack([history(r,im,p,g) for im in ims for g in (0,1)]))
            peaks1.append([fingerprint(im) for im in ims])
        states.append(states1);peaks.append(peaks1)
    return np.stack(start),np.asarray(states),np.asarray(peaks)
def choose(m,start):
    return np.where(m.probe(T(start)).numpy()>=0,1,-1).astype("int8")
def ridge(x,y):
    x,y=x.astype("float64"),y.astype("float64");xm=x.mean(0);ym=y.mean(0)
    a=x-xm;reg=1e-3*np.trace(a.T@a)/a.shape[1]
    w=np.linalg.solve(a.T@a+max(reg,1e-8)*np.eye(a.shape[1]),a.T@(y-ym))
    return dict(rw=w.astype("float32"),rb=(ym-xm@w).astype("float32"))
@torch.inference_mode()
def calibrate(rel):
    children=[rel/"models"/name/"freeze.json" for _,_,name in triples()]
    index(rel,"frozen.json",children+[p.with_name("last.pt") for p in children])
    b=bank("calibration");i=np.arange(len(b["start"]));files=[rel/"frozen.json"]
    for arm,seed,name in triples():
        m=load(rel,arm,seed);p=choose(m,b["start"]);k=(p>0).astype(int)
        x=T(b["hist"][i,:,:,k].reshape(len(i)*4,-1));drive=T(np.repeat(p,4))
        h=m.observe(x);z=m.representation(h,drive)
        fit=ridge(z.numpy(),b["motion"][i,:,:,k].reshape(-1,4))
        if arm in ("partial","detached"):fit.update(axes(b["eye"][i,:,:,k],m))
        dst=rel/"calibration"/(name+".npz");dst.parent.mkdir(exist_ok=True);save(dst,**fit);files.append(dst)
    index(rel,"calibrated.json",files)
def score_chunks(fn,h,p,g):
    return torch.cat([fn(hi,pi,gi) for hi,pi,gi in zip(h.split(768),p.split(768),g.split(768))])
@torch.inference_mode()
def intended(m,arm,start,states,peaks,fit):
    n=len(start);G=len(EXAM);i=np.arange(n);p=choose(m,start);k=(p>0).astype(int)
    x=T(states[i,k].reshape(n*4,-1));pp=T(np.repeat(p,4));h=m.observe(x);z=m.representation(h,pp)
    mask=x.clone();mask[:,F:2*F+12]=0
    zblind=m.representation(m.observe(mask),pp)
    pred=lambda a:np.where(a.numpy()@fit["rw"]+fit["rb"]>=0,1,-1).astype("int8").reshape(n,4,4)
    def repeat(a):return a[:,None,:].expand(-1,G,-1).reshape(-1,a.shape[-1])
    g=T(np.tile(EXAM,(n*4,1)));hh=repeat(h); pg=pp.repeat_interleave(G)
    u=score_chunks(m.policy,hh,pg,g).reshape(n,4,G)
    out=dict(probe=p,peak=peaks[i,k],group=pred(z),blind=pred(zblind))
    if arm in SUP:out["eye"]=m.eye(z).numpy().reshape(n,4,12)
    us=[u]
    if arm in ("partial","detached"):
        src=z.reshape(n,4,-1);donor=src[:,[1,0,3,2]]
        q,nul=T(fit["Q"]),T(fit["N"]);coef=(donor-src)@q
        physical=(src+coef@q.T).reshape_as(z); unrel=(src+coef@nul.T).reshape_as(z)
        for v in (physical,unrel):
            us.append(score_chunks(lambda a,b,c:m.issue(a,b,c),hh,repeat(v),g).reshape(n,4,G))
        out["patched_eye"]=m.eye(physical).numpy().reshape(n,4,12)
        out["max_norm_gap"]=np.float32(((physical-z).norm(dim=1)-(unrel-z).norm(dim=1)).abs().max())
    out["u"]=np.where(torch.stack(us,-1).numpy()>=0,1,-1).astype("int8")
    return out
def seal(rel,panel):
    check(rel,"calibrated.json");check(rel,"frozen.json")
    root=rel/panel; d=root/"sealed"; d.mkdir(parents=True)
    start,n=C["panels"][panel];ids=np.arange(start,start+n,dtype="int32")
    s,x,pk=prehistory([Rig(int(i)) for i in ids]);files=[]
    f=d/"items.npz";save(f,ids=ids,goals=EXAM);files.append(f)
    for arm,seed,name in triples():
        m=load(rel,arm,seed);fit=read(rel/"calibration"/(name+".npz"))
        f=d/(name+".npz");save(f,**intended(m,arm,s,x,pk,fit));files.append(f)
    index(rel,f"{panel}/before.json",files+[rel/"calibrated.json"])
def execute(rel,panel):
    check(rel,f"{panel}/before.json");root=rel/panel;d=root/"sealed"; out=root/"observed";out.mkdir()
    meta=read(d/"items.npz");ids,goals=meta["ids"],meta["goals"]
    names=[name for _,_,name in triples()];sets={name:read(d/(name+".npz")) for name in names}
    audited={name:np.empty(a["u"].shape,dtype=np.bool_) for name,a in sets.items()}
    left=np.empty((len(ids),4,4),"int8");opt=np.empty((len(ids),4,2,12),"float32")
    witnesses=np.empty((len(ids),4,2),dtype="S64")
    for i,layout in enumerate(ids):
        r=Rig(int(layout))
        for state in range(4):
            w,g=divmod(state,2)
            for pi,p in enumerate((-1,1)):
                use=[(name,a) for name,a in sets.items() if a["probe"][i]==p]
                pieces=[a["u"][i,state].ravel() for _,a in use]
                digest=fingerprint(r.world(w).probe(p))
                if any(a["peak"][i,w]!=digest for _,a in use):raise ValueError("实际探查与封存前缀不同")
                commands=np.concatenate(pieces) if pieces else np.empty(0,"int8")
                vel,ref,eye,hs=actual(r,w,g,p,commands,digest)
                if pi and not np.array_equal(left[i,state],ref):raise ValueError("试探复位后物理不一致")
                left[i,state]=ref;opt[i,state,pi]=eye;witnesses[i,state]=hs
                off=0
                for (name,a),q in zip(use,pieces):
                    shape=a["u"][i,state].shape;v=vel[off:off+len(q)].reshape(*shape,4)
                    audited[name][i,state]=goal_wins(v,goals[:,None]);off+=len(q)
    f=out/"camera.npz";save(f,left=left,reference_eye=opt,frame_sha=witnesses);files=[f,root/"before.json"]
    for name,y in audited.items():
        f=out/(name+".npz");save(f,wins=y);files.append(f)
    index(rel,f"{panel}/after.json",files)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=DEFAULT);a=parser.parse_args()
    torch.set_num_threads(2);calibrate(a.out)
    for panel in C["panels"]:seal(a.out,panel)
    for panel in C["panels"]:execute(a.out,panel)

