"""先冻结并校准，再封存全部意向；最后在真实虚拟机构中逐提交驱动、从相机取证。"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .rig import C,PROTO,REL,EXAM,Rig,bank,save,read,sha,dump,goal_wins,optical_delta,lane_travel
from .control import T,MODES,sign,fingerprint,load_net,collect,axes,decisions
def models():
    return [(a,s,f"{a}-{s}") for a in C['arms'] for s in C['seeds']]
def check(rel,manifest):
    data=json.loads((rel/manifest).read_text(encoding="utf8"))
    if data["protocol"]!=sha(PROTO):raise ValueError("协议指纹改变")
    for key,checksum in data["files"].items():
        if sha(rel/key)!=checksum:raise ValueError("证据链错误："+key)
    return data
def lock(rel,filename,paths,**fields):
    dump(rel/filename,dict(protocol=sha(PROTO),files={p.relative_to(rel).as_posix():sha(p) for p in paths},
                           utc_ns=time.time_ns(),**fields))
@torch.inference_mode()
def calibration(rel):
    d=rel/"calibration";d.mkdir(exist_ok=False);f=[rel/"experience.json"]
    for _,_,name in models():f.extend([rel/"models"/name/"freeze.json",rel/"models"/name/"last.pt"])
    lock(rel,"frozen.json",f);b=bank("calibration");files=[rel/"frozen.json"]
    for arm,seed,name in models():
        m=load_net(rel,arm,seed)
        if arm=="direct":continue
        p=sign(m.probe(T(b["start"]))).numpy();idx=(p>0).astype(int)
        y=np.stack([b["eye"][i,:,:,k] for i,k in enumerate(idx)])
        fn=d/(name+".npz");save(fn,**axes(y,m));files.append(fn)
    lock(rel,"calibrated.json",files)
def seal(rel):
    check(rel,"frozen.json");check(rel,"calibrated.json")
    d=rel/"sealed";d.mkdir();start,n=C["exam"];rigs=[Rig(start+i) for i in range(n)];files=[rel/"calibrated.json"]
    card=d/"goals.npz";save(card,goals=EXAM,ids=np.arange(start,start+n,dtype=np.int32));files.append(card)
    for arm,seed,name in models():
        m=load_net(rel,arm,seed);c=collect(m,rigs);basis=None if arm=="direct" else read(rel/"calibration"/(name+".npz"))
        f=d/(name+".npz");save(f,**decisions(m,arm,c,EXAM,basis));files.append(f)
        print("SEALED",name,flush=True)
    lock(rel,"before.json",files,modes=MODES)
def actual(r,s,g,probe,cmds,phash):
    w=r.world(s);peak=w.probe(probe)
    if fingerprint(peak)!=phash:raise ValueError("复位重放的真实经历不一致")
    w.set_clutch(g);sides=np.r_[cmds.ravel(),-1,1].astype(np.int32)
    pose,hand=w.stroke_copies(sides);vel=np.empty((len(sides),4),np.int8);eyes=[];hashes=[]
    # 每个提交都从同一位置推进。相同动作产生完全相同的姿态，只渲染每种实际姿态一次。
    for index,side in ((-2,-1),(-1,1)):
        subset=sides==side
        if not (np.all(pose[subset]==pose[index]) and np.all(hand[subset]==hand[index])):raise ValueError("错误的相机复用")
        im=r.board.render(pose[index],hand[index])
        vel[subset]=lane_travel(r.base,im);eyes.append(optical_delta(r.base,im)[1]);hashes.append(fingerprint(im))
    if not np.array_equal(vel[-2],-vel[-1]):raise ValueError("两个把手的实物后果不互补")
    return vel[:-2].reshape(*cmds.shape,4),vel[-2],eyes[int(probe>0)],hashes
def execute(rel):
    check(rel,"frozen.json");check(rel,"calibrated.json");check(rel,"before.json")
    d=rel/"observed";d.mkdir();files=[rel/"before.json"];gg=read(rel/"sealed/goals.npz");goals,ids=gg["goals"],gg["ids"]
    for arm,seed,name in models():
        z=read(rel/"sealed"/(name+".npz"));out=[]
        for i,num in enumerate(ids):
            r=Rig(int(num));part=[]
            if fingerprint(r.base)!=z["base"][i]:raise ValueError("实际初始画面不符")
            for state in range(4):
                s,g=divmod(state,2);v,a,eye,hs=actual(r,s,g,z["probe"][i],z["u"][i,state],z["peak"][i,s])
                part.append((goal_wins(v,goals[:,None]),a,eye,np.asarray(hs)))
            out.append(tuple(map(np.stack,zip(*part))))
        f=d/(name+".npz");save(f,**dict(zip(("wins","left","reference_eye","frame_sha"),map(np.stack,zip(*out)))));files.append(f)
        print("EXECUTED",name,flush=True)
    lock(rel,"after.json",files)
def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("calibrate","seal","execute"));p.add_argument("--release",type=Path,default=REL)
    a=p.parse_args();torch.set_num_threads(2);{"calibrate":calibration,"seal":seal,"execute":execute}[a.stage](a.release)
if __name__=="__main__":main()
