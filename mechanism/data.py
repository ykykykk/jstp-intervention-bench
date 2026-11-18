"""双台虚拟致动器、感觉轨迹、目标先于行动的行为日志，以及三继电器试卷。"""
from pathlib import Path
import hashlib, json
import numpy as np
from jstp.arena import scene
from jstp.v7_world import TABLE

ROOT=Path(__file__).resolve().parents[1]
PROTO=Path(__file__).with_name("protocol.json")
CFG=json.loads(PROTO.read_text(encoding="utf8"))
R=96
CTX=R*3+9
CODE=np.array([2,7,0,5,6,1,3,4],np.int64) # 不是寄存器重排或按位取反

def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save(p,**d):
    with Path(p).open('xb') as f:np.savez_compressed(f,**d)
def read(p):
    with np.load(p,allow_pickle=False) as x:return dict(x)
def grid(v):return v.reshape(8,8,12,8,3).mean((1,3,4)).astype(np.float32).ravel()
def retina(a,b):return np.clip(4*grid(np.abs(a.astype(np.int16)-b.astype(np.int16)))/255,0,1)
def pulse(w,b,drive,axis=1,reset=False):
    if drive==0:return w.camera()
    delta=np.zeros(2,int);delta[axis]=4
    w.press(b.grips[int(drive>0)]); im=w.move(delta)
    if reset:w.move(-delta)
    w.release();return im
def images(s,j):
    w=s.world(j);base=w.camera();peak=pulse(w,s.board,-1,axis=0,reset=True)
    if not np.array_equal(base,w.camera()):raise ValueError("probe did not reset")
    return base,peak
def act(s,j,own,external=0):
    # 保持是物理上的零输出，不是“取两个把手分数的较大者”。
    w=s.world(int(j))
    if external:pulse(w,s.board,int(external))
    return pulse(w,s.board,int(own))
def flags_of(base,peak):
    return np.array([np.any(base[:TABLE,32*i:32*i+32]!=peak[:TABLE,32*i:32*i+32]) for i in range(3)])
def apparatus(first,n,novel,teaching=False):
    bank=dict(base=[],past=[],observed_move=[])
    if teaching:bank["futures"]=[]
    for seed in range(first,first+n):
        s=scene(seed,novel); past=[]; flags=[]; fs=[]
        for j in (0,1):
            b,p=images(s,j);past.append(retina(b,p));flags.append(flags_of(b,p))
            if teaching:fs.append([retina(b,act(s,j,a)) for a in (-1,0,1)])
        bank["base"].append(grid(b)/255);bank["past"].append(past);bank["observed_move"].append(flags)
        if teaching:bank["futures"].append(fs)
    return {k:np.asarray(v) for k,v in bank.items()}

# row: layout, task wiring, nuisance wiring, goal lane, target moves?, automatic carriage?,
#      external signed drive, resting side, previous issued drive, reset+repair, issued drive (TRAIN only).
def episodes(bank,first,teach=False):
    rng=np.random.default_rng(first+187); out=[]
    for i,marks in enumerate(bank["observed_move"]):
        for j in (0,1):
            for k in (0,1):
                for lane in range(3):
                    home=(-1,1)[(i//2+lane)%2]
                    for want in (0,1):
                        # Demonstrator reads the visible response at the instructed location, not a hidden link or candidate score.
                        a=-1 if bool(marks[j,lane])==bool(want) else 1
                        for auto in (0,1):
                            ext=a if auto else 0
                            row=[i,j,k,lane,want,auto,ext,home,0,0]
                            if not teach:out.append(row);continue
                            issue=0 if auto else a
                            lapse=not auto and rng.random()<.08
                            if lapse:issue=-a
                            out.append(row+[issue])
                            if lapse:out.append(row[:8]+[issue,1,a])
    return np.asarray(out,np.int32)

def context(bank,r):
    i,j,k,lane,w,auto,ext,home,old,repair=r[:,:10].T
    return np.concatenate((bank["base"][i],bank["past"][i,j],bank["past"][i,k],
        np.eye(3,dtype=np.float32)[lane],np.column_stack((w*2-1,auto,ext,home,old,repair))),1).astype(np.float32)
def target_cam(bank,r,drive):
    # 只给记录、独立物理干预或封卷后的裁判使用。原始观察与未来数组分离。
    i,j,k=r[:,:3].T;a=np.asarray(drive,dtype=int)+1
    return np.concatenate((bank["futures"][i,j,a],bank["futures"][i,k,a]),axis=1)
def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    records={}
    for key in ("train","validation"):
        first,n=CFG[key];b=apparatus(first,n,False,True);r=episodes(b,first,True)
        save(out/(key+".npz"),**b,rows=r)
        records[key]=dict(layouts=n,episodes=len(r),corrections=int(r[:,9].sum()),file=sha(out/(key+".npz")))
    dump(out/"manifest.json",dict(protocol=sha(PROTO),records=records))

def bits(x):return ((np.asarray(x)[...,None]>>np.arange(3))&1).astype(np.int64)
def index(b):return (b*np.array([1,2,4])).sum(axis=-1)
def programs(rng,n,depth):
    src=rng.integers(3,size=(n,depth));dst=(src+rng.integers(1,3,size=(n,depth)))%3
    return np.stack([src,dst,rng.integers(2,size=(n,depth))],axis=-1)
def circuit(initial,cards):
    # Physical rule: a switch changes target iff source voltage matches the gate polarity.
    x=initial.copy(); ix=np.arange(len(x));trace=[]
    for src,dst,pol in cards.transpose(1,2,0):
        x[ix,dst]^=(x[ix,src]==pol);trace.append(x.copy())
    return np.stack(trace,1)
def training_programs(rng,n,depth):
    init=bits(rng.integers(8,size=n));cards=programs(rng,n,depth)
    return init,cards,rng.integers(3,size=n),circuit(init,cards)
def test_programs(depth):
    rng=np.random.default_rng(CFG["lang_exam_seed"]+depth);n=CFG["language_pairs_per_depth"]
    init=bits(rng.integers(8,size=n));cards=programs(rng,n,depth); twin=cards.copy();twin[:,0,2]^=1
    a,b=circuit(init,cards),circuit(init,twin)
    query=np.array([rng.choice(np.where(x!=y)[0]) for x,y in zip(a[:,-1],b[:,-1])])
    return (np.repeat(init,2,0),np.stack([cards,twin],1).reshape(n*2,depth,3),np.repeat(query,2),
            np.stack([a,b],1).reshape(n*2,depth,3))
