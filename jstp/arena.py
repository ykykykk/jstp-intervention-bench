"""三件双驱实验台。隐藏接线仅供世界执行器使用，控制模型的签名不含接线。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .v7_world import Board,World,H,W,D,TABLE,BG
from .common import config,save_npz,read_npz,sha,new_json,PROTOCOL

@dataclass
class Scene:
    board:Board
    singleton:int
    probe:np.ndarray
    options:np.ndarray
    target:int
    other:int
    goal_flip:bool

    def goals(self,compound):
        # 先指定可达到的目标，后交换 A/B 的挂接；任务方从不询问哪个把手正确。
        lanes=[self.target,self.other] if compound else [self.target]
        mask=np.zeros((TABLE,W),np.uint8)
        for lane in lanes: mask[:,32*lane:32*(lane+1)]=255
        ims=[]
        for which in (0,1):
            pos=self.board.pieces.copy()
            for lane in lanes:
                if compound:
                    moved=(lane==self.singleton) ^ bool(which) ^ self.goal_flip
                else: moved=bool(which)^self.goal_flip
                if moved: pos[lane,1]+=self.options[0,3]
            ims.append(self.board.render(pos,self.board.grips)[:TABLE])
        return np.stack(ims),mask

    def world(self,permutation): return CoupledWorld(self.board,self.singleton,permutation)

class CoupledWorld(World):
    def __init__(self,board,singleton,permutation):
        super().__init__(board,permutation)
        self._singleton=singleton
    def press(self,xy):
        if self._held is not None: raise RuntimeError("重复按下")
        for i,c in enumerate(self.g):
            if np.all(np.abs(c-xy)<=[4,2]):
                group=tuple(j for j in range(3) if ((j==self._singleton)==(i==self._joined)))
                self._held=(group,(i,)); return
        super().press(xy)

def scene(seed,heldout=False):
    rng=np.random.default_rng(seed+912_000_000); base=Board.make(seed); extra=Board.make(seed+430_000_000)
    p=np.column_stack([np.array([16,48,80])+rng.integers(-2,3,3),rng.integers(14,30,3)])
    b=Board(p,base.grips,base.textures+(extra.textures[0],),base.pads)
    k=seed%2; dx=D*(1 if (seed//2)%2 else -1); dy=D*(1 if (seed//4)%2 else -1)
    m=1 if heldout else (0,2)[(seed//8)%2]
    lane=int(rng.integers(3)); other=(lane+int(rng.integers(1,3)))%3
    return Scene(b,m,np.array([*b.grips[k],dx,0]),np.array([[*c,0,dy] for c in b.grips]),lane,other,bool(rng.integers(2)))

def sense(s):
    h=[s.world(j).probe(s.probe) for j in range(2)]
    if not np.array_equal(h[0][2],h[1][2]) or np.array_equal(h[0][1],h[1][1]):
        raise ValueError("未满足同末帧、异证据的配对条件")
    return h[0][2],np.stack([q[1] for q in h])

def execute(s,j,cmd):
    w=s.world(j); w.probe(s.probe); return w.execute(cmd)

def direct(seed):
    s=scene(seed); r=np.random.default_rng(seed+710_000_000)
    lane=(seed//4)%3; d=((D,0),(-D,0),(0,D),(0,-D))[seed%4]
    a=np.array([*(s.board.pieces[lane]+r.integers(-2,3,2)),*d]); w=s.world(0)
    return dict(c=w.camera(),a=a.astype(np.float32),f=w.execute(a))

def pair(seed):
    s=scene(seed); c,e=sense(s); g,mask=s.goals(False)
    f=np.asarray([[execute(s,j,a) for a in s.options] for j in (0,1)])
    return dict(c=c,e=e,p=s.probe.astype(np.float32),a=s.options.astype(np.float32),f=f,g=g,mask=mask)

def collect(root):
    root.mkdir(parents=True,exist_ok=True); manifest={}
    for split in ("direct_train","train","val"):
        first,n=config()["splits"][split]; fn=direct if split=="direct_train" else pair
        path=root/(split+".npz")
        if path.exists(): raise FileExistsError(f"数据集已存在，不覆盖：{path}")
        d={}
        for i in range(n):
            row=fn(first+i)
            if not d: d={k:np.empty((n,*v.shape),v.dtype) for k,v in row.items()}
            for k,v in row.items(): d[k][i]=v
        save_npz(path,**d); manifest[split]={"sha256":sha(path),"n":n}
        print("DATA",split,n,flush=True)
    new_json(root/"manifest.json",{"schema":sha(PROTOCOL),"data":manifest})
