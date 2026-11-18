"""本地四载荷试验台。控制器只能读取公开相机、仪表和自身已发出的驱动。"""
from pathlib import Path
from itertools import product,combinations
import json
import numpy as np
from jstp.v7_world import Board,BG
from mechanism.data import sha,dump,save,read

ROOT=Path(__file__).resolve().parents[1]
PROTO=Path(__file__).with_name("protocol.json")
C=json.loads(PROTO.read_text(encoding="utf8"))
REL=ROOT/"evidence/late-goals"
PAIRS=list(combinations(range(4),2))
XX=np.arange(96,dtype=np.float32)
YY=np.arange(44,dtype=np.float32)
K=np.exp(-.5*((np.arange(4,96,8,dtype=np.float32)[:,None]-XX)/8.5)**2)
F=72
HIST=2*F+12+1+4+1

def cards(supports):
    result=[]
    for support in supports:
        for signs in product((-1,1),repeat=len(support)-1):
            x=np.zeros(4,np.int8);x[list(support)]=(1,*signs)
            result.extend([x,-x])
    return np.stack(result)
TRAIN=cards([[i] for i in C["targets"]["single"]]+C["targets"]["train_triples"])
EXAM=cards(C["targets"]["exam_triples"])
if set(map(tuple,TRAIN))&set(map(tuple,EXAM)):raise ValueError("训练与考试目标相交")

class Rig:
    def __init__(self,seed):
        rng=np.random.default_rng(seed);a=Board.make(seed);b=Board.make(seed+210000000)
        centers=np.column_stack(([12,36,60,84]+rng.integers(-1,2,4),rng.integers(16,29,4)))
        self.board=Board(centers,a.grips,a.textures+b.textures,a.pads)
        self._links=rng.choice([-1,1],4);self.mark=np.zeros(4,np.int8)
        self.mark[list(PAIRS[int(rng.integers(6))])]=1
        self.home=int(rng.choice([-1,1]));self.base=self.board.render(centers,self.board.grips)
        self.seed=seed
    def world(self,wiring):return Trial(self,wiring)

class Trial:
    def __init__(self,rig,wiring):
        self.r=rig;self.p=rig.board.pieces.copy();self.hand=rig.board.grips.copy()
        self._links=rig._links*np.where(wiring&rig.mark,-1,1);self.probed=False;self.gear=None
    def camera(self):return self.r.board.render(self.p,self.hand)
    def probe(self,side):
        if self.probed or self.gear is not None or side not in (-1,1):raise ValueError("探查时序或把手错误")
        k=int(side>0);dx=-side*4*self._links
        self.p[:,0]+=dx;self.hand[k,0]+=4;peak=self.camera()
        self.p[:,0]-=dx;self.hand[k,0]-=4;self.probed=True
        if not np.array_equal(self.camera(),self.r.base):raise ValueError("横向试探未精确复位")
        return peak
    def set_clutch(self,gear):
        if not self.probed or self.gear is not None or gear not in (0,1):raise ValueError("离合设置时序错误")
        self.gear=gear
    def stroke(self,side):
        p,h=self.stroke_copies(np.array([side]));self.p=p[0];self.hand=h[0]
        return self.camera()
    def stroke_copies(self,sides):
        """从这一复位态分出等价的实物副本，每个已提交指令分别推进位置和把手。"""
        sides=np.asarray(sides)
        if self.gear is None or sides.ndim!=1 or not np.isin(sides,[-1,1]).all():raise ValueError("正式驱动或时序错误")
        p=np.repeat(self.p[None],len(sides),0);h=np.repeat(self.hand[None],len(sides),0)
        p[:,:,1]+=-sides[:,None]*4*self._links*np.where(self.gear&self.r.mark,-1,1)
        h[np.arange(len(sides)),(sides>0).astype(int),1]+=4;self.gear=None
        return p,h

def brightness(im):
    return ((im[:48].astype(np.float32)-BG.astype(np.float32)).mean(-1)).clip(min=0)
def small(im):
    return brightness(im).reshape(6,8,12,8).mean((1,3)).ravel()/150
def optics(im):
    f=brightness(im)[:44];mass=K@f.sum(0);yc=K@(YY@f)
    xc=K@(f.sum(0)*XX)
    return np.stack((xc,yc))/np.maximum(mass,1e-6)
def optical_delta(base,im):return (optics(im)-optics(base))/4
def lane_travel(base,im):
    # 独立的任务裁判：只从实际图像测重心，不取机构挂接或预测器输出。
    def cy(x):
        f=brightness(x)[:44].reshape(44,4,24).sum(2)
        return (YY@f)/f.sum(0)
    dy=cy(im)-cy(base)
    if not np.allclose(np.abs(dy),4,atol=.002):raise ValueError("裁判未看见四个完整的行程")
    return np.sign(dy).astype(np.int8)
def first(r):return np.r_[small(r.base),np.float32(r.home)].astype(np.float32)
def history(r,peak,side,gear):
    return np.concatenate((small(r.base),small(peak)-small(r.base),optical_delta(r.base,peak)[0],
                           [side],r.mark,[2*gear-1])).astype(np.float32)
def experience(seed):
    r=Rig(seed);x=np.empty((2,2,2,HIST),np.float32)
    y=np.empty((2,2,2,12),np.float32);motion=np.empty((2,2,2,4),np.int8)
    for s,g in product(range(2),repeat=2):
        for pi,p in enumerate((-1,1)):
            w=r.world(s);peak=w.probe(p);w.set_clutch(g);x[s,g,pi]=history(r,peak,p,g)
            frame=w.stroke(p);y[s,g,pi]=optical_delta(r.base,frame)[1]
            motion[s,g,pi]=lane_travel(r.base,frame)
    return first(r),x,y,motion
def bank(split):
    start,n=C[split];rows=list(map(experience,range(start,start+n)))
    return dict(zip(("start","hist","eye","motion"),map(np.stack,zip(*rows))))
def goal_wins(velocity,goals):
    return (np.asarray(velocity)*np.asarray(goals)).sum(-1)>0
