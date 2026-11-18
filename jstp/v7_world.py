"""V7 独立窗口的虚拟手和物理台。网络只能读 RGB 和公开的鼠标命令。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

W, H, TABLE, D = 96, 64, 43, 6
BG = np.array([16, 23, 34], np.uint8)
PALETTE = np.array([[233,145,74],[90,205,170],[164,145,230],[93,183,229],[224,129,162],[210,197,106]])
SPLITS = {"direct_train": (1_000_000,4096), "direct_val": (2_000_000,192),
          "pairs_train": (3_000_000,3072), "pairs_val": (4_000_000,96),
          "direct_test": (5_000_000,192), "pairs_test": (6_000_000,128)}


@dataclass
class Board:
    pieces: np.ndarray
    grips: np.ndarray
    textures: tuple
    pads: tuple

    @classmethod
    def make(cls, seed):
        r = np.random.default_rng(seed)
        points = np.array([[r.integers(17,33), r.integers(15,30)],
                           [r.integers(64,80), r.integers(15,30)]], dtype=int)
        grips = np.array([[r.integers(17,33),52],[r.integers(64,80),52]], dtype=int)
        def tile():
            h,w = int(r.integers(8,14)),int(r.integers(9,15))
            base = PALETTE[int(r.integers(len(PALETTE)))] + r.integers(-12,13,3)
            yy,xx = np.mgrid[:h,:w]
            period,tilt,offset = r.integers(3,7),r.integers(0,3),r.integers(0,7)
            grain = np.where((xx+tilt*yy+offset)%period == 0, -29, 0)
            color = np.clip(base + grain[...,None],0,255).astype(np.uint8)
            color[[0,-1],:] //= 2; color[:,[0,-1]] //= 2
            color[[0,0,-1,-1],[0,-1,0,-1]] = BG
            return color
        def pad(k):
            a = np.empty((5,9,3),np.uint8); a[:] = (103,129,152)
            a[0,1:-1] = (177,198,211); a[-1,1:-1] = (76,98,123)
            for x in ((4,) if k==0 else (3,5)): a[1:-1,x]=(238,226,181)
            a[[0,0,-1,-1],[0,-1,0,-1]] = BG
            return a
        return cls(points,grips,(tile(),tile()),(pad(0),pad(1)))

    def render(self, pieces, grips):
        img = np.empty((H,W,3),np.uint8); img[:] = BG
        for centers,textures in ((pieces,self.textures),(grips,self.pads)):
            for (x,y),a in zip(centers,textures):
                h,w = a.shape[:2]; l,t = int(x)-w//2,int(y)-h//2
                if l<0 or t<0 or l+w>W or t+h>H:
                    raise ValueError("物体移出相机范围")
                img[t:t+h,l:l+w] = a
        return img

    def goals(self,dy):
        # 任务方作画，不知道隐藏挂接，不调用 World，不模拟任何候选把手。
        result=[]
        for both in (False,True):
            p=self.pieces.copy(); p[0,1]+=dy
            if both: p[1,1]+=dy
            result.append(self.render(p,self.grips)[:TABLE])
        return np.stack(result)


class World:
    def __init__(self, board, joined_grip):
        self.board, self._joined = board, joined_grip
        self.p, self.g = board.pieces.copy(),board.grips.copy()
        self._held = None

    def camera(self): return self.board.render(self.p,self.g)

    def press(self, xy):
        if self._held is not None: raise RuntimeError("鼠标已按住")
        self._held = ((),())
        for i,(center,a) in enumerate(zip(self.p,self.board.textures)):
            l,t=center-np.array(a.shape[1::-1])//2
            if l<=xy[0]<l+a.shape[1] and t<=xy[1]<t+a.shape[0]:
                self._held=((i,),()); return
        for i,c in enumerate(self.g):
            if np.all(np.abs(c-xy)<=[4,2]):
                self._held=(((0,1) if i==self._joined else (0,)),(i,)); return

    def move(self, delta):
        if self._held is None: raise RuntimeError("未按住不能拖动")
        for i in self._held[0]: self.p[i]+=delta
        for i in self._held[1]: self.g[i]+=delta
        return self.camera()

    def release(self):
        if self._held is None: raise RuntimeError("鼠标未按住")
        self._held=None

    def probe(self, cmd):
        start=self.camera(); self.press(cmd[:2])
        peak=self.move(cmd[2:]); end=self.move(-cmd[2:]); self.release()
        if not np.array_equal(start,end): raise RuntimeError("试探没有精确回到原位")
        return start,peak,end

    def execute(self, cmd):
        self.press(cmd[:2]); result=self.move(cmd[2:]); self.release()
        return result


def setup(seed):
    b=Board.make(seed)
    # 每连续八个布局均衡涵盖试探把手、试探方向和任务方向，外观另由 seed 随机采样。
    k=seed%2; horizontal=D*(1 if (seed//2)%2 else -1); dy=D*(1 if (seed//4)%2 else -1)
    probe=np.array([*b.grips[k],horizontal,0])
    cmds=np.array([[*g,0,dy] for g in b.grips])
    return b,probe,cmds,dy


def direct_setup(seed):
    b=Board.make(seed); r=np.random.default_rng(seed+71_000_000)
    side=seed%2; coord=b.pieces[side]+r.integers(-2,3,2)
    dx,dy=((D,0),(-D,0),(0,D),(0,-D))[(seed//2)%4]
    return b,np.array([*coord,dx,dy])


def direct_record(seed):
    b,cmd=direct_setup(seed); w=World(b,seed%2)
    coord=cmd[:2]
    hold=w.execute(np.array([*coord,0,0])); cur=w.camera()
    return dict(current=cur,cmd=cmd.astype(np.float32),future=w.execute(cmd),hold=hold)


def pair_record(seed):
    b,probe,cmds,dy=setup(seed); peaks=[]; futures=[]
    for joint in (0,1):
        w=World(b,joint); _,peak,current=w.probe(probe); peaks.append(peak)
        branches=[]
        for cmd in cmds:
            w=World(b,joint); w.probe(probe); branches.append(w.execute(cmd))
        futures.append(branches)
    return dict(current=current,peak=np.stack(peaks),probe=probe.astype(np.float32),
                cmd=cmds.astype(np.float32),goal=b.goals(dy),future=np.asarray(futures))


def collect(split, path:Path, tick=None):
    first,n=SPLITS[split]; rec=direct_record if split.startswith("direct") else pair_record
    arrays={}
    for i in range(n):
        row=rec(first+i)
        if not arrays:
            arrays={k:np.empty((n,*v.shape),v.dtype) for k,v in row.items()}
        for k,v in row.items(): arrays[k][i]=v
        if tick and i%128==0: tick(i,n,row)
    # 唯一运行目录、每个采集文件只提交一次。监控窗口不持有待替换的文件句柄。
    with path.open("xb") as f: np.savez_compressed(f,**arrays)
    return arrays
