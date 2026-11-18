"""可见像素规则：不用真值坐标/连接。三带位置、平移、互补双驱是明确给予的先验。"""
from __future__ import annotations
import numpy as np
from .v7_world import TABLE,W

def render_move(c,bits,cmd):
    # 每条固定观察带宽 32；物体和两把手在规定行程内不跨区域。
    dy=int(cmd[3]); a=c.copy()
    for lane,yes in enumerate(bits):
        if yes:
            band=slice(32*lane,32*lane+32)
            a[:TABLE,band]=np.roll(c[:TABLE,band],dy,axis=0)
    pad=slice(0,48) if cmd[0]<48 else slice(48,96)
    a[TABLE:,pad]=np.roll(c[TABLE:,pad],dy,axis=0)
    return a

def rule(c,e,p,a,key):
    result=[]
    for cur,peaks,probe,cmds in zip(c,e,p,a):
        pair=[]
        for peak in peaks:
            # 信号被删的对照不应被当作真实的三路均不动；按均衡先验边缘化未知接线。
            support=[tuple(bool(j&(1<<i)) for i in range(3)) for j in range(1,7)]
            if key=="one_bit":
                bit=not np.array_equal(peak[:TABLE,64:],cur[:TABLE,64:])
                support=[s for s in support if s[2]==bit]
            elif key=="visible_rule":
                seen=tuple(not np.array_equal(peak[:TABLE,32*i:32*i+32],cur[:TABLE,32*i:32*i+32]) for i in range(3))
                if any(seen):support=[s for s in support if seen==s]
            else:raise KeyError(key)
            fut=[]
            for opt in cmds:
                same=probe[0]==opt[0]
                fut.append(np.mean([render_move(cur,s if same else [not x for x in s],opt) for s in support],axis=0,dtype=np.float32)/255.)
            pair.append(fut)
        result.append(pair)
    return np.asarray(result)
