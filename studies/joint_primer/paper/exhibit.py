"""Make the paper's real-camera strip and a standalone viewer of sealed trials and training traces."""
import base64,io,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mechanism.data import read
from lategoal.rig import Rig
from lategoal.control import fingerprint
from studies.joint_primer.train import slots
from studies.joint_primer.exam import jread
from studies.joint_primer.official_train import runs as official_runs
D=Path(__file__).resolve().parent;E=D.parent/'evidence'
def img(v):return Image.fromarray(v)
def b64(raw):return base64.b64encode(raw).decode('ascii')
def png(v):
    out=io.BytesIO();img(v).save(out,'PNG');return 'data:image/png;base64,'+b64(out.getvalue())
def pack(x):return b64(np.packbits(x.reshape(-1)).tobytes())
def compact(v):return np.rint(v*100).astype('int16').reshape(-1).tolist()
def main():
    items=read(E/'sealed/items.npz');cam=read(E/'observed/camera.npz')
    layouts=[];n=8
    for i,seed in enumerate(items['ids'][:n]):
        r=Rig(int(seed));wits=[];fin=[]
        if fingerprint(r.base)!=items['base'][i]:raise ValueError('base image mismatch')
        for s in (0,1):
            for p in (-1,1):
                peak=r.world(s).probe(p)
                if fingerprint(peak)!=items['peak'][i,int(p>0),s]:raise ValueError('probe mismatch')
                wits.append(png(peak))
            for g in (0,1):
                for a in (-1,1):
                    tri=r.world(s);tri.probe(-1);tri.set_clutch(g);frame=tri.stroke(a)
                    if fingerprint(frame)!=cam['frame_sha'][i,s,g,int(a>0)]:raise ValueError('actual frame mismatch')
                    fin.append(png(frame))
        layouts.append(dict(seed=int(seed),base=png(r.base),peak=wits,final=fin,left=cam['left'][i].tolist()))
        if i==0:
            tri=r.world(0);peak=tri.probe(-1);reset=tri.camera();post=[]
            for a in (-1,1):
                t=r.world(0);t.probe(-1);t.set_clutch(0);post.append(t.stroke(a))
            frames=[r.base,peak,reset]+post
            captions=['01  BEFORE','02  PROBE A','03  RESET','04a  STROKE A','04b  STROKE B']
            band=Image.new('RGB',(5*202+12,169),'#edf3f1');dr=ImageDraw.Draw(band)
            font=ImageFont.load_default(size=16)
            for k,(f,c) in enumerate(zip(frames,captions)):
                off=k*202+8;dr.text((off,7),c,fill='#175448',font=font)
                pic=img(f).resize((192,128),Image.Resampling.NEAREST)
                band.paste(pic,(off,32))
            (D/'figures').mkdir(exist_ok=True);band.save(D/'figures/one-history.png')
    policies={}
    from_official=E/'official'
    if not jread(from_official/'after.json')['camera_matches_primary']:raise ValueError('actual view cannot be shared')
    for arm,k,seed,name in slots()+official_runs():
        official=arm in ('smol','dino');root=from_official if official else E
        b=read(root/'sealed'/(name+'.npz'));u=b['u'][:n];win=read(root/'observed'/(name+'.npz'))['wins'][:n]
        if official:u=u[...,None];win=win[...,None]
        curves=(root if official else E/'train')/'models'/name/'curve.jsonl'
        curve=[json.loads(line) for line in curves.read_text(encoding='utf8').splitlines()]
        p=dict(u=pack(u>0),w=pack(win),variants=win.shape[-1],curve=curve)
        if 'eye' in b:p['eye']=compact(b['eye'][:n,...,0,:])
        if 'patched_eye' in b:p['patch']=compact(b['patched_eye'][:n])
        policies[name]=p
    doc=(D/'viewer-template.html').read_text(encoding='utf8')
    payload=dict(layouts=layouts,goals=items['goals'].tolist(),policies=policies)
    (D/'evidence-room.html').write_text(doc.replace('/*__RECORDS__*/null',json.dumps(payload,ensure_ascii=False,separators=(',',':'))),encoding='utf8',newline='\n')
    print('exhibit:',len(layouts),'layouts,',len(policies),'frozen policies')
if __name__=='__main__':main()
