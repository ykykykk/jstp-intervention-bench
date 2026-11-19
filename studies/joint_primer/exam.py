"""Freeze -> calibrate on validation only -> seal all intentions -> operate the independent rig."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from mechanism.data import sha,save,read,dump
from lategoal.rig import Rig,EXAM,F,goal_wins
from lategoal.control import T,axes,fingerprint
from lategoal.exam import actual
from studies.partial_futures.exam import prehistory,ridge
from .train import C,DATA,PROTO,HERE,ROOT,SUP,slots,load

def hashes(paths,root):return {f.relative_to(root).as_posix():sha(f) for f in paths}
def ensure(root,rows):
    for name,expected in rows.items():
        if sha(root/name)!=expected:raise ValueError('changed evidence: '+name)
def jread(p):return json.loads(p.read_text(encoding='utf8'))
def quantile_sign(x):return np.where(x>=0,1,-1).astype(np.int8)
@torch.inference_mode()
def seal(out):
    tr=out/'train';frozen=jread(tr/'frozen.json')
    ensure(tr,{'schedule.npz':frozen['schedule'],'design-lock.json':frozen['design']})
    ensure(ROOT,jread(tr/'design-lock.json')['sources'])
    if sha(DATA/'validation.npz')!=C['validation_sha256']:raise ValueError('validation changed')
    v=read(DATA/'validation.npz');vx=T(v['hist'].reshape(-1,v['hist'].shape[-1]))
    vy=np.repeat(v['motion'][:,:,:,0:1,:],2,axis=3).reshape(-1,4)
    ids=np.arange(C['exam'][0],sum(C['exam']),dtype='int32')
    rigs=[Rig(int(i)) for i in ids];_,states,peaks=prehistory(rigs)
    n=len(rigs);shape=(n,2,2,2)  # layout, wiring, clutch, probe side
    x=T(states.transpose(0,2,1,3).reshape(n*8,-1));blind=x.clone();blind[:,F:2*F+12]=0
    dest=out/'sealed';dest.mkdir(exist_ok=False);meta=dest/'items.npz'
    save(meta,ids=ids,goals=EXAM,base=np.asarray([fingerprint(r.base) for r in rigs]),peak=peaks)
    paths=[meta,tr/'frozen.json']; G=T(EXAM)
    for arm,k,seed,name in slots():
        ensure(tr/'models'/name,{'freeze.json':frozen['models'][name]})
        m=load(tr,arm,k,seed);vh=m.representation(m.observe(vx));w=ridge(vh.numpy(),vy)
        h=m.observe(x);z=m.representation(h)
        decode=lambda x:quantile_sign(x.numpy()@w['rw']+w['rb']).reshape(*shape,4)
        record=dict(group=decode(z),blind_group=decode(m.representation(m.observe(blind))),**w)
        u0=torch.stack([m.score(h,g.expand(len(x),-1)) for g in G],1)
        choices=[quantile_sign(u0.numpy()).reshape(*shape,len(G))]
        if arm in SUP:
            zz=[m.future(h,torch.full((len(h),),side,dtype=torch.long)) for side in (0,1)]
            record['eye']=torch.stack([m.eye(vv) for vv in zz],1).numpy().reshape(*shape,2,12)
        if arm=='joint':
            ax=axes(v['eye'][:,:,:,0],m);record.update(ax)
            q,nul=T(ax['Q']),T(ax['N']);donor=z.reshape(*shape,-1).flip(2).reshape_as(z)
            coeff=(donor-z)@q
            physical=z+coeff@q.T;irrelevant=z+coeff@nul.T
            record['norm_gap']=np.float32(((physical-z).norm(dim=1)-(irrelevant-z).norm(dim=1)).abs().max())
            record['patched_eye']=m.eye(physical).numpy().reshape(*shape,12)
            for replacement in (physical,irrelevant):
                logits=m.motor(h,replacement)@G.T
                choices.append(quantile_sign(logits.numpy()).reshape(*shape,len(G)))
        target=dest/(name+'.npz');save(target,**record,u=np.stack(choices,axis=-1));paths.append(target)
    dump(out/'before.json',dict(protocol=sha(PROTO),files=hashes(paths,out),exam_code=sha(Path(__file__)),
         dependencies=hashes([ROOT/'lategoal/rig.py',ROOT/'lategoal/control.py',ROOT/'lategoal/exam.py',
         ROOT/'studies/partial_futures/exam.py',ROOT/'jstp/v7_world.py'],ROOT),sealed_ns=time.time_ns()))
def execute(out):
    before=jread(out/'before.json');ensure(out,before['files']);ensure(ROOT,before['dependencies'])
    if before['protocol']!=sha(PROTO) or before['exam_code']!=sha(Path(__file__)):raise ValueError('locked exam changed')
    dest=out/'observed';dest.mkdir(exist_ok=False);src=out/'sealed';items=read(src/'items.npz')
    goals=items['goals'];ids=items['ids'];n=len(ids)
    policies={name:read(src/(name+'.npz'))['u'] for *_,name in slots()}
    wins={name:np.empty(u.shape,dtype=np.bool_) for name,u in policies.items()}
    left=np.empty((n,2,2,4),'int8');optics=np.empty((n,2,2,2,12),'float32')
    witness=np.empty((n,2,2,2),dtype='S64')
    for i,layout in enumerate(ids):
        rig=Rig(int(layout))
        if fingerprint(rig.base)!=items['base'][i]:raise ValueError('initial camera mismatch')
        for s in range(2):
            for g in range(2):
                for pi,p in enumerate((-1,1)):
                    rows=[u[i,s,g,pi].ravel() for u in policies.values()]
                    vel,ref,eye,hs=actual(rig,s,g,p,np.concatenate(rows),items['peak'][i,pi,s])
                    if pi and (not np.array_equal(left[i,s,g],ref) or not np.array_equal(witness[i,s,g],hs)):
                        raise ValueError('probe did not reset to physically equivalent state')
                    left[i,s,g]=ref;optics[i,s,g,pi]=eye;witness[i,s,g]=hs
                    off=0
                    for (name,u),row in zip(policies.items(),rows):
                        sz=len(row);v=vel[off:off+sz].reshape(*u.shape[-2:],4)
                        wins[name][i,s,g,pi]=goal_wins(v,goals[:,None]);off+=sz
    camera=dest/'camera.npz';save(camera,left=left,eye=optics,frame_sha=witness)
    paths=[camera,out/'before.json']
    for name,a in wins.items():
        f=dest/(name+'.npz');save(f,wins=a);paths.append(f)
    dump(out/'after.json',dict(protocol=sha(PROTO),files=hashes(paths,out),executed_ns=time.time_ns()))
    print('CAMERA',n,'new layouts;',len(wins),'frozen policies',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(2);seal(a.out);execute(a.out)
