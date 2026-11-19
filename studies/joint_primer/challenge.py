"""Post-main robustness check: prohibit goal-label gradients from corrupting modelled mechanics."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from mechanism.data import sha,save,read,dump
from mechanism.train import config
from lategoal.rig import Rig,EXAM,goal_wins
from lategoal.control import T,fingerprint
from lategoal.exam import actual
from studies.partial_futures.exam import prehistory
from .train import C,HERE,DATA,COMB,SUP,annotations,slots,load,tensor,make
from .exam import jread,ensure,hashes,quantile_sign
from .analyze import counts,unpack,interval

PLAN=HERE/'challenge-plan.json';P=jread(PLAN);NEW='world-isolated'
def newslots():return [(NEW,k,s,f'{NEW}-{k}-{s}') for k in P['label_budgets'] for s in P['seeds']]
def fit(out,dest,device):
    orig=out/'train';lock=jread(orig/'design-lock.json')
    if sha(orig/'schedule.npz')!=lock['schedule'] or sha(DATA/'train.npz')!=lock['train']:raise ValueError('not the main training data')
    plan=read(orig/'schedule.npz');bank=read(DATA/'train.npz');x=T(bank['hist'].reshape(-1,bank['hist'].shape[-1])).to(device)
    eye=T(bank['eye'].reshape(-1,12)).to(device)
    phys=tensor(plan['physical'].astype('int64'),device);side=tensor(plan['side'].astype('int64'),device)
    visible=tensor(COMB[plan['sensors']],device)
    for _,k,s,name in newslots():
        d=dest/'models'/name;d.mkdir(parents=True,exist_ok=False);config(s,device);m=make('world',C['width']).to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=C['lr'],weight_decay=1e-4)
        select=plan['perm'][:k].astype(np.int64);card,y=annotations(bank,select,plan['cards'],device)
        sel=tensor(select,device);perm=tensor((plan['label']%k).astype('int64'),device);t=time.monotonic()
        with (d/'curve.jsonl').open('x',encoding='utf8') as log:
            for step,(li,ri,si,mask) in enumerate(zip(perm,phys,side,visible),1):
                with torch.no_grad():
                    h=m.observe(x[sel[li]])
                    a=m.eye(m.step(h,m.events.weight[0]));b=m.eye(m.step(h,m.events.weight[1]))
                score=((m.value(b)-m.value(a))*card[li]).sum(-1)
                err=m.eye(m.future(m.observe(x[ri]),si)).gather(1,mask)-eye[(ri//2)*2+si].gather(1,mask)
                loss=F.softplus(-y[li]*score).mean()+C['optical_weight']*err.square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step()
                if step%128==0:
                    value=float(loss.detach())
                    if not np.isfinite(value):raise FloatingPointError((k,s,step))
                    log.write(json.dumps(dict(update=step,loss=value,seconds=round(time.monotonic()-t,3)))+'\n')
        with (d/'last.pt').open('xb') as f:torch.save(m.cpu().state_dict(),f)
        dump(d/'freeze.json',dict(seed=s,budget=k,weights=sha(d/'last.pt'),schedule=lock['schedule'],plan=sha(PLAN),
             code=sha(Path(__file__)),architecture=sha(HERE/'models.py'),updates=step,seconds=round(time.monotonic()-t,3),
             device=torch.cuda.get_device_name(device) if device.startswith('cuda') else device))
        print('FROZEN',name,flush=True)
def getmodel(out,d,arm,k,s,name):
    if arm!=NEW:return load(out/'train',arm,k,s)
    fold=d/'models'/name;meta=jread(fold/'freeze.json')
    ensure(fold,{'last.pt':meta['weights']})
    if meta['plan']!=sha(PLAN) or meta['architecture']!=sha(HERE/'models.py') or meta['code']!=sha(Path(__file__)):
        raise ValueError('changed supplement')
    m=make('world',C['width']).eval();m.load_state_dict(torch.load(fold/'last.pt',weights_only=True,map_location='cpu'));return m
@torch.inference_mode()
def exam(out,dest):
    runs=slots()+newslots();d=dest/'sealed';d.mkdir()
    ids=np.arange(P['exam'][0],sum(P['exam']),dtype='int32');n=len(ids);rigs=[Rig(int(i)) for i in ids]
    _,states,peaks=prehistory(rigs);x=T(states.transpose(0,2,1,3).reshape(n*8,-1));goals=T(EXAM)
    mfile=d/'items.npz';save(mfile,ids=ids,peak=peaks,goals=EXAM,base=np.asarray([fingerprint(r.base) for r in rigs]))
    paths=[mfile]
    for arm,k,s,name in runs:
        m=getmodel(out,dest,arm,k,s,name);h=m.observe(x);us=torch.stack([m.score(h,g.expand(len(x),-1)) for g in goals],1)
        row=dict(u=quantile_sign(us.numpy()).reshape(n,2,2,2,-1))
        if arm in SUP or arm==NEW:
            preds=[m.eye(m.future(h,torch.full((len(h),),p,dtype=torch.long))).numpy() for p in (0,1)]
            row['eye']=np.stack(preds,1).reshape(n,2,2,2,2,12)
        f=d/(name+'.npz');save(f,**row);paths.append(f)
    newmeta=[dest/'models'/name/'freeze.json' for *_,name in newslots()]
    dump(dest/'before.json',dict(plan=sha(PLAN),code=sha(Path(__file__)),main_freeze=sha(out/'train/frozen.json'),
         files=hashes(paths+newmeta,dest),sealed_ns=time.time_ns()))
    ensure(dest,jread(dest/'before.json')['files'])
    rows={name:read(d/(name+'.npz')) for *_,name in runs};wins={name:np.empty(row['u'].shape,bool) for name,row in rows.items()}
    optic=np.empty((n,2,2,2,12),'float32'); frames=np.empty((n,2,2,2),dtype='S64')
    for i,r in enumerate(rigs):
        for j in (0,1):
            for g in (0,1):
                for pi,p in enumerate((-1,1)):
                    us=np.stack([b['u'][i,j,g,pi] for b in rows.values()])
                    v,_,e,hs=actual(r,j,g,p,us,peaks[i,pi,j]);optic[i,j,g,pi]=e
                    if pi and not np.array_equal(frames[i,j,g],hs):raise ValueError('reset changed physics')
                    frames[i,j,g]=hs
                    scored=goal_wins(v,EXAM)
                    for hit,vv in zip(wins.values(),scored):hit[i,j,g,pi]=vv
    res=dest/'observed';res.mkdir();cam=res/'camera.npz';save(cam,eye=optic,frame_sha=frames)
    files=[cam,dest/'before.json'];layout={};per=[]
    for arm,k,s,name in runs:
        w=wins[name];row=rows[name];f=res/(name+'.npz');save(f,wins=w);files.append(f)
        ct={'pair':counts(w[...,::2]&w[...,1::2])}
        if 'eye' in row:ct['optical_rmse']=counts((row['eye']-optic[:,:,:,None,:,:])**2)
        layout[name]=ct;per.append(dict(arm=arm,budget=k,seed=s,**{t:float(unpack(v,sqrt=t=='optical_rmse')) for t,v in ct.items()}))
    dump(dest/'after.json',dict(files=hashes(files,dest),executed_ns=time.time_ns()))
    draw=np.random.default_rng(C['bootstrap'][0]).integers(0,n,size=(C['bootstrap'][1],n));main=[];diff=[]
    for k in P['label_budgets']:
        pool={}
        for arm in C['arms']+[NEW]:
            ls=[layout[f'{arm}-{k}-{s}'] for s in P['seeds']];data={key:sum(v[key] for v in ls) for key in ls[0]}
            pool[arm]=data['pair'];main.append(dict(arm=arm,budget=k,**{t:interval(unpack(v,sqrt=t=='optical_rmse'),unpack(v,draw,t=='optical_rmse')) for t,v in data.items()}))
        for a,b in [('joint',NEW),(NEW,'world')]:
            diff.append(dict(budget=k,a=a,b=b,delta=interval(unpack(pool[a])-unpack(pool[b]),unpack(pool[a],draw)-unpack(pool[b],draw))))
    dump(dest/'results.json',dict(plan=sha(PLAN),execution=sha(dest/'after.json'),n_layouts=n,rows=main,contrasts=diff,per_seed=per))
    for dd in diff:print(dd,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--device',default='cuda');a=p.parse_args()
    dest=a.out/'challenge';dest.mkdir(exist_ok=False)
    if not (a.out/'analysis/summary.json').exists():raise ValueError('this is explicitly a post-main follow-up')
    fit(a.out,dest,a.device);torch.set_num_threads(2);exam(a.out,dest)
