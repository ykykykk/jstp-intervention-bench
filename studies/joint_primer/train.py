"""Train fixed models on the exact, existing optical records; never open the exam in this process."""
from pathlib import Path
import argparse, itertools, json, platform, time
import numpy as np
import torch
from torch.nn import functional as F
from mechanism.data import read,save,sha,dump
from mechanism.train import config
from lategoal.rig import TRAIN
from .models import make

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
PROTO=HERE/'protocol.json'; C=json.loads(PROTO.read_text(encoding='utf8'))
DATA=ROOT/C['source']; SUP={'joint','aux','world','masked'}
COMB=np.asarray(list(itertools.combinations(range(12),4)),np.int64)
SOURCES=[PROTO,HERE/'models.py',Path(__file__).resolve(),ROOT/'lategoal/rig.py',ROOT/'jstp/v7_world.py']
def slots():
    return [(a,k,s,f'{a}-{k}-{s}') for k in C['label_budgets'] for s in C['seeds'] for a in C['arms']]
def uniform_cycles(rng,n,total):
    if total%n:raise ValueError('budget must cover whole cycles')
    return np.concatenate([rng.permutation(n) for _ in range(total//n)])
def schedule(out,n):
    rng=np.random.default_rng(C['plan_seed']);shape=(C['updates'],C['batch']);total=int(np.prod(shape))
    perm=rng.permutation(n)
    cards=np.empty(n,np.uint8); decks=np.concatenate([rng.permutation(len(TRAIN)) for _ in range((n+len(TRAIN)-1)//len(TRAIN))])
    cards[perm]=decks[:n]
    phy=uniform_cycles(rng,n,total); label=uniform_cycles(rng,n,total)
    u=lambda a:a.reshape(shape).astype(np.uint16)
    save(out/'schedule.npz',perm=perm.astype(np.uint16),cards=cards,physical=u(phy),label=u(label),
         side=rng.integers(2,size=shape,dtype=np.uint8),sensors=u(rng.integers(len(COMB),size=total)))
def tensor(x,device):return torch.as_tensor(x,device=device)
def annotations(bank,rows,card_ids,device):
    motion=bank['motion'].reshape(-1,4)
    g=TRAIN[card_ids[rows]]
    a=(motion[(rows//2)*2]*g).sum(1)>0;b=(motion[(rows//2)*2+1]*g).sum(1)>0
    if np.any(a==b):raise ValueError('recorded alternatives do not have one successful actuator')
    return tensor(g.astype('float32'),device),tensor(np.where(a,-1.,1.).astype('float32'),device)
def fit(out,device,arm,k,seed,bank,val,plan,locked):
    d=out/'models'/f'{arm}-{k}-{seed}';d.mkdir(parents=True,exist_ok=False)
    config(seed,device);m=make(arm,C['width']).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=C['lr'],weight_decay=1e-4)
    flat=lambda b,key:b[key].reshape(-1,b[key].shape[-1])
    x=tensor(flat(bank,'hist'),device); eyes=tensor(flat(bank,'eye'),device)
    select=plan['perm'][:k].astype(np.int64);card,y=annotations(bank,select,plan['cards'],device)
    sel=tensor(select,device); perm=tensor((plan['label']%k).astype('int64'),device)
    phys=tensor(plan['physical'].astype('int64'),device);side=tensor(plan['side'].astype('int64'),device)
    visible=tensor(COMB[plan['sensors']],device)
    vi=np.arange(len(flat(val,'hist')))
    vcard,vy=annotations(val,vi,(vi*11%len(TRAIN)),device)
    vx=tensor(flat(val,'hist'),device)
    t=time.monotonic()
    with (d/'curve.jsonl').open('x',encoding='utf8',buffering=1) as log:
        for step,(li,ri,si,mask) in enumerate(zip(perm,phys,side,visible),1):
            h=m.observe(x[sel[li]]);g=card[li];target=y[li]
            act=m.loss(h,g,target) if arm=='flow' else F.softplus(-target*m.score(h,g)).mean()
            loss=act
            if arm in SUP:
                z=m.future(m.observe(x[ri]),si)
                error=m.eye(z).gather(1,mask)-eyes[(ri//2)*2+si].gather(1,mask)
                loss=loss+C['optical_weight']*error.square().mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step()
            if step%128==0:
                with torch.no_grad():
                    value=float(loss.detach())
                    if not np.isfinite(value):raise FloatingPointError((arm,k,seed,step))
                    rec=dict(update=step,loss=value,validation_action=float(((m.score(m.observe(vx),vcard)>=0)==(vy>0)).float().mean()),
                             seconds=round(time.monotonic()-t,3))
                    log.write(json.dumps(rec)+'\n')
    with (d/'last.pt').open('xb') as f:torch.save(m.cpu().state_dict(),f)
    dump(d/'freeze.json',dict(arm=arm,budget=k,seed=seed,weights=sha(d/'last.pt'),protocol=sha(PROTO),
        schedule=locked['schedule'],sources=locked['sources'],registered_parameters=sum(p.numel() for p in m.parameters()),
        final_gradient_parameters=sum(p.numel() for p in m.parameters() if p.grad is not None),updates=step,
        seconds=round(time.monotonic()-t,3),torch=str(torch.__version__),python=platform.python_version(),
        device=torch.cuda.get_device_name(device) if device.startswith('cuda') else device))
    print('FROZEN',arm,k,seed,rec,flush=True)
def load(out,arm,k,seed):
    d=out/'models'/f'{arm}-{k}-{seed}';meta=json.loads((d/'freeze.json').read_text(encoding='utf8'))
    if meta['weights']!=sha(d/'last.pt') or meta['protocol']!=sha(PROTO) or meta['sources']['studies/joint_primer/models.py']!=sha(HERE/'models.py'):
        raise ValueError('frozen model mismatch: '+str(d))
    m=make(arm,C['width']).eval();m.load_state_dict(torch.load(d/'last.pt',map_location='cpu',weights_only=True));return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--device',default='cuda');a=p.parse_args()
    out=a.out.resolve()
    for key in ('train','validation'):
        if sha(DATA/(key+'.npz'))!=C[key+'_sha256']:raise ValueError('data fingerprint changed: '+key)
    bank,val=[read(DATA/(key+'.npz')) for key in ('train','validation')]
    out.mkdir(parents=True,exist_ok=False)
    schedule(out,len(bank['hist'])*8); plan=read(out/'schedule.npz')
    locked=dict(sources={p.relative_to(ROOT).as_posix():sha(p) for p in SOURCES},schedule=sha(out/'schedule.npz'),
                train=C['train_sha256'],validation=C['validation_sha256'],started_ns=time.time_ns())
    dump(out/'design-lock.json',locked)
    for arm,k,seed,name in slots():fit(out,a.device,arm,k,seed,bank,val,plan,locked)
    dump(out/'frozen.json',dict(design=sha(out/'design-lock.json'),schedule=sha(out/'schedule.npz'),
        completed_ns=time.time_ns(),models={name:sha(out/'models'/name/'freeze.json') for *_,name in slots()}))
if __name__=='__main__':main()
