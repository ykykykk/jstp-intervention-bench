"""Make all commitments, execute the actual simulated device, then compare on the *same* layouts."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from lategoal.rig import EXAM,TRAIN,Rig,goal_wins
from lategoal.control import fingerprint
from lategoal.exam import actual
from mechanism.data import sha,read,save,dump
from studies.partial_futures.exam import prehistory
from .train import ROOT,C
from .exam import ensure,hashes,jread,quantile_sign
from .analyze import counts,unpack,interval
from .official_train import MAIN,DEFAULT,P,PLAN,WORK,runs,restore,visual,TokenPrefix,dino_eye

def frozen(out):
    lock=jread(out/'frozen.json')
    ensure(out,{'design.json':lock['design'],**{'models/'+n+'/freeze.json':v for n,v in lock['models'].items()}})
    ensure(ROOT,P['source_locks']);ensure(MAIN,{'sealed/items.npz':P['exam_items_sha256']})
    if set(lock['models'])!={name for *_,name in runs()}:raise ValueError('not all eighteen weights frozen')
def check_items():
    items=read(MAIN/'sealed/items.npz')
    if not np.array_equal(items['goals'],EXAM) or not np.array_equal(items['ids'],np.arange(C['exam'][0],sum(C['exam']))):
        raise ValueError('not the original 192 layouts and 16 goal cards')
    return items
@torch.inference_mode()
def scores(arm,m,x,v):
    goals=torch.as_tensor(EXAM,device=x.device).float();N=len(x);G=len(goals)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if arm=='dino':
            eyes=[];rows=[]
            for xx,vv in zip(x.split(128),v.split(128)):
                one=torch.ones_like(xx[:,:1]);a=dino_eye(m,vv,xx,-one);b=dino_eye(m,vv,xx,one)
                eyes.append(torch.stack((a,b),1).float().cpu().numpy())
                rows.append(((m.utility(b)-m.utility(a))@goals.T).float().cpu().numpy())
            return np.concatenate(rows),np.concatenate(eyes)
        replay=TokenPrefix(m.native);rows=[]
        for flat in torch.arange(N*G,device='cuda').split(96):
            row=flat//G;j=flat%G
            z=replay.score(m,v[row],x[row],goals[j],j+len(TRAIN))
            rows.append(z.float().cpu().numpy())
    return np.concatenate(rows).reshape(N,G),None
def capture(out):
    """Materialize only the public pre-action input, using the original simulator environment."""
    items=check_items();rigs=[Rig(int(i)) for i in items['ids']]
    _,states,pk=prehistory(rigs)
    if not np.array_equal(items['peak'],pk) or not np.array_equal(items['base'],[fingerprint(r.base) for r in rigs]):
        raise ValueError('different image history or probe')
    history=np.ascontiguousarray(states.transpose(0,2,1,3).reshape(len(rigs),2,2,2,-1))
    f=out/'public-prefix.npz';save(f,hist=history,ids=items['ids'],goals=items['goals'])
    dump(out/'prefix.json',dict(files={f.name:sha(f)},items=sha(MAIN/'sealed/items.npz'),numpy=np.__version__))
def seal(out):
    frozen(out);items=check_items();rec=jread(out/'prefix.json');ensure(out,rec['files'])
    ensure(WORK/'upstream-smol-processor',jread(out/'design.json')['processor'])
    pref=read(out/'public-prefix.npz')
    if rec['items']!=sha(MAIN/'sealed/items.npz') or not np.array_equal(pref['ids'],items['ids']) or not np.array_equal(pref['goals'],EXAM):
        raise ValueError('different official evaluation inputs')
    x=torch.as_tensor(pref['hist'].reshape(-1,pref['hist'].shape[-1]),device='cuda')
    shape=(len(items['ids']),2,2,2);d=out/'sealed';d.mkdir(exist_ok=False);paths=[out/'frozen.json',out/'prefix.json'];v=None;prev=None
    for arm,k,s,name in runs():
        m=restore(out,arm,k,s)
        if arm!=prev:v=visual(m,arm,x);prev=arm
        sc,eye=scores(arm,m,x,v)
        if not np.isfinite(sc).all() or (eye is not None and not np.isfinite(eye).all()):raise ValueError(name+' produced nonfinite predictions')
        a=sc.reshape(*shape,len(EXAM));row=dict(u=quantile_sign(a),score=a)
        if eye is not None:row['eye']=eye.reshape(*shape,2,12)
        f=d/(name+'.npz');save(f,**row);paths.append(f);print('SEALED',name,flush=True);del m
    dump(out/'before.json',dict(plan=sha(PLAN),files=hashes(paths,out),item_sha=sha(MAIN/'sealed/items.npz'),
         code=sha(Path(__file__)),primary_before=sha(MAIN/'before.json'),sealed_ns=time.time_ns()))
def check_before(out):
    record=jread(out/'before.json')
    if record['plan']!=sha(PLAN) or record['code']!=sha(Path(__file__)) or record['item_sha']!=sha(MAIN/'sealed/items.npz') or record['primary_before']!=sha(MAIN/'before.json'):
        raise ValueError('official intention/panel identity changed')
    frozen(out);ensure(out,record['files'])
def execute(out):
    check_before(out);items=check_items();old=jread(MAIN/'after.json')
    ensure(MAIN,{'observed/camera.npz':old['files']['observed/camera.npz']})
    refcam=read(MAIN/'observed/camera.npz')
    d=out/'observed';d.mkdir(exist_ok=False);names=[n for *_,n in runs()]
    issued={n:read(out/'sealed'/(n+'.npz'))['u'] for n in names}
    wins={n:np.empty(a.shape,bool) for n,a in issued.items()}
    observed={k:np.empty_like(a) for k,a in refcam.items()}
    for i,num in enumerate(items['ids']):
        r=Rig(int(num))
        if fingerprint(r.base)!=items['base'][i]:raise ValueError('changed initial camera')
        for s in (0,1):
            for g in (0,1):
                for pi,p in enumerate((-1,1)):
                    actions=np.stack([a[i,s,g,pi] for a in issued.values()])
                    vel,left,optic,hs=actual(r,s,g,p,actions,items['peak'][i,pi,s])
                    if (not np.array_equal(hs,refcam['frame_sha'][i,s,g]) or not np.array_equal(left,refcam['left'][i,s,g])
                        or not np.array_equal(optic,refcam['eye'][i,s,g,pi])):
                        raise ValueError('camera outcomes differ from introductory study')
                    observed['frame_sha'][i,s,g]=hs;observed['left'][i,s,g]=left;observed['eye'][i,s,g,pi]=optic
                    correct=goal_wins(vel,EXAM)
                    for w,c in zip(wins.values(),correct):w[i,s,g,pi]=c
    f=d/'camera.npz';save(f,**observed);paths=[out/'before.json',f]
    for n,w in wins.items():
        f=d/(n+'.npz');save(f,wins=w);paths.append(f)
    dump(out/'after.json',dict(files=hashes(paths,out),official_actions=int(sum(w.size for w in wins.values())),
        layouts=len(items['ids']),camera_matches_primary=True,executed_ns=time.time_ns()))
def analyze(out):
    after=jread(out/'after.json');ensure(out,after['files']);check_before(out);items=check_items()
    original=jread(MAIN/'after.json');before=jread(MAIN/'before.json')
    if sha(MAIN/'after.json')!=P['source_locks']['studies/joint_primer/evidence/after.json']:raise ValueError('changed main results')
    N=len(items['ids']);draw=np.random.default_rng(C['bootstrap'][0]).integers(0,N,(C['bootstrap'][1],N))
    bank={};per=[];rows=[];contrasts=[];names=['joint']+P['models']
    for k in P['budgets']:
        for arm in names:
            for seed in P['seeds']:
                ident=f'{arm}-{k}-{seed}';base=MAIN if arm=='joint' else out
                if arm=='joint':
                    for folder,rec in [('observed',original),('sealed',before)]:
                        key=f'{folder}/{ident}.npz';ensure(MAIN,{key:rec['files'][key]})
                w=read(base/'observed'/(ident+'.npz'))['wins']
                u=read(base/'sealed'/(ident+'.npz'))['u']
                if arm=='joint':w,u=w[...,0],u[...,0]
                if w.shape!=(N,2,2,2,16) or u.shape!=w.shape:raise ValueError('unequal testing opportunities')
                pair=w[...,::2]&w[...,1::2];same=u[...,::2]==u[...,1::2]
                if int(w.sum())!=2*int(pair.sum())+int(same.sum()):raise ValueError('opposite-card action/outcome inconsistency')
                d=dict(pair=counts(pair),single=counts(w),same_action=counts(same));bank[ident]=d
                per.append(dict(arm=arm,budget=k,seed=seed,pairs_won=int(pair.sum()),pairs=int(pair.size),
                    metrics={key:float(unpack(v)) for key,v in d.items()}))
    for k in P['budgets']:
        pooled={}
        for a in names:
            hh=[bank[f'{a}-{k}-{s}'] for s in P['seeds']]
            d={key:sum(row[key] for row in hh) for key in hh[0]};pooled[a]=d['pair']
            rows.append(dict(arm=a,budget=k,metrics={key:interval(unpack(c),unpack(c,draw)) for key,c in d.items()},
                pairs_won=int(d['pair'][:,0].sum()),pairs=int(d['pair'][:,1].sum())))
        for a in P['models']:
            x,y=pooled['joint'],pooled[a];delta=unpack(x)-unpack(y);sim=unpack(x,draw)-unpack(y,draw)
            seeds=[float(unpack(bank[f'joint-{k}-{s}']['pair'])-unpack(bank[f'{a}-{k}-{s}']['pair'])) for s in P['seeds']]
            fam=interval(delta,sim,97.5) if k==128 else None
            contrasts.append(dict(budget=k,other=a,delta95=interval(delta,sim),family=fam,seed_deltas=seeds,
                primary_pass=(fam['lo']>0 and sum(v>0 for v in seeds)>=2) if fam else None))
    dest=out/'analysis';dest.mkdir(exist_ok=False)
    dump(dest/'summary.json',dict(plan=sha(PLAN),execution=sha(out/'after.json'),main=sha(MAIN/'after.json'),
         layouts=N,seeds=P['seeds'],new_fits=len(runs()),new_actions=after['official_actions'],rows=rows,paired=contrasts,
         both_primary_pass=all(c['primary_pass'] for c in contrasts if c['budget']==128)))
    dump(dest/'per-seed.json',per)
    save(dest/'layout-counts.npz',ids=items['ids'],**{n+'__'+k:v for n,d in bank.items() for k,v in d.items()})
    with (dest/'metrics.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['method','annotations','metric','estimate','lo95','hi95'])
        for row in rows:
            for key,est in row['metrics'].items():w.writerow([row['arm'],row['budget'],key,est['value'],est['lo'],est['hi']])
    for c in contrasts:print('JSTP minus',c['other'],c['budget'],c['delta95'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--stage',choices=('all','capture','seal','execute','analyze'),default='all');a=p.parse_args()
    torch.set_num_threads(2)
    for fn in (capture,seal,execute,analyze):
        if a.stage in ('all',fn.__name__):fn(a.out)
