"""Freeze -> seal all signed commitments -> actually operate the published rig -> paired layout analysis."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from lategoal.rig import EXAM,Rig,goal_wins
from lategoal.control import fingerprint
from lategoal.exam import actual
from studies.joint_primer.train import load as load_joint
from studies.joint_primer.exam import quantile_sign
from studies.joint_primer.analyze import counts,unpack,interval
from studies.joint_primer.official_exam import check_items, frozen as official_frozen, scores as official_scores
from studies.joint_primer.official_train import restore,visual
from .common import C,E,HERE,MAIN,OFF,P,PLAN,SCHEDULE,jread,ensure,sha,save,read,dump,runs
from .train import feature_bank,model

def frozen():
    paths={}
    for arm in ('joint',*P['fresh_external']):
        d=E/arm; path=d/'frozen.json';meta=jread(path)
        if arm=='joint':
            if meta['plan'] not in (sha(HERE/'initial-plan.json'),sha(PLAN)) or meta['train']!=C['train_sha256'] or meta['schedule']!=sha(SCHEDULE):
                raise ValueError('original JSTP data, plan or schedule changed')
        elif meta['lock']['plan']!=sha(PLAN) or meta['lock']['fit_code']!=sha(HERE/'train.py'):
            raise ValueError('different adapter training protocol: '+arm)
        if set(meta['models'])!={n for _,_,n in runs(arm)}:raise ValueError('incomplete '+arm)
        ensure(d,{'models/'+name+'/freeze.json':value for name,value in meta['models'].items()})
        paths[arm+'/frozen.json']=sha(path)
    return paths

@torch.inference_mode()
def seal():
    paths=frozen();official_frozen(OFF);items=check_items(); ensure(OFF,jread(OFF/'prefix.json')['files'])
    ensure(OFF,{'frozen.json':jread(OFF/'before.json')['files']['frozen.json']})
    prefix=read(OFF/'public-prefix.npz')
    if not np.array_equal(prefix['ids'],items['ids']) or not np.array_equal(prefix['goals'],EXAM):raise ValueError('unequal test layout or goal order')
    x=prefix['hist'];shape=x.shape[:-1]; xc=torch.as_tensor(x.reshape(-1,x.shape[-1]))
    goals=torch.as_tensor(EXAM).float();archive=jread(OFF/'before.json')['files'];historical_sign_differences={}
    dest=E/'sealed';dest.mkdir(exist_ok=False)
    for arm in P['arms']:
        native_visual=None
        if arm in P['fresh_external']:
            f,meta=feature_bank(arm,'exam')
            trainmeta=jread(E/arm/'train-features.json')
            for key in ('base_sha256','upstream_commit','code','plan'):
                if meta[key]!=trainmeta[key]: raise ValueError((arm,key))
        for k,s,name in runs(arm):
            if arm=='joint':
                m=load_joint(E/arm,arm,k,s);h=m.observe(xc)
                a=torch.stack([m.score(h,g.expand(len(h),-1)) for g in goals],1).numpy().reshape(*shape,16)
                u=quantile_sign(a)
            elif arm in P['archived']:
                rec=jread(OFF/'models'/name/'freeze.json');orig=jread(OFF/'models'/f'smol-{k}-{s}'/'freeze.json')
                if (rec['data']!=C['train_sha256'] or rec['schedule']!=sha(SCHEDULE) or
                    rec['annotation_pool_sha256']!=orig['annotation_pool_sha256']):raise ValueError(name+' unequal archived inputs')
                m=restore(OFF,arm,k,s);public=xc.cuda()
                if native_visual is None:native_visual=visual(m,arm,public)
                logits,_=official_scores(arm,m,public,native_visual)
                a=logits.reshape(*shape,16);u=quantile_sign(a)
                rel='sealed/'+name+'.npz';ensure(OFF,{rel:archive[rel]})
                historical_sign_differences[name]=int(np.count_nonzero(u!=read(OFF/rel)['u']))
                del m
            else:
                d=E/arm/'models'/name;rec=jread(d/'freeze.json');ensure(d,{'adapter.pt':rec['weights']})
                orig=jread(OFF/'models'/f'smol-{k}-{s}'/'freeze.json')
                if orig['annotation_pool_sha256']!=rec['annotation_pool_sha256']:raise ValueError(name+' unequal annotations')
                m=model(arm,f,meta).cuda().eval()
                m.load_state_dict(torch.load(d/'adapter.pt',map_location='cuda',weights_only=True))
                inst=xc[:,144:].cuda();out=[]
                for idx in torch.arange(len(xc),device='cuda').split(256):
                    out.append((m(f,idx,inst)[:,:4]@goals.cuda().T).cpu().numpy())
                a=np.concatenate(out).reshape(*shape,16);u=quantile_sign(a)
            if not np.isfinite(a).all() or tuple(u.shape)!=(len(items['ids']),2,2,2,16): raise ValueError(name)
            path=dest/(name+'.npz');save(path,u=u,score=a)
            paths[path.relative_to(E).as_posix()]=sha(path)
        if arm in P['fresh_external']: del f
        print('SEALED',arm,flush=True)
    dump(E/'before.json',dict(files=paths,plan=sha(PLAN),items=sha(MAIN/'sealed/items.npz'),
        official_seal=sha(OFF/'before.json'),recomputed_vs_archived_sign_differences=historical_sign_differences,
        code=sha(Path(__file__)),sealed_ns=time.time_ns()))

def execute():
    lock=jread(E/'before.json');ensure(E,lock['files'])
    if lock['plan']!=sha(PLAN) or lock['code']!=sha(Path(__file__)):raise ValueError('intent or executor changed')
    items=check_items();ref=MAIN/'observed/camera.npz';ensure(MAIN,{'observed/camera.npz':jread(MAIN/'after.json')['files']['observed/camera.npz']})
    old=read(ref);names=[n for a in P['arms'] for _,_,n in runs(a)]
    commands={n:read(E/'sealed'/(n+'.npz'))['u'] for n in names};wins={n:np.empty_like(v,dtype=bool) for n,v in commands.items()}
    dest=E/'observed';dest.mkdir(exist_ok=False);paths={'before.json':sha(E/'before.json')}
    for i,seed in enumerate(items['ids']):
        r=Rig(int(seed))
        if fingerprint(r.base)!=items['base'][i]:raise ValueError('initial camera mismatch')
        for w in (0,1):
            for g in (0,1):
                for pi,p in enumerate((-1,1)):
                    a=np.stack([cmd[i,w,g,pi] for cmd in commands.values()])
                    vel,left,eye,hs=actual(r,w,g,p,a,items['peak'][i,pi,w])
                    if not (np.array_equal(old['frame_sha'][i,w,g],hs) and np.array_equal(old['eye'][i,w,g,pi],eye)
                            and np.array_equal(old['left'][i,w,g],left)): raise ValueError((i,w,g,pi,'camera'))
                    for buf,success in zip(wins.values(),goal_wins(vel,EXAM),strict=True):buf[i,w,g,pi]=success
    for n,v in wins.items():
        path=dest/(n+'.npz');save(path,wins=v);paths[path.relative_to(E).as_posix()]=sha(path)
    dump(E/'after.json',dict(files=paths,camera=sha(ref),real_decisions=sum(v.size for v in wins.values()),
         layouts=len(items['ids']),matched_all_camera_hashes=True,completed_ns=time.time_ns()))

def analyze():
    after=jread(E/'after.json');ensure(E,after['files']);before=jread(E/'before.json');ensure(E,before['files'])
    N=after['layouts'];draw=np.random.default_rng(C['bootstrap'][0]).integers(0,N,(C['bootstrap'][1],N))
    data={};detail=[];table=[];contrasts=[];replay={}
    for a in P['arms']:
        for k,s,n in runs(a):
            w=read(E/'observed'/(n+'.npz'))['wins'];u=read(E/'sealed'/(n+'.npz'))['u']
            pair=w[...,::2]&w[...,1::2];same=u[...,::2]==u[...,1::2]
            if int(w.sum())!=2*int(pair.sum())+int(same.sum()):raise ValueError(n+' opposite card invariant')
            d={k:counts(v) for k,v in [('pair',pair),('single',w),('same',same)]};data[n]=d
            detail.append(dict(arm=a,budget=k,seed=s,won=int(pair.sum()),pairs=pair.size,single_won=int(w.sum()),
                               singles=w.size,same_count=int(same.sum()),**{key:float(unpack(v)) for key,v in d.items()}))
            if a in ('joint',*P['archived']):
                base=MAIN if a=='joint' else OFF;key='observed/'+n+'.npz'
                ensure(base,{key:jread(base/'after.json')['files'][key]});old=read(base/key)['wins']
                replay[n]=int((w!=(old[...,0] if a=='joint' else old)).sum())
    for k in P['budgets']:
        pooled={}
        for a in P['arms']:
            pack=[data[n] for kk,_,n in runs(a) if k==kk]
            obs={key:sum(t[key] for t in pack) for key in pack[0]};pooled[a]=obs['pair']
            table.append(dict(arm=a,budget=k,won=int(obs['pair'][:,0].sum()),pairs=int(obs['pair'][:,1].sum()),
                **{key:interval(unpack(v),unpack(v,draw)) for key,v in obs.items()}))
        for a in P['arms']:
            if a=='joint':continue
            j,b=pooled['joint'],pooled[a];val=unpack(j)-unpack(b);rep=unpack(j,draw)-unpack(b,draw)
            contrasts.append(dict(other=a,budget=k,delta95=interval(val,rep),primary=interval(val,rep,100-5/(len(P['arms'])-1)) if k==128 else None,
                seed_deltas=[float(unpack(data[f'joint-{k}-{s}']['pair'])-unpack(data[f'{a}-{k}-{s}']['pair'])) for s in P['seeds']]))
    dump(E/'summary.json',dict(layouts=N,per_checkpoint_pairs=12288,exam_dates='same historical panel, not newly blind',
         execution_hash=sha(E/'after.json'),actual_commands=after['real_decisions'],rows=table,contrasts=contrasts,archived_score_mismatches=replay))
    dump(E/'per-seed.json',detail)
    for row in table:print(row['arm'],row['budget'],row['won'],row['pairs'],row['pair'],flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','seal','execute','analyze'],default='all',nargs='?');a=p.parse_args()
    for fn in (seal,execute,analyze):
        if a.stage in ('all',fn.__name__):fn()

