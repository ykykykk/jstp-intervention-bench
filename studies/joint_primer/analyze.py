"""Paired estimates from actual camera execution; independent unit = layout, not goal card."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from mechanism.data import read,dump,save,sha
from .train import C,PROTO,slots
from .exam import ensure,jread

def counts(flag,eligible=None):
    a=np.asarray(flag);n=len(a)
    if eligible is None:den=np.full(n,a[0].size)
    else:
        eligible=np.broadcast_to(eligible,a.shape)
        a=a*eligible;den=eligible.reshape(n,-1).sum(1)
    return np.column_stack((a.reshape(n,-1).sum(1),den)).astype('float64')
def unpack(x,draw=None,sqrt=False):
    x=x.sum(0) if draw is None else x[draw].sum(1)
    v=x[...,0]/x[...,1]
    return np.sqrt(v) if sqrt else v
def interval(v,rep,level=95):
    q=(100-level)/200
    return dict(value=float(v),lo=float(np.quantile(rep,q)),hi=float(np.quantile(rep,1-q)),level=level)
def analyze(out):
    obs=out/'observed';before=jread(out/'before.json');after=jread(out/'after.json')
    ensure(out,after['files']);ensure(out,before['files'])
    if before['protocol']!=sha(PROTO):raise ValueError('exam protocol changed')
    gt=read(obs/'camera.npz');items=read(out/'sealed/items.npz');goals=items['goals']
    if not np.array_equal(goals[::2],-goals[1::2]):raise ValueError('not opposite goal pairs')
    n=len(items['ids']);left=gt['left'];optic=gt['eye'];signed=np.where((left[...,None,:]*goals).sum(-1)>0,-1,1)
    own=signed[:,:,:,None,:];donor=np.flip(own,axis=2);change=own!=donor
    N=C['bootstrap'][1];draw=np.random.default_rng(C['bootstrap'][0]).integers(0,n,size=(N,n))
    allrows={};per=[];record=[];pooled={}
    for arm,k,seed,name in slots():
        b=read(out/'sealed'/(name+'.npz'));w=read(obs/(name+'.npz'))['wins'];u=b['u']
        nat=w[...,0]
        if not np.array_equal(nat,u[...,0]==own):raise ValueError('record and camera score disagree')
        d={'single':counts(nat),'pair':counts(nat[...,::2]&nat[...,1::2])}
        for key in ('group','blind_group'):
            # All four signs, AND both clutches of the same layout, wiring and prior probe.
            d[key]=counts((b[key]==left[:,:,:,None,:]).all(-1).all(2))
        if 'eye' in b:d['optical_rmse']=counts((b['eye']-optic[:,:,:,None,:,:])**2)
        if arm=='joint':
            d['directed']=counts(nat&(u[...,1]==donor),change)
            d['null']=counts(nat&(u[...,2]==donor),change)
            d['false']=counts(nat&(u[...,1]!=own),~change)
            real=optic[:,:,:,None,0,:];alternative=np.flip(real,axis=2)
            dist=lambda x,y:((x-y)**2).sum(-1)
            src=b['eye'][...,0,:];patched=b['patched_eye']
            d['donor_match']=counts((dist(src,real)<dist(src,alternative))&(dist(patched,alternative)<dist(patched,real)))
        allrows[name]=d
        per.append(dict(arm=arm,budget=k,seed=seed,metrics={key:float(unpack(x,sqrt=key=='optical_rmse')) for key,x in d.items()}))
    for k in C['label_budgets']:
        for arm in C['arms']:
            raw=[allrows[f'{arm}-{k}-{s}'] for s in C['seeds']]
            d={key:sum(r[key] for r in raw) for key in raw[0]}
            pooled[arm,k]=d;ms={}
            for key,x in d.items():
                rt=key=='optical_rmse';ms[key]=interval(unpack(x,sqrt=rt),unpack(x,draw,rt))
            def delta(key,other):
                return interval(unpack(d[key])-unpack(d[other]),unpack(d[key],draw)-unpack(d[other],draw))
            ms['history_group_gap']=delta('group','blind_group')
            if arm=='joint':
                ms['over_null']=delta('directed','null');ms['over_false']=delta('directed','false')
            record.append(dict(arm=arm,budget=k,metrics=ms))
    contrasts=[]
    for k in C['label_budgets']:
        for arm in C['arms'][1:]:
            a,b=pooled['joint',k]['pair'],pooled[arm,k]['pair']
            dif=unpack(a)-unpack(b);rep=unpack(a,draw)-unpack(b,draw)
            contrasts.append(dict(budget=k,other=arm,delta=interval(dif,rep),family=interval(dif,rep,99)))
    destination=out/'analysis';destination.mkdir(exist_ok=False)
    dump(destination/'summary.json',dict(protocol=sha(PROTO),execution=sha(out/'after.json'),n_layouts=n,
         rows=record,paired=contrasts,uncertainty='fixed 3 initializations; resample 192 whole layouts paired across arms'))
    dump(destination/'per-seed.json',per)
    save(destination/'layout-counts.npz',ids=items['ids'],**{name+'__'+key:x for name,d in allrows.items() for key,x in d.items()})
    with (destination/'metrics.csv').open('x',encoding='utf8',newline='') as f:
        wr=csv.writer(f);wr.writerow(['method','annotations','metric','estimate','lo95','hi95'])
        for r in record:
            for key,v in r['metrics'].items():wr.writerow([r['arm'],r['budget'],key,v['value'],v['lo'],v['hi']])
    print('paired success % / functional grouping %')
    for r in record:print(r['arm'],r['budget'],*(round(100*r['metrics'][s]['value'],2) for s in ('pair','group')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);analyze(p.parse_args().out)
