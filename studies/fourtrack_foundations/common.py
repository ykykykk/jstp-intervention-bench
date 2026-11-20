from pathlib import Path
import json
from mechanism.data import read, save, sha, dump
from studies.joint_primer.train import HERE as PRIMER, ROOT, C, DATA, SOURCES, fit
from studies.joint_primer.exam import ensure

HERE=Path(__file__).resolve().parent
E=HERE/'evidence'
WORK=ROOT/'work'
CACHE=WORK/'fourtrack-foundations'
PLAN=HERE/'plan.json'
P=json.loads(PLAN.read_text(encoding='utf8'))
MAIN=PRIMER/'evidence'
OFF=MAIN/'official'
SCHEDULE=MAIN/'train/schedule.npz'
def jread(path): return json.loads(path.read_text(encoding='utf8'))
def runs(arm):
    for k in P['budgets']:
        for s in P['seeds']:
            yield k,s,f'{arm}-{k}-{s}'
def training_data():
    official=jread(PRIMER/'official-plan.json')
    ensure(DATA,{k+'.npz':C[k+'_sha256'] for k in ('train','validation')})
    ensure(SCHEDULE.parent,{SCHEDULE.name:official['schedule_sha256']})
    return read(DATA/'train.npz'),read(SCHEDULE)
def data_locks():
    return dict(plan=sha(PLAN),train=C['train_sha256'],schedule=sha(SCHEDULE),
                sources={f.relative_to(ROOT).as_posix():sha(f) for f in SOURCES})

def train_joint():
    import torch
    torch.set_num_threads(2)
    bank,sched=training_data(); val=read(DATA/'validation.npz')
    d=E/'joint'; d.mkdir(parents=True,exist_ok=True)
    locks=data_locks()
    for k,s,name in runs('joint'):
        f=d/'models'/name/'freeze.json'
        if f.exists():
            meta=jread(f); ensure(f.parent,{'last.pt':meta['weights']})
            if meta['schedule']!=locks['schedule'] or meta['sources']!=locks['sources']: raise ValueError(name)
            continue
        fit(d,'cuda','joint',k,s,bank,val,sched,locks)
    target=d/'frozen.json'
    if not target.exists():
        dump(target,dict(**locks,models={n:sha(d/'models'/n/'freeze.json') for _,_,n in runs('joint')}))
if __name__=='__main__': train_joint()
