"""Train only the named external adapters. The full JSTP uses the untouched paper trainer."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mechanism.train import config
from studies.joint_primer.train import annotations, COMB
from .features import feature_code
from .common import CACHE, E, P, PLAN, HERE, jread, ensure, sha, dump, runs, data_locks, training_data

class VideoReadout(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.token=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,32),nn.SiLU())
        self.read=nn.Sequential(nn.Linear(16*32+18,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,28))
    def forward(self,f,idx,instruments):
        t=self.token(f['v'][idx].float()).flatten(1)
        return self.read(torch.cat((t,instruments[idx]),1))

class NativeTerminal(nn.Module):
    def __init__(self,f):
        super().__init__()
        w,b=f['weight'],f['bias']
        self.project=nn.Linear(w.shape[1],len(w))
        with torch.no_grad():self.project.weight.copy_(w);self.project.bias.copy_(b)
        self.hidden=nn.Sequential(nn.LayerNorm(w.shape[1]),nn.Linear(w.shape[1],64),nn.SiLU())
        self.read=nn.Sequential(nn.Linear(64+18+2*len(w),96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,28))
    def forward(self,f,idx,instruments):
        h=f['v'][idx].float()
        return self.read(torch.cat((self.hidden(h),self.project(h),f['native'][idx],instruments[idx]),1))

def feature_bank(arm,split):
    path=CACHE/f'{arm}-{split}.pt';meta=jread(E/arm/(split+'-features.json'))
    if meta['file']!=sha(path) or meta['plan']!=sha(PLAN) or meta['code']!=feature_code(arm):
        raise ValueError('frozen input cache changed: '+str(path))
    return torch.load(path,map_location='cuda',weights_only=True),meta

def model(arm,f,meta):
    if arm in ('act','diffusion'): return NativeTerminal(f)
    if arm=='vjepa21': return VideoReadout(f['v'].shape[-1])
    raise ValueError(arm)

def fit(arm):
    bank,sched=training_data();f,meta=feature_bank(arm,'train');root=E/arm
    lock=dict(**data_locks(),fit_code=sha(Path(__file__)),feature_record=sha(root/'train-features.json'))
    inst=torch.as_tensor(bank['hist'].reshape(-1,162)[:,144:],device='cuda')
    eyes=torch.as_tensor(bank['eye'].reshape(-1,12),device='cuda')
    tensor=lambda a:torch.as_tensor(np.asarray(a,dtype='int64'),device='cuda')
    phys=tensor(sched['physical']);side=tensor(sched['side']);mask=tensor(COMB[sched['sensors']])
    for k,s,name in runs(arm):
        d=root/'models'/name;freeze=d/'freeze.json'
        if freeze.exists():
            rec=jread(freeze);ensure(d,{'adapter.pt':rec['weights']})
            if rec['lock']!=lock: raise ValueError(name+' belongs to another design')
            continue
        d.mkdir(parents=True,exist_ok=False);config(s,'cuda')
        ids=sched['perm'][:k].astype('int64');g,y=annotations(bank,ids,sched['cards'],'cuda')
        ix=tensor(ids);li=tensor(sched['label']%k)
        m=model(arm,f,meta).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
        tic=time.monotonic()
        with (d/'curve.jsonl').open('x',encoding='utf8') as stream:
            for step,(l,r,a,visible) in enumerate(zip(li,phys,side,mask),1):
                pred=m(f,torch.cat((ix[l],r)),inst)
                motor,optic=pred.split(len(l));score=(motor[:,:4]*g[l]).sum(1)
                act_loss=F.softplus(-score*y[l]).mean()
                measured=optic[:,4:].reshape(-1,2,12)[torch.arange(len(r),device='cuda'),a]
                err=(measured-eyes[(r//2)*2+a]).gather(1,visible)
                loss=act_loss+3*err.square().mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),3); opt.step()
                if step%128==0:
                    v=float(loss.detach())
                    if not np.isfinite(v):raise FloatingPointError((name,step))
                    stream.write(json.dumps(dict(step=step,loss=v,seconds=round(time.monotonic()-tic,2)))+'\n')
        ck=d/'adapter.pt'
        with ck.open('xb') as stream:torch.save(m.cpu().state_dict(),stream)
        pool=np.column_stack((ids,sched['cards'][ids],y.cpu().numpy().astype('int64'))).astype('<i8')
        digest=hashlib.sha256(np.ascontiguousarray(pool).tobytes()).hexdigest()
        dump(freeze,dict(lock=lock,weights=sha(ck),seed=s,budget=k,updates=step,batch=128,
              annotation_pool_sha256=digest,annotation_draws=step*128,physical_draws=step*128,
              trainable=sum(p.numel() for p in m.parameters()),seconds=round(time.monotonic()-tic,2)))
        print('FROZEN',name,round(time.monotonic()-tic,2),flush=True)
    if not (root/'frozen.json').exists():
        dump(root/'frozen.json',dict(lock=lock,models={n:sha(root/'models'/n/'freeze.json') for _,_,n in runs(arm)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=P['fresh_external']);fit(p.parse_args().arm)

