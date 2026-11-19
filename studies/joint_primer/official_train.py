"""New official-base adaptations. Uses the introductory study's fixed annotations and sample order."""
import argparse,hashlib,json,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from torch.nn import functional as F
from mechanism.data import read,dump,sha
from mechanism.train import config
from studies.partial_futures.transfer import make,params,picture,WORK,P as BASE
from .train import HERE,ROOT,DATA,COMB,annotations
from .exam import jread,ensure

PLAN=HERE/'official-plan.json'; P=jread(PLAN); MAIN=HERE/'evidence'; DEFAULT=MAIN/'official'
FOLD={'smol':WORK/'upstream-smolvla','dino':WORK/'upstream-dino'}
def runs():
    return [(a,k,s,f'{a}-{k}-{s}') for a in P['models'] for k in P['budgets'] for s in P['seeds']]
def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def base_model(arm,seed):
    config(seed,'cuda')
    m=make(arm,FOLD[arm],WORK/'upstream-smol-processor').eval()
    m.start.requires_grad_(False)  # The introductory exam balances both probes; no auxiliary home label.
    return m
def trainable(m):return params(m)
def torch_index(a):return torch.as_tensor(np.asarray(a,dtype='int64'),device='cuda')
@torch.no_grad()
def visual(m,arm,x):
    features=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for b in x.split(64):
            if arm=='smol':
                v=m.native.vlm_with_expert.embed_image(2*picture(b,128)-1)
                v=v*v.new_tensor(v.shape[-1]**.5)
            else:
                im=picture(b,56);mu=im.new_tensor([.485,.456,.406])[None,:,None,None]
                sd=im.new_tensor([.229,.224,.225])[None,:,None,None]
                v=m.backbone(pixel_values=(im-mu)/sd).last_hidden_state[:,1:]
            features.append(v)
    return torch.cat(features)
class TokenPrefix:
    """Call the *installed native* training/sampling methods, replacing only a frozen vision call by its output."""
    def __init__(self,core):
        self.core=core
        if core.add_image_special_tokens or core.prefix_length!=0:raise ValueError('unexpected pinned Smol prefix format')
    def __getattr__(self,key):return getattr(self.core,key)
    def embed_prefix(self,images,img_masks,lang_tokens,lang_masks,state):
        v,=images
        # Native language embedding, trainable state projection, and block attention masks are unchanged.
        a,p,f=self.core.embed_prefix([],[],lang_tokens,lang_masks,state=state)
        mask=img_masks[0][:,None].expand(-1,v.shape[1])
        return torch.cat((v,a),1),torch.cat((mask,p),1),torch.cat((torch.zeros_like(mask),f),1)
    def inputs(self,m,v,x,g,idx):
        state=F.pad(torch.cat((x[:,144:],g),1),(0,10))
        return [v],[torch.ones(len(x),device=x.device,dtype=torch.bool)],m.words[idx],m.mask[idx],state
    def train_loss(self,m,v,x,g,idx,y):
        act=x.new_zeros((len(x),1,32));act[:,0,0]=y
        noise=torch.zeros_like(act);noise[:,0,0]=torch.randn_like(y)
        losses=type(self.core).forward(self,*self.inputs(m,v,x,g,idx),actions=act,noise=noise)
        return losses[:,:,0].float().mean()
    def score(self,m,v,x,g,idx):
        noise=x.new_zeros((len(x),1,32))
        return type(self.core).sample_actions(self,*self.inputs(m,v,x,g,idx),noise=noise)[:,0,0]
@torch.no_grad()
def check_cache(m,arm,x):
    """One training-input identity check, not used for model selection."""
    from lategoal.rig import TRAIN
    x=x[:2];v=visual(m,arm,x);ids=torch.arange(2,device='cuda');g=torch.as_tensor(TRAIN[:2],device='cuda').float()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if arm=='smol':
            replay=TokenPrefix(m.native)
            direct=m.native.embed_prefix(*m.inputs(x,g,ids))
            cached=replay.embed_prefix(*replay.inputs(m,v,x,g,ids))
        else:
            direct=(m.candidates(x),)
            cached=(torch.stack([dino_eye(m,v,x,torch.full_like(x[:,:1],p)) for p in (-1,1)],1),)
    gaps=[]
    for a,b in zip(direct,cached):
        if not torch.equal(a,b):raise ValueError('frozen visual cache disagrees with native path: '+arm)
        gaps.append(float((a.float()-b.float()).abs().max()))
    return gaps
def dino_eye(m,v,x,side):
    conditioning=m.body(torch.cat((x[:,144:],side),1))
    return m.eye(m.transition(m.img(v)+conditioning[:,None]).flatten(1))
def fit(m,arm,k,seed,name,out,bank,plan,x,v,locks):
    d=out/'models'/name;d.mkdir(parents=True,exist_ok=False)
    ids=plan['perm'][:k].astype('int64');g,y=annotations(bank,ids,plan['cards'],'cuda')
    card_ix=torch_index(plan['cards'][ids]);idx=torch_index(ids);lbl=torch_index(plan['label']%k)
    phys=torch_index(plan['physical']);side=torch_index(plan['side']);mask=torch_index(COMB[plan['sensors']])
    eyes=torch.as_tensor(bank['eye'].reshape(-1,12),device='cuda')
    par=trainable(m);opt=torch.optim.AdamW(par.values(),lr=BASE[arm]['lr'],weight_decay=1e-4)
    flow=TokenPrefix(m.native) if arm=='smol' else None
    micro=64 if arm=='smol' else P['batch']
    tic=time.monotonic()
    with (d/'curve.jsonl').open('x',encoding='utf8',newline='\n',buffering=1) as log:
        for step,(l,r,si,vis) in enumerate(zip(lbl,phys,side,mask),1):
            opt.zero_grad(set_to_none=True);losses=[]
            for il,ir,j,visible in zip(l.split(micro),r.split(micro),si.split(micro),vis.split(micro)):
                ii=idx[il]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if arm=='smol':loss=flow.train_loss(m,v[ii],x[ii],g[il],card_ix[il],y[il])
                    else:
                        xx,vv=x[ii],v[ii];one=torch.ones_like(xx[:,:1])
                        a=dino_eye(m,vv,xx,-one);b=dino_eye(m,vv,xx,one)
                        score=((m.utility(b)-m.utility(a)).float()*g[il]).sum(-1)
                        pred=dino_eye(m,v[ir],x[ir],(2*j.float()-1)[:,None]).float()
                        err=(pred-eyes[(ir//2)*2+j]).gather(1,visible)
                        loss=F.softplus(-score*y[il]).mean()+3*err.square().mean()
                (loss*(len(il)/P['batch'])).backward();losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(par.values(),3);opt.step()
            if step==1 or step%64==0:
                value=float(torch.stack(losses).mean())
                if not np.isfinite(value):raise FloatingPointError((name,step))
                rec=dict(step=step,loss=value,seconds=round(time.monotonic()-tic,2))
                log.write(json.dumps(rec)+'\n')
                if step%256==0:print('TRAIN',name,rec,flush=True)
    ck=d/'adapter.pt'
    with ck.open('xb') as f:torch.save({n:p.detach().cpu() for n,p in par.items()},f)
    dump(d/'freeze.json',dict(adapter=sha(ck),plan=sha(PLAN),code=sha(Path(__file__)),base=BASE[arm]['sha256'],
        arm=arm,budget=k,seed=seed,data=locks['train'],schedule=locks['schedule'],processor=locks['processor'],
        annotation_pool_sha256=array_hash(np.column_stack((ids,plan['cards'][ids],y.cpu().numpy().astype('int64'))).astype('<i8')),
        annotation_draws=step*P['batch'],optical_training_draws=step*P['batch'] if arm=='dino' else 0,
        updates=step,effective_batch=P['batch'],microbatch=micro,parameters=sum(p.numel() for p in m.parameters()),
        adapted=sum(p.numel() for p in par.values()),seconds=round(time.monotonic()-tic,2),utc_ns=time.time_ns(),
        device=torch.cuda.get_device_name(0),versions={k:version(k) for k in ('torch','transformers','lerobot','numpy','safetensors','einops')}))
    print('FROZEN',name,flush=True)
def restore(out,arm,k,seed):
    name=f'{arm}-{k}-{seed}';d=out/'models'/name;meta=jread(d/'freeze.json')
    if meta['adapter']!=sha(d/'adapter.pt') or meta['plan']!=sha(PLAN) or meta['base']!=BASE[arm]['sha256'] or meta['code']!=sha(Path(__file__)):
        raise ValueError('stored adapter identity mismatch: '+name)
    m=base_model(arm,seed);p=trainable(m)
    weights=torch.load(d/'adapter.pt',map_location='cpu',weights_only=True)
    if set(weights)!=set(p):raise ValueError('adapted parameter names mismatch')
    with torch.no_grad():
        for key,value in weights.items():p[key].copy_(value)
    return m
def train(out):
    ensure(ROOT,P['source_locks']);schedule=MAIN/'train/schedule.npz'
    ensure(MAIN,{'train/schedule.npz':P['schedule_sha256']});ensure(DATA,{'train.npz':P['train_sha256']})
    out.mkdir(parents=True,exist_ok=True)
    proc=WORK/'upstream-smol-processor'
    identity=dict(plan=sha(PLAN),train=P['train_sha256'],schedule=sha(schedule),code=sha(Path(__file__)),
                  processor={f.name:sha(f) for f in sorted(proc.glob('*.json'))+sorted(proc.glob('*.txt'))})
    lock=out/'design.json'
    if lock.exists():
        if jread(lock)!=identity:raise ValueError('cannot resume with different code/data')
    else:dump(lock,identity)
    bank=read(DATA/'train.npz');pl=read(schedule);hist=bank['hist']
    x=torch.as_tensor(hist.reshape(-1,hist.shape[-1]),device='cuda')
    for arm in P['models']:
        v=None
        for a,k,s,name in runs():
            if a!=arm:continue
            f=out/'models'/name/'freeze.json'
            if f.exists():
                meta=jread(f)
                if meta['adapter']!=sha(f.with_name('adapter.pt')) or meta['code']!=identity['code']:raise ValueError('incomplete frozen run: '+name)
                continue
            m=base_model(a,s)
            if v is None:
                gaps=check_cache(m,a,x);print('NATIVE-CACHE',a,gaps,flush=True)
                v=visual(m,a,x)
            fit(m,a,k,s,name,out,bank,pl,x,v,identity);del m
        del v
    files={name:sha(out/'models'/name/'freeze.json') for *_,name in runs()}
    dump(out/'frozen.json',dict(design=sha(lock),models=files,frozen_ns=time.time_ns()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);train(p.parse_args().out)
