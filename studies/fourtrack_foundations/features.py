"""Official upstream forwards on public inputs only. Cache deterministic frozen computation, not answers."""
import argparse, subprocess, sys, time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .common import CACHE, C, E, MAIN, OFF, P, PLAN, WORK, jread, ensure, sha, dump, read, training_data

def grids(x):
    base=x[:,:72].reshape(-1,1,6,12)
    delta=x[:,72:144].reshape_as(base)
    return base,delta

class Video:
    def __init__(self):
        code=WORK/'upstream-vjepa21-code'
        head=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
        if head!=P['vjepa21']['revision']: raise ValueError('V-JEPA code revision')
        sys.path.insert(0,str(code))
        from src.hub.backbones import vjepa2_1_vit_base_384, _clean_backbone_key
        enc,pred=vjepa2_1_vit_base_384(pretrained=False)
        del pred
        path=WORK/'upstream-vjepa21'/'vjepa2_1_vitb_dist_vitG_384.pt'
        pack=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        enc.load_state_dict(_clean_backbone_key(pack['ema_encoder']),strict=True)
        self.net=enc.requires_grad_(False).eval().cuda()
        self.meta=dict(base_sha256=sha(path),upstream_commit=head,
                       frozen_used_parameters=sum(v.numel() for v in enc.parameters()),
                       upstream='Meta official EMA encoder; native video tokens before/peak, no AC planner')
    def __call__(self,x):
        base,delta=grids(x)
        frames=torch.stack([5*base,5*base,5*(base+delta),5*(base+delta)],1).clamp(0,1)
        b=len(x)
        v=F.interpolate(frames.flatten(0,1),size=(96,192),mode='bilinear',align_corners=False)
        mean=v.new_tensor([.485,.456,.406])[None,:,None,None];sd=v.new_tensor([.229,.224,.225])[None,:,None,None]
        v=((v-mean)/sd).reshape(b,4,3,96,192).transpose(1,2)
        with torch.autocast('cuda',dtype=torch.bfloat16): tokens=self.net(v)
        # Two observed tubelets; retain a coarse spatial grid in each, not a single CLS/mean vector.
        t=tokens.reshape(b*2,6,12,-1).permute(0,3,1,2)
        t=F.adaptive_avg_pool2d(t.float(),(2,4)).permute(0,2,3,1)
        return {'v':t.reshape(b,16,-1).half().cpu()}

def feature_code(arm):
    import hashlib,inspect
    from .policies import Policy,normalized_image
    # Editing one upstream bridge cannot invalidate a different model's completed forward.
    parts=[grids,generate,Video] if arm=='vjepa21' else [grids,generate,Policy,normalized_image]
    return hashlib.sha256('\n'.join(inspect.getsource(obj) for obj in parts).encode()).hexdigest()

def generate(arm,split):
    from .policies import Policy
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);CACHE.mkdir(parents=True,exist_ok=True)
    out=CACHE/f'{arm}-{split}.pt'
    if (E/arm/(split+'-features.json')).exists(): raise FileExistsError(out)
    if split=='train':
        bank,_=training_data();hist=bank['hist'];source=C['train_sha256']
    else:
        for a in ('joint',*P['fresh_external']):
            if not (E/a/'frozen.json').exists(): raise RuntimeError('freeze every fit before held-out prefix: '+a)
        rec=jread(OFF/'prefix.json');ensure(OFF,rec['files'])
        if rec['items']!=sha(MAIN/'sealed/items.npz'):raise ValueError('wrong exam')
        hist=read(OFF/'public-prefix.npz')['hist'];source=sha(OFF/'public-prefix.npz')
    x=torch.as_tensor(hist.reshape(-1,hist.shape[-1]),device='cuda')
    m=Video() if arm=='vjepa21' else Policy(arm)
    observed=x[:,:144] if arm=='vjepa21' else m.inputs_key(x)
    _,idx,inv=np.unique(observed.cpu().numpy(),axis=0,return_index=True,return_inverse=True)
    inputs=x[torch.as_tensor(idx,device=x.device)]
    buf={};tic=time.monotonic();done=0
    with torch.inference_mode():
        for group in inputs.split(32):
            row=m(group);done+=len(group)
            for key,v in row.items():
                if not torch.isfinite(v).all():raise FloatingPointError((arm,done,key))
                buf.setdefault(key,[]).append(v)
            if done==len(group) or done//512!=(done-len(group))//512:
                print('FEATURE',arm,split,done,'/',len(inputs),'seconds',round(time.monotonic()-tic,1),flush=True)
    inv=torch.from_numpy(inv.reshape(-1));arr={key:torch.cat(rows)[inv] for key,rows in buf.items()}
    if arm!='vjepa21':arr.update(m.projection())
    with out.open('xb') as f:torch.save(arr,f)
    (E/arm).mkdir(parents=True,exist_ok=True)
    dump(E/arm/(split+'-features.json'),dict(**m.meta,source=source,plan=sha(PLAN),code=feature_code(arm),
         file=sha(out),rows=len(x),distinct_forward_rows=len(inputs),seconds=round(time.monotonic()-tic,2),
         versions={n:version(n) for n in (('torch','transformers','timm') if arm=='vjepa21' else (('torch','lerobot') if arm=='act' else ('torch','lerobot','diffusers')))}))
    print('FEATURES READY',arm,split,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=P['fresh_external']);p.add_argument('split',choices=['train','exam'])
    a=p.parse_args();generate(a.arm,a.split)
