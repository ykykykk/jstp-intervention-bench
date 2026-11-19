"""Replay exactly the v3 bank/minibatch order. Never consume the calibration or examination splits."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from mechanism.data import sha,dump
from mechanism.train import config
from lategoal.rig import TRAIN
from lategoal.train import tensors,batch
from .models import make

HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/"protocol.json"; C=json.loads(PROTOCOL.read_text(encoding="utf8"))
DATA=HERE/"data"; DEFAULT=HERE/"evidence"
SUP=("partial","detached","world")
def logloss(v,y):return F.softplus(-v*y).mean()
def fit(rel,arm,seed,tr,va,device):
    path=rel/"models"/f"{arm}-{seed}"; path.mkdir(parents=True,exist_ok=False)
    config(seed,device); net=make(arm,C["width"]).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=C["lr"],weight_decay=.0001)
    rng=np.random.default_rng(seed+813); goals=torch.tensor(TRAIN,device=device).float()
    val=batch(np.random.default_rng(90018),1024,va,goals); times=time.monotonic()
    with (path/"curve.jsonl").open("x",encoding="utf8",buffering=1) as log:
        for k in range(1,C["updates"]+1):
            start,x,p,g,y,f,alt,falt=batch(rng,C["batch"],tr,goals)
            h=net.observe(x)
            loss=.18*logloss(net.probe(start),start[:,-1])
            loss=loss+(net.loss(h,g,y) if arm=="flow" else logloss(net.policy(h,p,g),y))
            if arm in SUP:
                preds=net.eye(net.future(torch.cat((h,h)),torch.cat((p,alt))))
                loss=loss+3*F.mse_loss(preds,torch.cat((f,falt)))
            if not torch.isfinite(loss):raise FloatingPointError((arm,seed,k))
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
            if k%256==0:
                with torch.no_grad():
                    vh=net.observe(val[1]); v=net.policy(vh,val[2],val[3])
                    record=dict(update=k,loss=float(loss.detach()),validation=float(((v>=0)==(val[4]>0)).float().mean()),
                        seconds=round(time.monotonic()-times,2))
                    if arm in SUP:record["optical_rmse"]=float(F.mse_loss(net.eye(net.future(vh,val[2])),val[5]).sqrt())
                log.write(json.dumps(record)+"\n")
    weights=path/"last.pt"
    with weights.open("xb") as stream:torch.save(net.cpu().state_dict(),stream)
    dump(path/"freeze.json",dict(arm=arm,seed=seed,weights=sha(weights),protocol=sha(PROTOCOL),data_sha=C["train_sha256"],
        code={x:sha(HERE/x) for x in ("models.py","train.py")},parameters=sum(p.numel() for p in net.parameters()),
        updates=k,examples=k*C["batch"],seconds=round(time.monotonic()-times,2),torch=str(torch.__version__),
        device=torch.cuda.get_device_name(device) if device.startswith("cuda") else device,python=platform.python_version()))
    print("FROZEN",arm,seed,record,flush=True)
def load(rel,arm,seed):
    d=rel/"models"/f"{arm}-{seed}";meta=json.loads((d/"freeze.json").read_text(encoding="utf8"))
    if meta["weights"]!=sha(d/"last.pt") or meta["protocol"]!=sha(PROTOCOL) or meta["code"]["models.py"]!=sha(HERE/"models.py"):
        raise ValueError("模型或协议指纹不符："+str(d))
    m=make(arm,C["width"]).eval();m.load_state_dict(torch.load(d/"last.pt",map_location="cpu",weights_only=True));return m
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=DEFAULT);p.add_argument("--device",default="cuda")
    p.add_argument("--arms",nargs="+",choices=C["arms"],default=C["arms"]);a=p.parse_args()
    for split in ("train","validation"):
        if sha(DATA/(split+".npz"))!=C[split+"_sha256"]:raise ValueError("数据不再是 v3 的原文件："+split)
    tr,va=[tensors(DATA/(s+".npz"),a.device) for s in ("train","validation")]
    for arm in a.arms:
        for seed in C["seeds"]:fit(a.out,arm,seed,tr,va,a.device)

