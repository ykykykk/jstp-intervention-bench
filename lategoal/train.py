import argparse,json,time,platform
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as fn
from .rig import C,PROTO,TRAIN,ROOT,REL,bank,save,read,sha,dump
from .net import Net,sign
from mechanism.train import config

def train_data(path):
    path.mkdir(parents=True,exist_ok=False);items={}
    for split in ("train","validation"):
        a=bank(split);p=path/(split+".npz");save(p,**a)
        items[split]=dict(layouts=len(a["start"]),executed_sensor_branches=int(np.prod(a["eye"].shape[:-1])),
                          distinct_training_cards=len(TRAIN),sha256=sha(p))
        print("DATA",split,items[split],flush=True)
    dump(path/"manifest.json",dict(protocol=sha(PROTO),records=items))
def tensors(p,dev):
    return {k:torch.as_tensor(v,device=dev).float() for k,v in read(p).items()}
def batch(rng,n,bank,goals):
    dev=goals.device;t=lambda x:torch.as_tensor(x,device=dev)
    i=t(rng.integers(len(bank["start"]),size=n));s,g,pi=map(t,rng.integers(2,size=(3,n)))
    card=goals[t(rng.integers(len(goals),size=n))];slot=t(rng.integers(2,size=n))
    x=bank["hist"][i,s,g,pi];p=pi.float()*2-1;target=torch.where((bank["motion"][i,s,g,0]*card).sum(-1)>0,-1.,1.)
    return bank["start"][i],x,p,card,target,bank["eye"][i,s,g,pi],slot.float()*2-1,bank["eye"][i,s,g,slot]
def forward(net,b,arm):
    start,x,drive,card,good,aim,alt,yalt=b
    probe=net.probe(start);h=net.observe(x);z=net.future(h,drive);motor=net.issue(h,z,card,arm)
    bce=lambda v,y:fn.softplus(-v*y).mean()
    loss=.18*bce(probe,start[:,-1])+bce(motor,good)
    err=(net.eye(z)-aim).square().mean()
    if arm!="direct":loss=loss+1.5*(err+(net.eye(net.future(h,alt))-yalt).square().mean())
    return loss,motor,err
def fit(src,rel,device):
    tr,va=[tensors(src/(k+".npz"),device) for k in ("train","validation")]
    goals=torch.tensor(TRAIN,device=device).float()
    for arm in C["arms"]:
        for seed in C["seeds"]:
            path=rel/"models"/f"{arm}-{seed}";path.mkdir(parents=True,exist_ok=False)
            config(seed,device);net=Net().to(device);opt=torch.optim.AdamW(net.parameters(),lr=C["lr"],weight_decay=.0001)
            rng=np.random.default_rng(seed+813);val=batch(np.random.default_rng(90018),1024,va,goals)
            tic=time.monotonic()
            with (path/"curve.jsonl").open("x",encoding="utf8",newline="\n",buffering=1) as log:
                for k in range(1,C["updates"]+1):
                    b=batch(rng,C["batch"],tr,goals);loss,_,_=forward(net,b,arm)
                    if not torch.isfinite(loss):raise ValueError("训练出现非有限损失")
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
                    if k%256==0:
                        with torch.no_grad():_,m,e=forward(net,val,arm)
                        log.write(json.dumps(dict(step=k,loss=float(loss.detach()),validation=float((sign(m)==val[4]).float().mean()),
                            optical_rmse=float(e.sqrt()) if arm!="direct" else None,seconds=round(time.monotonic()-tic,2)))+"\n")
            f=path/"last.pt"
            with f.open("xb") as w:torch.save(net.cpu().state_dict(),w)
            dump(path/"freeze.json",dict(weights=sha(f),protocol=sha(PROTO),experience=sha(src/"manifest.json"),seed=seed,
                 updates=k,parameters=sum(v.numel() for v in net.parameters()),torch=str(torch.__version__),numpy=np.__version__,
                 python=platform.python_version(),device=torch.cuda.get_device_name(device) if device.startswith("cuda") else device,
                 seconds=round(time.monotonic()-tic,2)))
            print("FROZEN",arm,seed,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("data","fit"));p.add_argument("--data",type=Path,default=ROOT/"work/late-experience")
    p.add_argument("--release",type=Path,default=REL);p.add_argument("--device",default="cuda");a=p.parse_args()
    if a.stage=="data":train_data(a.data)
    else:
        from shutil import copyfile
        a.release.mkdir(parents=True,exist_ok=False);copyfile(a.data/"manifest.json",a.release/"experience.json")
        fit(a.data,a.release,a.device)
if __name__=="__main__":main()
