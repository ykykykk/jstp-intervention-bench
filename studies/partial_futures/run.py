"""Continue from fully frozen weights, never from an incomplete training directory."""
import argparse
import torch
from pathlib import Path
from mechanism.data import sha
from lategoal.train import tensors
from .train import C,DATA,DEFAULT,fit,load
from .exam import calibrate,seal,execute
from .analyze import analyze

def run(out,device):
    for s in ("train","validation"):
        if sha(DATA/(s+".npz"))!=C[s+"_sha256"]:raise ValueError("数据指纹不符："+s)
    tr,va=[tensors(DATA/(s+".npz"),device) for s in ("train","validation")]
    for arm in C["arms"]:
        for seed in C["seeds"]:
            d=out/"models"/f"{arm}-{seed}"
            if (d/"freeze.json").is_file():load(out,arm,seed)
            else:fit(out,arm,seed,tr,va,device)
    del tr,va
    torch.set_num_threads(2)
    if not (out/"calibrated.json").exists():calibrate(out)
    for panel in C["panels"]:
        if not (out/panel/"before.json").exists():seal(out,panel)
    for panel in C["panels"]:
        if not (out/panel/"after.json").exists():execute(out,panel)
    if not (out/"analysis/summary.json").exists():analyze(out)
    print("COMPLETE",out,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=DEFAULT);p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.out,a.device)

