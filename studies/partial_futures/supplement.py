"""Record the matched schedule and paired layout contrasts for the separately budgeted resource track."""
import json,hashlib
import numpy as np
from mechanism.data import dump,read,save,sha
from .analyze import summary,distribution,count
from .transfer import DEFAULT,C,P,PLAN,DATA
from lategoal.rig import TRAIN

def main():
    d=DEFAULT/"transfer";n=P["records"];rng=np.random.default_rng(P["seed"])
    schedule=np.stack((rng.permutation(n),rng.integers(len(TRAIN),size=n)),1).astype("<i4")
    digest=hashlib.sha256(schedule.tobytes()).hexdigest();save(d/"schedule.npz",record_and_card=schedule)
    names=("matched-partial","matched-direct","matched-world","dino","smol");out={}
    for panel,(_,k) in C["panels"].items():
        draw=np.random.default_rng(44178).multinomial(k,np.full(k,1/k),3000)
        each={};ids=None
        for a in names:
            p=d/a;meta=json.loads((p/"freeze.json").read_text(encoding="utf8"))
            if "schedule" in meta and meta["schedule"]!=digest:raise ValueError("采样排程不同")
            if meta["data"]!=sha(DATA/"train.npz"):raise ValueError("训练来源不同")
            before=json.loads((p/(panel+"-before.json")).read_text(encoding="utf8"))
            if before["intents"]!=sha(p/(panel+"-sealed.npz")) or before["weights"]!=sha(p/"adapter.pt"):raise ValueError("封卷不符")
            m=json.loads((p/(panel+"-result.json")).read_text(encoding="utf8"));obs=p/(panel+"-observed.npz")
            if sha(obs)!=m["observed"]:raise ValueError("实测不符")
            v=read(p/(panel+"-sealed.npz"))["ids"]
            if ids is not None and not np.array_equal(ids,v):raise ValueError("布局配对失败")
            ids=v;y=read(obs)["wins"];each[a]=distribution(count(y[...,::2]&y[...,1::2]),draw)
        ref=each["matched-partial"];out[panel]={a:summary(*(i-j for i,j in zip(ref,v))) for a,v in each.items() if a!="matched-partial"}
    dump(d/"paired-contrasts.json",dict(schedule_sha256=digest,plan=sha(PLAN),train=sha(DATA/"train.npz"),contrasts=out))
if __name__=="__main__":main()

