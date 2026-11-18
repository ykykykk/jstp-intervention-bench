"""不比打包时间戳；逐方法、种子、布局比较冻结选择、实际成败和物理量。"""
import argparse,json
from pathlib import Path
import numpy as np
def compare(ref,new):
    summary={}
    for split in ("test_id","test_link"):
        a=ref/"exam"/split;b=new/"exam"/split
        if (a/"observations.json").read_bytes()!=(b/"observations.json").read_bytes():raise ValueError("考题或历史不同")
        names=sorted(x.name for x in a.glob("*.npz"))
        if names!=sorted(x.name for x in b.glob("*.npz")):raise ValueError("方法集不同")
        changed=0; max_dy=0.; n=0
        for name in names:
            with np.load(a/name) as f,np.load(b/name) as g:
                changed+=int(np.count_nonzero(f["choice"]!=g["choice"]));n+=f["choice"].size
                if "dy" in f:max_dy=max(max_dy,float(np.nanmax(np.abs(f["dy"]-g["dy"]))))
        with np.load(ref/"observed"/split/"outcomes.npz") as f,np.load(new/"observed"/split/"outcomes.npz") as g:
            if not np.array_equal(f["actual_sha"],g["actual_sha"]):raise ValueError("真实模拟器轨迹不同")
            different=sum(int(np.count_nonzero(f[k]!=g[k])) for k in f.files if k not in ("actual_sha","dy"))
        summary[split]=dict(compared_decisions=int(n),changed_decisions=changed,changed_success=different,max_pixel_dy_delta=max_dy)
    return summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("rerun",type=Path);p.add_argument("--out",type=Path);a=p.parse_args();v=compare(a.reference,a.rerun)
    print(json.dumps(v,indent=2))
    if a.out:
        with a.out.open("x",encoding="utf8",newline="\n") as f:json.dump(v,f,indent=2);f.write("\n")
