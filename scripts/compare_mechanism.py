"""逐实验单元比较独立重训；不要求时钟或序列化容器字节一致。"""
import argparse,json
from pathlib import Path
import numpy as np

def compare(a,b):
    ra=json.loads((a/"experience.json").read_text());rb=json.loads((b/"experience.json").read_text())
    if ra!=rb:raise ValueError("training records differ")
    report={}
    for name in ("test_familiar","test_novel","language"):
        left=a/"sealed"/name;right=b/"sealed"/name
        fs=sorted(p.name for p in left.glob("*.npz"))
        if fs!=sorted(p.name for p in right.glob("*.npz")):raise ValueError("different methods")
        decisions=wrong=cam_diffs=payload_diffs=0; max_float=0.
        for fn in fs:
            with np.load(left/fn) as x,np.load(right/fn) as y:
                if set(x.files)!=set(y.files):raise ValueError("different ledger fields")
                for k in x.files:
                    v,w=x[k],y[k]
                    if v.shape!=w.shape:raise ValueError("different ledger shapes")
                    if k=="u" or k.endswith("_answer"):
                        decisions+=v.size;wrong+=int(np.count_nonzero(v!=w))
                    elif k in ("prospect","donor","terminal"):cam_diffs+=int(np.count_nonzero(v!=w))
                    elif k.endswith("_message"):payload_diffs+=int(np.count_nonzero(v!=w))
                    elif v.dtype.kind=="f":max_float=max(max_float,float(np.max(np.abs(v.astype(float)-w))))
                    elif not np.array_equal(v,w):raise ValueError("different inputs/proposals: "+fn+":"+k)
        report[name]=dict(decisions=decisions,changed_decisions=wrong,changed_quantized_camera_elements=cam_diffs,
                          changed_discrete_messages=payload_diffs,largest_other_float_change=max_float)
    observed={}
    for f in sorted((a/"observed").rglob("*.npz")):
        g=b/"observed"/f.relative_to(a/"observed")
        with np.load(f) as x,np.load(g) as y:
            if set(x.files)!=set(y.files):raise ValueError("different physical audit fields")
            for k in x.files:
                if not np.array_equal(x[k],y[k]):observed[k]=observed.get(k,0)+int(np.count_nonzero(x[k]!=y[k]))
    summary=lambda p:json.loads((p/"analysis/summary.json").read_text())
    sa,sb=summary(a),summary(b)
    return dict(same_observed_training_records=True,ledgers=report,changed_observation_elements=observed,
                identical_gates=sa["gates"]==sb["gates"],identical_all_layout_intervals=sa["metrics"]==sb["metrics"])
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("rerun",type=Path);p.add_argument("--out",type=Path)
    a=p.parse_args();d=compare(a.reference,a.rerun);txt=json.dumps(d,indent=2);print(txt)
    if a.out:
        with a.out.open("x",encoding="utf8",newline="\n") as f:f.write(txt+"\n")
