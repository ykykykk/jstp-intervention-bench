"""只取权重与冻结元数据，重新试探、校准、封卷及执行；不读取原提交来选动作。"""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def run(src,out,report):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(src/"models",out/"models");shutil.copyfile(src/"experience.json",out/"experience.json")
    for s in ("calibrate","seal","execute"):
        subprocess.run([sys.executable,"-u","-m","lategoal.exam",s,"--release",str(out)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-m","lategoal.analysis","--release",str(out)],cwd=ROOT,check=True)
    changed={};decisions=0;files=0
    for part in ("calibration","sealed","observed"):
        originals=sorted(p.relative_to(src) for p in (src/part).glob("*.npz"))
        if not originals or originals!=sorted(p.relative_to(out) for p in (out/part).glob("*.npz")):raise ValueError(part)
        for p in originals:
            with np.load(src/p) as a,np.load(out/p) as b:
                if set(a.files)!=set(b.files):raise ValueError(str(p))
                for k in a.files:
                    x,y=a[k],b[k]
                    if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError(str(p)+":"+k)
                    if not np.array_equal(x,y):changed[p.as_posix()+":"+k]=int(np.count_nonzero(x!=y))
                    if k=="u":decisions+=x.size
            files+=1
    equal=(src/"analysis/summary.json").read_bytes()==(out/"analysis/summary.json").read_bytes()
    r=dict(kind="same-machine frozen-checkpoint replay, not independent retraining",decisions=decisions,npz_files=files,
           changed_array_elements=changed,identical_summary=equal)
    print(json.dumps(r));target=report or out/"replay.json"
    with target.open("x",encoding="utf8",newline="\n") as f:json.dump(r,f,indent=2);f.write("\n")
    if changed or not equal:raise SystemExit("复演与归档有差异；见报告。")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,default=ROOT/"evidence/late-goals")
    p.add_argument("--out",type=Path,default=ROOT/"work/late-frozen-replay");p.add_argument("--report",type=Path)
    a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.report)
