"""在独立新目录生成体验，训练、冻结、封存、执行，再重算。"""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(out,device):
    out.mkdir(parents=True,exist_ok=False);data=out/"experience";rel=out/"release"
    def call(*a):subprocess.run([sys.executable,"-u","-m",*map(str,a)],cwd=ROOT,check=True)
    call("lategoal.train","data","--data",data)
    call("lategoal.train","fit","--data",data,"--release",rel,"--device",device)
    for s in ("calibrate","seal","execute"):call("lategoal.exam",s,"--release",rel)
    call("lategoal.analysis","--release",rel)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path("work/late-rebuild"));p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.out.resolve(),a.device)
