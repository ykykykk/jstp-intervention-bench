"""从新目录重建两项实验及其全部证据；不会拿附带的检查点初始化模型。"""
import argparse,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(out,device):
    out.mkdir(parents=True,exist_ok=False);bank=out/"experience";rel=out/"release"
    def call(*args):
        print("+",*args,flush=True);subprocess.run([sys.executable,"-u","-m",*map(str,args)],cwd=ROOT,check=True)
    call("mechanism.train","data","--data",bank)
    rel.mkdir();shutil.copyfile(bank/"manifest.json",rel/"experience.json")
    call("mechanism.train","fit","--data",bank,"--release",rel,"--device",device)
    for stage in ("seal","score"):call("mechanism.exam",stage,"--release",rel,"--device",device)
    call("mechanism.analyze","--release",rel)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path("work/rebuilt-mechanism"));p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.out.resolve(),a.device)
