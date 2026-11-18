"""从空目录复算论文主实验。不会覆盖随发行附带的封存成绩。"""
import argparse,shutil,subprocess,sys
from pathlib import Path

def call(cmd,root):
    print("+",*cmd,flush=True);subprocess.run([sys.executable,"-u",*map(str,cmd)],cwd=root,check=True)
def run(work,out,device):
    root=Path(__file__).resolve().parents[1]
    if any(p.exists() for p in (work/"data",work/"features",out)):
        raise SystemExit("复现实验须使用新的工作/结果目录，不覆盖历史数据。")
    call(["scripts/prepare_external.py","--work",work],root)
    for stage in ("data","features"):call(["-m","jstp.prepare",stage,"--work",work,"--device",device],root)
    meta=out/"training";meta.mkdir(parents=True,exist_ok=False)
    for src,name in ((root/"configs/paper.json","protocol.json"),(work/"data/manifest.json","data-manifest.json"),(work/"features/manifest.json","dino-cache-manifest.json")):
        shutil.copyfile(src,meta/name)
    call(["-m","jstp.train","--work",work,"--release",out,"--device",device],root)
    for phase in ("seal","score"):call(["-m","jstp.exam",phase,"--work",work,"--release",out,"--device",device],root)
    call(["-m","jstp.analysis","--release",out],root)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--work",type=Path,default=Path("work/reproduction"));p.add_argument("--release",type=Path,default=Path("work/reproduction-results"));p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.work.resolve(),a.release.resolve(),a.device)
