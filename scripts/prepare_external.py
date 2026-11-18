"""下载到用户自己的缓存；不随仓库再分发第三方权重，不执行未经锁定的远程代码。"""
from pathlib import Path
import argparse,hashlib,json,subprocess,urllib.request,os

def main(work):
    cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/paper.json").read_text(encoding="utf8"))["dino"]
    dst=work/"external";dst.mkdir(parents=True,exist_ok=True)
    for name,remote,key in (("dinov2","facebookresearch/dinov2","encoder_commit"),("dino_wm","gaoyuezhou/dino_wm","predictor_commit")):
        path=dst/name; ref=cfg[key]
        if not path.exists():
            subprocess.run(["git","init","-q",str(path)],check=True)
            subprocess.run(["git","-C",str(path),"remote","add","origin","https://github.com/"+remote+".git"],check=True)
            subprocess.run(["git","-C",str(path),"fetch","--depth","1","origin",ref],check=True)
            subprocess.run(["git","-C",str(path),"checkout","-q","--detach","FETCH_HEAD"],check=True)
        got=subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
        if got!=ref:raise SystemExit(f"{path} 不是发布锁定的提交，请选择新的缓存目录")
    dest=dst/cfg["checkpoint_name"]
    def digest(path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for q in iter(lambda:f.read(1<<20),b""):h.update(q)
        return h.hexdigest()
    if not dest.exists():
        temp=dest.with_suffix(".download")
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"+cfg["checkpoint_name"]
        with urllib.request.urlopen(url,timeout=120) as src,temp.open("xb") as f:
            for chunk in iter(lambda:src.read(1<<20),b""):f.write(chunk)
        if digest(temp)!=cfg["encoder_sha256"]:raise RuntimeError("下载的官方权重哈希不符")
        os.replace(temp,dest)
    if digest(dest)!=cfg["encoder_sha256"]:raise RuntimeError("权重哈希不符")
    print("已就绪：两个固定提交和官方 DINOv2 权重。")
if __name__=="__main__":
    pa=argparse.ArgumentParser();pa.add_argument("--work",type=Path,default=Path("work"));main(pa.parse_args().work)
