"""Download only pinned official artifacts, outside Git, and check their LFS SHA-256."""
from pathlib import Path
import hashlib,json
from huggingface_hub import snapshot_download
D=Path(__file__).resolve().parent;WORK=D.parents[1]/"work"; P=json.loads((D/"transfer-plan.json").read_text(encoding="utf8"))
def get(repo,rev,dst,pats):
    snapshot_download(repo_id=repo,revision=rev,local_dir=WORK/dst,allow_patterns=pats)
if __name__=="__main__":
    for key,folder in (("dino","upstream-dino"),("smol","upstream-smolvla")):
        s=P[key];get(s["model"],s["revision"],folder,["*.json","model.safetensors"])
        with (WORK/folder/"model.safetensors").open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=s["sha256"]:raise ValueError(key+" 权重校验失败")
    get("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",P["smol"]["processor_revision"],"upstream-smol-processor",["*.json","*.txt","*.jinja","*.model"])

