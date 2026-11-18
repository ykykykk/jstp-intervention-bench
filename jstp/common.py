from __future__ import annotations
import hashlib,json,os,random
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"configs"/"paper.json"
def config(): return json.loads(PROTOCOL.read_text(encoding="utf8"))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def arrsha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def new_json(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf8",newline="\n") as f:
        json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
def read_json(p): return json.loads(Path(p).read_text(encoding="utf8"))
def save_npz(path,**kw):
    with Path(path).open("xb") as f: np.savez_compressed(f,**kw)
def read_npz(p):
    with np.load(p,allow_pickle=False) as f: return {k:f[k] for k in f.files}
def mname(k,s): return f"{k}-{s}"
def configure(seed,device):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available(): raise RuntimeError("指定 CUDA 但设备不可用")
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    torch.set_num_threads(2)
