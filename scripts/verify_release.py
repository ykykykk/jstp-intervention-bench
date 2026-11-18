"""公开包的 SHA-256 校验。默认只需要 Python 标准库，不需要 Git 或 GPU。"""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "MANIFEST.sha256"

def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def write():
    paths = subprocess.check_output(["git","-C",str(ROOT),"ls-files","-z"]).decode("utf8").split("\0")
    files = sorted(p for p in paths if p and p != MANIFEST)
    if not files: raise SystemExit("生成清单须先暂存拟发行的文件。")
    (ROOT/MANIFEST).write_text("".join(f"{digest(ROOT/p)}  {p}\n" for p in files),
                               encoding="utf8",newline="\n")
    print(len(files),"release files indexed")

def verify():
    lines=(ROOT/MANIFEST).read_text(encoding="utf8").splitlines()
    seen=set()
    for line in lines:
        expected,rel=line.split("  ",1)
        path=(ROOT/rel).resolve()
        if path in seen or not path.is_relative_to(ROOT) or path==ROOT/MANIFEST:
            raise ValueError("清单路径无效："+rel)
        seen.add(path)
        if not path.is_file() or digest(path)!=expected:
            raise SystemExit("不一致："+rel)
    print(f"SHA-256 OK: {len(seen)} files")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--write",action="store_true")
    write() if p.parse_args().write else verify()
