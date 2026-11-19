"""Retarget the two archived budget-bridge modules without changing the code hashes recorded at freeze."""
import argparse
from pathlib import Path
from mechanism.data import read,sha
from . import bridge,supplement
from .train import C,DATA
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if out==bridge.DEFAULT.resolve():raise ValueError("请使用单独的重建目录")
    if sha(DATA/"train.npz")!=C["train_sha256"]:raise ValueError("训练文件指纹错误")
    for name in ("dino","smol"):
        if not (out/"transfer"/name/"fresh-result.json").is_file():raise ValueError("请先完成官方迁移条件："+name)
    bridge.DEFAULT=supplement.DEFAULT=out
    data=read(DATA/"train.npz")
    for arm in ("partial","direct","world"):bridge.run(arm,data)
    supplement.main()

