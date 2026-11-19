"""Make a portable, offline viewer from training logs and sealed outcomes; no live inference is claimed."""
import json
from pathlib import Path
import numpy as np
from .train import HERE,DEFAULT,C
from lategoal.rig import EXAM
from mechanism.data import read

def build():
    e=DEFAULT;panel=e/"fresh";parent=HERE/"paper";data={}
    for arm in C["arms"]:
        curves=[]
        for s in C["seeds"]:
            records=[json.loads(l) for l in (e/"models"/f"{arm}-{s}"/"curve.jsonl").read_text().splitlines()]
            curves.append([[i["update"],i["validation"]] for i in records])
        s=C["seeds"][0];z=read(panel/"sealed"/f"{arm}-{s}.npz");w=read(panel/"observed"/f"{arm}-{s}.npz")["wins"]
        data[arm]=dict(curves=curves,probe=z["probe"][:6].tolist(),u=z["u"][:6].tolist(),w=w[:6].astype(int).tolist())
    payload=dict(arms=data,g=EXAM.tolist(),ids=read(panel/"sealed/items.npz")["ids"][:6].tolist())
    template=(parent/"viewer-template.html").read_text(encoding="utf8")
    (parent/"evidence-room.html").write_text(template.replace("/*RECORDS*/",json.dumps(payload,separators=(",",":"))),encoding="utf8",newline="\n")
if __name__=="__main__":build()

